from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from bqskit.ir import CircuitLocation
from bqskit.ir import Gate
from bqskit.ir.gates import ConstantGate
from bqskit.ir.gates import U3Gate
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.ir.opt import Instantiater
from bqskit.qis import UnitaryMatrix
from bqskit.qis.state import StateSystem
from bqskit.qis.state import StateVector
from jax import Array
from jax._src.lib import xla_extension as xe
from scipy.stats import unitary_group

from qfactorjax.qfactor import _apply_padding_and_flatten
from qfactorjax.qfactor import _remove_padding_and_create_matrix
from qfactorjax.singlelegedtensor import LHSTensor
from qfactorjax.singlelegedtensor import RHSTensor
from qfactorjax.singlelegedtensor import SingleLegSideTensor
from qfactorjax.unitary_acc import VariableUnitaryGateAcc
from qfactorjax.unitarymatrixjax import UnitaryMatrixJax

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.state.state import StateLike
    from bqskit.qis.state.system import StateSystemLike
    from bqskit.qis.unitary.unitarymatrix import UnitaryLike

_logger = logging.getLogger('bqskit.instant.qf-sample-jax')


jax.config.update('jax_enable_x64', True)


class TermCondition(Enum):
    UNKNOWN = 0
    REACHED_TARGET = 1
    EXCEEDED_MAX_ITER = 2
    PLATEAU_DETECTED = 3
    EXCEEDED_TRAINING_SET_SIZE = 4


class QFactorSampleJax(Instantiater):

    def __init__(
        self,
        dist_tol: float = 1e-8,
        max_iters: int = 100000,
        min_iters: int = 2,
        beta: float = 0.0,
        amount_of_validation_states: int = 2,
        num_params_coef: float = 1.0,
        overtrain_relative_threshold: float = 0.1,
        diff_tol_r: float = 1e-4,     # Relative criteria for distance change
        plateau_windows_size: int = 6,
        exact_amount_of_states_to_train_on: int | None = None,
    ):

        if not isinstance(dist_tol, float) or dist_tol > 0.5:
            raise TypeError('Invalid distance threshold.')

        if not isinstance(max_iters, int) or max_iters < 0:
            raise TypeError('Invalid maximum number of iterations.')

        if not isinstance(min_iters, int) or min_iters < 0:
            raise TypeError('Invalid minimum number of iterations.')

        self.dist_tol = dist_tol
        self.max_iters = max_iters
        self.min_iters = min_iters

        self.beta = beta
        self.amount_of_validation_states = amount_of_validation_states
        self.num_params_coef = num_params_coef
        self.overtrain_ratio = overtrain_relative_threshold
        self.diff_tol_r = diff_tol_r
        self.plateau_windows_size = plateau_windows_size
        self.exact_amount_of_states_to_train_on =\
            exact_amount_of_states_to_train_on

        self.targets_training_set_size_cache: dict[int, int] = {}

    def instantiate(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
        x0: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:

        return self.multi_start_instantiate(circuit, target, 1)

    def multi_start_instantiate_inplace(
        self,
        circuit: Circuit,
        target: UnitaryLike | StateLike | StateSystemLike,
        num_starts: int,
    ) -> None:
        """
        Instantiate `circuit` to best implement `target` with multiple starts.

        See :func:`multi_start_instantiate` for more info.

        Notes:
            This method is a version of :func:`multi_start_instantiate`
            that modifies `circuit` in place rather than returning a copy.
        """
        target = self.check_target(target)
        params = self.multi_start_instantiate(circuit, target, num_starts)
        circuit.set_params(params)

    async def multi_start_instantiate_async(
        self,
        circuit: Circuit,
        target: UnitaryLike | StateLike | StateSystemLike,
        num_starts: int,
    ) -> npt.NDArray[np.float64]:

        return self.multi_start_instantiate(circuit, target, num_starts)

    def multi_start_instantiate(
        self,
        circuit: Circuit,
        target: UnitaryLike | StateLike | StateSystemLike,
        num_starts: int,
    ) -> npt.NDArray[np.float64]:

        if len(circuit) == 0:
            return np.array([])

        circuit = circuit.copy()

        gates_list = []
        for op in circuit:
            if isinstance(op.gate, VariableUnitaryGate):
                new_gate = VariableUnitaryGateAcc(
                    op.gate.num_qudits,
                    op.gate.radixes,
                )
                gates_list.append(new_gate)
            else:
                gates_list.append(op.gate)

        target_hash = target.__hash__()
        target = UnitaryMatrixJax(self.check_target(target))
        radixes = target.radixes
        num_qudits = target.num_qudits
        locations = tuple([op.location for op in circuit])
        gates = tuple(gates_list)
        biggest_gate_dim = max(g.dim for g in circuit.gate_set)

        if target_hash in self.targets_training_set_size_cache:
            initial_amount_of_training_states =\
                self.targets_training_set_size_cache[target_hash]
        elif self.exact_amount_of_states_to_train_on is None:
            amount_of_params_in_circuit = 0
            for g in gates:
                amount_of_params_in_circuit += g.num_params
            initial_amount_of_training_states = int(
                np.sqrt(amount_of_params_in_circuit) * self.num_params_coef,
            )
        else:
            initial_amount_of_training_states =\
                self.exact_amount_of_states_to_train_on

            if np.prod(radixes) < initial_amount_of_training_states:
                _logger.warning(
                    f'Requested to use '
                    f'{initial_amount_of_training_states} training '
                    'states, while the prod of the radixes is '
                    f'{np.prod(radixes)}. This algorithm shines when we have '
                    f'much less training states than 2^prod(radixes)',
                )

        amount_of_training_states = initial_amount_of_training_states

        validation_states_kets = self.generate_random_states(
            self.amount_of_validation_states, int(np.prod(radixes)),
        )

        generate_untrys_only_once = 'GEN_ONCE' in os.environ

        if generate_untrys_only_once:
            untrys = []

            for gate in gates:
                size_of_untry = 2**gate.num_qudits

                if isinstance(gate, VariableUnitaryGateAcc):
                    pre_padding_untrys = [
                        unitary_group.rvs(size_of_untry) for
                        _ in range(num_starts)
                    ]
                else:
                    pre_padding_untrys = [
                        gate.get_unitary().numpy for
                        _ in range(num_starts)
                    ]

                untrys.append([
                    _apply_padding_and_flatten(
                        pre_padd, gate, biggest_gate_dim,
                    ) for pre_padd in pre_padding_untrys
                ])

            untrys = jnp.array(np.stack(untrys, axis=1))

        term_condition = None
        should_double_the_training_size = True
        while term_condition is None:
            if not generate_untrys_only_once:
                untrys = []

                for gate in gates:
                    size_of_untry = 2**gate.num_qudits

                    if isinstance(gate, VariableUnitaryGateAcc):
                        pre_padding_untrys = [
                            unitary_group.rvs(size_of_untry) for
                            _ in range(num_starts)
                        ]
                    else:
                        pre_padding_untrys = [
                            gate.get_unitary().numpy for
                            _ in range(num_starts)
                        ]

                    untrys.append([
                        _apply_padding_and_flatten(
                            pre_padd, gate, biggest_gate_dim,
                        ) for pre_padd in pre_padding_untrys
                    ])

                untrys = jnp.array(np.stack(untrys, axis=1))

            training_states_kets = self.generate_random_states(
                amount_of_training_states, int(np.prod(radixes)),
            )

            results = self.safe_call_jited_vmaped_state_sample_sweep(
                target, num_starts, radixes, num_qudits, locations, gates,
                validation_states_kets, untrys, training_states_kets,
            )

            final_untrys, training_costs, validation_costs, iteration_counts, \
                plateau_windows = results

            it = iteration_counts[0]
            untrys = final_untrys
            best_start = jnp.argmin(training_costs)

            if any(training_costs < self.dist_tol):
                _logger.debug(
                    f'Terminated: {it} c1 = {training_costs} <= dist_tol.\n'
                    f'Best start is {best_start}',
                )
                term_condition = TermCondition.REACHED_TARGET
            elif it >= self.max_iters:
                _logger.debug(
                    f'Terminated {it}: iteration limit reached. c1 = '
                    f'{training_costs}',
                )
                term_condition = TermCondition.EXCEEDED_MAX_ITER
            elif it > self.min_iters:
                val_to_train_diff = validation_costs - training_costs

                if np.all(np.all(plateau_windows, axis=1)):
                    _logger.debug(
                        f'Terminated: {it} plateau detected in all'
                        f' multistarts c1 = {training_costs}',
                    )
                    term_condition = TermCondition.PLATEAU_DETECTED

                elif all(
                    val_to_train_diff > self.overtrain_ratio * training_costs,
                ):
                    _logger.debug(
                        f'Terminated: {it} overtraining detected in'
                        f' all multistarts',
                    )

                else:
                    term_condition = TermCondition.UNKNOWN
            else:
                term_condition = TermCondition.UNKNOWN

            if term_condition == TermCondition.UNKNOWN:
                _logger.error(
                    f'Terminated with no good reason after {it} iterations '
                    f'with c1s {training_costs}.',
                )

            if (
                term_condition == TermCondition.REACHED_TARGET
                or term_condition == TermCondition.PLATEAU_DETECTED
                or term_condition == TermCondition.EXCEEDED_MAX_ITER
            ):
                self.targets_training_set_size_cache[target_hash] =\
                    amount_of_training_states

            if term_condition is None:
                if should_double_the_training_size:
                    amount_of_training_states *= 2
                else:
                    amount_of_training_states +=\
                        initial_amount_of_training_states

                if amount_of_training_states > np.prod(radixes):
                    term_condition = TermCondition.EXCEEDED_TRAINING_SET_SIZE
                    _logger.debug(
                        'Stopping as we reached the max number of'
                        ' training states',
                    )

        params: list[Sequence[float]] = []
        for untry, gate in zip(untrys[best_start], gates):
            if isinstance(gate, ConstantGate):
                params.extend([])
            else:
                params.extend(
                    gate.get_params(
                        _remove_padding_and_create_matrix(untry, gate),
                    ),
                )

        return np.array(params)

    def safe_call_jited_vmaped_state_sample_sweep(
            self,
            target: UnitaryMatrixJax,
            num_starts: int,
            radixes: tuple[int, ...],
            num_qudits: int,
            locations: tuple[CircuitLocation, ...],
            gates: tuple[Gate, ...],
            validation_states_kets: Array,
            untrys: Array,
            training_states_kets: Array,
    ) -> tuple[Array, Array[float], Array[float], Array[int], Array[bool]]:
        """We couldn't find a way to check if we are going to allocate more than
        the GPU memory, so we created this "safe" function that calls qfactor-
        sample and then if OOM exception is caught it recursively calls qfactor-
        sample with half the multistarts."""

        try:
            results = _jited_loop_vmaped_state_sample_sweep(
                target, num_qudits, radixes, locations, gates, untrys,
                self.dist_tol, self.max_iters, self.beta,
                num_starts, self.min_iters, self.diff_tol_r,
                self.plateau_windows_size, self.overtrain_ratio,
                training_states_kets, validation_states_kets,
            )

        except xe.XlaRuntimeError as e:

            if num_starts == 1:
                _logger.error(
                    f'Got a runtime error {e}, while {num_starts = } ,exiting',
                )
                raise e

            _logger.debug(
                f'Got a runtime error {e} will try re-run with half the starts',
            )

            mid_point = num_starts // 2
            first_half_untrys = untrys[:mid_point]
            second_half_untrys = untrys[mid_point:]

            results1 = self.safe_call_jited_vmaped_state_sample_sweep(
                target, mid_point, radixes, num_qudits, locations, gates,
                validation_states_kets, first_half_untrys,
                training_states_kets,
            )

            results2 = self.safe_call_jited_vmaped_state_sample_sweep(
                target, num_starts - mid_point, radixes, num_qudits, locations,
                gates, validation_states_kets, second_half_untrys,
                training_states_kets,
            )

            # TODO: Fix the typing ignore here
            results = tuple(
                jnp.concatenate((results1[i], results2[i]))
                for i in range(5)
            )  # type: ignore

        return results

    @staticmethod
    def get_method_name() -> str:
        """Return the name of this method."""
        return 'qfactor_jax_batched_jit'

    @staticmethod
    def can_internally_perform_multistart() -> bool:
        """Probes if the instantiater can internally perform multistart."""
        return True

    @staticmethod
    def is_capable(circuit: Circuit) -> bool:
        """Return true if the circuit can be instantiated."""
        return all(
            isinstance(
                gate, (
                    VariableUnitaryGate,
                    VariableUnitaryGateAcc, U3Gate, ConstantGate,
                ),
            )
            for gate in circuit.gate_set
        )

    @staticmethod
    def get_violation_report(circuit: Circuit) -> str:
        """
        Return a message explaining why `circuit` cannot be instantiated.

        Args:
            circuit (Circuit): Generate a report for this circuit.

        Raises:
            ValueError: If `circuit` can be instantiated with this
                instantiater.
        """

        invalid_gates = {
            gate
            for gate in circuit.gate_set
            if not isinstance(
                gate, (
                    VariableUnitaryGate,
                    VariableUnitaryGateAcc,
                    U3Gate,
                    ConstantGate,
                ),
            )
        }

        if len(invalid_gates) == 0:
            raise ValueError('Circuit can be instantiated.')

        return (
            'Cannot instantiate circuit with qfactor because'
            ' the following gates are not locally optimizable with jax: %s.'
            % ', '.join(str(g) for g in invalid_gates)
        )

    @staticmethod
    def generate_random_states(
        amount_of_states: int,
        size_of_state: int,
    ) -> list[npt.NDArray[np.complex128]]:
        """
        Generate a list of random state vectors (kets) using random unitary
        matrices.

        This function generates a specified number of random quantum state
        vectors (kets) by creating random unitary matrices and extracting
        their first columns.

        Args:
            amount_of_states (int): The number of random states to generate.
            size_of_state (int): The dimension of each state vector (ket).

        Returns:
            list of ndarrays: A list containing random quantum state vectors.
                            Each ket is represented as a numpy ndarray of
                            shape (size_of_state, 1).
        """
        states_kets = []
        states_to_add = amount_of_states
        while states_to_add > 0:
            # We generate a random unitary and take its columns
            rand_unitary = unitary_group.rvs(size_of_state)
            states_to_add_in_step = min(states_to_add, size_of_state)
            for i in range(states_to_add_in_step):
                states_kets.append(rand_unitary[:, i:i + 1])
            states_to_add -= states_to_add_in_step

        return states_kets


def _loop_vmaped_state_sample_sweep(
    target: UnitaryMatrixJax, num_qudits: int, radixes: tuple[int, ...],
    locations: tuple[CircuitLocation, ...],
    gates: tuple[Gate, ...], untrys: Array,
    dist_tol: float, max_iters: int, beta: float,
    amount_of_starts: int, min_iters: int,
    diff_tol_r: float, plateau_windows_size: int,
    overtrain_ratio: float, training_states_kets: Array,
    validation_states_kets: Array,
) -> tuple[Array, Array[float], Array[float], Array[int], Array[bool]]:

    # Calculate the bras for the validation and training states
    validation_states_bras = jax.vmap(
        lambda ket: ket.T.conj(),
    )(jnp.array(validation_states_kets))

    training_states_bras = jax.vmap(
        lambda ket: ket.T.conj(),
    )(jnp.array(training_states_kets))

    # Calculate the A and B0 tensor
    target_dagger = target.T.conj()
    A_train = RHSTensor(
        list_of_states=training_states_bras,
        num_qudits=num_qudits, radixes=radixes,
    )
    A_train.apply_left(target_dagger, range(num_qudits))

    A_val = RHSTensor(
        list_of_states=validation_states_bras,
        num_qudits=num_qudits, radixes=radixes,
    )
    A_val.apply_left(target_dagger, range(num_qudits))

    B0_train = LHSTensor(
        list_of_states=training_states_kets,
        num_qudits=num_qudits, radixes=radixes,
    )
    B0_val = LHSTensor(
        list_of_states=validation_states_kets,
        num_qudits=num_qudits, radixes=radixes,
    )

    # In JAX the body of a while must be a function that accepts and returns
    # the same type, and also the check should be a function that accepts it
    # and return a boolean

    def should_continue(
        loop_var: tuple[
            Array, Array[float], Array[float], Array[int], Array[bool],
        ],
    ) -> Array[bool]:
        _, training_costs, validation_costs, \
            iteration_counts, plateau_windows = loop_var

        any_reached_required_tol = jnp.any(
            jax.vmap(
                lambda cost: cost <= dist_tol,
            )(training_costs),
        )

        reached_max_iteration = iteration_counts[0] > max_iters
        above_min_iteration = iteration_counts[0] > min_iters

        val_to_train_diff = validation_costs - training_costs
        all_reached_over_training = jnp.all(
            val_to_train_diff > overtrain_ratio * training_costs,
        )

        all_reached_plateau = jnp.all(
            jnp.all(plateau_windows, axis=1),
        )

        return jnp.logical_not(
            jnp.logical_or(
                any_reached_required_tol,
                jnp.logical_or(
                    reached_max_iteration,
                    jnp.logical_and(
                        above_min_iteration,
                        jnp.logical_or(
                            all_reached_over_training,
                            all_reached_plateau,
                        ),
                    ),
                ),
            ),
        )

    def _while_body_to_be_vmaped(
        loop_var: tuple[
            Array, Array[float], Array[float], Array[int], Array[bool],
        ],
    ) -> tuple[
            Array, Array[float], Array[float], Array[int], Array[bool],
    ]:

        untrys, training_cost, validation_cost, iteration_count, \
            plateau_window = loop_var

        untrys_as_matrixes: list[UnitaryMatrixJax] = []
        for gate_index, gate in enumerate(gates):
            untrys_as_matrixes.append(
                UnitaryMatrixJax(
                    _remove_padding_and_create_matrix(
                        untrys[gate_index], gate,
                    ), gate.radixes,
                ),
            )
        prev_training_cost = training_cost

        untrys_as_matrixes, training_cost, validation_cost =\
            state_sample_single_sweep(
                locations, gates, untrys_as_matrixes,
                beta, A_train, A_val, B0_train, B0_val,
            )

        iteration_count += 1

        have_detected_plateau_in_curr_iter = jnp.abs(
            prev_training_cost - training_cost,
        ) <= diff_tol_r * jnp.abs(training_cost)

        plateau_window = jnp.concatenate(
            (
                jnp.array([have_detected_plateau_in_curr_iter]),
                plateau_window[:-1],
            ),
        )

        biggest_gate_dim = max(g.dim for g in gates)
        final_untrys_padded = jnp.array([
            _apply_padding_and_flatten(
                untry.numpy.flatten(
                ), gate, biggest_gate_dim,
            ) for untry, gate in zip(untrys_as_matrixes, gates)
        ])

        return (
            final_untrys_padded, training_cost, validation_cost,
            iteration_count, plateau_window,
        )

    while_body_vmaped = jax.vmap(_while_body_to_be_vmaped)

    initial_loop_var = (
        untrys,
        jnp.ones(amount_of_starts),  # train_cost
        jnp.ones(amount_of_starts),  # val_cost
        jnp.zeros(amount_of_starts, dtype=int),  # iter_count
        np.zeros((amount_of_starts, plateau_windows_size), dtype=bool),
    )

    if 'PRINT_LOSS_QFACTOR' in os.environ:
        loop_var = initial_loop_var
        i = 1
        while should_continue(loop_var):
            loop_var = while_body_vmaped(loop_var)

            untrys, training_costs, validation_costs, iteration_counts, \
                plateau_windows = loop_var
            _logger.debug(f'TRAINLOSS{i}: {training_costs}')
            _logger.debug(f'VALLOSS{i}: {validation_costs}')
            i += 1
        r = loop_var
    else:
        r = jax.lax.while_loop(
            should_continue, while_body_vmaped, initial_loop_var,
        )

    final_untrys, training_costs, validation_costs, iteration_counts, \
        plateau_windows = r

    return (
        final_untrys, training_costs, validation_costs,
        iteration_counts, plateau_windows,
    )


def state_sample_single_sweep(
    locations: tuple[CircuitLocation, ...],
    gates: tuple[Gate, ...],
    untrys: list[UnitaryMatrixJax], beta: float,
    A_train: RHSTensor, A_val: RHSTensor,
    B0_train: LHSTensor, B0_val: LHSTensor,
) -> tuple[list[UnitaryMatrixJax], float, float]:

    amount_of_gates = len(gates)
    B = [B0_train]
    for location, utry in zip(locations[:-1], untrys[:-1]):
        B.append(B[-1].copy())
        B[-1].apply_right(utry, location)

    # iterate over every gate from right to left and update it
    new_untrys_rev: list[UnitaryMatrixJax] = []
    a_train: RHSTensor = A_train.copy()
    a_val: RHSTensor = A_val.copy()
    for idx in reversed(range(amount_of_gates)):
        b = B[idx]
        gate = gates[idx]
        location = locations[idx]
        utry = untrys[idx]
        if gate.num_params > 0:
            env = SingleLegSideTensor.calc_env(b, a_train, location)
            utry = gate.optimize(
                env.T, get_untry=True,
                prev_untry=utry, beta=beta,
            )

        new_untrys_rev.append(utry)
        a_train.apply_left(utry, location)
        a_val.apply_left(utry, location)

    untrys = new_untrys_rev[::-1]

    training_cost = calc_cost(A_train, B0_train, a_train)

    validation_cost = calc_cost(A_val, B0_val, a_val)

    return untrys, training_cost, validation_cost


def calc_cost(A: RHSTensor, B0: LHSTensor, a: RHSTensor) -> float:
    cost = 2 * (
        1 - jnp.real(
            SingleLegSideTensor.calc_env(B0, a, [])[0],
        ) / A.single_leg_radix
    )

    return jnp.squeeze(cost)


if 'NO_JIT_QFACTOR' in os.environ or 'PRINT_LOSS_QFACTOR' in os.environ:
    _jited_loop_vmaped_state_sample_sweep = _loop_vmaped_state_sample_sweep
else:
    _jited_loop_vmaped_state_sample_sweep = jax.jit(
        _loop_vmaped_state_sample_sweep, static_argnums=(
            1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13,
        ),
    )
