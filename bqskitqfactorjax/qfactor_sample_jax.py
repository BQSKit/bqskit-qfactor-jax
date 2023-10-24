from __future__ import annotations

import logging
import os
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
from scipy.stats import unitary_group

from bqskitqfactorjax.qfactor_jax import _apply_padding_and_flatten
from bqskitqfactorjax.qfactor_jax import _remove_padding_and_create_matrix
from bqskitqfactorjax.singlelegedtensor import LHSTensor
from bqskitqfactorjax.singlelegedtensor import RHSTensor
from bqskitqfactorjax.singlelegedtensor import SingleLegSideTensor
from bqskitqfactorjax.unitary_acc import VariableUnitaryGateAcc
from bqskitqfactorjax.unitarymatrixjax import UnitaryMatrixJax

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.state.state import StateLike
    from bqskit.qis.state.system import StateSystemLike
    from bqskit.qis.unitary.unitarymatrix import UnitaryLike

_logger = logging.getLogger(__name__)

jax.config.update('jax_enable_x64', True)


class QFactorSampleJax(Instantiater):

    def __init__(
        self,
        dist_tol: float = 1e-10,
        max_iters: int = 100000,
        min_iters: int = 1000,
        beta: float = 0.0,
        amount_of_validation_states: int = 2,
        num_params_coef: float = 1.0,
        overtrain_relative_threshold: float = 0.1,
        diff_tol_r: float=1e-4,     # Relative criteria for distance change
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
        self.exact_amount_of_states_to_train_on = exact_amount_of_states_to_train_on

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

        # A very ugly casting
        for op in circuit:
            g = op.gate
            if isinstance(g, VariableUnitaryGate):
                g.__class__ = VariableUnitaryGateAcc

        target = UnitaryMatrixJax(self.check_target(target))
        locations = tuple([op.location for op in circuit])
        gates = tuple([op.gate for op in circuit])
        biggest_gate_size = max(gate.num_qudits for gate in gates)
        radixes = target.radixes
        num_qudits = target.num_qudits

        
        # Initial number of states to use can b
        if self.exact_amount_of_states_to_train_on is None:
            amount_of_params_in_circuit = 0
            for g in gates:
                amount_of_params_in_circuit +=  g.num_params
            initial_amount_of_training_states = int(
                np.sqrt(amount_of_params_in_circuit) * self.num_params_coef
            )
        else:
            initial_amount_of_training_states =\
                self.exact_amount_of_states_to_train_on

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
                        pre_padd, gate, biggest_gate_size,
                    ) for pre_padd in pre_padding_untrys
                ])

            untrys = jnp.array(np.stack(untrys, axis=1))
        
        have_sufficient_states = False
        should_double_the_training_size = True
        while not have_sufficient_states:
            
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
                            pre_padd, gate, biggest_gate_size,
                        ) for pre_padd in pre_padding_untrys
                    ])

                untrys = jnp.array(np.stack(untrys, axis=1))

            

            training_states_kets = self.generate_random_states(
                amount_of_training_states, int(np.prod(radixes)),
            )

            final_untrys, training_costs, validation_costs, iteration_counts, plateau_windows = \
                _jited_loop_vmaped_state_sample_sweep(
                    target, num_qudits, radixes, locations, gates, untrys,
                    self.dist_tol, self.max_iters, self.beta,
                    num_starts, self.min_iters, self.diff_tol_r,
                    self.plateau_windows_size, self.overtrain_ratio,
                    training_states_kets, validation_states_kets,
                )

            it = iteration_counts[0]
            untrys = final_untrys
            best_start = jnp.argmin(training_costs)

            have_sufficient_states = True
            if any(training_costs < self.dist_tol):
                _logger.debug(
                    f'Terminated: {it} c1 = {training_costs} <= dist_tol.\n'
                    f'Best start is {best_start}',
                )
            elif it >= self.max_iters:
                _logger.debug(f'Terminated {it}: iteration limit reached. c1 = {training_costs}')
            elif it > self.min_iters:
                if all(validation_costs - training_costs > self.overtrain_ratio * training_costs):
                    _logger.debug(
                        f'Terminated: {it} overtraining detected in all multistarts',
                    )
                    have_sufficient_states = False
                elif np.all(np.all(plateau_windows, axis=1)):
                    _logger.debug(
                        f'Terminated: {it} plateau detected in all multistarts c1 = {training_costs}',
                    )
            else:
                _logger.error(
                    f'Terminated with no good reason after {it} iterations '
                    f'with c1s {training_costs}.',
                )
        
            if should_double_the_training_size:
                amount_of_training_states *= 2
            else:
                amount_of_training_states += initial_amount_of_training_states

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
    diff_tol_r:float, plateau_windows_size:int,
    overtrain_ratio: float, training_states_kets: Array,
    validation_states_kets: Array,
) -> tuple[Array, Array[float], Array[float], Array[int]]:

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
            Array, Array[float], Array[float], Array[int],Array[bool]
        ],
    ) -> Array[bool]:
        _, training_costs, validation_costs, iteration_counts, plateau_windows = loop_var

        any_reached_required_tol = jnp.any(
            jax.vmap(
                lambda cost: cost <= dist_tol,
            )(training_costs),
        )

        reached_max_iteration = iteration_counts[0] > max_iters
        above_min_iteration = iteration_counts[0] > min_iters

        all_reached_over_training = jnp.all(
            validation_costs - training_costs > overtrain_ratio * training_costs,
        )

        all_reached_plateau =  jnp.all(
            jnp.all(plateau_windows, axis=1) 
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
                            all_reached_plateau
                        )
                    ),
                ),
            ),
        )

    def _while_body_to_be_vmaped(
        loop_var: tuple[
            Array, Array[float], Array[float], Array[int], Array[bool]
        ],
    ) -> tuple[
            Array, Array[float], Array[float], Array[int],
    ]:

        untrys, training_cost, validation_cost, iteration_count, plateau_window = loop_var

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
            prev_training_cost - training_cost
        ) <= diff_tol_r * jnp.abs(training_cost)

        plateau_window = jnp.concatenate(
            (jnp.array([have_detected_plateau_in_curr_iter]), plateau_window[:-1]),
        )

        biggest_gate_size = max(gate.num_qudits for gate in gates)
        final_untrys_padded = jnp.array([
            _apply_padding_and_flatten(
                untry.numpy.flatten(
                ), gate, biggest_gate_size,
            ) for untry, gate in zip(untrys_as_matrixes, gates)
        ])

        return (
            final_untrys_padded, training_cost, validation_cost,
            iteration_count, plateau_window
        )

    while_body_vmaped = jax.vmap(_while_body_to_be_vmaped)

    initial_loop_var = (
        untrys,
        jnp.ones(amount_of_starts),  # train_cost
        jnp.ones(amount_of_starts),  # val_cost
        jnp.zeros(amount_of_starts, dtype=int),  # iter_count
        np.zeros((amount_of_starts, plateau_windows_size), dtype=bool)
    )

    if 'PRINT_LOSS_QFACTOR' in os.environ:
        loop_var = initial_loop_var
        i = 1
        while should_continue(loop_var):
            loop_var = while_body_vmaped(loop_var)
            
            untrys, training_costs, validation_costs, iteration_counts, plateau_windows = loop_var
            print(f'TRAINLOSS{i}:', training_costs)
            print(f'VALLOSS{i}:', validation_costs)
            i += 1
        r = loop_var
    else:
        r = jax.lax.while_loop(should_continue, while_body_vmaped, initial_loop_var)
    
    final_untrys, training_costs, validation_costs, iteration_counts, plateau_windows = r

    return final_untrys, training_costs, validation_costs, iteration_counts, plateau_windows


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
            1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13
        ),
    )
