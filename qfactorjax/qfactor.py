from __future__ import annotations

import functools
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
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.opt.instantiater import Instantiater
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number
from jax import Array
from scipy.stats import unitary_group

from qfactorjax.unitary_acc import VariableUnitaryGateAcc
from qfactorjax.unitarybuilderjax import UnitaryBuilderJax
from qfactorjax.unitarymatrixjax import UnitaryMatrixJax

if TYPE_CHECKING:
    from bqskit.qis.state.state import StateLike
    from bqskit.qis.state.system import StateSystemLike
    from bqskit.qis.unitary.unitarymatrix import UnitaryLike

_logger = logging.getLogger('bqskit.instant.qf-jax')

jax.config.update('jax_enable_x64', True)


class QFactorJax(Instantiater):
    """The QFactor batch circuit instantiater."""

    def __init__(
        self,
        dist_tol: float = 1e-10,
        diff_tol_a: float = 1e-12,
        diff_tol_r: float = 1e-6,
        max_iters: int = 100000,
        min_iters: int = 1000,
        reset_iter: int = 40,
        plateau_windows_size: int = 8,
        diff_tol_step_r: float = 0.1,
        diff_tol_step: int = 200,
        beta: float = 0.0,
    ):
        """
        Constructs and configures a QFactor Instantiator using JAX.

        Parameters:
            dist_tol (float): Allowed distance between the target unitary and
                the parameterized circuit.
            diff_tol_a (float): Minimum absolute improvement used in the
                short plateau detection mechanism.
            diff_tol_r (float): Minimum relative improvement used in the
                short plateau detection mechanism.
            max_iters (int): Maximum number of iterations.
            min_iters (int): Minimum number of iterations before stopping.
            reset_iter (int): Number of iterations before resetting the unitary
                 tensor to avoid floating point issues.
            plateau_window_size (int): Window size used to detect plateaus -
                every multistart needs to have at least one iteration flagged
                as a plateau, to stop the instantiation.
            diff_tol_step_r (float): Minimum relative improvement used in the
                long plateau detection mechanism.
            diff_tol_step (int): Number of iterations where the long plateau
                detection mechanism is used.
            beta (float): Regularization parameter. Valid values [0.0 - 1.0].
        """

        if not is_real_number(diff_tol_a):
            raise TypeError(
                f'Expected float for diff_tol_a, got {type(diff_tol_a)}.',
            )

        if diff_tol_a > 0.5:
            raise ValueError(
                'Invalid absolute difference threshold, must be less'
                f' than 0.5, got {diff_tol_a}.',
            )

        # TODO: Fix rest of input validation prints to be more informative
        if not is_real_number(diff_tol_r) or diff_tol_r > 0.5:
            raise TypeError('Invalid relative difference threshold.')

        if not is_real_number(dist_tol) or dist_tol > 0.5:
            raise TypeError('Invalid distance threshold.')

        if not is_integer(max_iters) or max_iters < 0:
            raise TypeError('Invalid maximum number of iterations.')

        if not is_integer(min_iters) or min_iters < 0:
            raise TypeError('Invalid minimum number of iterations.')

        if not min_iters < max_iters:
            raise TypeError(
                'Minimum number of iterations must be smaller then'
                'maximum number of iterations.',
            )

        if not is_integer(reset_iter):
            raise TypeError('Invalid minimum number of iterations.')

        if not is_integer(plateau_windows_size) or plateau_windows_size < 0:
            raise TypeError('Invalid plateau windows size of iterations.')

        if not is_real_number(diff_tol_step_r) or diff_tol_step_r < 0:
            raise TypeError(
                'Invalid relative difference threshold for long'
                ' plateau detection.',
            )

        if not is_integer(diff_tol_step) or diff_tol_step < 0:
            raise TypeError('Invalid step size of long plateau detection.')

        if not is_real_number(beta) or beta < 0:
            raise TypeError('Invalid beta parameter')

        self.diff_tol_a = diff_tol_a
        self.diff_tol_r = diff_tol_r
        self.dist_tol = dist_tol
        self.max_iters = max_iters
        self.min_iters = min_iters
        self.reset_iter = reset_iter
        self.plateau_windows_size = plateau_windows_size
        self.diff_tol_step_r = diff_tol_step_r
        self.diff_tol_step = diff_tol_step
        self.beta = beta

    def instantiate(
        self,
        circuit: Circuit,
        target: UnitaryLike,
        x0: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Instantiate `circuit' and return optimal params (single-start)."""
        return self.multi_start_instantiate(circuit, target, 1)

    def multi_start_instantiate_inplace(
        self,
        circuit: Circuit,
        target: UnitaryLike,
        num_starts: int,
    ) -> None:
        """
        Instantiate `circuit` to best implement `target` with multiple starts.

        See :func:`multi_start_instantiate` for more info.

        Notes:
            This method is a version of :func:`multi_start_instantiate`
            that modifies `circuit` in place rather than returning a copy.
        """
        params = self.multi_start_instantiate(circuit, target, num_starts)
        circuit.set_params(params)

    async def multi_start_instantiate_async(
        self,
        circuit: Circuit,
        target: UnitaryLike,
        num_starts: int,
    ) -> npt.NDArray[np.float64]:
        """Instantiate `circuit' and return optimal params asynchronously."""
        return self.multi_start_instantiate(circuit, target, num_starts)

    def multi_start_instantiate(
        self,
        circuit: Circuit,
        target: UnitaryLike | StateLike | StateSystemLike,
        num_starts: int,
    ) -> npt.NDArray[np.float64]:
        """Instantiate `circuit' and return optimal params."""
        if len(circuit) == 0:
            return np.array([])

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

        target = UnitaryMatrixJax(self.check_target(target))
        locations = tuple([op.location for op in circuit])
        gates = tuple(gates_list)
        biggest_gate_dim = max(g.dim for g in circuit.gate_set)

        untrys = []

        for gate in gates:
            size_of_untry = gate.dim

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
        res_var = _sweep2_jited(
            target, locations, gates, untrys, self.reset_iter, self.dist_tol,
            self.diff_tol_a, self.diff_tol_r, self.plateau_windows_size,
            self.max_iters, self.min_iters, num_starts, self.diff_tol_step_r,
            self.diff_tol_step, self.beta,
        )

        it = res_var['iteration_counts'][0]
        c1s = res_var['c1s']
        untrys = res_var['untrys']
        best_start = jnp.argmin(jnp.abs(c1s))

        if any(res_var['curr_reached_required_tol_l']):
            _logger.debug(
                f'Terminated: {it} c1 = {c1s} <= dist_tol.\n'
                f'Best start is {best_start}',
            )
        elif all(res_var['curr_plateau_calc_l']):
            _logger.debug(
                'Terminated: |c1 - c2| = '
                ' <= diff_tol_a + diff_tol_r * |c1|.',
            )

            _logger.debug(
                f'Terminated: {it} c1 = {c1s} Reached plateau.\n'
                f'Best start is {best_start}',
            )
        elif all(res_var['curr_step_calc_l']):
            _logger.debug(
                'Terminated: |prev_step_c1| - |c1| '
                ' <= diff_tol_step_r * |prev_step_c1|.',
            )
            _logger.debug(
                f'Terminated: {it} c1 = {c1s} Reached plateau.\n'
                f'Best start is {best_start}',
            )

        elif it >= self.max_iters:
            _logger.debug('Terminated: iteration limit reached.')

        else:
            _logger.error(
                f'Terminated with no good reason after {it} iteration '
                f'with c1s {c1s}.',
            )
        params: list[Array] = []
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
                    VariableUnitaryGateAcc,
                    ConstantGate,
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


def _initialize_circuit_tensor(
    target_num_qudits: int,
    target_radixes: Sequence[int],
    locations: Sequence[CircuitLocation],
    target_mat: Array,
    untrys: list[Array],
) -> UnitaryBuilderJax:

    target_untry_builder = UnitaryBuilderJax(
        target_num_qudits, target_radixes, target_mat.conj().T,
    )

    for loc, untry in zip(locations, untrys):
        target_untry_builder.apply_right(
            untry, loc, check_arguments=False,
        )

    return target_untry_builder


def _single_sweep(
    locations: tuple[CircuitLocation, ...], gates: tuple[Gate, ...],
    amount_of_gates: int, target_untry_builder: UnitaryBuilderJax,
    untrys: list[Array], beta: float = 0.0,
) -> tuple[UnitaryBuilderJax, Array]:
    # from right to left
    for k in reversed(range(amount_of_gates)):
        gate = gates[k]
        location = locations[k]
        untry = untrys[k]

        # Remove current gate from right of circuit tensor
        target_untry_builder.apply_right(
            untry, location, inverse=True, check_arguments=False,
        )

        # Update current gate
        if gate.num_params > 0:
            env = target_untry_builder.calc_env_matrix(location)
            untry = gate.optimize(
                env, get_untry=True, prev_untry=untry,
                beta=beta,
            )
            untrys[k] = untry

            # Add updated gate to left of circuit tensor
        target_untry_builder.apply_left(
            untry, location, check_arguments=False,
        )

        # from left to right
    for k in range(amount_of_gates):
        gate = gates[k]
        location = locations[k]
        untry = untrys[k]

        # Remove current gate from left of circuit tensor
        target_untry_builder.apply_left(
            untry, location, inverse=True, check_arguments=False,
        )

        # Update current gate
        if gate.num_params > 0:
            env = target_untry_builder.calc_env_matrix(location)
            untry = gate.optimize(env, True, beta, untry)
            untrys[k] = untry

            # Add updated gate to right of circuit tensor
        target_untry_builder.apply_right(
            untry, location, check_arguments=False,
        )

    return target_untry_builder, untrys


def _single_sweep_sim(
    locations: tuple[CircuitLocation, ...], gates: tuple[Gate, ...],
    amount_of_gates: int, target_untry_builder: UnitaryBuilderJax,
    untrys: list[Array], beta: float = 0.0,
) -> list[Array]:

    new_untrys = []
    # from right to left
    for k in reversed(range(amount_of_gates)):
        gate = gates[k]
        location = locations[k]
        untry = untrys[k]

        # Remove current gate from right of circuit tensor
        target_untry_builder.apply_right(
            untry, location, inverse=True, check_arguments=False,
        )

        # Update current gate
        if gate.num_params > 0:
            env = target_untry_builder.calc_env_matrix(location)
            new_untrys.append(gate.optimize(env, True, beta, untry))
        else:
            new_untrys.append(untry)

        target_untry_builder.apply_left(
            untry, location, check_arguments=False,
        )

    return new_untrys[::-1]


def _apply_padding_and_flatten(
        untry: Array, gate: Gate, max_gate_dim: int,
) -> Array:
    zero_pad_size = (max_gate_dim)**2 - (gate.dim)**2
    if zero_pad_size > 0:
        zero_pad = jnp.zeros(zero_pad_size)
        return jnp.concatenate((untry, zero_pad), axis=None)
    else:
        return jnp.array(untry.flatten())


def _remove_padding_and_create_matrix(
        untry: Array, gate: Gate,
) -> Array:
    size_to_keep = gate.dim**2
    return untry[:size_to_keep].reshape((gate.dim, gate.dim))


def Loop_vars(
    untrys: list[Array], c1s: Array, plateau_windows: Array,
    curr_plateau_calc_l: Array, curr_reached_required_tol_l: Array,
    iteration_counts: Array, target_untry_builders: Array, prev_step_c1s: Array,
    curr_step_calc_l: Array,
) -> dict[str, Array]:
    d = {}
    d['untrys'] = untrys
    d['c1s'] = c1s
    d['plateau_windows'] = plateau_windows
    d['curr_plateau_calc_l'] = curr_plateau_calc_l
    d['curr_reached_required_tol_l'] = curr_reached_required_tol_l
    d['iteration_counts'] = iteration_counts
    d['target_untry_builders'] = target_untry_builders
    d['prev_step_c1s'] = prev_step_c1s
    d['curr_step_calc_l'] = curr_step_calc_l

    return d


def _sweep2(
    target: UnitaryMatrixJax,
    locations: tuple[CircuitLocation, ...],
    gates: tuple[Gate, ...],
    untrys: Array,
    n: int,
    dist_tol: float,
    diff_tol_a: float,
    diff_tol_r: float,
    plateau_windows_size: int,
    max_iters: int,
    min_iters: int,
    amount_of_starts: int,
    diff_tol_step_r: float,
    diff_tol_step: int,
    beta: float,
) -> dict[str, Array]:
    c1s = jnp.array([1.0] * amount_of_starts)
    plateau_windows = jnp.array(
        [[0] * plateau_windows_size for
         _ in range(amount_of_starts)], dtype=bool,
    )

    prev_step_c1s = jnp.array([1.0] * amount_of_starts)

    def should_continue(var: dict[str, Array]) -> bool:
        return jnp.logical_not(
            jnp.logical_or(
                jnp.any(var['curr_reached_required_tol_l']),
                jnp.logical_or(
                    var['iteration_counts'][0] > max_iters,
                    jnp.logical_and(
                        var['iteration_counts'][0] > min_iters,
                        jnp.logical_or(
                            jnp.all(var['curr_plateau_calc_l']),
                            jnp.all(var['curr_step_calc_l']),
                        ),
                    ),
                ),
            ),
        )

    def _while_body_to_be_vmaped(
        untrys: Array, c1: float, plateau_window: Array,
        curr_plateau_calc: bool, curr_reached_required_tol: bool,
        iteration_count: int, target_untry_builder_tensor: Array,
        prev_step_c1: float, curr_step_calc: float,
    ) -> tuple[Array, float, Array, bool, bool, int, Array, float, float]:
        amount_of_gates = len(gates)
        amount_of_qudits = target.num_qudits
        target_radixes = target.radixes

        untrys_as_matrixes = []
        for gate_index, gate in enumerate(gates):
            untrys_as_matrixes.append(
                UnitaryMatrixJax(
                    _remove_padding_and_create_matrix(
                        untrys[gate_index], gate,
                    ), gate.radixes,
                ),
            )
        untrys = untrys_as_matrixes

        if 'All_ENVS' in os.environ:

            target_untry_builder_tensor = _initialize_circuit_tensor(
                amount_of_qudits, target_radixes, locations,
                target.numpy, untrys,
            ).tensor

            target_untry_builder = UnitaryBuilderJax(
                amount_of_qudits, target_radixes,
                tensor=target_untry_builder_tensor,
            )

            iteration_count = iteration_count + 1

            untrys = _single_sweep_sim(
                locations, gates, amount_of_gates, target_untry_builder,
                untrys, beta,
            )

            target_untry_builder_tensor = _initialize_circuit_tensor(
                amount_of_qudits, target_radixes, locations, target.numpy,
                untrys,
            ).tensor

            target_untry_builder = UnitaryBuilderJax(
                amount_of_qudits, target_radixes,
                tensor=target_untry_builder_tensor,
            )
        else:

            # initialize every "n" iterations of the loop
            operand_for_if_init = (untrys, target_untry_builder_tensor)
            initialize_body = lambda x: _initialize_circuit_tensor(
                amount_of_qudits, target_radixes, locations,
                target.numpy, x[0],
            ).tensor
            no_initialize_body = lambda x: x[1]

            target_untry_builder_tensor = jax.lax.cond(
                iteration_count % n == 0, initialize_body,
                no_initialize_body, operand_for_if_init,
            )

            target_untry_builder = UnitaryBuilderJax(
                amount_of_qudits, target_radixes,
                tensor=target_untry_builder_tensor,
            )

            iteration_count = iteration_count + 1

            target_untry_builder, untrys = _single_sweep(
                locations, gates, amount_of_gates, target_untry_builder,
                untrys, beta,
            )

        c2 = c1
        dim = target_untry_builder.dim
        untry_res = target_untry_builder.tensor.reshape((dim, dim))
        c1 = jnp.abs(jnp.trace(untry_res))
        c1 = 1 - (c1 / (2 ** amount_of_qudits))

        curr_plateau_part = jnp.abs(
            c1 - c2,
        ) <= diff_tol_a + diff_tol_r * jnp.abs(c1)
        curr_plateau_calc = functools.reduce(
            jnp.bitwise_or, plateau_window,
        ) | curr_plateau_part
        plateau_window = jnp.concatenate(
            (jnp.array([curr_plateau_part]), plateau_window[:-1]),
        )
        curr_reached_required_tol = c1 < dist_tol

        # Checking the plateau in a step
        operand_for_if_plateau = (c1, prev_step_c1, curr_step_calc)
        reached_step_body = lambda x: (
            jnp.abs(x[0]), (x[1] - jnp.abs(x[0])) < diff_tol_step_r * x[1],
        )
        not_reached_step_body = lambda x: (x[1], x[2])

        prev_step_c1, curr_step_calc = jax.lax.cond(
            (iteration_count + 1) % diff_tol_step == 0,
            reached_step_body, not_reached_step_body, operand_for_if_plateau,
        )

        biggest_gate_dim = max(gate.dim for gate in gates)
        final_untrys_padded = jnp.array([
            _apply_padding_and_flatten(
                untry.numpy.flatten(
                ), gate, biggest_gate_dim,
            ) for untry, gate in zip(untrys, gates)
        ])

        return (
            final_untrys_padded, c1, plateau_window, curr_plateau_calc,
            curr_reached_required_tol, iteration_count,
            target_untry_builder.tensor, prev_step_c1, curr_step_calc,
        )

    while_body_vmaped = jax.vmap(_while_body_to_be_vmaped)

    def while_body(var: dict[str, Array]) -> dict[str, Array]:
        return Loop_vars(
            *while_body_vmaped(
                var['untrys'], var['c1s'],
                var['plateau_windows'],
                var['curr_plateau_calc_l'],
                var['curr_reached_required_tol_l'],
                var['iteration_counts'],
                var['target_untry_builders'],
                var['prev_step_c1s'],
                var['curr_step_calc_l'],
            ),
        )

    dim = np.prod(target.radixes)
    initial_untray_builders_values = jnp.array([
        jnp.identity(
            dim, dtype=jnp.complex128,
        ).reshape(target.radixes * 2) for _ in range(amount_of_starts)
    ])

    initial_loop_var = Loop_vars(
        untrys, c1s, plateau_windows, jnp.array([False] * amount_of_starts),
        jnp.array([False] * amount_of_starts),
        jnp.array([0] * amount_of_starts),
        initial_untray_builders_values, prev_step_c1s,
        jnp.array([False] * amount_of_starts),
    )

    if 'PRINT_LOSS_QFACTOR' in os.environ:
        loop_var = initial_loop_var
        i = 1
        while should_continue(loop_var):
            loop_var = while_body(loop_var)

            print('LOSS:', i, loop_var['c1s'])
            i += 1
        res_var = loop_var
    else:
        res_var = jax.lax.while_loop(
            should_continue, while_body, initial_loop_var,
        )

    return res_var


if 'NO_JIT_QFACTOR' in os.environ:
    _sweep2_jited = _sweep2
else:
    _sweep2_jited = jax.jit(
        _sweep2, static_argnums=(
            1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
        ),
    )
