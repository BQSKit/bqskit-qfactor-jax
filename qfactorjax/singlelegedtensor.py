from __future__ import annotations

from typing import Sequence
from typing import TypeVar

import jax.numpy as jnp
from bqskit.ir import CircuitLocation
from jax import Array

from qfactorjax.unitarymatrixjax import UnitaryMatrixJax


T = TypeVar('T', bound='SingleLegSideTensor')


class SingleLegSideTensor():
    """
    The class represents a tensor that has only a single leg in one of his
    sides.

    The single leg will always be index 0
    """

    def __init__(
        self, num_qudits: int, radixes: Sequence[int] = [],
        list_of_states: Array = jnp.array([]),
        tensor: Array | None = None, single_leg_radix: int | None = None,
    ) -> None:

        if len(list_of_states) > 0:
            first_state = list_of_states[0]

            assert any(d == 1 for d in first_state.shape)
            assert all(s.shape == first_state.shape for s in list_of_states)

            self.num_qudits = num_qudits
            self.num_of_legs = num_qudits + 1
            self.radixes = tuple(
                radixes if len(radixes) > 0 else [2] * num_qudits,
            )
            self.single_leg_radix = len(list_of_states)
            self.tensor = jnp.array(list_of_states).reshape(
                self.single_leg_radix, *self.radixes,
            )
        elif tensor is not None and single_leg_radix is not None:
            self.tensor = tensor
            self.single_leg_radix = single_leg_radix
            self.num_qudits = num_qudits
            self.num_of_legs = num_qudits + 1
            self.radixes = tuple(
                radixes if len(radixes) > 0 else [2] * num_qudits,
            )
        else:
            raise RuntimeError("can't create the instance")

    def copy(self: T) -> T:
        return self.__class__(
            tensor=self.tensor.copy(),
            num_qudits=self.num_qudits,
            radixes=self.radixes,
            single_leg_radix=self.single_leg_radix,
        )

    @staticmethod
    def calc_env(
        left: LHSTensor, right: RHSTensor,
        indexes_to_leave_open: Sequence[int],
    ) -> Array:

        # verify correct shape
        assert left.radixes == right.radixes
        assert left.single_leg_radix == right.single_leg_radix, \
            f'{left.single_leg_radix} != {right.single_leg_radix}'

        left_contraction_indexs = list(range(left.num_qudits + 1))
        right_contraction_indexs = list(range(left.num_qudits + 1))

        size_of_open = len(indexes_to_leave_open)

        for leg_num, i in enumerate(indexes_to_leave_open):
            left_contraction_indexs[i + 1] = size_of_open + \
                leg_num + left.num_of_legs
            right_contraction_indexs[i + 1] = leg_num + left.num_of_legs

        env_tensor = jnp.einsum(
            left.tensor, left_contraction_indexs,
            right.tensor, right_contraction_indexs,
        )
        env_mat = env_tensor.reshape((2**size_of_open, -1))

        return env_mat


class RHSTensor(SingleLegSideTensor):

    def apply_left(
            self,
            utry: UnitaryMatrixJax,
            location: Sequence[CircuitLocation],
    ) -> None:
        """
        Apply the specified unitary on the left of this rhs tensor

        ..
                 .------.   .-----.
              1 -|      |---|     |
              2 -| gate |---|     .
                 '------'   .     .
                            .     |- 0
                            .     .
              n ------------|     |
                            '-----'

        """

        utry_tensor = utry.get_tensor_format()
        utry_size = len(utry.radixes)

        utry_tensor_indexs = [
            i + self.num_of_legs for i in range(utry_size)
        ] + [1 + l for l in location]

        rhs_tensor_indexes = list(range(self.num_of_legs))
        output_tensor_index = list(range(self.num_of_legs))

        # matching the leg indexs of the utry
        for i, loc in enumerate(location):
            rhs_tensor_indexes[1 + loc] = i + self.num_of_legs

        self.tensor = jnp.einsum(
            utry_tensor, utry_tensor_indexs,
            self.tensor, rhs_tensor_indexes, output_tensor_index,
        )


class LHSTensor(SingleLegSideTensor):

    def apply_right(
            self,
            utry: UnitaryMatrixJax,
            location: Sequence[CircuitLocation],
    ) -> None:
        """
        Apply the specified unitary on the right of this lhs tensor.

        The reuslt looks like this          .-----.   .------.          | |---|
        |- 1          |     |---| utry |- 2          .     . '------'        0-|
        .          .     .          .     .          | |------------ n '-----'
        """

        utry_tensor = utry.get_tensor_format()
        utry_size = len(utry.radixes)

        utry_tensor_indexs = [1 + l for l in location] + \
            [i + self.num_of_legs for i in range(utry_size)]
        lhs_tensor_indexes = list(range(self.num_of_legs))
        output_tensor_index = list(range(self.num_of_legs))

        # matching the leg indexs of the utry
        for i, loc in enumerate(location):
            lhs_tensor_indexes[1 + loc] = i + self.num_of_legs

        self.tensor = jnp.einsum(
            utry_tensor, utry_tensor_indexs,
            self.tensor, lhs_tensor_indexes, output_tensor_index,
        )
