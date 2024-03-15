from __future__ import annotations

from typing import Any
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
import numpy.typing as npt
from bqskit.qis.unitary.unitary import Unitary
from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.docs import building_docs
from bqskit.utils.typing import is_square_matrix
from jax import Array

if not building_docs():
    from numpy.lib.mixins import NDArrayOperatorsMixin
else:
    class NDArrayOperatorsMixin:  # type: ignore
        pass


class UnitaryMatrixJax(NDArrayOperatorsMixin):
    def __init__(
        self,
        input: UnitaryLike,
        radixes: Sequence[int] = [],
        _from_tree: bool = False,
    ) -> None:

        # Stop any actual logic when building documentation
        if building_docs():
            self._utry = jnp.array([])
            return

        # Copy constructor
        if isinstance(input, (UnitaryMatrixJax, UnitaryMatrix)):
            self._utry = jnp.array(input.numpy)
            self._radixes = input.radixes
            self._dim = input.dim
            return

        if len(radixes) != 0:
            self._radixes = tuple(radixes)
        else:
            dim = len(input)
            self._dim = dim
            # Check if unitary dimension is a power of two
            if dim & (dim - 1) == 0:
                self._radixes = tuple([2] * int(np.round(np.log2(dim))))

        # Check if unitary dimension is a power of three
            elif 3 ** int(np.round(np.log(dim) / np.log(3))) == dim:  # noqa
                radixes = [3] * int(np.round(np.log(dim) / np.log(3)))
                self._radixes = tuple(radixes)

        if (
            type(input) is not object
            and type(input) is not jax.core.ShapedArray
                and not _from_tree
        ):
            dim = np.prod(self._radixes)
            self._utry = jnp.array(input, dtype=jnp.complex128).reshape(
                (dim, dim),
            )  # make sure its a square matrix
        else:
            self._utry = input

    @property
    def radixes(self) -> tuple[int, ...]:
        """The number of orthogonal states for each qudit."""
        return getattr(self, '_radixes')

    @property
    def dim(self) -> int:
        """The matrix dimension for this unitary."""
        if hasattr(self, '_dim'):
            return self._dim

        return int(np.prod(self.radixes))

    @property
    def num_params(self) -> int:
        """The number of real parameters this unitary-valued function takes."""
        return getattr(self, '_num_params')

    @property
    def num_qudits(self) -> int:
        """The number of qudits this unitary can act on."""
        if hasattr(self, '_num_qudits'):
            return self._num_qudits

        return len(self.radixes)

    @staticmethod
    def identity(dim: int, radixes: Sequence[int] = []) -> UnitaryMatrixJax:
        """
        Construct an identity UnitaryMatrix.

        Args:
            dim (int): The dimension of the identity matrix.

            radixes (Sequence[int]): The number of orthogonal states
                for each qudit, defaults to qubits.

        Returns:
            UnitaryMatrixJax: An identity matrix.

        Raises:
            ValueError: If `dim` is non-positive.
        """
        if dim <= 0:
            raise ValueError('Invalid dimension for identity matrix.')
        return UnitaryMatrixJax(jnp.identity(dim), radixes)

    @staticmethod
    def closest_to(
        M: npt.NDArray[np.complex128],
        radixes: Sequence[int] = [],
    ) -> UnitaryMatrixJax:
        """
        Calculate and return the closest unitary to a given matrix.

        Calculate the unitary matrix U that is closest with respect to the
        operator norm distance to the general matrix M.

        Args:
            M (np.ndarray): The matrix input.

            radixes (Sequence[int]): The radixes for the Unitary.

        Returns:
            (UnitaryMatrix): The unitary matrix closest to M.

        References:
            D.M.Reich. “Characterisation and Identification of Unitary Dynamics
            Maps in Terms of Their Action on Density Matrices”
        """
        if not is_square_matrix(M):
            raise TypeError('Expected square matrix.')

        V, _, Wh = jla.svd(M)

        return UnitaryMatrixJax(V @ Wh, radixes)

    @property
    def numpy(self) -> Array:
        """The JaxNumPy array holding the unitary."""
        return self._utry

    @property
    def jaxnumpy(self) -> Array:
        """The JaxNumPy array holding the unitary."""
        return self._utry

    @staticmethod
    def random(
        num_qudits: int,
        radixes: Sequence[int] = [],
    ) -> UnitaryMatrixJax:
        return UnitaryMatrixJax(UnitaryMatrix.random(num_qudits, radixes))

    @staticmethod
    def from_file(filename: str) -> UnitaryMatrixJax:
        """Load a unitary from a file."""
        return UnitaryMatrixJax(jnp.loadtxt(filename, dtype=jnp.complex128))

    @property
    def T(self) -> UnitaryMatrixJax:
        """The transpose of the unitary."""
        return UnitaryMatrixJax(self._utry.T, self.radixes)

    def conj(self) -> UnitaryMatrixJax:
        """Return the complex conjugate unitary matrix."""
        return UnitaryMatrixJax(self._utry.conj(), self.radixes)

    def get_unitary(self, params: Any) -> UnitaryMatrixJax:
        """Return the same object, satisfies the :class:`Unitary` API."""
        return self

    @property
    def dagger(self) -> UnitaryMatrixJax:
        """The conjugate transpose of the unitary."""
        return self.conj().T

    def get_tensor_format(self) -> Array:
        """
        Converts the unitary matrix operation into a tensor network format.

        Indices are counted top to bottom, right to left:
                 .-----.
              n -|     |- 0
            n+1 -|     |- 1
                 .     .
                 .     .
                 .     .
           2n-1 -|     |- n-1
                 '-----'

         Returns     Array: A tensor representing this matrix.
        """

        return self._utry.reshape(self.radixes + self.radixes)

    def __eq__(self, other: object) -> bool:
        """Check if `self` is approximately equal to `other`."""
        if isinstance(other, Unitary):
            other_unitary = other.get_unitary()
            if self._utry.shape != other_unitary.shape:
                return False
            return np.allclose(self, other_unitary)

        if isinstance(other, np.ndarray):
            return np.allclose(self, other)

        return NotImplemented

    def __array__(
        self,
        dtype: np.typing.DTypeLike = jnp.complex128,
    ) -> Array:
        """Implements NumPy API for the UnitaryMatrix class."""
        if dtype != jnp.complex128:
            raise ValueError(
                'UnitaryMatrixJax only supports JAX-Complex128 dtype.',
            )

        return self._utry

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Array,
        **kwargs: dict[str, Any],
    ) -> UnitaryMatrixJax | Array[jnp.complex128]:
        """Implements NumPy API for the UnitaryMatrix class."""
        if method != '__call__':
            return NotImplemented

        non_unitary_involved = False
        args: list[npt.NDArray[Any]] = []
        for input in inputs:
            if isinstance(input, UnitaryMatrixJax):
                args.append(input.numpy)
            else:
                args.append(input)
                non_unitary_involved = True

        out = ufunc(*args, **kwargs)

        # The results are unitary
        # if only unitaries are involved
        # and unitaries are closed under the specific operation.
        convert_back = (
            not non_unitary_involved and (
                ufunc.__name__ == 'conjugate'
                or ufunc.__name__ == 'matmul'
                or ufunc.__name__ == 'negative'
                or ufunc.__name__ == 'positive'
            )
            or (
                ufunc.__name__ == 'multiply'
                and all(
                    jnp.isscalar(input) or isinstance(
                        input, UnitaryMatrixJax,
                    )
                    for input in inputs
                )
                and all(
                    jnp.abs(jnp.abs(input) - 1) <= 1e-14
                    for input in inputs if jnp.isscalar(input)
                )
            )
        )

        if convert_back:
            return UnitaryMatrixJax(out, self.radixes)

        return out

    def _tree_flatten(
            self,
    ) -> tuple[tuple[Array], dict[str, Any]]:
        children = (self._utry,)  # arrays / dynamic values
        aux_data = {
            'radixes': self._radixes,
            '_from_tree': True,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(
        cls, aux_data: dict[str, Any],
        children: tuple[Array],
    ) -> UnitaryMatrixJax:
        return cls(*children, **aux_data)


jax.tree_util.register_pytree_node(
    UnitaryMatrixJax,
    UnitaryMatrixJax._tree_flatten,
    UnitaryMatrixJax._tree_unflatten,
)
