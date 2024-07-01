from __future__ import annotations

from random import randint
from random import sample

import jax
import jax.numpy as jnp
import pytest
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import HGate
from scipy.stats import unitary_group

from qfactorjax.singlelegedtensor import LHSTensor
from qfactorjax.singlelegedtensor import RHSTensor
from qfactorjax.singlelegedtensor import SingleLegSideTensor
from qfactorjax.unitarymatrixjax import UnitaryMatrixJax


jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize(
    'num_qubits, N',
    [(randint(2, 7), randint(3, 10)) for _ in range(6)],
)
def test_full_contraction_with_complex_conj(num_qubits: int, N: int) -> None:
    random_kets = [unitary_group.rvs(2**num_qubits)[:, :1] for _ in range(N)]
    random_bras = [ket.T.conj() for ket in random_kets]

    rhs = RHSTensor(list_of_states=random_bras, num_qudits=num_qubits)
    lhs = LHSTensor(list_of_states=random_kets, num_qudits=num_qubits)

    res = SingleLegSideTensor.calc_env(lhs, rhs, []).reshape(1)[0]

    assert jnp.isclose(res, N)


@pytest.mark.parametrize(
    'num_qubits, N',
    [(randint(2, 7), randint(3, 10)) for _ in range(3)],
)
def test_apply_left_H(num_qubits: int, N: int) -> None:
    random_kets = [unitary_group.rvs(2**num_qubits)[:, :1] for _ in range(N)]
    random_bras = [ket.T.conj() for ket in random_kets]

    rhs = RHSTensor(list_of_states=random_bras, num_qudits=num_qubits)
    orig = rhs.copy()

    H_mat = UnitaryMatrixJax(HGate().get_unitary())
    location = sorted(sample(range(num_qubits), 1))
    rhs.apply_left(H_mat, location)
    assert not all(jnp.isclose(rhs.tensor, orig.tensor).reshape(-1))
    rhs.apply_left(H_mat, location)
    assert all(jnp.isclose(rhs.tensor, orig.tensor).reshape(-1))


@pytest.mark.parametrize(
    'num_qubits, N',
    [(randint(2, 7), randint(3, 10)) for _ in range(3)],
)
def test_apply_right_H(num_qubits: int, N: int) -> None:
    random_kets = [unitary_group.rvs(2**num_qubits)[:, :1] for _ in range(N)]
    random_bras = [ket.T.conj() for ket in random_kets]

    rhs = LHSTensor(list_of_states=random_bras, num_qudits=num_qubits)
    orig = rhs.copy()

    H_mat = UnitaryMatrixJax(HGate().get_unitary())
    location = sorted(sample(range(num_qubits), 1))
    rhs.apply_right(H_mat, location)
    assert not all(jnp.isclose(rhs.tensor, orig.tensor).reshape(-1))
    rhs.apply_right(H_mat, location)
    assert all(jnp.isclose(rhs.tensor, orig.tensor).reshape(-1))


@pytest.mark.parametrize(
    'num_qubits, N',
    [(randint(2, 7), randint(3, 10)) for _ in range(3)],
)
def test_apply_left_CX(num_qubits: int, N: int) -> None:
    random_kets = [unitary_group.rvs(2**num_qubits)[:, :1] for _ in range(N)]
    random_bras = [ket.T.conj() for ket in random_kets]

    rhs = RHSTensor(list_of_states=random_bras, num_qudits=num_qubits)
    orig = rhs.copy()

    H_mat = UnitaryMatrixJax(CXGate().get_unitary())
    location = sorted(sample(range(num_qubits), 2))
    rhs.apply_left(H_mat, location)
    assert not all(jnp.isclose(rhs.tensor, orig.tensor).reshape(-1))
    rhs.apply_left(H_mat, location)
    assert all(jnp.isclose(rhs.tensor, orig.tensor).reshape(-1))


@pytest.mark.parametrize(
    'num_qubits, N',
    [(randint(2, 7), randint(3, 10)) for _ in range(3)],
)
def test_apply_right_CX(num_qubits: int, N: int) -> None:
    random_kets = [unitary_group.rvs(2**num_qubits)[:, :1] for _ in range(N)]
    random_bras = [ket.T.conj() for ket in random_kets]

    rhs = LHSTensor(list_of_states=random_bras, num_qudits=num_qubits)
    orig = rhs.copy()

    H_mat = UnitaryMatrixJax(CXGate().get_unitary())
    location = sorted(sample(range(num_qubits), 2))
    rhs.apply_right(H_mat, location)
    assert not all(jnp.isclose(rhs.tensor, orig.tensor).reshape(-1))
    rhs.apply_right(H_mat, location)
    assert all(jnp.isclose(rhs.tensor, orig.tensor).reshape(-1))
