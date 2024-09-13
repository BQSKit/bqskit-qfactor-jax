"""
Numerical Instantiation is the foundation of many of BQSKit's algorithms.

This is the same instantiation example as in BQSKit using the GPU implementation
of QFactor
"""
from __future__ import annotations

import numpy as np
from bqskit import Circuit
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.qis.unitary import UnitaryMatrix

from qfactorjax.qfactor_sample_jax import QFactorSampleJax


def run_toffoli_instantiation(dist_tol_requested: float = 1e-10) -> float:
    qfactor_gpu_instantiator = QFactorSampleJax(

        dist_tol=dist_tol_requested,       # Stopping criteria for distance

        max_iters=100000,      # Maximum number of iterations
        min_iters=10,          # Minimum number of iterations

        # Regularization parameter - [0.0 - 1.0]
        # Increase to overcome local minima at the price of longer compute
        beta=0.0,

        amount_of_validation_states=2,
        num_params_coef=1,
        overtrain_relative_threshold=0.1,
    )

    # We will optimize towards the Toffoli unitary.
    toffoli = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ])
    toffoli = UnitaryMatrix(toffoli)

    # Start with the circuit structure
    circuit = Circuit(3)
    circuit.append_gate(VariableUnitaryGate(2), [1, 2])
    circuit.append_gate(VariableUnitaryGate(2), [0, 2])
    circuit.append_gate(VariableUnitaryGate(2), [1, 2])
    circuit.append_gate(VariableUnitaryGate(2), [0, 2])
    circuit.append_gate(VariableUnitaryGate(2), [0, 1])

    # Instantiate the circuit template with QFactor
    circuit.instantiate(
        toffoli,
        multistarts=16,
        method=qfactor_gpu_instantiator,
    )

    # Calculate and print final distance
    dist = circuit.get_unitary().get_distance_from(toffoli, 1)
    return dist


if __name__ == '__main__':
    dist = run_toffoli_instantiation()
    print('Final Distance: ', dist)
