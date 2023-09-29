"""
Numerical Instantiation is the foundation of many of BQSKit's algorithms.

This is the same instantiation example as in BQSKit using the GPU implementation
of QFactor
"""
from __future__ import annotations

import numpy as np
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.qis.unitary import UnitaryMatrix

from bqskitqfactorjax.qfactor_jax import QFactor_jax


def run_toffoli_instantiation(dist_tol_requested = 1e-10):
    qfactr_gpu_instantiator = QFactor_jax(

            dist_tol  = dist_tol_requested,       # Stopping criteria for distance

            max_iters = 100000,      # Maximum number of iterations
            min_iters = 10,          # Minimum number of iterations

            #One step plateau detection -
            #diff_tol_a + diff_tol_r ∗ |c(i)| <= |c(i)|-|c(i-1)|
            diff_tol_a = 0.0,       # Stopping criteria for distance change
            diff_tol_r = 1e-10,     # Relative criteria for distance change

            #Long plateau detection -
            # diff_tol_step_r*|c(i-diff_tol_step)| <= |c(i)|-|c(i-diff_tol_step)|
            diff_tol_step_r = 0.1, #The relative improvment expected
            diff_tol_step   = 200,   #The interval in which to check the improvment

            #Regularization parameter - [0.0 - 1.0]
            # Increase to overcome local minimumas at the price of longer compute
            beta = 0.0,
    )



    # We will optimize towards the toffoli unitary.
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


    # Instantiate the circuit template with qfactor
    circuit.instantiate(
        toffoli,
        multistarts = 16,
        method=qfactr_gpu_instantiator,
    )

    # Calculate and print final distance
    dist = circuit.get_unitary().get_distance_from(toffoli, 1)
    return dist

if __name__ == '__main__':
    dist = run_toffoli_instantiation()
    print('Final Distance: ', dist)
