"""
Numerical Instantiation is the foundation of many of BQSKit's algorithms.

This example demonstrates building a circuit template that can implement the
toffoli gate and then instantiating it to be the gate.
"""
from __future__ import annotations

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.qis.unitary import UnitaryMatrix

from bqskitgpu.qfactor_jax import QFactor_jax



qfactr_gpu_instantiator = QFactor_jax(diff_tol_a = 0.0,
        diff_tol_r = 1e-10,
        dist_tol = 1e-10,
        max_iters= 100000,
        min_iters = 10,
        diff_tol_step_r = 0.1,
        diff_tol_step = 200,
        beta = 0.0)



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
    # diff_tol_a=1e-12,   # Stopping criteria for distance change
    # diff_tol_r=1e-6,    # Relative criteria for distance change
    # dist_tol=1e-10,     # Stopping criteria for distance
    # max_iters=100000,   # Maximum number of iterations
    # min_iters=10,     # Minimum number of iterations
    # beta=0,   # Larger numbers slowdown optimization
    # # to avoid local minima
)

# Calculate and print final distance
dist = circuit.get_unitary().get_distance_from(toffoli, 1)
print('Final Distance: ', dist)

# You can use synthesis to convert the `VariableUnitaryGate`s to
# native gates. Alternatively, you can build a circuit directly out of
# native gates and use the default instantiater to instantiate directly.
