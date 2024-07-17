from __future__ import annotations

from timeit import default_timer as timer

import numpy as np
from bqskit import Circuit
from bqskit.compiler import Compiler
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import SimpleLayerGenerator
from bqskit.qis.unitary import UnitaryMatrix

from qfactorjax.qfactor_sample_jax import QFactorSampleJax


num_multistarts = 32


qfactor_sample_gpu_instantiator = QFactorSampleJax(

    dist_tol=1e-8,       # Stopping criteria for distance

    max_iters=100000,      # Maximum number of iterations
    min_iters=6,          # Minimum number of iterations

    # Regularization parameter - [0.0 - 1.0]
    # Increase to overcome local minima at the price of longer compute
    beta=0.0,
    amount_of_validation_states=2,

    diff_tol_r=1e-4,
    overtrain_relative_threshold=0.1,
)


instantiate_options = {
    'method': qfactor_sample_gpu_instantiator,
    'multistarts': num_multistarts,
}

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
in_circuit = Circuit.from_unitary(toffoli)


search_layer_gen = SimpleLayerGenerator(
    two_qudit_gate=CNOTGate(), single_qudit_gate_1=VariableUnitaryGate(1),
)

passes = [
    QSearchSynthesisPass(
        layer_generator=search_layer_gen,
        instantiate_options=instantiate_options,
    ),
]

with Compiler('localhost') as compiler:
    start = timer()
    out_circuit = compiler.compile(in_circuit, passes)
    end = timer()
    run_time = end - start


dist = out_circuit.get_unitary().get_distance_from(in_circuit.get_unitary(), 1)

print(
    f'QSearch took {run_time} '
    f'seconds using QFactor-Sample JAX instantiation method.',
)

print(
    f'Circuit finished with gates: {out_circuit.gate_counts}, '
    f'and the distance is {dist}',
)
