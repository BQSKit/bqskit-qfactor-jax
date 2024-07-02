from __future__ import annotations

import os

from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate

from examples.gate_deletion_syth import run_gate_del_flow_example
from examples.toffoli_instantiation import run_toffoli_instantiation
from examples.toffoli_instantiation_using_sampling import\
    run_toffoli_instantiation as run_toffoli_instantiation_using_sampling


def test_toffoli_instantiation() -> None:
    distance = run_toffoli_instantiation()
    assert distance <= 1e-10


def test_toffoli_instantiation_using_sampling() -> None:
    distance = run_toffoli_instantiation_using_sampling()
    assert distance <= 1e-10


def test_gate_del_synth() -> None:

    if 'AMOUNT_OF_WORKERS' in os.environ:
        amount_of_workers = int(os.environ['AMOUNT_OF_WORKERS'])
    else:
        amount_of_workers = 10

    in_circuit, out_circuit, run_time = run_gate_del_flow_example(
        amount_of_workers,
    )

    out_circuit_gates_count = out_circuit.gate_counts
    assert out_circuit_gates_count[CNOTGate()] == 44
    assert out_circuit_gates_count[U3Gate()] == 56
