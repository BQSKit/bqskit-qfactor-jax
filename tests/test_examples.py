from bqskit.ir.gates import CNOTGate, U3Gate
from examples.toffoli_instantiation import run_toffoli_instantiation
from examples.gate_deletion_syth import run_gate_del_flow_example


def test_toffoli_instantiation():
    distance = run_toffoli_instantiation()
    assert distance <=1e-10


def test_gate_del_synth():
    in_circuit, out_circuit, run_time = run_gate_del_flow_example()

    out_circuit_gates_count = out_circuit.gate_counts
    assert out_circuit_gates_count[CNOTGate()] == 44
    assert out_circuit_gates_count[U3Gate()] == 56

