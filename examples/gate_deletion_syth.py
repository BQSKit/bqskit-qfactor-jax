"""
This example shows how to resynthesize a circuit using a gate deletion flow,
that utilizses Qfacto's GPU implementation

"""

import logging
import os
from timeit import default_timer as timer
from bqskit import Circuit, compile
from bqskit.compiler import Compiler, CompilationTask
from bqskit.passes import QuickPartitioner
from bqskit.passes import ForEachBlockPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import UnfoldPass
from bqskit.passes import ToU3Pass
# from bqskit.passes import FromU3ToVariablePass

#####################################################################
# Remove when this pass is in bqskit

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.point import CircuitPoint

class FromU3ToVariablePass(BasePass):
    """Converts U3 Gates to single-qubit variable unitary gates."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        # _logger.debug('Converting U3Gates to VariableUnitaryGate')
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, U3Gate):
                params = VariableUnitaryGate.get_params(op.get_unitary())
                point = CircuitPoint(cycle, op.location[0])
                vgate = VariableUnitaryGate(op.num_qudits, op.radixes)
                circuit.replace_gate(
                    point, vgate, op.location, params,
                )

#####################################################################

from bqskitgpu.qfactor_jax import QFactor_jax



def run_gate_del_flow_example():
        # The circuit to resynthesize
        file_path = os.path.dirname(__file__) + '/grover5.qasm'

        # Compiler runtime configuration
        amount_of_workers = 10

        # Set the size of paritions
        partition_size = 4

        # QFactor hyperparameters - 
        # see intantiation example for more detiles on the parameters
        num_multistarts = 32
        max_iters = 100000
        min_iters = 3
        diff_tol_r = 1e-5
        diff_tol_a = 0.0
        dist_tol = 1e-10

        diff_tol_step_r = 0.1
        diff_tol_step = 200

        beta = 0



        print(f"Will compile {file_path}")

        # Read the QASM circuit
        in_circuit = Circuit.from_file(file_path)

        # Preoare the instantiator
        batched_instantiation = QFactor_jax(
                                        diff_tol_r=diff_tol_r,
                                        diff_tol_a=diff_tol_a,
                                        min_iters=min_iters,
                                        max_iters=max_iters,
                                        dist_tol=dist_tol,
                                        diff_tol_step_r=diff_tol_step_r,
                                        diff_tol_step = diff_tol_step,
                                        beta=beta)
        instantiate_options={
                'method': batched_instantiation,
                'multistarts': num_multistarts,
                        }


        # Prepare the comiplation passes
        passes =[
                # Convert U3's to VU
                FromU3ToVariablePass(),

                # Split the circuit into partitions
                QuickPartitioner(partition_size),

                # For each partition perform scanning gate removal using QFactor jax
                ForEachBlockPass([
                        ScanningGateRemovalPass(instantiate_options=instantiate_options),  
                        ]),

                # Combine the partitions back into a circuit
                UnfoldPass(),
                
                # Convert back the VariablueUnitaires into U3s
                ToU3Pass()
                ]


        # Create the compilation task
        

        with Compiler(
                num_workers=amount_of_workers,
                runtime_log_level=logging.INFO,
                ) as compiler:

                print(f"Starting gate deletion flow using Qfactor JAX")
                start = timer()
                out_circuit = compiler.compile(in_circuit, passes)
                end = timer()
                run_time = end - start


        return in_circuit, out_circuit, run_time


if __name__ == '__main__':

        in_circuit, out_circuit, run_time = run_gate_del_flow_example()

        print(
                f"Partitioning + Synthesis took {run_time}"
                f"seconds using Qfactor JAX instantiation method."
        )

        print(
                f"Circuit finished with gates: {out_circuit.gate_counts}, "
                f"while started with {in_circuit.gate_counts}"
        )
