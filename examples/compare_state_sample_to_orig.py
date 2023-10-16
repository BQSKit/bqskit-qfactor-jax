from __future__ import annotations

import sys
import time
from bqskit.ir.circuit import Circuit
from bqskit.compiler import Compiler, CompilationTask
# from inst_pass import InstPass

from bqskitqfactorjax.qfactor_sample_jax import QFactor_sample_jax
from bqskitqfactorjax.qfactor_jax import QFactor_jax
from bqskit.passes import ToVariablePass

dist_tol_requested = float(sys.argv[2])
num_mutlistarts = 32

num_params_coeff = float(sys.argv[3])


instantiate_options = {
                        'multistarts': num_mutlistarts,
                    }


qfactr_gpu_instantiator = QFactor_jax(

    dist_tol=dist_tol_requested,       # Stopping criteria for distance

    max_iters=100000,      # Maximum number of iterations
    min_iters=1,          # Minimum number of iterations

    # One step plateau detection -
    # diff_tol_a + diff_tol_r ∗ |c(i)| <= |c(i)|-|c(i-1)|
    diff_tol_a=0.0,       # Stopping criteria for distance change
    diff_tol_r=1e-10,     # Relative criteria for distance change

    # Long plateau detection -
    # diff_tol_step_r*|c(i-diff_tol_step)| <= |c(i)|-|c(i-diff_tol_step)|
    diff_tol_step_r=0.1,  # The relative improvement expected
    diff_tol_step=200,  # The interval in which to check the improvement

    # Regularization parameter - [0.0 - 1.0]
    # Increase to overcome local minima at the price of longer compute
    beta=0.0,
)


qfactr_sample_gpu_instantiator = QFactor_sample_jax(

    dist_tol=dist_tol_requested,       # Stopping criteria for distance

    max_iters=100000,      # Maximum number of iterations
    min_iters=10,          # Minimum number of iterations

    # Regularization parameter - [0.0 - 1.0]
    # Increase to overcome local minima at the price of longer compute
    beta=0.0,

    amount_of_validation_states=2,
    num_params_coeff=num_params_coeff, # indicates the ratio between the sum of parameters in the circuits to the sample size.
    overtrain_ratio=1 / 32,
)

# file_name = 'heisenberg64_10q_block_100.qasm'
# file_name = 'heisenberg64_10q_block_036.qasm'
file_name = sys.argv[1]
# orig_10q_block_cir = Circuit.from_file('adder63_10q_block_47.qasm')
# orig_10q_block_cir = Circuit.from_file('adder63_10q_block_63.qasm')

print(f'Will use {file_name} {dist_tol_requested = } {num_mutlistarts = } {num_params_coeff = }')

orig_10q_block_cir = Circuit.from_file(f'examples/{file_name}')

with Compiler(num_workers=1) as compiler:
    task = CompilationTask(orig_10q_block_cir, [ToVariablePass()])
    task_id = compiler.submit(task)
    orig_10q_block_cir_vu = compiler.result(task_id)


tic = time.perf_counter()
target = orig_10q_block_cir_vu.get_unitary()
time_to_simulate_circ = time.perf_counter() - tic
print(f"Time to simulate was {time_to_simulate_circ}")

tic = time.perf_counter()
orig_10q_block_cir_vu.instantiate(target, multistarts=num_mutlistarts, method=qfactr_sample_gpu_instantiator)
sample_inst_time = time.perf_counter() - tic
inst_sample_dist_from_target = orig_10q_block_cir_vu.get_unitary().get_distance_from(target, 1)

print(f'sample method {sample_inst_time = } {inst_sample_dist_from_target = } {num_params_coeff = }')

tic = time.perf_counter()
orig_10q_block_cir_vu.instantiate(target, multistarts=num_mutlistarts, method=qfactr_gpu_instantiator)
full_inst_time = time.perf_counter() - tic
inst_dist_from_target = orig_10q_block_cir_vu.get_unitary().get_distance_from(target, 1)

print(f'full method {full_inst_time = } {inst_dist_from_target = }')

# with Compiler(num_workers=num_mutlistarts+1) as compiler:
#     task = CompilationTask(orig_10q_block_cir, [InstPass(instantiate_options, target)])
#     tic = time.perf_counter()
#     task_id = compiler.submit(task)
#     circ = compiler.result(task_id) # type: ignore
#     ceres_inst_time = time.perf_counter() - tic
#     ceres_inst_dist_from_target = circ.get_unitary().get_distance_from(target, 1)

# print(f'CERES inst {ceres_inst_time = } {ceres_inst_dist_from_target = }')