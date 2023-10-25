from __future__ import annotations

import time
from bqskit.ir.circuit import Circuit
from bqskit.compiler import Compiler, CompilationTask
# from inst_pass import InstPass

from bqskitqfactorjax.qfactor_sample_jax import QFactorSampleJax
from bqskitqfactorjax.qfactor_jax import QFactor_jax
from bqskit.passes import ToVariablePass
from bqskit import enable_logging

enable_logging(verbose=True)



import argparse

parser = argparse.ArgumentParser(description='Arguments for performance analysis')

parser.add_argument('--input_qasm', type=str, required=True)
parser.add_argument('--multistarts', type=int, default=32)
parser.add_argument('--max_iters', type=int,  default=300)
parser.add_argument('--dist_tol', type=float,  default=1e-8)
parser.add_argument('--num_params_coef', type=int,  default=1)
parser.add_argument('--exact_amount_of_sample_states', type=int)
parser.add_argument('--overtrain_relative_threshold', type=float,  default=0.1)



params = parser.parse_args()
    

print(params)

file_name = params.input_qasm
dist_tol_requested = params.dist_tol
num_mutlistarts = params.multistarts
max_iters = params.max_iters

num_params_coef = params.num_params_coef

exact_amount_of_sample_states = params.exact_amount_of_sample_states
overtrain_relative_threshold = params.overtrain_relative_threshold


instantiate_options = {
                        'multistarts': num_mutlistarts,
                    }


qfactor_gpu_instantiator = QFactor_jax(

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


qfactor_sample_gpu_instantiator = QFactorSampleJax(

    dist_tol=dist_tol_requested,       # Stopping criteria for distance

    max_iters=max_iters,      # Maximum number of iterations
    min_iters=5,          # Minimum number of iterations

    # Regularization parameter - [0.0 - 1.0]
    # Increase to overcome local minima at the price of longer compute
    beta=0.0,

    amount_of_validation_states=2,
    num_params_coef=num_params_coef, # indicates the ratio between the sum of parameters in the circuits to the sample size.
    overtrain_relative_threshold=overtrain_relative_threshold,
    exact_amount_of_states_to_train_on = exact_amount_of_sample_states,
)


print(f'Will use {file_name} {dist_tol_requested = } {num_mutlistarts = } {num_params_coef = }')

orig_10q_block_cir = Circuit.from_file(f'{file_name}')

with Compiler(num_workers=1) as compiler:
    task = CompilationTask(orig_10q_block_cir, [ToVariablePass()])
    task_id = compiler.submit(task)
    orig_10q_block_cir_vu = compiler.result(task_id)


tic = time.perf_counter()
target = orig_10q_block_cir_vu.get_unitary()
time_to_simulate_circ = time.perf_counter() - tic
print(f"Time to simulate was {time_to_simulate_circ}")

tic = time.perf_counter()
orig_10q_block_cir_vu.instantiate(target, multistarts=num_mutlistarts, method=qfactor_sample_gpu_instantiator)
sample_inst_time = time.perf_counter() - tic
inst_sample_dist_from_target = orig_10q_block_cir_vu.get_unitary().get_distance_from(target, 1)

print(f'sample method {sample_inst_time = } {inst_sample_dist_from_target = } {num_params_coef = }')

# tic = time.perf_counter()
# orig_10q_block_cir_vu.instantiate(target, multistarts=num_mutlistarts, method=qfactr_gpu_instantiator)
# full_inst_time = time.perf_counter() - tic
# inst_dist_from_target = orig_10q_block_cir_vu.get_unitary().get_distance_from(target, 1)

# print(f'full method {full_inst_time = } {inst_dist_from_target = }')

# with Compiler(num_workers=num_mutlistarts+1) as compiler:
#     task = CompilationTask(orig_10q_block_cir, [InstPass(instantiate_options, target)])
#     tic = time.perf_counter()
#     task_id = compiler.submit(task)
#     circ = compiler.result(task_id) # type: ignore
#     ceres_inst_time = time.perf_counter() - tic
#     ceres_inst_dist_from_target = circ.get_unitary().get_distance_from(target, 1)

# print(f'CERES inst {ceres_inst_time = } {ceres_inst_dist_from_target = }')
