#!/bin/bash
#SBATCH --job-name=checking_run_time_var
#SBATCH -A m4141_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 8:50:00
#SBATCH -n 1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --output=./logs/comparing_full_and_sample-shor_10q-%j.txt




date
uname -a
module load python
module load nvidia
conda activate my_env
nvidia-cuda-mps-control -d


XLA_PYTHON_CLIENT_PREALLOCATE=False python ./examples/compare_state_sample_to_orig.py examples/shor26_10q_block_134.qasm 1e-9 0.05