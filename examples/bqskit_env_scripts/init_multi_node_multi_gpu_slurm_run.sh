#!/bin/bash
#SBATCH --job-name=<job_name>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t <time_to_run>
#SBATCH -n <number_of_nodes>
#SBATCH --gpus=<total number of GPUs, not nodes>
#SBATCH --output=<full_path_to_log_file>

scratch_dir=<temp_dir>

date
uname -a

### load any modules needed and activate the conda enviorment
module load <module1>
module load <module2>
conda activate <conda-env-name>


echo "starting BQSKit managers on all nodes"
srun run_workers_and_managers.sh <number_of_gpus_per_node> <number_of_workers_per_gpu> &
managers_pid=$!

managers_started_file=$scratch_dir/managers_${SLURM_JOB_ID}_started
n=<number_of_nodes>


# Wait until  all the the  managers have started
while [[ ! -f "$managers_started_file" ]]
do
        sleep 0.5
done

while [ "$(cat "$managers_started_file" | wc -l)" -lt "$n" ]; do
    sleep 1
done

echo "starting BQSKit server on main node"
bqskit-server $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ') &> $scratch_dir/bqskit_logs/server_${SLURM_JOB_ID}.log &
server_pid=$!

uname -a >> $scratch_dir/server_${SLURM_JOB_ID}_started

echo "will run python your command"

python <Your command>

date

echo "Killing the server"
kill -2 $server_pid