#!/bin/bash

hostname=$(uname -n)
unique_id=bqskit_${RANDOM}
amount_of_gpus=<Number of GPUS to use in the node>
amount_of_workers_per_gpu=<Number of workers per GPU>
total_amount_of_workers=$(($amount_of_gpus * $amount_of_workers_per_gpu))
scratch_dir=<temp_dir>

wait_for_outgoing_thread_in_manager_log() {
    while [[ ! -f "$manager_log_file" ]]
    do
            sleep 0.5
    done

    while ! grep -q "Started outgoing thread." $manager_log_file; do
            sleep 1
    done
}

wait_for_server_to_connect(){
    while [[ ! -f "$server_log_file" ]]
    do
            sleep 0.5
    done

    while ! grep -q "Connected to manager" $server_log_file; do
            sleep 1
    done
}

mkdir -p $scratch_dir/bqskit_logs

manager_log_file=$scratch_dir/bqskit_logs/manager_${unique_id}.log
server_log_file=$scratch_dir/bqskit_logs/server_${unique_id}.log

echo "Will start bqskit runtime with id $unique_id gpus = $amount_of_gpus and workers per gpu = $amount_of_workers_per_gpu"

# Clean old server and manager logs, if exists
rm -f $manager_log_file
rm -f $server_log_file

echo "Starting MPS server"
nvidia-cuda-mps-control -d

echo "starting BQSKit managers"

bqskit-manager -x -n$total_amount_of_workers -vvv &> $manager_log_file &
manager_pid=$!
wait_for_outgoing_thread_in_manager_log

echo "starting BQSKit server"
bqskit-server $hostname -vvv &>> $server_log_file &
server_pid=$!

wait_for_server_to_connect

echo "Starting $total_amount_of_workers workers on $amount_of_gpus gpus"
for (( gpu_id=0; gpu_id<$amount_of_gpus; gpu_id++ ))
do
    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu_id bqskit-worker $amount_of_workers_per_gpu > $scratch_dir/bqskit_logs/workers_${SLURM_JOB_ID}_${hostname}_${gpu_id}.log &
done

wait

echo "Stop MPS on $hostname"
echo quit | nvidia-cuda-mps-control