#!/bin/bash
set -x

export HIP_FORCE_DEV_KERNARG=1
export PYTORCH_TUNABLEOP_ENABLED=1
#export MIOPEN_ENABLE_LOGGING_CMD=1

#set all_reduce config file name
config_file="allreduce_config.txt"
if [[ -f "$1" ]]; then
	config_file=$1
fi	
allreduce_config_file=$config_file

#save results for each test case to add into final csv file
results="all_reduce test args,,,\nReduceOp,dtype,tensor size, average latency (seconds)"

#for each line in rnn sizes file run the test
while read -r line; do
    args="$line"
    echo "run all_reduce test with $args"
    torchrun --nnodes=1  --nproc_per_node=8 allreduce_test.py $args |& tee allreduce_run.log
    val=$(cat allreduce_run.log | grep "Average latency (seconds): "|awk '{print $18}')
    echo "performance: $val average latency (seconds)"
    run_args=$(cat allreduce_run.log | grep "args: " | head -n 1 |awk '{print $2}')
    results+="\n$run_args,$val"

done < "$allreduce_config_file"

# unset printing trace 
set +x
echo -e "results:\n$results"
test_date=$(date +'%Y%m%d%H%M%S')
echo -e $results > allreduce_perf_$test_date.csv
