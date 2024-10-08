#!/bin/bash
# run_70b_throughput_fp8.sh
BASE_DIR=/app
VLLM_DIR=$BASE_DIR/vllm
# MODEL=~/data/llama2/llama-2-70b-chat-hf/
MODEL=~/data/llama3/llama-3-70b-instruct-hf/
#MODEL=/mnt/data/jorge/models/llama-2-70b-chat-hf
MODEL_SIZE=`echo $MODEL | sed 's/.*\(.[0-9][bB]\).*/\1/'`
 
export HIP_FORCE_DEV_KERNARG=1
 
TP="8"
OUTPUT_LEN="128"
INPUT_LEN="2048"
# INPUT_OUTPUT_LEN="1000 2000 4000 8000"
#NUM_PROMPTS="1 4 8 16 32 64 128 256"
NUM_PROMPTS="1 4 8"
 
for tp in $TP;
do
    for np in $NUM_PROMPTS;
    do
        for output_len in $OUTPUT_LEN;
        do
            for input_len in $INPUT_LEN;
            do
            echo python3 $VLLM_DIR/benchmarks/benchmark_throughput.py --model $MODEL  \
            --input-len $input_len --output-len $output_len --tensor-parallel-size $tp --num-prompts $np --dtype float16 --quantization fp8 --worker-use-ray
            python3 $VLLM_DIR/benchmarks/benchmark_throughput.py --model $MODEL  \
            --input-len $input_len --output-len $output_len --tensor-parallel-size $tp --num-prompts $np  --dtype float16 --quantization fp8 --worker-use-ray
            done
        done
    done
done
