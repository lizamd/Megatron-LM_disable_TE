#!/bin/bash

# set -x

export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
#export NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re6,bnxt_re7,bnxt_re8,bnxt_re9
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export NCCL_SOCKET_IFNAME=ens14np0
export NCCL_PROTO=Simple

# parsing input arguments
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done
export CUDA_DEVICE_MAX_CONNECTIONS=1

export GLOO_SOCKET_IFNAME=ens14np0

TIME_STAMP=$(date +"%Y-%m-%d_%H-%M-%S")

TEE_OUTPUT="${TEE_OUTPUT:-1}"
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-0}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
NO_TRAINING="${NO_TRAINING:-0}" # NO_TRAINING=1: for computing metrics only
ENABLE_PROFILING="${ENABLE_PROFILING:-0}"
echo "NO_TRAINING=$NO_TRAINING"

CWD=`pwd`
GPUS_PER_NODE=`python3 -c "import torch; print(torch.cuda.device_count())"`
# Change for multinode config
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MODEL_SIZE="${MODEL_SIZE:-70}"
TP="${TP:-8}"
PP="${PP:-1}"
MBS="${MBS:-4}"
BS="${BS:-16}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
TOTAL_ITERS="${TOTAL_ITERS:-4}"
SEQ_PARALLEL="${SEQ_PARALLEL:-1}" 
CONTI_PARAMS="${CONTI_PARAMS:-0}"
OPTIMIZER="${OPTIMIZER:-sgd}"
TE_FP16="${TE_FP16:-0}"



EXPERIMENT_DIR="experiment"
mkdir -p $EXPERIMENT_DIR

CHECKPOINT_PATH=$EXPERIMENT_DIR/ckpts
rm -rf $CHECKPOINT_PATH
mkdir -p $CHECKPOINT_PATH
DATA_DIR="/root/.cache/data"
mkdir -p $DATA_DIR

TOKENIZER_MODEL=$EXPERIMENT_DIR/tokenizer.model

# Download the tokenizer model
if ! [ -f "$TOKENIZER_MODEL" ]; then
wget -O $TOKENIZER_MODEL https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/resolve/main/tokenizer.model
fi

# Prepare the dataset
echo 'import argparse
from pathlib import Path
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=False, default="tmp/data",
                       help="Path to output JSON")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    dataset = load_dataset("bookcorpus", split="train")
    dataset.to_json(out_dir / "bookcorpus_megatron.json")' > prepare_bookcorpus_megatron_dataset.py

DATA_PATH=${DATA_DIR}/bookcorpus_text_sentence

# if ! [ -f "${DATA_DIR}/bookcorpus_text_sentence.idx" ]; then
#   echo "Dataset file does not exist, creating..."
#   python3 prepare_bookcorpus_megatron_dataset.py --out-dir ${DATA_DIR}
#   python3 tools/preprocess_data.py --input ${DATA_DIR}/bookcorpus_megatron.json  --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model ${EXPERIMENT_DIR}/tokenizer.model --output-prefix ${DATA_DIR}/bookcorpus --workers `nproc` --split-sentences
#   python3 tools/preprocess_data.py --input ${DATA_DIR}/bookcorpus_megatron.json  --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model ${EXPERIMENT_DIR}/tokenizer.model --output-prefix ${DATA_DIR}/bookcorpus --workers `nproc` --split-sentences
# else
#   echo "Dataset file already exist."
# fi


MAX_POSITION_EMBEDDINGS=32768

DEFAULT_LOG_DIR="${EXPERIMENT_DIR}/${NNODES}nodes_rank${NODE_RANK}_train_${MODEL_SIZE}B_mbs${MBS}_bs${BS}_tp${TP}_pp${PP}_optim_${OPTIMIZER}_iter${TOTAL_ITERS}/nocompile${NO_TORCH_COMPILE}_TE_FP16_${TE_FP16}/${TIME_STAMP}"
LOG_DIR="${LOG_DIR:-${DEFAULT_LOG_DIR}}"
TRAIN_LOG="${LOG_DIR}/output.log"
mkdir -p $LOG_DIR
echo $TRAIN_LOG

if [[ $MODEL_SIZE -eq 7 ]]; then
        HIDDEN_SIZE=4096 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=11008 # e.g. llama-13b: 13824
        NUM_LAYERS=32 # e.g. llama-13b: 40
        NUM_HEADS=32 # e.g. llama-13b: 40
        SEQ_LENGTH=$SEQ_LENGTH
        MAX_POSITION_EMBEDDINGS=$MAX_POSITION_EMBEDDINGS
        NUM_KV_HEADS=32 # llama2 70B uses GQA
elif [[ $MODEL_SIZE -eq 13 ]]; then
        HIDDEN_SIZE=5120 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=13824 # e.g. llama-13b: 13824
        NUM_LAYERS=40 # e.g. llama-13b: 40
        NUM_HEADS=40 # e.g. llama-13b: 40
        SEQ_LENGTH=$SEQ_LENGTH
        MAX_POSITION_EMBEDDINGS=$MAX_POSITION_EMBEDDINGS
        NUM_KV_HEADS=40 # llama2 70B uses GQA
elif [[ $MODEL_SIZE -eq 70 ]]; then
        HIDDEN_SIZE=8192 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=28672 # e.g. llama-13b: 13824
        NUM_LAYERS=80 # e.g. llama-13b: 40
        NUM_HEADS=64 # e.g. llama-13b: 40
        NUM_KV_HEADS=8 # llama2 70B uses GQA
        SEQ_LENGTH=$SEQ_LENGTH
        MAX_POSITION_EMBEDDINGS=$MAX_POSITION_EMBEDDINGS
else
        echo "Model size not supported."
        exit 1
fi

GROUP_SIZE=$(( ${NUM_HEADS} / ${NUM_KV_HEADS} ))
NUM_GROUPS=$(( ${NUM_HEADS} / ${GROUP_SIZE} ))


PROFILING_DIR="${LOG_DIR}/trace_using_tuned_gemm"


GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-position-embedding \
    --disable-bias-linear \
    --swiglu \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --micro-batch-size $MBS \
    --global-batch-size $BS \
    --lr 3.0e-8 \
    --train-iters $TOTAL_ITERS \
    --lr-decay-style cosine \
    --min-lr 3.0e-9 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction .01 \
    --optimizer $OPTIMIZER \
    --no-async-tensor-model-parallel-allreduce \
    --clip-grad 1.0 \
    --bf16 \
    --no-masked-softmax-fusion
"
    # --no-masked-softmax-fusion \

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --split 949,50,1 \
    --mock-data
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --log-throughput \
    --no-save-optim \
    --eval-iters -1
"

    # --save-interval $TOTAL_ITERS \
    # --eval-interval $TOTAL_ITERS \

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

EXTRA_ARGS="
    --group-query-attention \
    --num-query-groups $NUM_GROUPS \
    --no-gradient-accumulation-fusion \
    --enable_profiling $ENABLE_PROFILING \
    --profiling_out_folder $PROFILING_DIR \
    --distributed-backend nccl \
    --distributed-timeout-minutes 30 \
    --use-distributed-optimizer \
    --overlap-param-gather \
    --overlap-grad-reduce \
    --no-check-for-nan-in-loss-and-grad
"

#     --overlap-param-gather \
#     --overlap-grad-reduce
#     --use-distributed-optimizer

if [ "$NO_TORCH_COMPILE" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --no-torch-compile"
fi

if [ "$USE_FLASH_ATTN" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --use-flash-attn"
fi

if [ "$SEQ_PARALLEL" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --sequence-parallel"
fi

if [ "$CONTI_PARAMS" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --use-contiguous-parameters-in-local-ddp"
fi

if [ "$TE_FP16" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --transformer-impl=transformer_engine \
    --fp8-margin=0 \
    --fp8-interval=1 \
    --fp8-amax-history-len=1024 \
    --fp8-amax-compute-algo=max
"
fi

run_cmd="
    torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $EXTRA_ARGS \
        --load $CHECKPOINT_PATH
"

if [ "$TEE_OUTPUT" -eq 0 ]; then 
    run_cmd="$run_cmd >& $TRAIN_LOG"
else
    run_cmd="$run_cmd |& tee $TRAIN_LOG"
fi

if [ "$NO_TRAINING" -eq 0 ]; then 
    eval $run_cmd
fi

echo 'import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog="Process Log")
    parser.add_argument("filename")
    args = parser.parse_args()

    with open(args.filename) as f:
        lines = f.readlines()
    lines = lines[2:-1]
    lines = [float(a) for a in lines]
    mean = np.mean(np.array(lines))
    print(mean)' > mean_log_value.py


# echo '============================================================================================================'
grep -Eo 'throughput per GPU [^|]*' $TRAIN_LOG | sed -E 's/.*throughput per GPU \(TFLOP\/s\/GPU\): ([0-9\.]+).*/\1/' > tmp.txt
PERFORMANCE=$(python3 mean_log_value.py tmp.txt)
echo "throughput per GPU: $PERFORMANCE" |& tee -a $TRAIN_LOG
rm tmp.txt

# echo '============================================================================================================'
grep -Eo 'elapsed time per iteration [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9\.]+).*/\1/' > tmp.txt
ETPI=$(python3 mean_log_value.py tmp.txt)
echo "elapsed time per iteration: $ETPI" |& tee -a $TRAIN_LOG

TIME_PER_ITER=$(python3 mean_log_value.py tmp.txt 2>/dev/null | awk '{printf "%.6f", $0}')
TGS=$(awk -v bs="$BS" -v sl="$SEQ_LENGTH" -v tpi="$TIME_PER_ITER" -v ws="$WORLD_SIZE" 'BEGIN {printf "%.6f", bs * sl * 1000/ (tpi * ws)}')
echo "tokens/GPU/s: $TGS" |& tee -a $TRAIN_LOG
rm tmp.txt

echo '============================================================================================================'
grep -Eo 'mem usages: [^|]*' $TRAIN_LOG | sed -E 's/.*mem usages: ([0-9\.]+).*/\1/' > tmp.txt
MEMUSAGE=$(python3 mean_log_value.py tmp.txt)
echo "mem usages: $MEMUSAGE" |& tee -a $TRAIN_LOG
rm tmp.txt



NUM_GROUPS=$(( ${NNODES} - 1 ))
if [[ $NODE_RANK -eq $NUM_GROUPS ]]; then
    '#Nodes	Model 	Seq Len	Micro batch	Global Batch	TP	PP	Tokens/Sec/GPU	TFLOPs/s/GPU	Memory Usage'
    echo "$NNODES	$MODEL_SIZE	$SEQ_LENGTH	$MBS	$BS	$TP	$PP	$TGS	$PERFORMANCE	$MEMUSAGE	$ETPI" |& tee -a ../out.csv
    echo "$NNODES	$MODEL_SIZE	$SEQ_LENGTH	$MBS	$BS	$TP	$PP	$TGS	$PERFORMANCE	$MEMUSAGE	$ETPI" |& tee -a out.csv
else
        echo "Not the final node; check another the output for another node!"
        exit 1
fi

