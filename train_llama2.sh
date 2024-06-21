#!/bin/bash

# set -x

TEE_OUTPUT="${TEE_OUTPUT:-0}"
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-0}"

CWD=`pwd`
GPUS_PER_NODE=`python -c "import torch; print(torch.cuda.device_count())"`
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6006
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export CUDA_DEVICE_MAX_CONNECTIONS=1

EXPERIMENT_DIR="experiment"
mkdir -p $EXPERIMENT_DIR

CHECKPOINT_PATH=$EXPERIMENT_DIR/ckpts
rm -rf $CHECKPOINT_PATH
mkdir -p $CHECKPOINT_PATH
DATA_DIR=$EXPERIMENT_DIR/data
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

if ! [ -f "${DATA_DIR}/bookcorpus_text_sentence.idx" ]; then
  echo "Dataset file does not exist, creating..."
  python3 prepare_bookcorpus_megatron_dataset.py --out-dir ${DATA_DIR}
  python3 tools/preprocess_data.py --input ${DATA_DIR}/bookcorpus_megatron.json  --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model ${EXPERIMENT_DIR}/tokenizer.model --output-prefix ${DATA_DIR}/bookcorpus --workers `nproc` --split-sentences
  python3 tools/preprocess_data.py --input ${DATA_DIR}/bookcorpus_megatron.json  --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model ${EXPERIMENT_DIR}/tokenizer.model --output-prefix ${DATA_DIR}/bookcorpus --workers `nproc` --split-sentences
else
  echo "Dataset file already exist."
fi

MODEL_SIZE="${MODEL_SIZE:-7}"
TP="${TP:-1}"
PP="${PP:-1}"
MBS="${MBS:-4}"
BS="${BS:-288}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
TOTAL_ITERS="${TOTAL_ITERS:-6}"

MAX_POSITION_EMBEDDINGS=4096

TRAIN_LOG="${EXPERIMENT_DIR}/train_${MODEL_SIZE}B_iter${TOTAL_ITERS}_mbs${MBS}_bs${BS}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}.log"

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
    --lr 3.0e-4 \
    --train-iters $TOTAL_ITERS \
    --lr-decay-style cosine \
    --min-lr 3.0e-5 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction .01 \
    --no-async-tensor-model-parallel-allreduce \
    --clip-grad 1.0 \
    --bf16
"
    # --no-masked-softmax-fusion \

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --split 949,50,1
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
    --master_port $MASTER_PORT
"

EXTRA_ARGS="
    --group-query-attention \
    --num-query-groups $NUM_GROUPS \
    --no-gradient-accumulation-fusion \
    --use-distributed-optimizer
"

if [ "$NO_TORCH_COMPILE" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --no-torch-compile"
fi

run_cmd="
    torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $EXTRA_ARGS \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH
"

if [ "$TEE_OUTPUT" -eq 0 ]; then 
    run_cmd="$run_cmd >& $TRAIN_LOG"
else
    run_cmd="$run_cmd |& tee $TRAIN_LOG"
fi

eval $run_cmd

echo 'import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog="Process Log")
    parser.add_argument("filename")
    args = parser.parse_args()

    with open(args.filename) as f:
        lines = f.readlines()
    lines = lines[3:-1]
    lines = [float(a) for a in lines]
    mean = np.mean(np.array(lines))
    print(mean)' > mean_log_value.py


# echo '============================================================================================================'
grep -Eo 'throughput per GPU [^|]*' $TRAIN_LOG | sed -E 's/.*throughput per GPU \(TFLOP\/s\/GPU\): ([0-9\.]+).*/\1/' > tmp.txt
echo "throughput per GPU: $(python mean_log_value.py tmp.txt)"
rm tmp.txt

# echo '============================================================================================================'
grep -Eo 'elapsed time per iteration [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9\.]+).*/\1/' > tmp.txt
echo "elapsed time per iteration: $(python mean_log_value.py tmp.txt)"
rm tmp.txt

echo '============================================================================================================'
grep -Eo 'mem usages: [^|]*' $TRAIN_LOG | sed -E 's/.*mem usages: ([0-9\.]+).*/\1/' > tmp.txt
echo "mem usages: $(python mean_log_value.py tmp.txt)"
rm tmp.txt

