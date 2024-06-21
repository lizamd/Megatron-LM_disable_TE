#!/bin/bash

# Prepare the tuning code
# git clone https://github.com/ROCm/pytorch_afo_testkit.git
# cd pytorch_afo_testkit && pip install -e . && cd ..

EXPERIMENT_DIR="experiment"
mkdir -p $EXPERIMENT_DIR

TP=1
PP=1
GPUS_PER_NODE=`python -c "import torch; print(torch.cuda.device_count())"`
DEVICES_IDS=`python -c "print(' '.join([str(a) for a in range($GPUS_PER_NODE)]))"`
DP=$(python -c "print(int($GPUS_PER_NODE/$TP/$PP))")
MODEL_SIZE=7
SEQ_LENGTH=2048
# SEQ_LENGTH=4096

for MBS in 6;
do
    rm -f *.csv

    ROCBLAS_DIR="${EXPERIMENT_DIR}/rocblas_${MODEL_SIZE}B_mbs${MBS}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}"
    ROCBLAS_FILE="${EXPERIMENT_DIR}/rocblas_${MODEL_SIZE}B_mbs${MBS}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}.yaml"
    ROCBLAS_LOG="${EXPERIMENT_DIR}/rocblas_${MODEL_SIZE}B_mbs${MBS}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}.log"

    # =============== search =============== #
    TOTAL_ITERS=6
    VBS=256
    BS=$(python -c "import math; print(int(math.ceil($VBS/($MBS*$DP))*$MBS*$DP))")
    TEE_OUTPUT=1 TORCH_BLAS_PREFER_HIPBLASLT=0 ROCBLAS_LAYER=4 TOTAL_ITERS=$TOTAL_ITERS MODEL_SIZE=$MODEL_SIZE TP=$TP PP=$PP MBS=$MBS BS=$BS SEQ_LENGTH=$SEQ_LENGTH PYTORCH_TUNABLEOP_ENABLED=0 bash train_llama2.sh 2>&1 | grep "\- { rocblas_function:" | uniq > $ROCBLAS_FILE

    python pytorch_afo_testkit/afo/tools/tuning/tune_from_rocblasbench.py $ROCBLAS_FILE --cuda_device $DEVICES_IDS >& $ROCBLAS_LOG 

    mkdir -p $ROCBLAS_DIR
    mv full_tuned*.csv $ROCBLAS_DIR
    # =============== search =============== #

    for VBS in 288;
    do
        rm -f *.csv

        BS=$(python -c "import math; print(int(math.ceil($VBS/($MBS*$DP))*$MBS*$DP))")
        TOTAL_ITERS=9
        TRAIN_LOG="${EXPERIMENT_DIR}/train_compile_gemmtuned_${MODEL_SIZE}B_iter${TOTAL_ITERS}_mbs${MBS}_bs${BS}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}.log"
        TORCH_BLAS_PREFER_HIPBLASLT=0 TOTAL_ITERS=$TOTAL_ITERS MODEL_SIZE=$MODEL_SIZE TP=$TP PP=$PP MBS=$MBS BS=$BS SEQ_LENGTH=$SEQ_LENGTH PYTORCH_TUNABLEOP_FILENAME=$ROCBLAS_DIR/full_tuned%d.csv PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_ENABLED=1 bash train_llama2.sh >& $TRAIN_LOG
        throughput=$(grep -Eo 'throughput per GPU: [^|]*' $TRAIN_LOG | sed -E 's/.*throughput per GPU: ([0-9\.]+).*/\1/'  | head -1)
        time_per_iteration=$(grep -Eo 'elapsed time per iteration: [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration: ([0-9\.]+).*/\1/'  | head -1)
        samples_per_sec=$(python -c "print($BS*1000/$time_per_iteration)")
        mem_usages=$(grep -Eo 'mem usages: [^|]*' $TRAIN_LOG | sed -E 's/.*mem usages: ([0-9\.]+).*/\1/'  | head -1)
        echo "MODEL_SIZE: $MODEL_SIZE MBS: $MBS BS: $BS TP: $TP PP: $PP SEQ_LENGTH: $SEQ_LENGTH samples/s: $samples_per_sec throughput: $throughput mem usages: $mem_usages"
    done
done
