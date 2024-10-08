#!/bin/bash

# Prepare the tuning code
# git clone https://github.com/ROCm/pytorch_afo_testkit.git
# cd pytorch_afo_testkit && pip install -e . && cd ..

EXPERIMENT_DIR="experiment"
mkdir -p $EXPERIMENT_DIR

TP=8
PP=1
GPUS_PER_NODE=`python -c "import torch; print(torch.cuda.device_count())"`
DEVICES_IDS=`python -c "print(' '.join([str(a) for a in range($GPUS_PER_NODE)]))"`
DP=$(python -c "print(int($GPUS_PER_NODE/$TP/$PP))")
MODEL_SIZE=70
SEQ_LENGTH=4096
# SEQ_LENGTH=4096

for MBS in 4;
do
    rm -f *.csv

    ROCBLAS_DIR="${EXPERIMENT_DIR}/rocblas_${MODEL_SIZE}B_mbs${MBS}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}"
    ROCBLAS_FILE="${EXPERIMENT_DIR}/rocblas_${MODEL_SIZE}B_mbs${MBS}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}.yaml"
    ROCBLAS_LOG="${EXPERIMENT_DIR}/rocblas_${MODEL_SIZE}B_mbs${MBS}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}.log"

    # =============== search =============== #
    TOTAL_ITERS=2
    VBS=128
    BS=$(python -c "import math; print(int(math.ceil($VBS/($MBS*$DP))*$MBS*$DP))")
    echo "Getting GEMM info..."
    #TEE_OUTPUT=1 TORCH_BLAS_PREFER_HIPBLASLT=0 ROCBLAS_LAYER=4 TOTAL_ITERS=$TOTAL_ITERS MODEL_SIZE=$MODEL_SIZE TP=$TP PP=$PP MBS=$MBS BS=$BS SEQ_LENGTH=$SEQ_LENGTH PYTORCH_TUNABLEOP_ENABLED=0 bash train_70b_n0.sh 2>&1 | grep "\- { rocblas_function:" | uniq > $ROCBLAS_FILE

    echo "Run GEMM tunning..."
    python pytorch_afo_testkit/afo/tools/tuning/tune_from_rocblasbench.py $ROCBLAS_FILE --cuda_device $DEVICES_IDS >& $ROCBLAS_LOG 

    mkdir -p $ROCBLAS_DIR
    mv full_tuned*.csv $ROCBLAS_DIR
    # =============== search =============== #

    echo "Tunning completed..."


done
