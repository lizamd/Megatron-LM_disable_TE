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
SEQ_PARALLEL=1
NO_TORCH_COMPILE=0
#vbs=256 default
vbs=8
mbs=2
# SEQ_LENGTH=4096
 
for MBS in $mbs;
do
    rm -f *.csv
 
    ROCBLAS_DIR="${EXPERIMENT_DIR}/rocblas_${MODEL_SIZE}B_mbs${MBS}_vbs${vbs}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}_nocompile${NO_TORCH_COMPILE}"
    ROCBLAS_FILE="${EXPERIMENT_DIR}/rocblas_${MODEL_SIZE}B_mbs${MBS}_vbs${vbs}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}_nocompile${NO_TORCH_COMPILE}.yaml"
    ROCBLAS_LOG="${EXPERIMENT_DIR}/rocblas_${MODEL_SIZE}B_mbs${MBS}_vbs${vbs}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}_nocompile${NO_TORCH_COMPILE}.log"
 
    # =============== search =============== #
    TOTAL_ITERS=2
    VBS=$vbs
    BS=$(python -c "import math; print(int(math.ceil($VBS/($MBS*$DP))*$MBS*$DP))")
 
    echo "Start tuning"
   # TEE_OUTPUT=1 TORCH_BLAS_PREFER_HIPBLASLT=0 ROCBLAS_LAYER=4 TOTAL_ITERS=$TOTAL_ITERS MODEL_SIZE=$MODEL_SIZE TP=$TP PP=$PP MBS=$MBS BS=$BS SEQ_LENGTH=$SEQ_LENGTH SEQ_PARALLEL=$SEQ_PARALLEL PYTORCH_TUNABLEOP_ENABLED=0 NO_TORCH_COMPILE=$NO_TORCH_COMPILE bash train_llama2_13b.sh 2>&1 | grep "\- { rocblas_function:" | uniq > $ROCBLAS_FILE
    echo "Tuning stopped"
 
    #python pytorch_afo_testkit-main/afo/tools/tuning/tune_from_rocblasbench.py $ROCBLAS_FILE --cuda_device $DEVICES_IDS >& $ROCBLAS_LOG
 
    #mkdir -p $ROCBLAS_DIR
    #mv full_tuned*.csv $ROCBLAS_DIR
    # =============== search =============== #
 
    for VBS in $vbs;
    do
        rm -f *.csv
 
        BS=$(python -c "import math; print(int(math.ceil($VBS/($MBS*$DP))*$MBS*$DP))")
        TOTAL_ITERS=4
        TRAIN_LOG="${EXPERIMENT_DIR}/train_compile_gemmtuned_${MODEL_SIZE}B_iter${TOTAL_ITERS}_mbs${MBS}_bs${BS}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}_nocompile${NO_TORCH_COMPILE}.log"
        TORCH_BLAS_PREFER_HIPBLASLT=0 TOTAL_ITERS=$TOTAL_ITERS MODEL_SIZE=$MODEL_SIZE TP=$TP PP=$PP MBS=$MBS BS=$BS SEQ_LENGTH=$SEQ_LENGTH SEQ_PARALLEL=$SEQ_PARALLEL PYTORCH_TUNABLEOP_FILENAME=$ROCBLAS_DIR/full_tuned%d.csv PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_ENABLED=1 NO_TORCH_COMPILE=$NO_TORCH_COMPILE bash train_llama2_13b.sh >& $TRAIN_LOG
        throughput=$(grep -Eo 'throughput per GPU: [^|]*' $TRAIN_LOG | sed -E 's/.*throughput per GPU: ([0-9\.]+).*/\1/'  | head -1)
        echo $throughput
        time_per_iteration=$(grep -Eo 'elapsed time per iteration: [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration: ([0-9\.]+).*/\1/'  | head -1)
        echo $time_per_iteration
        samples_per_sec=$(python -c "print($BS*1000/$time_per_iteration)")
        token_throughput=$(python -c "print($samples_per_sec*$SEQ_LENGTH/8)")
        mem_usages=$(grep -Eo 'mem usages: [^|]*' $TRAIN_LOG | sed -E 's/.*mem usages: ([0-9\.]+).*/\1/'  | head -1)
        echo "MODEL_SIZE: $MODEL_SIZE MBS: $MBS BS: $BS TP: $TP PP: $PP SEQ_LENGTH: $SEQ_LENGTH samples/s: $samples_per_sec throughput: $throughput token throughput: $token_throughput mem usages: $mem_usages" |& tee -a $TRAIN_LOG
    done
done