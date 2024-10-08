#!/bin/bash
#git clone https://github.com/EleutherAI/cookbook

### ---------------------------------------------------------------------------
###                              BOOK KEEPING VARS
### ---------------------------------------------------------------------------
rm -rf /opt/conda/envs/py_3.9/lib/python3.9/site-packages/numpy-1.20.3.dist-info/

export PROFILE_DIR=$PWD/
OUTPUT_DIR=$PWD/results
mkdir -p $OUTPUT_DIR

### ---------------------------------------------------------------------------
###                           USER CONFIGURABLE VARS
### ---------------------------------------------------------------------------
#export TORCH_PROF="--profile"
export PYTORCH_TUNABLEOP_ENABLED=1 
export PYTORCH_TUNABLEOP_FILENAME=/root/tuned_gemm.csv
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=1
export HIP_FORCE_DEV_KERNARG=1 
export ALL_SIZES=0

### ---------------------------------------------------------------------------
###                                  EXECUTION
### ---------------------------------------------------------------------------

if [[ $ALL_SIZES == 0 ]]; then
    afo gemm --pytorch --m_range 128 200 128 -n 2048 -k 2048 $TORCH_PROF --output_file=$OUTPUT_DIR/mm_m_range2048.txt --csv_file $OUTPUT_DIR/mm_m_range2048.csv

    afo gemm --pytorch --m_range 128 20000 128 -n 4096 -k 4096 $TORCH_PROF --output_file=$OUTPUT_DIR/mm_m_range4096.txt  --csv_file $OUTPUT_DIR/mm_m_range4096.csv
    
    afo gemm --pytorch --m_range 128 10000 128 -n 8192 -k 8192 $TORCH_PROF --output_file=$OUTPUT_DIR/mm_m_range8192.txt  --csv_file $OUTPUT_DIR/mm_m_range8192.csv
else
    set -x
    SCRIPT=$(realpath "$0")
    SCRIPTPATH=$(dirname "$SCRIPT") 
    SIZES_PATH="$SCRIPTPATH/sizes/sizes_ranges.txt"
    while read -r line; do
        args="$line"
        sizefile=`echo $args | sed -e 's/ /_/g' -e 's/--//g' -e 's/-//g' `"_"$(basename ${SIZES_PATH})
        sizefile=`echo $sizefile | sed -e 's/\.txt//g'`

        if [[ $line != \#* && $line != "" ]]; then
            # Ignoring black and comment lines
            echo "run mm_flops test with $args and save the results in $sizefile file"
            afo gemm --pytorch $args --output_file=$OUTPUT_DIR/mm_${sizefile}.txt --csv_file=$TEMP_CSV
            printf $HEADER | cat - $TEMP_CSV > $OUTPUT_DIR/mm_${sizefile}.csv
        fi
    done < "$SIZES_PATH"
fi
