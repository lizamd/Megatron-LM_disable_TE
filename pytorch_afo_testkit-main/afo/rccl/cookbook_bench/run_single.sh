#!/bin/bash
NODES="8"
export HIP_FORCE_DEV_KERNARG=1
funcs="all-reduce"
#"all-reduce all-gather broadcast pt2pt all-to-all"
backends="rccl-test pytorch"
dtypes="float32 float16 double int8 bfloat16 uint8 int32 int64 uint64"

for func in $funcs; do
    for dtype in $dtypes; do
    	for backend in $backends; do
            echo "Running: function=${func}, backend=${backend}, dtype=${dtype}, nodes=${NODES}"
            python rccl_benchmark.py --function $func --backend $backend --nproc_per_node $NODES --dtype $dtype --single --maxsize 1024
        done
    done
done

