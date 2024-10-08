#!/bin/bash
NODES="2 4 8"
export HIP_FORCE_DEV_KERNARG=1
funcs="all-reduce all-gather broadcast pt2pt all-to-all"
backends="rccl-test pytorch"
dtypes="float32 float16 bfloat16 int8"

for func in $funcs; do
    for dtype in $dtypes; do
        for backend in $backends; do
 	    echo "Running: function=${func}, backend=${backend}, dtype=${dtype}, nodes=${NODES}"
	    python rccl_benchmark.py --function $func --backend $backend --nproc $NODES --dtype $dtype
	done
    done
done

