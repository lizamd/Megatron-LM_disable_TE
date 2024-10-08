#!/bin/bash
NODES="8"
export HIP_FORCE_DEV_KERNARG=1
func="all-to-all"
#"all-reduce all-gather broadcast pt2pt all-to-all"
backend="pytorch"
#"rccl-test pytorch"
dtype="float16"
#"float32 float16"
#rpd=runTracer.sh

$rpd python rccl_benchmark.py --function $func --backend $backend --nproc_per_node $NODES --dtype $dtype

if [[ -v rpd ]]; then
  if [[ ! -d results/traces/ ]]; then
    mkdir -p results/traces/
  fi
  mv trace.rpd results/traces/trace_${backend}_${func}_${dtype}_tp${NODES}.rpd
  echo "trace.rpd saved to results/traces/"
fi

