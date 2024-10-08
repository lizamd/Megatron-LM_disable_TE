#!/bin/bash

set -euo pipefail

# Usage instructions
# this script requires one argument, the full path to the ATI tool for your system
#
# NOTE: this script must be run as sudo (for ATITOOL)

atitool=$1

modes=("read" "write" "copy" "mul" "add" "triad" "dot")
sizes=(268435456 536870912)  # 256 * 1024 * 1024 and 512 * 1024 * 1024
iters=(100 1000 10000)

for m in ${modes[@]}; do
  for s in ${sizes[@]}; do
    for i in ${iters[@]}; do
      "$atitool" -i=0 -pmlogall -pmperiod=50 -pmnoesckey -pmstopcheck -pmoutput="babelstream_${m}_${s}_${i}".csv 2>&1 >/dev/null &
      ./hip-stream -e -s $s "--$m-only" -n $i --device 0 > "babelstream_${m}_${s}_${i}".txt 2>&1 < /dev/null
      touch ./terminate.txt
      # allow cooldown
      sleep 20
    done
  done
done
