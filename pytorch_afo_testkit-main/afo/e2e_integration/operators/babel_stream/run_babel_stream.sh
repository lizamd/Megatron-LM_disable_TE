#!/bin/bash

set -x
export HIP_FORCE_DEV_KERNARG=1

./hip-stream -e --std -s $((256 * 1024 * 1024)) --float --mibibytes |& tee stream.txt

echo "Read `awk '/Read/ {print $2}' stream.txt` MiB/s"
echo "Copy `awk '/Copy/ {print $2}' stream.txt` MiB/s"
echo "Triad `awk '/Triad/ {print $2}' stream.txt` MiB/s"
echo "The average for meta term reading is "

performance=$(awk '/Read/ {sum+=$2} /Copy/ {sum+=$2} /Triad/ {sum+=$2} END {print sum/3}' stream.txt)
echo "$performance MiB/s"
result="Babel Stream 1G,,\nperformance:, $performance,MiB/s"
echo -e $result > babelstream_1G_perf.csv


./hip-stream -e --std --float --mibibytes  |& tee stream_default.txt
echo "Read `awk '/Read/ {print $2}' stream_default.txt` MiB/s"
echo "Copy `awk '/Copy/ {print $2}' stream_default.txt` MiB/s"
echo "Triad `awk '/Triad/ {print $2}' stream_default.txt` MiB/s"
echo "The average for meta term reading is "

performance=$(awk '/Read/ {sum+=$2} /Copy/ {sum+=$2} /Triad/ {sum+=$2} END {print sum/3}' stream_default.txt)
echo "$performance MiB/s"

set +x
