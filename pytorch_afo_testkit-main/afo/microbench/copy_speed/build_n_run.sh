set -x
/opt/rocm/bin/hipcc  hipPerfBufferCopySpeed.cpp timer.cpp test_common.cpp -std=c++17 -o hipPerfBufferCopySpeed
./hipPerfBufferCopySpeed warmup |& tee output.txt
python3 collect.py
