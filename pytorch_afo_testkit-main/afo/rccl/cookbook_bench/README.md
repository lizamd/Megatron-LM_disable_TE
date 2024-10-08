# AFO RCCL Benchmark Script
This script is designed to facilitate the benchmarking of RCCL operations. It supports various benchmark backends, different RCCL functions, and allows for varying the number of processes per node (`nproc_per_node`) and tensor data types (`dtype`). The script parses the output from these benchmark runs and saves the results 
to a CSV file.

## Installation

```bash
./install.sh
```

## run_all

```bash
./run_all.sh
```

## RCCL tracer for ML model

```bash
# run your model with RCCL tracing flag enabled
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL

./run_your_ml_model.sh > model_rccl_log.txt
```
1. **Pass the RCCL log to `rccl_nccl_parser`:**  
   The parser processes the RCCL log and extracts the associated kernel information and store it into a yaml file.

2. **Benchmark Option:**  
   - The `--benchmark` flag determines whether the user wants to benchmark the collected RCCL kernels."
   - If benchmarking isn't required immediately, it can be deferred by later using the generated YAML file as input.


```bash
python rccl_nccl_parser.py --input model_rccl_log.txt --output [OPTIONAL] --benchmark
```

## Usage
Below are the details of the scripts:
- `rccl_benchmark.py`: This Python script benchmarks RCCL operations, accommodating various benchmark backends, different RCCL functions, a range of processes per node (nproc_per_node), and tensor data types (dtype).
- `run_all.sh`: This main script measures throughput and Bus Bandwidth (Gbps) using various combinations in the configuration.

To run the benchmarks, execute the run_allreduce_bench.py script with appropriate arguments. Here's an example command to run the benchmark:
```python
python rccl_benchmark.py --function $func --backend $backend --nproc_per_node $NODES --dtype $dtype
```
```
--function: Different RCCL functions such as all-reduce, all-gather, etc
--backend: User could choose from rccl-test backend or pytorch backend
--dtype: Data type for the tensors. Options include float16 and float32. Default is float16.
--nproc_per_node: Specifies a list of process counts per node for the benchmarks. Accepts multiple 
values. Example: --nproc_per_node 2 4 6 8
--results_dir: Target directory to store results.  Default is results.
```


## Output 
The script will output the results to the terminal and save them to a CSV file. The CSV file will contains metrics for each benchmark configuration:

```python
NP Size: The number of processes used.
Size: Size of the tensor involved in the all_reduce operation.
Description: Additional details about the tensor, typically its dimensions.
Duration(us): Time taken for the all_reduce operation, in microseconds.
Throughput (Gbps): The achieved throughput in gigabits per second.
BusBW (Gbps): Bandwidth utilized by the operation, in gigabits per second.
Data Type: The tensor data type used in the benchmark.
Backend: Indicate which benchmark backend you use. (pytorch cookbook or rccl-test)
```
