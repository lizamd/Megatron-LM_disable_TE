# pytorch_afo_testkit
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This repository contains the PyTorch test items that considered as part of the AFO criteria.
The testkit will include the following test cases:
1. library level tests for key AI operators, including GEMM, CONV, Allreduce and others.
2. PyTorch level tests for key AI operators in same scope.
3. misc ROCm lower level test cases that can serve as sanity test for MI300x platform perf. 

## Getting Started

### Installation
`pip3 install -e .`

### Usage
The main cli for the module is provided through the `afo` command. The package is then subdivided into sub-commands per operations i.e: `conv`, `gemm`, `rccl`

We additionally provide some bash scripts to pet up known configurations.

#### Scripts
For a quick run of our default configurations we provide some bash scripts 

For matrix multiplication kernels:
`bash afo/gemm/run.sh`

For convolution kernels:
`bash afo/convolution/run.sh`

For all reduce operations:
`bash afo/all_reduce/cookbook_bench/run.sh`

#### Runnning benchmark suits
The `afo` command can be used to run sets of pre-seleted configurations for easy testing. Use the `--benchmarks` command for any operation to access this benchmarks. By default the `--benchmarks` command will run all configurations, to limit runtime we advice the use of the `--section` flag to specifiy the desired configuration.

For example `afo gemm --benchmarks --section genericLLM_n lstm --pytorch --cuda_device 0 1 2 3 4 5 6` will run the `genericLLM_n` and the `lstm` preset configurations using pytorch as the backend on GPUs 1-6

Each pre-set configuration is contained in the following YAML files
- [gemm](afo/gemm/benchmarks/benchmarks.yaml)
- [conv](afo/convolution/benchmarks/benchmarks.yaml)

#### Direct call
The `afo` command can also be used to make direct calls for provided configurations
 
For example: `afo gemm --pytorch -m 1024 -n 1024 -k 1024` will run a 1kx1k times a 1kx1k gemm, using pytorch as the backend.

`afo --help` for a list of all sub-commands and `afo SUBCOMAMND --help` for help with any particular sub-command e.g.: `afo gemm --help`

<p align="center">🛸</p>
