# Convolution Benchmarks

This directory contains scripts and utilities for benchmarking PyTorch and MIOpen operations, particularly focusing on convolution operations. The benchmarks are aimed at evaluating performance metrics such as elapsed time, throughput, bandwidth and arithmetic intensity.

## Usage

The provided script can be used to benchmark various convolution operations. Below are the details of the main scripts:

- `conv_flops.py`: Python script for benchmarking convolution operations. Note: File located in cookbook/benchmarks/sizing/
- `collect_common.py`:  Python script for managing both the Pytorch and MIOpen implementation, script handles collection for a range of sizes comprising of input_size, kernel_size, stride, and padding.
- `run.sh`: Bash script for running full scan of sizes with NHWC
 Here's an example command to run the convolution benchmark:

```python

Unified PyTorch and MIOpen benchmarking.

optional arguments:
  -h, --help show this help message and exit
  --input_channels INPUT_CHANNELS
  --output_channels OUTPUT_CHANNELS
  --batch_size BATCH_SIZE
  --input_size INPUT_SIZE
  --kernel_size KERNEL_SIZE
  --stride STRIDE
  --padding PADDING
  --input_channels_list INPUT_CHANNELS_LIST
  --output_channels_list OUTPUT_CHANNELS_LIST
  --batch_size_list BATCH_SIZE_LIST
  --batch_size_range BATCH_SIZE_RANGE
  --input_size_list INPUT_SIZE_LIST
  --input_size_range INPUT_SIZE_RANGE
  --kernel_size_list KERNEL_SIZE_LIST
  --kernel_size_range KERNEL_SIZE_RANGE
  --stride_list STRIDE_LIST
  --stride_range STRIDE_RANGE
  --padding_list PADDING_LIST
  --padding_range PADDING_RANGE
  --num_iterations NUM_ITERATIONS
  --num_warmup_iterations NUM_WARMUP_ITERATIONS
  --cuda_device CUDA_DEVICE
  --output_file OUTPUT_FILE
  --csv_file CSV_FILE
  --verbose Log to stdout besides output_file
  --profile {none,torch,rpd} Profiling mode: "none", "torch", or "rpd"
  --backend BACKEND Comma-separated list of backends for convolution benchmarks. Options: pytorch,miopen
  --append Appends results to CSV file, recording benchmarks across multiple runs.

```

## Examples: Running Pytorch Benchmarks

Runing the convolution benchmark with custom parameters for Pytorch can be done by invoking the "--backend" flag. Since the default backend is Pytorch there is no need to add the "--backend" flag. This command will run conv2d for pytorch and return benchmark results in results/pytorch_conv.csv . 

```python

afo conv --batch_size_list "1 64" --input_size_list "64 128" --kernel_size_list "3 4" --stride_list "1 3" --input_channels "32" --output_channels "64" --padding_list "1 2" --verbose

```

## Examples: Running MIOpen Benchmarks

Runing the convolution benchmark with custom parameters for MIOpen can be done by invoking the "--backend "miopen"" flag. This command will run a range of input args for miopenDriver and return benchmark results in results/miopen_conv.csv .

```python

afo conv --batch_size_list "1 64" --input_size_list "64 128" --kernel_size_list "3 4" --stride_list "1 3" --input_channels "32" --output_channels "64" --padding_list "1 2" --backend "miopen" --verbose

```

## Examples: Running Comparison between Pytorch/MIOpen

To compare convolution benchmarks for Pytorch vs MIOpen, invoke both via "--backend "pytorch,miopen" " flag. This command will run both pytorch/miopen benchmarks for a range of simmilar input args and return a comparison benchmark results in results/combined_benchmark.xlsx

```python

afo conv --batch_size_list "1 64" --input_size_list "64 128" --kernel_size_list "3 4" --stride_list "1 3" --input_channels "32" --output_channels "64" --padding_list "1 2" --backend "miopen,pytorch" --verbose

```

## Examples: Profiling with Pytorch

AFO provides two methods of profiling convolutional benchmarks. One of which is Pytorch profiling. This is available for Pytorch Benchmarking only and is not supported for profiling MIOpen Benchmarks. Profiling can be done on either a single input or multiple inputs. By providing the "--profile "torch"" flag the command will display the top kernel information and provide a trace.json for each specific input arg passed in. 

```python

PROFILE_DIR="results/" afo conv --batch_size_list "1 64" --input_size_list "64 128" --kernel_size_list "3 4" --stride_list "1 3" --input_channels "32" --output_channels "64" --padding_list "1 2"  --verbose --profile "torch"

```


## Examples: Profiling with RPD

AFO also provides a method of profiling convolutional benchmarks with rpd. This is available for both Pytorch and MIOpen Benchmarks. Profiling can be done on either a single input or multiple inputs. By providing the "--profile "rpd"" flag the command will display the top kernel information and provide a trace.rpd for each specific input arg passed in. Both the pytorch and miopen comparison results can be found in results/

```python

# Install RPD
bash ../tools/install_rpd.sh

PROFILE_DIR="results/" afo conv --batch_size_list "1 64" --input_size_list "64 128" --kernel_size_list "3 4" --stride_list "1 3" --input_channels "32" --output_channels "64" --padding_list "1 2" --backend "miopen,pytorch" --verbose --profile "rpd"

```

## Collecting full range of sizes
 
AFO provides a script that is responsible for running a scan of different input size arguments. The command will run both pytorch/miopen for a large range of input sizes with NHWC (not default for ROCm Pytorch currently). Both the pytorch and miopen comparison results including tracefiles can be found in results/

```python

bash afo/convolution/run.sh 

```

## Appending runs to csv 

AFO provides a feature that facilitates generating CSV outputs in a way that prevents overwriting existing results. By utilizing the "--append" flag, users can run multiple AFO convolutional benchmarks consecutively, with each result being added incrementally to the same output file. This feature is also useful for iterative benchmarks that process a range of inputs within a loop, ensuring that each input's result is appended to the existing data as it is processed. This allows for storing results in real-time, rather than waiting until the entire run is complete to generate results..

```python

afo conv --batch_size_list "1 64" --input_size_list "64 128" --kernel_size_list "3 4" --stride_list "1 3" --input_channels "32" --output_channels "64" --padding_list "1 2" --backend "miopen,pytorch" --verbose --append

```

