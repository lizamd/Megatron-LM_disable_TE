# How to Run `all_reduce` Performance Test

## Run `all_reduce` with Default Configurations
The `allreduce/run.sh` runs `all_reduce` performance test using different configurations saved in `allreduce_config.txt` on 8 GPUs. The tests run `all_reduce` with `dist.ReduceOp.SUM`, `dist.ReduceOp.PRODUCT`, `dist.ReduceOp.MAX`, `dist.ReduceOp.MIN` operators using `bf16` and `fp16` data types with a tensor of size `(16384,16384)`.

```ruby
cd allreduce
bash run.sh
```
The performance in terms of average latency in seconds will be saved in `allreduce_perf<*>.csv` file.

## Run `allreduce` with Customized Configurations

The script also can run the `allreduce` test with customized configurations on 8 GPUs. Suppose `custom_config.txt` contains your `allreduce` test settings. The test runs as:

```ruby
cd allreduce
bash run.sh custom_config.txt
``` 
The performance will be saved in `allreduce_perf<*>.csv` file.

## Run `all_reduce` for a Specific Configuration

The test can be run using `allreduce/allreduce_test.py`. For example, the command below runs the `allreduce` with `SUM` operator using a tensor of size `(1024,1024)` with `dtype` `torch.bfloat16` on 8 GPUs.

```ruby
cd allreduce
torchrun --nnodes=1 --nproc_per_node=8 allreduce_test.py --ReduceOp SUM --all_reduce_dim 1024 --dtype bf16
```
