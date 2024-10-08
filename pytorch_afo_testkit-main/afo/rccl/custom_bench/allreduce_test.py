import os
import torch
import torch.distributed as dist
import argparse
from time import perf_counter
import numpy as np


def get_global_rank() -> int:
    return int(os.environ.get("RANK") or dist.get_rank())


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK") or 0)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def init_process_group():
    # Initialize process group and set device.
    dist.init_process_group(backend="nccl")
    barrier()
    torch.cuda.set_device(f"cuda:{get_local_rank()}")


def measure_latency(
    rank,
    group,
    reduceop=dist.ReduceOp.SUM,
    all_reduce_dim: int = 2**14,
    dtype=torch.float16,
):
    """simple collective communication."""
    # warm up
    print(f"Rank {rank} started warm up repeat 2 times")
    for _ in range(2):
        dist.barrier(group)
        tensor = torch.randn((all_reduce_dim, all_reduce_dim), dtype=dtype).to(
            "cuda:" + str(rank)
        )
        # print(tensor)
        dist.all_reduce(tensor, op=reduceop, group=group)
        # print(f'Rank {rank} has data {tensor[0]}')
    print(f"Rank {rank} done  warm up.")
    latencies = []
    # Timed run
    print(f"Rank {rank} started all_reduce test, repeat 10 time")
    for _ in range(10):
        # dist.barrier(group)
        tensor = torch.randn((all_reduce_dim, all_reduce_dim), dtype=dtype).to(
            "cuda:" + str(rank)
        )
        start_time = perf_counter()
        dist.all_reduce(tensor, op=reduceop, group=group)
        # print(f'Rank {rank} has data {tensor[0]}')
        latency = perf_counter() - start_time
        latencies.append(latency)
    print(f"Rank {rank} done all_reduce test.")
    # Compute run statistics
    time_avg_s = np.mean(latencies)
    time_std_s = np.std(latencies)
    time_p95_s = np.percentile(latencies, 95)

    return time_avg_s, time_p95_s, time_std_s


def test(args):
    rank = get_global_rank()
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    group = dist.new_group(list(range(0, local_world_size)))

    results = torch.zeros((3,), device="cuda:" + str(rank))
    # time_avg_s, time_p95_s, time_std_s
    results[0], results[1], results[2] = measure_latency(
        rank,
        group,
        reduceop=args.ReduceOp,
        all_reduce_dim=args.all_reduce_dim,
        dtype=args.dtype,
    )
    # print(f'Rank: {rank} done all_reduce test P95 latency (seconds): {results[1]:.9f}; Average latency (seconds): {results[0]:.9f} +\- {results[2]:.9f};')

    # reduce all to rank 0 process and display the results
    torch.distributed.reduce(results, 0)
    if dist.get_rank() == 0:
        print(
            f"Rank {rank} dispalies average for all ranks for op {args.ReduceOp}  P95 latency (seconds): {results[1]/local_world_size:.9f}; Average latency (seconds): {results[0]/local_world_size:.9f} +\- {results[2]/local_world_size:.9f};"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--all_reduce_dim", type=int, default=2**14)
    parser.add_argument("-o", "--ReduceOp", type=str, default="SUM")
    parser.add_argument("-t", "--dtype", type=str, default="fp16")
    args = parser.parse_args()
    return args


def main():
    # get test parameters
    args = parse_args()

    # set dtype; fp16 is default
    if args.dtype == "fp32":
        args.dtype = torch.float32
    elif args.dtype == "bf16":
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float16

    if args.ReduceOp == "MAX":
        args.ReduceOp = dist.ReduceOp.MAX
    elif args.ReduceOp == "MIN":
        args.ReduceOp = dist.ReduceOp.MIN
    elif args.ReduceOp == "PRODUCT":
        args.ReduceOp = dist.ReduceOp.PRODUCT
    else:  # default
        args.ReduceOp = dist.ReduceOp.SUM

    # initial a process on each device
    print(
        f'args: {args.ReduceOp},{args.dtype},"[{args.all_reduce_dim},{args.all_reduce_dim}]"'
    )
    init_process_group()
    test(args)


if __name__ == "__main__":
    main()
