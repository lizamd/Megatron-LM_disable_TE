import torch
import sys
import os

from communication.utils import (
    init_processes,
    print_rank_0,
    print_header,
    get_bw,
    get_metric_strings,
    sync_all,
    max_numel,
    convert_size,
    benchmark_parser,
)
from communication.constants import *  # noqa: F403

COMMS_BENCH_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(COMMS_BENCH_DIR)


def timed_broadcast(input, start_event, end_event, args):
    if args.dist == "torch":
        import torch.distributed as dist
    elif args.dist == "deepspeed":
        import deepspeed.comm as dist

    sync_all()
    # Warmups, establish connections, etc.
    for i in range(args.warmups):
        dist.broadcast(input, 0, async_op=args.async_op)
    sync_all()

    # time the actual comm op trials times and average it
    start_event.record()
    for i in range(args.trials):
        dist.broadcast(input, 0, async_op=args.async_op)
    end_event.record()
    sync_all()
    duration = start_event.elapsed_time(end_event) / 1000

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.element_size() * input.nelement()
    n = dist.get_world_size()
    tput, busbw = get_bw("broadcast", size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(
        args, tput, busbw, avg_duration
    )
    desc = f"{input.nelement()}x{input.element_size()}"

    if not args.raw:
        size = convert_size(size)

    print_rank_0(
        f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}"
    )


def run_broadcast(local_rank, args):
    if args.dist == "torch":
        import torch.distributed as dist
    elif args.dist == "deepspeed":
        import deepspeed.comm as dist

    # Prepare benchmark header
    print_header(args, "broadcast")

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if args.single:
        sync_all()
        global_rank = dist.get_rank()
        try:
            torch_dtype = getattr(torch, args.dtype)
            temp_tensor = torch.empty((), dtype=torch_dtype)
            dtype_size = temp_tensor.element_size()
            mat = torch.ones(
                world_size,
                (args.maxsize // world_size) // dtype_size,
                dtype=torch_dtype,
            ).cuda(local_rank)
            sync_all()
            input = (mat.mul_(global_rank)).view(-1)
            del mat
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                if dist.get_rank() == 0:
                    print("WARNING: Ran out of GPU memory. Exiting comm op.")
                sync_all()
                return
            else:
                raise e
        sync_all()
        timed_broadcast(input, start_event, end_event, args)
    elif args.scan:
        M_LIST = []
        for x in (2**p for p in range(1, args.maxsize)):
            M_LIST.append(x)

        sync_all()
        # loop over various tensor sizes
        for M in M_LIST:
            global_rank = dist.get_rank()
            try:
                mat = torch.ones(world_size, M, dtype=getattr(torch, args.dtype)).cuda(
                    local_rank
                )
                sync_all()
                input = (mat.mul_(global_rank)).view(-1)
                del mat
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if dist.get_rank() == 0:
                        print("WARNING: Ran out of GPU memory. Exiting comm op.")
                    sync_all()
                    break
                else:
                    raise e
            sync_all()
            timed_broadcast(input, start_event, end_event, args)
    else:
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        # Don't need output tensor, so we double mem_factor
        elements_per_gpu = max_numel(
            comm_op="broadcast",
            dtype=getattr(torch, args.dtype),
            mem_factor=args.mem_factor * 2,
            local_rank=local_rank,
            args=args,
        )
        try:
            mat = torch.ones(elements_per_gpu, dtype=getattr(torch, args.dtype)).cuda(
                local_rank
            )
            input = (mat.mul_(global_rank)).view(-1)
        except RuntimeError as e:
            if "out of memory" in str(e):
                if dist.get_rank() == 0:
                    print(
                        "WARNING: Ran out of GPU memory. Try to reduce the --mem-factor argument!"
                    )
                sync_all()
                return
        sync_all()
        timed_broadcast(input, start_event, end_event, args)


if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    rank = args.local_rank
    init_processes(local_rank=rank, args=args)
    run_broadcast(local_rank=rank, args=args)
