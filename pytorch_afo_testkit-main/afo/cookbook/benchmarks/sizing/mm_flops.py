import sys
import numpy as np
import argparse
import os
import pandas as pd
from multiprocessing import Pool, set_start_method, Manager
import tqdm

from afo.cookbook.benchmarks.sizing.utils import Tee, benchmark_mm
from afo.tools.utils import df_to_csv

NUM_PROC_PER_GPU = 1
GEMM_CONFIG_KEYS = "M,N,K,B,transA,transB,dtype"


def add_mm_args(parser, outfile_dir: str = ".", default_output_to_file: bool = False):
    m_group = parser.add_mutually_exclusive_group(required=True)
    m_group.add_argument(
        "-m",
        nargs="+",
        type=int,
        help="The first dimension of the GEMM, enter any number of arguments",
    )
    m_group.add_argument(
        "--m_range",
        nargs="+",
        type=int,
        help="The first dimension of the GEMM, [start,stop,step]",
    )

    n_group = parser.add_mutually_exclusive_group(required=True)
    n_group.add_argument(
        "-n",
        nargs="*",
        type=int,
        help="The shared dimension of the GEMM, enter any number of arguments",
    )
    n_group.add_argument(
        "--n_range",
        nargs="+",
        type=int,
        help="The shared dimension of the GEMM, [start,stop,step]",
    )

    k_group = parser.add_mutually_exclusive_group(required=True)
    k_group.add_argument(
        "-k",
        nargs="*",
        type=int,
        help="The last dimension of the GEMM, enter any number of arguments",
    )
    k_group.add_argument(
        "--k_range",
        nargs="+",
        type=int,
        help="The last dimension of the GEMM, [start,stop,step]",
    )

    b_group = parser.add_mutually_exclusive_group(required=False)
    b_group.add_argument(
        "-b",
        nargs="*",
        default=[1],
        type=int,
        help="The batch dimension of the GEMM, enter any number of arguments",
    )
    k_group.add_argument(
        "--b_range",
        nargs="+",
        type=int,
        help="The batch dimension of the GEMM, [start,stop,step]",
    )

    parser.add_argument(
        "--profile",
        default=False,
        action="store_true",
        # action=argparse.BooleanOptionalAction,
        help="Profile a given run",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=200,
        help="The number of iterations used to benchmark each GEMM",
    )
    parser.add_argument(
        "--num_warmup_iterations",
        type=int,
        default=50,
        help="The number of warmup iterations",
    )
    parser.add_argument(
        "--cuda_device",
        nargs="*",
        type=int,
        default=[0],
        help="The cuda device(s) to run the benchmark on",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None if default_output_to_file else f"{outfile_dir}/results/mm.out",
    )
    parser.add_argument("--csv_file", type=str, default=f"{outfile_dir}/results/mm.csv")

    parser.add_argument(
        "--transpose",
        type=str,
        default="TN",
        choices=["NN", "TN", "NT", "TT"],
        help="The Memory orientation of Matrix A and B",
    )

    parser.add_argument(
        "--verbose",
        default=True,
        # action=argparse.BooleanOptionalAction,
        action="store_true",
        help="log to stdout besides output_file?",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="gemm data type for operation",
    )


# Need a wrapper to unpack args since tqdm doesn't work with starmap
def wrapper(kwargs):
    return distributed_benchmark_mm(**kwargs)


def distributed_benchmark_mm(queue=None, **kwargs):
    if queue is None:
        # Single GPU Case
        elapsed, cpu_elapsed, overhead, overhead_p, throughput, bw, ai = benchmark_mm(
            **kwargs
        )
    else:
        gpu_id = queue.get()
        try:
            elapsed, cpu_elapsed, overhead, overhead_p, throughput, bw, ai = (
                benchmark_mm(**kwargs, device=gpu_id)
            )
        finally:
            queue.put(gpu_id)
    entry = {
        "M": [kwargs["m"]],
        "N": [kwargs["n"]],
        "K": [kwargs["k"]],
        "B": [kwargs["b"]],
        "dtype": [kwargs["dtype"]],
        "transA": [kwargs["transpose"][0]],
        "transB": [kwargs["transpose"][1]],
        "elapsed_time_min(us)": [round(elapsed[0] * 10**6, 3)],
        "elapsed_time_max(us)": [round(elapsed[1] * 10**6, 3)],
        "elapsed_time_avg(us)": [round(elapsed[2] * 10**6, 3)],
        "elapsed_cpu_time_min(us)": [round(cpu_elapsed[0] * 10**6, 3)],
        "elapsed_cpu_time_max(us)": [round(cpu_elapsed[1] * 10**6, 3)],
        "elapsed_cpu_time_avg(us)": [round(cpu_elapsed[2] * 10**6, 3)],
        "overhead_min(us)": [round(overhead[0] * 10**6, 3)],
        "overhead_max(us)": [round(overhead[1] * 10**6, 3)],
        "overhead_avg(us)": [round(overhead[2] * 10**6, 3)],
        "overhead_min(%)": [round(overhead_p[0] * 100)],
        "overhead_max(%)": [round(overhead_p[1] * 100)],
        "overhead_avg(%)": [round(overhead_p[2] * 100)],
        "throughput(TF/s)": [round(throughput, 2)],
        "bandwidth(GB/s)": [round(bw, 2)],
        "arithmetic_intensity": [round(ai, 2)],
    }

    return entry


def mm_flops(args):
    output_csv = pd.DataFrame()

    m = args.m
    n = args.n
    k = args.k
    b = args.b

    if m is None:
        start, stop, step = args.m_range
        m = np.arange(start, stop + 1, step)
    if n is None:
        start, stop, step = args.n_range
        n = np.arange(start, stop + 1, step)
    if k is None:
        start, stop, step = args.k_range
        k = np.arange(start, stop + 1, step)
    if b is None:
        start, stop, step = args.b_range
        b = np.arange(start, stop + 1, step)

    if args.output_file:
        sys.stdout = Tee(args.output_file, args.verbose)

    # Set up defaults for single GPU run
    processes = 1
    queue = None
    try:
        set_start_method("spawn")
    except RuntimeError:
        # Context can only be set once, pass if we are calling mm_flops in a loop.
        pass

    if len(args.cuda_device) > 1:
        print(f"Distibuted run on devices: {args.cuda_device}")
        # Define number of processes and GPUs to use
        processes = NUM_PROC_PER_GPU * len(args.cuda_device)
        queue = Manager().Queue()
        for dev in args.cuda_device:
            queue.put(dev)

    # loop through all sizes to benchmark
    inputs = [
        {
            "queue": queue,
            "m": M,
            "n": N,
            "k": K,
            "b": B,
            "dtype": args.dtype,
            "transpose": args.transpose,
            "num_iterations": args.num_iterations,
            "num_warmup_iterations": args.num_warmup_iterations,
            "profiling": args.profile,
        }
        for M in m
        for N in n
        for K in k
        for B in b
    ]
    with Pool(processes) as pool:
        for entry in tqdm.tqdm(
            pool.imap_unordered(wrapper, inputs), total=len(inputs), ascii=" 🛸="
        ):
            df = pd.DataFrame(entry)
            output_csv = pd.concat([output_csv, df])
        pool.close()
        pool.join()
    print(f">>>Output: {output_csv}")

    df_to_csv(output_csv, args.csv_file, keys=GEMM_CONFIG_KEYS, index=False)


if __name__ == "__main__":
    file_dir = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    add_mm_args(parser, outfile_dir=file_dir, default_output_to_file=True)
    args = parser.parse_args()
    mm_flops(args)
