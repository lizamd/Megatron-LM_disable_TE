import torch
import os
import pandas as pd
import sys
import argparse

from afo.cookbook.benchmarks.sizing.utils import Tee, benchmark_convolution
from afo.tools.utils import df_to_csv


def add_conv_args(parser):
    single_value_args = [
        "input_channels",
        "output_channels",
        "batch_size",
        "input_size",
        "kernel_size",
        "stride",
        "padding",
    ]
    for arg in single_value_args:
        parser.add_argument(f"--{arg}", type=int, default=None)

    list_args = [
        "input_channels_list",
        "output_channels_list",
        "batch_size_list",
        "input_size_list",
        "kernel_size_list",
        "stride_list",
        "padding_list",
    ]
    for arg in list_args:
        parser.add_argument(
            f"--{arg}", type=lambda s: [int(item) for item in s.split()], default=None
        )

    range_args = [
        "batch_size_range",
        "input_size_range",
        "kernel_size_range",
        "stride_range",
        "padding_range",
    ]
    for arg in range_args:
        parser.add_argument(
            f"--{arg}", type=lambda s: list(map(int, s.split())), default=None
        )

    parser.add_argument("--num_iterations", type=int, default=200)
    parser.add_argument("--num_warmup_iterations", type=int, default=50)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--output_file", type=str, default="results/conv.out")
    parser.add_argument("--csv_file", type=str, default="results/conv.csv")
    parser.add_argument(
        "--verbose", action="store_true", help="Log to stdout besides output_file"
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["none", "torch", "rpd"],
        default="none",
        help='Profiling mode: "none", "torch", or "rpd"',
    )


def get_range(list_value, range_value, single_value):
    """Helper function to create a usable range from list, range, or single value arguments."""
    if list_value:
        return list_value
    elif range_value:
        start, stop, step = range_value
        return range(start, stop + 1, step)
    elif single_value is not None:
        return range(single_value, single_value + 1, 1)
    return []


def conv_flops(args):
    torch.cuda.set_device(args.cuda_device)

    output_csv = pd.DataFrame(
        columns=[
            "batch_size",
            "input_size",
            "kernel_size",
            "stride",
            "padding",
            "input_channels",
            "output_channels",
            "elapsed_time(us)",
            "throughput(TF/s)",
            "bandwidth(GB/s)",
            "arithmetic_intensity",
        ]
    )

    batch_size_range = get_range(
        args.batch_size_list, args.batch_size_range, args.batch_size
    )
    input_size_range = get_range(
        args.input_size_list, args.input_size_range, args.input_size
    )
    kernel_size_range = get_range(
        args.kernel_size_list, args.kernel_size_range, args.kernel_size
    )
    stride_range = get_range(args.stride_list, args.stride_range, args.stride)
    padding_range = get_range(args.padding_list, args.padding_range, args.padding)
    input_channels_range = get_range(
        args.input_channels_list, None, args.input_channels
    )
    output_channels_range = get_range(
        args.output_channels_list, None, args.output_channels
    )

    configs = (
        "batch_size,input_size,kernel_size,stride,padding,"
        "input_channels,output_channels"
    )

    file_dir = os.path.abspath(os.path.dirname(__file__))
    output_file = os.path.join(file_dir, args.output_file)

    sys.stdout = Tee(output_file, args.verbose)

    for batch_size in batch_size_range:
        for input_size in input_size_range:
            for kernel_size in kernel_size_range:
                for stride in stride_range:
                    for padding in padding_range:
                        for input_channels in input_channels_range:
                            for output_channels in output_channels_range:
                                elapsed, throughput, bw, ai = benchmark_convolution(
                                    input_channels,
                                    output_channels,
                                    batch_size,
                                    input_size,
                                    kernel_size,
                                    stride,
                                    padding,
                                    args.num_iterations,
                                    args.num_warmup_iterations,
                                    args.profile,
                                )

                                entry = {
                                    "batch_size": [batch_size],
                                    "input_size": [input_size],
                                    "kernel_size": [kernel_size],
                                    "stride": [stride],
                                    "padding": [padding],
                                    "input_channels": [input_channels],
                                    "output_channels": [output_channels],
                                    "elapsed_time(us)": [round(elapsed * 10**6, 3)],
                                    "throughput(TF/s)": [round(throughput, 2)],
                                    "bandwidth(GB/s)": [round(bw, 2)],
                                    "arithmetic_intensity": [ai],
                                }

                                df = pd.DataFrame(entry)
                                output_csv = pd.concat(
                                    [output_csv, df], ignore_index=True
                                )

                                if args.append:
                                    df_to_csv(
                                        df,
                                        args.csv_file,
                                        keys=configs,
                                        mode="a",
                                        index=False,
                                    )

    if not args.append:
        df_to_csv(output_csv, args.csv_file, keys=configs, index=False)

    print(f"{output_csv}")


if __name__ == "__main__":
    # Setup dynamic args
    parser = argparse.ArgumentParser()
    add_conv_args()
    args = parser.parse_args()

    conv_flops(args)
