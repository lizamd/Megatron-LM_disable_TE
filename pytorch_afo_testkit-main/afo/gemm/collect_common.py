import argparse
import subprocess
import pandas as pd
import os
import tempfile
import numpy as np
import re
from collections.abc import Iterable
from multiprocessing import Pool

from afo.tools.utils import df_to_csv, is_float_try
from afo.tools.csv_compare import csvMerger
from afo.cookbook.benchmarks.sizing.mm_flops import (
    add_mm_args,
    mm_flops,
    GEMM_CONFIG_KEYS,
)


def generate_rocbench_yaml_line(args, m, n, k, b):
    rocblas_function = "rocblas_gemm_ex"
    additional_args = ""
    if b != 1:
        rocblas_function = "rocblas_gemm_strided_batched_ex"
        additional_args = (
            f"stride_a: {m*k},"
            f"stride_b: {n*k},"
            f"stride_c: {n*m},"
            f"stride_b: {n*m},"
            f"batch_count: {b},"
        )

    return (
        f"- {{"
        f" rocblas_function: {rocblas_function},"
        f" atomics_mode: atomics_allowed,"
        f" a_type: f16_r,"
        f" b_type: f16_r,"
        f" c_type: f16_r,"
        f" d_type: f16_r,"
        f" compute_type: f32_r,"
        f" transA: {args.transpose[0]},"
        f" transB: {args.transpose[1]},"
        f" M: {m},"
        f" N: {n},"
        f" K: {k},"
        f" alpha: 1,"
        f" lda: {m},"
        f" beta: 0,"
        f" ldb: {k},"
        f" ldc: {m},"
        f" ldd: {m},"
        f" flags: none,"
        f" iters: {args.num_iterations},"
        f" cold_iters: {args.num_warmup_iterations},"
        f" {additional_args}"
        f" }}{os.linesep}"
    )


def call_rocbench(inputs):
    device, yaml = inputs
    fp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    fp.writelines(yaml)
    fp.seek(0)
    fp.close()

    cmd = f"rocblas-bench --device {device} --yaml {fp.name}"
    print(cmd)
    process = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
    )
    os.unlink(fp.name)
    return process.stdout.decode("utf-8")


def run_and_collect_rocblas_metrics(args):
    results = []
    # XXX: If the list of headers grows more, split into its own selection function.
    header = (
        r"transA,"
        r"transB,"
        r"M,"
        r"N,"
        r"K,"
        r"alpha,"
        r"lda,"
        r"beta,"
        r"ldb,"
        r"ldc,"
        r"ldd,"
        r"batch_count,"
        r"rocblas-Gflops,"
        r"us"
        r"|"
        r"transA,"
        r"transB,"
        r"M,"
        r"N,"
        r"K,"
        r"alpha,"
        r"lda,"
        r"stride_a,"
        r"beta,"
        r"ldb,"
        r"stride_b,"
        r"ldc,"
        r"stride_c,"
        r"ldd,"
        r"stride_d,"
        r"batch_count,"
        r"rocblas-Gflops,"
        r"us"
    )
    header_re = re.compile(header)

    # Construct yamls, evenly divided by the number of devices to use.
    i = 0
    n_devices = len(args.cuda_device)
    rocblas_yamls = [""] * n_devices
    for m in args.m if isinstance(args.m, Iterable) else [args.m]:
        for n in args.n if isinstance(args.n, Iterable) else [args.n]:
            for k in args.k if isinstance(args.k, Iterable) else [args.k]:
                for b in args.b if isinstance(args.b, Iterable) else [args.b]:
                    # Generate configs
                    rocblas_yamls[i % n_devices] += generate_rocbench_yaml_line(
                        args, m, n, k, b
                    )
                    i += 1

    # Assign a device to use per yaml constructed
    inputs = []
    i = 0
    for yaml in rocblas_yamls:
        inputs.insert(i, (args.cuda_device[i], yaml))
        i += 1

    # Run benchmark
    raw_results = ""
    with Pool(n_devices) as pool:
        for result in pool.imap_unordered(call_rocbench, inputs):
            raw_results += result

    print(raw_results)

    # TODO: When we have batched jobs we need to redo the line fetch logic
    header_matched = False
    for line in raw_results.splitlines():
        if header_re.match(line):
            header_matched = line
        elif header_matched:
            d = {
                k: is_float_try(v.strip())
                for k, v in zip(header_matched.split(","), line.split(","))
            }
            results.append(d)
            header_matched = False

    # XXX: To allow batched and not batched to live in the same CSV going to remove
    # the strided fields. TODO: Come up with a more sustainable solution
    for r in results:
        r.pop("stride_a", None)
        r.pop("stride_b", None)
        r.pop("stride_c", None)
        r.pop("stride_d", None)

    df = pd.DataFrame(results)
    # Rename to align to mm_flops output
    df = df.rename(
        columns={
            "us": "elapsed_time(us)",
            "rocblas-Gflops": "throughput(TF/s)",
            "batch_count": "B",
        }
    )

    df["throughput(TF/s)"] = df["throughput(TF/s)"].apply(lambda x: x / 1000.0)
    df_to_csv(df, args.csv_file, keys=GEMM_CONFIG_KEYS, index=False)


def generate_ranges(args):
    if args.m is None:
        start, stop, step = args.m_range
        args.m = np.arange(start, stop + 1, step)
    if args.n is None:
        start, stop, step = args.n_range
        args.n = np.arange(start, stop + 1, step)
    if args.k is None:
        start, stop, step = args.k_range
        args.k = np.arange(start, stop + 1, step)


def merge_and_save_benchmark_csvs(pytorch_csv, rocbench_csv, output_csv):
    merger = csvMerger(pytorch_csv, rocbench_csv, output_csv, "_pytorch", "_rocbench")
    merged_df = merger.merge_csv()
    if output_csv.endswith(".xlsx"):
        merger.df_to_excel(merged_df)
    else:
        merged_df.to_csv(output_csv, index=False)


def add_cc_args(parser):
    parser.add_argument("--pytorch", action="store_true", help="Enable Pytorch")
    parser.add_argument("--rocbench", action="store_true", help="Enable MIOpen")
    parser.add_argument("--debug", action="store_true", help="More versbose debug")


def main(command_line=None):
    # Setup dynamic args
    parser = argparse.ArgumentParser()
    add_mm_args(parser, default_output_to_file=True)
    add_cc_args(parser)

    args = parser.parse_args(command_line)

    # Either is required
    if not (args.pytorch or args.rocbench):
        print(
            f"Either pytorch:{args.pytorch}"
            f" or rocbench:{args.rocbench} is required to run benchmark"
        )
        exit(1)

    generate_ranges(args)

    csv_base = os.path.dirname(args.csv_file)
    csv_file_name = os.path.basename(args.csv_file)

    if args.pytorch:
        print("Collecting Pytorch Benchmarks ...")
        try:
            if args.rocbench:
                # Rename the output csv if we are doing both
                pytorch_csv_file = f"{csv_base}/torch_{csv_file_name}"
                args.csv_file = pytorch_csv_file
            mm_flops(args)
            print("Finished Collecting Pytorch Benchmarks")
        except Exception as e:
            print(f"Error collecting PyTorch benchmarks: {e}")
            args.pytorch = False
            if args.debug:
                raise

    if args.rocbench:
        print("Collecting rocBench Benchmarks ...")
        try:
            if args.pytorch:
                # Rename the output csv if we are doing both
                rocbench_csv_file = f"{csv_base}/rocbench_{csv_file_name}"
                args.csv_file = rocbench_csv_file
            run_and_collect_rocblas_metrics(args)
            print("Finished Collecting rocBench Benchmarks")
        except Exception as e:
            print(f"Error collecting rocBench benchmarks: {e}")
            args.rocbench = False
            if args.debug:
                raise

    if args.pytorch and args.rocbench:
        print("Merging Data For Comparison")
        # Merge the CSV files into a combined benchmark file
        merge_and_save_benchmark_csvs(
            pytorch_csv_file, rocbench_csv_file, f"{csv_base}/combined_benchmark.xlsx"
        )
        print("Merge Complete!")


if __name__ == "__main__":
    main()
