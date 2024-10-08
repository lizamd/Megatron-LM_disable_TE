import pandas as pd
import os
import subprocess
import argparse
import logging

from ..tools.utils import df_to_csv, query_rpd_database
from ..tools.csv_compare import csvMerger
from ..cookbook.benchmarks.sizing.conv_flops import add_conv_args, conv_flops, get_range


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", filename="benchmark.log", filemode="w"
    )


def parse_backend(arg):
    return set(arg.split(","))


def generate_ranges(args):
    """Generate configuration ranges based on provided arguments."""
    ranges = {
        "batch_size_range": get_range(
            args.batch_size_list, args.batch_size_range, args.batch_size
        ),
        "input_size_range": get_range(
            args.input_size_list, args.input_size_range, args.input_size
        ),
        "kernel_size_range": get_range(
            args.kernel_size_list, args.kernel_size_range, args.kernel_size
        ),
        "stride_range": get_range(args.stride_list, args.stride_range, args.stride),
        "padding_range": get_range(args.padding_list, args.padding_range, args.padding),
        "input_channels_range": get_range(
            args.input_channels_list, None, args.input_channels
        ),
        "output_channels_range": get_range(
            args.output_channels_list, None, args.output_channels
        ),
    }

    return ranges


def run_miopen_driver(args, config):
    """Runs the MIOpen driver and returns the output."""
    base_command = [
        "convfp16",
        "-n",
        str(config["batch_size"]),
        "-c",
        str(config["input_channels"]),
        "-H",
        str(config["input_size"]),
        "-W",
        str(config["input_size"]),
        "-k",
        str(config["output_channels"]),
        "-y",
        str(config["kernel_size"]),
        "-x",
        str(config["kernel_size"]),
        "-p",
        str(config["padding"]),
        "-q",
        str(config["padding"]),
        "-u",
        str(config["stride"]),
        "-v",
        str(config["stride"]),
        "-l",
        "1",
        "-j",
        "1",
        "-m",
        "conv",
        "-g",
        "1",
        "-F",
        "1",
        "-t",
        "1",
        "--in_layout",
        "NHWC",
        "--out_layout",
        "NHWC",
        "--fil_layout",
        "NHWC",
    ]

    if args.profile == "rpd":
        profile_dir = os.getenv("PROFILE_DIR", default="")
        os.makedirs(profile_dir, exist_ok=True)
        trace_file_name = os.path.join(
            profile_dir,
            f"conv_trace_i{config['input_size']}_k{config['kernel_size']}_s{config['stride']}_p{config['padding']}_miopen.rpd",
        )

        command = [
            "runTracer.sh",
            "-o",
            trace_file_name,
            "/opt/rocm/bin/MIOpenDriver",
        ] + base_command
    else:
        command = ["/opt/rocm/bin/MIOpenDriver"] + base_command

    process = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if args.profile == "rpd":
        query_rpd_database(trace_file_name)
    return process.stdout


def parse_miopen_driver_output(output):
    """Parses MIOpen driver output for relevant metrics."""
    metrics = {}
    gflops_index = 11
    bandwidth_index = 12
    time_index = 13
    lines = output.splitlines()

    for i, line in enumerate(lines):
        if line.startswith("stats: name"):
            if i + 1 < len(lines):
                values_line = lines[i + 1]
                values = values_line.split(",")
                try:
                    gflops = float(values[gflops_index].strip())
                    bandwidth = float(values[bandwidth_index].strip())
                    time_ms = float(values[time_index].strip())

                    elapsed_time_us = time_ms * 1000
                    metrics = {
                        "elapsed_time(us)": elapsed_time_us,
                        "throughput(TF/s)": gflops / 1000,
                        "bandwidth(GB/s)": bandwidth,
                    }
                except ValueError:
                    logging.error(f"Error parsing metrics from line: {values_line}")
                break

    return metrics


def run_and_collect_miopen_metrics(args, ranges):
    """Runs MIOpen benchmarks and collects results into a dataframe."""
    results = []
    configs = (
        "batch_size,input_size,kernel_size,stride,padding,"
        "input_channels,output_channels"
    )

    output_csv = pd.DataFrame(
        columns=[
            "batch_size",
            "input_size",
            "kernel_size",
            "stride",
            "padding",
            "input_channels",
            "output_channels",
        ]
    )

    for batch_size in ranges["batch_size_range"]:
        for input_size in ranges["input_size_range"]:
            for kernel_size in ranges["kernel_size_range"]:
                for input_channels in ranges["input_channels_range"]:
                    for output_channels in ranges["output_channels_range"]:
                        for stride in ranges["stride_range"]:
                            for padding in ranges["padding_range"]:

                                output = run_miopen_driver(
                                    args,
                                    {
                                        "batch_size": batch_size,
                                        "input_size": input_size,
                                        "kernel_size": kernel_size,
                                        "input_channels": input_channels,
                                        "output_channels": output_channels,
                                        "stride": stride,
                                        "padding": padding,
                                    },
                                )

                                metrics = parse_miopen_driver_output(output)
                                entry = [
                                    {
                                        "batch_size": batch_size,
                                        "input_size": input_size,
                                        "kernel_size": kernel_size,
                                        "stride": stride,
                                        "padding": padding,
                                        "input_channels": input_channels,
                                        "output_channels": output_channels,
                                        **metrics,
                                    }
                                ]

                                df = pd.DataFrame(entry)
                                output_csv = pd.concat([output_csv, df])

                                if args.append:
                                    df_to_csv(
                                        df,
                                        args.csv_file,
                                        keys=configs,
                                        mode="a",
                                        index=False,
                                    )

                                if args.verbose:
                                    logging.info(f"Completed run:\n{df}")
    if not args.append:
        df_to_csv(output_csv, args.csv_file, keys=configs, index=False)

    logging.info(f"Benchmark Summary:\n{output_csv}")


def merge_and_save_benchmark_csvs(pytorch_csv, miopen_csv, output_csv):
    """Merges the given CSVs into a single output."""
    merger = csvMerger(pytorch_csv, miopen_csv, output_csv, "_pytorch", "_miopen")
    merged_df = merger.merge_csv()
    if output_csv.endswith(".xlsx"):
        merger.df_to_excel(merged_df)
    else:
        merged_df.to_csv(output_csv, index=False)


def add_cc_args(parser):
    parser.add_argument(
        "--backend",
        type=parse_backend,
        default=set(["pytorch"]),
        help="Comma-separated list of backends for convolution benchmarks. Options: pytorch, miopen",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Appends results to CSV file, recording benchmarks across multiple runs.",
    )


def main(command_line=None):
    setup_logging()

    parser = argparse.ArgumentParser()
    add_conv_args(parser)
    add_cc_args(parser)
    args = parser.parse_args(command_line)

    csv_base = os.path.dirname(args.csv_file)
    csv_file_name = os.path.basename(args.csv_file)

    ranges = generate_ranges(args)

    if "miopen" in args.backend:
        logging.info("Collecting MIOpen Benchmarks ...")
        try:
            miopen_csv_file = f"{csv_base}/miopen_{csv_file_name}"
            args.csv_file = miopen_csv_file
            run_and_collect_miopen_metrics(args, ranges)
            logging.info("Finished Collecting MIOpen Benchmarks")
        except Exception as e:
            logging.error(f"Error collecting MIOpen benchmarks: {e}")
            args.miopen = False

    if "pytorch" in args.backend:
        logging.info("Collecting Pytorch Benchmarks ...")
        try:
            pytorch_csv_file = f"{csv_base}/pytorch_{csv_file_name}"
            args.csv_file = pytorch_csv_file
            conv_flops(args)
            logging.info("Finished Collecting Pytorch Benchmarks")
        except Exception as e:
            logging.error(f"Error collecting PyTorch benchmarks: {e}")
            args.pytorch = False
            raise

    if "pytorch" in args.backend and "miopen" in args.backend:
        logging.info("Merging Data For Comparison")
        merge_and_save_benchmark_csvs(
            pytorch_csv_file, miopen_csv_file, f"{csv_base}/combined_benchmark.xlsx"
        )
        logging.info("Merge Complete!")


if __name__ == "__main__":
    main()
