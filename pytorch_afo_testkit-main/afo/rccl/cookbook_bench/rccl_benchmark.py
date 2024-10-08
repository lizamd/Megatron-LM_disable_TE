import argparse
import subprocess
import re
import pandas as pd
import os
import torch


def benchmark(function, backend, dtype, nproc_per_node, results_dir, single, maxsize):
    os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
    entries = []
    os.makedirs(results_dir, exist_ok=True)
    func = function
    for nproc in nproc_per_node:
        if backend == "pytorch":
            if single:
                way_to_run = "--single"
            else:
                way_to_run = "--scan"
            command = [
                "torchrun",
                "--nproc_per_node={}".format(nproc),
                "../../cookbook/benchmarks/communication/run_all.py",
                way_to_run,
                "--" + function,
                "--dist",
                "torch",
                "--backend",
                "nccl",
                "--dtype",
                dtype,
                "--maxsize",
                str(maxsize),
            ]
        else:
            if function == "all-reduce":
                func = "all_reduce_perf"
            elif function == "all-gather":
                func = "all_gather_perf"
            elif function == "broadcast":
                func = "broadcast_perf"
            elif function == "pt2pt":
                func = "sendrecv_perf"
            elif function == "all-to-all":
                func = "alltoall_perf"

            if single:
                b = maxsize
                e = maxsize
            else:
                tensor = torch.empty((), dtype=getattr(torch, dtype))
                item_size = tensor.element_size()
                b = 4 * item_size * nproc / 2
                e = 16 * item_size * nproc / 2
                e = str(e) + "M"

            func = "./rccl-tests/build/" + func

            command = [
                func,
                "-b",
                str(b),
                "-e",
                str(e),
                "-f",
                "2",
                "-d",
                dtype,
                "-g",
                str(nproc),
            ]
        # Launch and capture output.
        result = subprocess.run(command, capture_output=True, text=True)
        output = result.stdout

        # print(result)

        if backend == "pytorch":
            # Parse the output
            pattern = re.compile(
                r"(\d+(\.\d+)?) (KB|MB|B)\s+(\d+x\d+)\s+(\d+\.\d+) (ms|us)\s+(\d+\.\d+)\s+(\d+\.\d+)"
            )

            # Parse the performance metrics from the output
            for match in pattern.finditer(output):
                (
                    size,
                    _,
                    size_unit,
                    description,
                    duration,
                    dur_unit,
                    throughput,
                    bus_bw,
                ) = match.groups()

                if dur_unit == "ms":
                    duration = float(duration) * 1000

                entries.append(
                    {
                        "NP Size": nproc,
                        "Size": size + " " + size_unit,
                        "Description": description,
                        "Duration(us)": duration,
                        "Throughput (Gbps)": float(throughput),
                        "BusBW (Gbps)": float(bus_bw),
                        "Data Type": dtype,
                        "Backend": backend,
                    }
                )
        else:
            pattern = re.compile(
                r"\s*(\d+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)"
            )
            for match in pattern.finditer(output):
                size, num_elem, size_type, _, _, duration, throughput, bus_bw = (
                    match.groups()
                )

                size = int(size)
                num_elem = int(num_elem)
                throughput = float(throughput) * 8
                bus_bw = float(bus_bw) * 8

                # rccl test all gather is wrong
                # issue reported to rccl test
                if int(num_elem) <= 0:
                    continue
                if func == "all-gather":
                    size = size // 2

                description = str(num_elem) + "x" + str(size // num_elem)

                if size // 1048576 > 0:
                    size = size / 1048576
                    size_unit = "MB"
                elif size // 1024 > 0:
                    size = size / 1024
                    size_unit = "KB"
                else:
                    size_unit = "B"

                entries.append(
                    {
                        "NP Size": nproc,
                        "Size": str(size) + " " + size_unit,
                        "Description": description,
                        "Duration(us)": duration,
                        "Throughput (Gbps)": throughput,
                        "BusBW (Gbps)": bus_bw,
                        "Data Type": dtype,
                        "Backend": backend,
                    }
                )

    # Convert data to a DataFrame
    df = pd.DataFrame(entries)
    # Print and save the DataFrame
    print(df)
    if single:
        df_filename = f"{results_dir}_{backend}_{function}_performance_{maxsize}_bytes_{dtype}.csv"
    else:
        df_filename = f"{results_dir}/{backend}_{function}_performance_{dtype}.csv"

    df.to_csv(df_filename, index=False)
    print(f"Output saved to {df_filename}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Benchmark all_gather operations with PyTorch and NCCL across different nproc_per_node values and data types."
    )
    parser.add_argument(
        "--function",
        type=str,
        default="all-reduce",
        choices=["all-reduce", "all-gather", "broadcast", "pt2pt", "all-to-all"],
        help="Functions in RCCL. Options are all-reduce, all-gather, broadcast, pt2pt and all-to-all. Default is all-reduce.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pytorch",
        choices=["pytorch", "rccl-test"],
        help="Backend for RCCL benchmark. Options are pytorch and rccl-test. Default is pytorch.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=[
            "float16",
            "float32",
            "double",
            "bfloat16",
            "int8",
            "uint8",
            "int32",
            "int64",
            "uint64",
        ],
        help="Data type for the tensors to be reduced. Options are float16 and float32. Default is float16.",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        nargs="+",
        default=[2],
        help="A list of nproc_per_node values to run the benchmarks with. Example: --nproc_per_node 2 4 6 8",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/perf",
        help="Directory to store the resulting csv files into.  Default is results/",
    )
    parser.add_argument("--single", action="store_true", help="Execute single setup")
    parser.add_argument(
        "--maxsize",
        type=int,
        default=24,
        help="Max message size as a power of 2",
    )

    args = parser.parse_args()

    benchmark(
        args.function,
        args.backend,
        args.dtype,
        args.nproc_per_node,
        args.results_dir,
        args.single,
        args.maxsize,
    )
