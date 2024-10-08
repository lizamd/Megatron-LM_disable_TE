import yaml
import argparse
from multiprocessing import Pool, set_start_method, Manager
import tqdm
import os

from afo.cookbook.benchmarks.sizing.mm_flops import wrapper


def add_args(parser):
    parser.add_argument(
        "input_yaml",
        metavar="ROCBLAS_BENCH YAML",
        type=argparse.FileType("r"),
        help="rocblas bench yaml file to be converted",
    )
    parser.add_argument(
        "--cuda_device",
        nargs="*",
        type=int,
        default=[0],
        help="The cuda device(s) to run the benchmark on",
    )


AFO_DTYPES = {
    "f16_r": "fp16",
    "bf16_r": "bf16",
    "f32_r": "fp32",
}


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)

    inputs = []

    args = parser.parse_args()
    input_yaml = yaml.safe_load(args.input_yaml)

    try:
        set_start_method("spawn")
    except RuntimeError:
        # Context can only be set once, pass if we are calling mm_flops in a loop.
        pass

    # XXX: Copied from mm_flops.py as POC. TODO: Properly support
    print(f"Distibuted run on devices: {args.cuda_device}")
    # Define number of processes and GPUs to use
    processes = 1 * len(args.cuda_device)
    queue = Manager().Queue()
    for dev in args.cuda_device:
        queue.put(dev)

    for gemm in input_yaml:
        # loop through all sizes to benchmark

        default_dtype = "f32_r" if "sgemm" in gemm["rocblas_function"] else "f16_r"

        # FIXME: This is a workaround hack to skip known bad configs with hipblaslt:
        if (
            gemm["transA"] == "N"
            and gemm["transB"] == "N"
            and (
                (gemm.get("a_type", default_dtype) == "f16_r")
                or (gemm.get("a_type", default_dtype) == "f32_r")
            )
        ):
            continue

        new = {
            "queue": queue,
            "m": gemm["M"],
            "n": gemm["N"],
            "k": gemm["K"],
            "b": gemm.get("batch_count", 1),
            "transpose": gemm["transA"] + gemm["transB"],
            "dtype": AFO_DTYPES.get(gemm.get("a_type", default_dtype), "fp16"),
            "num_iterations": 1,
            "num_warmup_iterations": 0,
            "profiling": False,
        }

        if new not in inputs:
            inputs.append(new)

    # Set untunable ops envs
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "afo_tune_device_%d_from_yaml.csv"
    with Pool(processes) as pool:
        for entry in tqdm.tqdm(
            pool.imap_unordered(wrapper, inputs), total=len(inputs), ascii=" 🛸="
        ):
            pass
        pool.close()
        pool.join()
    # Get csv files
    csvs = [
        filename for filename in os.listdir(".") if filename.endswith("from_yaml.csv")
    ]

    combined_csv = ""
    for csv in csvs:
        with open(csv, "r") as f:
            if combined_csv == "":
                combined_csv = f.read()
            else:
                for line in f:
                    if not line.startswith("Validator"):
                        combined_csv += line
    # Writeout combined files per device
    for device in args.cuda_device:
        with open(f"full_tuned{device}.csv", "w") as f:
            f.write(combined_csv)

    print("To enable tuning please set:")
    print("export PYTORCH_TUNABLEOP_FILENAME=full_tuned%d.csv")
    print("export PYTORCH_TUNABLEOP_TUNING=0")
    print("export PYTORCH_TUNABLEOP_ENABLED=1")


if __name__ == "__main__":
    main()
