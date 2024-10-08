import os
import argparse
import re
import yaml
import rccl_benchmark

coll_op_map = {
    "Broadcast": "broadcast_perf",
    "Reduce": "reduce_perf",
    "AllGather": "all_gather_perf",
    "ReduceScatter": "reduce_scatter_perf",
    "AllReduce": "all_reduce_perf",
    "Gather": "gather_perf",
    "Scatter": "scatter_perf",
    "AllToAll": "alltoall_perf",
    "AllToAllv": "alltoallv_perf",
    "Send": "sendrecv_perf",
    "Recv": "sendrecv_perf",
}

reduction_op_map = {
    "0": "sum",
    "1": "prod",
    "2": "max",
    "3": "min",
    "4": "all",
}

data_types_map = {
    "0": "int8",
    "1": "uint8",
    "2": "int32",
    "3": "uint32",
    "4": "int64",
    "5": "uint64",
    "6": "half",
    "7": "float",
    "8": "double",
    "9": "bfloat16",
    # "10": "ncclNumTypes Equivalent?"
}

data_type_bytes_map = {
    "0": 1,
    "1": 1,
    "2": 4,
    "3": 4,
    "4": 8,
    "5": 8,
    "6": 2,
    "7": 4,
    "8": 8,
    "9": 2,
    # "10": Not sure.
}

yaml_map = {
    "broadcast_perf": "broadcast",
    "all_reduce_perf": "all-reduce",
    "all_gather_perf": "all-gather",
    "alltoall_perf": "all-to-all",
    "sendrecv_perf": "pt2pt",
}


def get_useful_info(log_file):
    fs = open(log_file, "r")
    lines = fs.readlines()
    fs.close()

    useful_lines = []
    for j in range(len(lines)):
        line = lines[j].rstrip()
        if "opCount" in line and "sendbuff" in line:
            useful_lines.append(line)

    return useful_lines


def parse_nccl_log(nccl_lines):
    commands = []
    for j in range(len(nccl_lines)):
        line = nccl_lines[j]
        split_list = line.split(" ")
        comm = split_list[split_list.index("INFO") + 1].replace(":", "")
        count = split_list[split_list.index("count") + 1]
        datatype = split_list[split_list.index("datatype") + 1]
        op_type = split_list[split_list.index("op") + 1]
        # root = split_list[split_list.index("root") + 1]
        nnranks = (
            next(item for item in split_list if "nranks" in item)
            .split("=")[1]
            .replace("]", "")
        )
        total_bytes = int(count) * data_type_bytes_map[datatype]

        test_cmd = (
            coll_op_map[comm.replace("mscclFunc", "")]
            + " -d "
            + data_types_map[datatype]
            + " -b "
            + str(total_bytes)
            + " -e "
            + str(total_bytes)
            + " -o "
            + reduction_op_map[op_type]
            + " -g "
            + str(nnranks)
        )
        commands.append((test_cmd, int(nnranks)))

    return commands


def generate_script(commands, output_script):
    filename = output_script + ".sh"
    fs = open(filename, "w")
    for j in range(len(commands)):
        fs.write(commands[j])
        fs.write("\n")
    fs.close()
    print("INFO: Dumped out the commands in a script named: {}".format(filename))


def run_benchmark(rccl_kernels, output_file):
    for kernel in rccl_kernels:
        if kernel["size"] == 0:
            continue
        for bend in ["rccl-test", "pytorch"]:
            rccl_benchmark.benchmark(
                function=kernel["command"],
                backend=bend,
                nproc_per_node=[kernel["num_gpus"]],
                maxsize=kernel["size"],
                dtype=kernel["dtype"],
                results_dir=output_file,
                single=True,
            )


def dump_counts_map(counts_map, output_file):
    filename = output_file + ".yaml"
    parsed_counts_list = []
    pattern = r"(\S+)\s+-d\s+(\S+)\s+-b\s+(\S+)\s+-e\s+(\S+)\s+-o\s+(\S+)\s+-g\s+(\S+)"
    for key, value in counts_map.items():
        matches = re.findall(pattern, key)
        if matches:
            command, dtype, size_lower, size_upper, _, num_gpus = matches[0]
            parsed_counts_map = {
                "command": yaml_map[command],
                "dtype": dtype,
                "size": max(int(size_lower), int(size_upper)),
                "num_gpus": int(num_gpus),
                "counts": value,
            }
        parsed_counts_list.append(parsed_counts_map)

    with open(filename, "w") as fs:
        yaml.dump(parsed_counts_list, fs, default_flow_style=False)
    print(
        "INFO: Dumped the count of each command in a YAML file named: {}".format(
            filename
        )
    )

    return parsed_counts_list


def load_counts_map(input_file):
    with open(input_file, "r") as fs:
        parsed_counts_list = yaml.safe_load(fs)

    if not isinstance(parsed_counts_list, list):
        raise ValueError("The YAML file is not in correct format")

    required_keys = {"command", "dtype", "size", "num_gpus", "counts"}
    for item in parsed_counts_list:
        if not isinstance(item, dict) or not required_keys.issubset(item):
            raise ValueError(
                "Each item in the list must be a dictionary with the required structure."
            )

    return parsed_counts_list


def get_unique_commands(commands_and_nranks):
    unique_values = []
    counts_map = {}
    nranks_map = {}
    for c_and_nr in commands_and_nranks:
        cmd = c_and_nr[0]
        nranks = c_and_nr[1]
        if cmd not in unique_values:
            counts_map[cmd] = 1
            nranks_map[cmd] = nranks
            unique_values.append(cmd)
        else:
            counts_map[cmd] = counts_map[cmd] + 1
    assert len(counts_map) == len(nranks_map)
    for cmd in counts_map.keys():
        # assert counts_map[cmd] % nranks_map[cmd] == 0
        counts_map[cmd] = int(counts_map[cmd] / nranks_map[cmd])
    return unique_values, counts_map


def main():
    os.makedirs("./RCCL_kernels_capture/traces/", exist_ok=True)
    os.makedirs("./RCCL_kernels_capture/perfs/", exist_ok=True)
    log_file = os.path.abspath(args.input)
    if log_file.lower().endswith((".yaml", ".yml")):
        rccl_kernels = load_counts_map(log_file)
    else:
        nccl_lines = get_useful_info(log_file)
        commands_and_nranks = parse_nccl_log(nccl_lines)
        new_commands, counts_map = get_unique_commands(commands_and_nranks)
        rccl_kernels = dump_counts_map(
            counts_map, "./RCCL_kernels_capture/traces/" + args.output + "_traces"
        )
    if args.benchmark:
        run_benchmark(
            rccl_kernels, "./RCCL_kernels_capture/perfs/" + args.output + "_perf"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Either pass the Log from app with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL or pass the processed yaml file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="model_rccl_kernels",
        help="Output files",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark the collected RCCL kernels"
    )
    args = parser.parse_args()
    main()
