import yaml
import pandas as pd
import argparse
import shlex
import os
import re

from afo.tools.utils import HEADER, FOOTER, KEYS_R, df_to_csv
from afo.tools.csv_compare import search_file


# Merge all csvs under a directory path
def merge_dir(directory="."):
    header_p = re.compile(HEADER)
    footer_p = re.compile(FOOTER)
    keys_p = re.compile(KEYS_R)
    keys = ""

    for root, dirs, _ in os.walk(directory, topdown=False):
        df_list = []
        # Can't use the files from os walk, no updated in real time.
        files = [f for f in os.listdir(root) if f.endswith(".csv")]
        for name in files:
            file_path = os.path.join(root, name)
            _, footer, keys = search_file(file_path, header_p, footer_p, keys_p)
            df = pd.read_csv(file_path, skiprows=footer + 1)
            df_list.append(df)
        if len(df_list) > 0:
            combined_csv = os.path.join(
                os.path.dirname(root), f"combined_{os.path.basename(root)}.csv"
            )
            big_df = pd.concat(df_list, ignore_index=True)
            # XXX: Assumes all keys are the same for all combined csvs.
            df_to_csv(big_df, combined_csv, keys=keys, index=False)


def run_from_yaml(
    command_line=None, func=None, yaml_loc="", results_dir="results/benchmarks"
):
    # Setup dynamic args
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--section",
        nargs="*",
        default="all",
        help="Pick which section(s) of the yaml to run",
    )

    args, unknown = parser.parse_known_args(command_line)

    # Special condition "all" to run all yaml
    all_sections = args.section == "all"

    with open(yaml_loc, "r") as f:
        sizes = yaml.safe_load(f)

    for section in sizes:
        if all_sections or section in args.section:
            for size in sizes[section]["args"]:
                inputs = shlex.split(f"{size}")
                inputs.extend(unknown)
                outfile = [
                    "--csv_file",
                    f"{results_dir}/{section}/{hash(size)}.csv",
                ]
                inputs.extend(outfile)
                func(inputs)

    merge_dir(results_dir)
