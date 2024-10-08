import os

from afo.convolution.collect_common import main as collect_common
from afo.tools.benchmarks_utils import run_from_yaml


SIZES_FILE_LOC = f"{os.path.dirname(os.path.abspath(__file__))}/benchmarks.yaml"


def main(command_line=None):
    run_from_yaml(command_line, collect_common, SIZES_FILE_LOC)


if __name__ == "__main__":
    main()
