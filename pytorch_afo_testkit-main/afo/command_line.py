import sys
from afo.gemm.collect_common import main as gemm
from afo.gemm.benchmarks.run_benchmarks import main as gemm_benchmarks
from afo.convolution.benchmarks.run_benchmarks import main as conv_benchmarks
from afo.convolution.collect_common import main as conv
import argparse


def full_help_message(parser):
    parser.print_help()

    # retrieve subparsers from parser
    subparsers_actions = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    # there will probably only be one subparser_action,
    # but better save than sorry
    for subparsers_action in subparsers_actions:
        # get all subparsers and print help
        for choice, subparser in subparsers_action.choices.items():
            print("Subparser '{}'".format(choice))
            print(subparser.format_help())

    parser.exit()


def cli():
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="mode")
    gemm_parser = subparsers.add_parser(
        "gemm", help="main interface for the GEMM benchmarks in afo", add_help=False
    )
    gemm_parser.add_argument(
        "--benchmarks", action="store_true", help="Runs full suites of benchmarks"
    )

    conv_parser = subparsers.add_parser(
        "conv",
        help="main interface for the convolution benchmarks in afo",
        add_help=False,
    )

    conv_parser.add_argument(
        "--benchmarks", action="store_true", help="Runs full suites of benchmarks"
    )

    subparsers.add_parser(
        "rccl", help="main interface for the RCCL benchmarks in afo", add_help=False
    )

    args, unknown = parser.parse_known_args()

    # Remove our args from the overall before passing down
    if args.mode == "gemm":
        sys.argv.remove(args.mode)
        if args.benchmarks:
            gemm_benchmarks(unknown)
        else:
            gemm()
    elif args.mode == "benchmark_gemms":
        sys.argv.remove(args.mode)
        gemm_benchmarks()
    elif args.mode == "conv":
        sys.argv.remove(args.mode)
        if args.benchmarks:
            conv_benchmarks(unknown)
        else:
            conv()
    else:
        full_help_message(parser)
