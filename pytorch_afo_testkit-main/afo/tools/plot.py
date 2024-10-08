import matplotlib
import matplotlib.pyplot as plt
import argparse
import yaml
from collections.abc import Iterable
import shlex
import math
import pandas as pd
from matplotlib import animation

from afo.cookbook.benchmarks.sizing.mm_flops import add_mm_args
from afo.gemm.sizes.run_sizes import SIZES_FILE_LOC
from afo.gemm.collect_common import generate_ranges


def plot_yaml():
    parser = argparse.ArgumentParser()
    add_mm_args(parser)

    with open(SIZES_FILE_LOC, "r") as f:
        sizes = yaml.safe_load(f)

    # Generate a list of markers and another of colors
    markers = ["", ".", ",", "o", "v", "^", "<", ">", "."]
    colors = ["", "r", "g", "b", "c", "m", "y", "k", "w"]
    title_pad = 0.985
    n_subplots = len(sizes) + 1
    n_cols = 3
    n_rows = math.ceil(n_subplots / n_cols)
    plot_pos = range(1, n_subplots + 1)
    cur_pos = 0
    fig = plt.figure()
    plt.style.use("dark_background")
    ax = fig.add_subplot(n_rows, n_cols, plot_pos[cur_pos], projection="3d")
    ax.set_title("Combined", y=title_pad)
    ax.set(xlabel="M", ylabel="N", zlabel="K")

    for section in sizes:
        x = []
        y = []
        z = []
        for size in sizes[section]["args"]:
            print(shlex.split(size))
            args = parser.parse_args(args=shlex.split(size))
            generate_ranges(args)
            for m in args.m if isinstance(args.m, Iterable) else [args.m]:
                for n in args.n if isinstance(args.n, Iterable) else [args.n]:
                    for k in args.k if isinstance(args.k, Iterable) else [args.k]:
                        x.append(m)
                        y.append(n)
                        z.append(k)
        cur_pos += 1
        ax_s = fig.add_subplot(n_rows, n_cols, plot_pos[cur_pos], projection="3d")
        ax_s.set_title(section, y=title_pad)
        ax_s.set(xlabel="M", ylabel="N", zlabel="K")
        ax_s.scatter(
            x,
            y,
            z,
            linewidth=0.1,
            label=section,
            marker=markers[cur_pos],
            color=colors[cur_pos],
        )
        ax.scatter(
            x,
            y,
            z,
            linewidth=0.1,
            label=section,
            marker=markers[cur_pos],
            color=colors[cur_pos],
        )

    fig.set_size_inches(25, 25)
    fig.legend()
    plt.tight_layout()
    plt.savefig("gemmspace.png", dpi=500)


def plot_cvs(args):
    csv = args.csv
    if csv.endswith(".xlsx"):
        df = pd.DataFrame(pd.read_excel(csv))
    else:
        df = pd.read_csv(csv, skiprows=3)

    color_map = matplotlib.colors.TwoSlopeNorm(vcenter=0)

    fig = plt.figure()
    plt.style.use("dark_background")
    ax = fig.add_subplot(projection="3d")
    ax.set_title("Percent difference time")
    ax.set(xlabel="M", ylabel="N", zlabel="K")
    path_collection = ax.scatter(
        df["M"],
        df["N"],
        df["K"],
        # linewidth=0.1,
        marker="v",
        c=df["elapsed_time_avg(us)_%_change"],
        cmap="RdYlGn",  # "Spectral",
        norm=color_map,
    )
    fig.set_size_inches(15, 15)
    fig.legend()
    plt.tight_layout()
    plt.colorbar(path_collection)
    if args.animate:

        def animate(i):
            sin = abs(math.sin(i / 100))
            angle = 90 - (sin * 90)
            ax.view_init(30, azim=angle)
            return fig

        anim = animation.FuncAnimation(fig, animate, frames=315)
        writergif = animation.PillowWriter(fps=30)
        anim.save("sanimation.gif", writer=writergif)
    else:
        plt.savefig("csv_visualized.png", dpi=250)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--yaml", action="store_true", help="Creats plot of the GEMM yaml file"
    )
    group.add_argument("--csv", action="store", help="creates plot of passed in csv")
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()

    if args.yaml:
        plot_yaml()
    if args.csv:
        plot_cvs(args)


if __name__ == "__main__":
    main()
