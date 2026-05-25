from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ppnn_experiments import load_pickle, plot_final_vs_lambda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--out-prefix", type=Path, default=Path("results/metric_vs_lambda"))
    parser.add_argument("--no-min-star", action="store_true")
    args = parser.parse_args()

    payloads = [load_pickle(path) for path in args.inputs]

    fig, ax = plot_final_vs_lambda(
        payloads,
        metric="loss_end",
        save_path=args.out_prefix.with_name(args.out_prefix.name + "_loss.pdf"),
        show_min_star=not args.no_min_star,
    )
    plt.close(fig)

    fig, ax = plot_final_vs_lambda(
        payloads,
        metric="acc_end",
        save_path=args.out_prefix.with_name(args.out_prefix.name + "_acc.pdf"),
        show_min_star=not args.no_min_star,
    )
    plt.close(fig)

    print(f"saved {args.out_prefix.with_name(args.out_prefix.name + '_loss.pdf')}")
    print(f"saved {args.out_prefix.with_name(args.out_prefix.name + '_acc.pdf')}")


if __name__ == "__main__":
    main()