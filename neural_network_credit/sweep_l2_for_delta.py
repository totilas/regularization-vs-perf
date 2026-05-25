from __future__ import annotations

import argparse
from pathlib import Path

from ppnn_experiments import run_lambda_sweep, save_pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--l2-grid", type=float, nargs="+", required=True)
    parser.add_argument("--test-frac", type=float, default=0.9)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_noise_only = args.delta == -1.0

    results, noise_baseline = run_lambda_sweep(
        delta=args.delta,
        l2_grid=args.l2_grid,
        n_runs=args.n_runs,
        num_iters=args.num_iters,
        layers=args.layers,
        learning_rate=args.learning_rate,
        mode="RS",
        verbose=args.verbose,
        include_noise_baseline=run_noise_only,
        run_noise_only=run_noise_only,
        test_frac=args.test_frac,
    )

    payload = {
        "delta": float(args.delta),
        "is_noise_only": run_noise_only,
        "meta": {
            "n_runs": args.n_runs,
            "num_iters": args.num_iters,
            "layers": args.layers,
            "learning_rate": args.learning_rate,
            "l2_grid": [float(x) for x in args.l2_grid],
            "test_frac": args.test_frac,
        },
        "results": results,
        "noise_baseline": noise_baseline,
    }
    save_pickle(args.out, payload)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

