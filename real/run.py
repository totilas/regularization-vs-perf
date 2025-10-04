import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
import typer
from typing_extensions import Annotated
from typer_config import use_yaml_config
import os
from housing import load_housing
from lsac import load_lsac
from typing import Dict, Any, List
from tqdm.auto import tqdm
from plot import plot_risk_vs_lambda_multi
from pathlib import Path
import json



def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
        n, _ = X.shape
        model = Ridge(alpha=n * lam, fit_intercept=False)
        model.fit(X, y)
        return model.coef_


def k_splits(n: int, splits: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    return [np.array(f, dtype=int) for f in np.array_split(idx, splits)]

def synth_labels_y_plus_perf(X: np.ndarray, theta_prev: np.ndarray, b_vec: np.ndarray, perf_mask: np.ndarray, y_base: np.ndarray) -> np.ndarray:
    x1 = X[:, perf_mask]
    t1 = theta_prev[perf_mask]
    return y_base + np.sum(x1 * (b_vec[perf_mask] * t1), axis=1)




app = typer.Typer()


@app.command()
@use_yaml_config()
def main(seed: int,
        out_dir: str,
        dataset: str,
        lambda_min: float,
        lambda_max: float,
        num_lambda: int,
        n_splits: int,
        train_size: int,
        steps: int,
        y_min:float,
        y_max:float):
     
    outdir = "results_perfo"
    os.makedirs(outdir, exist_ok=True)

    assert steps <= n_splits - 1

    lam_grid = np.linspace(lambda_min, lambda_max, num_lambda)


    if dataset == "housing":
        X_df, y_series, feature_names = load_housing()
        perf_names = ["MedInc", "AveBedrms", "AveOccup"]
    elif dataset == "lsac":
        X_df, y_series, feature_names = load_lsac()
        perf_names = ['Unnamed: 0', 'decile1b', 'decile3', 'other', 'asian', 'black', 'hisp', 'pass_bar', 'tier']
    else:
        raise ValueError("unknown dataset")
    
    n, d = X_df.shape
    folds = k_splits(n, n_splits, seed)

    base = max(1, n // n_splits)
    chunk_cap = max(1, min(base, int(train_size)))

    name_to_idx = {nm: i for i, nm in enumerate(X_df.columns)}
    perf_idx = np.array([name_to_idx[nm] for nm in perf_names if nm in name_to_idx], dtype=int)
    perf_mask = np.zeros(d, dtype=bool)
    if perf_idx.size > 0:
        perf_mask[perf_idx] = True

    X_all = X_df.values.astype(float)
    y_all = y_series.values.astype(float)

    part_stats = []
    noise_vars = []
    b_values = [0.0, 0.05, 0.1, 0.15, 0.2]

    for t in range(1, 1 + steps):
        idx = folds[t][:chunk_cap]
        n_t = int(idx.size)
        kappa = float(d) / max(1, n_t)
        part_stats.append(dict(step=t, n_train=n_t, d=d, kappa=kappa))
        try:
            ols_t = LinearRegression().fit(X_all[idx], y_all[idx])
            resid_t = y_all[idx] - ols_t.predict(X_all[idx])
            df_t = max(1, n_t - d - 1)
            var_t = float(np.dot(resid_t, resid_t) / df_t)
        except Exception:
            var_t = float(np.nan)
        noise_vars.append(var_t)

    mean_noise_var = float(np.nanmean(noise_vars))

    results: Dict[str, Any] = dict(
            dataset=dataset,
            feature_names=feature_names,
            performative_cols=perf_names,
            b_values=b_values,
            lambda_grid=lam_grid.tolist(),
            steps=steps,
            n_splits=n_splits,
            train_size=train_size,
            effective_train_chunk_size=chunk_cap,
            training_part_stats=part_stats,
            mean_training_noise_variance=mean_noise_var,
            folds=[f.tolist() for f in folds],
            curves=[],
    )

    all_risk_curves: List[List[float]] = []
    all_lambda_stars: List[float] = []
    all_optimal_risks: List[float] = []
    all_b_vals: List[float] = []

    for b in b_values:
        b_vec = np.zeros(d)
        if perf_mask.any():
            b_vec[perf_mask] = b
        test_risk_curve: List[float] = []

        for lam in tqdm(lam_grid, desc=f"b={b}"):
            theta_prev = np.zeros(d)

            for t in range(1, 1 + steps):
                idx = folds[t][:chunk_cap]
                X_step = X_all[idx, :]
                y_base = y_all[idx]
                y_step = synth_labels_y_plus_perf(X_step, theta_prev, b_vec, perf_mask, y_base)
                theta_prev = ridge_fit(X_step, y_step, lam)

            test_idx = folds[-1]
            X_test = X_all[test_idx, :]
            y_test = y_all[test_idx]
            y_hat = X_test @ theta_prev
            mse = float(((y_test - y_hat) ** 2).mean())
            if lam == 0:
                print(mse)
            denom = float((y_test ** 2).mean())
            if lam == 0:
                print(denom, "de", theta_prev)
            risk = float(mse / max(denom, 1e-12))
            test_risk_curve.append(risk)

        best_idx = int(np.argmin(test_risk_curve))
        lambda_star = float(lam_grid[best_idx])
        optimal_risk = float(test_risk_curve[best_idx])

        print(f"[{dataset} | b={b}] lambda* = {lambda_star}, optimal risk = {optimal_risk}")
        for s in part_stats:
            print(f"[{dataset} | step={s['step']}] n_train={s['n_train']} d={s['d']} kappa={s['kappa']}")
        print(f"[{dataset}] mean training noise variance = {mean_noise_var}")

        results["curves"].append(
            dict(
                dataset=dataset,
                b=float(b),
                risk=test_risk_curve,
                lambda_star=lambda_star,
                optimal_risk=optimal_risk,
            )
        )

        all_risk_curves.append(list(test_risk_curve))
        all_lambda_stars.append(lambda_star)
        all_optimal_risks.append(optimal_risk)
        all_b_vals.append(float(b))

    try:
        plot_risk_vs_lambda_multi(
            lam_grid=lam_grid,
            curves=all_risk_curves,
            b_values=all_b_vals,
            out_dir=out_dir,
            dataset=dataset,
            y_min=y_min,
            y_max=y_max
        )
    except Exception as e:
            print(e)

    
    out_json = Path(out_dir) / f"results_{dataset}.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    return results


if __name__ == "__main__":
    app()