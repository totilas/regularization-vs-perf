from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import whynot.gym as gym

import scripts.utils_torch as utils_torch


def forward(features, model):
    features_t = torch.from_numpy(features)
    with torch.no_grad():
        y_hat = model(features_t)
    return y_hat.numpy()


def accuracy(preds, epsilon, labels):
    return ((preds > epsilon / 2.0) * epsilon == labels).mean()

def make_env():
    return gym.make("Credit-v0")


def split_train_test(features, labels, test_frac=0.9):
    n = features.shape[0]
    n_test = int(round(test_frac * n))
    n_test = max(1, min(n - 1, n_test))
    perm = np.random.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return (
        features[train_idx],
        labels[train_idx],
        features[test_idx],
        labels[test_idx],
    )


def agent_shift(features, env, mode, model=None):
    if mode != "RS":
        raise ValueError(f"Unsupported mode: {mode}")
    preds = forward(features, model)
    n = features.shape[0]
    r = np.random.uniform(0, 1, (n,))
    resample_indices = np.random.randint(0, n, (n,))
    new_indices = np.where(r < (model.epsilon - preds), resample_indices, np.arange(n))

    strategic_features = features[:, env.config.changeable_features]
    new_strategic_features = strategic_features[new_indices]
    new_features = np.copy(features)
    new_features[:, env.config.changeable_features] = new_strategic_features
    return new_features

def tensor_repeated_risk_minimization(
    epsilon,
    learning_rate,
    num_iters,
    l2_penalty,
    mode,
    layers,
    do_shift=True,
    verbose=False,
    test_frac=0.2,
):
    env = make_env()
    env.config.epsilon = epsilon
    env.config.l2_penalty = l2_penalty
    env.config.mode = mode

    loss_start, loss_end, acc_start, acc_end, theta_gaps, f_theta_gaps = [], [], [], [], [], []
    theta_history = []

    if layers == 1:
        model = utils_torch.onelayer_NN(epsilon, mode)
    else:
        model = utils_torch.twolayers_NN(epsilon, mode)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    base_dataset = env.initial_state.values()
    base_features, base_labels = base_dataset["features"], base_dataset["labels"]
    base_labels = base_labels.astype("float64")
    base_labels[base_labels == 1] = epsilon
    train_features, train_labels, test_features, test_labels = split_train_test(
        base_features, base_labels, test_frac=test_frac
    )

    train_features_t = torch.from_numpy(train_features)
    train_labels_t = torch.from_numpy(train_labels)

    eval_bool = False
    utils_torch.fit_logistic_regression(
        train_features_t, train_labels_t, l2_penalty, model, optimizer, eval_bool, mode=mode
    )

    preds = forward(test_features, model)
    base_acc = accuracy(preds, epsilon, test_labels)
    if verbose:
        print(f"BASE TEST ACCURACY:{base_acc}")

    current_train_features = np.copy(train_features)
    current_test_features = np.copy(test_features)

    for step in range(num_iters):
        if do_shift:
            train_features_strat = agent_shift(current_train_features, env, mode, model)
            test_features_strat = agent_shift(current_test_features, env, mode, model)
        else:
            train_features_strat = np.copy(current_train_features)
            test_features_strat = np.copy(current_test_features)

        train_features_strat_t = torch.from_numpy(train_features_strat)
        train_labels_t = torch.from_numpy(train_labels)

        test_features_strat_t = torch.from_numpy(test_features_strat)
        test_labels_t = torch.from_numpy(test_labels)

        eval_bool = True
        loss_start.append(
            utils_torch.evaluate_logistic_loss(
                test_features_strat_t, test_labels_t, l2_penalty, model, eval_bool, mode=mode
            ).item()
        )
        eval_bool = False

        preds = forward(test_features_strat, model)

        theta_list = []
        with torch.no_grad():
            for weight in model.parameters():
                theta_list.append(weight.detach().clone().numpy())

        start_acc = accuracy(preds, epsilon, test_labels)
        acc_start.append(start_acc)
        if verbose and step % 5 == 0:
            print(f"iteration:{step}, start_test_accuracy:{start_acc}")

        utils_torch.fit_logistic_regression(
            train_features_strat_t, train_labels_t, l2_penalty, model, optimizer, eval_bool, mode=mode
        )

        eval_bool = True
        loss_end.append(
            utils_torch.evaluate_logistic_loss(
                test_features_strat_t, test_labels_t, l2_penalty, model, eval_bool, mode=mode
            ).item()
        )
        eval_bool = False

        predss = forward(test_features_strat, model)
        end_acc = accuracy(predss, epsilon, test_labels)
        if verbose and step % 5 == 0:
            print(f"iteration:{step}, end_test_accuracy:{end_acc}")
        acc_end.append(end_acc)

        theta_new_list = []
        with torch.no_grad():
            for weight in model.parameters():
                theta_new_list.append(weight.detach().clone().numpy())

        theta_gap = 0.0
        for i in range(len(theta_list)):
            theta_gap += np.linalg.norm(theta_list[i] - theta_new_list[i])
        theta_gaps.append(theta_gap)

        f_theta_gap = np.sqrt(np.mean((preds - predss) ** 2))
        f_theta_gaps.append(f_theta_gap)

        theta_history.append(theta_new_list)

        current_train_features = train_features_strat
        current_test_features = test_features_strat

    return {
        "loss_start": np.array(loss_start),
        "loss_end": np.array(loss_end),
        "acc_start": np.array(acc_start),
        "acc_end": np.array(acc_end),
        "theta_gaps": np.array(theta_gaps),
        "f_theta_gaps": np.array(f_theta_gaps),
        "theta_history": np.array(theta_history, dtype=object),
    }

def run_config_grid(
    configs,
    n_runs,
    num_iters,
    layers,
    learning_rate,
    l2_penalty,
    mode="RS",
    verbose=False,
    test_frac=0.9,
):
    results = {}
    for label, delta, do_shift in configs:
        runs = []
        for seed in range(n_runs):
            np.random.seed(seed)
            torch.manual_seed(seed)
            runs.append(
                tensor_repeated_risk_minimization(
                    delta,
                    learning_rate,
                    num_iters,
                    l2_penalty,
                    mode,
                    layers,
                    do_shift=do_shift,
                    verbose=verbose,
                    test_frac=test_frac,
                )
            )
        stacked = {}
        for key in runs[0]:
            stacked[key] = np.array([run[key] for run in runs], dtype=object if key == "theta_history" else None)
        results[label] = stacked
    return results


def run_lambda_sweep(
    delta,
    l2_grid,
    n_runs,
    num_iters,
    layers,
    learning_rate,
    mode="RS",
    verbose=False,
    include_noise_baseline=False,
    run_noise_only=False,
    test_frac=0.9,
):
    out = {}
    if not run_noise_only:
        for l2_penalty in l2_grid:
            print('Current lambda', l2_penalty)
            runs = []
            for seed in range(n_runs):
                print('Current run', seed, '/', n_runs)
                np.random.seed(seed)
                torch.manual_seed(seed)
                runs.append(
                    tensor_repeated_risk_minimization(
                        float(delta),
                        learning_rate,
                        num_iters,
                        float(l2_penalty),
                        mode,
                        layers,
                        do_shift=True,
                        verbose=verbose,
                        test_frac=test_frac,
                    )
                )
            stacked = {}
            for key in runs[0]:
                stacked[key] = np.array([run[key] for run in runs], dtype=object if key == "theta_history" else None)
            out[float(l2_penalty)] = stacked

    noise = None
    if include_noise_baseline:
        noise = {}
        for l2_penalty in l2_grid:
            runs = []
            for seed in range(n_runs):
                np.random.seed(seed)
                torch.manual_seed(seed)
                runs.append(
                    tensor_repeated_risk_minimization(
                        0.0,
                        learning_rate,
                        num_iters,
                        float(l2_penalty),
                        mode,
                        layers,
                        do_shift=False,
                        verbose=verbose,
                        test_frac=test_frac,
                    )
                )
            stacked = {}
            for key in runs[0]:
                stacked[key] = np.array([run[key] for run in runs], dtype=object if key == "theta_history" else None)
            noise[float(l2_penalty)] = stacked
    return out, noise


def save_pickle(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_pickle(path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def summarize_final_iteration(results):
    summary = {}
    for label, out in results.items():
        summary[label] = {
            "loss_end_final_mean": out["loss_end"][:, -1].mean(),
            "loss_end_final_std": out["loss_end"][:, -1].std(),
            "acc_end_final_mean": out["acc_end"][:, -1].mean(),
            "acc_end_final_std": out["acc_end"][:, -1].std(),
        }
    return summary


def plot_last_experiment(results, num_iters, save_path=None):
    x = np.arange(num_iters)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    for label, out in results.items():
        m, s = out["loss_end"].mean(0), out["loss_end"].std(0)
        axes[0, 0].plot(x, m, "*-", label=label)
        axes[0, 0].fill_between(x, m - s, m + s, alpha=0.15)

        m, s = out["acc_end"].mean(0), out["acc_end"].std(0)
        axes[0, 1].plot(x, 100 * m, "*-", label=label)
        axes[0, 1].fill_between(x, 100 * (m - s), 100 * (m + s), alpha=0.15)

        m, s = out["theta_gaps"].mean(0), out["theta_gaps"].std(0)
        axes[1, 0].plot(x, m, "*-", label=label)
        axes[1, 0].fill_between(x, m - s, m + s, alpha=0.15)

        m, s = out["f_theta_gaps"].mean(0), out["f_theta_gaps"].std(0)
        axes[1, 1].plot(x, m, "*-", label=label)
        axes[1, 1].fill_between(x, m - s, m + s, alpha=0.15)

    axes[0, 0].set_title("Log Risk", fontsize=26)
    axes[0, 1].set_title("Accuracy", fontsize=26)
    axes[1, 0].set_title(r"$\|\theta_{t+1}-\theta_t\|$", fontsize=26)
    axes[1, 1].set_title(r"$\|f_{\theta_{t+1}}-f_{\theta_t}\|$", fontsize=26)

    for ax in axes.ravel():
        ax.set_xlabel("Iteration", fontsize=22)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(axis="both", labelsize=14)

    axes[0, 1].legend(fontsize=13)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


def _metric_label(metric):
    if metric == "loss_end":
        return "Loss"
    if metric == "log_loss_end":
        return "Log Loss"
    if metric == "acc_end":
        return "Accuracy"
    raise ValueError(f"Unknown metric: {metric}")


def _final_metric_values(out, metric):
    vals = out[:, -1]
    if metric == "loss_end":
        return vals
    if metric == "log_loss_end":
        return np.log(vals)
    if metric == "acc_end":
        return 100 * vals
    raise ValueError(f"Unknown metric: {metric}")

"""
def plot_final_vs_lambda(
    sweep_payloads,
    metric,
    save_path=None,
    show_min_star=True,
    log_x=True,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    payloads = sorted(sweep_payloads, key=lambda p: p["delta"])

    for payload in payloads:
        lams = np.array(sorted(payload["results"].keys()), dtype=float)
        means, stds = [], []
        for lam in lams:
            vals = _final_metric_values(payload["results"][lam][metric], metric)
            means.append(vals.mean())
            stds.append(vals.std())
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)

        ax.errorbar(
            lams,
            means,
            yerr=stds,
            fmt="*-",
            capsize=3,
            label=fr"$\delta={payload['delta']:g}$",
        )

        if show_min_star and means.size:
            idx = int(np.nanargmin(means)) if metric != "acc_end" else int(np.nanargmax(means))
            ax.scatter([lams[idx]], [means[idx]], marker="+", s=80, linewidths=2, zorder=5)

    if payloads and "noise_baseline" in payloads[0] and payloads[0]["noise_baseline"] is not None:
        lams = np.array(sorted(payloads[0]["noise_baseline"].keys()), dtype=float)
        means, stds = [], []
        for lam in lams:
            vals = _final_metric_values(payloads[0]["noise_baseline"][lam][metric], metric)
            means.append(vals.mean())
            stds.append(vals.std())
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)

        ax.errorbar(
            lams,
            means,
            yerr=stds,
            fmt="k--",
            capsize=3,
            linewidth=2,
            label="noise only",
        )

        if show_min_star and means.size:
            idx = int(np.nanargmin(means)) if metric != "acc_end" else int(np.nanargmax(means))
            ax.scatter([lams[idx]], [means[idx]], marker="+", s=80, linewidths=2, zorder=5, c="k")

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel(r"$\lambda$", fontsize=16)
    ax.set_ylabel(_metric_label(metric), fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax

"""
def plot_final_vs_lambda(
    sweep_payloads,
    metric,
    save_path=None,
    show_min_star=True,
    log_x=True,
):
    import seaborn as sns

    sns.set_theme(style="darkgrid", context="paper")

    payloads = sorted(sweep_payloads, key=lambda p: p["delta"])
    curve_payloads = [p for p in payloads if not p.get("is_noise_only", False)]

    n_curves = max(len(curve_payloads), 1)
    purple_palette = sns.color_palette("rocket_r", n_colors=n_curves + 2)[2:]

    fig, ax = plt.subplots(figsize=(8, 5))

    for j, payload in enumerate(curve_payloads):
        lams = np.array(sorted(payload["results"].keys()), dtype=float)
        means, stds = [], []
        for lam in lams:
            vals = _final_metric_values(payload["results"][lam][metric], metric)
            means.append(vals.mean())
            stds.append(vals.std())
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)

        ax.errorbar(
            lams,
            means,
            yerr=stds,
            fmt="*-",
            capsize=3,
            color=purple_palette[j],
            linewidth=2,
            label=fr"$\delta={payload['delta']:g}$",
        )

        if show_min_star and means.size:
            idx = int(np.nanargmin(means)) if metric != "acc_end" else int(np.nanargmax(means))
            ax.scatter(
                [lams[idx]],
                [means[idx]],
                marker="+",
                s=80,
                linewidths=2,
                zorder=5,
                c="r",
            )

    noise_payload = next((p for p in payloads if p.get("noise_baseline") is not None), None)
    if noise_payload is not None:
        lams = np.array(sorted(noise_payload["noise_baseline"].keys()), dtype=float)
        means, stds = [], []
        for lam in lams:
            vals = _final_metric_values(noise_payload["noise_baseline"][lam][metric], metric)
            means.append(vals.mean())
            stds.append(vals.std())
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)

        ax.errorbar(
            lams,
            means,
            yerr=stds,
            fmt="k--",
            capsize=3,
            linewidth=2,
            label="noise only",
        )

        if show_min_star and means.size:
            idx = int(np.nanargmin(means)) if metric != "acc_end" else int(np.nanargmax(means))
            ax.scatter(
                [lams[idx]],
                [means[idx]],
                marker="+",
                s=80,
                linewidths=2,
                zorder=5,
                c="r",
            )

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(_metric_label(metric))

    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    ax.tick_params(axis="both", which="major", labelsize=14)

    ax.legend(loc="best")

    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontsize(12)
        leg.set_title(leg.get_title().get_text(), prop={"size": 13})

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax