from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

sns.set_theme(style="darkgrid", context="paper")

def _apply_scientific(ax: plt.Axes) -> None:
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 3))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

def plot_risk_vs_lambda_multi(
    lam_grid: np.ndarray,
    curves,
    b_values,
    out_dir,
    dataset,
    y_min,
    y_max
) -> Path:
    out_dir_plot = Path(out_dir) / "risk_vs_lambda_all_b" / dataset
    out_dir_plot.mkdir(parents=True, exist_ok=True)
    risk_concat = np.concatenate([np.asarray(c).ravel() for c in curves])
    df = pd.DataFrame({
        "λ": np.tile(np.asarray(lam_grid).ravel(), len(b_values)),
        "risk": risk_concat,
        "b": np.repeat(np.asarray(b_values, dtype=float).ravel(), len(lam_grid)),
    })
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.lineplot(data=df, x="λ", y="risk", hue="b", linewidth=2.0, ax=ax, legend=True)
    lam_grid_arr = np.asarray(lam_grid).ravel()
    mins = []
    for curve, b in zip(curves, np.asarray(b_values, dtype=float).ravel()):
        c = np.asarray(curve).ravel()
        if c.size == 0:
            continue
        idx = int(np.nanargmin(c))
        mins.append((lam_grid_arr[idx], c[idx], float(b)))
    if mins:
        mins_df = pd.DataFrame(mins, columns=["λ", "risk", "b"]).sort_values("b")
        # "+" markers at each minimum
        ax.scatter(mins_df["λ"], mins_df["risk"], marker="+", s=60, linewidths=2, zorder=5, c='r',label=r"$\lambda^*$")
        # Line connecting the minima (ordered by b)
        ax.plot(mins_df["λ"], mins_df["risk"], linestyle="--", linewidth=2, zorder=4, c="r")
    #ax.set_title(f"Risk vs λ (all b) | {dataset}")
    ax.legend()
    ax.set_xlabel("λ")
    ax.set_ylabel("Risk")
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontsize(12)
        leg.set_title(leg.get_title().get_text(), prop={"size": 13})
    if False:
        ax.set_ylim(.38, .55)
        ax.set_ylim(.695, .725)
        ax.set_ylim(0.78, .9)
    ax.set_ylim(y_min, y_max)
    _apply_scientific(ax)
    fig.tight_layout()
    pdf_path = out_dir_plot / "risk_vs_lambda_all_b.pdf"
    png_path = out_dir_plot / "risk_vs_lambda_all_b.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path
