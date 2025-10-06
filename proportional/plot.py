# ridge_plot_p_from_results.py
import os, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import seaborn as sns
from pathlib import Path


from typing import List, Optional

import typer
from typing_extensions import Annotated
from typer_config import use_toml_config

app = typer.Typer()


## Option 1:
# Input: folder where all the runs are stored (we can copy-paste that by modifying the name once to get only the n parameter in the name)
# For each curves (.npz with multiple runid), we plot the curve with errors bars in a different file pdf.


## Option 2:
# the input is the same
# we give certain parameters in input as well.

from collections import defaultdict


def format_label(s: str) -> str:
    s = s.replace("b_", "b=")\
         .replace("gamma_", r"\gamma=")\
         .replace("kappa_", r"\kappa=")\
         .replace("rho_", r"\rho=")\
         .replace("sigma_", r"\sigma=")\
         .replace("unbalanced_", "g=")\
    
    parts = s.split("_")
    return ",".join(parts)


@app.command()
def main(npz_directory: Path):

    sns.set_theme(style="darkgrid", context="paper")
    palette = sns.color_palette("husl", 8)


    c_0 = defaultdict(list)
    c_2 = defaultdict(list)
    for file in sorted(npz_directory.glob(f"run_*.npz")):
        file = np.load(file, allow_pickle=True)
        param = file["param"].item()
        del param["c"]
        if np.isclose(file["c"], 0):
            c_0["_".join(f"{k}_{v}" for k, v in sorted(param.items()))].append(file)
        elif np.isclose(file["c"], 0.2):
            c_2["_".join(f"{k}_{v}" for k, v in sorted(param.items()))].append(file)
        else:
            assert False

    # assert set(c_0.keys()) == set(c_2.keys())

    intersting_keys = []
    for k in c_0.keys():
        f0 = c_0[k]
        if k not in c_2:
            continue
        f2 = c_2[k]

        risks2 = np.mean(np.vstack([f["risks"] for f in f2]), axis=0)
        risks0 = np.mean(np.vstack([f["risks"] for f in f0]), axis=0)
        assert risks2.shape == (len(f0[0]["risks"]),)

        
        if np.min(risks0) < np.min(risks2):
            intersting_keys.append(k)
            print(k)

    # Perform the plots for the intersting keys

    fig, ax = plt.subplots()
    for i, k in enumerate(intersting_keys):
        f0 = c_0[k]
        f2 = c_2[k]

        lam = f0[0]["lambdas"]
        n = f0[0]["n"]

        R2 = np.vstack([f["risks"] for f in f2])
        R0 = np.vstack([f["risks"] for f in f0])

        m2 = R2.mean(axis=0)
        s2 = R2.std(axis=0, ddof=1)

        m0 = R0.mean(axis=0)
        s0 = R0.std(axis=0, ddof=1)

        ax.plot(lam, m2, marker="x",  color=palette[i%8], markevery=5, label=fr"${format_label(k)}$")
        ax.fill_between(lam, m2 - s2, m2 + s2, color=palette[i%8], alpha=0.2)

        ax.plot(lam, m0, marker="o",  color=palette[i%8], markevery=5)
        ax.fill_between(lam, m0 - s0, m0 + s0, color=palette[i%8], alpha=0.2)


    # ---- Final touches ----
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Risk")
    ax.set_ylim(0, 2)
    # ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    
    ax.tick_params(axis="both", which="major", labelsize=14)

    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontsize(12)
        leg.set_title(leg.get_title().get_text(), prop={"size": 13})

    #fig.tight_layout(rect=[0, 0, 1.85, 1])
    fig.savefig(f"risk_{n}.pdf")
    print(f"Saved risk_{n}.pdf")

if __name__ == "__main__":
    app()