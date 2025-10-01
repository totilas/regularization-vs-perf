# ridge_plot_p_from_results.py
import os, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns



from typing import List, Optional

import typer
from typing_extensions import Annotated
from typer_config import use_yaml_config

app = typer.Typer()


@app.command()
@use_yaml_config()
def main( output: str,
         to_plot: Annotated[Optional[List[str]], typer.Option()] = None,
         legends: Annotated[Optional[List[str]], typer.Option()] = None):

    sns.set_theme(style="darkgrid", context="paper")
    palette = sns.color_palette("husl", 8)



    fig, ax = plt.subplots()


    for i,curve in enumerate(to_plot):
        # ---- Performative: results_perfo/ ----
        outdir2 = "results_perfo"
        files2 = sorted(glob.glob(os.path.join(outdir2, f"run_{curve}_*.npz")))
        if len(files2) >= 1:
            risks2 = []
            lam_ref2 = None
            for f in files2:
                d = np.load(f)
                if lam_ref2 is None:
                    lam_ref2 = d["lambdas"]
                else:
                    np.testing.assert_allclose(lam_ref2, d["lambdas"], rtol=0, atol=1e-12)
                risks2.append(d["risks"])

            R2 = np.vstack(risks2)
            m2, s2 = R2.mean(axis=0), R2.std(axis=0, ddof=1)

            ax.plot(lam_ref2, m2, marker="s",  color=palette[i], label=legends[i])
            ax.fill_between(lam_ref2, m2 - s2, m2 + s2, color=palette[i],alpha=0.2)
        else:
            print("No performative result files found in 'results_perfo' (skipping).")



    # ---- Final touches ----
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Risk")
    ax.legend()
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


    fig.tight_layout()
    fig.savefig(f"risk_{output}.pdf")
    print(f"Saved risk_{output}.pdf")

if __name__ == "__main__":
    app()