"""Plot Sobol sensitivity indices from a GSA CSV as horizontal bar charts.

Reads the long-form CSV written by ``python -m peloton.gsa --method sobol``
(columns: metric, param, S1, S1_conf, ST, ST_conf) and draws, for every metric,
a horizontal grouped bar chart of the first-order (S1) and total-order (ST)
indices per parameter, with the SALib confidence intervals as error bars.

Usage:
    python scripts/plot_gsa_sobol.py [--in data/gsa_sobol.csv] [--out plots/gsa_sobol.png]
"""

import argparse
import os

try:
    import matplotlib

    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))  # headless by default
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pandas, matplotlib and numpy are required. Install with "
        "'pip install pandas matplotlib numpy'."
    ) from e


def plot_sobol(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    required = {"metric", "param", "S1", "ST"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing Sobol columns: {sorted(missing)}")

    metrics = list(dict.fromkeys(df["metric"]))   # preserve file order
    params = list(dict.fromkeys(df["param"]))
    y = np.arange(len(params))
    h = 0.38                                       # half-height of each grouped bar

    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows),
                             squeeze=False, sharey=True)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["metric"] == metric].set_index("param").reindex(params)
        s1, st = sub["S1"].to_numpy(), sub["ST"].to_numpy()
        s1c = sub.get("S1_conf", pd.Series(0, index=sub.index)).to_numpy()
        stc = sub.get("ST_conf", pd.Series(0, index=sub.index)).to_numpy()

        # Horizontal bars: parameters on the y-axis, index value along the x-axis.
        ax.barh(y + h / 2, s1, height=h, xerr=s1c, label="S1 (first-order)",
                color="#4C78A8", ecolor="gray", capsize=2, error_kw={"lw": 0.8})
        ax.barh(y - h / 2, st, height=h, xerr=stc, label="ST (total-order)",
                color="#F58518", ecolor="gray", capsize=2, error_kw={"lw": 0.8})

        # Annotate each bar with its value, written horizontally at the bar tip.
        for yi, v in zip(y + h / 2, s1):
            ax.text(v, yi, f" {v:.2f}", va="center", ha="left", fontsize=7)
        for yi, v in zip(y - h / 2, st):
            ax.text(v, yi, f" {v:.2f}", va="center", ha="left", fontsize=7)

        ax.set_yticks(y)
        ax.set_yticklabels(params)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(metric, fontsize=10)
        ax.grid(axis="x", ls=":", alpha=0.5)

    # Blank any unused panes in the grid.
    for k in range(len(metrics), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle(f"Sobol sensitivity indices — {os.path.basename(csv_path)}",
                 y=0.995, fontsize=12)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=2, frameon=False)
    fig.supxlabel("sensitivity index (error bars = SALib 95% CI)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}  ({len(metrics)} metrics x {len(params)} params)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", default="data/gsa_sobol.csv")
    p.add_argument("--out", dest="out_path", default="plots/gsa_sobol.png")
    args = p.parse_args()
    plot_sobol(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
