#!/usr/bin/env python3
"""Plot a Sobol GSA CSV as horizontal error-bar charts with value labels.

One subplot per metric; within each, every parameter gets a first-order (S1) and
total-order (ST) point, drawn as a horizontal error bar (±_conf) with the index
value annotated at the bar tip.

    uv run python scripts/plot_gsa_sobol.py data/gsa_sobol_from_dumps.csv \
        --out data/gsa_sobol_from_dumps.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# (column, conf column, label, colour)
SERIES = [
    ("S1", "S1_conf", "S1 (first-order)", "#1f77b4"),
    ("ST", "ST_conf", "ST (total-order)", "#d62728"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="GSA Sobol CSV (metric, param, S1, S1_conf, ST, ST_conf)")
    p.add_argument("--out", default=None, help="output image (default: <csv>.png)")
    p.add_argument("--title", default=None, help="figure suptitle")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    metrics = list(dict.fromkeys(df["metric"]))      # preserve CSV order
    params = list(dict.fromkeys(df["param"]))
    y = range(len(params))
    offset = 0.18                                    # vertical split between S1 and ST

    ncol = 2
    nrow = -(-len(metrics) // ncol)                  # ceil
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.2 * nrow), squeeze=False)

    for ax, metric in zip(axes.flat, metrics):
        sub = df[df["metric"] == metric].set_index("param").reindex(params)
        for k, (col, conf, label, colour) in enumerate(SERIES):
            yk = [yi + (offset if k == 0 else -offset) for yi in y]
            vals, errs = sub[col].to_numpy(), sub[conf].to_numpy()
            ax.errorbar(vals, yk, xerr=errs, fmt="o", color=colour, label=label,
                        capsize=3, markersize=6, elinewidth=1.4, zorder=3)
            # value label just past the error-bar tip
            for v, e, yy in zip(vals, errs, yk):
                ax.annotate(f"{v:.3f}", (v + e, yy), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=8,
                            color=colour)

        ax.axvline(0, color="0.7", lw=0.8, zorder=1)
        ax.set_yticks(list(y))
        ax.set_yticklabels(params)
        ax.set_ylim(-0.6, len(params) - 0.4)
        ax.set_xlabel("Sobol index")
        ax.set_title(metric, fontweight="bold")
        ax.grid(axis="x", color="0.92", zorder=0)
        ax.margins(x=0.18)                           # headroom for labels

    # blank any unused panels
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(args.title or f"Sobol sensitivity indices — {Path(args.csv).name}",
                 y=1.04, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = Path(args.out) if args.out else Path(args.csv).with_suffix(".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
