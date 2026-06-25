"""Plot Sobol second-order (pairwise S2) indices from a GSA CSV as heatmaps.

Reads the long-form CSV written by ``python -m peloton.gsa --method sobol`` when
second-order is on (columns: metric, param_1, param_2, S2, S2_conf) and draws, for
every metric, a symmetric parameter x parameter heatmap of the pairwise
interaction indices S2[i,j]. A large off-diagonal cell means those two knobs
*interact* (their joint effect exceeds the sum of their individual effects) -- the
detail behind any ST > S1 gap in the first/total-order plot. The diagonal is
undefined (a parameter has no second-order index with itself) and left blank.

Usage:
    python scripts/plot_gsa_sobol_s2.py [--in data/gsa_sobol_S2.csv] [--out plots/gsa_sobol_s2.png]
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


def plot_sobol_s2(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    required = {"metric", "param_1", "param_2", "S2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing second-order columns: {sorted(missing)}")

    metrics = list(dict.fromkeys(df["metric"]))                  # file order
    params = list(dict.fromkeys(df["param_1"].tolist() + df["param_2"].tolist()))
    idx = {p: i for i, p in enumerate(params)}
    D = len(params)

    # Shared colour scale (vmin=0) so interaction strength is comparable across metrics.
    gmax = float(np.nanmax(df["S2"].to_numpy()))
    gmax = gmax if gmax > 0 else 1.0

    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.6 * nrows), squeeze=False)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#dddddd")                                      # masked diagonal -> grey
    im = None

    for k, metric in enumerate(metrics):
        ax = axes[k // ncols][k % ncols]
        M = np.full((D, D), np.nan)
        for _, r in df[df["metric"] == metric].iterrows():
            i, j = idx[r["param_1"]], idx[r["param_2"]]
            M[i, j] = M[j, i] = r["S2"]                          # S2 is symmetric
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=gmax)

        # Annotate each cell with its value; pick text colour for contrast.
        for i in range(D):
            for j in range(D):
                if np.isnan(M[i, j]):
                    continue
                shade = "white" if M[i, j] > 0.6 * gmax else "black"
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color=shade)

        ax.set_xticks(range(D)); ax.set_yticks(range(D))
        ax.set_xticklabels(params, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(params, fontsize=7)
        ax.set_title(metric, fontsize=10)

    for k in range(len(metrics), nrows * ncols):                # blank unused panes
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle(f"Sobol second-order indices (S2) — {os.path.basename(csv_path)}",
                 y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.9, 0.97))
    if im is not None:
        # Dedicated colourbar axis on the far right so it never overlaps a subplot.
        cax = fig.add_axes((0.92, 0.30, 0.015, 0.40))
        fig.colorbar(im, cax=cax, label="S2 (pairwise interaction)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}  ({len(metrics)} metrics, {D}x{D} pairs)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", default="data/gsa_sobol_S2.csv")
    p.add_argument("--out", dest="out_path", default="plots/gsa_sobol_s2.png")
    args = p.parse_args()
    plot_sobol_s2(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
