"""Plot evolution of learned coefficients from a learning CSV.

Reads the CSV written by ``python main.py learn --out learning.csv`` and plots
selected parameter means vs generation, with std shaded regions when available.

Usage:
    python scripts\plot_learning.py --in learning.csv --out plots/learning.png

By default plots every "*_mean" column found in the CSV.
"""
import argparse
import os

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except Exception as e:
    raise RuntimeError("pandas, matplotlib and numpy are required. Install with 'pip install pandas matplotlib numpy'.")


def plot_learning(csv_path: str, out_path: str, params: list | None = None):
    df = pd.read_csv(csv_path)
    if "generation" not in df.columns:
        raise ValueError("Input CSV must contain a 'generation' column.")

    # Select parameter mean columns if none specified
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    if params:
        cols = [p for p in params if p in df.columns]
    else:
        cols = mean_cols

    if not cols:
        raise ValueError("No parameter mean columns found to plot.")

    n = len(cols)
    # Choose layout: up to 4 cols wide
    cols_per_row = min(4, n)
    rows = (n + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(4 * cols_per_row, 3 * rows), squeeze=False)

    generations = df['generation'].values

    for i, cname in enumerate(cols):
        r = i // cols_per_row
        c = i % cols_per_row
        ax = axes[r][c]
        mean = df[cname].values
        std_col = cname.replace("_mean", "_std")
        if std_col in df.columns:
            std = df[std_col].values
            ax.fill_between(generations, mean - std, mean + std, alpha=0.2)
        ax.plot(generations, mean, marker='o', ms=3)
        ax.set_title(cname)
        ax.set_xlabel('Generation')
        ax.grid(alpha=0.3)

    # Turn off empty subplots
    for j in range(n, rows * cols_per_row):
        r = j // cols_per_row
        c = j % cols_per_row
        axes[r][c].axis('off')

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f'Wrote learning plot to {out_path}')


def main():
    p = argparse.ArgumentParser(description='Plot learning CSV')
    p.add_argument('--in', dest='infile', default='learning.csv')
    p.add_argument('--out', dest='outfile', default='plots/learning.png')
    p.add_argument('--params', dest='params', default=None,
                   help='Comma-separated list of parameter columns to plot (exact column names).')
    args = p.parse_args()

    params = args.params.split(',') if args.params else None
    plot_learning(args.infile, args.outfile, params)


if __name__ == '__main__':
    main()
