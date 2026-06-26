"""Create heatmaps from Sobol GSA per-run agent dumps.

Reads a Sobol *agent-dump* tree produced by ``peloton.gsa_dump`` (the
scripts/job-cpu-rome-gsa-sobol-dump.sh single-node job and its parallel
...-dump-array.sh sibling): a ``sobol/`` directory holding ``X.npy`` (the design
matrix; the array job also drops a ``sample_index.csv``) and one ``sample_<i>/``
folder per design row, each containing a ``gen_<g>/`` bundle per evolution
generation. For each sample only the *final* generation is used -- the end-state
the field evolved to. It filters the design matrix to the samples nearest a
target recovery rate and utility decay and plots metrics over
``breakaway_speed_frac`` x ``k_s``.

Examples:
    python scripts/gsa_sobol_heatmaps.py \
        --sobol-dir /path/to/123456-abcdef-sobol-dump \
        --out-dir /path/to/heatmaps

    python scripts/gsa_sobol_heatmaps.py \
        --sobol-dir /path/to/123456-abcdef-sobol-dump/sobol \
        --bins 12 --slice-nearest 200
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from peloton.gsa import PROBLEM_SOBOL


# Dump layout: sobol/sample_<i>/gen_<g>/<tables>. One folder per design row, one
# bundle per evolution generation inside it.
SAMPLE_RE = re.compile(r"^sample_(?P<sample>\d+)$")
GEN_RE = re.compile(r"^gen_(?P<gen>\d+)$")

# Column order of X.npy == the Sobol problem's parameter order (gsa_dump samples
# PROBLEM_SOBOL); used to rebuild the index when only X.npy is present.
SOBOL_PARAM_NAMES = PROBLEM_SOBOL["names"]


METRICS = {
    "mean_finish_time": "Mean finishing time (s)",
    "mean_stamina": "Mean stamina fraction",
    "group_time_frac": "Fraction of rider-time in group",
    "coop_alpha": "Mean coop alpha",
    "leave_alpha": "Mean leave alpha",
    "follow_alpha": "Mean follow alpha",
}


def _has_design(d: Path) -> bool:
    """A dump dir is identifiable by its design matrix (X.npy) or sample index."""
    return (d / "X.npy").exists() or (d / "sample_index.csv").exists()


def resolve_sobol_dir(path: Path) -> Path:
    """Accept either the sobol/ dump directory itself or a parent containing it."""
    if _has_design(path):
        return path
    child = path / "sobol"
    if _has_design(child):
        return child
    raise FileNotFoundError(
        f"Could not find X.npy or sample_index.csv in {path} or {child}. "
        "Pass the dump directory (or its parent) that holds the sample_<i>/ folders."
    )


def load_index(sobol_dir: Path) -> pd.DataFrame:
    """Load the design matrix as a DataFrame with a ``sample_idx`` column.

    Prefers ``sample_index.csv`` (written by the array job) and otherwise rebuilds
    it from ``X.npy`` (single-node job), labelling the columns with the Sobol
    parameter names. Row order is the design-row / ``sample_<i>`` index either way.
    """
    idx_csv = sobol_dir / "sample_index.csv"
    if idx_csv.exists():
        index = pd.read_csv(idx_csv)
        if "sample_idx" not in index.columns:
            index = index.reset_index().rename(columns={"index": "sample_idx"})
        return index
    x_npy = sobol_dir / "X.npy"
    if x_npy.exists():
        X = np.load(x_npy)
        index = pd.DataFrame(X, columns=SOBOL_PARAM_NAMES)
        index.insert(0, "sample_idx", np.arange(len(X)))
        return index
    raise FileNotFoundError(f"No sample_index.csv or X.npy in {sobol_dir}")


class BundleError(Exception):
    """A generation bundle is missing or only partially written -- e.g. a 0-byte
    parquet left by a job that was killed mid-write (disk quota / timeout). Such a
    generation is skipped so only samples with complete data are plotted."""


def read_table(run_dir: Path, stem: str) -> pd.DataFrame:
    """Read a parquet table, falling back to CSV.

    A legitimately empty table (e.g. a finish_order from a race nobody finished)
    comes back as an empty DataFrame. A truncated/corrupt parquet -- a 0-byte file
    or one pyarrow can't open -- raises ``BundleError`` so the caller can skip that
    generation instead of feeding garbage into the metrics."""
    parquet_path = run_dir / f"{stem}.parquet"
    csv_path = run_dir / f"{stem}.csv"
    if parquet_path.exists():
        if parquet_path.stat().st_size == 0:
            raise BundleError(f"0-byte parquet (incomplete write): {parquet_path}")
        try:
            return pd.read_parquet(parquet_path)
        except Exception as exc:  # ArrowInvalid (truncated), OSError, ...
            raise BundleError(f"unreadable parquet {parquet_path}: {exc}") from exc
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    raise FileNotFoundError(f"Missing {stem}.parquet or {stem}.csv in {run_dir}")


def choose_slice(index: pd.DataFrame, target_recovery: float, target_decay: float,
                 nearest: int | None, min_samples: int) -> tuple[pd.DataFrame, dict]:
    """Select rows close to target recovery_rate and utility_decay.

    Exact closest values are used first. For continuous Sobol designs this may be
    very sparse, so the selection is widened to the nearest rows by 2D normalized
    distance when fewer than ``min_samples`` rows are found, or when
    ``nearest`` is explicitly supplied.
    """
    rec_col = "recovery_rate"
    decay_col = "utility_decay"
    closest_recovery = index[rec_col].iloc[(index[rec_col] - target_recovery).abs().argmin()]
    closest_decay = index[decay_col].iloc[(index[decay_col] - target_decay).abs().argmin()]

    exact = index[
        np.isclose(index[rec_col], closest_recovery)
        & np.isclose(index[decay_col], closest_decay)
    ].copy()

    rec_span = max(index[rec_col].max() - index[rec_col].min(), np.finfo(float).eps)
    decay_span = max(index[decay_col].max() - index[decay_col].min(), np.finfo(float).eps)
    dist = np.sqrt(
        ((index[rec_col] - target_recovery) / rec_span) ** 2
        + ((index[decay_col] - target_decay) / decay_span) ** 2
    )

    desired = nearest if nearest is not None else min_samples
    if nearest is not None or len(exact) < min_samples:
        keep = dist.nsmallest(min(desired, len(index))).index
        selected = index.loc[keep].copy()
        mode = "nearest"
    else:
        selected = exact
        mode = "exact_closest_values"

    selected["_slice_distance"] = dist.loc[selected.index]
    info = {
        "mode": mode,
        "target_recovery_rate": target_recovery,
        "target_utility_decay": target_decay,
        "closest_recovery_rate": float(closest_recovery),
        "closest_utility_decay": float(closest_decay),
        "n_selected_samples": int(len(selected)),
    }
    return selected, info


def summarize_run(run_dir: Path) -> dict:
    agent_ts = read_table(run_dir, "agent_timeseries")
    model_ts = read_table(run_dir, "model_timeseries")
    agent_meta = read_table(run_dir, "agent_meta")
    finish_order = read_table(run_dir, "finish_order")

    group_time_frac = float((agent_ts["group_size"] > 1).mean())
    mean_stamina = float(model_ts["MeanStamina"].mean())
    mean_finish_time = float(finish_order["finish_time"].mean()) if len(finish_order) else math.nan

    def meta_mean(col: str) -> float:
        return float(agent_meta[col].mean()) if col in agent_meta.columns else math.nan

    return {
        "mean_finish_time": mean_finish_time,
        "mean_stamina": mean_stamina,
        "group_time_frac": group_time_frac,
        "coop_alpha": meta_mean("coeff.coop.alpha"),
        "leave_alpha": meta_mean("coeff.leave.alpha"),
        "follow_alpha": meta_mean("coeff.follow.alpha"),
        "n_finished": int(len(finish_order)),
        "n_agent_rows": int(len(agent_ts)),
    }


def _latest_complete_generation(gen_dirs: list[Path]) -> tuple[int, dict] | None:
    """Summarise the highest generation whose bundle fully reads.

    Walks generations from the last toward the first (closest to the evolution
    end-state first) and returns ``(generation, summary)`` for the first complete
    one, or ``None`` if every generation is missing/partially written -- the case
    when a disk-quota/timeout kill truncated the sample's final dumps."""
    for gen_dir in reversed(gen_dirs):
        generation = int(GEN_RE.match(gen_dir.name).group("gen"))
        try:
            return generation, summarize_run(gen_dir)
        except (BundleError, FileNotFoundError) as exc:
            print(f"[skip-gen] {exc}")
    return None


def collect_runs(sobol_dir: Path, selected: pd.DataFrame) -> pd.DataFrame:
    """One row per selected sample, read from its latest *complete* generation --
    the closest available end-state (gen folders sort numerically by zero-pad).
    Samples whose dirs are absent or hold no complete generation are dropped, so a
    partially-finished dump (e.g. cut short by disk quota) still plots."""
    selected_ids = set(int(v) for v in selected["sample_idx"])
    param_lookup = selected.set_index("sample_idx")
    rows = []
    skipped_samples = 0

    for sample_dir in sorted(p for p in sobol_dir.iterdir() if p.is_dir()):
        sample_match = SAMPLE_RE.match(sample_dir.name)
        if not sample_match:
            continue
        sample_idx = int(sample_match.group("sample"))
        if sample_idx not in selected_ids:
            continue

        gen_dirs = sorted(g for g in sample_dir.iterdir()
                          if g.is_dir() and GEN_RE.match(g.name))
        result = _latest_complete_generation(gen_dirs) if gen_dirs else None
        if result is None:
            skipped_samples += 1
            print(f"[skip] no complete generation in {sample_dir}")
            continue
        generation, summary = result

        params = param_lookup.loc[sample_idx]
        rows.append({
            "sample_idx": sample_idx,
            "generation": generation,
            "run_dir": str(sample_dir / f"gen_{generation:04d}"),
            "recovery_rate": float(params["recovery_rate"]),
            "breakaway_speed_frac": float(params["breakaway_speed_frac"]),
            "utility_decay": float(params["utility_decay"]),
            "k_s": float(params["k_s"]),
            "slice_distance": float(params["_slice_distance"]),
            **summary,
        })

    if not rows:
        raise RuntimeError(
            "No sample with a complete generation was found among the selected "
            "rows. Check --sobol-dir and the slice settings."
        )
    if skipped_samples:
        print(f"[warn] skipped {skipped_samples} selected sample(s) with no complete data")
    print(f"[collect] using {len(rows)} sample(s) with data")
    return pd.DataFrame(rows)


def add_bins(df: pd.DataFrame, bins: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df.copy()
    out["breakaway_bin"] = pd.cut(out["breakaway_speed_frac"], bins=bins, include_lowest=True)
    out["ks_bin"] = pd.cut(out["k_s"], bins=bins, include_lowest=True)

    break_labels = [interval_label(v) for v in out["breakaway_bin"].cat.categories]
    ks_labels = [interval_label(v) for v in out["ks_bin"].cat.categories]
    out["breakaway_bin_label"] = out["breakaway_bin"].map(interval_label)
    out["ks_bin_label"] = out["ks_bin"].map(interval_label)
    return out, break_labels, ks_labels


def interval_label(interval) -> str:
    return f"{interval.mid:.3f}"


def plot_heatmap(summary: pd.DataFrame, metric: str, break_labels: list[str],
                 ks_labels: list[str], out_path: Path) -> None:
    grid = summary.pivot(index="ks_bin_label", columns="breakaway_bin_label", values=metric)
    grid = grid.reindex(index=ks_labels, columns=break_labels)

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(grid.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(METRICS[metric])
    ax.set_xlabel("breakaway_speed_frac")
    ax.set_ylabel("k_s")
    ax.set_xticks(np.arange(len(break_labels)))
    ax.set_yticks(np.arange(len(ks_labels)))
    ax.set_xticklabels(break_labels, rotation=45, ha="right")
    ax.set_yticklabels(ks_labels)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(METRICS[metric])

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            value = grid.iat[y, x]
            if pd.notna(value):
                ax.text(x, y, f"{value:.2g}", ha="center", va="center",
                        color="white", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sobol-dir", required=True, type=Path,
                        help="Sobol dump directory, or parent directory containing sobol/")
    parser.add_argument("--out-dir", default=Path("gsa_heatmaps"), type=Path)
    parser.add_argument("--target-recovery-rate", type=float, default=0.1)
    parser.add_argument("--target-utility-decay", type=float, default=0.4)
    parser.add_argument("--slice-nearest", type=int, default=None,
                        help="Use the N nearest samples in recovery/decay space. "
                             "Default: use exact closest values unless sparse.")
    parser.add_argument("--min-samples", type=int, default=100,
                        help="Minimum samples before automatically widening to nearest rows.")
    parser.add_argument("--bins", type=int, default=10,
                        help="Number of bins on each heatmap axis.")
    args = parser.parse_args()

    sobol_dir = resolve_sobol_dir(args.sobol_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    index = load_index(sobol_dir)

    selected, slice_info = choose_slice(
        index=index,
        target_recovery=args.target_recovery_rate,
        target_decay=args.target_utility_decay,
        nearest=args.slice_nearest,
        min_samples=args.min_samples,
    )

    print(f"[slice] mode={slice_info['mode']}")
    print(
        "[slice] target recovery_rate={target_recovery_rate:.4g}, "
        "utility_decay={target_utility_decay:.4g}; closest exact values are "
        "recovery_rate={closest_recovery_rate:.4g}, "
        "utility_decay={closest_utility_decay:.4g}".format(**slice_info)
    )
    print(f"[slice] selected {len(selected)} sample(s)")

    per_run = collect_runs(sobol_dir, selected)
    per_run_path = args.out_dir / "per_run_metrics.csv"
    per_run.to_csv(per_run_path, index=False)

    binned, break_labels, ks_labels = add_bins(per_run, args.bins)
    summary = (
        binned
        .groupby(["ks_bin_label", "breakaway_bin_label"], observed=False)
        [list(METRICS)]
        .mean()
        .reset_index()
    )
    counts = (
        binned
        .groupby(["ks_bin_label", "breakaway_bin_label"], observed=False)
        .size()
        .reset_index(name="n_runs")
    )
    summary = summary.merge(counts, on=["ks_bin_label", "breakaway_bin_label"], how="left")
    summary_path = args.out_dir / "heatmap_cell_summary.csv"
    summary.to_csv(summary_path, index=False)

    for metric in METRICS:
        plot_heatmap(
            summary=summary,
            metric=metric,
            break_labels=break_labels,
            ks_labels=ks_labels,
            out_path=args.out_dir / f"heatmap_{metric}.png",
        )

    print(f"[done] wrote run metrics: {per_run_path}")
    print(f"[done] wrote heatmap table: {summary_path}")
    print(f"[done] wrote {len(METRICS)} heatmap PNGs to {args.out_dir}")


if __name__ == "__main__":
    main()
