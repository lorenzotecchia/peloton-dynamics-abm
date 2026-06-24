"""Quantify the extent of learning in the peloton evolution system.

Measures:
  1. Population convergence: how much coefficient std declines (0 = full convergence)
  2. Coefficient drift: mean shift magnitude per generation
  3. Mean finishing time across completed riders
  4. Role specialization: whether diversity (std) is maintained or collapses
  5. Stability: how volatile is coefficient change in later generations

Usage:
    python scripts/quantify_learning.py --in learning.csv
"""

import argparse
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


MEAN_FINISHING_TIME_COL = 'mean_finishing_time'


def quantify_convergence(df) -> dict:
    """Measure how much the population std (diversity) declines over generations."""
    std_cols = [c for c in df.columns if c.endswith('_std')]
    if not std_cols:
        return {}
    
    # Average std across all parameters at each generation
    df['avg_std'] = df[std_cols].mean(axis=1)
    
    initial_std = df['avg_std'].iloc[0]
    final_std = df['avg_std'].iloc[-1]
    convergence_frac = 1 - (final_std / initial_std) if initial_std > 0 else 0
    
    return {
        'initial_avg_std': initial_std,
        'final_avg_std': final_std,
        'convergence_fraction': convergence_frac,  # 0 = no convergence, 1 = full convergence
        'std_decline_pct': 100 * convergence_frac,
    }


def quantify_coefficient_drift(df) -> dict:
    """Measure how much coefficient means change between consecutive generations."""
    mean_cols = [c for c in df.columns if c.endswith('_mean')]
    if not mean_cols:
        return {}
    
    # Compute absolute change in means between consecutive generations
    changes = []
    for i in range(1, len(df)):
        for col in mean_cols:
            delta = abs(df[col].iloc[i] - df[col].iloc[i-1])
            changes.append(delta)
    
    changes = np.array(changes)
    
    # Early vs. late change (first half vs. second half of generations)
    mid = len(df) // 2
    early_changes = []
    late_changes = []
    
    for i in range(1, len(df)):
        for col in mean_cols:
            delta = abs(df[col].iloc[i] - df[col].iloc[i-1])
            if i < mid:
                early_changes.append(delta)
            else:
                late_changes.append(delta)
    
    early_changes = np.array(early_changes)
    late_changes = np.array(late_changes)
    
    return {
        'mean_drift_per_gen': float(np.mean(changes)),
        'max_drift_per_gen': float(np.max(changes)),
        'early_drift_mean': float(np.mean(early_changes)),
        'late_drift_mean': float(np.mean(late_changes)),
        'drift_deceleration': float(np.mean(early_changes) - np.mean(late_changes)),
    }


def quantify_finish_rate(df) -> dict:
    """Measure whether riders finish the race (n_finished should approach n_agents=50)."""
    if 'n_finished' not in df.columns:
        return {}
    
    n_finished = df['n_finished'].values
    target = 50  # default n_agents
    
    initial_finish_rate = n_finished[0] / target
    final_finish_rate = n_finished[-1] / target
    
    return {
        'initial_finish_rate': float(initial_finish_rate),
        'final_finish_rate': float(final_finish_rate),
        'finish_rate_change': float(final_finish_rate - initial_finish_rate),
        'mean_finish_rate': float(np.mean(n_finished) / target),
    }


def quantify_role_specialization(df) -> dict:
    """Measure whether distinct roles emerge (std maintained) or converge (std -> 0)."""
    std_cols = [c for c in df.columns if c.endswith('_std')]
    if not std_cols:
        return {}
    
    # If std stays high = diverse roles. If std -> 0 = homogeneous.
    # We measure the "rebound" of std late vs. early.
    df['avg_std'] = df[std_cols].mean(axis=1)
    
    initial_std = df['avg_std'].iloc[0]
    min_std = df['avg_std'].min()
    final_std = df['avg_std'].iloc[-1]
    
    # "Rebound" = how much std increases after minimum
    rebound = final_std - min_std
    min_generation = df['avg_std'].idxmin()
    
    return {
        'min_std_generation': int(min_generation),
        'min_std_value': float(min_std),
        'rebound_after_min': float(rebound),
        'specialization_sign': 'roles_maintained' if final_std > 0.5 * initial_std else 'homogeneous',
    }


def quantify_learning_phases(df) -> dict:
    """Identify if there are distinct learning phases (fast vs. slow learning)."""
    std_cols = [c for c in df.columns if c.endswith('_std')]
    if not std_cols:
        return {}
    
    df['avg_std'] = df[std_cols].mean(axis=1)
    
    # Split into thirds and measure convergence rate in each
    n = len(df)
    third = n // 3
    
    rates = []
    for phase, (start, end) in enumerate([
        (0, third),
        (third, 2*third),
        (2*third, n)
    ]):
        if start < end:
            phase_std = df['avg_std'].iloc[start:end].values
            if len(phase_std) > 1:
                rate = (phase_std[0] - phase_std[-1]) / phase_std[0] if phase_std[0] > 0 else 0
                rates.append(rate)
    
    return {
        'phase_1_convergence_rate': float(rates[0]) if len(rates) > 0 else 0,
        'phase_2_convergence_rate': float(rates[1]) if len(rates) > 1 else 0,
        'phase_3_convergence_rate': float(rates[2]) if len(rates) > 2 else 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--in', dest='infile', default='learning.csv', help='Input CSV from learning run')
    p.add_argument('--out', dest='outfile', default=None, help='Optional: save summary plot')
    args = p.parse_args()
    
    df = pd.read_csv(args.infile)
    
    print(f"Learning data: {len(df)} generations, columns: {df.columns.tolist()}\n")
    
    # Compute all metrics
    conv = quantify_convergence(df)
    drift = quantify_coefficient_drift(df)
    finish = quantify_finish_rate(df)
    roles = quantify_role_specialization(df)
    phases = quantify_learning_phases(df)
    
    print("=" * 60)
    print("POPULATION CONVERGENCE")
    print("=" * 60)
    for k, v in conv.items():
        print(f"  {k:.<40} {v:>10.4f}")
    
    print("\nCOEFFICIENT DRIFT")
    print("=" * 60)
    for k, v in drift.items():
        print(f"  {k:.<40} {v:>10.6f}")
    
    print("\nFINISH RATE")
    print("=" * 60)
    for k, v in finish.items():
        print(f"  {k:.<40} {v:>10.4f}")
    
    print("\nROLE SPECIALIZATION")
    print("=" * 60)
    for k, v in roles.items():
        if isinstance(v, str):
            print(f"  {k:.<40} {v:>10s}")
        else:
            print(f"  {k:.<40} {v:>10.6f}")
    
    print("\nLEARNING PHASES")
    print("=" * 60)
    for k, v in phases.items():
        print(f"  {k:.<40} {v:>10.4f}")
    
    # Generate summary plot
    if args.outfile:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Average std (convergence)
        ax = axes[0, 0]
        mean_cols = [c for c in df.columns if c.endswith('_mean')]
        std_cols = [c for c in df.columns if c.endswith('_std')]
        if std_cols:
            df['avg_std'] = df[std_cols].mean(axis=1)
            ax.plot(df['generation'], df['avg_std'], marker='o', linewidth=2, label='Average coefficient std')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Std Dev (diversity)')
            ax.set_title('Population Convergence')
            ax.grid(alpha=0.3)
        
        # Plot 2: Mean drift per generation
        ax = axes[0, 1]
        if mean_cols:
            drifts_per_gen = []
            for i in range(1, len(df)):
                gen_drift = np.mean([abs(df[col].iloc[i] - df[col].iloc[i-1]) for col in mean_cols])
                drifts_per_gen.append(gen_drift)
            ax.plot(range(1, len(drifts_per_gen) + 1), drifts_per_gen, marker='s', linewidth=2, label='Mean drift')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Avg coefficient change')
            ax.set_title('Coefficient Drift per Generation')
            ax.grid(alpha=0.3)
        
        # Plot 3: Mean finishing time
        ax = axes[1, 0]
        if MEAN_FINISHING_TIME_COL not in df.columns:
            raise ValueError(
                "Cannot plot mean finishing time: input CSV has no recognized "
                f"'{MEAN_FINISHING_TIME_COL}' column. Regenerate learning.csv "
                "with `python main.py learn`."
            )
        ax.plot(
            df['generation'],
            df[MEAN_FINISHING_TIME_COL],
            marker='d',
            linewidth=2,
            label='Mean finishing time',
        )
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mean finishing time')
        ax.set_title('Mean Finishing Time Over Generations')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Plot 4: Sample coefficient trajectories (pick a few interesting ones)
        ax = axes[1, 1]
        sample_cols = [c for c in mean_cols if 'alpha' in c][:3]  # First 3 alpha params
        colors = plt.cm.Set1(np.linspace(0, 1, len(sample_cols)))
        for col, color in zip(sample_cols, colors):
            ax.plot(df['generation'], df[col], marker='o', label=col, color=color)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Coefficient value')
        ax.set_title('Sample Parameter Trajectories')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(args.outfile, dpi=200)
        print(f"\nSummary plot saved to {args.outfile}")


if __name__ == '__main__':
    main()
