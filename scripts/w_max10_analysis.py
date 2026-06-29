"""Analyze w_max10 (engine) correlations with learned strategy parameters.

This script:
1. Runs a single race with agent-level data dump
2. Extracts w_max10 and strategy coefficients per agent
3. Computes correlations between engine strength and learned strategies
4. Compares high-engine agents' performance vs. role-based expectations
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Repo root, so main.py runs correctly regardless of the cwd this script is called from.
REPO_ROOT = Path(__file__).resolve().parent.parent


def run_dump(seed: int, max_steps: int = 2000, out_dir: str = None) -> str:
    """Run main.py dump and return the output directory."""
    cmd = ["uv", "run", "python", "main.py", "dump", 
           "--seed", str(seed), 
           "--max-steps", str(max_steps)]
    
    if out_dir:
        cmd.extend(["--out-dir", out_dir])
    
    logger.info(f"Running race dump with seed {seed}...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    
    if result.returncode != 0:
        logger.error(f"Dump failed: {result.stderr}")
        return None
    
    # Extract output directory from stdout or use default
    if out_dir:
        return out_dir
    
    # Parse output to find directory
    for line in result.stderr.split('\n'):
        if 'analysis_output' in line:
            return line.split('/')[-1] if '/' in line else None
    
    # Default fallback
    return "analysis_output"


def load_agent_data(dump_dir: str) -> pd.DataFrame:
    """Load agent-level data from dump directory."""
    dump_path = Path(dump_dir)
    
    if not dump_path.exists():
        logger.error(f"Dump directory not found: {dump_path}")
        return None
    
    # Look for agent CSV files
    agent_files = list(dump_path.glob("*agents*.csv"))
    
    if not agent_files:
        logger.warning(f"No agent CSV files found in {dump_path}")
        return None
    
    logger.info(f"Loading agent data from {agent_files[0]}")
    return pd.read_csv(agent_files[0])


def compute_correlations(df: pd.DataFrame) -> dict:
    """Compute correlations between w_max10 and strategy parameters."""
    correlations = {}
    
    if 'w_max10' not in df.columns:
        logger.warning("w_max10 column not found in agent data")
        return correlations
    
    # Find all strategy coefficient columns
    coeff_cols = [c for c in df.columns if any(
        prefix in c for prefix in ['coop.', 'leave.', 'follow.']
    ) and any(suffix in c for suffix in ['alpha', 'beta', 'gamma', 'delta'])]
    
    logger.info(f"\nComputing correlations (w_max10 vs. coefficients):")
    logger.info(f"  n_agents: {len(df)}")
    
    for col in coeff_cols:
        if df[col].notna().sum() > 2:
            try:
                r, p_val = stats.pearsonr(df['w_max10'], df[col])
                correlations[col] = {
                    'r': r,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                }
                
                if p_val < 0.05:
                    logger.info(f"  {col:30s}: r={r:7.4f} (p={p_val:.4f}) **")
                else:
                    logger.info(f"  {col:30s}: r={r:7.4f} (p={p_val:.4f})")
            except:
                pass
    
    return correlations


def analyze_role_specialization(df: pd.DataFrame, n_quantiles: int = 3) -> dict:
    """Group agents by w_max10 quantiles and analyze strategy differences."""
    if 'w_max10' not in df.columns:
        return {}
    
    logger.info(f"\n" + "="*80)
    logger.info("ROLE SPECIALIZATION BY ENGINE STRENGTH")
    logger.info("="*80)
    
    df['w_max10_quantile'] = pd.qcut(df['w_max10'], q=n_quantiles, labels=['Weak', 'Medium', 'Strong'])
    
    results = {}
    
    for quantile in ['Weak', 'Medium', 'Strong']:
        group = df[df['w_max10_quantile'] == quantile]
        
        logger.info(f"\n{quantile} Engines (n={len(group)}):")
        logger.info(f"  w_max10: {group['w_max10'].mean():.1f} ± {group['w_max10'].std():.1f} W")
        
        # Compute mean strategy parameters for this group
        coeff_cols = [c for c in df.columns if any(
            prefix in c for prefix in ['coop.', 'leave.', 'follow.']
        ) and any(suffix in c for suffix in ['alpha', 'beta', 'gamma', 'delta'])]
        
        group_means = {}
        for col in coeff_cols:
            if col in group.columns and group[col].notna().sum() > 0:
                mean_val = group[col].mean()
                group_means[col] = mean_val
                
                # Show significant coefficients
                if abs(mean_val) > 0.2:
                    logger.info(f"    {col:25s}: {mean_val:7.4f}")
        
        results[quantile] = group_means
        
        # Compare performance (if available)
        if 'finish_order' in group.columns or 'distance' in group.columns:
            if 'finish_order' in group.columns:
                logger.info(f"  Avg finish position: {group['finish_order'].mean():.1f}")
    
    return results


def compare_strategies(best_strategy_coeffs: dict, agent_df: pd.DataFrame) -> dict:
    """Compare best-learned strategy coefficients with agent population."""
    logger.info(f"\n" + "="*80)
    logger.info("COMPARING BEST-LEARNED STRATEGY WITH POPULATION")
    logger.info("="*80)
    
    if not best_strategy_coeffs:
        logger.warning("No best strategy coefficients provided")
        return {}
    
    comparison = {}
    
    for coeff_name, learned_value in best_strategy_coeffs.items():
        if coeff_name in agent_df.columns:
            pop_mean = agent_df[coeff_name].mean()
            pop_std = agent_df[coeff_name].std()
            
            z_score = (learned_value - pop_mean) / pop_std if pop_std > 0 else 0
            
            comparison[coeff_name] = {
                'learned': learned_value,
                'population_mean': pop_mean,
                'population_std': pop_std,
                'z_score': z_score,
                'extreme': abs(z_score) > 2
            }
            
            if abs(z_score) > 2:
                logger.info(f"  {coeff_name:30s}: {learned_value:7.4f} (pop: {pop_mean:7.4f} ± {pop_std:7.4f}) [z={z_score:6.2f}] **")
            else:
                logger.info(f"  {coeff_name:30s}: {learned_value:7.4f} (pop: {pop_mean:7.4f} ± {pop_std:7.4f}) [z={z_score:6.2f}]")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Analyze w_max10 (engine) correlations with learned strategies"
    )
    parser.add_argument(
        "--batch-report", required=True,
        help="Path to batch learning analysis report (to get best strategy coefficients)"
    )
    parser.add_argument(
        "--seed", type=int, default=999,
        help="Seed for test race (default: 999)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=2000,
        help="Max steps for test race (default: 2000)"
    )
    parser.add_argument(
        "--dump-dir", default=None,
        help="Use existing dump directory (skip running new race)"
    )
    
    args = parser.parse_args()
    
    # Load batch report to get best strategy
    report_file = Path(args.batch_report)
    if not report_file.exists():
        logger.error(f"Report not found: {report_file}")
        return
    
    logger.info(f"Loading best strategy from: {report_file}")
    with open(report_file) as f:
        report = json.load(f)
    
    if not report.get('best_strategies'):
        logger.error("No best strategies in report")
        return
    
    best_strategy = report['best_strategies'][0]
    logger.info(f"Best strategy (seed {best_strategy['seed']}): utility_mean={best_strategy['utility_mean']:.4f}")
    
    # Run dump or load existing
    if args.dump_dir:
        dump_dir = args.dump_dir
        logger.info(f"Using existing dump directory: {dump_dir}")
    else:
        dump_dir = run_dump(args.seed, args.max_steps)
    
    # Load agent data
    agent_df = load_agent_data(dump_dir)
    if agent_df is None or agent_df.empty:
        logger.error("Failed to load agent data")
        return
    
    # Analyze
    logger.info(f"\nAgent dataset: {len(agent_df)} agents × {len(agent_df.columns)} features")
    
    # Compute correlations
    correlations = compute_correlations(agent_df)
    
    # Role specialization analysis
    roles = analyze_role_specialization(agent_df)
    
    # Compare best strategy with population
    best_coeffs = best_strategy.get('coefficients', {})
    comparison = compare_strategies(best_coeffs, agent_df)
    
    # Save results
    results = {
        'best_strategy_seed': best_strategy['seed'],
        'best_strategy_utility': best_strategy['utility_mean'],
        'n_agents': len(agent_df),
        'correlations': {k: float(v['r']) for k, v in correlations.items()},
        'role_specialization': roles,
        'comparison': {k: {
            'learned': v['learned'],
            'population_mean': float(v['population_mean']),
            'z_score': float(v['z_score'])
        } for k, v in comparison.items()}
    }
    
    output_file = Path(dump_dir) / "w_max10_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Analysis saved to: {output_file}")
    logger.info("\nKey findings:")
    logger.info(f"  - Strong correlations (|r| > 0.3): {sum(1 for c in correlations.values() if abs(c['r']) > 0.3)}")
    logger.info(f"  - Extreme strategy parameters (|z| > 2): {sum(1 for c in comparison.values() if c['extreme'])}")


if __name__ == "__main__":
    main()
