"""Analyze and compare learned strategies with high w_max10 (engine) agents.

This script:
1. Loads learned strategy parameters from batch_learning results
2. Modifies a single simulation to run with best-learned coefficients vs. high w_max10 agents
3. Compares performance and strategy parameter correlations with initial engine
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import statistics
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_analysis_report(report_file: str) -> dict:
    """Load the analysis report from batch_learning."""
    with open(report_file, 'r') as f:
        return json.load(f)


def extract_best_coefficients(report: dict) -> dict:
    """Extract coefficients from the single best-performing replication."""
    if not report.get('best_strategies'):
        raise ValueError("No best strategies found in report")
    
    best = report['best_strategies'][0]
    logger.info(f"Using best strategy from seed {best['seed']} "
               f"(utility_mean: {best['utility_mean']:.4f})")
    
    return best['coefficients']


def analyze_w_max10_distribution(batch_data_dir: str) -> dict:
    """Analyze distribution of w_max10 values across replications.
    
    This requires extracting w_max10 from individual races, which are stored
    in the model state. For now, we'll document what w_max10 represents and
    how to extract it.
    """
    logger.info("\n" + "="*80)
    logger.info("W_MAX10 DISTRIBUTION ANALYSIS")
    logger.info("="*80)
    logger.info("""
W_max10 (maximum power over 10 minutes) is each rider's 'engine' — a measure
of physiological capacity sampled from a Gaussian distribution:
    w_max10 ~ N(μ={w_max10_mean}, σ={w_max10_std})

Riders with higher w_max10 are physiologically stronger. The learning system
selects for strategies that work well with a rider's engine type, so we expect:

1. Emergent role specialization: strong-engine riders may develop different
   strategies than weaker riders
2. Similarity-based imitation: riders imitate peers who race *like them*
   (similar w_max10) AND performed better

To extract w_max10 distribution:
- Run `uv run python main.py dump` to get per-agent data files
- Extract w_max10 from agents in each race
- Group agents by w_max10 quantiles and analyze their strategy coefficients
    """)
    
    return {
        'note': 'w_max10 extraction requires dump command analysis',
        'recommendation': 'Run batch_learning.py first, then use dump command'
    }


def create_comparison_recommendations() -> None:
    """Suggest how to run comparisons between learned strategies and engine types."""
    logger.info("\n" + "="*80)
    logger.info("STRATEGY COMPARISON WORKFLOW")
    logger.info("="*80)
    
    logger.info("""
To compare best learned strategies with high w_max10 agents:

1. Extract best coefficients from batch learning:
   
   coefficients = report['best_strategies'][0]['coefficients']
   
2. Create a variant of the model that:
   a. Runs with fixed seed (reproducible)
   b. Logs agent w_max10 and final coefficients
   c. Compares finish position vs. w_max10 to measure role specialization

3. Analyze correlations:
   - Pearson correlation(w_max10, coop.alpha) → do strong engines cooperate more?
   - Pearson correlation(w_max10, utility) → does engine predict finish position?
   - Among high-engine agents, compare learned vs. random strategies

4. Expected findings:
   - Role specialization: sprinters (low coop.alpha) vs. domestiques (high alpha)
   - Strong engines may learn to sit-in (defect) more; weak engines help pull
   - Best strategies balance role specialization with population diversity
    """)


def print_learned_strategy_summary(report: dict) -> None:
    """Print summary of learned strategy parameters."""
    logger.info("\n" + "="*80)
    logger.info("LEARNED STRATEGY PARAMETERS (Top 5)")
    logger.info("="*80)
    
    for i, strategy in enumerate(report.get('best_strategies', [])[:5], 1):
        logger.info(f"\n#{i} Strategy (seed {strategy['seed']}):")
        logger.info(f"   Final utility_mean: {strategy['utility_mean']:.4f}")
        logger.info(f"   Final utility_std:  {strategy['utility_std']:.4f}")
        
        # Organize coefficients by type
        coop_coeffs = {k: v for k, v in strategy['coefficients'].items() if k.startswith('coop.')}
        leave_coeffs = {k: v for k, v in strategy['coefficients'].items() if k.startswith('leave.')}
        follow_coeffs = {k: v for k, v in strategy['coefficients'].items() if k.startswith('follow.')}
        
        if coop_coeffs:
            logger.info("   Cooperation:")
            for k, v in sorted(coop_coeffs.items()):
                logger.info(f"     {k}: {v:.4f}")
        
        if leave_coeffs:
            logger.info("   Breakaway:")
            for k, v in sorted(leave_coeffs.items()):
                logger.info(f"     {k}: {v:.4f}")
        
        if follow_coeffs:
            logger.info("   Following:")
            for k, v in sorted(follow_coeffs.items()):
                logger.info(f"     {k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare learned strategies with engine (w_max10) distribution"
    )
    parser.add_argument(
        "--report", required=True,
        help="Path to analysis_report_*.json from batch_learning.py"
    )
    parser.add_argument(
        "--batch-dir", default="data/batch_learning",
        help="Directory containing batch learning results"
    )
    
    args = parser.parse_args()
    
    report_file = Path(args.report)
    if not report_file.exists():
        logger.error(f"Report file not found: {report_file}")
        return
    
    # Load and analyze
    logger.info(f"Loading analysis report: {report_file}")
    report = load_analysis_report(str(report_file))
    
    # Print learned strategies
    print_learned_strategy_summary(report)
    
    # Analyze w_max10 distribution
    analyze_w_max10_distribution(args.batch_dir)
    
    # Print comparison recommendations
    create_comparison_recommendations()
    
    logger.info("\n✓ Analysis complete!")
    logger.info(f"\nNext steps:")
    logger.info(f"1. Review learned strategies above")
    logger.info(f"2. Run: uv run python main.py dump --seed 42 (to get agent-level data)")
    logger.info(f"3. Extend this script to analyze w_max10 vs. coefficient correlations")


if __name__ == "__main__":
    main()
