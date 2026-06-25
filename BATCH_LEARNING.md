# Batch Learning Workflow: 100 Replications + Strategy Analysis

This workflow runs 100 independent replications of learning simulations to characterize strategy parameter distributions and compare learned strategies with agent engine capabilities (w_max10).

## Overview

### What This Does

1. **Batch Runner (`batch_learning.py`)**: Runs 100 replications of `main.py learn --generation 100` with different seeds
   - Each replication runs 100 generations of evolutionary learning
   - Collects per-generation trajectory of strategy coefficients
   - Extracts final strategy parameters and performance metrics

2. **Strategy Analyzer (`analyze_strategies.py`)**: Compares best-learned strategies with agent engine distribution
   - Aggregates strategy parameter distributions across all 100 replications
   - Identifies top 5 best-performing strategies
   - Analyzes w_max10 (engine) distribution and correlations

### Key Concepts

- **w_max10**: Each rider's maximum power output (watts) over 10 minutes — their "engine"
  - Sampled at agent initialization from: `w_max10 ~ N(mean=300, std=30)` [configurable]
  - Determines physiological capacity and drafting efficiency
  - Used for similarity-based imitation: riders learn from peers with similar engines

- **Strategy Coefficients**: Learned parameters for cooperation/breakaway/following decisions
  - `coop.alpha, coop.beta, coop.gamma`: Cooperation strategy parameters
  - `leave.alpha, leave.beta, leave.gamma`: Breakaway decision parameters
  - `follow.alpha, follow.beta, follow.gamma`: Group-following parameters

- **Evolutionary Learning**: Between races, agents update coefficients by imitating better-performing peers with similar w_max10
  - Top 20% become "donors" for coefficient updates
  - Bottom 30% copy/blend coefficients from donors
  - Mutation adds diversity

## Quick Start

### Run 100 Replications (takes ~2-4 hours on typical hardware)

```bash
# Full 100 replications with 100 generations each
uv run python batch_learning.py

# Or customize:
uv run python batch_learning.py \
  --replications 100 \
  --generations 100 \
  --max-steps 400 \
  --output-dir data/batch_learning
```

### Analyze Results

After batch run completes, find the latest report:
```bash
ls -t data/batch_learning/analysis_report_*.json | head -1
```

Then analyze:
```bash
uv run python analyze_strategies.py \
  --report data/batch_learning/analysis_report_YYYYMMDD_HHMMSS.json
```

### Load and Analyze Existing Results

If simulations already completed:
```bash
uv run python batch_learning.py --skip-run --replications 100
```

## Output Files

### In `data/batch_learning/`:

- **`replication_NNN.csv`**: Per-generation trajectory for replication N
  - Columns: `generation, n_finished, [coefficient stats], [utility stats]`
  - Use to trace learning curves

- **`analysis_report_YYYYMMDD_HHMMSS.json`**: Main summary
  ```json
  {
    "num_replications": 100,
    "coefficient_distributions": {
      "coop.alpha_mean": {
        "mean": 0.42,
        "std": 0.15,
        "min": 0.05,
        "max": 0.89
      },
      ...
    },
    "utility_distributions": {...},
    "best_strategies": [
      {
        "rank": 1,
        "seed": 47,
        "utility_mean": 0.523,
        "coefficients": {...}
      },
      ...
    ]
  }
  ```

- **`coefficient_distributions_YYYYMMDD_HHMMSS.csv`**: All raw coefficient values
  - One row per replication, columns for each coefficient
  - Use for: histograms, scatter plots, correlation analysis

## Analysis Patterns

### 1. View Learning Curves

```python
import pandas as pd

# Load a single replication's trajectory
df = pd.read_csv('data/batch_learning/replication_042.csv')

# Plot coefficient evolution
df[['generation', 'coop.alpha_mean', 'coop.beta_mean']].plot(x='generation')

# Show utility improvement
df[['generation', 'utility_mean', 'utility_std']].plot(x='generation')
```

### 2. Compare Best Strategies

```python
import json

with open('data/batch_learning/analysis_report_*.json') as f:
    report = json.load(f)

# Show top 5 strategies
for strategy in report['best_strategies'][:5]:
    print(f"Seed {strategy['seed']}: utility_mean={strategy['utility_mean']:.4f}")
    print(f"  Coefficients: {strategy['coefficients']}")
```

### 3. Strategy Parameter Distribution Statistics

```python
import json
import pandas as pd

with open('data/batch_learning/analysis_report_*.json') as f:
    report = json.load(f)

# Print coefficient distribution summary
for key, dist in report['coefficient_distributions'].items():
    print(f"{key:30s}: μ={dist['mean']:7.4f} σ={dist['std']:7.4f}")
```

## Interpreting Results

### Expected Findings

1. **Role Specialization**: Final population splits into distinct strategies
   - Sprinters: low `coop.alpha` (defect more), high final utility despite small population
   - Domestiques: high `coop.alpha` (cooperate), lower utility but enable team

2. **Engine-Strategy Correlation**: Learned strategies differ by w_max10
   - Strong engines: may learn aggressive strategies (breakaway, sit-in)
   - Weak engines: learn cooperative strategies (help pace, lead-out)

3. **Convergence**: Mean coefficients stabilize by generation 50-80
   - Std decreases initially (convergence) but may increase if multiple equilibria

4. **Utility Improvement**: Mean utility should increase over generations (learning works)
   - Final `utility_mean` ~ 0.3–0.5 (exponential decay, 20 riders)

### Comparing with High w_max10 Agents

To identify which strategy parameters correlate with engine strength:

```bash
# Run a single race with agent-level data dump
uv run python main.py dump --seed 123

# This creates per-agent CSVs in analysis_output/YYYYMMDD-HHMMSS/
# You can then compute:
#   correlation(w_max10, coop.alpha) across all agents
#   correlation(w_max10, final_position) to measure role utility
```

## Advanced: Running on HPC (Snellius)

To submit 100 replications as a single Slurm job array:

```bash
# Create a job script that calls batch_learning.py
cat > submit_batch.sh << 'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

cd /path/to/peloton-dynamics-abm
uv run python batch_learning.py --replications 100
EOF

sbatch submit_batch.sh
```

Alternatively, submit each replication as a separate job and merge results:

```bash
for seed in {0..99}; do
  sbatch --job-name=learn_$seed \
    -c 1 --time=00:30:00 \
    -o jobs/logs/learn_$seed.log \
    bash -c "cd /path/to/repo && uv run python main.py learn --generations 100 --seed $seed --out data/batch_learning/replication_$(printf '%03d' $seed).csv"
done
```

## Troubleshooting

### Simulation Too Slow?

- Reduce generations: `--generations 50`
- Reduce max_steps per race: `--max-steps 200`
- Or submit to HPC with parallel processing

### Out of Disk Space?

Each replication produces a ~50–100 KB CSV file. 100 replications ≈ 5–10 MB. If disk is full:
```bash
rm -f data/batch_learning/replication_*.csv  # Keep only final report
```

### Missing Dependencies?

```bash
uv sync  # Re-install from uv.lock
```

## Next Steps

1. ✅ Run batch_learning.py (100 replications)
2. ✅ View analysis report
3. Run dump command to extract per-agent w_max10 data
4. Extend analyze_strategies.py to correlate w_max10 ↔ learned coefficients
5. Compare learned strategies in a head-to-head race simulation
6. Plot strategy distributions and learning curves for publication

---

**Questions?** Check CLAUDE.md for project architecture details.
