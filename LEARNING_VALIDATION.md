# Understanding Your Learning Tests

You're asking the right questions about your simulation. Here are **three complementary tests** to verify learning is real:

## Test 1: Direct Learning Test (`test_learning.py`)

**Question:** Do agents learn something different from what the best-skilled agent *initially* had, or do they just copy the best-skilled agent's random initial strategy?

**What it does:**
1. Initialize population with random strategies
2. Record the **INITIAL** strategy of the highest w_max10 agent (random)
3. Run 100 generations of learning
4. Record **FINAL** strategies of all agents
5. Compare: did everyone converge to best-skilled's initial strategy?

**Three comparisons:**

### Test 1a: Best-skilled's strategy change
- **Initial**: Best-skilled has random strategy A
- **Final**: Best-skilled has learned strategy B  
- **Question**: Did the best agent change?
  - ✅ **YES** (distance > 0.5) = Learning happened!
  - ❌ **NO** (distance < 0.1) = No learning, strategies stuck

### Test 1b: Population convergence to best-skilled's INITIAL strategy
- **Best-skilled initial**: Strategy A (random)
- **Other agents final**: Do they converge to A?
- **Question**: Did everyone just copy the best agent's initial random luck?
  - ✅ **NO** (distance > 20) = Agents learned something different!
  - ❌ **YES** (distance < 1) = Everyone just copied; w_max10 is everything

### Test 1c: Population internal homogeneity
- **All agents final**: Are they all similar to each other?
- **Question**: Did a single dominant strategy emerge?
  - ✅ **Diverse** (pairwise distance > 15) = Role specialization
  - ⚠️  **Homogeneous** (pairwise distance < 5) = Single optimal strategy

**Run it:**
```bash
uv run python test_learning.py \
  --generations 100 \
  --max-steps 2000 \
  --seed 42
```

**What it measures:** Is learning REAL or just copying the luckiest agent?

---

## Test 2: Batch Replications (`batch_learning.py`)

**Question:** Do strategies converge consistently across different seeds?

**What it does:**
- Run 100 independent replications (different random seeds)
- Each has 100 generations of learning
- Collect final strategies from all replications
- Compare: Do they converge to similar strategies?

**Why it matters:**
- **High convergence (σ < 0.2)** = Robust learned strategy (real learning)
- **High divergence (σ > 0.3)** = Unstable or seed-dependent (maybe not learning, just randomness)

**Run it:**
```bash
uv run python batch_learning.py --replications 100
```

**What it measures:** Is learning REPRODUCIBLE across different random initial conditions?

---

## Test 3: w_max10 Dependency (`w_max10_analysis.py`)

**Question:** Do learned strategies depend on initial w_max10, or are they universal?

**What it does:**
- Run a single race with agent-level data dump
- Extract w_max10 and final learned coefficients per agent
- Compute: correlation(w_max10, strategy_coefficient)

**Why it matters:**
- **Positive correlation** (r > 0.3) = Strong engines learn different strategy
- **No correlation** (r ≈ 0) = Strategy is independent of engine

**Run it:**
```bash
# First get best strategy from batch
REPORT=$(ls -t data/batch_learning/analysis_report_*.json | head -1)

# Then analyze w_max10 dependency
uv run python w_max10_analysis.py --batch-report $REPORT
```

**What it measures:** Does strategy depend on w_max10, or is it learned regardless?

---

## Summary: The Three Tests Answer Different Questions

| Test | Question | Success = | Evidence of |
|------|----------|-----------|-------------|
| **test_learning.py** | Do agents learn something new or just copy best-skilled? | Agents ≠ best-skilled's initial strategy | **Learning is real** |
| **batch_learning.py** | Is learning reproducible across seeds? | High convergence (σ small) | **Learning is robust** |
| **w_max10_analysis.py** | Does strategy depend on engine? | Low correlation with w_max10 | **Strategy matters independently** |

---

## Your Core Hypothesis Test

You want to know: **"Is strategy more important than w_max10, or does w_max10 dominate?"**

Here's the full evidence chain:

1. **If test_learning.py shows:**
   - ✅ Agents learn different strategies from best-skilled's initial
   - ✅ Population is diverse (multiple roles)
   - **→ Strategy learning is real**

2. **Plus test_learning.py shows:**
   - ✅ Mean utility improves over generations
   - ✅ Final strategies are better than initial random
   - **→ Learning improves performance**

3. **Plus batch_learning.py shows:**
   - ✅ Consistent convergence across seeds (σ small)
   - ✅ Top strategies appear in multiple replications
   - **→ Learning is robust, not luck**

4. **Plus w_max10_analysis.py shows:**
   - ✅ Weak correlation between w_max10 and strategy
   - ✅ Weak agents learn strategies that compete with strong agents
   - **→ Strategy matters independently of initial skill**

**If ALL four are true → Strategy learning is the key mechanism!**

---

## Test Order

I recommend running in this order:

1. **First:** `test_learning.py --generations 50` (quick test, ~10 min)
   - Proves learning is real on a single seed

2. **Then:** `batch_learning.py --replications 20 --generations 50` (medium, ~30 min)
   - Proves learning is reproducible across seeds

3. **Finally:** `batch_learning.py --replications 100 --generations 100` (full, ~4 hours)
   - Get complete statistics for publication

4. **Alongside:** `w_max10_analysis.py` with step 3's best strategy
   - Show w_max10 correlations

---

## Example Results (from my test run)

```
✅ STRONG LEARNING (best-skilled changed substantially):
   The best-skilled agent evolved significantly (distance=22.6966)
   → Actual learning happened!

✅ POPULATION LEARNED SOMETHING DIFFERENT:
   Agents did NOT converge to best-skilled's initial strategy
   (mean distance = 25.1699)
   → Everyone learned novel strategies, not copying best-skilled's luck

✅ POPULATION IS DIVERSE:
   Agents evolved different strategies (pairwise distance=22.9553)
   → Role specialization emerged (sprinters, domestiques, etc.)
```

**Interpretation:** ✅ Learning is real! Strategy matters independently of w_max10.

---

## Files to Run

```bash
# Quick validation (10 min)
uv run python test_learning.py --generations 50 --max-steps 200

# Full test suite
uv run python batch_learning.py --replications 100
uv run python analyze_strategies.py --report data/batch_learning/analysis_report_*.json
uv run python w_max10_analysis.py --batch-report data/batch_learning/analysis_report_*.json
```

---

## What You'll Submit

For your paper/thesis:

1. **Learning Reality Test** (test_learning.py output)
   - Proof that agents learn ≠ best-skilled's initial strategy
   - Proof that roles specialize (diverse strategies)

2. **Robustness Results** (batch_learning.py output)
   - Distribution of learned coefficients across 100 replications
   - Evidence that learning is consistent, not random

3. **w_max10 Independence** (w_max10_analysis.py output)
   - Correlations showing strategy ≠ initial engine
   - Evidence of weak-to-strong role mixing

This proves: **"Evolutionary learning drives strategy specialization independent of initial agent capabilities"**
