"""Test whether learning is real or just copying the best-skilled agent's initial strategy."""

import copy
import json
import statistics
from pathlib import Path
import pandas as pd
import numpy as np


def extract_coefficients_snapshot(riders: list, label: str = "snapshot") -> dict:
    """Capture current coefficients of all riders."""
    snapshot = {}
    for i, rider in enumerate(riders):
        snapshot[f"rider_{i:02d}"] = {
            "w_max10": rider.w_max10,
            "coeffs": copy.deepcopy(rider.coeffs),
            "utility": getattr(rider, "utility", 0.0),
        }
    return snapshot


def find_best_skilled_agents(riders: list, top_n: int = 1) -> list:
    """Find agents with highest w_max10 (best initial skill)."""
    sorted_riders = sorted(riders, key=lambda r: r.w_max10, reverse=True)
    return sorted_riders[:top_n]


def compare_strategies(coeffs_dict1: dict, coeffs_dict2: dict) -> dict:
    """Compute similarity between two strategy coefficient dicts."""
    diffs = []
    for key in coeffs_dict1:
        if key in coeffs_dict2:
            for param in coeffs_dict1[key]:
                if param in coeffs_dict2[key]:
                    diff = abs(coeffs_dict1[key][param] - coeffs_dict2[key][param])
                    diffs.append(diff)

    if not diffs:
        return None

    euclidean = np.sqrt(sum(d**2 for d in diffs))
    return {
        "euclidean_distance": euclidean,
        "mean_abs_diff": statistics.mean(diffs),
        "max_abs_diff": max(diffs),
    }


def run_learning_with_tracking(
    generations: int = 100, max_steps: int = 2000, seed: int = None
):
    """Run learning and track initial best-skilled vs. final learned strategies."""
    from peloton.model import PelotonModel
    from peloton.config import PelotonConfig
    from peloton.evolution import evolve, _coeff_stats, _utility_stats

    output_dir = Path("data/learning_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = PelotonConfig(seed=seed)

    # Generation 0: initialize population
    model = PelotonModel(config=cfg)

    # CAPTURE: Initial state
    best_skilled_initial = find_best_skilled_agents(model.riders, top_n=1)[0]
    initial_best_coeffs = copy.deepcopy(best_skilled_initial.coeffs)
    initial_best_w_max10 = best_skilled_initial.w_max10

    # Run learning loop
    population = None
    for gen in range(generations):
        if gen > 0:
            model = PelotonModel(config=cfg, population=population)

        # Run one race
        for _ in range(max_steps):
            if not model.running:
                break
            model.step()

        evolve(model.riders, model)
        population = [copy.deepcopy(rider.coeffs) for rider in model.riders]

    # CAPTURE: Final state
    best_skilled_final = find_best_skilled_agents(model.riders, top_n=1)[0]
    final_best_coeffs = copy.deepcopy(best_skilled_final.coeffs)

    # ANALYSIS
    initial_vs_final_best = compare_strategies(initial_best_coeffs, final_best_coeffs)

    convergence_to_best = []
    for rider in model.riders:
        if rider.unique_id != best_skilled_initial.unique_id:
            comparison = compare_strategies(initial_best_coeffs, rider.coeffs)
            if comparison:
                convergence_to_best.append(comparison["euclidean_distance"])

    pairwise_diffs = []
    riders_list = list(model.riders)
    for i in range(len(riders_list)):
        for j in range(i + 1, min(i + 5, len(riders_list))):
            comp = compare_strategies(riders_list[i].coeffs, riders_list[j].coeffs)
            if comp:
                pairwise_diffs.append(comp["euclidean_distance"])

    results = {
        "seed": seed,
        "generations": generations,
        "best_skilled_distance": float(initial_vs_final_best["euclidean_distance"]),
        "population_mean_distance_to_best": (
            float(statistics.mean(convergence_to_best)) if convergence_to_best else 0
        ),
        "pairwise_distance": (
            float(statistics.mean(pairwise_diffs)) if pairwise_diffs else 0
        ),
    }

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_learning_with_tracking(
        generations=args.generations, max_steps=args.max_steps, seed=args.seed
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
