"""
Multi-Seed Experiment Runner
=============================
Runs the same experiment with multiple seeds for statistical comparison.

Usage:
    python -m src.training.run_seeds --experiment e0_baseline --seeds 42 123 456
"""

import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime

from src.training.train import train_experiment
from src.utils.helpers import RUNS_DIR, ensure_dir


def run_multi_seed(
    experiment_name: str,
    seeds: list[int],
    model_name: str = None,
    extra_args: dict = None,
):
    """Run an experiment with multiple seeds and aggregate results."""
    print("=" * 60)
    print(f"  MULTI-SEED EXPERIMENT: {experiment_name}")
    print(f"  Seeds: {seeds}")
    print("=" * 60)

    all_results = []

    for i, seed in enumerate(seeds):
        print(f"\n{'─' * 60}")
        print(f"  Seed {seed} ({i + 1}/{len(seeds)})")
        print(f"{'─' * 60}")

        try:
            results = train_experiment(
                experiment_name=experiment_name,
                model_name=model_name,
                seed=seed,
                extra_args=extra_args,
            )
            all_results.append({
                "seed": seed,
                "status": "success",
            })
        except Exception as e:
            print(f"  ❌ Seed {seed} failed: {e}")
            all_results.append({
                "seed": seed,
                "status": "failed",
                "error": str(e),
            })

    # Save multi-seed summary
    summary_dir = ensure_dir(RUNS_DIR / experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = summary_dir / f"multi_seed_summary_{timestamp}.yaml"

    summary = {
        "experiment": experiment_name,
        "seeds": seeds,
        "results": all_results,
        "note": (
            "Compare metrics across seeds in each run directory. "
            "If metric differences < std across seeds, "
            "the difference is not statistically significant."
        ),
    }

    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(f"\n✅ Multi-seed experiment complete!")
    print(f"   Summary: {summary_path}")
    print(f"   Successful: {sum(1 for r in all_results if r['status'] == 'success')}/{len(seeds)}")


def main():
    parser = argparse.ArgumentParser(description="Run experiment with multiple seeds")
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Experiment config name",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds to use (default: 42 123 456)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override model",
    )
    args = parser.parse_args()

    run_multi_seed(
        experiment_name=args.experiment,
        seeds=args.seeds,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
