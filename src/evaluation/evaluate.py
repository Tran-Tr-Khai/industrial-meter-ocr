"""
Evaluation Script
==================
Evaluate trained models and compare experiment results.

Usage:
    # Evaluate a specific run
    python -m src.evaluation.evaluate --run-dir runs/e0_baseline/run_name

    # Compare multiple experiments
    python -m src.evaluation.evaluate --compare runs/e0_baseline runs/e1_model_capacity
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

from src.utils.helpers import PROJECT_ROOT, PROCESSED_DIR, RUNS_DIR


def evaluate_single_run(run_dir: Path, data_yaml: Path = None):
    """Evaluate a single trained model."""
    if data_yaml is None:
        data_yaml = PROCESSED_DIR / "data.yaml"

    # Find best model
    best_model = run_dir / "weights" / "best.pt"
    if not best_model.exists():
        print(f"  ❌ No best.pt found in {run_dir}")
        return None

    print(f"\n📊 Evaluating: {run_dir.name}")
    print(f"   Model: {best_model}")
    print(f"   Data:  {data_yaml}")

    model = YOLO(str(best_model))

    # Validate on val set
    print(f"\n   Validation set results:")
    val_results = model.val(data=str(data_yaml), split="val", verbose=False)

    metrics = {
        "val": {
            "mAP50": round(float(val_results.box.map50), 4),
            "mAP50-95": round(float(val_results.box.map), 4),
            "precision": round(float(val_results.box.mp), 4),
            "recall": round(float(val_results.box.mr), 4),
        }
    }

    print(f"     mAP50:     {metrics['val']['mAP50']:.4f}")
    print(f"     mAP50-95:  {metrics['val']['mAP50-95']:.4f}")
    print(f"     Precision: {metrics['val']['precision']:.4f}")
    print(f"     Recall:    {metrics['val']['recall']:.4f}")

    # Validate on test set
    print(f"\n   Test set results:")
    test_results = model.val(data=str(data_yaml), split="test", verbose=False)

    metrics["test"] = {
        "mAP50": round(float(test_results.box.map50), 4),
        "mAP50-95": round(float(test_results.box.map), 4),
        "precision": round(float(test_results.box.mp), 4),
        "recall": round(float(test_results.box.mr), 4),
    }

    print(f"     mAP50:     {metrics['test']['mAP50']:.4f}")
    print(f"     mAP50-95:  {metrics['test']['mAP50-95']:.4f}")
    print(f"     Precision: {metrics['test']['precision']:.4f}")
    print(f"     Recall:    {metrics['test']['recall']:.4f}")

    # Measure inference speed
    import time
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n   Inference speed ({device}):")

    # Get a test image
    test_img_dir = PROCESSED_DIR / "test" / "images"
    test_images = list(test_img_dir.glob("*.jpg"))[:10]

    if test_images:
        times = []
        for img in test_images:
            t0 = time.time()
            model.predict(str(img), verbose=False)
            times.append((time.time() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        metrics["inference_ms"] = round(avg_ms, 2)
        print(f"     Avg: {avg_ms:.1f} ms/image")

    return metrics


def compare_experiments(experiment_dirs: list[Path]):
    """Compare results across multiple experiments."""
    print("\n" + "=" * 80)
    print("  EXPERIMENT COMPARISON")
    print("=" * 80)

    all_metrics = {}

    for exp_dir in experiment_dirs:
        exp_dir = Path(exp_dir)
        if not exp_dir.exists():
            print(f"  ⚠ Directory not found: {exp_dir}")
            continue

        # Find all run directories
        run_dirs = sorted([
            d for d in exp_dir.iterdir()
            if d.is_dir() and (d / "weights" / "best.pt").exists()
        ])

        if not run_dirs:
            print(f"  ⚠ No trained models in: {exp_dir}")
            continue

        for run_dir in run_dirs:
            metrics = evaluate_single_run(run_dir)
            if metrics:
                all_metrics[run_dir.name] = metrics

    if not all_metrics:
        print("  No results to compare.")
        return

    # Print comparison table
    print(f"\n{'─' * 80}")
    print(f"  {'Experiment':<35} {'mAP50':>8} {'mAP50-95':>10} {'Prec':>8} {'Rec':>8} {'ms':>8}")
    print(f"{'─' * 80}")

    for name, m in all_metrics.items():
        test = m.get("test", m.get("val", {}))
        inf_ms = m.get("inference_ms", 0)
        print(
            f"  {name[:35]:<35} "
            f"{test.get('mAP50', 0):>8.4f} "
            f"{test.get('mAP50-95', 0):>10.4f} "
            f"{test.get('precision', 0):>8.4f} "
            f"{test.get('recall', 0):>8.4f} "
            f"{inf_ms:>8.1f}"
        )

    print(f"{'─' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare experiments")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to a specific run directory to evaluate",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        default=None,
        help="Paths to experiment directories to compare",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        metrics = evaluate_single_run(run_dir)
        if metrics:
            # Save metrics
            out_path = run_dir / "evaluation_metrics.yaml"
            with open(out_path, "w") as f:
                yaml.dump(metrics, f, default_flow_style=False)
            print(f"\n📄 Metrics saved to: {out_path}")

    elif args.compare:
        compare_experiments([Path(p) for p in args.compare])

    else:
        # Default: compare all experiments in runs/
        if RUNS_DIR.exists():
            exp_dirs = sorted([
                d for d in RUNS_DIR.iterdir() if d.is_dir()
            ])
            if exp_dirs:
                compare_experiments(exp_dirs)
            else:
                print("No experiments found in runs/")
        else:
            print("No runs/ directory. Train a model first.")


if __name__ == "__main__":
    main()
