"""
Training Script for YOLOv8 Experiments
=======================================
Supports all experiments defined in the research plan.

Usage:
    # Run baseline experiment
    python -m src.training.train --experiment e0_baseline

    # Run with custom config
    python -m src.training.train --experiment e1_model_capacity --model yolov8s.pt

    # Run with specific seed
    python -m src.training.train --experiment e0_baseline --seed 42
"""

import argparse
import time
import yaml
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

from src.utils.helpers import (
    PROJECT_ROOT, PROCESSED_DIR, RUNS_DIR, CONFIGS_DIR,
    set_seed, ensure_dir,
)


def load_experiment_config(experiment_name: str) -> dict:
    """Load experiment configuration from YAML file."""
    config_path = CONFIGS_DIR / f"{experiment_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Available configs: {list(CONFIGS_DIR.glob('*.yaml'))}"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def train_experiment(
    experiment_name: str,
    model_name: str = None,
    seed: int = None,
    extra_args: dict = None,
):
    """
    Run a training experiment.

    Args:
        experiment_name: Name of the experiment config (without .yaml)
        model_name: Override model name from config
        seed: Override seed from config
        extra_args: Additional YOLO training arguments
    """
    # Load config
    config = load_experiment_config(experiment_name)
    exp_config = config.get("experiment", {})
    train_config = config.get("training", {})
    aug_config = config.get("augmentation", {})

    # Override with CLI args
    if model_name:
        train_config["model"] = model_name
    if seed is not None:
        train_config["seed"] = seed

    current_seed = train_config.get("seed", 42)
    set_seed(current_seed)

    is_resume = False
    if model_name and str(model_name).endswith("last.pt"):
        is_resume = True
        print(f"Resume training from checkpoint: {model_name}")

    # Setup paths
    data_yaml = PROCESSED_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Data config not found: {data_yaml}\n"
            "Run 'python -m src.data.prepare_data' first."
        )

    # Experiment naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_name}_seed{current_seed}_{timestamp}"
    project_dir = str(RUNS_DIR / experiment_name)

    print("=" * 60)
    print(f"  EXPERIMENT: {exp_config.get('name', experiment_name)}")
    print(f"  Description: {exp_config.get('description', 'N/A')}")
    print(f"  Model: {train_config.get('model', 'yolov8n.pt')}")
    print(f"  Seed: {current_seed}")
    print(f"  Run name: {run_name}")
    print(f"  Data: {data_yaml}")
    print("=" * 60)

    # Initialize model
    model_path = train_config.get("model", "yolov8n.pt")
    model = YOLO(model_path)

    # Build training arguments
    train_args = {
        "data": str(data_yaml),
        "epochs": train_config.get("epochs", 100),
        "imgsz": train_config.get("imgsz", 640),
        "batch": train_config.get("batch", 16),
        "patience": train_config.get("patience", 20),
        "seed": current_seed,
        "project": project_dir,
        "name": run_name,
        "single_cls": train_config.get("single_cls", True),
        "resume": is_resume, 
        "verbose": True,
        "save": True,
        "plots": True,
    }

    # Optimizer settings
    if "optimizer" in train_config:
        train_args["optimizer"] = train_config["optimizer"]
    if "lr0" in train_config:
        train_args["lr0"] = train_config["lr0"]
    if "lrf" in train_config:
        train_args["lrf"] = train_config["lrf"]
    if "weight_decay" in train_config:
        train_args["weight_decay"] = train_config["weight_decay"]

    # Augmentation settings
    if aug_config:
        aug_mapping = {
            "hsv_h": "hsv_h",
            "hsv_s": "hsv_s",
            "hsv_v": "hsv_v",
            "degrees": "degrees",
            "translate": "translate",
            "scale": "scale",
            "shear": "shear",
            "perspective": "perspective",
            "flipud": "flipud",
            "fliplr": "fliplr",
            "mosaic": "mosaic",
            "mixup": "mixup",
            "copy_paste": "copy_paste",
            "erasing": "erasing",
        }
        for key, yolo_key in aug_mapping.items():
            if key in aug_config:
                train_args[yolo_key] = aug_config[key]

    # Extra override args
    if extra_args:
        train_args.update(extra_args)

    # Train
    print(f"\n🚀 Starting training...")
    start_time = time.time()

    results = model.train(**train_args)

    elapsed = time.time() - start_time
    print(f"\n⏱ Training completed in {elapsed / 60:.1f} minutes")

    # Validate on test set
    print(f"\n📊 Evaluating on test set...")
    test_metrics = model.val(
        data=str(data_yaml),
        split="test",
        project=project_dir,
        name=f"{run_name}_test",
    )

    # Save experiment summary
    summary = {
        "experiment": experiment_name,
        "config": config,
        "seed": current_seed,
        "training_time_minutes": round(elapsed / 60, 2),
        "run_name": run_name,
        "model_path": str(model_path),
        "results_dir": str(Path(project_dir) / run_name),
    }

    summary_path = Path(project_dir) / run_name / "experiment_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(f"\n✅ Experiment complete!")
    print(f"   Results: {Path(project_dir) / run_name}")
    print(f"   Summary: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 training experiment")
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Experiment config name (e.g., e0_baseline)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override model (e.g., yolov8s.pt)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override batch size",
    )
    args = parser.parse_args()

    extra_args = {}
    if args.epochs:
        extra_args["epochs"] = args.epochs
    if args.batch:
        extra_args["batch"] = args.batch

    train_experiment(
        experiment_name=args.experiment,
        model_name=args.model,
        seed=args.seed,
        extra_args=extra_args if extra_args else None,
    )


if __name__ == "__main__":
    main()
