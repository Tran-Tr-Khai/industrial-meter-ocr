"""
Utility helpers for the industrial meter OCR project.
"""

import os
import random
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIGS_DIR = PROJECT_ROOT / "configs"


# ──────────────────────────────────────────────
# Original (broken) Roboflow class names
# ──────────────────────────────────────────────
ROBOFLOW_CLASSES = {
    0: "collaborate_text",
    1: "understand_text",
    2: "active_learning_text",
    3: "separator",
    4: "electricity_meter_v1",
    5: "Reading",          # ← the only meaningful class
    6: "roboflow_text",
    7: "dataset_101_text",
    8: "meter_v1",
    9: "github_link_text",
}

# Class 5 = "Reading" is the kWh display region
READING_CLASS_ID = 5


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def polygon_to_bbox(coords: list[float]) -> tuple[float, float, float, float]:
    """
    Convert polygon coordinates to bounding box (YOLO format).

    Args:
        coords: list of [x1, y1, x2, y2, ..., xn, yn]

    Returns:
        (x_center, y_center, width, height) in normalized coords
    """
    xs = coords[0::2]
    ys = coords[1::2]

    x_min = max(0.0, min(xs))
    x_max = min(1.0, max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(1.0, max(ys))

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return (x_center, y_center, width, height)


def parse_yolo_line(line: str) -> dict:
    """
    Parse a single YOLO annotation line.
    Handles both bbox format (5 values) and polygon format (>5 values).

    Returns:
        dict with keys: class_id, bbox (x_center, y_center, w, h), is_polygon
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]

    if len(coords) == 4:
        # Standard bbox: x_center, y_center, width, height
        bbox = tuple(coords)
        is_polygon = False
    else:
        # Polygon: convert to bbox
        bbox = polygon_to_bbox(coords)
        is_polygon = True

    return {
        "class_id": class_id,
        "bbox": bbox,
        "is_polygon": is_polygon,
    }


def format_yolo_line(class_id: int, bbox: tuple) -> str:
    """Format a YOLO bbox annotation line."""
    x, y, w, h = bbox
    return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
