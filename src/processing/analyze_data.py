"""
Dataset Analysis & Statistics
==============================
Generates comprehensive statistics about the processed (or raw) dataset.

Usage:
    python -m src.data.analyze_data [--data-dir data/processed]
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np

from src.utils.helpers import (
    PROJECT_ROOT, PROCESSED_DIR, RAW_DIR,
    parse_yolo_line, ROBOFLOW_CLASSES, READING_CLASS_ID,
)


def analyze_split(split_dir: Path) -> dict:
    """Analyze a single data split."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if not img_dir.exists():
        return None

    stats = {
        "num_images": 0,
        "num_labels": 0,
        "num_annotations": 0,
        "class_counts": Counter(),
        "annotations_per_image": [],
        "bbox_sizes": [],  # (width, height) in normalized coords
        "image_sizes": [],  # (w, h) in pixels
        "images_without_annotations": 0,
        "polygon_count": 0,
    }

    image_files = sorted(img_dir.glob("*"))
    for img_path in image_files:
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue

        stats["num_images"] += 1

        # Read image dimensions
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            stats["image_sizes"].append((w, h))

        # Read label
        label_path = lbl_dir / (img_path.stem + ".txt")
        if label_path.exists():
            stats["num_labels"] += 1
            lines = label_path.read_text().strip().splitlines()
            ann_count = 0
            for line in lines:
                parsed = parse_yolo_line(line)
                if parsed:
                    ann_count += 1
                    stats["class_counts"][parsed["class_id"]] += 1
                    stats["bbox_sizes"].append(
                        (parsed["bbox"][2], parsed["bbox"][3])
                    )
                    if parsed["is_polygon"]:
                        stats["polygon_count"] += 1

            stats["num_annotations"] += ann_count
            stats["annotations_per_image"].append(ann_count)

            if ann_count == 0:
                stats["images_without_annotations"] += 1
        else:
            stats["annotations_per_image"].append(0)
            stats["images_without_annotations"] += 1

    return stats


def print_report(all_stats: dict, data_dir: Path):
    """Print a comprehensive analysis report."""
    print("\n" + "=" * 70)
    print("  DATASET ANALYSIS REPORT")
    print(f"  Source: {data_dir}")
    print("=" * 70)

    total_images = 0
    total_annotations = 0

    for split_name, stats in all_stats.items():
        if stats is None:
            continue

        print(f"\n{'─' * 50}")
        print(f"  Split: {split_name.upper()}")
        print(f"{'─' * 50}")
        print(f"  Images:              {stats['num_images']}")
        print(f"  Labels:              {stats['num_labels']}")
        print(f"  Total annotations:   {stats['num_annotations']}")
        print(f"  No annotations:      {stats['images_without_annotations']}")

        total_images += stats["num_images"]
        total_annotations += stats["num_annotations"]

        if stats["annotations_per_image"]:
            api = stats["annotations_per_image"]
            print(f"  Annotations/image:   "
                  f"mean={np.mean(api):.1f}, "
                  f"min={min(api)}, max={max(api)}")

        if stats["bbox_sizes"]:
            ws = [s[0] for s in stats["bbox_sizes"]]
            hs = [s[1] for s in stats["bbox_sizes"]]
            print(f"  BBox width (norm):   "
                  f"mean={np.mean(ws):.3f}, "
                  f"min={min(ws):.3f}, max={max(ws):.3f}")
            print(f"  BBox height (norm):  "
                  f"mean={np.mean(hs):.3f}, "
                  f"min={min(hs):.3f}, max={max(hs):.3f}")

        if stats["image_sizes"]:
            iw = [s[0] for s in stats["image_sizes"]]
            ih = [s[1] for s in stats["image_sizes"]]
            unique_sizes = set(stats["image_sizes"])
            print(f"  Image resolutions:   {len(unique_sizes)} unique")
            for sz in sorted(unique_sizes):
                count = stats["image_sizes"].count(sz)
                print(f"    {sz[0]}x{sz[1]}: {count} images")

        if stats["class_counts"]:
            print(f"  Class distribution:")
            for cls_id in sorted(stats["class_counts"]):
                count = stats["class_counts"][cls_id]
                pct = count / stats["num_annotations"] * 100 if stats["num_annotations"] else 0
                name = ROBOFLOW_CLASSES.get(cls_id, f"class_{cls_id}")
                print(f"    [{cls_id}] {name}: {count} ({pct:.1f}%)")

        if stats["polygon_count"] > 0:
            print(f"  Polygon annotations: {stats['polygon_count']}")

    print(f"\n{'=' * 70}")
    print(f"  TOTALS: {total_images} images, {total_annotations} annotations")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset statistics")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (default: data/processed, fallback: data/raw)",
    )
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif PROCESSED_DIR.exists() and any(PROCESSED_DIR.iterdir()):
        data_dir = PROCESSED_DIR
    else:
        data_dir = RAW_DIR

    # Analyze each split
    splits = ["train", "valid", "test"]
    all_stats = {}
    for split in splits:
        split_dir = data_dir / split
        print(f"📊 Analyzing '{split}'...")
        all_stats[split] = analyze_split(split_dir)

    print_report(all_stats, data_dir)


if __name__ == "__main__":
    main()
