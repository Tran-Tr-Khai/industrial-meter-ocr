"""
Data Preparation Pipeline
=========================
Processes raw Roboflow data into clean YOLO format for experiments.

Outputs are written to data/processed/ — raw data is never modified.

Usage:
    python -m src.data.prepare_data [--mode single_cls|multi_cls|reading_only]
"""

import argparse
import shutil
import yaml
from pathlib import Path
from collections import Counter

from src.utils.helpers import (
    PROJECT_ROOT, RAW_DIR, PROCESSED_DIR,
    READING_CLASS_ID, ROBOFLOW_CLASSES,
    parse_yolo_line, format_yolo_line, ensure_dir,
)


# ──────────────────────────────────────────────
# Processing modes
# ──────────────────────────────────────────────
MODE_SINGLE_CLS = "single_cls"      # All annotations → class 0
MODE_READING_ONLY = "reading_only"  # Keep only class 5 (Reading) → class 0
MODE_MULTI_CLS = "multi_cls"        # Keep all classes as-is (for E2 experiment)


def process_label_file(
    src_path: Path,
    dst_path: Path,
    mode: str,
    stats: dict,
) -> bool:
    """
    Process a single label file.

    Returns True if at least one annotation was written.
    """
    lines = src_path.read_text().strip().splitlines()

    output_lines = []
    for line in lines:
        parsed = parse_yolo_line(line)
        if parsed is None:
            stats["skipped_lines"] += 1
            continue

        cls_id = parsed["class_id"]
        bbox = parsed["bbox"]
        stats["total_annotations"] += 1
        stats["class_counts"][cls_id] += 1

        if parsed["is_polygon"]:
            stats["polygons_converted"] += 1

        if mode == MODE_SINGLE_CLS:
            # Everything becomes class 0
            output_lines.append(format_yolo_line(0, bbox))

        elif mode == MODE_READING_ONLY:
            # Only keep class 5 (Reading = kWh display)
            if cls_id == READING_CLASS_ID:
                output_lines.append(format_yolo_line(0, bbox))
            else:
                stats["filtered_out"] += 1

        elif mode == MODE_MULTI_CLS:
            # Keep original class IDs
            output_lines.append(format_yolo_line(cls_id, bbox))

    if output_lines:
        dst_path.write_text("\n".join(output_lines) + "\n")
        return True
    else:
        # Write empty file (image with no valid annotations)
        dst_path.write_text("")
        stats["empty_labels"] += 1
        return False


def process_split(split_name: str, mode: str, stats: dict) -> int:
    """
    Process a single data split (train/valid/test).

    Returns the number of images processed.
    """
    # Map directory names: raw uses 'valid', we keep it consistent
    raw_split_dir = RAW_DIR / split_name
    if not raw_split_dir.exists():
        # Try alternate name
        alt = "valid" if split_name == "val" else "val"
        raw_split_dir = RAW_DIR / alt
        if not raw_split_dir.exists():
            print(f"  ⚠ Split '{split_name}' not found, skipping.")
            return 0

    raw_img_dir = raw_split_dir / "images"
    raw_lbl_dir = raw_split_dir / "labels"

    proc_img_dir = ensure_dir(PROCESSED_DIR / split_name / "images")
    proc_lbl_dir = ensure_dir(PROCESSED_DIR / split_name / "labels")

    if not raw_img_dir.exists() or not raw_lbl_dir.exists():
        print(f"  ⚠ Images or labels missing for '{split_name}', skipping.")
        return 0

    count = 0
    image_files = sorted(raw_img_dir.glob("*"))
    for img_path in image_files:
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue

        # Find corresponding label
        label_name = img_path.stem + ".txt"
        label_path = raw_lbl_dir / label_name

        # Copy image
        dst_img = proc_img_dir / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # Process label
        dst_lbl = proc_lbl_dir / label_name
        if label_path.exists():
            process_label_file(label_path, dst_lbl, mode, stats)
        else:
            # No label file — write empty (background image)
            dst_lbl.write_text("")
            stats["missing_labels"] += 1

        count += 1

    return count


def generate_data_yaml(mode: str) -> Path:
    """Generate data.yaml for the processed dataset."""
    yaml_path = PROCESSED_DIR / "data.yaml"

    if mode == MODE_MULTI_CLS:
        nc = len(ROBOFLOW_CLASSES)
        names = list(ROBOFLOW_CLASSES.values())
    else:
        nc = 1
        names = ["reading"]

    data = {
        "path": str(PROCESSED_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": nc,
        "names": names,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare data for experiments")
    parser.add_argument(
        "--mode",
        choices=[MODE_SINGLE_CLS, MODE_READING_ONLY, MODE_MULTI_CLS],
        default=MODE_SINGLE_CLS,
        help=(
            f"Processing mode: "
            f"'{MODE_SINGLE_CLS}' = all classes → 0 (default), "
            f"'{MODE_READING_ONLY}' = only class 5 (Reading) → 0, "
            f"'{MODE_MULTI_CLS}' = keep all classes"
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing processed data before processing",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"  DATA PREPARATION PIPELINE")
    print(f"  Mode: {args.mode}")
    print(f"  Raw:  {RAW_DIR}")
    print(f"  Out:  {PROCESSED_DIR}")
    print("=" * 60)

    # Optionally clean
    if args.clean and PROCESSED_DIR.exists():
        print("\n🧹 Cleaning processed directory...")
        shutil.rmtree(PROCESSED_DIR)

    ensure_dir(PROCESSED_DIR)

    # Stats tracking
    stats = {
        "total_annotations": 0,
        "polygons_converted": 0,
        "filtered_out": 0,
        "skipped_lines": 0,
        "empty_labels": 0,
        "missing_labels": 0,
        "class_counts": Counter(),
    }

    # Process each split
    splits = ["train", "valid", "test"]
    for split in splits:
        print(f"\n📁 Processing '{split}' split...")
        n = process_split(split, args.mode, stats)
        print(f"   ✓ {n} images processed")

    # Generate data.yaml
    yaml_path = generate_data_yaml(args.mode)
    print(f"\n📄 Generated: {yaml_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  PROCESSING SUMMARY")
    print("=" * 60)
    print(f"  Total annotations read:   {stats['total_annotations']}")
    print(f"  Polygons → bbox:          {stats['polygons_converted']}")
    print(f"  Filtered out:             {stats['filtered_out']}")
    print(f"  Skipped (bad format):     {stats['skipped_lines']}")
    print(f"  Empty labels:             {stats['empty_labels']}")
    print(f"  Missing labels:           {stats['missing_labels']}")
    print(f"\n  Original class distribution:")
    for cls_id in sorted(stats["class_counts"]):
        name = ROBOFLOW_CLASSES.get(cls_id, f"unknown_{cls_id}")
        count = stats["class_counts"][cls_id]
        marker = " ← Reading (kWh display)" if cls_id == READING_CLASS_ID else ""
        print(f"    Class {cls_id} ({name}): {count}{marker}")

    print(f"\n✅ Done! Data saved to: {PROCESSED_DIR}")
    print(f"   YAML config: {yaml_path}")


if __name__ == "__main__":
    main()
