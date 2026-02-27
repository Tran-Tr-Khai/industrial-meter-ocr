"""
Error Analysis
===============
Detailed analysis of model predictions: false positives, false negatives,
and failure patterns.

Usage:
    python -m src.evaluation.error_analysis --run-dir runs/e0_baseline/run_name
"""

import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO

from src.utils.helpers import (
    PROCESSED_DIR, ensure_dir, parse_yolo_line,
)


def load_ground_truth(label_path: Path) -> list[dict]:
    """Load ground truth bounding boxes from a label file."""
    if not label_path.exists():
        return []

    bboxes = []
    lines = label_path.read_text().strip().splitlines()
    for line in lines:
        parsed = parse_yolo_line(line)
        if parsed:
            bboxes.append(parsed)
    return bboxes


def compute_iou(box1: tuple, box2: tuple) -> float:
    """
    Compute IoU between two YOLO-format boxes (x_center, y_center, w, h).
    """
    x1_c, y1_c, w1, h1 = box1
    x2_c, y2_c, w2, h2 = box2

    x1_min, x1_max = x1_c - w1 / 2, x1_c + w1 / 2
    y1_min, y1_max = y1_c - h1 / 2, y1_c + h1 / 2
    x2_min, x2_max = x2_c - w2 / 2, x2_c + w2 / 2
    y2_min, y2_max = y2_c - h2 / 2, y2_c + h2 / 2

    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = w1 * h1 + w2 * h2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def analyze_predictions(
    run_dir: Path,
    data_dir: Path = None,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
):
    """
    Analyze predictions vs ground truth for error patterns.
    """
    if data_dir is None:
        data_dir = PROCESSED_DIR

    best_model = run_dir / "weights" / "best.pt"
    if not best_model.exists():
        print(f"❌ No model found: {best_model}")
        return

    model = YOLO(str(best_model))

    # Analyze test set
    test_img_dir = data_dir / "test" / "images"
    test_lbl_dir = data_dir / "test" / "labels"

    if not test_img_dir.exists():
        print("❌ Test images not found")
        return

    output_dir = ensure_dir(run_dir / "error_analysis")

    stats = {
        "total_images": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "fp_images": [],
        "fn_images": [],
        "perfect_images": [],
        "confidence_distribution": {
            "tp": [],
            "fp": [],
        },
    }

    image_files = sorted(test_img_dir.glob("*"))
    for img_path in image_files:
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue

        stats["total_images"] += 1

        # Ground truth
        gt_bboxes = load_ground_truth(test_lbl_dir / (img_path.stem + ".txt"))

        # Predictions
        results = model.predict(str(img_path), conf=conf_threshold, verbose=False)
        pred_boxes = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    # Convert xyxy to xywh normalized
                    xyxy = box.xyxy[0].cpu().numpy()
                    img = cv2.imread(str(img_path))
                    h, w = img.shape[:2]
                    x_center = ((xyxy[0] + xyxy[2]) / 2) / w
                    y_center = ((xyxy[1] + xyxy[3]) / 2) / h
                    bw = (xyxy[2] - xyxy[0]) / w
                    bh = (xyxy[3] - xyxy[1]) / h
                    conf = float(box.conf[0])
                    pred_boxes.append({
                        "bbox": (x_center, y_center, bw, bh),
                        "conf": conf,
                    })

        # Match predictions to ground truth
        gt_matched = [False] * len(gt_bboxes)
        pred_matched = [False] * len(pred_boxes)

        for pi, pred in enumerate(pred_boxes):
            best_iou = 0
            best_gi = -1
            for gi, gt in enumerate(gt_bboxes):
                if gt_matched[gi]:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= iou_threshold and best_gi >= 0:
                gt_matched[best_gi] = True
                pred_matched[pi] = True
                stats["true_positives"] += 1
                stats["confidence_distribution"]["tp"].append(pred["conf"])
            else:
                stats["false_positives"] += 1
                stats["confidence_distribution"]["fp"].append(pred["conf"])

        # Count false negatives (unmatched ground truth)
        fn_count = sum(1 for m in gt_matched if not m)
        stats["false_negatives"] += fn_count

        # Categorize image
        fp_count = sum(1 for m in pred_matched if not m)
        if fp_count > 0:
            stats["fp_images"].append(img_path.name)
        if fn_count > 0:
            stats["fn_images"].append(img_path.name)
        if fp_count == 0 and fn_count == 0 and len(gt_bboxes) > 0:
            stats["perfect_images"].append(img_path.name)

        # Save visualization for errors
        if fp_count > 0 or fn_count > 0:
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            # Draw GT in green
            for gi, gt in enumerate(gt_bboxes):
                x, y, bw, bh = gt["bbox"]
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                color = (0, 255, 0) if gt_matched[gi] else (0, 0, 255)
                label = "GT" if gt_matched[gi] else "FN"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw predictions
            for pi, pred in enumerate(pred_boxes):
                x, y, bw, bh = pred["bbox"]
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                if pred_matched[pi]:
                    color = (255, 200, 0)  # Cyan = TP
                    label = f"TP {pred['conf']:.2f}"
                else:
                    color = (0, 0, 255)  # Red = FP
                    label = f"FP {pred['conf']:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imwrite(str(output_dir / img_path.name), img)

    # ── Generate report ─────────────────────────────
    total_gt = stats["true_positives"] + stats["false_negatives"]
    total_pred = stats["true_positives"] + stats["false_positives"]
    precision = stats["true_positives"] / total_pred if total_pred else 0
    recall = stats["true_positives"] / total_gt if total_gt else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    report = {
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
        "total_images": stats["total_images"],
        "total_ground_truth": total_gt,
        "total_predictions": total_pred,
        "true_positives": stats["true_positives"],
        "false_positives": stats["false_positives"],
        "false_negatives": stats["false_negatives"],
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "perfect_images": len(stats["perfect_images"]),
        "fp_images": stats["fp_images"],
        "fn_images": stats["fn_images"],
    }

    if stats["confidence_distribution"]["tp"]:
        report["tp_conf_mean"] = round(np.mean(stats["confidence_distribution"]["tp"]), 4)
    if stats["confidence_distribution"]["fp"]:
        report["fp_conf_mean"] = round(np.mean(stats["confidence_distribution"]["fp"]), 4)

    # Save report
    report_path = output_dir / "error_analysis_report.yaml"
    with open(report_path, "w") as f:
        yaml.dump(report, f, default_flow_style=False)

    # Print report
    print("\n" + "=" * 60)
    print("  ERROR ANALYSIS REPORT")
    print("=" * 60)
    print(f"  IoU threshold:  {iou_threshold}")
    print(f"  Conf threshold: {conf_threshold}")
    print(f"  Total images:   {stats['total_images']}")
    print(f"  TP / FP / FN:   {stats['true_positives']} / {stats['false_positives']} / {stats['false_negatives']}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1:             {f1:.4f}")
    print(f"  Perfect images: {len(stats['perfect_images'])}/{stats['total_images']}")

    if stats["fp_images"]:
        print(f"\n  ⚠ False Positive images ({len(stats['fp_images'])}):")
        for name in stats["fp_images"][:10]:
            print(f"    - {name}")

    if stats["fn_images"]:
        print(f"\n  ⚠ False Negative images ({len(stats['fn_images'])}):")
        for name in stats["fn_images"][:10]:
            print(f"    - {name}")

    print(f"\n  📁 Error visualizations: {output_dir}")
    print(f"  📄 Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze model prediction errors")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the training run directory",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU threshold for matching (default: 0.5)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    args = parser.parse_args()

    analyze_predictions(
        run_dir=Path(args.run_dir),
        iou_threshold=args.iou,
        conf_threshold=args.conf,
    )


if __name__ == "__main__":
    main()
