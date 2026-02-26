"""
Giai đoạn 4: Inference & Crop ảnh (OBB)
=========================================
Load model OBB đã train, chạy dự đoán, vẽ bounding box NGHIÊNG (OBB)
và cắt crop ảnh thẳng (axis-aligned crop) để dùng cho OCR.

Điều kiện tiên quyết:
  - Đã chạy xong 2_train_yolo.py -> best.pt đã có trong:
    runs/detect/meter_model/weights/best.pt

Chạy:
  # Thử nhanh 1 ảnh
  python 3_inference_test.py

  # Batch toàn bộ tập test
  python 3_inference_test.py --batch

  # Chỉ định ảnh khác
  python 3_inference_test.py --img ./Data/test/images/test_0005.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO


# ─── Cấu hình ────────────────────────────────────────────────────────────────
WEIGHTS      = "runs/detect/meter_model/weights/best.pt"
TEST_IMG_DIR = "./Data/test/images"
CROP_OUT_DIR = "./crops"          # Thư mục lưu ảnh cắt
CONF_THRESH  = 0.25               # Ngưỡng confidence tối thiểu
# ─────────────────────────────────────────────────────────────────────────────


def predict_single(model: YOLO, img_path: str, save_annotated: bool = True) -> None:
    """Dự đoán 1 ảnh, vẽ OBB polygon, cắt axis-aligned crop, lưu file."""
    if not os.path.isfile(img_path):
        print(f"[ERROR] Không tìm thấy ảnh: {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] cv2 không đọc được ảnh: {img_path}")
        return

    results = model.predict(img_path, conf=CONF_THRESH, verbose=False)
    os.makedirs(CROP_OUT_DIR, exist_ok=True)

    basename = os.path.splitext(os.path.basename(img_path))[0]
    crop_count = 0

    for result in results:
        # OBB result: result.obb.xyxyxyxy  ->  (N, 4, 2)  float32 tensor
        if result.obb is None or len(result.obb) == 0:
            print(f"  [INFO] Không phát hiện đối tượng nào trong: {img_path}")
            continue

        # Tọa độ 4 đỉnh polygon (pixel)
        polys  = result.obb.xyxyxyxy.cpu().numpy()   # (N, 4, 2)
        confs  = result.obb.conf.cpu().numpy()        # (N,)
        clsids = result.obb.cls.cpu().numpy().astype(int)  # (N,)

        class_names = result.names  # dict {id: name}

        for i, (poly, conf, cls_id) in enumerate(zip(polys, confs, clsids)):
            cls_name = class_names.get(cls_id, str(cls_id))
            label    = f"{cls_name} {conf:.2f}"

            # ── 1. Vẽ OBB polygon lên ảnh gốc ──────────────────────────────
            pts = poly.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            x_text = int(poly[:, 0].min())
            y_text = int(poly[:, 1].min()) - 8
            cv2.putText(img, label, (x_text, max(y_text, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ── 2. Cắt axis-aligned crop (dùng cho OCR) ─────────────────────
            x1 = max(0, int(poly[:, 0].min()))
            y1 = max(0, int(poly[:, 1].min()))
            x2 = min(img.shape[1], int(poly[:, 0].max()))
            y2 = min(img.shape[0], int(poly[:, 1].max()))

            crop = img[y1:y2, x1:x2]
            crop_path = os.path.join(CROP_OUT_DIR, f"{basename}_crop_{crop_count:02d}.jpg")
            cv2.imwrite(crop_path, crop)
            print(f"  [CROP] {cls_name} ({conf:.2f}) -> lưu tại: {crop_path}")
            crop_count += 1

    # ── 3. Lưu ảnh annotated ────────────────────────────────────────────────
    if save_annotated:
        out_path = os.path.join(CROP_OUT_DIR, f"{basename}_annotated.jpg")
        cv2.imwrite(out_path, img)
        print(f"  [SAVE] Ảnh annotated: {out_path}")

    if crop_count == 0:
        print(f"  [WARN] Không cắt được crop nào. Kiểm tra lại CONF_THRESH={CONF_THRESH}")
    else:
        print(f"  [OK]   Tổng {crop_count} crop đã lưu vào: {CROP_OUT_DIR}/")


def predict_batch(model: YOLO) -> None:
    """Chạy inference trên toàn bộ thư mục test."""
    img_exts = (".jpg", ".jpeg", ".png", ".bmp")
    imgs = [
        os.path.join(TEST_IMG_DIR, f)
        for f in sorted(os.listdir(TEST_IMG_DIR))
        if f.lower().endswith(img_exts)
    ]

    if not imgs:
        print(f"[ERROR] Không tìm thấy ảnh nào trong: {TEST_IMG_DIR}")
        return

    print(f"[BATCH] Sẽ xử lý {len(imgs)} ảnh từ: {TEST_IMG_DIR}\n")
    for idx, img_path in enumerate(imgs, 1):
        print(f"[{idx:03d}/{len(imgs)}] {os.path.basename(img_path)}")
        predict_single(model, img_path, save_annotated=True)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference YOLOv8-OBB + crop cho OCR")
    parser.add_argument("--img",   type=str, default=None,  help="Đường dẫn 1 ảnh cụ thể")
    parser.add_argument("--batch", action="store_true",      help="Chạy batch toàn bộ tập test")
    parser.add_argument("--weights", type=str, default=WEIGHTS, help="Đường dẫn file .pt")
    args = parser.parse_args()

    # Kiểm tra weights
    if not os.path.isfile(args.weights):
        print(f"[ERROR] Không tìm thấy weights: {args.weights}")
        print("        Hãy chạy 2_train_yolo.py trước!")
        sys.exit(1)

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    if args.batch:
        predict_batch(model)
    else:
        # Mặc định: lấy ảnh test đầu tiên
        img_path = args.img
        if img_path is None:
            imgs = sorted(
                f for f in os.listdir(TEST_IMG_DIR)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            if not imgs:
                print(f"[ERROR] Không có ảnh nào trong {TEST_IMG_DIR}")
                sys.exit(1)
            img_path = os.path.join(TEST_IMG_DIR, imgs[0])
            print(f"[AUTO] Dùng ảnh đầu tiên: {img_path}")

        predict_single(model, img_path)

    print("\nBước tiếp theo: đưa ảnh crop vào PaddleOCR hoặc EasyOCR để đọc số!")


if __name__ == "__main__":
    main()
