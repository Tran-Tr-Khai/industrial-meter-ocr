"""
Giai đoạn 3: Training YOLOv8-OBB Model
========================================
Dataset này dùng định dạng OBB (Oriented Bounding Box) — mỗi nhãn có
9 giá trị: class x1 y1 x2 y2 x3 y3 x4 y4

Do đó PHẢI dùng model: yolov8n-obb.pt  (không phải yolov8n.pt thông thường)

Điều kiện tiên quyết:
  1. Đã chạy:  python 1_clean_data.py
  2. Đã kiểm tra đường dẫn trong Data/data.yaml

Chạy: python 2_train_yolo.py
"""

from ultralytics import YOLO


def train() -> None:
    # ─── OBB model ────────────────────────────────────────────────────────────
    # yolov8n-obb : Nano — nhanh nhất, phù hợp PoC / máy không có GPU mạnh
    # yolov8s-obb : Small — tốt hơn ~10-15 % mAP nếu có GPU >= 6 GB
    model = YOLO("yolov8n-obb.pt")

    print("=" * 60)
    print("  Bắt đầu training YOLOv8-OBB...")
    print("  Tiến trình được lưu tại: runs/detect/meter_model/")
    print("=" * 60)

    results = model.train(
        data    = "./Data/data.yaml",   # File cấu hình dataset
        epochs  = 50,                   # 50 epoch là đủ cho PoC
        imgsz   = 640,                  # Kích thước ảnh chuẩn
        batch   = 16,                   # Giảm xuống 8 nếu Out-of-Memory

        # ── Tối ưu ──────────────────────────────────────────────────────────
        optimizer = "AdamW",            # AdamW thường hội tụ nhanh hơn SGD
        lr0       = 0.001,              # Learning rate ban đầu
        weight_decay = 0.0005,

        # ── Augmentation nhẹ (an toàn cho OBB) ──────────────────────────────
        degrees   = 10,                 # Xoay ảnh ±10°
        fliplr    = 0.5,                # Lật ngang 50%
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,

        # ── Output ──────────────────────────────────────────────────────────
        project   = "runs/detect",      # Thư mục lưu kết quả
        name      = "meter_model",      # Tên folder con
        exist_ok  = True,               # Ghi đè nếu đã tồn tại
        plots     = True,               # Vẽ biểu đồ loss & mAP
        save      = True,
    )

    print("\n" + "=" * 60)
    print("  Training hoàn tất!")
    print(f"  Best weights: runs/detect/meter_model/weights/best.pt")
    print(f"  mAP50 cuối:   {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print("=" * 60)
    print("\n  Dấu hiệu thành công:")
    print("  - box_loss  đi XUỐNG đều qua các epoch")
    print("  - mAP50     đi LÊN, tiệm cận 0.9 trở lên")
    print("\n  Bước tiếp theo: python 3_inference_test.py")


if __name__ == "__main__":
    train()
