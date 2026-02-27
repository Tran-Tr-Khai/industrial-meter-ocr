# Electric Meter Detection — YOLOv8 Pipeline

> Phát hiện vùng hiển thị số của đồng hồ điện trong ảnh thực tế,
> sử dụng YOLOv8 (single-class) và chuẩn bị crop ảnh cho bước OCR.

---

## Mục lục

1. [Giới thiệu Dataset](#1-giới-thiệu-dataset)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Research Workflow](#3-research-workflow)
4. [Yêu cầu môi trường](#4-yêu-cầu-môi-trường)
5. [Kết quả E0 Baseline](#5-kết-quả-e0-baseline)
6. [Bảng Experiments](#6-bảng-experiments)
7. [Bước tiếp theo — OCR](#7-bước-tiếp-theo--ocr)

---

## 1. Giới thiệu Dataset

| Thuộc tính        | Chi tiết                                                                 |
|-------------------|--------------------------------------------------------------------------|
| **Tên**           | Electric Meter — v3                                                      |
| **Nguồn**         | [Roboflow Universe](https://universe.roboflow.com/wattwise/electric-meter-wzeeg-zk5fd) |
| **License**       | CC BY 4.0                                                                |
| **Tổng ảnh**      | 712 ảnh                                                                  |
| **Format nhãn**   | YOLOv8 (polygon → bbox, single class)                                    |
| **Số class**      | 1 — `display` (vùng hiển thị số đồng hồ)                                |
| **Thương hiệu**   | Genus, MicroTech Industries, General Electric, Bentex                    |

### Phân chia tập dữ liệu

| Split     | Số ảnh | Tỉ lệ |
|-----------|-------:|------:|
| **train** |   621  | 87 %  |
| **valid** |    65  |  9 %  |
| **test**  |    26  |  4 %  |
| **Tổng**  |   712  | 100 % |

> **Lưu ý:** Data raw tại `data/raw/` **không bao giờ bị ghi đè**.
> Script `prepare_data.py` chuẩn hóa polygon → bbox và remap class ID → 0,
> output tại `data/processed/`.

---

## 2. Cấu trúc thư mục

```
industrial_meter_ocr/
│
├── data/
│   ├── raw/                    ← Data gốc từ Roboflow (không sửa)
│   └── processed/
│       ├── data.yaml           ← Cấu hình dataset cho YOLO (nc=1)
│       ├── train/images+labels/
│       ├── valid/images+labels/
│       └── test/images+labels/
│
├── configs/                    ← YAML config cho từng experiment
│   ├── e0_baseline.yaml
│   ├── e1_model_capacity.yaml
│   ├── e2_single_vs_multi.yaml
│   ├── e3_augmentation.yaml
│   └── e4_orientation.yaml
│
├── src/
│   ├── data/
│   │   ├── prepare_data.py     ← Raw → processed (polygon→bbox, remap class)
│   │   └── analyze_data.py     ← Thống kê dataset
│   ├── training/
│   │   ├── train.py            ← Chạy từng experiment
│   │   └── run_seeds.py        ← Multi-seed cho statistical comparison
│   ├── evaluation/
│   │   ├── evaluate.py         ← So sánh metrics giữa experiments
│   │   └── error_analysis.py   ← Phân tích FP/FN với visualization
│   └── utils/
│       └── helpers.py          ← Shared utilities
│
├── runs/
│   └── e0_baseline/
│       └── e0_baseline_seed42_20260226_134709/
│           ├── weights/best.pt ← Model tốt nhất
│           ├── evaluation_metrics.yaml
│           ├── error_analysis/ ← Ảnh debug FP/FN
│           └── results.csv
│
├── docs/
│   ├── plan.md                 ← Research plan & experimental design
│   └── Summary.md              ← Tổng hợp phân tích về dataset
│
├── requirements.txt
└── yolov8n.pt                  ← Pretrained weights (base model)
```

---

## 3. Research Workflow

### Bước 1 — Chuẩn bị data

```bash
# Xử lý raw → processed (single class, polygon → bbox)
python -m src.data.prepare_data

# Xem thống kê dataset
python -m src.data.analyze_data
```

### Bước 2 — Chạy experiments

```bash
# E0: Baseline (ĐÃ HOÀN THÀNH)
python -m src.training.train --experiment e0_baseline

# E1: So sánh kích thước model
python -m src.training.train -e e1_model_capacity -m yolov8n.pt
python -m src.training.train -e e1_model_capacity -m yolov8s.pt
python -m src.training.train -e e1_model_capacity -m yolov8m.pt

# E3: Augmentation mạnh hơn
python -m src.training.train --experiment e3_augmentation

# E4: Orientation robustness (flipud + rotation)
python -m src.training.train --experiment e4_orientation

# Chạy multi-seed (3 seeds) để kiểm tra statistical significance
python -m src.training.run_seeds --experiment e0_baseline --seeds 42 123 456
```

### Bước 3 — Đánh giá

```bash
# So sánh tất cả experiments
python -m src.evaluation.evaluate

# Phân tích lỗi (FP/FN) của một run cụ thể
python -m src.evaluation.error_analysis --run-dir runs/e0_baseline/e0_baseline_seed42_20260226_134709
```

---

## 4. Yêu cầu môi trường

```bash
pip install -r requirements.txt
```

| Package         | Phiên bản tối thiểu | Ghi chú                         |
|-----------------|---------------------|---------------------------------|
| `ultralytics`   | ≥ 8.0               | Bao gồm YOLOv8 support          |
| `opencv-python` | ≥ 4.5               | Đọc/ghi/vẽ ảnh                 |
| `torch`         | ≥ 2.0               | Tự cài khi install ultralytics  |
| `pyyaml`        | ≥ 6.0               | Đọc/ghi config và metrics       |
| Python          | ≥ 3.9               |                                 |

**Môi trường đề xuất:** WSL2 Ubuntu trên Windows.

---

## 5. Kết quả E0 Baseline

**Run:** `e0_baseline_seed42_20260226_134709`

### Cấu hình training

| Tham số        | Giá trị      | Ghi chú                                             |
|----------------|--------------|-----------------------------------------------------|
| `model`        | `yolov8n.pt` | Nano — nhẹ, phù hợp baseline                       |
| `epochs`       | 100          | Kết thúc đủ 100 epoch                              |
| `imgsz`        | 640          | Chuẩn YOLO                                         |
| `batch`        | 16           |                                                     |
| `patience`     | 20           | Early stopping nếu không cải thiện                 |
| `single_cls`   | `True`       | Gộp mọi class thành 1                              |
| `optimizer`    | `auto`       | YOLO tự chọn (AdamW)                               |
| `seed`         | 42           |                                                     |
| `device`       | CPU          |                                                     |

### Metrics

| Split    | mAP50  | mAP50-95 | Precision | Recall |
|----------|-------:|----------:|----------:|-------:|
| **val**  | 0.884  | 0.6025    | 0.8555    | 0.8433 |
| **test** | 0.891  | 0.6377    | 0.9132    | 0.9048 |

- **Inference time:** ~38 ms/image (CPU)

### Error Analysis (tập test — 26 ảnh)

| Metric                         | Giá trị |
|-------------------------------|--------:|
| Total images                  | 26      |
| Perfect predictions           | 17      |
| True Positives (TP)           | 57      |
| False Positives (FP)          | 10      |
| False Negatives (FN)          | 6       |
| F1-score (conf=0.25, IoU=0.5) | 0.877   |

**FP chủ yếu:** logo/nhãn hiệu bị nhận nhầm là màn hình (ảnh 10, 32, Mtr32, Mtr35, Mtr40, img17).  
**FN chủ yếu:** màn hình bị lật ngược hoặc bị che khuất (ảnh 75, img12, img37).

### Model weights

```
runs/e0_baseline/e0_baseline_seed42_20260226_134709/weights/best.pt
```

---

## 6. Bảng Experiments

| ID  | Config                    | Câu hỏi nghiên cứu                        | Trạng thái         |
|-----|---------------------------|-------------------------------------------|--------------------|
| E0  | `e0_baseline.yaml`        | YOLOv8n baseline — mốc so sánh           | ✅ Hoàn thành       |
| E1  | `e1_model_capacity.yaml`  | n vs s vs m — trade-off accuracy/speed   | ⏳ Chưa chạy        |
| E2  | `e2_single_vs_multi.yaml` | single_cls vs multi-class                 | ⏳ Chưa chạy        |
| E3  | `e3_augmentation.yaml`    | Lighting robustness (HSV/rotation/mixup)  | ⏳ Chưa chạy        |
| E4  | `e4_orientation.yaml`     | Rotation aug đủ không, hay cần OBB?      | ⏳ Chưa chạy        |

---

## 7. Bước tiếp theo — OCR

Sau khi có `best.pt`, chạy inference để crop vùng màn hình, sau đó đưa qua OCR:

```python
# Ví dụ với PaddleOCR
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')
result = ocr.ocr('crops/test_0000_crop_00.jpg', cls=True)

for line in result[0]:
    print(line[1][0])  # In ra chuỗi số đọc được
```

| Thư viện    | Ưu điểm                              | Nhược điểm                    |
|-------------|--------------------------------------|-------------------------------|
| PaddleOCR   | Chính xác cao, hỗ trợ góc nghiêng   | Nặng hơn, cài phức tạp hơn   |
| EasyOCR     | Cài dễ (`pip install easyocr`)       | Chậm hơn trên CPU             |

---

*Dataset nguồn: [Roboflow Universe — WattWise](https://universe.roboflow.com/wattwise/electric-meter-wzeeg-zk5fd) | License: CC BY 4.0*

