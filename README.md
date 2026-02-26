# Electric Meter Detection — YOLOv8 Pipeline

> Phát hiện vùng hiển thị số của đồng hồ điện trong ảnh thực tế,
> sử dụng YOLOv8 (single-class) và chuẩn bị crop ảnh cho bước OCR.

---

## Mục lục

1. [Giới thiệu Dataset](#1-giới-thiệu-dataset)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Pipeline 3 bước](#3-pipeline-3-bước)
   - [Bước 1 — Data Cleaning](#bước-1--data-cleaning--standardization)
   - [Bước 2 — Training](#bước-2--training)
   - [Bước 3 — Inference & Crop](#bước-3--inference--crop)
4. [Yêu cầu môi trường](#4-yêu-cầu-môi-trường)
5. [Kết quả mong đợi](#5-kết-quả-mong-đợi)
6. [Bước tiếp theo — OCR](#6-bước-tiếp-theo--ocr)

---

## 1. Giới thiệu Dataset

| Thuộc tính        | Chi tiết                                                                 |
|-------------------|--------------------------------------------------------------------------|
| **Tên**           | Electric Meter — v3                                                      |
| **Nguồn**         | [Roboflow Universe](https://universe.roboflow.com/wattwise/electric-meter-wzeeg-zk5fd) |
| **License**       | CC BY 4.0                                                                |
| **Tổng ảnh**      | 712 ảnh                                                                  |
| **Format nhãn**   | YOLOv8 (polygon / bbox)                                                  |
| **Số class**      | 1 — `display` (vùng hiển thị số đồng hồ)                                |

### Phân chia tập dữ liệu

| Split     | Số ảnh | Tỉ lệ |
|-----------|-------:|------:|
| **train** |   621  | 87 %  |
| **valid** |    65  |  9 %  |
| **test**  |    26  |  4 %  |
| **Tổng**  |   712  | 100 % |

### Vấn đề tên file gốc

File ảnh và nhãn có tên lộn xộn từ Roboflow (ví dụ:
`10_JPG.rf.944bef43184d2204dfd2188f06283f64.jpg`).
Dù model vẫn học được, khi debug cực kỳ khó tra cứu.
→ **Bước 1** của pipeline giải quyết vấn đề này.

---

## 2. Cấu trúc thư mục

```
industrial_meter_ocr/
│
├── Data/
│   ├── data.yaml               ← Cấu hình dataset cho YOLO (nc=1)
│   ├── train/
│   │   ├── images/             ← Ảnh training (621 files)
│   │   └── labels/             ← Nhãn tương ứng (.txt)
│   ├── valid/
│   │   ├── images/             ← Ảnh validation (65 files)
│   │   └── labels/
│   └── test/
│       ├── images/             ← Ảnh test (26 files)
│       └── labels/
│
├── 1_clean_data.py             ← Bước 1: Chuẩn hóa tên file + remap class ID
├── 2_train_yolo.py             ← Bước 2: Training YOLOv8 single-class
├── 3_inference_test.py         ← Bước 3: Inference + crop OCR
│
├── runs/
│   └── detect/
│       └── meter_model/
│           ├── weights/
│           │   ├── best.pt     ← Model tốt nhất (dùng cho inference)
│           │   └── last.pt
│           └── results.png     ← Biểu đồ loss & mAP
│
└── crops/                      ← Ảnh crop đầu ra (input cho OCR)
```

---

## 3. Pipeline 3 bước

### Bước 1 — Data Cleaning & Standardization

**Script:** [1_clean_data.py](1_clean_data.py)

Thực hiện 2 việc tự động:

**1a. Đổi tên file** sang định dạng chuẩn:

```
Trước:  10_JPG.rf.944bef43184d2204dfd2188f06283f64.jpg / .txt
Sau:    train_0000.jpg  ←→  train_0000.txt
        train_0001.jpg  ←→  train_0001.txt
        ...
        valid_0000.jpg  ←→  valid_0000.txt
        test_0000.jpg   ←→  test_0000.txt
```

**1b. Remap class ID → 0** trong mọi file nhãn:

Dataset gốc dùng nhiều class ID khác nhau. Vì `data.yaml` khai báo `nc: 1`,
mọi annotation phải là class `0` (`display`). Script tự động fix việc này.

**Quy tắc an toàn của script:**
- Ảnh không có file nhãn tương ứng → **bỏ qua**, in cảnh báo.
- Luôn chạy `--dry-run` trước để xem preview.
- Thứ tự đổi tên theo `sorted()` → nhất quán mỗi lần chạy.

```bash
# Xem trước (không sửa gì)
python 1_clean_data.py --dry-run

# Thực sự chuẩn hóa
python 1_clean_data.py
```

---

### Bước 2 — Training

**Script:** [2_train_yolo.py](2_train_yolo.py)

```bash
python 2_train_yolo.py
```

| Tham số        | Giá trị      | Lý do chọn                                          |
|----------------|--------------|-----------------------------------------------------|
| `model`        | `yolov8n.pt` | Nano — nhanh, nhẹ, phù hợp PoC (E0 Baseline)       |
| `epochs`       | 100          | E0 Baseline theo plan.md                            |
| `imgsz`        | 640          | Chuẩn YOLO, cân bằng tốc độ và độ chính xác        |
| `batch`        | 16           | Giảm xuống 8 nếu Out-of-Memory                      |
| `single_cls`   | `True`       | Gộp mọi class thành 1 — tránh lỗi label index      |
| `optimizer`    | AdamW        | Hội tụ nhanh hơn SGD trên dataset nhỏ              |

**Dấu hiệu training thành công** (xem `runs/detect/meter_model/results.png`):

```
box_loss   ↓↓↓  (giảm đều, không dao động mạnh)
cls_loss   ↓↓↓
mAP50      ↑↑↑  (tăng, tiệm cận ≥ 0.85+)
```

Thời gian ước tính: **10–15 phút** (GPU) / **45–60 phút** (CPU).

---

### Bước 3 — Inference & Crop

**Script:** [3_inference_test.py](3_inference_test.py)

```bash
# Test nhanh 1 ảnh đầu tiên trong tập test
python 3_inference_test.py

# Chỉ định ảnh cụ thể
python 3_inference_test.py --img ./Data/test/images/test_0005.jpg

# Batch toàn bộ tập test
python 3_inference_test.py --batch
```

Script thực hiện 3 việc cho mỗi ảnh:

1. **Vẽ bounding box** màu xanh lá lên ảnh gốc.
2. **Cắt crop** theo bounding box → lưu vào `crops/`.
3. **Lưu ảnh annotated** để kiểm tra trực quan.

```
Input:  Data/test/images/test_0000.jpg
Output:
  crops/test_0000_crop_00.jpg    ← Crop sẵn sàng cho OCR
  crops/test_0000_annotated.jpg  ← Ảnh debug có khung bbox
```

---

## 4. Yêu cầu môi trường

```bash
pip install ultralytics opencv-python
```

| Package         | Phiên bản tối thiểu | Ghi chú                         |
|-----------------|---------------------|---------------------------------|
| `ultralytics`   | ≥ 8.0               | Bao gồm YOLOv8 support          |
| `opencv-python` | ≥ 4.5               | Đọc/ghi/vẽ ảnh                 |
| `torch`         | ≥ 2.0               | Tự cài khi install ultralytics  |
| Python          | ≥ 3.9               |                                 |

**Môi trường đề xuất:** WSL2 Ubuntu trên Windows với CUDA GPU.

---

## 5. Kết quả mong đợi

Sau khi hoàn thành pipeline, thư mục `crops/` chứa các ảnh như:

```
crops/
├── test_0000_crop_00.jpg     ← Vùng màn hình đồng hồ đã cắt
├── test_0000_annotated.jpg   ← Ảnh gốc có khung bbox
├── test_0001_crop_00.jpg
└── ...
```

`test_0000_crop_00.jpg` sẽ là ảnh chỉ chứa **màn hình số của đồng hồ điện** — input lý tưởng cho bước OCR.

---

## 6. Bước tiếp theo — OCR

Đưa ảnh crop vào một trong hai thư viện OCR sau để đọc ra chỉ số:

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

