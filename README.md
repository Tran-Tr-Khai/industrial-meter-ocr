# Electric Meter Detection — YOLOv8-OBB Pipeline

> Phát hiện vùng hiển thị số của đồng hồ điện trong ảnh thực tế,
> sử dụng YOLOv8 Oriented Bounding Box (OBB) và chuẩn bị crop ảnh cho bước OCR.

---

## Mục lục

1. [Giới thiệu Dataset](#1-giới-thiệu-dataset)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Pipeline 4 bước](#3-pipeline-4-bước)
   - [Bước 1 — Data Cleaning](#bước-1--data-cleaning--standardization)
   - [Bước 2 — Cấu hình data.yaml](#bước-2--cấu-hình-datayaml)
   - [Bước 3 — Training](#bước-3--training)
   - [Bước 4 — Inference & Crop](#bước-4--inference--crop)
4. [Yêu cầu môi trường](#4-yêu-cầu-môi-trường)
5. [Kết quả mong đợi](#5-kết-quả-mong-đợi)
6. [Bước tiếp theo — OCR](#6-bước-tiếp-theo--ocr)

---

## 1. Giới thiệu Dataset

| Thuộc tính        | Chi tiết                                                                 |
|-------------------|--------------------------------------------------------------------------|
| **Tên**           | Electric Meter — v4                                                      |
| **Nguồn**         | [Roboflow Universe](https://universe.roboflow.com/wattwise/electric-meter-wzeeg-zk5fd) |
| **License**       | CC BY 4.0                                                                |
| **Export lúc**    | 12/02/2026                                                               |
| **Tổng ảnh**      | 1 813 ảnh                                                                |
| **Format nhãn**   | **YOLOv8 OBB** (Oriented Bounding Box)                                  |
| **Số class**      | 11 (class hữu ích là `reading` — index 5)                               |

### Phân chia tập dữ liệu

| Split     | Số ảnh | Tỉ lệ |
|-----------|-------:|------:|
| **train** | 1 522  | 84 %  |
| **test**  |   226  | 12 %  |
| **valid** |    65  |  4 %  |
| **Tổng**  | 1 813  | 100 % |

### Tại sao dùng OBB?

Đồng hồ điện thường được gắn nghiêng hoặc chụp từ góc xiên.
OBB cho phép bounding box **xoay theo góc thực** của vật thể,
giúp crop ảnh sát hơn và tăng độ chính xác OCR về sau.

```
Standard BBox (AABB)         OBB (Oriented)
┌─────────────────┐           ╱‾‾‾‾‾‾‾╲
│   ┌───────┐     │          ╱ đồng hồ ╲
│   │đồng hồ│     │          ╲         ╱
│   └───────┘     │           ╲_______╱
└─────────────────┘
  Crop thừa nhiều              Crop sát hơn
```

### Vấn đề tên file gốc

File ảnh và nhãn có tên lộn xộn từ Roboflow (ví dụ:
`235736161_238243881636164_..._jpg.rf.b38b1db1.jpg`).
Dù model vẫn học được, khi debug cực kỳ khó tra cứu.
→ **Bước 1** của pipeline sẽ giải quyết vấn đề này.

---

## 2. Cấu trúc thư mục

```
images_processing_pj/
│
├── Data/
│   ├── data.yaml               ← Cấu hình dataset cho YOLO
│   ├── train/
│   │   ├── images/             ← Ảnh training (1 522 files)
│   │   └── labels/             ← Nhãn OBB tương ứng (.txt)
│   ├── valid/
│   │   ├── images/             ← Ảnh validation (65 files)
│   │   └── labels/
│   └── test/
│       ├── images/             ← Ảnh test (226 files)
│       └── labels/
│
├── 1_clean_data.py             ← Bước 1: Chuẩn hóa tên file
├── 2_train_yolo.py             ← Bước 3: Training YOLOv8-OBB
├── 3_inference_test.py         ← Bước 4: Inference + crop OCR
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

### Format file nhãn OBB

Mỗi dòng trong file `.txt` có **9 giá trị**:

```
<class_id>  <x1> <y1>  <x2> <y2>  <x3> <y3>  <x4> <y4>
```

Tọa độ được chuẩn hóa về `[0, 1]` theo kích thước ảnh.
Ví dụ thực tế từ dataset:

```
5  0.0007 0.4294  0.8049 0.4480  0.7802 0.8692  0.0084 0.9016
```

---

## 3. Pipeline 4 bước

### Bước 1 — Data Cleaning & Standardization

**Script:** [1_clean_data.py](1_clean_data.py)

Đổi tên toàn bộ cặp `(ảnh, nhãn)` về định dạng chuẩn:

```
Trước:  235736161_238243881636164_..._jpg.rf.b38b1.jpg
        235736161_238243881636164_..._jpg.rf.b38b1.txt

Sau:    train_0000.jpg  ←→  train_0000.txt
        train_0001.jpg  ←→  train_0001.txt
        ...
        valid_0000.jpg  ←→  valid_0000.txt
        test_0000.jpg   ←→  test_0000.txt
```

**Quy tắc an toàn của script:**
- Ảnh không có file nhãn tương ứng → **bỏ qua**, in cảnh báo.
- Luôn chạy `--dry-run` trước để xem preview không thay đổi thật.
- Thứ tự đổi tên theo `sorted()` → kết quả nhất quán mỗi lần chạy.

```bash
# Xem trước (không sửa gì)
python 1_clean_data.py --dry-run

# Thực sự đổi tên
python 1_clean_data.py
```

---

### Bước 2 — Cấu hình data.yaml

**File:** [Data/data.yaml](Data/data.yaml)

Sau khi chạy bước 1, mở `Data/data.yaml` và **kiểm tra** trường `path`:

```yaml
path: /home/tntkhai/images_processing_pj/Data   # ← sửa nếu tên user khác

train: train/images
val:   valid/images
test:  test/images

nc: 11
names:
  5: reading        # Class chính — vùng hiển thị số đồng hồ
  ...
```

> **Lưu ý:** `nc` phải là `11` vì file nhãn dùng index `5`.
> Nếu đặt `nc: 1` sẽ bị lỗi `index out of range` ngay khi training.

---

### Bước 3 — Training

**Script:** [2_train_yolo.py](2_train_yolo.py)

```bash
python 2_train_yolo.py
```

| Tham số        | Giá trị  | Lý do chọn                                     |
|----------------|----------|------------------------------------------------|
| `model`        | `yolov8n-obb.pt` | Phải dùng **-obb** vì format nhãn là OBB |
| `epochs`       | 50       | Đủ để hội tụ cho dataset ~1 500 ảnh           |
| `imgsz`        | 640      | Chuẩn YOLO, cân bằng tốc độ và độ chính xác  |
| `batch`        | 16       | Giảm xuống 8 nếu Out-of-Memory                |
| `optimizer`    | AdamW    | Hội tụ nhanh hơn SGD trên dataset nhỏ        |

**Dấu hiệu training thành công** (xem `runs/detect/meter_model/results.png`):

```
box_loss   ↓↓↓  (giảm đều, không dao động mạnh)
cls_loss   ↓↓↓
mAP50      ↑↑↑  (tăng, tiệm cận ≥ 0.90)
```

Thời gian ước tính: **15–20 phút** (GPU) / **60–90 phút** (CPU).

---

### Bước 4 — Inference & Crop

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

1. **Vẽ OBB polygon** màu xanh lá lên ảnh gốc.
2. **Cắt axis-aligned crop** (dùng bounding box bao quanh polygon) → lưu vào `crops/`.
3. **Lưu ảnh annotated** để kiểm tra trực quan.

```
Input:  Data/test/images/test_0000.jpg
Output:
  crops/test_0000_crop_00.jpg    ← Crop sẵn sàng cho OCR
  crops/test_0000_annotated.jpg  ← Ảnh debug có khung OBB
```

---

## 4. Yêu cầu môi trường

```bash
# Cài đặt dependencies
pip install ultralytics opencv-python
```

| Package         | Phiên bản tối thiểu | Ghi chú                         |
|-----------------|---------------------|---------------------------------|
| `ultralytics`   | ≥ 8.0               | Bao gồm YOLOv8-OBB support     |
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
├── test_0000_annotated.jpg   ← Ảnh gốc có khung OBB
├── test_0001_crop_00.jpg
└── ...
```

Mẫu `test_0000_crop_00.jpg` sẽ là ảnh chỉ chứa **màn hình số của đồng hồ điện**,
cắt khá sát — đây là input lý tưởng cho bước OCR tiếp theo.

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
