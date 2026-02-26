Giờ mình sẽ thiết kế cho bạn một **experimental plan đúng kiểu research**, không phải kiểu “train xong là xong”.

Bối cảnh:

* Bài toán: Detect vùng hiển thị kWh trên công tơ điện
* Single class
* Nhiều thương hiệu: Genus, MicroTech, GE, Bentex
* Ảnh thực tế, điều kiện ánh sáng khác nhau
* Model base: YOLOv8

---

# Research Objective (phải rõ ngay từ đầu)

**Mục tiêu chính:**

> Tìm cấu hình tối ưu để detect chính xác vùng hiển thị kWh trong môi trường thực tế đa thương hiệu.

**Câu hỏi nghiên cứu:**

1. YOLOv8n có đủ capacity không?
2. Single class có tốt hơn multi-class không?
3. Data augmentation ảnh hưởng thế nào?
4. Model có generalize sang thương hiệu chưa thấy không?

---

# 1. Dataset Protocol (rất quan trọng)

## 1.1 Split chiến lược

KHÔNG random đơn giản.

Chia theo:

* Train: 70%
* Val: 15%
* Test: 15%

Nhưng quan trọng hơn:

### Cross-brand evaluation

Ví dụ:

| Setup | Train on          | Test on    |
| ----- | ----------------- | ---------- |
| E1    | All brands        | All brands |
| E2    | Genus + MicroTech | GE         |
| E3    | Genus             | MicroTech  |
| E4    | All except Bentex | Bentex     |

Đây mới là test generalization thật sự.

---

# 2. Baseline Experiment

### E0 – Baseline chuẩn

Model: `yolov8n.pt`
single_cls=True
Epoch: 100
Image size: 640
Default augmentation

Metrics:

* mAP50
* mAP50-95
* Precision
* Recall
* Inference time (ms/image)

Đây là mốc so sánh cho tất cả experiment sau.

---

# 3. Model Capacity Study

## E1 – So sánh kích thước model

| Model   | Hypothesis      |
| ------- | --------------- |
| yolov8n | Nhẹ, nhanh      |
| yolov8s | Tăng mAP        |
| yolov8m | Có thể overkill |

Mục tiêu:

> Tìm trade-off giữa accuracy và speed.

Nếu mAP tăng <2% nhưng inference chậm 40% → không đáng.

---

# 4. Single Class vs Multi Class

## E2 – Thử multi-class (nếu có annotation)

Class:

* display_region
* brand_label
* LED_status

So sánh:

* mAP display_region
* Stability
* Convergence speed

Hypothesis:

Single class có thể:

* Học nhanh hơn
* Ít overfit hơn

---

# 5. Data Augmentation Study

## E3 – Lighting robustness

Vì dataset có:

* LCD xanh
* LCD cam
* Không đèn nền
* Cơ học

Test:

1. No augmentation
2. Brightness/contrast augmentation
3. Random rotation ±15°
4. Motion blur

So sánh:

* mAP trên tập ảnh tối
* mAP trên tập ảnh sáng

---

# 6. Orientation Robustness

Vì có ảnh lật ngược:

## E4 – Rotation augmentation vs không

Hypothesis:

Nếu rotation augmentation giúp tăng mAP test orientation → không cần OBB.

Nếu không → cân nhắc OBB.

---

# 7. Error Analysis Protocol

Sau mỗi experiment:

Phân tích:

1. False positives xuất hiện ở đâu?

   * Logo?
   * LED?
   * Vỏ máy?

2. False negatives:

   * Màn hình tối?
   * Bị glare?
   * Góc nghiêng?

Không chỉ nhìn mAP.

---

# 8. Statistical Comparison

Không chỉ train 1 lần.

Mỗi experiment:

* Train 3 seeds khác nhau
* Lấy mean ± std

Nếu khác biệt < std → không có ý nghĩa thống kê.

---

# 9. Generalization Stress Test

Tạo tập:

* Ảnh mới hoàn toàn
* Chụp bằng điện thoại khác
* Góc khác

Test zero-shot generalization.

Đây mới là đánh giá thực tế.

---

# 10. Kết luận nghiên cứu

Cuối cùng bạn sẽ trả lời được:

* Model nhỏ nhất đạt performance đủ tốt là gì?
* Augmentation nào thực sự cần?
* Có cần OBB không?
* Single class có đủ không?

---

# Nếu viết thành báo cáo research

Cấu trúc:

1. Introduction
2. Related Work (YOLO-based detection)
3. Dataset Description
4. Experimental Setup
5. Ablation Study
6. Generalization Study
7. Discussion
8. Conclusion

---

# Thành thật với bạn

Nếu làm đúng plan này, project của bạn không còn là “train YOLO demo”.

Nó thành:

> Một nghiên cứu bài bản về robust meter display detection trong môi trường thực tế.

---

Nếu bạn muốn, mình có thể:

* Thiết kế luôn bảng log template để bạn ghi experiment
* Hoặc giúp bạn chọn 5 experiment quan trọng nhất để làm trước
* Hoặc phân tích risk overfitting dựa trên số lượng ảnh bạn đang có

Bạn đang có khoảng bao nhiêu ảnh annotated?
