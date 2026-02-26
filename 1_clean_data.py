"""
Giai đoạn 1: Data Cleaning & Standardization
=============================================
Script này đổi tên toàn bộ ảnh + nhãn OBB (.txt) sang định dạng chuẩn:
  train_0000.jpg  <->  train_0000.txt
  valid_0000.jpg  <->  valid_0000.txt
  test_0000.jpg   <->  test_0000.txt

LƯU Ý QUAN TRỌNG:
  - Format nhãn trong dataset này là OBB (Oriented Bounding Box),
    mỗi dòng có 9 giá trị: class x1 y1 x2 y2 x3 y3 x4 y4
  - Script KHÔNG sửa nội dung file nhãn, chỉ đổi tên.
  - Ảnh không có nhãn tương ứng sẽ bị BỎ QUA và in cảnh báo.

Chạy: python 1_clean_data.py
      hoặc: python 1_clean_data.py --dry-run  (xem trước, không đổi tên thật)
"""

import os
import argparse

# ─── Cấu hình ────────────────────────────────────────────────────────────────
BASE_DIR = "./Data"          # Thư mục chứa train, valid, test
SUB_DIRS = ["train", "valid", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# ─────────────────────────────────────────────────────────────────────────────


def standardize_filenames(dry_run: bool = False) -> None:
    mode = "[DRY-RUN] " if dry_run else ""
    print(f"{'='*60}")
    print(f"  {mode}Bắt đầu dọn dẹp tên file...")
    print(f"  Base directory : {os.path.abspath(BASE_DIR)}")
    print(f"{'='*60}\n")

    grand_total = 0

    for sub in SUB_DIRS:
        img_dir = os.path.join(BASE_DIR, sub, "images")
        lbl_dir = os.path.join(BASE_DIR, sub, "labels")

        if not os.path.isdir(img_dir):
            print(f"  [SKIP] Không tìm thấy thư mục: {img_dir}\n")
            continue

        # Lấy TẤT CẢ file ảnh, sắp xếp để thứ tự nhất quán giữa các lần chạy
        all_files = sorted(
            f for f in os.listdir(img_dir)
            if f.lower().endswith(IMG_EXTS)
        )

        count_ok = 0
        count_skip = 0

        for filename in all_files:
            name_root, ext = os.path.splitext(filename)

            old_img_path = os.path.join(img_dir, filename)
            old_lbl_path = os.path.join(lbl_dir, name_root + ".txt")

            # Bỏ qua nếu không có nhãn tương ứng
            if not os.path.isfile(old_lbl_path):
                print(f"  [WARN] Không có nhãn cho: {filename} -> Bỏ qua.")
                count_skip += 1
                continue

            # Tạo tên mới: <sub>_<index:04d>.<ext>
            new_name_root = f"{sub}_{count_ok:04d}"
            new_img_path  = os.path.join(img_dir, new_name_root + ext.lower())
            new_lbl_path  = os.path.join(lbl_dir, new_name_root + ".txt")

            # Tránh ghi đè nếu file đích đã tồn tại và khác file nguồn
            if old_img_path == new_img_path:
                count_ok += 1
                continue

            if not dry_run:
                os.rename(old_img_path, new_img_path)
                os.rename(old_lbl_path, new_lbl_path)
            else:
                print(f"  {filename:60s}  ->  {new_name_root + ext.lower()}")

            count_ok += 1

        grand_total += count_ok
        status = "[DRY-RUN]" if dry_run else "OK"
        print(f"  [{status}] '{sub}': {count_ok} files đổi tên, {count_skip} files bỏ qua.\n")

    print(f"{'='*60}")
    print(f"  Tổng cộng: {grand_total} files đã được chuẩn hóa.")
    print(f"  Bước tiếp theo: Sửa Data/data.yaml rồi chạy 2_train_yolo.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuẩn hóa tên file ảnh + nhãn YOLO-OBB")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Xem trước kết quả mà không thực sự đổi tên file"
    )
    args = parser.parse_args()

    standardize_filenames(dry_run=args.dry_run)
