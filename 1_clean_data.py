"""
Giai đoạn 1: Data Cleaning & Standardization
=============================================
Script này thực hiện 2 bước chuẩn hoá cho dataset YOLO single-class:

  BƯỚC 1 — Đổi tên file sang định dạng chuẩn:
    train_0000.jpg  <->  train_0000.txt
    valid_0000.jpg  <->  valid_0000.txt
    test_0000.jpg   <->  test_0000.txt

  BƯỚC 2 — Remap class ID về 0 (single-class mode):
    Mọi annotation trong .txt đều được đặt class = 0 ("display"),
    khớp với data.yaml (nc: 1) và plan.md E0 Baseline.

  - Ảnh không có nhãn tương ứng sẽ bị BỎ QUA và in cảnh báo.

Chạy: python 1_clean_data.py
      hoặc: python 1_clean_data.py --dry-run  (xem trước, không thay đổi thật)
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
    print(f"  Bước tiếp theo: chạy 2_train_yolo.py")
    print(f"{'='*60}")


def remap_class_ids(dry_run: bool = False) -> None:
    """
    Bước 2: Đặt lại class ID = 0 cho toàn bộ file nhãn.
    Lý do: data.yaml có nc=1 (single-class mode), class ID gốc != 0
    sẽ bị Ultralytics bỏ qua vì vượt quá nc.
    """
    mode = "[DRY-RUN] " if dry_run else ""
    print(f"\n{'='*60}")
    print(f"  {mode}Remap class IDs → 0 (single-class mode)...")
    print(f"{'='*60}\n")

    for sub in SUB_DIRS:
        lbl_dir = os.path.join(BASE_DIR, sub, "labels")
        if not os.path.isdir(lbl_dir):
            print(f"  [SKIP] Không tìm thấy: {lbl_dir}\n")
            continue

        txt_files = sorted(f for f in os.listdir(lbl_dir) if f.endswith(".txt"))
        count_changed = 0

        for fname in txt_files:
            fpath = os.path.join(lbl_dir, fname)
            with open(fpath, "r") as f:
                lines = f.readlines()

            new_lines = []
            changed = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] != "0":
                    parts[0] = "0"
                    new_lines.append(" ".join(parts) + "\n")
                    changed = True
                else:
                    new_lines.append(line)

            if changed:
                if not dry_run:
                    with open(fpath, "w") as f:
                        f.writelines(new_lines)
                count_changed += 1

        status = "[DRY-RUN]" if dry_run else "OK"
        print(f"  [{status}] '{sub}': {count_changed}/{len(txt_files)} files cập nhật class ID → 0\n")

    print(f"{'='*60}")
    print(f"  Remap hoàn tất. Mọi annotation đã là class 0 ('display').")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chuẩn hóa tên file ảnh + nhãn YOLO (single-class)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Xem trước kết quả mà không thực sự thay đổi file"
    )
    args = parser.parse_args()

    standardize_filenames(dry_run=args.dry_run)
    remap_class_ids(dry_run=args.dry_run)
