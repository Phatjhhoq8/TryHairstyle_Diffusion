"""
Trích xuất ảnh gốc từ K-Hairstyle dataset dựa trên ID trong metadata.jsonl.
Lưu vào processed/ground_truth_images/ để dùng cho việc chấm điểm huấn luyện.

Cách dùng:
    python extract_eval_images.py
"""

import os
import sys
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
KHAIRSTYLE_DIR = PROJECT_DIR / "backend" / "data" / "dataset" / "khairstyle"
DATASET_DIRS = [
    KHAIRSTYLE_DIR / "training" / "images",
    KHAIRSTYLE_DIR / "validation" / "images",
]
PROCESSED_DIR = PROJECT_DIR / "backend" / "training" / "processed"
METADATA_PATH = PROCESSED_DIR / "metadata.jsonl"
OUTPUT_DIR = PROCESSED_DIR / "ground_truth_images"


def build_id_to_path_index(dataset_dirs):
    """
    Quét toàn bộ thư mục dataset (training + validation), tạo dict mapping: image_id → full_path.
    Ví dụ: "CP032677-006" → ".../0001.가르마/0126.CP032677/CP032677-006.jpg"
    """
    id_map = {}
    
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            print(f"  ⚠️ Bỏ qua (không tồn tại): {dataset_dir}")
            continue
        
        print(f">>> Đang quét: {dataset_dir}")
        count = 0
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            for img_path in dataset_dir.rglob(ext):
                img_id = img_path.stem
                id_map[img_id] = img_path
                count += 1
        print(f"    Tìm thấy {count} ảnh.")
    
    print(f">>> Tổng index: {len(id_map)} ảnh từ {len(dataset_dirs)} thư mục.")
    return id_map


def extract_images():
    """Đọc metadata.jsonl, tìm ảnh gốc theo ID, copy vào ground_truth_images/."""
    
    # Kiểm tra metadata tồn tại
    if not METADATA_PATH.exists():
        print(f"❌ Không tìm thấy metadata: {METADATA_PATH}")
        return
    
    if not any(d.exists() for d in DATASET_DIRS):
        print(f"❌ Không tìm thấy dataset directories")
        return
    
    # Tạo thư mục output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Bước 1: Index toàn bộ ảnh trong dataset (training + validation) → dict {id: path}
    id_map = build_id_to_path_index(DATASET_DIRS)
    
    # Bước 2: Đọc metadata.jsonl, lấy danh sách ID
    print(f">>> Đọc metadata: {METADATA_PATH}")
    metadata_entries = []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                metadata_entries.append(entry)
            except json.JSONDecodeError:
                continue
    
    print(f">>> Tổng cộng {len(metadata_entries)} entry trong metadata.")
    
    # Bước 3: Copy ảnh gốc vào processed/ground_truth_images/
    found = 0
    not_found = 0
    skipped = 0
    
    for entry in tqdm(metadata_entries, desc="Trích xuất ảnh"):
        img_id = entry.get("id", "")
        if not img_id:
            continue
        
        # Kiểm tra nếu đã tồn tại (skip để tránh copy lại)
        dst_path = OUTPUT_DIR / f"{img_id}.png"
        if dst_path.exists():
            skipped += 1
            continue
        
        # Tìm ảnh gốc từ index
        src_path = id_map.get(img_id)
        if src_path is None or not src_path.exists():
            not_found += 1
            continue
        
        # Copy ảnh (giữ nguyên chất lượng)
        shutil.copy2(str(src_path), str(dst_path))
        found += 1
    
    print(f"\n{'='*50}")
    print(f"✅ KẾT QUẢ TRÍCH XUẤT:")
    print(f"   - Copy thành công: {found}")
    print(f"   - Đã có sẵn (skip): {skipped}")
    print(f"   - Không tìm thấy:  {not_found}")
    print(f"   - Tổng metadata:   {len(metadata_entries)}")
    print(f"   - Output:          {OUTPUT_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    extract_images()
