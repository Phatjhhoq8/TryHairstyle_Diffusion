"""
Precompute Face Masks — SegFormer + 3DDFA V2 Convex Hull.

Kết hợp 2 tầng tạo face_mask:
  1. SegFormer: semantic segmentation (face classes 1-12)
  2. 3DDFA V2: convex hull từ 3D mesh projection (fallback khi SegFormer miss)

Tạo face_mask.png cho mỗi sample trong mỗi chunk processed_XXX.
Face mask dùng làm "vùng cấm" khi mask augmentation mở rộng mask tóc.

Usage:
    python precompute_face_masks.py --data_dir "C:/Users/Admin/Desktop/Drive"
    python precompute_face_masks.py --data_dir "/content/drive/MyDrive/..." --device cuda
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# SegFormer face classes (skin, nose, glasses, eyes, brows, ears, mouth, lips)
SEGFORMER_FACE_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}


# ==============================================================
# TẦNG 1: SegFormer
# ==============================================================

def load_segformer(device="cuda"):
    """Load SegFormer model."""
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    import torch
    
    model_id = "jonathandinu/face-parsing"
    local_path = str(PROJECT_DIR / "backend" / "models" / "segformer_face_parsing")
    
    if os.path.exists(local_path):
        processor = SegformerImageProcessor.from_pretrained(local_path)
        model = SegformerForSemanticSegmentation.from_pretrained(local_path)
        print(f"  ✅ SegFormer loaded từ local: {local_path}")
    else:
        processor = SegformerImageProcessor.from_pretrained(model_id)
        model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        print(f"  ✅ SegFormer loaded từ HuggingFace: {model_id}")
    
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    model.to(device)
    model.eval()
    return processor, model, device


def get_face_mask_segformer(image_bgr, processor, model, device):
    """
    Tầng 1: SegFormer → binary face mask.
    
    Returns:
        numpy array (H, W), uint8 [0, 255] — 255=face
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        h, w = image_bgr.shape[:2]
        upsampled = F.interpolate(
            logits, size=(h, w),
            mode="bilinear", align_corners=False
        )
        parsing = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()
    
    # Tạo binary face mask
    face_mask = np.zeros_like(parsing, dtype=np.uint8)
    for cls in SEGFORMER_FACE_CLASSES:
        face_mask[parsing == cls] = 255
    
    return face_mask


# ==============================================================
# TẦNG 2: 3DDFA V2 Convex Hull (fallback)
# ==============================================================

def load_face_pipeline():
    """
    Load TrainingFaceDetector + TrainingPoseEstimator + TrainingReconstructor3D.
    Reuse modules từ training pipeline.
    """
    from backend.app.services.face_detector import TrainingFaceDetector
    from backend.app.services.pose_estimator import TrainingPoseEstimator
    from backend.app.services.reconstructor_3d import TrainingReconstructor3D
    
    detector = TrainingFaceDetector()
    pose_estimator = TrainingPoseEstimator()
    reconstructor = TrainingReconstructor3D()
    
    available = {
        "detector": detector.isAvailable(),
        "pose": pose_estimator.isAvailable(),
        "3d": reconstructor.isAvailable()
    }
    print(f"  ✅ Face pipeline loaded: {available}")
    
    return detector, pose_estimator, reconstructor


def get_face_mask_3ddfa(image_bgr, detector, pose_estimator, reconstructor):
    """
    Tầng 2: YOLOv8 detect → 3DDFA V2 reconstruct → convex hull → face mask.
    
    Returns:
        numpy array (H, W), uint8 [0, 255] — 255=face, hoặc None nếu thất bại
    """
    h, w = image_bgr.shape[:2]
    
    # Detect faces
    faces = detector.detect(image_bgr) if detector.isAvailable() else []
    if not faces:
        return None
    
    face_mask = np.zeros((h, w), dtype=np.uint8)
    
    for face_data in faces:
        bbox = face_data["bbox"]
        
        # Estimate pose
        pose_result = pose_estimator.estimate(image_bgr, bbox)
        
        # 3D Reconstruct
        recon_result = reconstructor.reconstruct(image_bgr, bbox)
        
        if recon_result is not None:
            # Lấy 3D vertices → chiếu xuống 2D → convex hull
            vertices = recon_result["vertices"]  # (3, N)
            xs = np.clip(vertices[0, :].astype(np.int32), 0, w - 1)
            ys = np.clip(vertices[1, :].astype(np.int32), 0, h - 1)
            
            pts = np.column_stack([xs, ys])
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(face_mask, hull, 255)
        else:
            # Fallback: dùng bbox làm face mask
            x1, y1, x2, y2 = [int(c) for c in bbox]
            # Co vào 10% để tránh bao quá rộng
            bw, bh = x2 - x1, y2 - y1
            x1 = max(0, x1 + int(bw * 0.1))
            y1 = max(0, y1 + int(bh * 0.1))
            x2 = min(w, x2 - int(bw * 0.1))
            y2 = min(h, y2 - int(bh * 0.1))
            face_mask[y1:y2, x1:x2] = 255
    
    return face_mask


# ==============================================================
# KẾT HỢP 2 TẦNG
# ==============================================================

def get_combined_face_mask(image_bgr, segformer_args, pipeline_args):
    """
    Kết hợp SegFormer + 3DDFA V2:
    - Luôn chạy SegFormer trước
    - Nếu SegFormer mask quá bé (< 1% ảnh) → bổ sung 3DDFA V2
    - Union 2 mask
    
    Args:
        segformer_args: (processor, model, device)
        pipeline_args: (detector, pose_estimator, reconstructor) hoặc None
    
    Returns:
        numpy (H, W), uint8 [0, 255]
    """
    processor, model, device = segformer_args
    h, w = image_bgr.shape[:2]
    
    # Tầng 1: SegFormer
    face_mask = get_face_mask_segformer(image_bgr, processor, model, device)
    
    segformer_area = (face_mask > 127).sum()
    total_pixels = h * w
    coverage = segformer_area / total_pixels
    
    # Nếu SegFormer bắt được đủ mặt (> 1% ảnh) → dùng luôn
    if coverage >= 0.01:
        # Dilate nhẹ để tạo buffer an toàn
        kernel = np.ones((5, 5), np.uint8)
        face_mask = cv2.dilate(face_mask, kernel, iterations=1)
        return face_mask
    
    # Tầng 2: SegFormer miss → thử 3DDFA V2
    if pipeline_args is not None:
        detector, pose_estimator, reconstructor = pipeline_args
        mask_3d = get_face_mask_3ddfa(image_bgr, detector, pose_estimator, reconstructor)
        
        if mask_3d is not None:
            # Union: kết hợp cả 2
            face_mask = np.maximum(face_mask, mask_3d)
    
    # Dilate nhẹ
    kernel = np.ones((5, 5), np.uint8)
    face_mask = cv2.dilate(face_mask, kernel, iterations=1)
    
    return face_mask


# ==============================================================
# XỬ LÝ CHUNK
# ==============================================================

def process_chunk(chunk_dir, segformer_args, pipeline_args):
    """
    Xử lý 1 chunk: tạo face_masks/ cho tất cả ảnh trong ground_truth_images/.
    
    Returns:
        (processed_count, skipped_count, enhanced_count)
    """
    gt_dir = chunk_dir / "ground_truth_images"
    out_dir = chunk_dir / "face_masks"
    
    if not gt_dir.exists():
        print(f"  ⚠️  Không tìm thấy {gt_dir}, skip.")
        return 0, 0, 0
    
    out_dir.mkdir(exist_ok=True)
    
    # Lấy danh sách ảnh
    image_files = sorted(list(gt_dir.glob("*.png")) + list(gt_dir.glob("*.jpg")))
    
    processed = 0
    skipped = 0
    enhanced = 0  # Số ảnh cần 3DDFA V2 fallback
    
    for img_path in tqdm(image_files, desc=f"  {chunk_dir.name}", leave=False):
        out_path = out_dir / f"{img_path.stem}.png"
        
        # Skip nếu đã tồn tại
        if out_path.exists():
            skipped += 1
            continue
        
        # Load ảnh
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"    ⚠️  Không đọc được: {img_path.name}")
            continue
        
        # Tạo face mask (kết hợp 2 tầng)
        face_mask = get_combined_face_mask(image_bgr, segformer_args, pipeline_args)
        
        # Check nếu 3DDFA was needed
        h, w = image_bgr.shape[:2]
        segformer_only = get_face_mask_segformer(
            image_bgr, segformer_args[0], segformer_args[1], segformer_args[2]
        )
        if (segformer_only > 127).sum() / (h * w) < 0.01:
            enhanced += 1
        
        # Lưu
        cv2.imwrite(str(out_path), face_mask)
        processed += 1
    
    return processed, skipped, enhanced


def main():
    parser = argparse.ArgumentParser(description="Precompute face masks (SegFormer + 3DDFA V2)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Đường dẫn tới thư mục chứa các chunk processed_XXX")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda hoặc cpu (default: cuda)")
    parser.add_argument("--no-3ddfa", action="store_true",
                        help="Chỉ dùng SegFormer, không dùng 3DDFA V2 fallback")
    parser.add_argument("--force", action="store_true",
                        help="Ghi đè file đã tồn tại")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Không tìm thấy: {data_dir}")
        return
    
    # Tìm tất cả chunk processed_*
    chunks = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("processed_")])
    
    if not chunks:
        print(f"❌ Không tìm thấy chunk processed_* trong {data_dir}")
        return
    
    print(f"📁 Data dir: {data_dir}")
    print(f"📦 Tìm thấy {len(chunks)} chunks")
    print()
    
    # Load SegFormer
    print("🔄 Loading SegFormer...")
    segformer_args = load_segformer(args.device)
    
    # Load 3DDFA V2 pipeline (nếu không bị disable)
    pipeline_args = None
    if not args.no_3ddfa:
        print("🔄 Loading Face Pipeline (YOLOv8 + 3DDFA V2)...")
        try:
            pipeline_args = load_face_pipeline()
        except Exception as e:
            print(f"  ⚠️  Không load được face pipeline: {e}")
            print(f"  → Chỉ dùng SegFormer")
    else:
        print("ℹ️  3DDFA V2 bị disable (--no-3ddfa)")
    
    print()
    
    # Nếu --force, xóa file cũ
    if args.force:
        print("⚠️  --force: sẽ ghi đè file đã tồn tại")
        for chunk in chunks:
            face_dir = chunk / "face_masks"
            if face_dir.exists():
                for f in face_dir.glob("*.png"):
                    f.unlink()
        print()
    
    # Xử lý từng chunk
    total_processed = 0
    total_skipped = 0
    total_enhanced = 0
    
    for chunk in chunks:
        print(f"📂 Processing: {chunk.name}")
        processed, skipped, enhanced = process_chunk(chunk, segformer_args, pipeline_args)
        total_processed += processed
        total_skipped += skipped
        total_enhanced += enhanced
        print(f"  ✅ Done: {processed} mới, {skipped} đã có, {enhanced} cần 3DDFA fallback")
    
    print()
    print(f"🎉 Hoàn tất!")
    print(f"   Tổng: {total_processed} face masks mới, {total_skipped} đã có")
    print(f"   3DDFA V2 enhanced: {total_enhanced} ảnh (SegFormer miss)")


if __name__ == "__main__":
    main()
