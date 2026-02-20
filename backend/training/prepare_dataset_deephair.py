"""
Prepare Dataset cho quá trình Deep Texture Hair Inpainting model
Yêu cầu:
- Sử dụng dataset K-Hairstyle
- Không chỉnh sửa các module có sẵn của training_face.
- Tạo: bald_images, hair_only_images, hair_patches, style_vectors, identity_embeddings.
"""

import os
import sys
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
# Thêm đường dẫn gốc vào PYTHONPATH để nhận diện package backend
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from backend.app.services.training_utils import setupLogger, getDevice, ensureDir
from backend.app.services.face_detector import TrainingFaceDetector
from backend.app.services.visualizer import TrainingVisualizer
from backend.app.services.embedder import TrainingEmbedder

logger = setupLogger("PrepareDatasetDeepHair")

INPUT_DIR = PROJECT_DIR / "backend" / "data" / "dataset" / "khairstyle" / "training" / "images"
LABEL_DIR = PROJECT_DIR / "backend" / "data" / "dataset" / "khairstyle" / "training" / "labels"
PROCESSED_DIR = PROJECT_DIR / "backend" / "training" / "processed"

DIR_BALD = PROCESSED_DIR / "bald_images"
DIR_HAIR_ONLY = PROCESSED_DIR / "hair_only_images"
DIR_PATCHES = PROCESSED_DIR / "hair_patches"
DIR_STYLE = PROCESSED_DIR / "style_vectors"
DIR_IDENTITY = PROCESSED_DIR / "identity_embeddings"

# Tạo thư mục
for d in [DIR_BALD, DIR_HAIR_ONLY, DIR_PATCHES, DIR_STYLE, DIR_IDENTITY]:
    ensureDir(str(d))

def generate_hair_patches(image_cv2, hair_mask, img_name, patch_size=128, stride=64, min_hair_ratio=0.85):
    """
    Cắt các patch ngẫu nhiên/cố định từ tóc (Mật độ mask > min_hair_ratio).
    """
    patches_saved = 0
    h, w = image_cv2.shape[:2]
    
    # Tìm bounding box chứa tóc để giảm khu vực scan
    y_idx, x_idx = np.where(hair_mask > 0)
    if len(y_idx) == 0:
        return 0
    
    y_min, y_max = np.min(y_idx), np.max(y_idx)
    x_min, x_max = np.min(x_idx), np.max(x_idx)
    
    for y in range(y_min, y_max - patch_size + 1, stride):
        for x in range(x_min, x_max - patch_size + 1, stride):
            mask_patch = hair_mask[y:y+patch_size, x:x+patch_size]
            hair_pixels = np.count_nonzero(mask_patch)
            ratio = hair_pixels / (patch_size * patch_size)
            
            if ratio >= min_hair_ratio:
                img_patch = image_cv2[y:y+patch_size, x:x+patch_size]
                patch_path = DIR_PATCHES / f"{img_name}_patch_{patches_saved:03d}.png"
                cv2.imwrite(str(patch_path), img_patch)
                patches_saved += 1
                
    return patches_saved


def generate_text_prompt_from_khairstyle_json(json_path, mapping_dict):
    """
    Đọc file JSON K-Hairstyle tương ứng và trích xuất thuộc tính để tạo SDXL Text Prompt.
    Sử dụng mapping_dict (từ khóa tiếng Hàn sang tiếng Anh) để dịch thuộc tính ra chuỗi chuẩn.
    Trả về string prompt rỗng nếu không có dữ liệu.
    """
    if not os.path.exists(json_path):
        return ""
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        hair_data = data
        if not hair_data:
            return ""
            
        # Trích xuất các thuộc tính cơ bản và map sang file dịch tiếng Anh
        def get_mapped_val(key):
            raw = str(hair_data.get(key, "")).strip()
            # Bỏ qua các giá trị mang nghĩa "khác", "không xác định"
            if not raw or raw in ["기타", "없음", "none", "other", "해당없음", "nan", "NaN"]:
                return ""
            return mapping_dict.get(raw, raw)
            
        color = get_mapped_val("color")
        style = get_mapped_val("basestyle")
        shape = get_mapped_val("curl")
        length = get_mapped_val("length")
        bangs = get_mapped_val("bang")
        volume = get_mapped_val("hair-width")
        
        # Build prompt: "a [length] [volume] [color] [shape] [style] hair style, [bangs] bangs"
        prompt_parts = []
        if length:
            prompt_parts.append(length)
        if volume:
            prompt_parts.append(f"{volume} volume")
        if color:
            prompt_parts.append(color)
        if shape:
            prompt_parts.append(shape)
        if style:
            prompt_parts.append(f"{style} hairstyle")
        else:
            prompt_parts.append("hairstyle")
            
        prompt = " ".join([p.lower() for p in prompt_parts if p]).strip()
        
        if bangs:
            prompt += f", with {bangs.lower()} bangs"
            
        return prompt
        
    except Exception as e:
        logger.warning(f"Loi doc file nhan {json_path}: {e}")
        return ""

def process_dataset():
    logger.info(f"Bat dau xu ly dataset tu: {INPUT_DIR}")
    
    # Init Models
    detector = TrainingFaceDetector()
    embedder = TrainingEmbedder(yawThreshold=45.0)
    
    # Load Mapping Dictionary
    mapping_dict = {}
    dict_path = PROJECT_DIR / "backend" / "training" / "mapping_dict.json"
    if dict_path.exists():
        with open(str(dict_path), 'r', encoding='utf-8') as f:
            mapping_dict = json.load(f)
        logger.info(f"Đã tải thành công {len(mapping_dict)} từ khóa dịch K-Hairstyle.")
    else:
        logger.warning(f"Chưa tìm thấy {dict_path}. Sẽ dùng raw labels. Khuyến cáo chạy script normalize_khairstyle.py trước.")
        
    # Find all images recursively
    image_extensions = {'.png', '.jpg', '.jpeg'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(INPUT_DIR.rglob(f"*{ext}")))
        image_files.extend(list(INPUT_DIR.rglob(f"*{ext.upper()}")))
        
    logger.info(f"Tim thay {len(image_files)} anh de xu ly.")
    
    successful = 0
    
    for idx, img_path_obj in enumerate(tqdm(image_files)):
        img_name = img_path_obj.stem
        img_path = str(img_path_obj)
        
        # Tìm file JSON label tương ứng
        # K-Hairstyle map 1-1 thư mục images -> labels
        rel_path = img_path_obj.relative_to(INPUT_DIR)
        json_path = LABEL_DIR / rel_path.parent / f"{img_name}.json"
        
        text_prompt = generate_text_prompt_from_khairstyle_json(str(json_path), mapping_dict)
        
        image_cv2 = cv2.imread(img_path)
        if image_cv2 is None:
            continue
            
        h, w = image_cv2.shape[:2]
        
        # 1. Detect Face
        faces = detector.detect(image_cv2)
        if not faces:
            continue
            
        # Select largest face
        faces.sort(key=lambda x: (x["bbox"][2]-x["bbox"][0]) * (x["bbox"][3]-x["bbox"][1]), reverse=True)
        main_face = faces[0]
        bbox = main_face["bbox"]
        
        # 2. Tạo Mask Tóc từ Polygon
        hair_mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            with open(str(json_path), 'r', encoding='utf-8') as f:
                hair_json = json.load(f)
            
            # Hàm phụ parse polygon từ str -> cv2 points
            def parse_polygon(poly_str):
                if not poly_str: return None
                try:
                    points_list = json.loads(poly_str)
                    if not points_list: return None
                    pts = np.array([[float(p['x']), float(p['y'])] for p in points_list], np.int32)
                    return pts.reshape((-1, 1, 2))
                except Exception as e:
                    logger.warning(f"Lỗi parse polygon: {e}")
                    return None
            
            p1 = parse_polygon(hair_json.get("polygon1"))
            p2 = parse_polygon(hair_json.get("polygon2"))
            
            if p1 is not None:
                cv2.fillPoly(hair_mask, [p1], 255)
            if p2 is not None:
                cv2.fillPoly(hair_mask, [p2], 255)
                
        except Exception as e:
            logger.error(f"Lỗi đọc JSON lấy polygon {img_name}: {e}")
            continue
        
        if np.count_nonzero(hair_mask) < 1000:
           # Quá ít tóc
           continue
           
        # Dilate 1 chút để bao phủ viền tóc rời
        kernel = np.ones((5,5),np.uint8)
        hair_mask_dilated = cv2.dilate(hair_mask, kernel, iterations=1)
        
        # 3. Create Bald Image (Inpainting vung toc)
        bald_image = cv2.inpaint(image_cv2, hair_mask_dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        # Pad to 1024x1024 hoac de nguyen
        
        # 4. Create Hair-Only Image (Alpha channel)
        b, g, r = cv2.split(image_cv2)
        hair_only_rgba = cv2.merge((b, g, r, hair_mask))
        
        # 5. Extract Identity Embedding
        # Estimate coarse yaw from bbox center ratio (or use 0 for simplicity, embedder handles 0)
        # We don't need accurate yaw here since Adaface is default for all when yaw is provided, let's use 0 to trigger insightface
        embedding_result = embedder.getEmbedding(image_cv2, bbox, yaw=0)
        if embedding_result is None:
            continue
            
        embedding = embedding_result["embedding"]
        
        # 6. Extract Style Vector Image (Cropped 224x224 hair for CLIP input)
        y_idx, x_idx = np.where(hair_mask > 0)
        y_m, y_x = np.min(y_idx), np.max(y_idx)
        x_m, x_x = np.min(x_idx), np.max(x_idx)
        cropped_hair_img = hair_only_rgba[y_m:y_x, x_m:x_x]
        style_img = cv2.resize(cropped_hair_img, (224, 224))
        
        # 7. Generate Patches for Texture
        num_patches = generate_hair_patches(image_cv2, hair_mask, img_name)
        
        # --- SAVE ---
        cv2.imwrite(str(DIR_BALD / f"{img_name}.png"), bald_image)
        cv2.imwrite(str(DIR_HAIR_ONLY / f"{img_name}.png"), hair_only_rgba)
        cv2.imwrite(str(DIR_STYLE / f"{img_name}.png"), style_img)
        np.save(str(DIR_IDENTITY / f"{img_name}.npy"), embedding)
        
        with open(str(PROCESSED_DIR / "metadata.jsonl"), "a", encoding="utf-8") as f:
            metadata = {
                "id": img_name,
                "bald": f"bald_images/{img_name}.png",
                "hair_only": f"hair_only_images/{img_name}.png",
                "style": f"style_vectors/{img_name}.png",
                "identity": f"identity_embeddings/{img_name}.npy",
                "num_patches": num_patches,
                "text_prompt": text_prompt
            }
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            
        successful += 1

    logger.info(f"Hoan tat! Xu ly thanh cong {successful}/{len(image_files)} anh. Metadata: {PROCESSED_DIR / 'metadata.jsonl'}")


if __name__ == "__main__":
    process_dataset()
