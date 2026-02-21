"""
Prepare Dataset cho quá trình Deep Texture Hair Inpainting model (Stage 1 & 2)
Tạo: bald_images, hair_only_images, hair_patches, style_vectors, identity_embeddings.
"""

import os
import cv2
import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from backend.app.services.training_utils import setupLogger, getDevice, ensureDir
from backend.app.services.face_detector import TrainingFaceDetector
from backend.app.services.embedder import TrainingEmbedder

logger = setupLogger("PrepareDatasetDeepHair")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_DIR / "backend" / "data" / "dataset" / "khairstyle" / "training" / "images"
LABEL_DIR = PROJECT_DIR / "backend" / "data" / "dataset" / "khairstyle" / "training" / "labels"
PROCESSED_DIR = PROJECT_DIR / "backend" / "training" / "processed"

DIR_BALD = PROCESSED_DIR / "bald_images"
DIR_HAIR_ONLY = PROCESSED_DIR / "hair_only_images"
DIR_PATCHES = PROCESSED_DIR / "hair_patches"
DIR_STYLE = PROCESSED_DIR / "style_vectors"
DIR_IDENTITY = PROCESSED_DIR / "identity_embeddings"

# Tạo thư mục lưu dataset
for d in [DIR_BALD, DIR_HAIR_ONLY, DIR_PATCHES, DIR_STYLE, DIR_IDENTITY]:
    ensureDir(str(d))

def generate_hair_patches(image_cv2, hair_mask, img_name, patch_size=128, stride=64, min_hair_ratio=0.85):
    """ Cắt các patch ngẫu nhiên/cố định từ tóc (Mật độ mask khối lượng tóc > min_hair_ratio). """
    patches_saved = 0
    h, w = image_cv2.shape[:2]
    
    y_idx, x_idx = np.where(hair_mask > 0)
    if len(y_idx) == 0: return 0
    
    y_min, y_max = np.min(y_idx), np.max(y_idx)
    x_min, x_max = np.min(x_idx), np.max(x_idx)
    
    for y in range(y_min, max(y_min+1, y_max - patch_size + 1), stride):
        for x in range(x_min, max(x_min+1, x_max - patch_size + 1), stride):
            if y+patch_size > h or x+patch_size > w: continue
            
            mask_patch = hair_mask[y:y+patch_size, x:x+patch_size]
            hair_pixels = np.count_nonzero(mask_patch)
            ratio = hair_pixels / (patch_size * patch_size)
            
            if ratio >= min_hair_ratio:
                img_patch = image_cv2[y:y+patch_size, x:x+patch_size]
                patch_path = DIR_PATCHES / f"{img_name}_patch_{patches_saved:03d}.png"
                cv2.imwrite(str(patch_path), img_patch)
                patches_saved += 1
    return patches_saved

def generate_text_prompt_from_khairstyle_json(json_data, mapping_dict):
    """ Build Prompt SDXL Dựa trên K-Hairstyle Metadata """
    try:
        hair_data = json_data
        def get_mapped_val(key):
            raw = str(hair_data.get(key, "")).strip()
            if not raw or raw in ["기타", "없음", "none", "other", "해당없음", "nan", "NaN"]:
                return ""
            return mapping_dict.get(raw, raw)
            
        color = get_mapped_val("color")
        style = get_mapped_val("basestyle")
        shape = get_mapped_val("curl")
        length = get_mapped_val("length")
        bangs = get_mapped_val("bang")
        volume = get_mapped_val("hair-width")
        
        prompt_parts = []
        if length: prompt_parts.append(length)
        if volume: prompt_parts.append(f"{volume} volume")
        if color: prompt_parts.append(color)
        if shape: prompt_parts.append(shape)
        if style: prompt_parts.append(f"{style} hairstyle")
        else: prompt_parts.append("hairstyle")
            
        prompt = " ".join([p.lower() for p in prompt_parts if p]).strip()
        if bangs: prompt += f", with {bangs.lower()} bangs"
            
        return prompt
    except Exception as e:
        logger.warning(f"Lỗi đọc JSON Label: {e}")
        return ""

# Phải nạp Object gốc để tránh Error Pickling cho multiprocessing
detector = None
embedder = None
mapping_dict_global = {}

def process_single_image(img_path_obj):
    """
    Hàm worker thực hiện xử lý trên 1 file ảnh. 
    Chạy đa luồng/đa tiến trình.
    """
    global detector, embedder, mapping_dict_global
    
    # Khởi tạo model lười (Lazy initialization) riêng cho từng Worker Tiến trình
    if detector is None:
        detector = TrainingFaceDetector()
    if embedder is None:
        embedder = TrainingEmbedder(yawThreshold=45.0)

    img_name = img_path_obj.stem
    img_path = str(img_path_obj)
    
    # Mapping file JSON
    rel_path = img_path_obj.relative_to(INPUT_DIR)
    
    # Xử lý tên file (Ảnh thường có dấu trừ '-', còn JSON có thể là gạch dưới '_')
    json_path = LABEL_DIR / rel_path.parent / f"{img_name}.json"
    
    if not json_path.exists():
        json_path = LABEL_DIR / rel_path.parent / f"{img_name.replace('-', '_')}.json"
        if not json_path.exists():
            return None # Bỏ qua

    # Load file nhãn một lần duy nhất
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        return None

    text_prompt = generate_text_prompt_from_khairstyle_json(json_data, mapping_dict_global)
    
    image_cv2 = cv2.imread(img_path)
    if image_cv2 is None: return None
    h, w = image_cv2.shape[:2]

    # --- BƯỚC 1: Segment Tóc bằng JSON Polygon (Loại bỏ Segformer) ---
    hair_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Hàm đọc và vẽ polygon từ JSON object caching
    def draw_polygons(poly_key):
        try:
            poly_data = json_data.get(poly_key, "")
            
            if isinstance(poly_data, str) and poly_data.strip():
                polygons = json.loads(poly_data)
            elif isinstance(poly_data, list):
                polygons = poly_data
            else:
                return
            
            for poly in polygons:
                if not poly: continue
                pts = np.array([[p["x"], p["y"]] for p in poly], dtype=np.int32)
                cv2.fillPoly(hair_mask, [pts], 255)
        except Exception as e:
            pass

    draw_polygons("polygon1")
    # draw_polygons("polygon2")
    
    if np.count_nonzero(hair_mask) < 1000: return None
    
    kernel = np.ones((5,5), np.uint8)
    hair_mask_dilated = cv2.dilate(hair_mask, kernel, iterations=1)
    
    # --- BƯỚC 2: Detect Face ---
    # Chỉ detect để lấy bbox cho AdaFace Embedding
    faces = detector.detect(image_cv2)
    if not faces: return None
    faces.sort(key=lambda x: (x["bbox"][2]-x["bbox"][0]) * (x["bbox"][3]-x["bbox"][1]), reverse=True)
    main_face = faces[0]
    bbox = main_face["bbox"]
    
    # --- BƯỚC 3: Tạo Dataset 1 (Bald Image - Ảnh trọc)
    bald_image = cv2.inpaint(image_cv2, hair_mask_dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    # --- BƯỚC 4: Tạo Dataset 2 (Hair Only)
    b, g, r = cv2.split(image_cv2)
    hair_only_rgba = cv2.merge((b, g, r, hair_mask))
    
    # --- BƯỚC 5: Tạo Dataset 5 (Identity Embedding ArcFace)
    embedding_result = embedder.getEmbedding(image_cv2, bbox, yaw=0)
    if embedding_result is None: return None
    embedding = embedding_result["embedding"]
    
    # --- BƯỚC 6: Tạo Dataset 4 (Style Vectors)
    y_idx, x_idx = np.where(hair_mask > 0)
    y_m, y_x = np.min(y_idx), np.max(y_idx)
    x_m, x_x = np.min(x_idx), np.max(x_idx)
    cropped_hair_img = hair_only_rgba[y_m:y_x, x_m:x_x]
    
    # Check lại Shape để chống crash OpenCV resize
    if cropped_hair_img.shape[0] < 10 or cropped_hair_img.shape[1] < 10:
        return None
        
    style_img = cv2.resize(cropped_hair_img, (224, 224))
    
    # --- BƯỚC 7: Tạo Dataset 3 (Deep Texture Hair Patches 128x128)
    num_patches = generate_hair_patches(image_cv2, hair_mask, img_name)
    
    # Ghi Disk Vout an toàn Multiprocess
    cv2.imwrite(str(DIR_BALD / f"{img_name}.png"), bald_image)
    cv2.imwrite(str(DIR_HAIR_ONLY / f"{img_name}.png"), hair_only_rgba)
    cv2.imwrite(str(DIR_STYLE / f"{img_name}.png"), style_img)
    np.save(str(DIR_IDENTITY / f"{img_name}.npy"), embedding)
    
    # Trả về chuỗi Metadata hoàn chỉnh để gom lại ghi 1 lần ở Thread gốc
    metadata = {
        "id": img_name,
        "bald": f"bald_images/{img_name}.png",
        "hair_only": f"hair_only_images/{img_name}.png",
        "style": f"style_vectors/{img_name}.png",
        "identity": f"identity_embeddings/{img_name}.npy",
        "num_patches": num_patches,
        "text_prompt": text_prompt
    }
    return metadata

def process_dataset():
    global mapping_dict_global
    
    logger.info(f"Bắt đầu xử lý Pipeline tạo Dataset (ĐA LUỒNG) từ: {INPUT_DIR}")
    
    dict_path = PROJECT_DIR / "backend" / "training" / "mapping_dict.json"
    if dict_path.exists():
        with open(str(dict_path), 'r', encoding='utf-8') as f:
            mapping_dict_global = json.load(f)
            
    image_files = list(INPUT_DIR.rglob("*.jpg")) + list(INPUT_DIR.rglob("*.png"))
    logger.info(f"Đã tìm thấy {len(image_files)} ảnh raw.")
    
    successful = 0
    metadata_lines = []
    
    max_workers = min(os.cpu_count() or 1, 8) # Limit to 8 to avoid VRAM exhaustion on Yolo
    logger.info(f"Sử dụng {max_workers} luồng tiến trình (ProcessPoolExecutor)...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Tqdm để theo dõi tiến độ Multiprocessing
        futures = {executor.submit(process_single_image, img_path): img_path for img_path in image_files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(image_files), desc="Tạo Dữ Liệu Đa Luồng"):
            result_meta = future.result()
            if result_meta:
                metadata_lines.append(json.dumps(result_meta, ensure_ascii=False) + "\n")
                successful += 1

    # Ghi Lịch Sử Metadata Label an toàn ở luồng chính
    if metadata_lines:
        with open(str(PROCESSED_DIR / "metadata.jsonl"), "a", encoding="utf-8") as f:
            f.writelines(metadata_lines)

    logger.info(f"Hoàn tất tạo Pipeline Dữ liệu! Thành công: {successful}/{len(image_files)}.")

if __name__ == "__main__":
    process_dataset()
