"""
Tool chuẩn hóa (Normalization Tool) cho dataset K-Hairstyle JSON.
Yêu cầu từ user:
- Chuẩn hóa các trường tiếng Hàn thành tiếng Anh/chuẩn tắc (canonical form)
- Có dictionary từ điển dịch
- Không chỉnh sửa polygon1, polygon2, path, filename, source
- Log lại thay đổi và list ra các từ unknown (chưa được chuẩn hóa)
- Chạy độc lập, dễ mở rộng
"""

import os
import json
import logging
from pathlib import Path
from tqdm import tqdm

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("normalization_tool.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NormalizeTool")

# =====================================================================
# TỪ ĐIỂN CHUẨN HÓA (MAPPING DICTIONARY)
# Có thể dễ dàng mở rộng bằng cách thêm key-value vào đây
# =====================================================================
MAPPING_DICT = {
    
}

# Các trường trong obj "hair" cần phải format
FIELDS_TO_NORMALIZE = [
    "basestyle", "basestyle-type", "length", "curl", "bang", 
    "loss", "side", "color", "hair-width", "natural-curl", 
    "damage", "melanin-color", "style", "volume", "shape"
]

def load_existing_dict(dict_path):
    if dict_path.exists():
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Không thể đọc {dict_path}, tạo mới... ({e})")
    
    # Khởi tạo file mới với gốc từ MAPPING_DICT
    return MAPPING_DICT.copy()

def save_dict(dict_path, data_dict):
    with open(dict_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

def extract_korean_dictionary(labels_dir, output_dict_path):
    logger.info(f"Bắt đầu quét JSON Dataset trong thư mục: {labels_dir}")
    labels_path = Path(labels_dir)
    out_path = Path(output_dict_path)
    
    if not labels_path.exists():
        logger.error("Thư mục labels không tồn tại!")
        return
        
    json_files = list(labels_path.rglob("*.json"))
    logger.info(f"Tìm thấy {len(json_files)} files JSON.")
    
    # Load từ điển hiện tại trên đĩa màng lên
    final_dict = load_existing_dict(out_path)
    
    new_words_count = 0
    save_interval = 100 # Lưu vào file sau mỗi 100 file JSON (hoặc khi có từ mới tùy chiến lược)
    
    for i, json_file in enumerate(tqdm(json_files, desc="Scanning JSONs")):
        has_new_word = False
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            hair_obj = data
            for field in FIELDS_TO_NORMALIZE:
                if field in hair_obj:
                    val = hair_obj[field]
                    if not isinstance(val, str):
                        continue
                        
                    val = val.strip()
                    if not val:
                        continue
                        
                    # Nếu chứa ký tự non-ascii (tiếng Hàn) và CHƯA có trong từ điển
                    if not val.isascii() and val not in final_dict:
                        final_dict[val] = "" # Khởi tạo rỗng để chờ user dịch
                        has_new_word = True
                        new_words_count += 1
                        
        except Exception as e:
            logger.error(f"Lỗi đọc file {json_file.name}: {e}")
            
        # Lưu file ngay lập tức nếu phát hiện từ tiếng Hàn mới, hoặc định kì
        if has_new_word or (i % save_interval == 0):
            save_dict(out_path, final_dict)
            
    # Lưu chốt hạ lần cuối
    save_dict(out_path, final_dict)
        
    # --- BÁO CÁT KẾT QUẢ ---
    logger.info("="*50)
    logger.info("KẾT QUẢ TRÍCH XUẤT TỪ ĐIỂN")
    logger.info("="*50)
    logger.info(f"Tổng số từ vựng (tiếng Hàn) hiện có: {len(final_dict)}")
    logger.info(f" - Số từ MỚI phát hiện đợt này: {new_words_count}")
    logger.info(f" => Tệp từ điển được CHÚNG TÔI LIÊN TỤC CẬP NHẬT TRỰC TIẾP TẠI: {output_dict_path}")
    if new_words_count > 0:
        logger.info("Vui lòng mở file mapping_dict.json và cập nhật nghĩa tiếng Anh.")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent.parent.parent.parent
    default_labels_dir = current_dir / "backend" / "data" / "dataset" / "khairstyle" / "training" / "labels"
    # default_labels_dir = current_dir / "backend" / "data" / "dataset" / "khairstyle" / "validation" / "labels"

    default_output_dict = Path(__file__).resolve().parent / "mapping_dict.json"
    
    import argparse
    parser = argparse.ArgumentParser(description="Tool trích xuất từ điển tiếng Hàn từ dataset K-Hairstyle")
    parser.add_argument("--dir", type=str, default=str(default_labels_dir), help="Đường dẫn trỏ đến thư mục nhãn (Labels)")
    parser.add_argument("--out", type=str, default=str(default_output_dict), help="Đường dẫn lưu file dictionary JSON xuất ra")
    
    args = parser.parse_args()
    extract_korean_dictionary(args.dir, args.out)
