import os
import json
import time
import logging
from pathlib import Path
from deep_translator import GoogleTranslator

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AutoTranslate")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DICT_PATH = Path(__file__).resolve().parent / "mapping_dict.json"

def load_dict():
    if not DICT_PATH.exists():
        return {}
    try:
        with open(DICT_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Lỗi đọc file dict: {e}")
        return {}

def save_dict(data):
    try:
        with open(DICT_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Lỗi lưu file dict: {e}")

def translate_empty_words():
    logger.info(f"Đang kiểm tra từ điển tại: {DICT_PATH}")
    
    translator = GoogleTranslator(source='ko', target='en')
    dictionary = load_dict()
    
    if not dictionary:
        logger.warning("Từ điển đang trống hoặc chưa được tạo ra!")
        return

    empty_keys = [k for k, v in dictionary.items() if str(v).strip() == ""]
    
    if not empty_keys:
        logger.info("Tuyệt vời! Tất cả các từ vựng trong từ điển đều ĐÃ ĐƯỢC DỊCH.")
        return
        
    logger.info(f"Phát hiện {len(empty_keys)} từ vựng tiếng Hàn chưa có nghĩa tiếng Anh. Bắt đầu dịch...")
    
    translated_count = 0
    # Dịch từng từ một (nếu dùng batching có thể nhanh hơn nhưng dễ bị limit, dịch chậm an toàn hơn)
    for word in empty_keys:
        try:
            # Gửi API dịch
            english_meaning = translator.translate(word)
            
            if english_meaning:
                # Chuẩn hóa kết quả dịch sang chữ thường
                english_meaning = english_meaning.lower().strip()
                dictionary[word] = english_meaning
                translated_count += 1
                logger.info(f"  [+] Đã dịch: {word} -> {english_meaning}")
                
                # Lưu liên tục sau mỗi từ dịch được (tránh bị block API giữa chừng mất kết quả)
                save_dict(dictionary)
                
            # Nghỉ 0.5s để tránh request limit từ Google Free API
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"  [-] Lỗi khi dịch từ '{word}': {e}")
            time.sleep(2) # Đợi lâu hơn nếu có lỗi (ví dụ block)

    logger.info("="*40)
    logger.info(f"Hoàn thành dịch tự động {translated_count}/{len(empty_keys)} từ!")
    logger.info("="*40)

def loop_monitor(interval_seconds=30):
    """ Hàm chạy vòng lặp để liên tục canh me từ mới """
    logger.info(f"Bắt đầu trình giám sát dịch tự động (Mỗi {interval_seconds}s check 1 lần)")
    try:
        while True:
            translate_empty_words()
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Đã tắt tiến trình dịch tự động.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tool tự động dịch mảng mapping_dict.json sang Tiếng Anh")
    parser.add_argument("--loop", action="store_true", help="Chạy ở chế độ giám sát vòng lặp liên tục")
    
    args = parser.parse_args()
    
    if args.loop:
        loop_monitor(30)
    else:
        translate_empty_words()
