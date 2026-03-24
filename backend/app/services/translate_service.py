"""
Dịch prompt từ tiếng Việt sang tiếng Anh bằng Google Translate (deep-translator).
Mục đích: Stable Diffusion chỉ hiểu prompt tiếng Anh, service này giúp
người dùng nhập tiếng Việt mà vẫn có kết quả tốt.
"""

from deep_translator import GoogleTranslator


def translate_vi_to_en(text: str) -> str:
    """
    Dịch text tiếng Việt sang tiếng Anh.
    Fallback: nếu API lỗi → trả về text gốc (không block pipeline).
    
    Args:
        text: Prompt tiếng Việt cần dịch
        
    Returns:
        Prompt đã dịch sang tiếng Anh
    """
    if not text or not text.strip():
        return text
    
    try:
        translated = GoogleTranslator(source='vi', target='en').translate(text.strip())
        print(f"  🌐 Translated: '{text.strip()}' → '{translated}'")
        return translated
    except Exception as e:
        # Fallback: trả về text gốc nếu dịch lỗi
        print(f"  ⚠️ Translation failed: {e}. Using original text.")
        return text
