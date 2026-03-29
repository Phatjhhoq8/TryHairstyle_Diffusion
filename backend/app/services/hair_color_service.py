"""
Hair Color Service — Đổi màu tóc bằng HSV Color Transfer.

Sử dụng hair mask (từ SegFormer) để blend màu mới lên vùng tóc.
Giữ nguyên texture (Value trong HSV) để kết quả tự nhiên.
"""

import cv2
import numpy as np
from PIL import Image


# Bảng màu preset phổ biến — key: tên màu, value: (H, S trong HSV range 0-179, 0-255)
PRESET_COLORS = {
    "black": {"hex": "#1a1a1a", "hsv": (0, 10, 30), "label": "Đen", "prompt": "black"},
    "dark_brown": {"hex": "#3b2213", "hsv": (15, 180, 60), "label": "Nâu đậm", "prompt": "dark brown"},
    "brown": {"hex": "#6b3a2a", "hsv": (12, 160, 110), "label": "Nâu", "prompt": "brown"},
    "light_brown": {"hex": "#a0673a", "hsv": (20, 150, 160), "label": "Nâu sáng", "prompt": "light brown"},
    "blonde": {"hex": "#d4a54a", "hsv": (28, 160, 210), "label": "Vàng (Blonde)", "prompt": "blonde"},
    "platinum": {"hex": "#e8dcc8", "hsv": (30, 40, 230), "label": "Bạch kim", "prompt": "platinum blonde"},
    "red": {"hex": "#8b2500", "hsv": (6, 220, 140), "label": "Đỏ", "prompt": "red"},
    "auburn": {"hex": "#922724", "hsv": (2, 190, 146), "label": "Nâu đỏ (Auburn)", "prompt": "auburn"},
    "ginger": {"hex": "#c45e28", "hsv": (16, 200, 196), "label": "Cam gừng", "prompt": "ginger"},
    "pink": {"hex": "#d4608a", "hsv": (165, 140, 212), "label": "Hồng", "prompt": "pink"},
    "blue": {"hex": "#2a52be", "hsv": (110, 190, 190), "label": "Xanh dương", "prompt": "blue"},
    "purple": {"hex": "#6a0dad", "hsv": (135, 220, 170), "label": "Tím", "prompt": "purple"},
    "green": {"hex": "#2e8b57", "hsv": (75, 180, 140), "label": "Xanh lá", "prompt": "green"},
    "silver": {"hex": "#c0c0c0", "hsv": (0, 5, 192), "label": "Bạc", "prompt": "silver"},
    "white": {"hex": "#f0ede5", "hsv": (30, 10, 240), "label": "Trắng", "prompt": "white"},
}

COLOR_NAME_ALIASES = {
    "den": "black",
    "đen": "black",
    "nau dam": "dark_brown",
    "nâu đậm": "dark_brown",
    "nau": "brown",
    "nâu": "brown",
    "nau sang": "light_brown",
    "nâu sáng": "light_brown",
    "vang": "blonde",
    "vàng": "blonde",
    "bach kim": "platinum",
    "bạch kim": "platinum",
    "do": "red",
    "đỏ": "red",
    "nau do": "auburn",
    "nâu đỏ": "auburn",
    "cam gung": "ginger",
    "cam gừng": "ginger",
    "hong": "pink",
    "hồng": "pink",
    "xanh duong": "blue",
    "xanh dương": "blue",
    "tim": "purple",
    "tím": "purple",
    "xanh la": "green",
    "xanh lá": "green",
    "bac": "silver",
    "bạc": "silver",
    "trang": "white",
    "trắng": "white",
}


class HairColorService:
    """
    Service đổi màu tóc bằng kỹ thuật HSV Color Transfer.

    Thuật toán:
    1. Chuyển ảnh sang HSV
    2. Trong vùng hair mask: thay H (Hue) + S (Saturation) bằng màu target
    3. Giữ nguyên V (Value/Brightness) → bảo toàn texture tóc
    4. Blend mượt bằng Gaussian blur trên biên mask
    """

    def colorize(
        self,
        image_pil: Image.Image,
        hair_mask_pil: Image.Image,
        target_color: str,
        intensity: float = 0.7
    ) -> Image.Image:
        """
        Áp dụng màu tóc mới lên vùng tóc.

        Args:
            image_pil: Ảnh gốc (PIL RGB)
            hair_mask_pil: Hair mask (PIL grayscale, 0/255)
            target_color: Tên màu preset (vd: 'blonde') hoặc hex code (vd: '#FF0000')
            intensity: Mức độ đậm/nhạt (0.0 = giữ nguyên, 1.0 = 100% màu mới)

        Returns:
            PIL Image — ảnh đã đổi màu tóc
        """
        # Clamp intensity
        intensity = max(0.0, min(1.0, intensity))
        if intensity == 0.0:
            return image_pil.copy()

        # Parse target color → HSV values
        targetH, targetS, targetV = self._parseColor(target_color)

        # Convert ảnh sang numpy
        imgRgb = np.array(image_pil)
        imgHsv = cv2.cvtColor(imgRgb, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Chuẩn bị mask
        maskNp = np.array(hair_mask_pil.convert("L"))
        maskFloat = (maskNp / 255.0).astype(np.float32)

        # Gaussian blur biên mask → blend mượt, tránh viền cứng
        maskBlur = cv2.GaussianBlur(maskFloat, (21, 21), 10)

        # Tạo ảnh HSV đã đổi màu
        colorizedHsv = imgHsv.copy()

        # Thay Hue
        colorizedHsv[:, :, 0] = targetH

        # Blend Saturation: giữ 1 phần saturation gốc để tự nhiên hơn
        originalS = imgHsv[:, :, 1]
        colorizedHsv[:, :, 1] = originalS * (1 - intensity * 0.8) + targetS * (intensity * 0.8)

        # Điều chỉnh Value nhẹ để phản ánh mức độ sáng/tối của màu target
        originalV = imgHsv[:, :, 2]
        # Tính ratio V target/V trung bình gốc (giới hạn 0.6–1.4 để không quá tối/sáng)
        avgOriginalV = np.mean(originalV[maskNp > 128]) if np.any(maskNp > 128) else 128
        vRatio = np.clip(targetV / (avgOriginalV + 1e-8), 0.6, 1.4)
        colorizedHsv[:, :, 2] = np.clip(originalV * (1 - intensity * 0.3 + intensity * 0.3 * vRatio), 0, 255)

        # Convert về RGB
        colorizedHsv = np.clip(colorizedHsv, 0, 255).astype(np.uint8)
        colorizedRgb = cv2.cvtColor(colorizedHsv, cv2.COLOR_HSV2RGB)

        # Blend: chỉ áp dụng trong vùng mask (với biên mờ)
        maskBlur3ch = np.stack([maskBlur * intensity] * 3, axis=-1)
        result = (colorizedRgb * maskBlur3ch + imgRgb * (1 - maskBlur3ch)).astype(np.uint8)

        return Image.fromarray(result)

    def _parseColor(self, color: str) -> tuple:
        """
        Parse tên màu hoặc hex code → (H, S, V) trong OpenCV range.

        Args:
            color: Tên preset ('blonde') hoặc hex ('#FF0000')

        Returns:
            tuple (H: 0-179, S: 0-255, V: 0-255)
        """
        # Kiểm tra preset
        colorKey = self.normalize_color_name(color)
        if colorKey in PRESET_COLORS:
            return PRESET_COLORS[colorKey]["hsv"]

        # Parse hex code
        if colorKey.startswith("#") and len(colorKey) in (4, 7):
            return self._hexToHsv(colorKey)

        # Fallback: trả về nâu đậm
        print(f"⚠️ Màu '{color}' không nhận dạng được, dùng mặc định nâu đậm")
        return PRESET_COLORS["dark_brown"]["hsv"]

    @staticmethod
    def normalize_color_name(color: str) -> str:
        if not color:
            return ""

        colorLower = color.lower().strip()
        if colorLower in PRESET_COLORS:
            return colorLower

        return COLOR_NAME_ALIASES.get(colorLower, colorLower)

    @staticmethod
    def get_prompt_color_name(color: str) -> str:
        if not color:
            return ""

        normalized = HairColorService.normalize_color_name(color)
        if normalized in PRESET_COLORS:
            return PRESET_COLORS[normalized].get("prompt", normalized.replace("_", " "))

        if normalized.startswith("#"):
            return normalized

        return normalized.replace("_", " ")

    def _hexToHsv(self, hexColor: str) -> tuple:
        """
        Convert hex color → HSV (OpenCV range).

        Args:
            hexColor: vd '#FF0000' hoặc '#F00'

        Returns:
            tuple (H: 0-179, S: 0-255, V: 0-255)
        """
        hexColor = hexColor.lstrip("#")
        if len(hexColor) == 3:
            hexColor = "".join([c * 2 for c in hexColor])

        r = int(hexColor[0:2], 16)
        g = int(hexColor[2:4], 16)
        b = int(hexColor[4:6], 16)

        # Tạo pixel RGB → convert sang HSV
        pixel = np.array([[[r, g, b]]], dtype=np.uint8)
        hsvPixel = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)

        return (
            int(hsvPixel[0, 0, 0]),
            int(hsvPixel[0, 0, 1]),
            int(hsvPixel[0, 0, 2])
        )

    @staticmethod
    def getPresetColors() -> dict:
        """
        Trả về danh sách màu preset.

        Returns:
            dict: {color_name: {hex, label}}
        """
        return {
            name: {"hex": info["hex"], "label": info["label"]}
            for name, info in PRESET_COLORS.items()
        }
