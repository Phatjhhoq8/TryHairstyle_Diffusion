"""
Segmentation Service — SegFormer cho face/hair segmentation.

Sử dụng SegFormer (jonathandinu/face-parsing) cho face/hair segmentation.
Cung cấp API tương thích ngược: get_mask(image_pil, target_class)
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from backend.app.services.training_utils import setupLogger, getDevice


# SegFormer config
SEGFORMER_MODEL_ID = "jonathandinu/face-parsing"
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SEGFORMER_LOCAL_PATH = str(BASE_DIR / "backend" / "models" / "segformer_face_parsing")

# SegFormer class mapping (jonathandinu/face-parsing)
# 0: background
# 1: skin, 2: nose, 3: eye_g (glasses), 4: l_eye, 5: r_eye
# 6: l_brow, 7: r_brow, 8: l_ear, 9: r_ear, 10: mouth
# 11: u_lip, 12: l_lip, 13: hair, 14: hat, 15: earring
# 16: necklace, 17: neck, 18: cloth

# Hair class trong SegFormer = 13
SEGFORMER_HAIR_CLASS = 13

# Face classes (skin, nose, eyes, brows, mouth, lips)
SEGFORMER_FACE_CLASSES = {1, 2, 3, 4, 5, 6, 7, 10, 11, 12}


class SegmentationService:
    """
    SegFormer-based segmentation service.
    
    SegFormer-based segmentation service.
    API: get_mask(image_pil, target_class)
    """
    
    def __init__(self):
        self.logger = setupLogger("Segmentation")
        self.device = getDevice()
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load SegFormer model."""
        try:
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            import os
            
            # Thử load local trước
            if os.path.exists(SEGFORMER_LOCAL_PATH):
                self.processor = SegformerImageProcessor.from_pretrained(SEGFORMER_LOCAL_PATH)
                self.model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_LOCAL_PATH)
                self.logger.info(f"SegFormer loaded từ local: {SEGFORMER_LOCAL_PATH}")
            else:
                self.processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL_ID)
                self.model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_ID)
                self.logger.info(f"SegFormer loaded từ HuggingFace: {SEGFORMER_MODEL_ID}")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            self.logger.error(f"Lỗi load SegFormer: {e}")
            self.model = None
    
    def get_parsing(self, image_cv2):
        """
        Chạy SegFormer trên ảnh và trả về parsing map.
        
        Args:
            image_cv2: numpy array (BGR)
        
        Returns:
            numpy array (H, W) — class ID cho mỗi pixel
        """
        if self.model is None or self.processor is None:
            return None
        
        try:
            import torch.nn.functional as F
            
            imageRgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
            pilImage = Image.fromarray(imageRgb)
            
            inputs = self.processor(images=pilImage, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                h, w = image_cv2.shape[:2]
                upsampled = F.interpolate(
                    logits, size=(h, w),
                    mode="bilinear", align_corners=False
                )
                parsing = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()
            
            return parsing
            
        except Exception as e:
            self.logger.error(f"Lỗi SegFormer inference: {e}")
            return None
    
    def get_mask(self, image_pil, target_class=17):
        """
        Trả về binary mask cho target class.
        
        Args:
            image_pil: PIL Image
            target_class: Class ID (17 = hair, auto-map sang SegFormer class 13)
        
        Returns:
            PIL Image — binary mask (0/255)
        """
        w, h = image_pil.size
        
        # Convert PIL → CV2
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Chạy SegFormer
        parsing = self.get_parsing(image_cv2)
        
        if parsing is None:
            # Fallback: trả về mask trống
            return Image.fromarray(np.zeros((h, w), dtype=np.uint8))
        
        # Map legacy class → SegFormer class
        if target_class == 17:
            # Legacy hair class 17 → SegFormer hair class 13
            segformer_class = SEGFORMER_HAIR_CLASS
        else:
            segformer_class = target_class
        
        # Tạo binary mask
        mask = np.zeros_like(parsing, dtype=np.uint8)
        mask[parsing == segformer_class] = 255
        
        # Resize nếu cần
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Dilate để mở rộng biên
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=2)
        
        return Image.fromarray(mask_dilated)
    
    def get_hair_and_face_mask(self, image_pil):
        """
        Trả về cả hair mask và face mask.
        
        Args:
            image_pil: PIL Image
        
        Returns:
            dict:
                - hair_mask: PIL Image (binary mask)
                - face_mask: PIL Image (binary mask)
                - parsing: numpy array (H, W) — full parsing map
        """
        w, h = image_pil.size
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        parsing = self.get_parsing(image_cv2)
        
        if parsing is None:
            empty = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
            return {"hair_mask": empty, "face_mask": empty, "parsing": None}
        
        # Hair mask
        hair_mask = np.zeros_like(parsing, dtype=np.uint8)
        hair_mask[parsing == SEGFORMER_HAIR_CLASS] = 255
        
        # Face mask (tổng hợp tất cả face classes)
        face_mask = np.zeros_like(parsing, dtype=np.uint8)
        for cls in SEGFORMER_FACE_CLASSES:
            face_mask[parsing == cls] = 255
        
        # Dilate
        kernel = np.ones((5, 5), np.uint8)
        hair_mask = cv2.dilate(hair_mask, kernel, iterations=2)
        face_mask = cv2.dilate(face_mask, kernel, iterations=1)
        
        return {
            "hair_mask": Image.fromarray(hair_mask),
            "face_mask": Image.fromarray(face_mask),
            "parsing": parsing
        }
