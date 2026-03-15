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

# Hat class = 14 — gộp vào hair để inpainting thay cả nón lẫn tóc
SEGFORMER_HAT_CLASS = 14

# Tất cả classes được coi là "hair" cho mask (tóc + nón)
SEGFORMER_HAIR_CLASSES = {SEGFORMER_HAIR_CLASS, SEGFORMER_HAT_CLASS}

# Face classes (skin, nose, eyes, brows, ears, mouth, lips)
SEGFORMER_FACE_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}


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
        
        # Tạo binary mask
        mask = np.zeros_like(parsing, dtype=np.uint8)
        
        if target_class == 17:
            # Legacy hair class 17 → SegFormer hair classes {13, 14}
            # Bao gồm cả nón (hat) để inpainting thay toàn bộ vùng tóc + nón
            for cls in SEGFORMER_HAIR_CLASSES:
                mask[parsing == cls] = 255
        else:
            mask[parsing == target_class] = 255
        
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
        
        # Hair mask (bao gồm cả nón/hat)
        hair_mask = np.zeros_like(parsing, dtype=np.uint8)
        for cls in SEGFORMER_HAIR_CLASSES:
            hair_mask[parsing == cls] = 255
        
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
    
    def _project_ref_mask_for_bald(self, face_np, ref_np, face_info=None):
        """
        Tạo scalp mask cho user trọc.
        
        Chiến lược 2 tầng:
        1. Nếu có 3D vertices (từ 3DDFA V2) → ngoại suy đỉnh đầu từ forehead vertices
           → scalp mask tự động khớp góc mặt user (yaw/pitch/roll)
        2. Fallback (không có 3D) → crop tóc mẫu, scale theo face width, paste lên trán

        Args:
            face_np: numpy (H, W), uint8 [0,255] — face mask user
            ref_np: numpy (H, W), uint8 [0,255] — hair mask reference
            face_info: face_info object (có thể có .vertices3D)
        Returns:
            PIL Image — mask tóc giả định cho user trọc
        """
        h, w = face_np.shape[:2]
        result = np.zeros((h, w), dtype=np.uint8)
        
        # Tìm bbox khuôn mặt user (cần cho cả 2 tầng)
        face_ys, face_xs = np.where(face_np > 127)
        if len(face_ys) == 0:
            return Image.fromarray(result)
        
        # ============================================================
        # TẦNG 1: 3DDFA V2 Scalp Projection (nếu có vertices3D)
        # ============================================================
        vertices3D = getattr(face_info, 'vertices3D', None) if face_info is not None else None
        
        if vertices3D is not None:
            try:
                result = self._create_scalp_mask_from_3d(vertices3D, face_np, h, w)
                if (result > 127).sum() > 50:  # Scalp mask sinh ra có nội dung
                    self.logger.info(f"Bald: 3DDFA V2 scalp mask generated ({(result > 127).sum()} pixels)")
                    return Image.fromarray(result)
                else:
                    self.logger.info("Bald: 3DDFA V2 scalp mask quá nhỏ, fallback OpenCV")
            except Exception as e:
                self.logger.warning(f"Bald: 3DDFA V2 scalp failed ({e}), fallback OpenCV")
        
        # ============================================================
        # TẦNG 2: OpenCV Fallback (paste tóc mẫu lên trán)
        # ============================================================
        return self._project_ref_mask_opencv(face_np, ref_np)
    
    def _create_scalp_mask_from_3d(self, vertices3D, face_np, h, w):
        """
        Ngoại suy scalp mask từ 3D mesh vertices.
        
        Lấy top 20% vertices (forehead) → tính pháp tuyến lên trên →
        chiếu các điểm "đỉnh đầu" giả lập → convex hull → fill.
        
        Args:
            vertices3D: numpy (3, N) — 3D face mesh vertices
            face_np: numpy (H, W) — face mask
            h, w: image dimensions
        Returns:
            numpy (H, W) uint8 — scalp mask
        """
        result = np.zeros((h, w), dtype=np.uint8)
        
        xs = vertices3D[0, :]  # x coords (image space)
        ys = vertices3D[1, :]  # y coords (image space)
        zs = vertices3D[2, :]  # z coords (depth)
        
        # Lấy top 20% vertices (vùng trán — y nhỏ nhất = cao nhất trên ảnh)
        y_threshold = np.percentile(ys, 20)
        top_mask = ys <= y_threshold
        top_xs = xs[top_mask]
        top_ys = ys[top_mask]
        top_zs = zs[top_mask]
        
        if len(top_xs) < 3:
            return result
        
        # Tính vector "lên trên" dựa trên face normal
        # Trung tâm mặt
        cx_face = np.mean(xs)
        cy_face = np.mean(ys)
        
        # Vector từ tâm mặt đến trung tâm trán (hướng "lên")
        cx_top = np.mean(top_xs)
        cy_top = np.mean(top_ys)
        
        up_dx = cx_top - cx_face
        up_dy = cy_top - cy_face
        up_len = np.sqrt(up_dx**2 + up_dy**2) + 1e-6
        up_dx /= up_len
        up_dy /= up_len
        
        # Khoảng cách chiều cao khuôn mặt (để scale mức ngoại suy)
        face_height = np.max(ys) - np.min(ys)
        
        # Ngoại suy ra 3 lớp điểm đỉnh đầu (30%, 60%, 90% face_height phía trên trán)
        scalp_points = []
        for layer_ratio in [0.3, 0.6, 0.9]:
            extend = face_height * layer_ratio
            for tx, ty in zip(top_xs, top_ys):
                new_x = tx + up_dx * extend
                new_y = ty + up_dy * extend
                scalp_points.append([new_x, new_y])
        
        # Thêm cả điểm trán gốc
        for tx, ty in zip(top_xs, top_ys):
            scalp_points.append([tx, ty])
        
        scalp_points = np.array(scalp_points, dtype=np.int32)
        
        # Clip vào trong ảnh
        scalp_points[:, 0] = np.clip(scalp_points[:, 0], 0, w - 1)
        scalp_points[:, 1] = np.clip(scalp_points[:, 1], 0, h - 1)
        
        # Convex hull → fill
        hull = cv2.convexHull(scalp_points)
        cv2.fillConvexPoly(result, hull, 255)
        
        # Trừ face_mask (bảo vệ khuôn mặt)
        face_buffer = cv2.dilate(face_np, np.ones((10, 10), np.uint8), iterations=2)
        result[face_buffer > 127] = 0
        
        # Smooth biên
        result = cv2.GaussianBlur(result.astype(np.float32), (7, 7), 0)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _project_ref_mask_opencv(self, face_np, ref_np):
        """
        OpenCV fallback cho user trọc: crop tóc mẫu, scale, paste lên trán.
        Dùng khi không có dữ liệu 3D.
        """
        h, w = face_np.shape[:2]
        result = np.zeros((h, w), dtype=np.uint8)
        
        face_ys, face_xs = np.where(face_np > 127)
        if len(face_ys) == 0:
            return Image.fromarray(result)
        
        face_top = np.min(face_ys)
        face_left = np.min(face_xs)
        face_right = np.max(face_xs)
        face_width = face_right - face_left + 1
        face_cx = (face_left + face_right) // 2
        
        ref_ys, ref_xs = np.where(ref_np > 127)
        if len(ref_ys) == 0:
            return Image.fromarray(result)
        
        ref_top = np.min(ref_ys)
        ref_bot = np.max(ref_ys)
        ref_left = np.min(ref_xs)
        ref_right = np.max(ref_xs)
        ref_crop = ref_np[ref_top:ref_bot+1, ref_left:ref_right+1]
        
        scale = (face_width * 1.3) / ref_crop.shape[1]
        new_w = int(ref_crop.shape[1] * scale)
        new_h = int(ref_crop.shape[0] * scale)
        ref_scaled = cv2.resize(ref_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        paste_left = face_cx - new_w // 2
        paste_top = face_top - int(new_h * 0.5)
        
        src_y1 = max(0, -paste_top)
        src_x1 = max(0, -paste_left)
        dst_y1 = max(0, paste_top)
        dst_x1 = max(0, paste_left)
        
        src_y2 = min(new_h, h - dst_y1 + src_y1)
        src_x2 = min(new_w, w - dst_x1 + src_x1)
        
        copy_h = src_y2 - src_y1
        copy_w = src_x2 - src_x1
        
        if copy_h > 0 and copy_w > 0:
            result[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = ref_scaled[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w]
        
        face_buffer = cv2.dilate(face_np, np.ones((10, 10), np.uint8), iterations=2)
        result[face_buffer > 127] = 0
        
        result = cv2.GaussianBlur(result.astype(np.float32), (7, 7), 0)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        self.logger.info(f"Bald OpenCV fallback: face_width={face_width}, scaled_hair={new_w}x{new_h}")
        
        return Image.fromarray(result)
    
    def expand_hair_mask(self, hair_mask_pil, face_mask_pil, ref_hair_mask_pil, face_info=None):
        """
        Dynamic Mask Expansion: mở rộng mask tóc user nếu tóc mẫu lớn hơn.
        
        So sánh diện tích mask tóc reference vs user.
        Nếu ref > user × 1.5 → mở rộng mask theo aspect ratio tóc mẫu.
        Luôn bảo vệ khuôn mặt (3 lớp).
        
        Args:
            hair_mask_pil: PIL Image — mask tóc user (0/255)
            face_mask_pil: PIL Image — mask mặt user (0/255)
            ref_hair_mask_pil: PIL Image — mask tóc reference (0/255)
            face_info: face_info object (optional, có .vertices3D cho bald)
        
        Returns:
            PIL Image — mask tóc đã mở rộng (hoặc giữ nguyên nếu không cần)
        """
        user_np = np.array(hair_mask_pil)
        face_np = np.array(face_mask_pil)
        ref_np = np.array(ref_hair_mask_pil)
        
        # Tính diện tích (số pixel trắng)
        user_area = (user_np > 127).sum()
        ref_area = (ref_np > 127).sum()
        
        # Tính aspect ratio mask tóc reference
        ref_ys, ref_xs = np.where(ref_np > 127)
        if len(ref_ys) == 0:
            return hair_mask_pil
        
        # === TRƯỜNG HỢP ĐẶC BIỆT: User trọc/tóc quá ngắn ===
        if user_area < 100:
            self.logger.info(f"Bald user detected (area={user_area}). Projecting scalp mask.")
            scalp_pil = self._project_ref_mask_for_bald(face_np, ref_np, face_info)
            scalp_np = np.array(scalp_pil)
            
            # Tính lại user_area từ scalp mask mới tạo
            user_np = scalp_np
            user_area = (user_np > 127).sum()
            
            # Nếu scalp mask cũng rỗng (3D + OpenCV đều fail) → trả mask gốc
            if user_area < 50:
                self.logger.warning("Bald: scalp mask rỗng, trả mask gốc.")
                return hair_mask_pil
            
            # KHÔNG return ở đây — cho scalp_mask chảy tiếp vào dilate bên dưới
            # để mở rộng thêm nếu tóc mẫu dài/phồng hơn scalp
        
        # Chỉ mở rộng nếu tóc mẫu lớn hơn đáng kể (× 1.5)
        if ref_area <= user_area * 1.5:
            if user_area < 100:
                # Bald user: scalp mask vừa đủ, không cần dilate thêm
                return Image.fromarray(user_np)
            return hair_mask_pil  # User có tóc, không cần mở rộng
        
        ref_width = np.max(ref_xs) - np.min(ref_xs) + 1
        ref_height = np.max(ref_ys) - np.min(ref_ys) + 1
        ratio = ref_width / (ref_height + 1)
        
        # Chọn kernel theo aspect ratio
        if ratio > 1.2:
            # Tóc phồng → kernel ngang rộng hơn dọc
            kernel = np.ones((8, 20), np.uint8)
        elif ratio < 0.6:
            # Tóc dài → kernel dọc dài hơn ngang
            kernel = np.ones((20, 8), np.uint8)
        else:
            # Cả hai → kernel đều
            kernel = np.ones((15, 15), np.uint8)
        
        # Tính số iterations dựa trên chênh lệch diện tích
        area_ratio = ref_area / (user_area + 1)
        iterations = min(int(area_ratio), 8)  # Tối đa 8 iterations
        iterations = max(1, iterations)
        
        # Dilate mask tóc user
        expanded = cv2.dilate(user_np, kernel, iterations=iterations)
        
        # 3 lớp bảo vệ khuôn mặt:
        # 1. Dilate face_mask thêm 10px làm buffer
        face_buffer = cv2.dilate(face_np, np.ones((10, 10), np.uint8), iterations=2)
        
        # 2. Trừ face_mask khỏi expanded mask
        expanded[face_buffer > 127] = 0
        
        # 3. Smooth biên bằng Gaussian blur
        expanded = cv2.GaussianBlur(expanded.astype(np.float32), (7, 7), 0)
        expanded = np.clip(expanded, 0, 255).astype(np.uint8)
        
        # Đảm bảo vùng tóc gốc luôn được giữ
        expanded = np.maximum(expanded, user_np)
        
        self.logger.info(
            f"Mask expanded: user_area={user_area}, ref_area={ref_area}, "
            f"ratio={ratio:.2f}, iterations={iterations}"
        )
        
        return Image.fromarray(expanded)
