"""
AdaFace Service - Quality Adaptive Margin for Face Recognition

Sử dụng AdaFace (CVPR 2022) để extract face embedding từ profile/partial faces.
AdaFace xử lý tốt hơn các trường hợp:
- Low quality images
- Profile/side faces 
- Partial occlusions

Reference: https://github.com/mk-minchul/AdaFace
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path

from backend.app.config import model_paths, settings


class AdaFaceService:
    """
    AdaFace embedding extraction service.
    Có thể extract embedding từ profile faces khi InsightFace fails.
    """
    
    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.model = None
        self.mtcnn = None
        self._load_model()
    
    def _load_model(self):
        """Load AdaFace model và MTCNN for alignment"""
        try:
            # Load MTCNN for face alignment
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(
                image_size=112,
                margin=0,
                min_face_size=20,
                thresholds=[0.5, 0.6, 0.6],  # Giảm threshold để detect partial faces
                factor=0.709,
                post_process=False,  # Không normalize, để ta xử lý
                device=self.device
            )
            print("[AdaFaceService] MTCNN loaded for face alignment")
            
            # Load AdaFace model
            model_path = model_paths.ADAFACE_MODEL
            if os.path.exists(model_path):
                self.model = self._load_pretrained_model(model_path)
                self.model.to(self.device)
                self.model.eval()
                print(f"[AdaFaceService] Model loaded from {model_path}")
            else:
                print(f"[AdaFaceService] Model not found at {model_path}")
                print("[AdaFaceService] Run: python backend/models/download_adaface.py")
                self.model = None
                
        except ImportError as e:
            print(f"[AdaFaceService] Missing dependency: {e}")
            print("[AdaFaceService] Run: pip install facenet-pytorch")
            self.mtcnn = None
            self.model = None
        except Exception as e:
            print(f"[AdaFaceService] Error loading model: {e}")
            self.model = None
    
    def _load_pretrained_model(self, model_path):
        """Load AdaFace pretrained model (IR backbone)"""
        # AdaFace sử dụng IR (Improved ResNet) backbone
        # Đây là simplified version, load từ checkpoint
        from backend.app.services.adaface_ir import iresnet100
        
        model = iresnet100()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'module.' prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        return model
    
    def _to_input(self, aligned_face_rgb):
        """
        Convert aligned face (RGB PIL Image) to model input tensor.
        AdaFace expects BGR 112x112, normalized with mean=0.5, std=0.5
        """
        # Convert to numpy
        if isinstance(aligned_face_rgb, Image.Image):
            img = np.array(aligned_face_rgb)
        else:
            img = aligned_face_rgb
            
        # Ensure 112x112
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        
        # RGB to BGR (AdaFace expects BGR)
        img_bgr = img[:, :, ::-1].copy()
        
        # Normalize: (img - 127.5) / 127.5 = img/127.5 - 1
        img_normalized = (img_bgr.astype(np.float32) - 127.5) / 127.5
        
        # HWC to CHW, add batch dimension
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1))
        img_tensor = img_tensor.unsqueeze(0).float()
        
        return img_tensor.to(self.device)
    
    def align_face(self, image_cv2, bbox=None):
        """
        Align face using MTCNN.
        
        Args:
            image_cv2: OpenCV image (BGR)
            bbox: Optional [x1, y1, x2, y2] from YOLO để crop trước
            
        Returns:
            PIL Image 112x112 (RGB) hoặc None
        """
        if self.mtcnn is None:
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        
        # Crop theo bbox nếu có (từ YOLO detection)
        if bbox is not None:
            x1, y1, x2, y2 = [int(c) for c in bbox]
            # Mở rộng margin
            h, w = image_rgb.shape[:2]
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)
            image_rgb = image_rgb[y1:y2, x1:x2]
        
        # Convert to PIL
        pil_image = Image.fromarray(image_rgb)
        
        try:
            # MTCNN extract aligned face
            # Returns tensor 3x112x112 or None
            aligned = self.mtcnn(pil_image)
            
            if aligned is not None:
                # Convert tensor to PIL
                # MTCNN returns tensor in range [0, 255] if post_process=False
                aligned_np = aligned.permute(1, 2, 0).cpu().numpy()
                aligned_np = np.clip(aligned_np, 0, 255).astype(np.uint8)
                return Image.fromarray(aligned_np)
            
        except Exception as e:
            print(f"[AdaFaceService] Alignment error: {e}")
        
        return None
    
    def get_embedding(self, image_cv2, bbox=None):
        """
        Extract face embedding từ ảnh.
        
        Args:
            image_cv2: OpenCV image (BGR)
            bbox: Optional [x1, y1, x2, y2] từ YOLO detection
            
        Returns:
            numpy array 512-d embedding hoặc None
        """
        if self.model is None:
            return None
        
        # Align face
        aligned = self.align_face(image_cv2, bbox)
        if aligned is None:
            return None
        
        # Convert to model input
        input_tensor = self._to_input(aligned)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(input_tensor)
            
            # Normalize
            embedding = F.normalize(embedding, p=2, dim=1)
            
        return embedding.cpu().numpy().flatten()
    
    def get_embedding_from_aligned(self, aligned_pil):
        """
        Extract embedding từ ảnh đã align sẵn.
        
        Args:
            aligned_pil: PIL Image 112x112 RGB
            
        Returns:
            numpy array 512-d embedding
        """
        if self.model is None:
            return None
        
        input_tensor = self._to_input(aligned_pil)
        
        with torch.no_grad():
            embedding = self.model(input_tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
            
        return embedding.cpu().numpy().flatten()
    
    def is_available(self):
        """Check if AdaFace model is loaded"""
        return self.model is not None and self.mtcnn is not None
