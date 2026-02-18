"""
Face Detector — YOLOv8-Face cho Training Pipeline.

Phát hiện tất cả khuôn mặt trong ảnh, trả về bounding box + confidence.
Hỗ trợ GPU nếu có.
"""

import os
import numpy as np
from pathlib import Path

from backend.training.utils import setupLogger, getDevice


# Đường dẫn model
BASE_DIR = Path(__file__).resolve().parent.parent.parent
YOLO_MODEL_PATH = str(BASE_DIR / "backend" / "models" / "yolov8n-face.pt")


class TrainingFaceDetector:
    """
    YOLOv8-Face detector cho training pipeline.
    Detect tất cả faces trong ảnh, kể cả partial/profile.
    """
    
    def __init__(self, modelPath=None, confThreshold=0.4):
        """
        Args:
            modelPath: Đường dẫn tới YOLOv8-Face weights (.pt)
            confThreshold: Ngưỡng confidence mặc định
        """
        self.logger = setupLogger("FaceDetector")
        self.modelPath = modelPath or YOLO_MODEL_PATH
        self.confThreshold = confThreshold
        self.model = None
        self.device = getDevice()
        self._loadModel()
    
    def _loadModel(self):
        """Load YOLOv8-Face model."""
        try:
            from ultralytics import YOLO
            
            if not os.path.exists(self.modelPath):
                self.logger.error(f"Model không tìm thấy: {self.modelPath}")
                return
            
            self.model = YOLO(self.modelPath)
            self.logger.info(f"YOLOv8-Face loaded từ {self.modelPath}")
        except ImportError:
            self.logger.error("Thiếu ultralytics. Chạy: pip install ultralytics")
        except Exception as e:
            self.logger.error(f"Lỗi load YOLO model: {e}")
    
    def detect(self, imageCv2, confThreshold=None):
        """
        Detect tất cả khuôn mặt trong ảnh.
        
        Args:
            imageCv2: numpy array (BGR) — ảnh đầu vào
            confThreshold: Ngưỡng confidence (None = dùng default)
        
        Returns:
            list of dict:
                - bbox: [x1, y1, x2, y2] (float)
                - confidence: float (0.0 - 1.0)
        """
        if self.model is None:
            self.logger.error("YOLO model chưa được load")
            return []
        
        threshold = confThreshold or self.confThreshold
        
        try:
            results = self.model(
                imageCv2,
                conf=threshold,
                verbose=False
            )
            
            faces = []
            for result in results:
                if result.boxes is None:
                    continue
                
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                for i in range(len(boxes)):
                    faces.append({
                        "bbox": boxes[i].tolist(),  # [x1, y1, x2, y2]
                        "confidence": float(confs[i])
                    })
            
            # Sắp xếp theo diện tích giảm dần (face lớn nhất trước)
            faces.sort(
                key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
                reverse=True
            )
            
            self.logger.info(f"Phát hiện {len(faces)} khuôn mặt (threshold={threshold})")
            return faces
            
        except Exception as e:
            self.logger.error(f"Lỗi detection: {e}")
            return []
    
    def detectBatch(self, images, confThreshold=None):
        """
        Batch detection cho nhiều ảnh cùng lúc.
        
        Args:
            images: list of numpy array (BGR)
            confThreshold: Ngưỡng confidence
        
        Returns:
            list of list — mỗi ảnh là 1 list faces
        """
        results = []
        for img in images:
            faces = self.detect(img, confThreshold)
            results.append(faces)
        return results
    
    def isAvailable(self):
        """Kiểm tra model đã load thành công chưa."""
        return self.model is not None
