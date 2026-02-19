"""
Face Detector — YOLOv8-Face cho Training Pipeline.

Phát hiện tất cả khuôn mặt trong ảnh, trả về bounding box + confidence.
Hỗ trợ GPU nếu có.
"""

import os
import numpy as np
from pathlib import Path

from backend.app.services.training_utils import setupLogger, getDevice


# Đường dẫn model
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
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
            
            # Sắp xếp theo confidence giảm dần
            faces.sort(key=lambda f: f["confidence"], reverse=True)
            
            # NMS: loại bỏ bbox trùng lặp (IoU > 0.5)
            faces = self._nmsFilter(faces, iouThreshold=0.5)
            
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
    
    def _nmsFilter(self, faces, iouThreshold=0.5):
        """
        Non-Maximum Suppression — loại bỏ bbox trùng lặp.
        
        Khi 2 bbox có IoU > threshold, giữ cái confidence cao hơn.
        Input đã sort theo confidence giảm dần.
        
        Args:
            faces: list of dict (đã sort theo confidence giảm dần)
            iouThreshold: float — ngưỡng IoU để coi là trùng
        
        Returns:
            list of dict — faces đã loại bỏ trùng lặp
        """
        if len(faces) <= 1:
            return faces
        
        keep = []
        suppressed = set()
        
        for i, faceA in enumerate(faces):
            if i in suppressed:
                continue
            keep.append(faceA)
            
            bA = faceA["bbox"]
            for j in range(i + 1, len(faces)):
                if j in suppressed:
                    continue
                bB = faces[j]["bbox"]
                
                iou = self._computeIoU(bA, bB)
                if iou > iouThreshold:
                    suppressed.add(j)
                    self.logger.info(
                        f"  NMS: loại bbox trùng (IoU={iou:.2f}, conf={faces[j]['confidence']:.3f})"
                    )
        
        return keep
    
    def _computeIoU(self, boxA, boxB):
        """Tính IoU giữa 2 bbox [x1, y1, x2, y2]."""
        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[2], boxB[2])
        y2 = min(boxA[3], boxB[3])
        
        interW = max(0, x2 - x1)
        interH = max(0, y2 - y1)
        interArea = interW * interH
        
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        union = areaA + areaB - interArea
        if union <= 0:
            return 0.0
        
        return interArea / union
    
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
