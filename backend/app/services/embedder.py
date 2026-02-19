"""
Embedder — Trích xuất Face Embedding cho Training Pipeline.

Logic theo góc quay:
- |yaw| < 45°: InsightFace ArcFace (512-d)
- |yaw| >= 45°: AdaFace IR-100 (512-d)

LƯU Ý: KHÔNG rotate/frontalize khuôn mặt.
Embedding phản ánh đúng góc mặt gốc.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from backend.app.services.training_utils import setupLogger, getDevice, normalizeEmbedding

# Đường dẫn
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
INSIGHTFACE_ROOT = str(BASE_DIR / "backend")
ADAFACE_MODEL_PATH = str(BASE_DIR / "backend" / "models" / "adaface_ir101_webface4m.ckpt")

# Ngưỡng yaw để chuyển model
YAW_THRESHOLD = 45.0


class TrainingEmbedder:
    """
    Trích xuất face embedding dựa theo góc yaw.
    
    - Pose nhỏ (|yaw| < 45°): InsightFace ArcFace
    - Pose lớn (|yaw| ≥ 45°): AdaFace IR-100
    """
    
    def __init__(self, yawThreshold=YAW_THRESHOLD):
        """
        Args:
            yawThreshold: Ngưỡng yaw để chọn model (mặc định 45°)
        """
        self.logger = setupLogger("Embedder")
        self.device = getDevice()
        self.yawThreshold = yawThreshold
        
        # Models
        self.insightApp = None
        self.adafaceModel = None
        self.mtcnn = None
        
        self._loadInsightFace()
        self._loadAdaFace()
    
    def _loadInsightFace(self):
        """Load InsightFace cho ArcFace embedding extraction."""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            self.insightApp = FaceAnalysis(
                name="antelopev2",
                root=INSIGHTFACE_ROOT,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.insightApp.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info("InsightFace ArcFace loaded cho embedding extraction")
        except Exception as e:
            self.logger.error(f"Lỗi load InsightFace: {e}")
            self.insightApp = None
    
    def _loadAdaFace(self):
        """Load AdaFace IR-100 cho profile face embedding."""
        try:
            # Load MTCNN cho alignment
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(
                image_size=112,
                margin=0,
                min_face_size=20,
                thresholds=[0.5, 0.6, 0.6],
                factor=0.709,
                post_process=False,
                device=self.device
            )
            
            # Load AdaFace model
            if not os.path.exists(ADAFACE_MODEL_PATH):
                self.logger.error(f"AdaFace model không tìm thấy: {ADAFACE_MODEL_PATH}")
                return
            
            from backend.app.services.adaface_ir import iresnet100
            
            model = iresnet100()
            checkpoint = torch.load(ADAFACE_MODEL_PATH, map_location="cpu")
            
            if "state_dict" in checkpoint:
                stateDict = checkpoint["state_dict"]
                stateDict = {k.replace("module.", ""): v for k, v in stateDict.items()}
            else:
                stateDict = checkpoint
            
            model.load_state_dict(stateDict, strict=False)
            model.to(self.device)
            model.eval()
            
            self.adafaceModel = model
            self.logger.info("AdaFace IR-100 loaded cho profile face embedding")
            
        except ImportError as e:
            self.logger.error(f"Thiếu dependency: {e}")
        except Exception as e:
            self.logger.error(f"Lỗi load AdaFace: {e}")
    
    def getEmbedding(self, imageCv2, bbox, yaw):
        """
        Trích xuất embedding theo góc yaw.
        
        KHÔNG rotate/frontalize khuôn mặt.
        
        Args:
            imageCv2: numpy array (BGR) — ảnh gốc
            bbox: [x1, y1, x2, y2]
            yaw: float — góc yaw (độ)
        
        Returns:
            dict:
                - embedding: numpy array (512,) — L2-normalized
                - model_name: str — "InsightFace" hoặc "AdaFace"
            hoặc None nếu extraction thất bại
        """
        absYaw = abs(yaw)
        
        if absYaw < self.yawThreshold:
            # CASE 1: Pose nhỏ → InsightFace ArcFace
            self.logger.info(f"|yaw|={absYaw:.1f}° < {self.yawThreshold}° → Dùng InsightFace")
            embedding = self._extractInsightFace(imageCv2, bbox)
            modelName = "InsightFace"
        else:
            # CASE 2: Pose lớn → AdaFace
            self.logger.info(f"|yaw|={absYaw:.1f}° >= {self.yawThreshold}° → Dùng AdaFace")
            embedding = self._extractAdaFace(imageCv2, bbox)
            modelName = "AdaFace"
        
        if embedding is None:
            # Fallback: thử model còn lại
            self.logger.warning(f"{modelName} extraction thất bại, thử fallback...")
            if modelName == "InsightFace":
                embedding = self._extractAdaFace(imageCv2, bbox)
                modelName = "AdaFace"
            else:
                embedding = self._extractInsightFace(imageCv2, bbox)
                modelName = "InsightFace"
        
        if embedding is None:
            self.logger.error("Cả 2 model đều không extract được embedding")
            return None
        
        # L2 normalize
        embedding = normalizeEmbedding(embedding)
        
        return {
            "embedding": embedding,
            "model_name": modelName
        }
    
    def _extractInsightFace(self, imageCv2, bbox):
        """
        Extract embedding bằng InsightFace ArcFace.
        
        Returns:
            numpy array (512,) hoặc None
        """
        if self.insightApp is None:
            return None
        
        try:
            faces = self.insightApp.get(imageCv2)
            if not faces:
                return None
            
            # Match face với bbox
            bestFace = None
            bestIou = 0
            for face in faces:
                iou = self._computeIou(face.bbox, bbox)
                if iou > bestIou:
                    bestIou = iou
                    bestFace = face
            
            if bestFace is None or bestIou < 0.3:
                return None
            
            # Lấy embedding (đã tích hợp sẵn trong InsightFace)
            if hasattr(bestFace, "embedding") and bestFace.embedding is not None:
                return bestFace.embedding
            
            return None
        except Exception as e:
            self.logger.error(f"Lỗi InsightFace embedding: {e}")
            return None
    
    def _extractAdaFace(self, imageCv2, bbox):
        """
        Extract embedding bằng AdaFace IR-100.
        Crop face → align (MTCNN) → forward model.
        
        Returns:
            numpy array (512,) hoặc None
        """
        if self.adafaceModel is None:
            return None
        
        try:
            # Crop theo bbox với margin
            imageRgb = cv2.cvtColor(imageCv2, cv2.COLOR_BGR2RGB)
            x1, y1, x2, y2 = [int(c) for c in bbox]
            h, w = imageRgb.shape[:2]
            
            # Mở rộng margin 20%
            marginX = int((x2 - x1) * 0.2)
            marginY = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - marginX)
            y1 = max(0, y1 - marginY)
            x2 = min(w, x2 + marginX)
            y2 = min(h, y2 + marginY)
            
            croppedRgb = imageRgb[y1:y2, x1:x2]
            pilImage = Image.fromarray(croppedRgb)
            
            # Align bằng MTCNN
            aligned = None
            if self.mtcnn is not None:
                try:
                    alignedTensor = self.mtcnn(pilImage)
                    if alignedTensor is not None:
                        alignedNp = alignedTensor.permute(1, 2, 0).cpu().numpy()
                        alignedNp = np.clip(alignedNp, 0, 255).astype(np.uint8)
                        aligned = alignedNp
                except Exception:
                    pass
            
            # Fallback: resize trực tiếp nếu MTCNN fail
            if aligned is None:
                aligned = cv2.resize(croppedRgb, (112, 112))
            
            # Chuyển sang input tensor cho AdaFace
            # AdaFace expects BGR, normalized (x - 127.5) / 127.5
            imgBgr = aligned[:, :, ::-1].copy()
            imgNormalized = (imgBgr.astype(np.float32) - 127.5) / 127.5
            imgTensor = torch.from_numpy(imgNormalized.transpose(2, 0, 1))
            imgTensor = imgTensor.unsqueeze(0).float().to(self.device)
            
            # Forward
            with torch.no_grad():
                embedding = self.adafaceModel(imgTensor)
                embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"Lỗi AdaFace embedding: {e}")
            return None
    
    def _computeIou(self, box1, box2):
        """Tính IoU giữa 2 bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def isAvailable(self):
        """Kiểm tra các model đã sẵn sàng."""
        return {
            "insightface": self.insightApp is not None,
            "adaface": self.adafaceModel is not None
        }
