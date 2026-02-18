"""
Pose Estimator — Landmark & Head Pose Estimation cho Training Pipeline.

Pipeline:
1. InsightFace (antelopev2) → 106 landmarks
2. 3DDFA V2 → yaw, pitch, roll từ 3DMM parameters

Fallback: nếu InsightFace miss face → dùng 3DDFA trực tiếp với bbox.
"""

import sys
import os
import numpy as np
from pathlib import Path

from backend.training.utils import setupLogger, getDevice

# Đường dẫn base
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TDDFA_DIR = str(BASE_DIR / "backend" / "models" / "3ddfa_v2")
INSIGHTFACE_ROOT = str(BASE_DIR / "backend")


class TrainingPoseEstimator:
    """
    Ước lượng landmarks 106 điểm và head pose (yaw/pitch/roll).
    
    Kết hợp InsightFace cho landmarks chính xác
    và 3DDFA V2 cho pose estimation robust.
    """
    
    def __init__(self):
        self.logger = setupLogger("PoseEstimator")
        self.device = getDevice()
        self.insightApp = None
        self.tddfa = None
        self._loadInsightFace()
        self._loadTDDFA()
    
    def _loadInsightFace(self):
        """Load InsightFace antelopev2 cho 106-point landmarks."""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            self.insightApp = FaceAnalysis(
                name="antelopev2",
                root=INSIGHTFACE_ROOT,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.insightApp.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info("InsightFace antelopev2 loaded cho landmark extraction")
        except Exception as e:
            self.logger.error(f"Lỗi load InsightFace: {e}")
            self.insightApp = None
    
    def _loadTDDFA(self):
        """Load 3DDFA V2 cho pose estimation."""
        try:
            # Thêm 3DDFA vào sys.path để import đúng
            if TDDFA_DIR not in sys.path:
                sys.path.insert(0, TDDFA_DIR)
            
            import yaml
            from TDDFA import TDDFA
            
            cfgPath = os.path.join(TDDFA_DIR, "configs", "mb1_120x120.yml")
            with open(cfgPath, "r") as f:
                cfg = yaml.safe_load(f)
            
            # Chuyển relative paths thành absolute paths
            cfg["checkpoint_fp"] = os.path.join(TDDFA_DIR, cfg["checkpoint_fp"])
            cfg["bfm_fp"] = os.path.join(TDDFA_DIR, cfg["bfm_fp"])
            
            # Sử dụng GPU nếu có
            gpuMode = str(self.device) == "cuda"
            cfg["gpu_mode"] = gpuMode
            
            self.tddfa = TDDFA(**cfg)
            self.logger.info("3DDFA V2 loaded cho pose estimation")
        except Exception as e:
            self.logger.error(f"Lỗi load 3DDFA V2: {e}")
            self.tddfa = None
    
    def estimate(self, imageCv2, bbox):
        """
        Ước lượng landmarks và head pose cho 1 khuôn mặt.
        
        Args:
            imageCv2: numpy array (BGR)
            bbox: [x1, y1, x2, y2] từ face detector
        
        Returns:
            dict:
                - landmarks_106: numpy array (106, 2) hoặc None
                - yaw: float (độ)
                - pitch: float (độ)
                - roll: float (độ)
                - method: str — phương pháp đã dùng
        """
        result = {
            "landmarks_106": None,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "method": "none"
        }
        
        # Bước 1: Thử InsightFace cho 106 landmarks
        landmarks106 = self._getInsightFaceLandmarks(imageCv2, bbox)
        if landmarks106 is not None:
            result["landmarks_106"] = landmarks106
        
        # Bước 2: Dùng 3DDFA V2 cho pose (yaw/pitch/roll)
        poseResult = self._getTDDFAPose(imageCv2, bbox)
        if poseResult is not None:
            result["yaw"] = poseResult["yaw"]
            result["pitch"] = poseResult["pitch"]
            result["roll"] = poseResult["roll"]
            result["method"] = "3DDFA"
        
        self.logger.info(
            f"Pose: yaw={result['yaw']:.1f}° pitch={result['pitch']:.1f}° "
            f"roll={result['roll']:.1f}° method={result['method']}"
        )
        return result
    
    def _getInsightFaceLandmarks(self, imageCv2, bbox):
        """
        Trích xuất 106 landmarks bằng InsightFace.
        
        Returns:
            numpy array (106, 2) hoặc None
        """
        if self.insightApp is None:
            return None
        
        try:
            faces = self.insightApp.get(imageCv2)
            if not faces:
                self.logger.warning("InsightFace không detect được face")
                return None
            
            # Tìm face có IoU cao nhất với bbox
            bestFace = self._matchFaceToBbox(faces, bbox)
            if bestFace is None:
                return None
            
            # Lấy 106 landmarks (nếu có)
            if hasattr(bestFace, "landmark_2d_106") and bestFace.landmark_2d_106 is not None:
                landmarks = bestFace.landmark_2d_106
                self.logger.info(f"Trích xuất {len(landmarks)} landmarks từ InsightFace")
                return landmarks
            
            # Fallback về 5 landmarks nếu không có 106
            if hasattr(bestFace, "kps") and bestFace.kps is not None:
                self.logger.warning("Chỉ có 5 landmarks, không có 106")
                return None
            
            return None
        except Exception as e:
            self.logger.error(f"Lỗi InsightFace landmarks: {e}")
            return None
    
    def _getTDDFAPose(self, imageCv2, bbox):
        """
        Tính yaw/pitch/roll bằng 3DDFA V2.
        
        Returns:
            dict {yaw, pitch, roll} hoặc None
        """
        if self.tddfa is None:
            return None
        
        try:
            # Chuẩn bị tham số cho 3DDFA
            # Import pose utils từ 3DDFA
            if TDDFA_DIR not in sys.path:
                sys.path.insert(0, TDDFA_DIR)
            from utils.pose import calc_pose
            
            # 3DDFA cần bbox format [x1, y1, x2, y2]
            bboxForTddfa = [bbox]
            
            paramLst, roiBoxLst = self.tddfa(imageCv2, bboxForTddfa)
            
            if not paramLst:
                return None
            
            # Tính pose từ 3DMM params
            P, pose = calc_pose(paramLst[0])
            yaw, pitch, roll = pose[0], pose[1], pose[2]
            
            return {
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll)
            }
        except Exception as e:
            self.logger.error(f"Lỗi 3DDFA pose: {e}")
            return None
    
    def _matchFaceToBbox(self, insightFaces, targetBbox):
        """
        Tìm InsightFace result có IoU cao nhất với target bbox.
        
        Args:
            insightFaces: list InsightFace face objects
            targetBbox: [x1, y1, x2, y2]
        
        Returns:
            Best matching face object hoặc None
        """
        bestFace = None
        bestIou = 0
        
        for face in insightFaces:
            faceBbox = face.bbox
            iou = self._computeIou(faceBbox, targetBbox)
            if iou > bestIou:
                bestIou = iou
                bestFace = face
        
        if bestIou < 0.3:
            self.logger.warning(f"IoU quá thấp ({bestIou:.2f}), skip InsightFace match")
            return None
        
        return bestFace
    
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
    
    def estimateBatch(self, imageCv2, bboxes):
        """
        Batch estimation cho nhiều faces trong 1 ảnh.
        
        Args:
            imageCv2: numpy array (BGR)
            bboxes: list of [x1, y1, x2, y2]
        
        Returns:
            list of dict — mỗi face 1 result
        """
        results = []
        for bbox in bboxes:
            result = self.estimate(imageCv2, bbox)
            results.append(result)
        return results
    
    def isAvailable(self):
        """Kiểm tra các model đã sẵn sàng chưa."""
        return {
            "insightface": self.insightApp is not None,
            "tddfa": self.tddfa is not None
        }
