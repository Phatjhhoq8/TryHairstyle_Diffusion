"""
Reconstructor 3D — 3DDFA V2 Wrapper cho Training Pipeline.

Dựng 3D Morphable Model (3DMM) và reconstruct dense 3D face mesh.
Giữ nguyên góc mặt (KHÔNG xoay về frontal).
"""

import sys
import os
import numpy as np
from pathlib import Path

from backend.training.utils import setupLogger, getDevice

# Đường dẫn
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TDDFA_DIR = str(BASE_DIR / "backend" / "models" / "3ddfa_v2")


class TrainingReconstructor3D:
    """
    3DDFA V2 wrapper cho 3D face reconstruction.
    
    Chức năng:
    - Reconstruct dense 3D face mesh
    - Trả về 3DMM parameters
    - Trả về 3D vertices
    - KHÔNG xoay/frontalize — giữ nguyên góc mặt gốc
    """
    
    def __init__(self):
        self.logger = setupLogger("Reconstructor3D")
        self.device = getDevice()
        self.tddfa = None
        self._loadModel()
    
    def _loadModel(self):
        """Load 3DDFA V2 model."""
        try:
            if TDDFA_DIR not in sys.path:
                sys.path.insert(0, TDDFA_DIR)
            
            import yaml
            from TDDFA import TDDFA
            
            cfgPath = os.path.join(TDDFA_DIR, "configs", "mb1_120x120.yml")
            with open(cfgPath, "r") as f:
                cfg = yaml.safe_load(f)
            
            # Absolute paths
            cfg["checkpoint_fp"] = os.path.join(TDDFA_DIR, cfg["checkpoint_fp"])
            cfg["bfm_fp"] = os.path.join(TDDFA_DIR, cfg["bfm_fp"])
            
            gpuMode = str(self.device) == "cuda"
            cfg["gpu_mode"] = gpuMode
            
            self.tddfa = TDDFA(**cfg)
            self.logger.info("3DDFA V2 loaded cho 3D reconstruction")
        except Exception as e:
            self.logger.error(f"Lỗi load 3DDFA V2: {e}")
            self.tddfa = None
    
    def reconstruct(self, imageCv2, bbox):
        """
        Reconstruct 3D face mesh từ ảnh 2D.
        
        KHÔNG xoay frontal — giữ nguyên góc mặt gốc.
        
        Args:
            imageCv2: numpy array (BGR)
            bbox: [x1, y1, x2, y2]
        
        Returns:
            dict:
                - param_lst: list — 3DMM parameters (62-d)
                - vertices: numpy array (3, N) — 3D vertices
                - roi_box: list — ROI box used
            hoặc None nếu reconstruction thất bại
        """
        if self.tddfa is None:
            self.logger.error("3DDFA V2 chưa được load")
            return None
        
        try:
            bboxForTddfa = [bbox]
            
            # Forward: tính 3DMM parameters
            paramLst, roiBoxLst = self.tddfa(imageCv2, bboxForTddfa)
            
            if not paramLst:
                self.logger.warning("3DDFA không trả về params")
                return None
            
            # Reconstruct 3D vertices (dense mesh)
            verticesLst = self.tddfa.recon_vers(
                paramLst, roiBoxLst,
                dense_flag=True
            )
            
            self.logger.info(
                f"3D reconstruction thành công: "
                f"params={len(paramLst[0])}d, "
                f"vertices shape={verticesLst[0].shape}"
            )
            
            return {
                "param_lst": paramLst,
                "vertices": verticesLst[0],  # (3, N)
                "roi_box": roiBoxLst[0]
            }
        except Exception as e:
            self.logger.error(f"Lỗi 3D reconstruction: {e}")
            return None
    
    def reconstructBatch(self, imageCv2, bboxes):
        """
        Batch reconstruct cho nhiều faces trong 1 ảnh.
        
        Args:
            imageCv2: numpy array (BGR)
            bboxes: list of [x1, y1, x2, y2]
        
        Returns:
            list of dict — mỗi face 1 result
        """
        results = []
        for bbox in bboxes:
            result = self.reconstruct(imageCv2, bbox)
            results.append(result)
        return results
    
    def isAvailable(self):
        """Kiểm tra model đã sẵn sàng chưa."""
        return self.tddfa is not None
