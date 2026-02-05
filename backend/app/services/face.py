
import cv2
import numpy as np
from backend.app.config import model_paths, settings

class FaceInfoService:
    def __init__(self):
        # Lazy import to prevent hang on module load in WSL
        import insightface
        from insightface.app import FaceAnalysis
        
        # Khởi tạo InsightFaceapp
        # allowed_modules=['detection', 'recognition'] để lấy embedding và kps
        # root trỏ về folder antelopev2 đã tải
        self.app = FaceAnalysis(
            name='antelopev2', 
            root=model_paths.INSIGHTFACE_ROOT, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def analyze(self, image_cv2):
        """
        Phân tích khuôn mặt từ ảnh OpenCV (BGR).
        Trả về face_info của khuôn mặt lớn nhất tìm thấy.
        """
        faces = self.app.get(image_cv2)
        if len(faces) == 0:
            return None
        
        # Sắp xếp lấy khuôn mặt to nhất (theo diện tích bbox)
        faces = sorted(
            faces, 
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), 
            reverse=True
        )
        return faces[0]

    def get_face_embedding(self, face_info):
        return face_info.embedding

    def get_keypoints(self, face_info):
        return face_info.kps
