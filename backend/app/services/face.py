"""
Face Detection Service - Sử dụng training modules đã refactor.

Pipeline:
1. TrainingFaceDetector (YOLOv8-Face + NMS) detect tất cả faces
2. InsightFace analyze để lấy embedding, keypoints, 106 landmarks
3. TrainingEmbedder cho embedding extraction theo yaw (InsightFace < 45° / AdaFace ≥ 45°)
"""

import cv2
import numpy as np
import os
from backend.app.config import model_paths, settings
from backend.app.services.face_detector import TrainingFaceDetector
from backend.app.services.embedder import TrainingEmbedder


class PartialFaceInfo:
    """Wrapper cho face không được InsightFace xử lý đầy đủ."""
    def __init__(self, bbox, embedding, kps=None, det_score=0.0):
        self.bbox = np.array(bbox)
        self.embedding = embedding
        self.kps = kps
        self.det_score = det_score
        self.is_partial = True


class FaceInfoService:
    """
    Service phân tích khuôn mặt.
    
    Sử dụng:
    - TrainingFaceDetector: YOLOv8-Face + NMS cho detection
    - InsightFace (antelopev2): landmarks, embedding cho frontal face
    - TrainingEmbedder: yaw-based embedding switching (InsightFace/AdaFace)
    """
    
    def __init__(self):
        import insightface
        from insightface.app import FaceAnalysis
        
        # YOLOv8-Face detector (từ training module, có NMS)
        self.yolo_detector = TrainingFaceDetector()
        
        # InsightFace cho landmark extraction + embedding
        self.app = FaceAnalysis(
            name='antelopev2', 
            root=model_paths.INSIGHTFACE_ROOT, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Training Embedder cho yaw-based embedding (lazy load)
        self.embedder = None
        self._embedder_init_attempted = False
        
        print("[FaceInfoService] Initialized with TrainingFaceDetector + InsightFace + TrainingEmbedder")
    
    def _init_embedder(self):
        """Lazy init TrainingEmbedder."""
        if self._embedder_init_attempted:
            return self.embedder is not None
        
        self._embedder_init_attempted = True
        try:
            self.embedder = TrainingEmbedder()
            avail = self.embedder.isAvailable()
            if not avail.get("insightface") and not avail.get("adaface"):
                print("[FaceInfoService] TrainingEmbedder: không có model nào available")
                self.embedder = None
                return False
            print(f"[FaceInfoService] TrainingEmbedder initialized: {avail}")
            return True
        except Exception as e:
            print(f"[FaceInfoService] TrainingEmbedder init failed: {e}")
            self.embedder = None
            return False
    
    def _compute_iou(self, box1, box2):
        """Tính IoU giữa 2 bbox."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def _match_yolo_to_insight(self, yolo_faces, insight_faces, iou_threshold=0.5):
        """Tìm YOLO faces không match với InsightFace (partial/profile)."""
        if len(insight_faces) == 0:
            return yolo_faces
        
        unmatched = []
        
        for yolo_face in yolo_faces:
            yolo_bbox = yolo_face['bbox']
            matched = False
            
            for insight_face in insight_faces:
                insight_bbox = insight_face.bbox.tolist()
                
                iou = self._compute_iou(yolo_bbox, insight_bbox)
                if iou > iou_threshold:
                    matched = True
                    break
            
            if not matched:
                unmatched.append(yolo_face)
        
        return unmatched

    def analyze_all(self, image_cv2, use_adaface_fallback=True):
        """
        Phân tích TẤT CẢ khuôn mặt từ ảnh OpenCV (BGR).
        
        Pipeline:
        1. YOLO detect tất cả faces (có NMS loại trùng)
        2. InsightFace analyze từng face
        3. Với faces mà InsightFace miss → dùng TrainingEmbedder
        """
        # Step 1: YOLO detection (có NMS tích hợp)
        yolo_faces = []
        if self.yolo_detector.isAvailable():
            yolo_faces = self.yolo_detector.detect(image_cv2)
            print(f"[FaceInfoService] YOLO detected {len(yolo_faces)} face(s) (with NMS)")
        
        # Step 2: InsightFace analysis
        insight_faces = self.app.get(image_cv2)
        print(f"[FaceInfoService] InsightFace detected {len(insight_faces)} face(s)")
        
        # DEBUG: Check 106 landmarks availability
        if len(insight_faces) > 0:
            has_106 = hasattr(insight_faces[0], 'landmark_2d_106') and insight_faces[0].landmark_2d_106 is not None
            print(f"[FaceInfoService] 106 Landmarks available: {has_106}")

        # Danh sách kết quả cuối cùng
        all_faces = list(insight_faces)
        
        # Step 3: Tìm partial faces (YOLO detect nhưng InsightFace không)
        if use_adaface_fallback and len(yolo_faces) > len(insight_faces):
            unmatched_yolo = self._match_yolo_to_insight(yolo_faces, insight_faces)
            
            if len(unmatched_yolo) > 0:
                print(f"[FaceInfoService] Found {len(unmatched_yolo)} partial face(s), trying TrainingEmbedder...")
                
                # Init TrainingEmbedder nếu chưa
                if self._init_embedder():
                    for yolo_face in unmatched_yolo:
                        bbox = yolo_face['bbox']
                        
                        # Dùng TrainingEmbedder (yaw=90 vì partial face thường profile)
                        result = self.embedder.getEmbedding(image_cv2, bbox, yaw=90.0)
                        
                        if result is not None:
                            partial_face = PartialFaceInfo(
                                bbox=bbox,
                                embedding=result["embedding"],
                                kps=None,
                                det_score=yolo_face['confidence']
                            )
                            all_faces.append(partial_face)
                            print(f"[FaceInfoService] ✓ {result['model_name']} extracted embedding for partial face")
                        else:
                            print(f"[FaceInfoService] ✗ TrainingEmbedder failed for partial face")
                else:
                    print(f"[FaceInfoService] TrainingEmbedder not available, skipping {len(unmatched_yolo)} partial face(s)")
        
        if len(all_faces) == 0:
            if len(yolo_faces) > 0:
                print("[FaceInfoService] WARNING: Có khuôn mặt nhưng không thể xử lý")
            return []
        
        # Sắp xếp theo diện tích bbox (lớn → nhỏ)
        all_faces = sorted(
            all_faces, 
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), 
            reverse=True
        )
        
        return all_faces

    def analyze(self, image_cv2, use_adaface_fallback=True):
        """
        Phân tích khuôn mặt từ ảnh OpenCV (BGR).
        Trả về face_info của khuôn mặt lớn nhất tìm thấy.
        (Backward compatible - dùng analyze_all() internally)
        """
        faces = self.analyze_all(image_cv2, use_adaface_fallback)
        if len(faces) == 0:
            return None
        return faces[0]

    def get_face_embedding(self, face_info):
        return face_info.embedding

    def get_keypoints(self, face_info):
        """
        Trả về keypoints. Ưu tiên 106 points nếu có.
        """
        if hasattr(face_info, 'landmark_2d_106') and face_info.landmark_2d_106 is not None:
             return face_info.landmark_2d_106
        return face_info.kps
    
    def get_5_keypoints(self, face_info):
        """
        Luôn trả về 5 points chuẩn (cho alignment/metrics cũ).
        """
        return face_info.kps

    def is_partial_face(self, face_info):
        """Check if face_info is partial face (from TrainingEmbedder)"""
        return getattr(face_info, 'is_partial', False)
    
    def get_detection_status(self, image_cv2):
        """
        Kiểm tra trạng thái detection.
        Dùng cho frontend để hiển thị thông báo phù hợp.
        
        Returns:
            dict với keys:
                - yolo_count: Số faces YOLO detect được
                - insight_count: Số faces InsightFace detect được  
                - adaface_count: Số partial faces xử lý được
                - status: 'success' | 'partial_faces_processed' | 'partial_faces_skipped' | 'no_face'
                - message: Thông báo cho user
        """
        yolo_faces = []
        if self.yolo_detector.isAvailable():
            yolo_faces = self.yolo_detector.detect(image_cv2)
        
        insight_faces = self.app.get(image_cv2)
        
        yolo_count = len(yolo_faces)
        insight_count = len(insight_faces)
        adaface_count = 0
        
        # Check partial faces với TrainingEmbedder
        if yolo_count > insight_count and self._init_embedder():
            unmatched = self._match_yolo_to_insight(yolo_faces, insight_faces)
            for yolo_face in unmatched:
                result = self.embedder.getEmbedding(image_cv2, yolo_face['bbox'], yaw=90.0)
                if result is not None:
                    adaface_count += 1
        
        total_processed = insight_count + adaface_count
        
        if total_processed > 0:
            if adaface_count > 0:
                return {
                    'yolo_count': yolo_count,
                    'insight_count': insight_count,
                    'adaface_count': adaface_count,
                    'status': 'partial_faces_processed',
                    'message': f'Phát hiện {yolo_count} khuôn mặt, {insight_count} xử lý đầy đủ, {adaface_count} xử lý một phần (góc nghiêng)'
                }
            return {
                'yolo_count': yolo_count,
                'insight_count': insight_count,
                'adaface_count': 0,
                'status': 'success',
                'message': f'Phát hiện {insight_count} khuôn mặt'
            }
        else:
            if yolo_count > 0:
                return {
                    'yolo_count': yolo_count,
                    'insight_count': 0,
                    'adaface_count': 0,
                    'status': 'partial_faces_skipped',
                    'message': 'Khuôn mặt nghiêng quá lớn, không thể xử lý. Vui lòng sử dụng ảnh có khuôn mặt rõ hơn.'
                }
            return {
                'yolo_count': 0,
                'insight_count': 0,
                'adaface_count': 0,
                'status': 'no_face',
                'message': 'Không phát hiện khuôn mặt trong ảnh'
            }
