"""
Face Detection Service - Hybrid YOLO + InsightFace + AdaFace

Pipeline:
1. YOLOv8-Face detect tất cả faces (kể cả partial/profile)
2. InsightFace analyze để lấy embedding, keypoints
3. Nếu InsightFace không detect được (partial) → dùng AdaFace fallback
"""

import cv2
import numpy as np
import os
from backend.app.config import model_paths, settings


class YOLOFaceDetector:
    """
    YOLOv8-Face detector cho việc phát hiện khuôn mặt.
    Có thể detect partial/profile faces tốt hơn InsightFace.
    """
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load YOLOv8-Face model"""
        try:
            from ultralytics import YOLO
            
            model_path = model_paths.YOLO_FACE_MODEL
            
            if not os.path.exists(model_path):
                print(f"[YOLOFaceDetector] Model not found at {model_path}")
                print("[YOLOFaceDetector] Run: python download_models.py")
                self.model = None
                return
            
            self.model = YOLO(model_path)
            print(f"[YOLOFaceDetector] Loaded model from {model_path}")
            
        except ImportError:
            print("[YOLOFaceDetector] ultralytics not installed. Run: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"[YOLOFaceDetector] Error loading model: {e}")
            self.model = None
    
    def detect(self, image_cv2, conf_threshold=0.4):
        if self.model is None:
            return []
        
        try:
            image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

            results = self.model(image_rgb, verbose=False, conf=conf_threshold)
            
            faces = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for box in boxes:
                    bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0].cpu().numpy())
                    
                    faces.append({
                        'bbox': bbox.tolist(),
                        'confidence': conf
                    })
            
            faces = sorted(
                faces,
                key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]),
                reverse=True
            )
            
            return faces
            
        except Exception as e:
            print(f"[YOLOFaceDetector] Detection error: {e}")
            return []
    
    def is_available(self):
        return self.model is not None


class PartialFaceInfo:
    def __init__(self, bbox, embedding, kps=None, det_score=0.0):
        self.bbox = np.array(bbox)
        self.embedding = embedding
        self.kps = kps
        self.det_score = det_score
        self.is_partial = True


class FaceInfoService:
    def __init__(self):
        import insightface
        from insightface.app import FaceAnalysis
        
        self.yolo_detector = YOLOFaceDetector()
        
        self.app = FaceAnalysis(
            name='antelopev2', 
            root=model_paths.INSIGHTFACE_ROOT, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # AdaFace cho partial face embedding (lazy load)
        self.adaface = None
        self._adaface_init_attempted = False
        
        print("[FaceInfoService] Initialized with hybrid YOLO + InsightFace + AdaFace")
    
    def _init_adaface(self):
        if self._adaface_init_attempted:
            return self.adaface is not None
        
        self._adaface_init_attempted = True
        try:
            from backend.app.services.adaface_service import AdaFaceService
            self.adaface = AdaFaceService()
            if not self.adaface.is_available():
                print("[FaceInfoService] AdaFace model not available")
                self.adaface = None
                return False
            print("[FaceInfoService] AdaFace initialized for profile face support")
            return True
        except Exception as e:
            print(f"[FaceInfoService] AdaFace init failed: {e}")
            self.adaface = None
            return False
    
    def _match_yolo_to_insight(self, yolo_faces, insight_faces, iou_threshold=0.5):
        if len(insight_faces) == 0:
            return yolo_faces
        
        unmatched = []
        
        for yolo_face in yolo_faces:
            yolo_bbox = yolo_face['bbox']
            matched = False
            
            for insight_face in insight_faces:
                insight_bbox = insight_face.bbox.tolist()
                
                # Tính IoU
                iou = self._compute_iou(yolo_bbox, insight_bbox)
                if iou > iou_threshold:
                    matched = True
                    break
            
            if not matched:
                unmatched.append(yolo_face)
        
        return unmatched
    
    def _compute_iou(self, box1, box2):
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

    def analyze_all(self, image_cv2, use_adaface_fallback=True):
        """
        Phân tích TẤT CẢ khuôn mặt từ ảnh OpenCV (BGR).
        
        Pipeline:
        1. YOLO detect tất cả faces
        2. InsightFace analyze từng face
        3. Với faces mà InsightFace miss → dùng AdaFace để extract embedding
        """
        # Step 1: YOLO detection (có thể detect partial faces)
        yolo_faces = []
        if self.yolo_detector.is_available():
            yolo_faces = self.yolo_detector.detect(image_cv2, conf_threshold=0.4)
            print(f"[FaceInfoService] YOLO detected {len(yolo_faces)} face(s)")
        
        # Step 2: InsightFace analysis
        insight_faces = self.app.get(image_cv2)
        print(f"[FaceInfoService] InsightFace detected {len(insight_faces)} face(s)")
        
        # Danh sách kết quả cuối cùng
        all_faces = list(insight_faces)
        
        # Step 3: Tìm partial faces (YOLO detect nhưng InsightFace không)
        if use_adaface_fallback and len(yolo_faces) > len(insight_faces):
            unmatched_yolo = self._match_yolo_to_insight(yolo_faces, insight_faces)
            
            if len(unmatched_yolo) > 0:
                print(f"[FaceInfoService] Found {len(unmatched_yolo)} partial face(s), trying AdaFace...")
                
                # Init AdaFace nếu chưa
                if self._init_adaface():
                    for yolo_face in unmatched_yolo:
                        bbox = yolo_face['bbox']
                        
                        # Extract embedding bằng AdaFace
                        embedding = self.adaface.get_embedding(image_cv2, bbox=bbox)
                        
                        if embedding is not None:
                            # Tạo PartialFaceInfo
                            partial_face = PartialFaceInfo(
                                bbox=bbox,
                                embedding=embedding,
                                kps=None,  # Không có keypoints cho partial faces
                                det_score=yolo_face['confidence']
                            )
                            all_faces.append(partial_face)
                            print(f"[FaceInfoService] ✓ AdaFace extracted embedding for partial face")
                        else:
                            print(f"[FaceInfoService] ✗ AdaFace failed for partial face")
                else:
                    print(f"[FaceInfoService] AdaFace not available, skipping {len(unmatched_yolo)} partial face(s)")
        
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
        return face_info.kps
    
    def is_partial_face(self, face_info):
        """Check if face_info is partial face (from AdaFace)"""
        return getattr(face_info, 'is_partial', False)
    
    def get_detection_status(self, image_cv2):
        """
        Kiểm tra trạng thái detection.
        Dùng cho frontend để hiển thị thông báo phù hợp.
        
        Returns:
            dict với keys:
                - yolo_count: Số faces YOLO detect được
                - insight_count: Số faces InsightFace detect được  
                - adaface_count: Số partial faces AdaFace xử lý được
                - status: 'success' | 'partial_faces_processed' | 'partial_faces_skipped' | 'no_face'
                - message: Thông báo cho user
        """
        yolo_faces = []
        if self.yolo_detector.is_available():
            yolo_faces = self.yolo_detector.detect(image_cv2, conf_threshold=0.4)
        
        insight_faces = self.app.get(image_cv2)
        
        yolo_count = len(yolo_faces)
        insight_count = len(insight_faces)
        adaface_count = 0
        
        # Check partial faces với AdaFace
        if yolo_count > insight_count and self._init_adaface():
            unmatched = self._match_yolo_to_insight(yolo_faces, insight_faces)
            for yolo_face in unmatched:
                embedding = self.adaface.get_embedding(image_cv2, bbox=yolo_face['bbox'])
                if embedding is not None:
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
