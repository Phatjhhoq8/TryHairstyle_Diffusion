"""
Face Detection Training Utilities
Sử dụng InsightFace (antelopev2) để phát hiện và xử lý khuôn mặt.

Tách từ: backend/app/services/face.py
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


class FaceDetector:
    """
    Face Detection Service sử dụng InsightFace.
    Dùng cho việc tiền xử lý dữ liệu training.
    """
    
    def __init__(self, model_name='antelopev2', model_root=None, device='cuda'):
        """
        Khởi tạo Face Detector.
        
        Args:
            model_name: Tên model InsightFace (default: antelopev2)
            model_root: Đường dẫn đến thư mục chứa model
            device: 'cuda' hoặc 'cpu'
        """
        import insightface
        from insightface.app import FaceAnalysis
        
        # Xác định providers theo device
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Xác định model_root
        if model_root is None:
            # Default: thư mục backend
            model_root = str(Path(__file__).parent.parent)
        
        self.app = FaceAnalysis(
            name=model_name,
            root=model_root,
            providers=providers
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print(f"[FaceDetector] Đã khởi tạo với model: {model_name}")
    
    def detect(self, image):
        """
        Phát hiện tất cả khuôn mặt trong ảnh (sắp xếp theo kích thước).
        
        Args:
            image: Ảnh OpenCV (BGR) hoặc đường dẫn file
            
        Returns:
            List các face_info objects (sắp xếp từ lớn đến nhỏ)
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {image}")
        
        faces = self.app.get(image)
        
        if len(faces) == 0:
            return []
        
        # Sắp xếp theo diện tích bbox (lớn → nhỏ)
        faces = sorted(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            reverse=True
        )
        return faces
    
    def detect_all(self, image):
        """Alias cho detect() - để nhất quán với FaceInfoService.analyze_all()"""
        return self.detect(image)
    
    def detect_largest(self, image):
        """
        Phát hiện khuôn mặt lớn nhất trong ảnh.
        (Backward compatible - dùng detect() internally)
        
        Args:
            image: Ảnh OpenCV (BGR) hoặc đường dẫn file
            
        Returns:
            face_info của khuôn mặt lớn nhất, hoặc None nếu không tìm thấy
        """
        faces = self.detect(image)
        if len(faces) == 0:
            return None
        return faces[0]
    
    def get_face_info(self, face):
        """
        Trích xuất thông tin từ face object.
        
        Args:
            face: face_info object từ InsightFace
            
        Returns:
            Dict chứa bbox, embedding, keypoints, landmarks
        """
        return {
            'bbox': face.bbox.tolist(),  # [x1, y1, x2, y2]
            'det_score': float(face.det_score),  # Confidence score
            'embedding': face.embedding,  # Face embedding (512-d)
            'kps': face.kps.tolist() if face.kps is not None else None,  # 5 keypoints
            'landmark_2d_106': face.landmark_2d_106.tolist() if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None else None,
        }
    
    def crop_face(self, image, face, margin=0.3):
        """
        Cắt khuôn mặt từ ảnh với margin.
        
        Args:
            image: Ảnh OpenCV (BGR)
            face: face_info object
            margin: Tỷ lệ margin quanh bbox (0.3 = 30%)
            
        Returns:
            Ảnh khuôn mặt đã crop
        """
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Tính margin
        w, h = x2 - x1, y2 - y1
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        # Mở rộng bbox với bounds checking
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(image.shape[1], x2 + margin_x)
        y2 = min(image.shape[0], y2 + margin_y)
        
        return image[y1:y2, x1:x2]
    
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Vẽ bounding box và keypoints lên ảnh.
        
        Args:
            image: Ảnh OpenCV (BGR)
            faces: List face_info objects
            color: Màu vẽ (BGR)
            thickness: Độ dày đường vẽ
            
        Returns:
            Ảnh đã vẽ
        """
        result = image.copy()
        
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Vẽ bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Vẽ keypoints
            if face.kps is not None:
                for kp in face.kps:
                    cv2.circle(result, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)
            
            # Vẽ confidence
            score = face.det_score
            cv2.putText(result, f"{score:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result


def filter_dataset_by_face(input_dir, output_dir, min_face_size=100, detector=None):
    """
    Lọc dataset, chỉ giữ ảnh có khuôn mặt đủ lớn.
    
    Args:
        input_dir: Thư mục chứa ảnh gốc
        output_dir: Thư mục lưu ảnh đã lọc
        min_face_size: Kích thước tối thiểu của khuôn mặt (pixels)
        detector: FaceDetector instance (tạo mới nếu None)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if detector is None:
        detector = FaceDetector()
    
    # Lấy danh sách ảnh
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    images = []
    for ext in extensions:
        images.extend(input_path.rglob(ext))
    
    print(f"[Filter] Tìm thấy {len(images)} ảnh")
    
    valid_count = 0
    for img_path in tqdm(images, desc="Filtering"):
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            face = detector.detect_largest(image)
            if face is None:
                continue
            
            # Kiểm tra kích thước
            bbox = face.bbox
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            if w >= min_face_size and h >= min_face_size:
                # Copy ảnh sang output
                output_file = output_path / img_path.name
                cv2.imwrite(str(output_file), image)
                valid_count += 1
                
        except Exception as e:
            print(f"Lỗi xử lý {img_path}: {e}")
    
    print(f"[Filter] Hoàn thành: {valid_count}/{len(images)} ảnh hợp lệ")


def extract_face_embeddings(image_dir, output_file, detector=None):
    """
    Trích xuất face embeddings từ tất cả ảnh trong thư mục.
    
    Args:
        image_dir: Thư mục chứa ảnh
        output_file: File .npy để lưu embeddings
        detector: FaceDetector instance
        
    Returns:
        Dict {filename: embedding}
    """
    image_path = Path(image_dir)
    
    if detector is None:
        detector = FaceDetector()
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    images = []
    for ext in extensions:
        images.extend(image_path.rglob(ext))
    
    print(f"[Embeddings] Xử lý {len(images)} ảnh...")
    
    embeddings = {}
    for img_path in tqdm(images, desc="Extracting"):
        try:
            face = detector.detect_largest(str(img_path))
            if face is not None:
                embeddings[img_path.name] = face.embedding
        except Exception as e:
            print(f"Lỗi: {img_path}: {e}")
    
    # Lưu file
    np.save(output_file, embeddings)
    print(f"[Embeddings] Đã lưu {len(embeddings)} embeddings vào {output_file}")
    
    return embeddings


# ===== CLI Test =====
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Detection Training Utilities")
    parser.add_argument("--image", type=str, help="Đường dẫn ảnh để test")
    parser.add_argument("--filter-dir", type=str, help="Thư mục ảnh cần lọc")
    parser.add_argument("--output-dir", type=str, help="Thư mục lưu ảnh đã lọc")
    parser.add_argument("--min-size", type=int, default=100, help="Kích thước mặt tối thiểu")
    
    args = parser.parse_args()
    
    detector = FaceDetector()
    
    if args.image:
        # Test single image
        print(f"[Test] Đang xử lý: {args.image}")
        image = cv2.imread(args.image)
        faces = detector.detect(image)
        print(f"[Test] Phát hiện {len(faces)} khuôn mặt")
        
        for i, face in enumerate(faces):
            info = detector.get_face_info(face)
            print(f"  Face {i+1}: bbox={info['bbox']}, score={info['det_score']:.3f}")
        
        # Vẽ và lưu kết quả
        result = detector.draw_faces(image, faces)
        output_path = "face_detection_result.png"
        cv2.imwrite(output_path, result)
        print(f"[Test] Đã lưu kết quả: {output_path}")
        
    elif args.filter_dir and args.output_dir:
        # Filter dataset
        filter_dataset_by_face(args.filter_dir, args.output_dir, args.min_size, detector)
    
    else:
        print("Sử dụng:")
        print("  python train_face_detection.py --image <path>")
        print("  python train_face_detection.py --filter-dir <input> --output-dir <output>")
