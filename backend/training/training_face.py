"""
Training Face Pipeline — Pipeline xử lý khuôn mặt production-ready.

Pipeline chính orchestrate toàn bộ quy trình:
1. Face Detection (YOLOv8-Face)
2. Landmark & Pose Estimation (InsightFace + 3DDFA V2)
3. Embedding Extraction (InsightFace ArcFace / AdaFace theo góc yaw)
4. 3D Reconstruction (3DDFA V2, chỉ khi |yaw| >= 45°)
5. Visualization & Segmentation (SegFormer)
6. Export: .npy + .png + .json cho mỗi face

Sẵn sàng tích hợp vào backend API.

LƯU Ý:
- KHÔNG rotate/frontalize khuôn mặt
- Embedding phản ánh đúng góc mặt gốc
- Hỗ trợ yaw từ 0° đến 90°
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from backend.app.services.training_utils import (
    setupLogger, getDevice, generateFilename,
    loadImageSafe, normalizeEmbedding, ensureDir,
    convertToWslPath
)
from backend.app.services.face_detector import TrainingFaceDetector
from backend.app.services.pose_estimator import TrainingPoseEstimator
from backend.app.services.embedder import TrainingEmbedder
from backend.app.services.reconstructor_3d import TrainingReconstructor3D
from backend.app.services.visualizer import TrainingVisualizer

# Ngưỡng yaw cho logic chuyển model
YAW_THRESHOLD = 45.0


class TrainingFacePipeline:
    """
    Pipeline xử lý khuôn mặt hoàn chỉnh cho training.
    
    Workflow cho mỗi ảnh:
    1. Detect all faces → bounding boxes
    2. Estimate pose → yaw/pitch/roll
    3. Extract embedding → InsightFace (|yaw|<45°) hoặc AdaFace (|yaw|≥45°)
    4. 3D reconstruction (nếu |yaw| ≥ 45°)
    5. Visualization + segmentation
    6. Export: embedding (.npy) + visualization (.png) + metadata (.json)
    """
    
    def __init__(self, yawThreshold=YAW_THRESHOLD):
        """
        Khởi tạo pipeline với tất cả các module.
        
        Args:
            yawThreshold: Ngưỡng yaw để chuyển model (mặc định 45°)
        """
        self.logger = setupLogger("TrainingFacePipeline")
        self.yawThreshold = yawThreshold
        
        self.logger.info("=" * 60)
        self.logger.info("Khởi tạo Training Face Pipeline...")
        self.logger.info("=" * 60)
        
        # Khởi tạo từng module
        self.logger.info("[1/5] Loading Face Detector...")
        self.detector = TrainingFaceDetector()
        
        self.logger.info("[2/5] Loading Pose Estimator...")
        self.poseEstimator = TrainingPoseEstimator()
        
        self.logger.info("[3/5] Loading Embedder...")
        self.embedder = TrainingEmbedder(yawThreshold=yawThreshold)
        
        self.logger.info("[4/5] Loading 3D Reconstructor...")
        self.reconstructor = TrainingReconstructor3D()
        
        self.logger.info("[5/5] Loading Visualizer...")
        self.visualizer = TrainingVisualizer()
        
        self.logger.info("=" * 60)
        self.logger.info("Pipeline sẵn sàng!")
        self._printModuleStatus()
        self.logger.info("=" * 60)
    
    def processImage(self, imagePath, outputDir):
        """
        Xử lý 1 ảnh: detect → pose → embed → visualize → export.
        
        Args:
            imagePath: str — đường dẫn file ảnh đầu vào
            outputDir: str — thư mục output
        
        Returns:
            list of dict — kết quả cho mỗi face:
                - face_id: int
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - yaw: float
                - pitch: float
                - roll: float
                - embedding_model: str
                - embedding_path: str
                - visualization_path: str
                - metadata_path: str
        """
        # Auto-convert Windows path → WSL path
        imagePath = convertToWslPath(imagePath)
        outputDir = convertToWslPath(outputDir)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing: {imagePath}")
        self.logger.info(f"Output dir: {outputDir}")
        self.logger.info(f"{'='*60}")
        
        # Load ảnh
        imageCv2 = loadImageSafe(imagePath)
        if imageCv2 is None:
            self.logger.error(f"Không thể load ảnh: {imagePath}")
            return []
        
        # Tạo output directory theo tên ảnh: output/{image_name}/
        imgName = os.path.splitext(os.path.basename(imagePath))[0]
        outputDir = os.path.join(outputDir, imgName)
        ensureDir(outputDir)
        
        # ==========================================
        # STEP 1: FACE DETECTION
        # ==========================================
        self.logger.info("\n--- STEP 1: Face Detection ---")
        faces = self.detector.detect(imageCv2)
        
        if not faces:
            self.logger.warning("Không phát hiện khuôn mặt nào trong ảnh")
            return []
        
        self.logger.info(f"Phát hiện {len(faces)} khuôn mặt")
        
        # ==========================================
        # STEP 2-6: Xử lý từng face
        # ==========================================
        results = []
        
        # Thu thập tất cả bboxes để visualizer loại trừ face lân cận
        allBboxes = [fd["bbox"] for fd in faces]
        
        for idx, faceData in enumerate(faces):
            faceId = idx + 1
            bbox = faceData["bbox"]
            confidence = faceData["confidence"]
            
            self.logger.info(f"\n--- Face #{faceId}/{len(faces)} ---")
            self.logger.info(f"  BBox: {[round(c, 1) for c in bbox]}")
            self.logger.info(f"  Confidence: {confidence:.3f}")
            
            faceResult = self._processSingleFace(
                imageCv2, bbox, confidence, faceId, outputDir, allBboxes
            )
            
            if faceResult is not None:
                results.append(faceResult)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Hoàn tất: {len(results)}/{len(faces)} faces processed")
        self.logger.info(f"{'='*60}")
        
        return results
    
    def _processSingleFace(self, imageCv2, bbox, confidence, faceId, outputDir, allBboxes=None):
        """
        Xử lý 1 khuôn mặt hoàn chỉnh.
        
        Returns:
            dict chứa kết quả, hoặc None nếu lỗi
        """
        try:
            # ==========================================
            # STEP 2: POSE ESTIMATION
            # ==========================================
            self.logger.info("  [Step 2] Pose Estimation...")
            poseResult = self.poseEstimator.estimate(imageCv2, bbox)
            
            yaw = poseResult["yaw"]
            pitch = poseResult["pitch"]
            roll = poseResult["roll"]
            absYaw = abs(yaw)
            
            self.logger.info(f"  Yaw={yaw:.1f}° Pitch={pitch:.1f}° Roll={roll:.1f}°")
            
            if absYaw < self.yawThreshold:
                self.logger.info(f"  → Pose nhỏ (|yaw|={absYaw:.1f}° < {self.yawThreshold}°)")
            else:
                self.logger.info(f"  → Pose lớn (|yaw|={absYaw:.1f}° >= {self.yawThreshold}°)")
            
            # ==========================================
            # STEP 3: EMBEDDING EXTRACTION
            # ==========================================
            self.logger.info("  [Step 3] Embedding Extraction...")
            embeddingResult = self.embedder.getEmbedding(imageCv2, bbox, yaw)
            
            if embeddingResult is None:
                self.logger.error(f"  Không thể extract embedding cho Face #{faceId}")
                return None
            
            embedding = embeddingResult["embedding"]
            modelName = embeddingResult["model_name"]
            self.logger.info(f"  Model: {modelName}, Embedding shape: {embedding.shape}")
            
            # ==========================================
            # STEP 4: 3D RECONSTRUCTION (chỉ khi pose lớn)
            # ==========================================
            reconResult = None
            if absYaw >= self.yawThreshold:
                self.logger.info("  [Step 4] 3D Reconstruction...")
                reconResult = self.reconstructor.reconstruct(imageCv2, bbox)
                if reconResult is not None:
                    self.logger.info(f"  3D vertices: {reconResult['vertices'].shape}")
                else:
                    self.logger.warning("  3D reconstruction thất bại (tiếp tục pipeline)")
            
            # ==========================================
            # STEP 5: GENERATE FILENAMES & SAVE
            # ==========================================
            # Tạo subfolder cho mỗi face: output/face_01/
            faceDir = os.path.join(outputDir, f"face_{faceId:02d}")
            ensureDir(faceDir)
            
            baseName = f"face_{faceId:02d}_yaw_{int(abs(yaw)):02d}"
            basePath = os.path.join(faceDir, baseName)
            
            # Paths
            embeddingPath = basePath + ".npy"
            metadataPath = basePath + ".json"
            
            # Lưu embedding (.npy)
            np.save(embeddingPath, embedding)
            self.logger.info(f"  Saved embedding: {embeddingPath}")
            
            # ==========================================
            # STEP 6: VISUALIZATION (4 ảnh)
            # ==========================================
            self.logger.info("  [Step 5] Visualization & Segmentation (4 images)...")
            
            # Chuẩn bị pose info cho visualizer
            poseInfoForVis = {
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll
            }
            
            # Lấy landmarks 106 từ pose result
            landmarks106 = poseResult.get("landmarks_106", None)
            
            # Lấy 3D vertices nếu có (cho face mask enhancement)
            vertices3D = reconResult["vertices"] if reconResult else None
            
            visPaths = self.visualizer.createVisualization(
                imageCv2, bbox, faceId, basePath,
                landmarks106=landmarks106,
                poseInfo=poseInfoForVis,
                vertices3D=vertices3D,
                allBboxes=allBboxes
            )
            
            # ==========================================
            # STEP 7: METADATA
            # ==========================================
            self.logger.info("  [Step 6] Saving Metadata...")
            metadata = {
                "face_id": faceId,
                "yaw": round(float(yaw), 2),
                "pitch": round(float(pitch), 2),
                "roll": round(float(roll), 2),
                "bbox": [round(float(c), 1) for c in bbox],
                "confidence": round(float(confidence), 4),
                "embedding_model": modelName,
                "embedding_path": os.path.basename(embeddingPath),
                "visualization_path": os.path.basename(visPaths) if visPaths else "",
                "embedding_dim": int(embedding.shape[0]),
                "pose_method": poseResult.get("method", "unknown"),
                "yaw_threshold": self.yawThreshold,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(metadataPath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            self.logger.info(f"  Saved metadata: {metadataPath}")
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"  Lỗi xử lý Face #{faceId}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def processBatch(self, imageDir, outputDir):
        """
        Batch processing cho tất cả ảnh trong thư mục.
        
        Args:
            imageDir: str — thư mục chứa ảnh đầu vào
            outputDir: str — thư mục output
        
        Returns:
            dict:
                - total_images: int
                - total_faces: int
                - results: list of list — kết quả cho mỗi ảnh
        """
        # Auto-convert Windows path → WSL path
        imageDir = convertToWslPath(imageDir)
        outputDir = convertToWslPath(outputDir)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"BATCH PROCESSING")
        self.logger.info(f"Input dir: {imageDir}")
        self.logger.info(f"Output dir: {outputDir}")
        self.logger.info(f"{'='*60}")
        
        # Tìm tất cả ảnh
        supportedExts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        imagePaths = []
        
        for f in sorted(os.listdir(imageDir)):
            ext = os.path.splitext(f)[1].lower()
            if ext in supportedExts:
                imagePaths.append(os.path.join(imageDir, f))
        
        if not imagePaths:
            self.logger.warning(f"Không tìm thấy ảnh trong {imageDir}")
            return {"total_images": 0, "total_faces": 0, "results": []}
        
        self.logger.info(f"Tìm thấy {len(imagePaths)} ảnh")
        
        allResults = []
        totalFaces = 0
        
        for i, imgPath in enumerate(imagePaths):
            self.logger.info(f"\n[{i+1}/{len(imagePaths)}] {os.path.basename(imgPath)}")
            
            # Tạo sub-folder cho mỗi ảnh
            imgName = os.path.splitext(os.path.basename(imgPath))[0]
            imgOutputDir = os.path.join(outputDir, imgName)
            
            results = self.processImage(imgPath, imgOutputDir)
            allResults.append(results)
            totalFaces += len(results)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"BATCH HOÀN TẤT")
        self.logger.info(f"Tổng: {len(imagePaths)} ảnh, {totalFaces} faces")
        self.logger.info(f"{'='*60}")
        
        return {
            "total_images": len(imagePaths),
            "total_faces": totalFaces,
            "results": allResults
        }
    
    def getModuleStatus(self):
        """
        Trả về trạng thái các module.
        
        Returns:
            dict — tên module → available (bool)
        """
        return {
            "detector": self.detector.isAvailable(),
            "pose_estimator": self.poseEstimator.isAvailable(),
            "embedder": self.embedder.isAvailable(),
            "reconstructor_3d": self.reconstructor.isAvailable(),
            "visualizer": self.visualizer.isAvailable()
        }
    
    def _printModuleStatus(self):
        """In trạng thái từng module."""
        status = self.getModuleStatus()
        for module, available in status.items():
            icon = "✅" if (available if isinstance(available, bool) else all(available.values())) else "❌"
            self.logger.info(f"  {icon} {module}: {available}")


# ==========================================
# CLI Entry Point
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Face Pipeline")
    parser.add_argument("--image", type=str, help="Đường dẫn ảnh đầu vào")
    parser.add_argument("--image-dir", type=str, help="Thư mục ảnh đầu vào (batch mode)")
    parser.add_argument("--output", type=str, default="backend/training/output",
                        help="Thư mục output")
    parser.add_argument("--yaw-threshold", type=float, default=45.0,
                        help="Ngưỡng yaw để chuyển model (mặc định 45°)")
    
    args = parser.parse_args()
    
    # Khởi tạo pipeline
    pipeline = TrainingFacePipeline(yawThreshold=args.yaw_threshold)
    
    if args.image:
        # Single image mode
        results = pipeline.processImage(args.image, args.output)
        print(f"\nKết quả: {len(results)} faces processed")
        for r in results:
            print(f"  Face #{r['face_id']}: yaw={r['yaw']}° model={r['embedding_model']}")
    
    elif args.image_dir:
        # Batch mode
        results = pipeline.processBatch(args.image_dir, args.output)
        print(f"\nKết quả batch: {results['total_images']} ảnh, {results['total_faces']} faces")
    
    else:
        print("Vui lòng cung cấp --image hoặc --image-dir")
        parser.print_help()
