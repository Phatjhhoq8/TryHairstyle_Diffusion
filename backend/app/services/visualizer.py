"""
Visualizer — Visualization & Face Segmentation cho Training Pipeline.

Tạo 4 ảnh visualize cho mỗi face:
1. BBox: Ảnh gốc + bounding box xanh lá
2. Segmentation: face=trắng, hair=đen, bg=xám
3. Geometry: Face geometry overlay (landmarks + wireframe vàng)
4. Red Mask: Face+Hair overlay đỏ trên ảnh gốc

Xuất 4 file PNG.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from backend.app.services.training_utils import setupLogger, getDevice, ensureDir

# SegFormer config
SEGFORMER_MODEL_ID = "jonathandinu/face-parsing"
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SEGFORMER_LOCAL_PATH = str(BASE_DIR / "backend" / "models" / "segformer_face_parsing")

# SegFormer class mapping (jonathandinu/face-parsing)
# 0: background
# 1: skin, 2: nose, 3: eye_g (glasses), 4: l_eye, 5: r_eye
# 6: l_brow, 7: r_brow, 8: l_ear, 9: r_ear
# 10: mouth, 11: u_lip, 12: l_lip
# 13: hair, 14: hat, 15: ear_r (earring)
# 16: neck_l (necklace), 17: neck, 18: cloth

# Nhóm face (skin + features)
FACE_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 17}

# Nhóm hair
HAIR_CLASSES = {13}

# Background = tất cả còn lại (0, 14, 16, 18)


class TrainingVisualizer:
    """
    Visualization và face segmentation cho training output.
    
    Tạo 4 ảnh cho mỗi face:
    1. _bbox.png — Bounding box xanh lá
    2. _seg.png — Segmentation mask (trắng/đen/xám)
    3. _geometry.png — Face geometry overlay (landmarks + wireframe)
    4. _red.png — Face+Hair red mask overlay
    """
    
    def __init__(self):
        self.logger = setupLogger("Visualizer")
        self.device = getDevice()
        self.segformer = None
        self.imageProcessor = None
        self._loadSegFormer()
    
    def _loadSegFormer(self):
        """Load SegFormer model. Ưu tiên local path, fallback HuggingFace hub."""
        try:
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            
            # Ưu tiên load từ local (đã download bằng download_models.py)
            modelSource = SEGFORMER_LOCAL_PATH
            if os.path.isdir(SEGFORMER_LOCAL_PATH):
                self.logger.info(f"Loading SegFormer từ local: {SEGFORMER_LOCAL_PATH}")
            else:
                # Fallback: download từ HuggingFace hub
                self.logger.info(f"Local không tìm thấy, loading SegFormer từ HuggingFace: {SEGFORMER_MODEL_ID}")
                modelSource = SEGFORMER_MODEL_ID
            
            self.imageProcessor = SegformerImageProcessor.from_pretrained(modelSource)
            model = SegformerForSemanticSegmentation.from_pretrained(modelSource)
            model.to(self.device)
            model.eval()
            
            self.segformer = model
            self.logger.info("SegFormer loaded cho face parsing (19 classes)")
        except ImportError:
            self.logger.error("Thiếu transformers. Chạy: pip install transformers")
        except Exception as e:
            self.logger.error(f"Lỗi load SegFormer: {e}")
            self.segformer = None
    
    def createVisualization(self, imageCv2, bbox, faceId, basePath, landmarks106=None, poseInfo=None, vertices3D=None, allBboxes=None):
        """
        Tạo 1 ảnh visualize ghép 2x2 cho 1 khuôn mặt.
        
        Layout:
        ┌─────────────┬──────────────┐
        │  1. BBox     │  2. Segment  │
        ├─────────────┼──────────────┤
        │  3. Geometry │  4. Red Mask │
        └─────────────┴──────────────┘
        
        Args:
            imageCv2: numpy array (BGR) — ảnh gốc
            bbox: [x1, y1, x2, y2]
            faceId: int — ID khuôn mặt
            basePath: str — đường dẫn base (không có extension)
            landmarks106: numpy array (106, 2) hoặc None
            poseInfo: dict {yaw, pitch, roll} hoặc None
            allBboxes: list of [x1, y1, x2, y2] — tất cả bboxes trong ảnh
        
        Returns:
            str: đường dẫn file đã lưu
        """
        ensureDir(os.path.dirname(basePath))
        h, w = imageCv2.shape[:2]
        
        # Chạy SegFormer trên vùng crop quanh bbox (isolate từng face)
        parsing = self._runSegFormerForFace(imageCv2, bbox)
        
        # Lọc bỏ face/hair của khuôn mặt lân cận (nếu có nhiều face)
        if parsing is not None and allBboxes is not None and len(allBboxes) > 1:
            parsing = self._filterParsingForFace(parsing, bbox, allBboxes)
        
        # Face mask enhanced bằng 3D mesh projection đã thực hiện ở trên.
        # Lưu ý: Phần code mở rộng hair mask (_dilateHairByYaw) gây rác mask (artifact)
        # đã bị cắt bỏ theo yêu cầu của user.
        
        # Tạo 4 ảnh con
        img1 = self._createBboxImage(imageCv2, bbox, faceId, poseInfo)
        
        segMask = self._createSegmentationMask(parsing, w, h)
        img2 = segMask if segMask is not None else np.full_like(imageCv2, 128)
        
        img3 = self._createGeometryImage(imageCv2, bbox, landmarks106, poseInfo, parsing, w, h)
        img4 = self._createRedMaskImage(imageCv2, parsing, w, h)
        
        # Ghép 2x2
        topRow = np.hstack([img1, img2])
        bottomRow = np.hstack([img3, img4])
        combined = np.vstack([topRow, bottomRow])
        
        # Lưu
        outputPath = basePath + ".png"
        cv2.imwrite(outputPath, combined)
        self.logger.info(f"  Saved visualization (2x2): {os.path.basename(outputPath)}")
        
        return outputPath
    
    def _runSegFormer(self, imageCv2):
        """
        Chạy SegFormer 1 lần, trả về parsing map (512x512).
        
        SegFormer output logits ở ~H/4 x W/4 → upsample về 512x512.
        
        Returns:
            numpy array (512, 512) class labels, hoặc None
        """
        if self.segformer is None:
            return None
        
        try:
            # Convert BGR → RGB → PIL
            imageRgb = cv2.cvtColor(imageCv2, cv2.COLOR_BGR2RGB)
            pilImage = Image.fromarray(imageRgb)
            
            # Preprocess bằng SegformerImageProcessor
            inputs = self.imageProcessor(images=pilImage, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.segformer(**inputs)
                logits = outputs.logits  # (1, 19, ~H/4, ~W/4)
                
                # Upsample về 512x512
                upsampled = F.interpolate(
                    logits,
                    size=(512, 512),
                    mode='bilinear',
                    align_corners=False
                )
                parsing = upsampled.argmax(dim=1)[0].cpu().numpy()
            
            return parsing
        except Exception as e:
            self.logger.error(f"Lỗi SegFormer forward: {e}")
            return None
    
    def _runSegFormerForFace(self, imageCv2, bbox, margin=0.8):
        """
        Crop vùng quanh bbox → chạy SegFormer → map parsing về full-image resolution.
        
        Đảm bảo mỗi face chỉ có parsing của chính nó, không lẫn face khác.
        
        Args:
            imageCv2: numpy array (BGR) — ảnh gốc full
            bbox: [x1, y1, x2, y2] — bounding box của face
            margin: float — tỉ lệ mở rộng bbox (0.8 = 80% mỗi phía)
        
        Returns:
            numpy array (imgH, imgW) class labels ở full-image resolution, hoặc None
        """
        if self.segformer is None:
            return None
        
        try:
            imgH, imgW = imageCv2.shape[:2]
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            
            # Mở rộng bbox — phía trên nhiều hơn để bắt tóc
            cx1 = max(0, int(x1 - bw * margin))
            cy1 = max(0, int(y1 - bh * margin * 1.5))  # Trên mở rộng thêm 50%
            cx2 = min(imgW, int(x2 + bw * margin))
            cy2 = min(imgH, int(y2 + bh * margin * 0.5))  # Dưới ít hơn (cổ/vai)
            
            # Crop ảnh
            cropped = imageCv2[cy1:cy2, cx1:cx2]
            
            if cropped.size == 0:
                self.logger.warning("  Crop region rỗng, fallback full image")
                return self._runSegFormer(imageCv2)
            
            cropH, cropW = cropped.shape[:2]
            self.logger.info(f"  SegFormer crop: ({cx1},{cy1})-({cx2},{cy2}) = {cropW}x{cropH}")
            
            # Chạy SegFormer trên crop
            cropParsing = self._runSegFormer(cropped)
            if cropParsing is None:
                return None
            
            # Scale parsing (512x512) về kích thước crop thực
            cropParsingResized = cv2.resize(
                cropParsing.astype(np.uint8),
                (cropW, cropH),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)
            
            # Tạo full-image parsing (toàn background = 0)
            fullParsing = np.zeros((imgH, imgW), dtype=np.int64)
            
            # Đặt crop parsing vào đúng vị trí
            fullParsing[cy1:cy2, cx1:cx2] = cropParsingResized
            
            return fullParsing
            
        except Exception as e:
            self.logger.error(f"  Lỗi SegFormer crop: {e}")
            return None
    
    def _filterParsingForFace(self, parsing, targetBbox, allBboxes):
        """
        Lọc parsing: chỉ giữ face/hair thuộc targetBbox, loại bỏ face lân cận.
        
        Chiến lược 2 lớp:
        1. Exclusion zones: tạo vùng cấm quanh các bbox khác
        2. Connected components: giữ face component gần targetBbox nhất
        
        Args:
            parsing: numpy (H, W) — full-image parsing
            targetBbox: [x1, y1, x2, y2] — bbox của face hiện tại
            allBboxes: list of [x1, y1, x2, y2] — tất cả bboxes
        
        Returns:
            numpy (H, W) — parsing đã lọc
        """
        try:
            pH, pW = parsing.shape[:2]
            filtered = parsing.copy()
            
            tx1, ty1, tx2, ty2 = targetBbox
            tCenterX = (tx1 + tx2) / 2
            tCenterY = (ty1 + ty2) / 2
            
            # ========================================
            # LỚP 1: Exclusion zones quanh bbox khác
            # ========================================
            # Với mỗi bbox khác, tạo vùng cấm (bbox thu nhỏ 20%)
            # — mọi face/hair pixel trong vùng này → background
            for otherBbox in allBboxes:
                ox1, oy1, ox2, oy2 = otherBbox
                oCenterX = (ox1 + ox2) / 2
                oCenterY = (oy1 + oy2) / 2
                
                # Bỏ qua bbox trùng (cùng face)
                dist = np.sqrt((tCenterX - oCenterX)**2 + (tCenterY - oCenterY)**2)
                if dist < 5:  # ~cùng bbox
                    continue
                
                # Tạo exclusion zone = bbox khác (co vào 20% để chắc chắn)
                ow, oh = ox2 - ox1, oy2 - oy1
                shrink = 0.2
                ex1 = max(0, int(ox1 + ow * shrink))
                ey1 = max(0, int(oy1 + oh * shrink))
                ex2 = min(pW, int(ox2 - ow * shrink))
                ey2 = min(pH, int(oy2 - oh * shrink))
                
                # Xóa face/hair trong exclusion zone
                for cls in FACE_CLASSES | HAIR_CLASSES:
                    regionMask = (filtered[ey1:ey2, ex1:ex2] == cls)
                    filtered[ey1:ey2, ex1:ex2][regionMask] = 0
            
            # ========================================
            # LỚP 2: Connected components cho FACE
            # ========================================
            # Giữ chỉ face component gần targetBbox nhất
            faceMask = np.zeros((pH, pW), dtype=np.uint8)
            for cls in FACE_CLASSES:
                faceMask[filtered == cls] = 255
            
            numLabels, labels = cv2.connectedComponents(faceMask)
            
            if numLabels > 2:  # Có nhiều hơn 1 face component
                bestLabel = -1
                bestDist = float('inf')
                
                for label in range(1, numLabels):
                    ys, xs = np.where(labels == label)
                    if len(xs) == 0:
                        continue
                    centroid = (np.mean(xs), np.mean(ys))
                    d = np.sqrt((centroid[0] - tCenterX)**2 + (centroid[1] - tCenterY)**2)
                    if d < bestDist:
                        bestDist = d
                        bestLabel = label
                
                # Xóa face components xa target
                removedPixels = 0
                for label in range(1, numLabels):
                    if label != bestLabel:
                        wrongFaceMask = (labels == label)
                        filtered[wrongFaceMask] = 0
                        removedPixels += np.sum(wrongFaceMask)
                
                if removedPixels > 0:
                    self.logger.info(f"  Loại bỏ {removedPixels} face pixels thuộc face khác")
            
            # ========================================
            # LỚP 3: Connected components cho HAIR
            # ========================================
            # Giữ hair components gần target face, loại components xa
            hairMask = np.zeros((pH, pW), dtype=np.uint8)
            for cls in HAIR_CLASSES:
                hairMask[filtered == cls] = 255
            
            numHairLabels, hairLabels = cv2.connectedComponents(hairMask)
            
            if numHairLabels > 2:  # Có nhiều hair components
                tbw = tx2 - tx1
                maxHairDist = tbw * 2.0  # Hair không nên xa quá 2x bbox width
                
                removedHair = 0
                for label in range(1, numHairLabels):
                    ys, xs = np.where(hairLabels == label)
                    if len(xs) == 0:
                        continue
                    centroid = (np.mean(xs), np.mean(ys))
                    d = np.sqrt((centroid[0] - tCenterX)**2 + (centroid[1] - tCenterY)**2)
                    
                    if d > maxHairDist:
                        wrongHairMask = (hairLabels == label)
                        filtered[wrongHairMask] = 0
                        removedHair += np.sum(wrongHairMask)
                
                if removedHair > 0:
                    self.logger.info(f"  Loại bỏ {removedHair} hair pixels xa face hiện tại")
            
            return filtered
            
        except Exception as e:
            self.logger.warning(f"  Lỗi filter parsing: {e}")
            return parsing
    
    # ============================================================
    # 1. BBOX IMAGE
    # ============================================================
    def _createBboxImage(self, imageCv2, bbox, faceId, poseInfo=None):
        """Ảnh gốc + bounding box xanh lá + label."""
        vis = imageCv2.copy()
        x1, y1, x2, y2 = [int(c) for c in bbox]
        
        # Vẽ bbox xanh lá
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label với pose info
        if poseInfo:
            label = f"Face #{faceId} | yaw={poseInfo.get('yaw', 0):.1f}"
        else:
            label = f"Face #{faceId}"
        
        labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(vis, (x1, y1 - labelSize[1] - 10), (x1 + labelSize[0] + 5, y1), (0, 255, 0), -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return vis
    
    # ============================================================
    # 2. SEGMENTATION MASK
    # ============================================================
    def _createSegmentationMask(self, parsing, targetW, targetH):
        """Face=trắng, Hair=đen, Background=xám."""
        if parsing is None:
            return None
        
        pH, pW = parsing.shape[:2]
        mask = np.full((pH, pW, 3), 128, dtype=np.uint8)  # Xám (background)
        
        for cls in FACE_CLASSES:
            mask[parsing == cls] = [255, 255, 255]  # Trắng
        
        for cls in HAIR_CLASSES:
            mask[parsing == cls] = [0, 0, 0]  # Đen
        
        if (pH, pW) != (targetH, targetW):
            mask = cv2.resize(mask, (targetW, targetH), interpolation=cv2.INTER_NEAREST)
        return mask
    
    # ============================================================
    # 3. GEOMETRY IMAGE
    # ============================================================
    def _createGeometryImage(self, imageCv2, bbox, landmarks106, poseInfo, parsing, targetW, targetH):
        """
        Ảnh gốc + hair overlay đỏ + face wireframe vàng + landmarks.
        Giống style ảnh _geometry.jpg mẫu.
        """
        vis = imageCv2.copy()
        h, w = vis.shape[:2]
        
        # Vẽ vùng hair đỏ bán trong suốt (từ parsing)
        if parsing is not None:
            pH, pW = parsing.shape[:2]
            hairMaskP = np.zeros((pH, pW), dtype=np.uint8)
            for cls in HAIR_CLASSES:
                hairMaskP[parsing == cls] = 255
            hairMask = cv2.resize(hairMaskP, (w, h), interpolation=cv2.INTER_NEAREST) if (pH, pW) != (h, w) else hairMaskP
            
            # Overlay đỏ cho hair
            redOverlay = np.zeros_like(vis)
            redOverlay[:, :, 2] = 200  # Kênh Red (BGR)
            vis[hairMask > 0] = cv2.addWeighted(vis, 0.4, redOverlay, 0.6, 0)[hairMask > 0]
        
        # Vẽ face contour bằng parsing (viền trắng)
        if parsing is not None:
            pH, pW = parsing.shape[:2]
            faceMaskP = np.zeros((pH, pW), dtype=np.uint8)
            for cls in FACE_CLASSES:
                faceMaskP[parsing == cls] = 255
            faceMask = cv2.resize(faceMaskP, (w, h), interpolation=cv2.INTER_NEAREST) if (pH, pW) != (h, w) else faceMaskP
            contours, _ = cv2.findContours(faceMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)
        
        # Vẽ landmarks 106 điểm + wireframe
        if landmarks106 is not None and len(landmarks106) >= 5:
            self._drawLandmarksWireframe(vis, landmarks106)
        
        # Text info góc trái trên
        if poseInfo:
            textLines = [
                f"Yaw: {poseInfo.get('yaw', 0):.1f} deg",
                f"Pitch: {poseInfo.get('pitch', 0):.1f} deg",
                f"Roll: {poseInfo.get('roll', 0):.1f} deg",
            ]
            y0 = 20
            for i, line in enumerate(textLines):
                cv2.putText(vis, line, (10, y0 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        
        return vis
    
    def _drawLandmarksWireframe(self, vis, landmarks):
        """
        Vẽ landmarks và face wireframe kiểu tam giác.
        
        Landmarks 106 hoặc 5 điểm đều hỗ trợ.
        """
        numPts = len(landmarks)
        
        if numPts >= 106:
            # Vẽ tất cả 106 points nhỏ (xanh dương)
            for pt in landmarks:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(vis, (x, y), 1, (255, 100, 0), -1)
            
            # Vẽ wireframe vàng nối các điểm chính
            # Contour mặt: điểm 0-32
            jawPts = landmarks[0:33].astype(np.int32)
            cv2.polylines(vis, [jawPts], False, (0, 200, 255), 1, cv2.LINE_AA)
            
            # Lông mày trái: 33-37
            if numPts > 37:
                browL = landmarks[33:38].astype(np.int32)
                cv2.polylines(vis, [browL], False, (0, 200, 255), 1, cv2.LINE_AA)
            
            # Lông mày phải: 42-46
            if numPts > 46:
                browR = landmarks[42:47].astype(np.int32)
                cv2.polylines(vis, [browR], False, (0, 200, 255), 1, cv2.LINE_AA)
            
            # Mũi: 51-54 (sống mũi), 55-59 (cánh mũi)
            if numPts > 59:
                noseBridge = landmarks[51:55].astype(np.int32)
                cv2.polylines(vis, [noseBridge], False, (0, 200, 255), 1, cv2.LINE_AA)
                noseWing = landmarks[55:60].astype(np.int32)
                cv2.polylines(vis, [noseWing], True, (0, 255, 0), 1, cv2.LINE_AA)
                # Vẽ vòng tròn ở đầu mũi
                noseTip = landmarks[54].astype(int)
                cv2.circle(vis, (noseTip[0], noseTip[1]), 8, (0, 255, 0), 1)
            
            # Mắt trái: 60-67
            if numPts > 67:
                eyeL = landmarks[60:68].astype(np.int32)
                cv2.polylines(vis, [eyeL], True, (0, 200, 255), 1, cv2.LINE_AA)
            
            # Mắt phải: 68-75
            if numPts > 75:
                eyeR = landmarks[68:76].astype(np.int32)
                cv2.polylines(vis, [eyeR], True, (0, 200, 255), 1, cv2.LINE_AA)
            
            # Môi ngoài: 76-87, Môi trong: 88-95
            if numPts > 87:
                lipOuter = landmarks[76:88].astype(np.int32)
                cv2.polylines(vis, [lipOuter], True, (0, 200, 255), 1, cv2.LINE_AA)
            if numPts > 95:
                lipInner = landmarks[88:96].astype(np.int32)
                cv2.polylines(vis, [lipInner], True, (0, 200, 255), 1, cv2.LINE_AA)
            
            # Wireframe tam giác nối các điểm chính (giống ảnh mẫu)
            keyIndices = [0, 16, 32, 33, 46, 51, 54, 60, 68, 76, 82]
            keyPts = []
            for idx in keyIndices:
                if idx < numPts:
                    keyPts.append(landmarks[idx].astype(int))
            
            # Vẽ các đường nối tam giác vàng
            for i in range(len(keyPts)):
                for j in range(i + 1, len(keyPts)):
                    cv2.line(vis, tuple(keyPts[i]), tuple(keyPts[j]), (0, 200, 255), 1, cv2.LINE_AA)
        
        elif numPts >= 5:
            # Fallback: chỉ vẽ 5 landmarks chính
            colors = [(0, 255, 255), (0, 255, 255), (0, 255, 0), (255, 100, 0), (255, 100, 0)]
            for i, pt in enumerate(landmarks[:5]):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(vis, (x, y), 3, colors[i % len(colors)], -1)
    
    # ============================================================
    # 4. RED MASK IMAGE
    # ============================================================
    def _createRedMaskImage(self, imageCv2, parsing, targetW, targetH):
        """
        Ảnh gốc + overlay đỏ bán trong suốt trên vùng face + hair.
        Giống style ảnh _red.jpg mẫu.
        """
        vis = imageCv2.copy()
        h, w = vis.shape[:2]
        
        if parsing is None:
            return vis
        
        pH, pW = parsing.shape[:2]
        
        # Tạo mask face + hair
        faceHairMaskP = np.zeros((pH, pW), dtype=np.uint8)
        for cls in FACE_CLASSES:
            faceHairMaskP[parsing == cls] = 255
        for cls in HAIR_CLASSES:
            faceHairMaskP[parsing == cls] = 255
        
        # Resize về kích thước gốc nếu cần
        faceHairMask = cv2.resize(faceHairMaskP, (w, h), interpolation=cv2.INTER_NEAREST) if (pH, pW) != (h, w) else faceHairMaskP
        
        # Overlay đỏ (BGR: Blue=0, Green=0, Red=200)
        redOverlay = np.zeros_like(vis)
        redOverlay[:, :, 2] = 200  # Kênh Red
        
        # Blend: 40% ảnh gốc + 60% đỏ trên vùng face+hair
        blended = cv2.addWeighted(vis, 0.4, redOverlay, 0.6, 0)
        vis[faceHairMask > 0] = blended[faceHairMask > 0]
        
        return vis
    

    # ============================================================
    # 6. ENHANCE FACE MASK VỚI 3D MESH
    # ============================================================
    def _enhanceFaceMaskWith3D(self, parsing, vertices3D):
        """
        Bổ sung face region bằng 3D mesh projection.
        
        Dùng convex hull từ 3D projected points để fill vùng face
        mà SegFormer bỏ sót (thường xảy ra khi profile >45°).
        
        Chỉ fill vào vùng background/cloth, KHÔNG ghi đè hair.
        
        Args:
            parsing: numpy (H, W) — parsing map (full-image resolution)
            vertices3D: numpy (3, N) — 3D vertices (x, y, z in image coords)
        
        Returns:
            numpy (H, W) — parsing map đã enhance
        """
        try:
            pH, pW = parsing.shape[:2]
            
            # Lấy tọa độ 2D projection từ 3D vertices
            # vertices3D đã ở image space, parsing cũng ở full-image resolution
            xs = vertices3D[0, :].astype(np.int32)
            ys = vertices3D[1, :].astype(np.int32)
            
            # Clip bounds
            xs = np.clip(xs, 0, pW - 1)
            ys = np.clip(ys, 0, pH - 1)
            
            # Tạo contour points cho convex hull
            pts = np.column_stack([xs, ys])
            
            # Convex hull bao quanh toàn bộ face mesh
            hull = cv2.convexHull(pts)
            
            # Fill convex hull → tạo face region mask
            meshMask = np.zeros((pH, pW), dtype=np.uint8)
            cv2.fillConvexPoly(meshMask, hull, 255)
            
            # Chỉ fill vào vùng background/cloth (KHÔNG ghi đè hair/hat)
            # background=0, cloth=18 → fill thành skin=1
            enhancedParsing = parsing.copy()
            fillableClasses = {0, 18}  # background, cloth (SegFormer)
            fillMask = np.isin(parsing, list(fillableClasses)) & (meshMask > 0)
            enhancedParsing[fillMask] = 1  # Đánh dấu là skin (class 1)
            
            filledPixels = np.sum(fillMask)
            self.logger.info(f"  3D mesh enhanced: {filledPixels} pixels bổ sung vào face mask")
            
            return enhancedParsing
            
        except Exception as e:
            self.logger.warning(f"  Lỗi enhance face mask với 3D: {e}")
            return parsing  # Fallback: trả về parsing gốc
    
    def isAvailable(self):
        """Kiểm tra SegFormer đã sẵn sàng chưa."""
        return self.segformer is not None
