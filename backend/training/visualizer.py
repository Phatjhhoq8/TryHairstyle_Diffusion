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

from backend.training.utils import setupLogger, getDevice, ensureDir

# Đường dẫn
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BISENET_CHECKPOINT = str(BASE_DIR / "backend" / "models" / "bisenet" / "79999_iter.pth")

# BiSeNet class mapping (CelebAMask-HQ 19 classes)
# 0: background
# 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye
# 6: eye_g (glasses), 7: l_ear, 8: r_ear, 9: ear_r (earring)
# 10: nose, 11: mouth, 12: u_lip, 13: l_lip
# 14: neck, 15: necklace, 16: cloth, 17: hair, 18: hat

# Nhóm face (skin + features)
FACE_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

# Nhóm hair
HAIR_CLASSES = {17}

# Background = tất cả còn lại (0, 15, 16, 18)


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
        self.bisenet = None
        self._loadBiSeNet()
    
    def _loadBiSeNet(self):
        """Load BiSeNet model cho face parsing."""
        try:
            from backend.app.services.bisenet_arch import BiSeNet
            
            if not os.path.exists(BISENET_CHECKPOINT):
                self.logger.error(f"BiSeNet checkpoint không tìm thấy: {BISENET_CHECKPOINT}")
                return
            
            model = BiSeNet(n_classes=19)
            checkpoint = torch.load(BISENET_CHECKPOINT, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            model.to(self.device)
            model.eval()
            
            self.bisenet = model
            self.logger.info("BiSeNet loaded cho face parsing")
        except Exception as e:
            self.logger.error(f"Lỗi load BiSeNet: {e}")
            self.bisenet = None
    
    def createVisualization(self, imageCv2, bbox, faceId, basePath, landmarks106=None, poseInfo=None):
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
        
        Returns:
            str: đường dẫn file đã lưu
        """
        ensureDir(os.path.dirname(basePath))
        h, w = imageCv2.shape[:2]
        
        # Chạy BiSeNet 1 lần, dùng cho cả 4 ảnh
        parsing = self._runBiSeNet(imageCv2)
        
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
    
    def _runBiSeNet(self, imageCv2):
        """
        Chạy BiSeNet 1 lần, trả về parsing map (512x512).
        
        Returns:
            numpy array (512, 512) class labels, hoặc None
        """
        if self.bisenet is None:
            return None
        
        try:
            imageRgb = cv2.cvtColor(imageCv2, cv2.COLOR_BGR2RGB)
            pilImage = Image.fromarray(imageRgb)
            pilImage = pilImage.resize((512, 512), Image.BILINEAR)
            
            imgArray = np.array(pilImage).astype(np.float32) / 255.0
            imgTensor = torch.from_numpy(imgArray.transpose(2, 0, 1))
            imgTensor = imgTensor.unsqueeze(0).to(self.device)
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            imgTensor = (imgTensor - mean) / std
            
            with torch.no_grad():
                output = self.bisenet(imgTensor)
                if isinstance(output, tuple):
                    output = output[0]
                parsing = output.squeeze(0).argmax(0).cpu().numpy()
            
            return parsing
        except Exception as e:
            self.logger.error(f"Lỗi BiSeNet forward: {e}")
            return None
    
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
        
        mask = np.full((512, 512, 3), 128, dtype=np.uint8)  # Xám (background)
        
        for cls in FACE_CLASSES:
            mask[parsing == cls] = [255, 255, 255]  # Trắng
        
        for cls in HAIR_CLASSES:
            mask[parsing == cls] = [0, 0, 0]  # Đen
        
        return cv2.resize(mask, (targetW, targetH), interpolation=cv2.INTER_NEAREST)
    
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
            hairMask512 = np.zeros((512, 512), dtype=np.uint8)
            for cls in HAIR_CLASSES:
                hairMask512[parsing == cls] = 255
            hairMask = cv2.resize(hairMask512, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Overlay đỏ cho hair
            redOverlay = np.zeros_like(vis)
            redOverlay[:, :, 2] = 200  # Kênh Red (BGR)
            vis[hairMask > 0] = cv2.addWeighted(vis, 0.4, redOverlay, 0.6, 0)[hairMask > 0]
        
        # Vẽ face contour bằng parsing (viền trắng)
        if parsing is not None:
            faceMask512 = np.zeros((512, 512), dtype=np.uint8)
            for cls in FACE_CLASSES:
                faceMask512[parsing == cls] = 255
            faceMask = cv2.resize(faceMask512, (w, h), interpolation=cv2.INTER_NEAREST)
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
        
        # Tạo mask face + hair
        faceHairMask512 = np.zeros((512, 512), dtype=np.uint8)
        for cls in FACE_CLASSES:
            faceHairMask512[parsing == cls] = 255
        for cls in HAIR_CLASSES:
            faceHairMask512[parsing == cls] = 255
        
        # Resize về kích thước gốc
        faceHairMask = cv2.resize(faceHairMask512, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Overlay đỏ (BGR: Blue=0, Green=0, Red=200)
        redOverlay = np.zeros_like(vis)
        redOverlay[:, :, 2] = 200  # Kênh Red
        
        # Blend: 40% ảnh gốc + 60% đỏ trên vùng face+hair
        blended = cv2.addWeighted(vis, 0.4, redOverlay, 0.6, 0)
        vis[faceHairMask > 0] = blended[faceHairMask > 0]
        
        return vis
    
    def isAvailable(self):
        """Kiểm tra BiSeNet đã sẵn sàng chưa."""
        return self.bisenet is not None
