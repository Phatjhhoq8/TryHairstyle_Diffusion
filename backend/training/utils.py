"""
Training Utils — Các hàm tiện ích chung cho training pipeline.

Bao gồm:
- Setup logging
- Auto-detect GPU/CPU
- Tạo tên file output
- Load ảnh an toàn
- L2-normalize embedding
"""

import os
import re
import logging
import numpy as np
import cv2
import torch
from datetime import datetime
from pathlib import Path


def convertToWslPath(winPath):
    """
    Tự động chuyển đường dẫn Windows sang WSL.
    
    Ví dụ:
        C:\\Users\\Admin\\test.jpg → /mnt/c/Users/Admin/test.jpg
        D:\\Data\\img.png → /mnt/d/Data/img.png
    
    Nếu đang chạy trên Windows hoặc path đã là Linux → giữ nguyên.
    
    Args:
        winPath: str — đường dẫn (Windows hoặc Linux)
    
    Returns:
        str — đường dẫn đã convert
    """
    pathStr = str(winPath)
    
    # Chỉ convert nếu đang trên Linux/WSL VÀ path có dạng Windows
    if os.name == "nt":
        # Đang trên Windows → giữ nguyên
        return pathStr
    
    # Kiểm tra pattern Windows: C:\ hoặc C:/
    match = re.match(r'^([A-Za-z]):[/\\](.*)$', pathStr)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).replace('\\', '/')
        return f"/mnt/{drive}/{rest}"
    
    # Path đã là Linux → giữ nguyên
    return pathStr


def setupLogger(name="TrainingFace", level=logging.INFO):
    """
    Tạo logger chuẩn với format thống nhất.
    
    Args:
        name: Tên logger
        level: Logging level
    
    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(name)s] %(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def getDevice():
    """
    Auto-detect GPU (CUDA) hoặc fallback CPU.
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpuName = torch.cuda.get_device_name(0)
        print(f"[Utils] Sử dụng GPU: {gpuName}")
    else:
        device = torch.device("cpu")
        print("[Utils] GPU không khả dụng, sử dụng CPU")
    return device


def generateFilename(faceId, yaw, extension="npy"):
    """
    Tạo tên file output theo format chuẩn.
    
    Format: face_XX_yaw_YY_YYYYMMDD.ext
    
    Args:
        faceId: ID khuôn mặt (int)
        yaw: Góc yaw (float, đơn vị độ)
        extension: Phần mở rộng file (npy, png, json)
    
    Returns:
        str: Tên file
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    yawInt = int(abs(yaw))
    return f"face_{faceId:02d}_yaw_{yawInt}_{timestamp}.{extension}"


def loadImageSafe(imagePath):
    """
    Load ảnh an toàn với error handling.
    
    Args:
        imagePath: Đường dẫn tới file ảnh
    
    Returns:
        numpy.ndarray (BGR) hoặc None nếu lỗi
    """
    logger = setupLogger("Utils")
    
    if not os.path.exists(imagePath):
        logger.error(f"File không tồn tại: {imagePath}")
        return None
    
    try:
        image = cv2.imread(str(imagePath))
        if image is None:
            logger.error(f"Không thể đọc ảnh: {imagePath}")
            return None
        logger.info(f"Loaded ảnh: {imagePath} — shape={image.shape}")
        return image
    except Exception as e:
        logger.error(f"Lỗi khi load ảnh {imagePath}: {e}")
        return None


def normalizeEmbedding(embedding):
    """
    L2-normalize embedding vector.
    
    Args:
        embedding: numpy array hoặc torch.Tensor
    
    Returns:
        numpy array đã normalize
    """
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().numpy()
    
    embedding = embedding.flatten().astype(np.float32)
    norm = np.linalg.norm(embedding)
    
    if norm < 1e-10:
        return embedding
    
    return embedding / norm


def ensureDir(dirPath):
    """
    Tạo thư mục nếu chưa tồn tại.
    
    Args:
        dirPath: Đường dẫn thư mục
    """
    os.makedirs(str(dirPath), exist_ok=True)


def cropFaceFromImage(image, bbox, margin=0.2):
    """
    Crop vùng khuôn mặt từ ảnh với margin mở rộng.
    
    Args:
        image: numpy array (BGR)
        bbox: [x1, y1, x2, y2]
        margin: Tỷ lệ margin mở rộng (0.0 - 1.0)
    
    Returns:
        numpy array — cropped face region
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(c) for c in bbox]
    
    # Mở rộng margin
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margin)
    my = int(bh * margin)
    
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    
    return image[y1:y2, x1:x2].copy()
