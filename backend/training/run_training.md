# Hướng dẫn chạy Training Face Pipeline

## Yêu cầu
- WSL2 (Ubuntu)
- Virtual environment `venv_wsl` đã cài đặt dependencies
- GPU NVIDIA (khuyến nghị, hỗ trợ CPU fallback)

## Pipeline Flow

```
Input Image
    │
    ▼
[1] YOLOv8-Face Detection ──► Bounding boxes
    │
    ▼
[2] InsightFace 106-Landmarks + 3DDFA V2 Pose ──► yaw, pitch, roll
    │
    ▼
[3] Embedding Extraction
    ├── |yaw| < 45°  ──► InsightFace (ArcFace)
    └── |yaw| ≥ 45°  ──► AdaFace (profile-optimized)
    │
    ▼
[4] 3D Reconstruction (chỉ khi |yaw| ≥ 45°)
    └── 3DDFA V2 ──► Dense 3D mesh vertices
    │
    ▼
[5] Visualization & Segmentation
    ├── SegFormer face parsing (19 classes)
    ├── 3D Mesh Face Enhancement (|yaw| ≥ 45°)
    │   └── Convex hull từ 3D vertices → mở rộng face mask
    ├── Directional Hair Dilation (|yaw| ≥ 45°)
    │   └── Asymmetric kernel → mở rộng tóc về phía sau ót
    └── Output 4 ảnh:
        ├── _bbox.png     — Bounding box xanh lá
        ├── _seg.png      — Segmentation mask (face=trắng, hair=đen, bg=xám)
        ├── _geometry.png — Landmarks + wireframe vàng
        └── _red.png      — Face+Hair overlay đỏ
    │
    ▼
[6] Save: .npy (embedding) + .json (metadata) + .png (visualization)
```

## Models sử dụng

| Model | Chức năng | Path |
|---|---|---|
| YOLOv8-Face | Face detection | `backend/models/yolov8n-face.pt` |
| InsightFace (antelopev2) | Landmarks + embedding | `~/.insightface/models/antelopev2/` |
| AdaFace IR-101 | Embedding cho profile | `backend/models/adaface_ir101_webface4m.ckpt` |
| 3DDFA V2 | 3D reconstruction + pose | `backend/models/3ddfa_v2/` |
| **SegFormer** | Face parsing (19 classes) | `backend/models/segformer_face_parsing/` |

> **Lưu ý:** Chạy `python download_models.py` để download tất cả models.

## Cách chạy

### 1. Activate môi trường

```bash
cd /mnt/c/Users/Admin/Desktop/TryHairStyle
source venv_wsl/bin/activate
```

### 2. Single Image

```bash
python -m backend.training.training_face --image path/to/test.jpg --output backend/training/output
```

### 3. Batch Processing (cả thư mục ảnh)

```bash
python -m backend.training.training_face --image-dir path/to/images/ --output backend/training/output
```

### 4. Tuỳ chỉnh ngưỡng yaw

```bash
python -m backend.training.training_face --image test.jpg --output backend/training/output --yaw-threshold 30
```

### 5. Sử dụng trong Python

```python
from backend.training.training_face import TrainingFacePipeline

pipeline = TrainingFacePipeline()
results = pipeline.processImage("path/to/image.jpg", "backend/training/output")
```

## Output

Mỗi khuôn mặt tạo files trong thư mục `output/face_XX/`:

| File | Mô tả |
|---|---|
| `face_XX_yaw_YY.npy` | Embedding vector 512-d (L2-normalized) |
| `face_XX_yaw_YY.png` | Visualization 2×2 (bbox + seg + geometry + red mask) |
| `face_XX_yaw_YY.json` | Metadata (yaw, pitch, roll, bbox, model, etc.) |

## Tham số CLI

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--image` | — | Đường dẫn ảnh đầu vào |
| `--image-dir` | — | Thư mục ảnh (batch mode) |
| `--output` | `backend/training/output` | Thư mục output |
| `--yaw-threshold` | `45.0` | Ngưỡng yaw chuyển model (độ) |

## Chuyển đổi đường dẫn Windows → WSL

Pipeline tự động convert đường dẫn Windows sang WSL:

| Windows | WSL |
|---|---|
| `C:\Users\Admin\Desktop\TryHairStyle\test.jpg` | `/mnt/c/Users/Admin/Desktop/TryHairStyle/test.jpg` |

```bash
python -m backend.training.training_face --image "C:\Users\Admin\Desktop\test.jpg" --output backend/training/output
```
