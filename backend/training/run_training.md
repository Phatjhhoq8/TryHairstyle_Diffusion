# Hướng dẫn chạy Training Face Pipeline

## Yêu cầu
- WSL2 (Ubuntu)
- Virtual environment `venv_wsl` đã cài đặt dependencies
- GPU NVIDIA (khuyến nghị, hỗ trợ CPU fallback)

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

Mỗi khuôn mặt tạo 3 files trong thư mục output:

| File | Mô tả |
|---|---|
| `face_XX_yaw_YY_YYYYMMDD.npy` | Embedding vector 512-d (L2-normalized) |
| `face_XX_yaw_YY_YYYYMMDD.png` | Ảnh gốc + bbox + segmentation mask |
| `face_XX_yaw_YY_YYYYMMDD.json` | Metadata (yaw, pitch, roll, bbox, model, etc.) |

## Tham số CLI

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--image` | — | Đường dẫn ảnh đầu vào |
| `--image-dir` | — | Thư mục ảnh (batch mode) |
| `--output` | `backend/training/output` | Thư mục output |
| `--yaw-threshold` | `45.0` | Ngưỡng yaw chuyển model (độ) |

## Chuyển đổi đường dẫn Windows → WSL

Pipeline tự động convert đường dẫn Windows sang WSL. Bạn có thể dùng cả 2 format:

| Windows | WSL |
|---|---|
| `C:\Users\Admin\Desktop\TryHairStyle\test.jpg` | `/mnt/c/Users/Admin/Desktop/TryHairStyle/test.jpg` |

Quy tắc: `C:\` → `/mnt/c/`, `D:\` → `/mnt/d/`, dấu `\` → `/`

Ví dụ — đường dẫn Windows cũng chạy được trong WSL:

```bash
python -m backend.training.training_face --image "C:\Users\Admin\Desktop\test.jpg" --output backend/training/output
```
