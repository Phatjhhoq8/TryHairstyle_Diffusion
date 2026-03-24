## Yêu cầu hệ thống

- **Python** 3.8
- **CUDA** 11.7+ và GPU NVIDIA (khuyến nghị ≥ 8GB VRAM)
- **Linux / WSL2** (Ubuntu 20.04+)
- **CMake** và **build-essential** (để build dlib)


## Cài đặt

### Cách 1: Dùng venv (khuyến nghị)

```bash
# 1. Tạo Virtual Environment
python3.8 -m venv hairfusion
source hairfusion/bin/activate

# 2. Cài PyTorch + CUDA 11.7
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117

# 3. Cài CMake (cần thiết để build dlib)
sudo apt-get update && sudo apt-get install -y cmake build-essential

# 4. Cài tất cả dependencies
pip install -r requirements.txt
```

### Cách 2: Dùng Conda

```bash
conda create -y -n hairfusion python=3.8
conda activate hairfusion
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

### Cách 3: Dùng Docker (đơn giản nhất)

**Yêu cầu:** [Docker](https://docs.docker.com/get-docker/) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# Build và chạy
docker-compose up --build

# Chạy nền
docker-compose up -d --build
```

Mở trình duyệt tại `http://localhost:7860`. Kết quả lưu trong `backend/data/output/`.


## Tải Model Weights

### 1) Preprocessing Models (bắt buộc)
Tải và lưu vào `backend/models/`:

| File | Kích thước | Link |
|------|-----------|------|
| `face_segment16.pth` | 50.8 MB | [Google Drive](https://drive.google.com/file/d/10GL030sNpVrxM9Ez0nXhHvs9-lsnZFGV/view?usp=sharing) |
| `shape_predictor_68_face_landmarks.dat` | 95.1 MB | [Google Drive](https://drive.google.com/file/d/1g4jTab8cNVmF2AjDz2N3uXu0cMvsvlC3/view?usp=sharing) |

### 2) VAE Model (bắt buộc)
- Tải `realisticVisionV60B1_v51VAE.safetensors` (~2GB) từ [CivitAI](https://civitai.com/models/4201?modelVersionId=130072)
- Lưu vào `backend/models/`

### 3) HairFusion Checkpoint (bắt buộc)
- Tải [hairfusion.zip (8.4GB)](https://drive.google.com/file/d/1QWB478Fc415CSqvDDe4ZczGTzh2FAXTO/view?usp=sharing)
- Giải nén và lưu vào `backend/logs/`

### Cấu trúc thư mục sau khi tải:
```
backend/
├── models/
│   ├── face_segment16.pth
│   ├── shape_predictor_68_face_landmarks.dat
│   └── realisticVisionV60B1_v51VAE.safetensors
└── logs/
    └── hairfusion/
        └── models/
            └── [Train]_[epoch=599]_[train_loss_epoch=0.3666].ckpt
```


## Chạy hệ thống

### Web UI (Gradio)

```bash
source hairfusion/bin/activate
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python -m backend.app.main
```

Mở trình duyệt tại `http://localhost:7860` để sử dụng giao diện.

**Hướng dẫn sử dụng:**
1. Upload ảnh khuôn mặt (Your Face)
2. Upload ảnh kiểu tóc mong muốn (Desired Hairstyle Reference)
3. Điều chỉnh Steps (mặc định 50) và Guidance Scale (mặc định 5.0)
4. Bấm **Generate**
5. Kết quả + file trung gian được lưu trong `backend/data/output/session_XXXXXXXX/`

### CLI Inference (Script gốc)

```bash
bash ./scripts/test.sh
```


## File trung gian (Output)

Mỗi session sẽ tạo thư mục `backend/data/output/session_XXXXXXXX/` chứa:

| Thư mục | Mô tả |
|---------|-------|
| `images/` | Ảnh đầu vào đã crop |
| `mask_hair/` | Mask vùng tóc |
| `mask_face/` | Mask vùng mặt |
| `nth/` | Keypoints visualization |
| `agnostic/` | Ảnh agnostic (giữ mặt, xoá tóc) |
| `agnostic-mask/` | Mask vùng agnostic |
| `keypoints/` | Toạ độ facial landmarks |
| `result.png` | **Ảnh kết quả cuối cùng** |


## Lưu ý khi sử dụng

- Ảnh đầu vào nên là **chân dung chính diện**, nền đơn giản
- Ảnh reference tóc cũng nên **chụp rõ kiểu tóc**, chính diện
- Hệ thống hoạt động tốt nhất với **khuôn mặt người trưởng thành**
- Tăng **Steps** (60-80) để có chất lượng cao hơn, giảm (20-30) để nhanh hơn
- Nếu dùng **WSL**, cần set `LD_LIBRARY_PATH` trước khi chạy


## Citation

If you find our work useful for your research, please cite us:
```
@inproceedings{chung2025hairfusion,
  title={What to Preserve and What to Transfer: Faithful, Identity-Preserving Diffusion-based Hairstyle Transfer},
  author={Chung, Chaeyeon and Park, Sunghyun and Kim, Jeongho and Choo, Jaegul},
  booktitle={The Association for the Advancement of Artificial Intelligence},
  year={2025}
}
```

## License

Licensed under the CC BY-NC-SA 4.0 license [https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
