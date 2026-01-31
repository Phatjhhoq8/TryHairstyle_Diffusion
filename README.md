# Try Hairstyle with Stable Diffusion

## 1. Yêu cầu hệ thống
- Python >= 3.10
- GPU NVIDIA hỗ trợ CUDA (khuyến nghị CUDA 11.8)
- Linux / WSL (Windows nên dùng WSL để ổn định)

--------------------------------------------------

## 2. Tạo và kích hoạt môi trường ảo

python -m venv venv
source venv/bin/activate

Nâng cấp pip:
pip install --upgrade pip

--------------------------------------------------

## 3. Cài đặt PyTorch (CUDA 11.8)

pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

--------------------------------------------------

## 4. Cài đặt xFormers (tối ưu bộ nhớ & tốc độ)

pip install xformers==0.0.33.post2 --no-deps

--------------------------------------------------

## 5. Kiểm tra PyTorch & CUDA

python - << 'EOF'
import torch, torchvision, torchaudio
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
EOF

Nếu cuda: True và hiện tên GPU → cài đặt thành công.

--------------------------------------------------

## 6. Tải dữ liệu (Dataset)

FFHQ (khuôn mặt chất lượng cao):
https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL

K-Hairstyle (kiểu tóc tham chiếu):
https://psh01087.github.io/K-Hairstyle/

--------------------------------------------------

## 7. Chuẩn bị HuggingFace CLI

pip install huggingface_hub
huggingface-cli login

(Lưu ý: cần tài khoản HuggingFace)

--------------------------------------------------

## 8. Tải các mô hình cần thiết

### 8.1 Stable Diffusion v1.5

mkdir -p models/stable-diffusion
cd models/stable-diffusion

hf download runwayml/stable-diffusion-v1-5 \
  --local-dir sd15

--------------------------------------------------

### 8.2 ControlNet Depth (giữ hình dạng đầu)

mkdir -p models/controlnet
cd models/controlnet

hf download lllyasviel/control_v11f1p_sd15_depth \
  --local-dir depth

--------------------------------------------------

### 8.3 IP-Adapter FaceID (giữ identity khuôn mặt)

mkdir -p models/ip_adapter_faceid
cd models/ip_adapter_faceid

hf download h94/IP-Adapter-FaceID \
  --local-dir .

--------------------------------------------------

### 8.4 IP-Adapter Plus (copy kiểu tóc từ ảnh mẫu)

mkdir -p models/ip_adapter_hair
cd models/ip_adapter_hair

hf download h94/IP-Adapter \
  --local-dir .

--------------------------------------------------

### 8.5 BiSeNet – Face Parsing (tách mask tóc)

mkdir -p models/bisenet
cd models/bisenet

Tải thủ công các file sau:

- 79999_iter.pth  
  https://huggingface.co/vivym/face-parsing-bisenet/blob/768606b84908769d31ddd78b2e1105319839edfa/79999_iter.pth

- best_dice_loss_mitou_0.655.pth  
  https://drive.google.com/file/d/1ulUgHwFct-vFwGCAfJ4Oa9DBlNDzm5r4/view

--------------------------------------------------

## 9. Ghi chú
- BiSeNet: tách mask tóc cho Inpainting
- IP-Adapter FaceID: giữ nguyên khuôn mặt
- IP-Adapter Plus: copy kiểu tóc
- ControlNet Depth: giữ hình dạng đầu & góc nhìn

--------------------------------------------------

## 10. Docker Training (Khuyến nghị)

### 10.1 Yêu cầu
- Docker với NVIDIA Container Toolkit
- WSL2 (Windows) hoặc Linux
- GPU NVIDIA với driver >= 525

### 10.2 Build Docker Image
```bash
# Trong WSL hoặc Linux
docker build -t hairstyle-training .
```

### 10.3 Chạy Training với Docker Compose
```bash
# GPU training với docker-compose
docker compose up training
```

### 10.4 Chạy Training trực tiếp
```bash
docker run --gpus all -v $(pwd):/app \
  hairstyle-training \
  python backend/training/train_ip_adapter.py \
    --num_epochs 100 \
    --train_batch_size 4 \
    --save_steps 500
```

--------------------------------------------------

## 11. Training IP-Adapter cho Hair Style

### 11.1 Chạy với WSL (không cần Docker)
```bash
# Kích hoạt môi trường
cd /mnt/c/Users/Admin/Desktop/TryHairStyle
source venv_wsl/bin/activate

# Chạy training
python backend/training/train_ip_adapter.py \
  --num_epochs 100 \
  --train_batch_size 4 \
  --save_steps 500 \
  --mixed_precision fp16
```

### 11.2 Các tham số training quan trọng
| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--num_epochs` | 100 | Số epoch training |
| `--train_batch_size` | 4 | Batch size (giảm nếu OOM) |
| `--learning_rate` | 1e-4 | Learning rate |
| `--save_steps` | 500 | Lưu checkpoint mỗi N steps |
| `--mixed_precision` | fp16 | Mixed precision (fp16/bf16/no) |

### 11.3 Output
Checkpoints sẽ được lưu tại: `backend/output/ip_adapter_hair/`

```
checkpoint-500/
├── ip_adapter.bin     # IP-Adapter weights (~89MB)
├── model.safetensors  # Full model state
├── optimizer.bin      # Optimizer state
└── ...
```

### 11.4 Sử dụng weights đã train
```python
import torch

# Load IP-Adapter weights
ckpt = torch.load("backend/output/ip_adapter_hair/checkpoint-XXX/ip_adapter.bin")
image_proj_weights = ckpt["image_proj"]
adapter_weights = ckpt["ip_adapter"]
```
