# Try Hairstyle with Stable Diffusion

## 1. Yêu cầu hệ thống
- Python >= 3.10
- GPU NVIDIA hỗ trợ CUDA (khuyến nghị CUDA 11.8)
- Linux / WSL (Windows nên dùng WSL để ổn định)

--------------------------------------------------

## 2. Tạo và kích hoạt môi trường ảo

```bash
python -m venv venv_wsl
source venv_wsl/bin/activate
```

--------------------------------------------------

## 3. Cài đặt Môi trường (Theo đúng thứ tự để tránh lỗi)

**Bước 1: Cài đặt PyTorch & xFormers (BẮT BUỘC TRƯỚC)**
*Lưu ý: Cần cài PyTorch hỗ trợ CUDA 12.1 trở lên cho RTX 3060.*

```bash
# Gỡ bản cũ nếu có
pip uninstall torch torchvision torchaudio xformers -y

# 1. Cài PyTorch (Stable 2.5.1 + CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. Cài xFormers (Tương thích với PyTorch đã cài)
pip install xformers
```

**Bước 2: Cài đặt các thư viện dự án (Dependencies)**

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

--------------------------------------------------

## 4. Kiểm tra PyTorch & CUDA

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

## 5. Tải dữ liệu (Dataset)

FFHQ (khuôn mặt chất lượng cao):
https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL

K-Hairstyle (kiểu tóc tham chiếu):
https://psh01087.github.io/K-Hairstyle/

--------------------------------------------------

## 6. Chuẩn bị HuggingFace CLI

pip install huggingface_hub
huggingface-cli login

(Lưu ý: cần tài khoản HuggingFace)

--------------------------------------------------

## 7. Tải Model & Thư viện (QUAN TRỌNG)

### Cách 1: Tự động (Khuyên dùng)
Mình đã chuẩn bị sẵn script để tải toàn bộ model cần thiết.

1. Đảm bảo đã kích hoạt môi trường ảo:
```bash
source venv_wsl/bin/activate
```
2. Chạy lệnh sau:
```bash
python download_models.py
```
Script sẽ tự động tải:
- ControlNet Depth (SDXL)
- InstantID & IP-Adapter
- Face Parsing models
- CLIP Image Encoder

### Cách 2: Tải Thủ công (Nếu script lỗi)

### 8.1 Stable Diffusion XL Inpainting (1024x1024)

mkdir -p models/stable-diffusion
cd models/stable-diffusion

hf download diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
  --local-dir sd_xl_inpainting

--------------------------------------------------

### 8.2 ControlNet Depth SDXL

mkdir -p models/controlnet_depth
cd models/controlnet_depth

hf download diffusers/controlnet-depth-sdxl-1.0 \
  --local-dir .

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

## 8. Ghi chú
- BiSeNet: tách mask tóc cho Inpainting
- IP-Adapter FaceID: giữ nguyên khuôn mặt
- IP-Adapter Plus: copy kiểu tóc
--------------------------------------------------

## 9. Troubleshooting (Các lỗi thường gặp và cách khắc phục)


### 9.1 Lỗi Xung đột thư viện (Transformers vs Diffusers)
- **Lỗi:** `ImportError: cannot import name 'MT5Tokenizer'` hoặc `Qwen2_5_VL...`
- **Nguyên nhân:** Xung đột phiên bản giữa `transformers`, `diffusers`, và `sentencepiece`.
- **Cách sửa:** Sử dụng phiên bản "điểm ngọt" đã test kỹ:
    ```bash
    pip install transformers==4.49.0
    ```

### 9.2 Lỗi Runtime Crash `NoneType` (IP-Adapter)
- **Lỗi:** `AttributeError: 'NoneType' object has no attribute 'image_projection_layers'`
- **Nguyên nhân:** Model IP-Adapter không load được nhưng code vẫn cố dùng.
- **Cách sửa:** Đã patch lại code `diffusion.py` để tự động bỏ qua IP-Adapter nếu load lỗi, giúp app không bị crash.

### 9.3 Lỗi Lệch kiểu dữ liệu (FP16 vs FP32)
- **Lỗi:** `RuntimeError: Input type (HalfTensor) and weight type (FloatTensor)...`
- **Nguyên nhân:** Model chính chạy FP16 nhưng IP-Adapter chạy FP32.
- **Cách sửa:** Đã update code để ép kiểu toàn bộ pipeline sang FP16.

### 9.4 Các lỗi cũ hơn
- **Lỗi `module 'torch' has no attribute 'xpu'`**: Do `accelerate` mới không tương thích Windows. Sửa bằng `pip install accelerate==0.26.0` (hoặc mới hơn nếu dùng WSL).
- **Lỗi `InstantX... ip-adapter.bin not found`**: Sai đường dẫn model. Sửa bằng cách dùng đường dẫn tuyệt đối.
--------------------------------------------------

## 10. Hướng dẫn Chạy Hệ thống (Run App)

### Cách 1: Chạy bằng Docker Compose (Khuyên dùng cho Production)
Dễ dàng nhất vì đã bao gồm Redis, API, và Worker.

```bash
# Bật Docker Desktop trước, sau đó chạy:
docker compose up -d backend worker redis
```
- API sẽ chạy tại: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

---

### Cách 2: Chạy trên WSL (Khuyên dùng cho Development)

> **Yêu cầu:** Mở **4 terminal WSL** riêng biệt (hoặc dùng tmux/screen).

#### Terminal 1: Khởi động Redis
```bash
# Cài Redis nếu chưa có
sudo apt update && sudo apt install redis-server -y

# Chạy Redis server
redis-server
```

#### Terminal 2: Khởi động Backend API (FastAPI)
```bash
# Di chuyển vào thư mục dự án
cd /mnt/c/Users/Admin/Desktop/TryHairStyle

# Kích hoạt môi trường ảo
source venv_wsl/bin/activate

# Chạy FastAPI server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
API sẽ chạy tại: `http://localhost:8000`
Swagger UI: `http://localhost:8000/docs`

#### Terminal 3: Khởi động Celery Worker (Xử lý AI Tasks)
```bash
# Di chuyển vào thư mục dự án
cd /mnt/c/Users/Admin/Desktop/TryHairStyle

# Kích hoạt môi trường ảo
source venv_wsl/bin/activate

# Chạy Celery worker (Thêm pool=solo để chạy ổn định trên Windows/WSL)
celery -A backend.app.tasks worker --loglevel=info --pool=solo
```

#### Terminal 4: Khởi động Frontend (ReactJS)
```bash
# Di chuyển vào thư mục frontend
cd /mnt/c/Users/Admin/Desktop/TryHairStyle/frontend

# Cài dependencies (lần đầu)
npm install

# Chạy dev server
npm run dev
```
Frontend sẽ chạy tại: `http://localhost:5173`

---

### Cách 3: Chạy trên Windows (Powershell) - Chỉ Frontend

> **Lưu ý:** Backend nên chạy trên WSL để tận dụng GPU với CUDA ổn định hơn.

**Chạy Frontend trên Windows:**
```powershell
cd C:\Users\Admin\Desktop\TryHairStyle\frontend
npm install
npm run dev
```
Frontend sẽ chạy tại: `http://localhost:5173`

--------------------------------------------------

## 11. Chạy Kiểm Thử (Verification Scripts)

Để kiểm tra nhanh hệ thống (Backend Logic), bạn có thể dùng script CLI:

### 11.1 CLI Test (Chạy ngầm)
Tự động chạy pipeline hair transfer với ảnh ngẫu nhiên từ FFHQ.
```bash
python backend/tests/test_cli_ffhq.py
```
- Kết quả lưu tại: `backend/output/cli_test_result.png`

### 11.2 UI Test (Giao diện trực quan - Gradio)
Bật giao diện web nhỏ gọn để chọn ảnh và chạy thử (Dùng để debug nhanh nếu không muốn chạy React).
```bash
python backend/tests/test_ui_gradio.py
```
- Truy cập: `http://127.0.0.1:7861`

--------------------------------------------------