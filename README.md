# 💇 TryHairStyle — Hệ thống Tạo Kiểu Tóc bằng AI

> Ứng dụng AI chuyển đổi kiểu tóc sử dụng Stable Diffusion XL, ControlNet, IP-Adapter và Face Parsing.
> Upload ảnh chân dung + ảnh kiểu tóc tham khảo → AI tạo ảnh kết quả với kiểu tóc mới.

---

## 📋 Mục Lục

1. [Tổng quan Kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Yêu cầu Hệ thống](#2-yêu-cầu-hệ-thống)
3. [Cài đặt & Chạy](#3-cài-đặt--chạy)
4. [Tải Models](#4-tải-models)
5. [Cấu trúc Thư mục](#5-cấu-trúc-thư-mục)
6. [API Endpoints](#6-api-endpoints)
7. [Kiểm thử](#7-kiểm-thử)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Tổng quan Kiến trúc

```
┌──────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite)                │
│                   http://localhost:5173 (dev)             │
│                   http://localhost:3000 (docker)          │
└────────────────────────┬─────────────────────────────────┘
                         │ /api/* (proxy)
┌────────────────────────▼─────────────────────────────────┐
│                Backend (FastAPI)  :8000                   │
│  POST /generate  │  POST /detect-faces  │  GET /status   │
└────────────┬─────────────────────────────────────────────┘
             │ Celery Task Queue
┌────────────▼─────────────────────────────────────────────┐
│               Celery Worker (GPU)                        │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐  │
│  │ Face     │ │ Mask     │ │ Diffusion │ │ Hair      │  │
│  │ Service  │ │ Service  │ │ Service   │ │ Color     │  │
│  │(InsightF)│ │(SegForm) │ │ (SDXL)    │ │ Service   │  │
│  └──────────┘ └──────────┘ └───────────┘ └───────────┘  │
└──────────────────────────────────────────────────────────┘
             │
┌────────────▼──┐
│  Redis :6379  │ (Message Broker)
└───────────────┘
```

### AI Pipeline

1. **Face Detection** — InsightFace + YOLOv8 phát hiện & align khuôn mặt
2. **Face/Hair Parsing** — SegFormer tách mask tóc và mặt (19 classes)
3. **Hair Transfer** — Stable Diffusion XL Inpainting + ControlNet Depth + IP-Adapter
4. **Hair Colorization** — Thay đổi màu tóc theo preset hoặc hex color

### Tech Stack

| Layer | Công nghệ |
|-------|-----------|
| Frontend | React 19, Vite 8, TailwindCSS 4 |
| Backend API | FastAPI, Pydantic, Uvicorn |
| Task Queue | Celery + Redis |
| AI/ML | PyTorch 2.4, Diffusers, Transformers, InsightFace |
| Segmentation | SegFormer (face parsing), SAM |
| Container | Docker, Docker Compose, NVIDIA Container Toolkit |

---

## 2. Yêu cầu Hệ thống

### Phần cứng
- **GPU NVIDIA** với ≥ 12GB VRAM (RTX 3060 trở lên)
- **RAM** ≥ 16GB
- **Disk** ≥ 20GB trống (cho models + Docker images)

### Phần mềm
- Python ≥ 3.10
- Node.js ≥ 18
- CUDA ≥ 12.1
- Docker + Docker Compose v2 (nếu chạy Docker)
- NVIDIA Container Toolkit (nếu chạy Docker)
- WSL2 (nếu dùng Windows)

---

## 3. Cài đặt & Chạy

### 🐳 Cách 1: Docker Compose (Khuyên dùng - Production)

```bash
# 1. Clone project
git clone <repo-url>
cd TryHairStyle

# 2. Cấu hình environment
cp .env.example .env
# Mở .env và điền HUGFACE_TOKEN

# 3. Đảm bảo models đã có trong backend/models/ (xem Mục 4)

# 4. Build & khởi động
docker compose up --build -d

# 5. Theo dõi logs
docker compose logs -f

# 6. Truy cập
#    Frontend:  http://localhost:3000
#    API Docs:  http://localhost:8000/docs
```

**Quản lý Docker:**
```bash
docker compose ps          # Xem trạng thái các service
docker compose logs -f celery-worker   # Xem logs AI worker
docker compose restart backend         # Restart backend
docker compose down                    # Dừng tất cả
docker compose up -d --build           # Rebuild & restart
```

**Docker Services:**

| Service | Container | Port | Mô tả |
|---------|-----------|------|--------|
| `redis` | tryhair_redis | 6379 | Message broker cho Celery |
| `backend` | tryhair_backend | 8000 | FastAPI server |
| `celery-worker` | tryhair_celery | — | AI model inference (GPU) |
| `frontend` | tryhair_frontend | 3000 | Web UI (Nginx + React build) |

---

### 🖥️ Cách 2: WSL Development (Phát triển)

> Mở **2 terminal** riêng biệt.

**Terminal 1 — Backend (Redis + FastAPI + Celery):**
```bash
cd /mnt/c/Users/<User>/Desktop/TryHairStyle
source venv_wsl/bin/activate
bash start.sh
```
Script `start.sh` tự động khởi động: Redis → FastAPI (port 8000) → Celery Worker.

**Terminal 2 — Frontend:**
```bash
cd /mnt/c/Users/<User>/Desktop/TryHairStyle/frontend
npm install   # lần đầu
npm run dev
```
Frontend chạy tại: `http://localhost:5173`

---

### ⚡ Cách 3: Windows PowerShell (Chỉ Frontend)

> Backend phải chạy trên WSL (Cách 2, Terminal 1).

```powershell
cd C:\Users\<User>\Desktop\TryHairStyle\frontend
npm install
npm run dev
```

---

## 4. Tải Models

### Tự động (Khuyên dùng)

```bash
source venv_wsl/bin/activate
python download_models.py
```

Script tự tải: ControlNet Depth, InstantID, IP-Adapter, SegFormer Face Parsing, CLIP, YOLOv8-Face, AdaFace, 3DDFA V2.

### Thủ công

| Model | Mục đích | Lệnh tải |
|-------|----------|----------|
| **SDXL Inpainting** | Base model | `hf download diffusers/stable-diffusion-xl-1.0-inpainting-0.1` |
| **ControlNet Depth** | Giữ cấu trúc khuôn mặt | `hf download diffusers/controlnet-depth-sdxl-1.0` |
| **IP-Adapter FaceID** | Giữ identity khuôn mặt | `hf download h94/IP-Adapter-FaceID` |
| **IP-Adapter Plus** | Copy kiểu tóc | `hf download h94/IP-Adapter` |
| **SegFormer** | Tách mask face/hair | `hf download jonathandinu/face-parsing` |
| **3DDFA V2** | 3D face alignment | `git clone https://github.com/cleardusk/3DDFA_V2.git` |

> **Lưu ý:** Cần đăng nhập HuggingFace CLI trước: `pip install huggingface_hub && huggingface-cli login`

---

## 5. Cấu trúc Thư mục

```
TryHairStyle/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI endpoints
│   │   ├── tasks.py             # Celery tasks (generate, detect, colorize)
│   │   ├── config.py            # Cấu hình paths, thông số
│   │   ├── schemas.py           # Pydantic data models
│   │   ├── services/
│   │   │   ├── face.py          # Face detection & alignment (InsightFace)
│   │   │   ├── mask.py          # Face/Hair parsing (SegFormer)
│   │   │   ├── diffusion.py     # Hair transfer pipeline (SDXL + ControlNet)
│   │   │   ├── hair_color_service.py  # Hair colorization
│   │   │   ├── embedder.py      # Hair texture embedding
│   │   │   ├── face_detector.py # YOLOv8 face detector
│   │   │   └── pose_estimator.py # 3D pose estimation
│   │   └── utils/
│   │       └── torch_patch.py   # Compatibility patches (PyTorch ↔ HuggingFace)
│   ├── models/                  # AI model weights (không commit)
│   ├── data/                    # Upload & output images
│   ├── tests/
│   │   ├── test_cli_ffhq.py     # CLI test với FFHQ
│   │   └── test_ui_gradio.py    # Gradio UI test (debug nhanh)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .dockerignore
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main app component
│   │   ├── api/hairApi.js       # API client
│   │   ├── components/
│   │   │   ├── ImageUpload.jsx  # Upload & preview ảnh
│   │   │   ├── DrawButton.jsx   # Nút VẼ TÓC
│   │   │   ├── ResultPanel.jsx  # Hiển thị kết quả
│   │   │   ├── FaceSelector.jsx # Popup chọn mặt (multi-face)
│   │   │   ├── ColorPicker.jsx  # Chọn màu tóc
│   │   │   ├── PromptInput.jsx  # Text prompt
│   │   │   └── Header.jsx
│   │   └── index.css            # Global styles
│   ├── Dockerfile               # Multi-stage build (Node → Nginx)
│   ├── nginx.conf               # Nginx config (SPA + API proxy)
│   ├── vite.config.js
│   └── package.json
├── docker-compose.yml           # Orchestrate tất cả services
├── start.sh                     # Script khởi động WSL (dev)
├── download_models.py           # Tải models tự động
├── .env.example                 # Template biến môi trường
└── README.md
```

---

## 6. API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| `POST` | `/generate` | Gửi 2 ảnh + prompt → tạo kiểu tóc (trả task_id) |
| `POST` | `/detect-faces` | Phát hiện khuôn mặt trong ảnh (trả task_id) |
| `GET` | `/status/{task_id}` | Kiểm tra trạng thái task |
| `GET` | `/colors` | Danh sách preset màu tóc |
| `GET` | `/random-pair` | 2 ảnh FFHQ ngẫu nhiên |
| `GET` | `/docs` | Swagger UI |

**Flow xử lý:**
```
POST /generate → task_id → GET /status/{id} (polling) → result_url
```

---

## 7. Kiểm thử

### CLI Test
```bash
python backend/tests/test_cli_ffhq.py
```
Chạy pipeline với ảnh FFHQ ngẫu nhiên. Kết quả lưu tại `backend/output/`.

### Gradio UI Test (Debug nhanh)
```bash
python backend/tests/test_ui_gradio.py
```
Giao diện web debug tại `http://127.0.0.1:7862` — upload ảnh, chọn mặt, và chạy thử pipeline trực tiếp.

### Kiểm tra PyTorch & CUDA
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 8. Troubleshooting

### Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|---------|
| `_is_hf_initialized` | Xung đột PyTorch ↔ HuggingFace | Đã patch tự động trong `torch_patch.py` |
| `module 'torch' has no attribute 'xpu'` | accelerate mới không tương thích | Đã patch tự động |
| `NoneType ... image_projection_layers` | IP-Adapter load lỗi | Code tự bỏ qua, app vẫn chạy |
| `HalfTensor vs FloatTensor` | Lệch FP16/FP32 | Pipeline đã ép sang FP16 |
| `ImportError: MT5Tokenizer` | Xung đột transformers/diffusers | `pip install transformers==4.49.0` |

### Docker

```bash
# Xem logs service cụ thể
docker compose logs celery-worker -f

# Restart service
docker compose restart backend

# Rebuild hoàn toàn
docker compose down && docker compose up --build -d

# Kiểm tra GPU trong container
docker compose exec celery-worker nvidia-smi
```

### Dataset

- **FFHQ** (khuôn mặt): https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL
- **K-Hairstyle** (kiểu tóc): https://psh01087.github.io/K-Hairstyle/

---

## 📄 License

Dự án phục vụ mục đích học tập và nghiên cứu.