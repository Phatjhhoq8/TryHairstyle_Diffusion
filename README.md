# TryHairStyle

Ứng dụng thử kiểu tóc ảo bằng AI sử dụng **Stable Diffusion XL Inpainting**, **ControlNet**, **IP-Adapter**, **InstantID** và **SegFormer**.

> [!NOTE]
> Repo này **không** commit virtual environment, model weights, dữ liệu sinh ra và `.env`.
> Sau khi clone, bạn cần tự tạo lại môi trường và tải model (xem hướng dẫn bên dưới).

---

## Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Cài đặt & Chạy Local](#2-cài-đặt--chạy-local)
3. [Chạy bằng Docker](#3-chạy-bằng-docker)
4. [Cấu trúc thư mục](#4-cấu-trúc-thư-mục)
5. [API chính](#5-api-chính)
6. [Kiểm thử & Debug](#6-kiểm-thử--debug)
7. [Troubleshooting](#7-troubleshooting)
8. [Dataset tham khảo](#8-dataset-tham-khảo)
9. [License](#9-license)

---

## 1. Yêu cầu hệ thống

### Phần cứng

| Thành phần | Yêu cầu |
|---|---|
| GPU | NVIDIA ≥ 12 GB VRAM (khuyến nghị) |
| RAM | ≥ 16 GB |
| Ổ đĩa | ≥ 20 GB trống |

### Phần mềm

| Phần mềm | Ghi chú |
|---|---|
| Python 3.10 / 3.11 | Backend |
| Node.js 18+ | Frontend |
| Redis | Message broker cho Celery |
| CUDA 12.1 | Nếu chạy local với GPU |
| WSL2 | Nếu dùng Windows cho backend |
| Docker Desktop + Compose v2 | Nếu chạy bằng container |
| NVIDIA Container Toolkit | Nếu dùng Docker với GPU |

---

## 2. Cài đặt & Chạy Local

### Bước 1 – Clone & cấu hình

```bash
git clone https://github.com/Phatjhhoq8/TryHairstyle_Diffusion
cd TryHairStyle
cp .env.example .env          # Điền HUGFACE_TOKEN vào file .env
```

### Bước 2 – Tạo virtual environment

**WSL / Linux:**

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

**PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### Bước 3 – Cài dependencies

```bash
pip install -r backend/requirements.txt   # Backend (PyTorch CUDA 12.1)
cd frontend && npm install && cd ..       # Frontend
```

### Bước 4 – Tải model

```bash
mkdir -p backend/models backend/data      # PowerShell: mkdir backend\models; mkdir backend\data
python download_models.py                 # Model sẽ lưu vào backend/models/
```

### Bước 5 – Khởi động dịch vụ

Mở **3 terminal** riêng biệt:

```bash
# Terminal 1 – Redis
redis-server --daemonize yes
redis-cli ping                            # Kết quả PONG = OK
```

```bash
# Terminal 2 – FastAPI + Celery
source .venv/bin/activate
export PYTHONPATH=$(pwd)
export REDIS_URL=redis://localhost:6379/0
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

```bash
# Terminal 3 – Celery worker (terminal mới)
source .venv/bin/activate
export PYTHONPATH=$(pwd)
export REDIS_URL=redis://localhost:6379/0
celery -A backend.app.tasks.celery_app worker --loglevel=info --pool=solo
```

```bash
# Terminal 4 – Frontend
cd frontend && npm run dev
```

### Địa chỉ truy cập

| Dịch vụ | URL |
|---|---|
| Frontend | `http://localhost:5173` |
| Backend API | `http://localhost:8000` |
| Swagger UI | `http://localhost:8000/docs` |

> [!TIP]
> **Windows:** Khuyến nghị chạy backend (Redis + FastAPI + Celery) trong WSL và frontend trong PowerShell.

---

## 3. Chạy bằng Docker

> [!IMPORTANT]
> Cần tải model **trước** khi build Docker vì `docker-compose.yml` mount thư mục `./backend/models` vào container.

```bash
# 1. Clone & cấu hình (nếu chưa)
git clone https://github.com/Phatjhhoq8/TryHairstyle_Diffusion
cd TryHairStyle
cp .env.example .env

# 2. Tải model trên host
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
mkdir -p backend/models backend/data
python download_models.py

# 3. Build & chạy
docker compose up --build -d

# 4. Kiểm tra
docker compose ps
docker compose logs -f backend
```

| Dịch vụ | URL |
|---|---|
| Frontend | `http://localhost:3000` |
| Swagger UI | `http://localhost:8000/docs` |

---

## 4. Cấu trúc thư mục

```
TryHairStyle/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── tasks.py             # Celery tasks
│   │   ├── config.py            # Cấu hình
│   │   ├── schemas.py           # Pydantic schemas
│   │   ├── services/            # Business logic
│   │   └── utils/               # Utilities
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── models/                  # ⚠ Không có sau clone
│   └── data/                    # ⚠ Không có sau clone
├── frontend/
│   ├── src/
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml
├── download_models.py
├── start.sh
├── .env.example
└── README.md
```

---

## 5. API chính

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/generate` | Upload ảnh user + ảnh tham chiếu, trả `task_id` |
| `POST` | `/detect-faces` | Phát hiện khuôn mặt |
| `GET` | `/status/{task_id}` | Kiểm tra trạng thái xử lý |
| `GET` | `/colors` | Danh sách màu tóc preset |
| `GET` | `/random-pair` | Cặp ảnh mẫu ngẫu nhiên |
| `GET` | `/docs` | Swagger UI |

**Flow cơ bản:**

```
POST /generate → task_id → GET /status/{task_id} → result_url
```

---

## 6. Kiểm thử & Debug

```bash
# Kiểm tra PyTorch + CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# CLI test
python backend/tests/test_cli_ffhq.py

# Gradio debug UI
python backend/tests/test_ui_gradio.py
```

---

## 7. Troubleshooting

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| Thiếu model | Clone sạch, model bị gitignore | `python download_models.py` |
| Không có `.env` | `.env` không được commit | `cp .env.example .env` |
| Lỗi import / thiếu package | Chưa cài dependencies | Tạo `.venv` và `pip install -r backend/requirements.txt` |
| Task không xử lý | Chưa chạy Redis hoặc Celery | Khởi động Redis + Celery worker |
| Frontend không gọi được API | Backend chưa chạy / sai port | Kiểm tra `http://localhost:8000/docs` |
| Docker không sinh ảnh | `backend/models/` rỗng trên host | Tải model trước khi `docker compose up` |

---

## 8. Dataset tham khảo

- **FFHQ:** [Google Drive](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL)
- **K-Hairstyle:** [https://psh01087.github.io/K-Hairstyle/](https://psh01087.github.io/K-Hairstyle/)

---

## 9. License

Dự án phục vụ mục đích học tập và nghiên cứu.
