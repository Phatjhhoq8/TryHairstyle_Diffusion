# Backend Dockerfile - TryHairStyle
# CUDA 12.1 + Python 3.10 cho AI inference với GPU

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Tránh interactive prompts khi cài đặt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Cài đặt Python và dependencies hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy requirements trước để tận dụng Docker cache
COPY requirements.txt ./backend/

# Cài đặt PyTorch với CUDA support trước
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir xformers

# Cài đặt các dependencies còn lại
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy toàn bộ source code vào backend/
COPY . ./backend/

# Tạo thư mục data nếu chưa có
RUN mkdir -p /workspace/backend/data/uploads /workspace/backend/data/output

# Set PYTHONPATH để import được module backend
ENV PYTHONPATH=/workspace

# Expose port cho FastAPI
EXPOSE 8000

# Default command: chạy FastAPI server
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
