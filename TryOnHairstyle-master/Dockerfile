# ============================================================
# HairFusion — Dockerfile
# Base: NVIDIA CUDA 11.7 + Python 3.8
# ============================================================
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-venv python3.8-dev python3-pip \
    cmake build-essential git wget \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.8 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Working directory
WORKDIR /app

# Install PyTorch first (cached layer)
RUN pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 \
    --index-url https://download.pytorch.org/whl/cu117

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose Gradio port
EXPOSE 7860

# Environment
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860')" || exit 1

# Run
CMD ["python", "-m", "backend.app.main"]
