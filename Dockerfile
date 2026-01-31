# TryHairStyle Training Environment
# Base: PyTorch with CUDA 12.1

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install training-specific dependencies
RUN pip install --no-cache-dir \
    accelerate>=0.25.0 \
    transformers>=4.36.0 \
    diffusers>=0.25.0 \
    peft>=0.7.0 \
    bitsandbytes>=0.41.0 \
    wandb \
    insightface \
    onnxruntime-gpu

# Copy project files
COPY . /app

# Set environment variables
ENV PYTHONPATH=/app
# GPU sẽ được tự động phát hiện bởi NVIDIA Container Toolkit

# Default command
CMD ["python", "backend/training/train_ip_adapter.py"]
