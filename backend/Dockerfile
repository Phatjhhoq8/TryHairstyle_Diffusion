FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    build-essential \
    git \
    curl \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install torchvision==0.19.1 torchaudio==2.4.1 && \
    grep -Ev '^(torch|torchvision|torchaudio)([<>=].*)?$' /tmp/requirements.txt > /tmp/requirements-docker.txt && \
    python -m pip install --use-deprecated=legacy-resolver -r /tmp/requirements-docker.txt && \
    python -m pip install \
        "typing_extensions==4.15.0" \
        "click==8.3.1" \
        "diffusers==0.36.0" \
        "transformers==5.1.0" \
        "accelerate==1.12.0" \
        "huggingface_hub==1.4.1" \
        "xformers==0.0.28.post1" \
        "ultralytics==8.4.12" \
        "onnx==1.20.1" \
        "onnxruntime==1.23.2"

COPY backend/ /app/backend/

RUN mkdir -p /app/backend/data/output /app/backend/data/uploads /app/backend/models

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
