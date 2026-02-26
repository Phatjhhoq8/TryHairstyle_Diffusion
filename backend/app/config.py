
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
MODELS_DIR = BACKEND_DIR / "models"
OUTPUT_DIR = BACKEND_DIR / "data" / "output"
UPLOAD_DIR = BACKEND_DIR / "data" / "uploads"

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Model Paths
class ModelPaths:
    # Stable Diffusion XL
    SDXL_BASE = str(MODELS_DIR / "stable-diffusion" / "sd_xl_inpainting")
    SDXL_REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"  # Fallback HuggingFace repo
    
    # Custom Training Checkpoints
    CUSTOM_INPAINTING_MODEL = str(BACKEND_DIR / "training" / "models" / "deep_hair_v1_best.safetensors")
    
    # Adapters (Local Paths from download_models.py structure)
    INSIGHTFACE_ROOT = str(BACKEND_DIR)
    
    # ControlNet (SDXL)
    CONTROLNET_DEPTH = str(MODELS_DIR / "controlnet_depth")
    
    # IP-Adapter SDXL
    IP_ADAPTER_PLUS_HAIR = str(MODELS_DIR / "ip_adapter_hair")  # Folder chứa weights
    IMAGE_ENCODER_PATH = str(MODELS_DIR / "image_encoder")

# App Settings
class Settings:
    HOST = "0.0.0.0"
    PORT = 8000
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    DEVICE = "cuda"
    
settings = Settings()
model_paths = ModelPaths()
