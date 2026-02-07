
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
    # Stable Diffusion XL
    SDXL_BASE = str(MODELS_DIR / "stable-diffusion" / "sd_xl_inpainting")
    # SDXL_BASE = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" # Fallback repo id if local missing
    SDXL_REFINER = str(MODELS_DIR / "sd_xl_refiner_1.0_0.9vae.safetensors")

    # SDXL_REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    
    # SD1.5 Paths (Fallback)
    SD15_BASE = str(MODELS_DIR / "stable-diffusion" / "sd15")
    
    # Adapters (Local Paths from download_models.py structure)
    INSIGHTFACE_ROOT = str(BACKEND_DIR)
    BISENET_CHECKPOINT = str(MODELS_DIR / "bisenet" / "79999_iter.pth")
    
    # ControlNet (SDXL)
    CONTROLNET_DEPTH = str(MODELS_DIR / "controlnet_depth")
    
    # Adapters
    INSTANTID_ADAPTER = str(MODELS_DIR / "instantid" / "ip-adapter.bin")
    IP_ADAPTER_PLUS_HAIR = str(MODELS_DIR / "ip_adapter_hair") # Folder
    
    # Specific file for SD1.5 IP Adapter
    IP_ADAPTER_SD15_PATH = str(MODELS_DIR / "ip_adapter_hair" / "ip-adapter-plus_sd15.bin")
    # Specific file for SDXL IP Adapter
    IP_ADAPTER_SDXL_PATH = str(MODELS_DIR / "ip_adapter_hair" / "ip-adapter-plus_sdxl_vit-h.bin")
    IMAGE_ENCODER_PATH = str(MODELS_DIR / "image_encoder")

    # Face Packing (cho InsightFace)
    ANTELOPEV2_PACK = str(MODELS_DIR / "antelopev2")

# App Settings
class Settings:
    HOST = "0.0.0.0"
    PORT = 8000
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    DEVICE = "cuda"
    
settings = Settings()
model_paths = ModelPaths()
