
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
MODELS_DIR = BACKEND_DIR / "models"
OUTPUT_DIR = BACKEND_DIR / "output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model Paths
class ModelPaths:
    # Stable Diffusion XL
    SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0" # Tải online hoặc trỏ vào local nếu đã tải full
    # Nếu dùng file safetensors đơn lẻ thì code load phải khác, ở đây giả định dùng diffusers repo structure
    # Hoặc trỏ vào folder đã download script:
    # SDXL_BASE = str(MODELS_DIR / "stable-diffusion" / "sd_xl_base_1.0.safetensors") 
    
    # Ở đây ta sẽ dùng repo id HuggingFace vì trong script download trước đó 
    # nó có vẻ download dạng cache hoặc folder diffusers. 
    # Tuy nhiên để tối ưu tốc độ và offline, ta nên trỏ vào folder local nếu có.
    # Dựa trên `ls` trước đó: backend/models/sd_xl_refiner... (chỉ thấy refiner?)
    # Tạm thời ta set default là repo ID để diffusers tự handle cache.
    SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
    
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

    # Face Packing (cho InsightFace)
    ANTELOPEV2_PACK = str(MODELS_DIR / "antelopev2")

# App Settings
class Settings:
    HOST = "0.0.0.0"
    PORT = 8000
    REDIS_URL = "redis://localhost:6379/0"
    DEVICE = "cuda"
    
settings = Settings()
model_paths = ModelPaths()
