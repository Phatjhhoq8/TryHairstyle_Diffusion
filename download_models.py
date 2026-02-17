
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

# Configuration
MODELS_DIR = Path("backend/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Define models to download
MODELS = {


    # --- SDXL (Advanced/Future Use) ---
    "sdxl_base": {
        "repo_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "type": "model",
        "path": MODELS_DIR / "stable-diffusion" / "sd_xl_inpainting"
    },

    "sdxl_refiner": {
        "repo_id": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "type": "file",
        "filename": "sd_xl_refiner_1.0_0.9vae.safetensors",
        "path": MODELS_DIR
    },
    
    "controlnet_depth": {
        "repo_id": "diffusers/controlnet-depth-sdxl-1.0",
        "type": "model",
        "path": MODELS_DIR / "controlnet_depth"
    },
    
    "image_encoder": {
        "repo_id": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "type": "model",
        "path": MODELS_DIR / "image_encoder"
    },
    
    # --- Adapters ---
    "ip_adapter_plus_sdxl": {
        "repo_id": "h94/IP-Adapter", 
        "type": "file",
        "filename": "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
        "local_filename": "ip-adapter-plus_sdxl_vit-h.bin",
        "path": MODELS_DIR / "ip_adapter_hair"
    },

    "face_parser": {
        "repo_id": "vivym/face-parsing-bisenet", 
        "type": "file",
        "filename": "79999_iter.pth",
        "path": MODELS_DIR / "bisenet"
    },
    
    # --- Face Detection & Embedding (NEW) ---
    "yolo_face": {
        "repo_id": "arnabdhar/YOLOv8-Face-Detection",
        "type": "file",
        "filename": "model.pt",
        "local_filename": "yolov8n-face.pt",
        "path": MODELS_DIR
    },
    
    "adaface": {
        "type": "gdrive",  # Special type for Google Drive
        "gdrive_id": "1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI",
        "filename": "adaface_ir101_webface4m.ckpt",
        "path": MODELS_DIR,
        "size_mb": 250
    }
}

def install_libraries():
    print("Installing necessary libraries for model download...")
    os.system("pip install huggingface_hub tqdm")

def download_file(repo_id, filename, target_dir, local_filename=None):
    print(f"\nDownloading {filename} from {repo_id}...")
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        
        # Rename if requested (e.g. to flatten structure or simplify names)
        if local_filename:
            destination = target_dir / local_filename
            # If the download created a nested structure (e.g. models/...), iterate to move it
            # But hf_hub_download with local_dir preserves structure. 
            # Simplified: Use os.rename if needed, or rely on hf_hub handling.
            # To strictly rename:
            # Rename/Move logic:
            full_downloaded_path = Path(file_path).resolve()
            destination = (target_dir / local_filename).resolve()
            
            # If the downloaded file is not at the destination (e.g. it's in a subdir 'models/')
            if full_downloaded_path != destination:
                 if not destination.parent.exists():
                     destination.parent.mkdir(parents=True)
                 
                 import shutil
                 print(f"  -> Moving from {full_downloaded_path} to {destination}")
                 shutil.copy(full_downloaded_path, destination)

        print("✅ Done.")
    except Exception as e:
        print(f"❌ Failed: {e}")

def download_model_repo(repo_id, target_dir, allow_patterns=None, ignore_patterns=None):
    print(f"\nDownloading repo {repo_id}...")
    try:
        # Defaults
        if not ignore_patterns:
            ignore_patterns = ["*.msgpack", "*.h5", "*.ot", "*.ckpt"] # Prefer Safetensors
            
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=ignore_patterns,
            allow_patterns=allow_patterns
        )
        print("✅ Done.")
    except Exception as e:
        print(f"❌ Failed: {e}")

def download_from_gdrive(gdrive_id, filename, target_dir):
    """Download file from Google Drive using gdown"""
    print(f"\nDownloading {filename} from Google Drive...")
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        os.system("pip install gdown")
        import gdown
    
    target_path = target_dir / filename
    if target_path.exists():
        print(f"✅ Already exists: {target_path}")
        return
    
    target_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    
    try:
        gdown.download(url, str(target_path), quiet=False)
        print("✅ Done.")
    except Exception as e:
        print(f"❌ Failed: {e}")

def main():
    install_libraries()
    
    print("="*50)
    print("TRYHAIRSTYLE MODEL DOWNLOAD MANAGER")
    print(f"Target Directory: {MODELS_DIR.absolute()}")
    print("="*50)

    for key, config in MODELS.items():
        target_path = config["path"]
        print(f"\n--- Processing: {key} ---")
        
        if config["type"] == "file":
             local_name = config.get("local_filename", None)
             download_file(config["repo_id"], config["filename"], target_path, local_name)
             
        elif config["type"] == "model":
            allow = config.get("allow_patterns", None)
            download_model_repo(config["repo_id"], target_path, allow_patterns=allow)
        
        elif config["type"] == "gdrive":
            download_from_gdrive(config["gdrive_id"], config["filename"], target_path)
            
    print("\n\n🎉 ALL DOWNLOADS COMPLETED!")
    print("Verify your 'backend/models' folder structure.")

if __name__ == "__main__":
    main()
