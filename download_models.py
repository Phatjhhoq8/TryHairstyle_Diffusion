
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

# Configuration
MODELS_DIR = Path("backend/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Define models to download
MODELS = {
    # 1. SDXL Base (Optional if using API, but good to have)
    # "sdxl": {
    #     "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
    #     "type": "model",
    #     "path": MODELS_DIR / "sdxl"
    # },
    
    # 2. ControlNet Depth (SDXL) - Critical for Hair Shape
    "controlnet_depth": {
        "repo_id": "diffusers/controlnet-depth-sdxl-1.0",
        "type": "model",
        "path": MODELS_DIR / "controlnet_depth"
    },
    
    # 3. Image Encoder (CLIP) - for IP-Adapter (Must be bigG for SDXL IP-Adapter)
    "image_encoder": {
        "repo_id": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "type": "model",
        "path": MODELS_DIR / "image_encoder"
    },
    
    # 4. IP-Adapter FaceID/Plus - for Identity & Hair
    "ip_adapter_faceid": {
        "repo_id": "h94/IP-Adapter-FaceID",
        "type": "file",
        "filename": "ip-adapter-faceid_sdxl.bin",
        "path": MODELS_DIR / "ip_adapter_faceid"
    },
    
    "ip_adapter_plus": {
        "repo_id": "h94/IP-Adapter", 
        "type": "file",
        # We need the SDXL Plus ViT-H version
        "filename": "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
        "path": MODELS_DIR / "ip_adapter_hair"
    },

    # 5. InstantID (includes its own ControlNet & IP-Adapter)
    "instantid": {
        "repo_id": "InstantX/InstantID",
        "type": "model",
        "path": MODELS_DIR / "instantid"
    },
    
    # 6. Face Parsing (BiSeNet) - Custom links usually, but we can try HF mirror
    "face_parser": {
        "repo_id": "vivym/face-parsing-bisenet", 
        "type": "file",
        "filename": "79999_iter.pth",
        "path": MODELS_DIR / "bisenet"
    }
}

def install_libraries():
    print("Installing necessary libraries for model download...")
    os.system("pip install huggingface_hub tqdm")

def download_file(repo_id, filename, target_dir):
    print(f"\nDownloading {filename} from {repo_id}...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        print("✅ Done.")
    except Exception as e:
        print(f"❌ Failed: {e}")

def download_model_repo(repo_id, target_dir, allow_patterns=None):
    print(f"\nDownloading full repo {repo_id}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.safetensors.index.json", "*.h5", "*.ot"],
            allow_patterns=allow_patterns
        )
        print("✅ Done.")
    except Exception as e:
        print(f"❌ Failed: {e}")

def main():
    install_libraries()
    
    print("="*50)
    print("STARTING MODEL DOWNLOAD MANAGER")
    print(f"Target Directory: {MODELS_DIR.absolute()}")
    print("="*50)

    for key, config in MODELS.items():
        target_path = config["path"]
        
        # specific manual fix for ip-adapter hair structure compatibility
        if key == "ip_adapter_plus":
             # We want to place it so code can find 'sdxl_models/ip-adapter_sdxl.bin' inside 'backend/models/ip_adapter_hair'
             # So we download to 'backend/models/ip_adapter_hair'
             pass

        if config["type"] == "file":
            download_file(config["repo_id"], config["filename"], target_path)
        elif config["type"] == "model":
            allow_patterns = None
            if key == "image_encoder":
                 # Only download essential files for the huge clip repo
                 allow_patterns = ["*.json", "*.bin", "*.safetensors", "*.txt"]
            
            download_model_repo(config["repo_id"], target_path, allow_patterns=allow_patterns)
            
    print("\n\n🎉 ALL DOWNLOADS COMPLETED!")
    print("Verify your 'backend/models' folder structure.")

if __name__ == "__main__":
    main()
