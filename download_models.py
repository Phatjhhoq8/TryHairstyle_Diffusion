
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

# Configuration
MODELS_DIR = Path("backend/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Define models to download
MODELS = {
    # --- SD1.5 (Primary Working Pipeline) ---
    "sd15_base": {
        "repo_id": "runwayml/stable-diffusion-v1-5",
        "type": "model",
        "path": MODELS_DIR / "stable-diffusion" / "sd15",
        # Download fp16 variants and critical config files for bandwidth efficiency
        "allow_patterns": ["*fp16.safetensors", "*.json", "*.txt", "tokenizer/*"] 
    },
    
    "ip_adapter_sd15": {
        "repo_id": "h94/IP-Adapter",
        "type": "file",
        "filename": "models/ip-adapter-plus_sd15.bin", # Correct path in repo
        "local_filename": "ip-adapter-plus_sd15.bin",   # Rename locally for clarity
        "path": MODELS_DIR / "ip_adapter_hair"
    },

    # --- SDXL (Advanced/Future Use) ---
    # "sdxl_base": {
    #     "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
    #     "type": "model",
    #     "path": MODELS_DIR / "stable-diffusion" / "sd_xl_base_1.0"
    # },
    
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
        "path": MODELS_DIR / "ip_adapter_hair"
    },

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
            
    print("\n\n🎉 ALL DOWNLOADS COMPLETED!")
    print("Verify your 'backend/models' folder structure.")

if __name__ == "__main__":
    main()
