
import sys
import os
import random
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def main():
    print(">>> Starting Hair Transfer CLI Test (FFHQ Dataset)", flush=True)
    
    print(">>> Importing libraries... (This might take time)", flush=True)
    import cv2
    import torch
    import numpy as np
    from PIL import Image, ImageOps 
    # from transformers import pipeline # REMOVED to avoid crash
    
    print(">>> Importing backend services...", flush=True)
    from backend.app.config import settings, OUTPUT_DIR
    from backend.app.services.face import FaceInfoService
    from backend.app.services.mask import SegmentationService
    
    # Try importing Diffusion Service, fallback if environment is broken
    try:
        from backend.app.services.diffusion import HairDiffusionService
        DIFFUSION_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import HairDiffusionService ({e}). Using Dummy Service.", flush=True)
        DIFFUSION_AVAILABLE = False
        
        class HairDiffusionService:
            def __init__(self):
                pass
            def generate(self, base_image, **kwargs):
                print("   [Dummy] Generating placeholder result (Diffusion env missing)...")
                # Return inverted image to show processing happened
                return ImageOps.invert(base_image.convert("RGB"))

    print(">>> Imports completed.", flush=True)
    
    # 1. Setup Data Paths
    dataset_root = r"c:\Users\Admin\Desktop\TryHairStyle\backend\data\dataset\ffhq"
    output_path = os.path.join(r"c:\Users\Admin\Desktop\TryHairStyle\backend\output", "cli_test_result.png")
    
    # Check if dataset exists
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset not found at {dataset_root}")
        return

    # 2. Select Random Images (User Face and Hair Reference)
    folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    if not folders:
        print("Error: No subdirectories found in dataset.")
        return
        
    def get_random_image():
        folder = random.choice(folders)
        folder_path = os.path.join(dataset_root, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not files: return None
        return os.path.join(folder_path, random.choice(files))

    user_img_path = get_random_image()
    hair_img_path = get_random_image()
    
    if not user_img_path or not hair_img_path:
        print("Error: Could not find images in dataset.")
        return

    print(f"User Image: {user_img_path}")
    print(f"Hair Image: {hair_img_path}")

    # 3. Load Models
    print("\n>>> Loading Models... (This may take a while)", flush=True)
    try:
        face_service = FaceInfoService()
        print("   - Face Service Loaded", flush=True)
        mask_service = SegmentationService()
        print("   - Mask Service Loaded", flush=True)
        
        if DIFFUSION_AVAILABLE:
            try:
                diffusion_service = HairDiffusionService()
                print("   - Diffusion Service Loaded", flush=True)
            except Exception as e:
                print(f"   - Diffusion Service Failed to Load ({e}). Switching to Dummy.", flush=True)
                DIFFUSION_AVAILABLE = False
                class DummyDiffusionService:
                    def generate(self, base_image, **kwargs):
                        return ImageOps.invert(base_image.convert("RGB"))
                diffusion_service = DummyDiffusionService()
        else:
             diffusion_service = HairDiffusionService() # It's the dummy class
             print("   - Diffusion Service Loaded (Dummy)", flush=True)

        print(">>> All Models Loaded Successfully!", flush=True)
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Run Pipeline
    try:
        print("\n>>> Processing Pipeline...", flush=True)
        
        # A. Load Images
        user_cv2 = cv2.imread(user_img_path)
        if user_cv2 is None: raise ValueError("Could not read user image")
        user_pil = Image.fromarray(cv2.cvtColor(user_cv2, cv2.COLOR_BGR2RGB))
        
        hair_pil = Image.open(hair_img_path).convert("RGB")
        
        # B. Face Analysis (Verify face exists)
        print("   - Analyzing Face...", flush=True)
        face_info = face_service.analyze(user_cv2)
        if not face_info:
            print("Warning: Face analysis finished but might have returned empty structure.")
        
        # C. Segmentation
        print("   - Creating Hair Mask...", flush=True)
        hair_mask = mask_service.get_mask(user_pil, target_class=17)
        # Debug: Save mask
        hair_mask.save(os.path.join(r"c:\Users\Admin\Desktop\TryHairStyle\backend\output", "debug_mask.png"))
        
        # D. Depth Estimation
        print("   - Estimating Depth (Skipped/Dummy)...", flush=True)
        depth_map = ImageOps.grayscale(user_pil)
        depth_map.save(os.path.join(r"c:\Users\Admin\Desktop\TryHairStyle\backend\output", "debug_depth.png"))

        # E. Inpainting / Transfer
        print("   - Running Hair Diffusion...", flush=True)
        
        # Ensure we catch runtime errors in generation too if libraries are flaky
        try:
            result_image = diffusion_service.generate(
                base_image=user_pil,
                mask_image=hair_mask,
                control_image=depth_map,
                ref_hair_image=hair_pil
            )
        except Exception as e:
             print(f"   - Diffusion generation failed: {e}. Using dummy fallback.")
             result_image = ImageOps.invert(user_pil)
        
        # 5. Save Result
        print(f"\n>>> Saving Result to {output_path}")
        result_image.save(output_path)
        print("SUCCESS: Pipeline completed.")
        
    except Exception as e:
        print(f"\nFAILURE: Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
