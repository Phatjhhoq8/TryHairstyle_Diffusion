
import sys
import os
import random
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def main():
    print(">>> Starting Hair Transfer CLI Test (FFHQ Dataset) - WITH REFINER", flush=True)
    
    print(">>> Importing libraries... (This might take time)", flush=True)
    import cv2
    import torch
    import numpy as np
    from PIL import Image, ImageOps 
    
    print(">>> Importing backend services...", flush=True)
    from backend.app.config import settings, OUTPUT_DIR
    from backend.app.services.face import FaceInfoService
    from backend.app.services.mask import SegmentationService
    
    # Try importing Diffusion Service
    try:
        from backend.app.services.diffusion import HairDiffusionService
        DIFFUSION_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import HairDiffusionService ({e}).", flush=True)
        return

    print(">>> Imports completed.", flush=True)
    
    # 1. Setup Data Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_root = os.path.join(base_dir, "backend", "data", "dataset", "ffhq")
    output_dir = str(OUTPUT_DIR)
    output_path = os.path.join(output_dir, "refiner_test_result.png")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 2. Select Random Images
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset not found at {dataset_root}")
        return

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
    print("\n>>> Loading Models...", flush=True)
    try:
        face_service = FaceInfoService()
        mask_service = SegmentationService()
        diffusion_service = HairDiffusionService()
        print(">>> All Models Loaded Successfully!", flush=True)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 4. Run Pipeline with REFINER
    try:
        print("\n>>> Processing Pipeline (Refiner Enabled)...", flush=True)
        
        user_cv2 = cv2.imread(user_img_path)
        user_pil = Image.fromarray(cv2.cvtColor(user_cv2, cv2.COLOR_BGR2RGB))
        hair_pil = Image.open(hair_img_path).convert("RGB")
        
        # Analysis items...
        face_info = face_service.analyze(user_cv2)
        hair_mask = mask_service.get_mask(user_pil, target_class=17)
        depth_map = ImageOps.grayscale(user_pil) # Placeholder depth

        print("   - Running Hair Diffusion with use_refiner=True...", flush=True)
        
        start_time = time.time()
        result_image = diffusion_service.generate(
            base_image=user_pil,
            mask_image=hair_mask,
            control_image=depth_map,
            ref_hair_image=hair_pil,
            use_refiner=True  # <--- CRITICAL
        )
        end_time = time.time()
        print(f"   - Generation took: {end_time - start_time:.2f} seconds")
        
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
