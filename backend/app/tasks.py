
import os
import cv2
import numpy as np
from PIL import Image
from celery import Celery
from backend.app.config import settings, OUTPUT_DIR
from backend.app.services.face import FaceInfoService
from backend.app.services.mask import SegmentationService
from backend.app.services.diffusion import HairDiffusionService # Heavy load

# Initialize Celery
# Note: In production, backend should be Redis/Database to store results. 
# Here using Redis for broker and backend.
celery_app = Celery(
    "hair_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

# Global Service Cache (Lazy Load)
_SERVICES = {
    "face": None,
    "mask": None,
    "diffusion": None,
    "loaded": False
}

def get_services():
    """Lazy load services to ensure they run in the worker process (safe for CUDA/Celery)"""
    if _SERVICES["loaded"]:
        return _SERVICES["face"], _SERVICES["mask"], _SERVICES["diffusion"]
        
    print(">>> Initializing AI Models in Worker Process...")
    try:
        if not _SERVICES["face"]:
             _SERVICES["face"] = FaceInfoService()
        if not _SERVICES["mask"]:
             _SERVICES["mask"] = SegmentationService()
        if not _SERVICES["diffusion"]:
             _SERVICES["diffusion"] = HairDiffusionService()
             
        _SERVICES["loaded"] = True
        print(">>> AI Models Loaded Successfully!")
        return _SERVICES["face"], _SERVICES["mask"], _SERVICES["diffusion"]
    except Exception as e:
        print(f"Critical Error loading models: {e}")
        import traceback
        traceback.print_exc()
        raise e

@celery_app.task(bind=True)
def process_hair_transfer(self, user_img_path: str, hair_img_path: str, prompt: str, use_refiner: bool = False):
    try:
        face_service, mask_service, diffusion_service = get_services()
    except Exception as e:
        return {"status": "FAILURE", "error": f"Model Load Failed: {str(e)}"}

    try:
        # Update state
        self.update_state(state='PROCESSING', meta={'step': 'Loading Images'})
        
        # 1. Load Images
        user_cv2 = cv2.imread(user_img_path)
        user_pil = Image.fromarray(cv2.cvtColor(user_cv2, cv2.COLOR_BGR2RGB))
        hair_pil = Image.open(hair_img_path).convert("RGB")
        
        # 2. Face Analysis
        self.update_state(state='PROCESSING', meta={'step': 'Face Analysis'})
        
        # Auto-rotate logic: Try 0, 90, 180, 270 degrees
        face_info = None
        for angle in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            if angle is not None:
                print(f"No face found. Rotating image {angle}...")
                user_cv2 = cv2.rotate(user_cv2, angle)
                # Update user_pil as well to match user_cv2 for later steps
                user_pil = Image.fromarray(cv2.cvtColor(user_cv2, cv2.COLOR_BGR2RGB))
            
            face_info = face_service.analyze(user_cv2)
            if face_info:
                print("Face detected successfully.")
                break
        
        if not face_info:
            return {"status": "FAILURE", "error": "No face detected in user image (tried multiple orientations)"}
        
        # 3. Segmentation (Create Mask)
        self.update_state(state='PROCESSING', meta={'step': 'Creating Hair Mask'})
        # Mask tóc từ biSeNet
        hair_mask = mask_service.get_mask(user_pil, target_class=17) # 17 is hair
        
        # 4. Depth Map (ControlNet Input)
        # Simple depth estimation (or use helper if available, for now using simple grayscale as placeholder or dedicated DepthEstimator)
        # Note: In real implementation, we should use Midas or similar. 
        # ControlNet requires a proper depth map.
        # Temp solution: Use ImageOps.invert(grayscale) if real depth not avail, OR better: use transformers pipeline.
        # But to avoid adding more deps/files now, let's assume we pass the User Image itself if Preprocessor is integrated in Pipe?
        # Standard ControlNet expects pre-processed depth map.
        # Let's create a dummy depth or better: assume user provides good lighting.
        # Actually, let's skip Depth estimation integration code to keep it simple, passing the GS image implies Canny/Depth depending on model.
        # ControlNet Depth expects normalized depth.
        # Important: For quality, we SHOULD run depth estimation.
        # Since 'transformers' is installed, let's use it quickly here inside task or service.
        from transformers import pipeline
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large") # Will download if not localized
        
        self.update_state(state='PROCESSING', meta={'step': 'Estimating Depth'})
        depth_map = depth_estimator(user_pil)['depth']
        
        # 5. Diffusion Inference
        self.update_state(state='PROCESSING', meta={'step': 'Generating Hair'})
        result_image = diffusion_service.generate(
            base_image=user_pil,
            mask_image=hair_mask,
            control_image=depth_map,
            ref_hair_image=hair_pil,
            prompt=prompt,
            use_refiner=use_refiner
        )
        
        # 6. Save Output
        filename = f"result_{self.request.id}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        result_image.save(save_path)
        
        return {
            "status": "SUCCESS", 
            "result_path": str(save_path),
            "url": f"/static/output/{filename}"
        }

    except Exception as e:
        return {"status": "FAILURE", "error": str(e)}
