
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

# Global Service Instances (Lazy Loading recommended usually, but initializing here for simplicity in worker)
# Worker process will load these ONCE when starting.
print("Loading AI Models...")
try:
    face_service = FaceInfoService()
    mask_service = SegmentationService()
    diffusion_service = HairDiffusionService()
    print("AI Models Loaded Successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    # In production, we might want to fail fast or handle this gracefully
    face_service = None

@celery_app.task(bind=True)
def process_hair_transfer(self, user_img_path: str, hair_img_path: str, prompt: str):
    if not face_service:
        return {"status": "FAILURE", "error": "Models not loaded properly"}

    try:
        # Update state
        self.update_state(state='PROCESSING', meta={'step': 'Loading Images'})
        
        # 1. Load Images
        user_cv2 = cv2.imread(user_img_path)
        user_pil = Image.fromarray(cv2.cvtColor(user_cv2, cv2.COLOR_BGR2RGB))
        hair_pil = Image.open(hair_img_path).convert("RGB")
        
        # 2. Face Analysis
        self.update_state(state='PROCESSING', meta={'step': 'Face Analysis'})
        face_info = face_service.analyze(user_cv2)
        if not face_info:
            return {"status": "FAILURE", "error": "No face detected in user image"}
        
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
            prompt=prompt
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
