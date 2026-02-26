
import os
import cv2
import torch
from PIL import Image
from datetime import datetime
from celery import Celery
from backend.app.config import settings, OUTPUT_DIR
from backend.app.services.face import FaceInfoService
from backend.app.services.mask import SegmentationService
from backend.app.services.diffusion import HairDiffusionService
from backend.app.services.hair_color_service import HairColorService

celery_app = Celery(
    "hair_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

_SERVICES = {
    "face": None,
    "mask": None,
    "diffusion": None,
    "depth": None,
    "color": None,
    "loaded": False
}

def get_services():
    """Lazy load services to ensure they run in the worker process (safe for CUDA/Celery)"""
    if _SERVICES["loaded"]:
        return _SERVICES["face"], _SERVICES["mask"], _SERVICES["diffusion"], _SERVICES["depth"], _SERVICES["color"]
        
    print(">>> Initializing AI Models in Worker Process...")
    try:
        if not _SERVICES["face"]:
             _SERVICES["face"] = FaceInfoService()
        if not _SERVICES["mask"]:
             _SERVICES["mask"] = SegmentationService()
        if not _SERVICES["diffusion"]:
             _SERVICES["diffusion"] = HairDiffusionService()
        if not _SERVICES["depth"]:
             from transformers import pipeline
             _SERVICES["depth"] = pipeline("depth-estimation", model="Intel/dpt-large")
             print(">>> Depth Estimator loaded (Intel/dpt-large)")
        if not _SERVICES["color"]:
             _SERVICES["color"] = HairColorService()
             print(">>> HairColorService loaded")
             
        _SERVICES["loaded"] = True
        print(">>> AI Models Loaded Successfully!")
        return _SERVICES["face"], _SERVICES["mask"], _SERVICES["diffusion"], _SERVICES["depth"], _SERVICES["color"]
    except Exception as e:
        print(f"Critical Error loading models: {e}")
        import traceback
        traceback.print_exc()
        raise e

@celery_app.task(bind=True)
def process_hair_transfer(self, user_img_path: str, hair_img_path: str, prompt: str, hair_color: str = None, color_intensity: float = 0.7):
    try:
        face_service, mask_service, diffusion_service, depth_estimator, color_service = get_services()
    except Exception as e:
        return {"status": "FAILURE", "error": f"Model Load Failed: {str(e)}"}

    try:
        # Tạo session folder cho lần inference này
        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.request.id[:8]}"
        session_dir = os.path.join(str(OUTPUT_DIR), session_name)
        os.makedirs(session_dir, exist_ok=True)
        print(f">>> Session folder: {session_dir}")
        
        # Update state
        self.update_state(state='PROCESSING', meta={'step': 'Loading Images'})
        
        # 1. Load Images
        user_cv2 = cv2.imread(user_img_path)
        if user_cv2 is None:
            return {"status": "FAILURE", "error": f"Cannot read user image: {user_img_path}"}
        user_pil = Image.fromarray(cv2.cvtColor(user_cv2, cv2.COLOR_BGR2RGB))
        hair_pil = Image.open(hair_img_path).convert("RGB")
        
        # 2. Face Analysis (phát hiện TẤT CẢ khuôn mặt)
        self.update_state(state='PROCESSING', meta={'step': 'Face Analysis'})
        
        # Auto-rotate logic: Try 0, 90, 180, 270 degrees
        # Lưu ảnh gốc để xoay từ đó (tránh xoay tích lũy)
        original_cv2 = user_cv2.copy()
        faces = []
        for angle in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            if angle is not None:
                print(f"No face found. Rotating image {angle}...")
                user_cv2 = cv2.rotate(original_cv2, angle)
                user_pil = Image.fromarray(cv2.cvtColor(user_cv2, cv2.COLOR_BGR2RGB))
            
            faces = face_service.analyze_all(user_cv2)
            if faces:
                print(f"Detected {len(faces)} face(s) successfully.")
                break
        
        if not faces:
            return {"status": "FAILURE", "error": "No face detected in user image (tried multiple orientations)"}
        
        # 3. Segmentation (Create Mask — lấy cả hair + face mask)
        self.update_state(state='PROCESSING', meta={'step': 'Creating Hair & Face Mask'})
        masks = mask_service.get_hair_and_face_mask(user_pil)
        hair_mask = masks["hair_mask"]
        face_mask = masks["face_mask"]
        
        # Mask tóc từ ảnh reference
        ref_masks = mask_service.get_hair_and_face_mask(hair_pil)
        ref_hair_mask = ref_masks["hair_mask"]
        
        # Lưu mask trung gian vào session folder
        hair_mask.save(os.path.join(session_dir, "hair_mask.png"))
        face_mask.save(os.path.join(session_dir, "face_mask.png"))
        ref_hair_mask.save(os.path.join(session_dir, "ref_hair_mask.png"))
        print(f"  ✅ Saved hair_mask, face_mask, ref_hair_mask → {session_dir}")
        
        # 4. Depth Map (ControlNet Input)
        # Dùng depth estimator đã cache trong _SERVICES (tránh reload mỗi task)
        self.update_state(state='PROCESSING', meta={'step': 'Estimating Depth'})
        depth_map = depth_estimator(user_pil)['depth']
        depth_map.save(os.path.join(session_dir, "depth_map.png"))
        print(f"  ✅ Saved depth_map → {session_dir}")
        
        # 5. Diffusion Inference
        self.update_state(state='PROCESSING', meta={'step': 'Generating Hair'})
        
        # Nếu có hair_color → thêm mô tả màu vào prompt
        finalPrompt = prompt
        if hair_color:
            from backend.app.services.hair_color_service import PRESET_COLORS
            colorLabel = hair_color
            if hair_color.lower() in PRESET_COLORS:
                colorLabel = PRESET_COLORS[hair_color.lower()]["label"]
            finalPrompt = f"{colorLabel} colored hair, {prompt}"
            print(f"  🎨 Hair color requested: {hair_color} → prompt: '{finalPrompt}'")
        
        original_size = user_pil.size  # (w, h) — lưu kích thước gốc
        result_image = diffusion_service.generate(
            base_image=user_pil,
            mask_image=hair_mask,
            control_image=depth_map,
            ref_hair_image=hair_pil,
            prompt=finalPrompt
        )
        
        # Resize kết quả về kích thước gốc để không bị méo
        if result_image.size != original_size:
            result_image = result_image.resize(original_size, Image.LANCZOS)
        
        # 5b. Post-process: đổi màu tóc nếu có yêu cầu
        if hair_color:
            self.update_state(state='PROCESSING', meta={'step': 'Applying Hair Color'})
            print(f"  🎨 Applying hair color: {hair_color} (intensity: {color_intensity})")
            # Tạo mask từ ảnh kết quả (vì kiểu tóc đã thay đổi)
            resultMasks = mask_service.get_hair_and_face_mask(result_image)
            resultHairMask = resultMasks["hair_mask"]
            result_image = color_service.colorize(
                result_image, resultHairMask, hair_color, color_intensity
            )
            print(f"  ✅ Hair color applied successfully")
        
        # 6. Save Output vào session folder
        save_path = os.path.join(session_dir, "result.png")
        result_image.save(save_path)
        print(f"  ✅ Saved result ({original_size}) → {session_dir}")
        
        # Cleanup GPU memory sau inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "status": "SUCCESS", 
            "result_path": str(save_path),
            "session_dir": str(session_dir),
            "url": f"/static/output/{session_name}/result.png"
        }

    except Exception as e:
        # Cleanup GPU memory ngay cả khi lỗi
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "FAILURE", "error": str(e)}


@celery_app.task(bind=True)
def process_hair_colorize(self, face_img_path: str, hair_color: str, intensity: float = 0.7):
    """
    Task chỉ đổi màu tóc (KHÔNG thay kiểu tóc).
    Nhanh hơn process_hair_transfer vì không cần Diffusion Model.
    """
    try:
        # Chỉ cần mask service và color service
        if not _SERVICES["mask"]:
            _SERVICES["mask"] = SegmentationService()
        if not _SERVICES["color"]:
            _SERVICES["color"] = HairColorService()
        mask_service = _SERVICES["mask"]
        color_service = _SERVICES["color"]
    except Exception as e:
        return {"status": "FAILURE", "error": f"Service Load Failed: {str(e)}"}

    try:
        # Tạo session folder
        session_name = f"color_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.request.id[:8]}"
        session_dir = os.path.join(str(OUTPUT_DIR), session_name)
        os.makedirs(session_dir, exist_ok=True)
        print(f">>> Color session: {session_dir}")
        
        # 1. Load ảnh
        self.update_state(state='PROCESSING', meta={'step': 'Loading Image'})
        face_pil = Image.open(face_img_path).convert("RGB")
        
        # 2. Segmentation — tạo hair mask
        self.update_state(state='PROCESSING', meta={'step': 'Creating Hair Mask'})
        masks = mask_service.get_hair_and_face_mask(face_pil)
        hair_mask = masks["hair_mask"]
        hair_mask.save(os.path.join(session_dir, "hair_mask.png"))
        
        # 3. Colorize
        self.update_state(state='PROCESSING', meta={'step': f'Applying Color: {hair_color}'})
        print(f"  🎨 Colorizing: {hair_color} (intensity: {intensity})")
        result_image = color_service.colorize(face_pil, hair_mask, hair_color, intensity)
        
        # 4. Save output
        save_path = os.path.join(session_dir, "result.png")
        result_image.save(save_path)
        print(f"  ✅ Color result saved → {save_path}")
        
        return {
            "status": "SUCCESS",
            "result_path": str(save_path),
            "session_dir": str(session_dir),
            "url": f"/static/output/{session_name}/result.png"
        }

    except Exception as e:
        return {"status": "FAILURE", "error": str(e)}
