
import os
import sys
import subprocess
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
def process_hair_transfer(self, user_img_path: str, hair_img_path: str, prompt: str, hair_color: str = None, color_intensity: float = 0.7, ai_model: str = "HairFusion", original_face_path: str = None, bbox: list = None):
    # Biến lưu kết quả trả về chung
    result_data = None

    # === Routing: Nếu chọn TryOnHairstyle → chạy subprocess cách ly ===
    if ai_model == "TryOnHairstyle":
        result_data = _run_tryonhairstyle(self, user_img_path, hair_img_path)
    else:
        # === Mặc định: HairFusion pipeline ===
        try:
            face_service, mask_service, diffusion_service, depth_estimator, color_service = get_services()
        except Exception as e:
            return {"status": "FAILURE", "error": f"Model Load Failed: {str(e)}"}

    if ai_model != "TryOnHairstyle":
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
        
        # Dynamic Mask: mở rộng mask nếu tóc mẫu lớn hơn tóc user
        hair_mask = mask_service.expand_hair_mask(hair_mask, face_mask, ref_hair_mask, faces[0] if faces else None)
        hair_mask.save(os.path.join(session_dir, "hair_mask_expanded.png"))
        
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
            prompt=finalPrompt,
            latent_injection_weight=0.3
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
        
        result_data = {
            "status": "SUCCESS", 
            "result_path": str(save_path),
            "session_dir": str(session_dir),
            "url": f"/static/output/{session_name}/result.png"
        }

      except Exception as e:
        # Cleanup GPU memory ngay cả khi lỗi
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result_data = {"status": "FAILURE", "error": str(e)}

    # Paste back logic for ALL models
    if result_data.get("status") == "SUCCESS" and original_face_path and bbox:
        print(f">>> Bắt đầu ghép kết quả trả về ảnh gốc... bbox: {bbox}")
        try:
            orig_img = Image.open(original_face_path).convert("RGB")
            res_img = Image.open(result_data["result_path"]).convert("RGB")
            
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            res_resized = res_img.resize((w, h), Image.LANCZOS)
            orig_img.paste(res_resized, (x1, y1))
            
            final_path = result_data["result_path"].replace("result.png", "result_full.png")
            orig_img.save(final_path)
            print(f"  ✅ Đã ghép mặt và lưu thành công: {final_path}")
            
            result_data["result_path"] = str(final_path)
            result_data["url"] = result_data["url"].replace("result.png", "result_full.png")
        except Exception as e:
            print(f"Error pasting back: {str(e)}")
            import traceback
            traceback.print_exc()

    return result_data


def _run_tryonhairstyle(self, user_img_path: str, hair_img_path: str):
    """
    Chạy mô hình TryOnHairstyle qua subprocess (cách ly hoàn toàn).
    Tránh xung đột namespace backend.app và phiên bản thư viện.
    """
    try:
        session_name = f"tryon_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.request.id[:8]}"
        session_dir = os.path.join(str(OUTPUT_DIR), session_name)
        os.makedirs(session_dir, exist_ok=True)
        print(f">>> TryOnHairstyle session: {session_dir}")
        
        self.update_state(state='PROCESSING', meta={'step': 'Running TryOnHairstyle Model'})
        
        # Đường dẫn tới thư mục TryOnHairstyle-master
        # __file__ = .../TryHairStyle/backend/app/tasks.py → cần 3 lần dirname để lên TryHairStyle
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        tryon_dir = os.path.join(project_root, "TryOnHairstyle-master")
        run_script = os.path.join(tryon_dir, "run_custom.py")
        
        if not os.path.exists(run_script):
            return {"status": "FAILURE", "error": f"TryOnHairstyle script not found: {run_script}"}
        
        # Dùng Python từ venv riêng của TryOnHairstyle (tránh xung đột thư viện)
        tryon_python = os.path.join(tryon_dir, "hairfusion", "bin", "python")
        if not os.path.exists(tryon_python):
            return {"status": "FAILURE", "error": f"TryOnHairstyle venv not found. Run: cd TryOnHairstyle-master && python3 -m venv hairfusion && source hairfusion/bin/activate && pip install -r requirements.txt"}
        
        # Thiết lập LD_LIBRARY_PATH cho WSL và CUDA/cuDNN trong venv
        # Dùng shell=True để export trước khi chạy Python → dynamic linker nhận biến RIGHT AWAY
        site_packages = os.path.join(tryon_dir, "hairfusion", "lib", "python3.8", "site-packages")
        torch_lib = os.path.join(site_packages, "torch", "lib")
        vision_lib = os.path.join(site_packages, "torchvision.libs")
        
        ld_path = f"/usr/lib/wsl/lib:{torch_lib}:{vision_lib}"
        
        # Tạo shell command với export LD_LIBRARY_PATH trước khi gọi Python
        shell_cmd = (
            f'export LD_LIBRARY_PATH="{ld_path}:${{LD_LIBRARY_PATH:-}}" && '
            f'"{tryon_python}" "{run_script}" '
            f'--face "{user_img_path}" '
            f'--hair "{hair_img_path}" '
            f'--output "{session_dir}"'
        )
        
        print(f"  Shell command: {shell_cmd}")
        
        # Chạy subprocess cách ly hoàn toàn qua bash shell
        result = subprocess.run(
            shell_cmd,
            shell=True,
            executable="/bin/bash",
            cwd=tryon_dir,
            capture_output=True,
            text=True,
            timeout=900  # Timeout 15 phút (model nặng, cần thời gian load + inference)
        )
        
        print(f"  TryOnHairstyle stdout: {result.stdout[-500:] if result.stdout else '(empty)'}")
        if result.stderr:
            print(f"  TryOnHairstyle stderr: {result.stderr[-500:]}")
        
        if result.returncode != 0:
            return {"status": "FAILURE", "error": f"TryOnHairstyle failed (code {result.returncode}): {result.stderr[-200:] if result.stderr else 'Unknown error'}"}
        
        # Tìm file kết quả trong session_dir
        result_path = os.path.join(session_dir, "result.png")
        if not os.path.exists(result_path):
            # Thử tìm bất kỳ file ảnh nào trong output
            for f in os.listdir(session_dir):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    result_path = os.path.join(session_dir, f)
                    break
        
        if not os.path.exists(result_path):
            return {"status": "FAILURE", "error": "TryOnHairstyle completed but no result image found."}
        
        print(f"  ✅ TryOnHairstyle result saved: {result_path}")
        return {
            "status": "SUCCESS",
            "result_path": str(result_path),
            "session_dir": str(session_dir),
            "url": f"/static/output/{session_name}/{os.path.basename(result_path)}"
        }
        
    except subprocess.TimeoutExpired:
        return {"status": "FAILURE", "error": "TryOnHairstyle timed out (>15 minutes)"}
    except Exception as e:
        return {"status": "FAILURE", "error": f"TryOnHairstyle error: {str(e)}"}


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

@celery_app.task(bind=True)
def process_detect_faces(self, image_path: str):
    """
    Quét và cắt các khuôn mặt từ ảnh đầu vào.
    Nới rộng bbox một chút (1.8x) để cắt trọn vẹn cả tóc và cổ, phục vụ ghép tóc tốt hơn.
    """
    try:
        if not _SERVICES["face"]:
            _SERVICES["face"] = FaceInfoService()
        face_service = _SERVICES["face"]
    except Exception as e:
        return {"status": "FAILURE", "error": f"Face Service Load Failed: {str(e)}"}

    try:
        session_name = f"detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.request.id[:8]}"
        session_dir = os.path.join(str(OUTPUT_DIR), session_name)
        os.makedirs(session_dir, exist_ok=True)
        print(f">>> Detect Faces session: {session_dir}")
        
        self.update_state(state='PROCESSING', meta={'step': 'Loading image and detecting'})
        user_cv2 = cv2.imread(image_path)
        if user_cv2 is None:
            return {"status": "FAILURE", "error": "Cannot read uploaded image."}
            
        faces = face_service.analyze_all(user_cv2)
        if not faces:
            return {"status": "SUCCESS", "faces": []}
            
        user_pil = Image.fromarray(cv2.cvtColor(user_cv2, cv2.COLOR_BGR2RGB))
        results = []
        
        print(f"  🔍 Detected {len(faces)} faces.")
        for i, f in enumerate(faces):
            bbox = f.bbox
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            
            # Lấy vùng crop hình vuông và đẩy tâm lên trên để lấy trọn mái tóc
            cx, cy = x1 + w/2, y1 + h/2
            
            # Mở rộng khung thành hình vuông gấp 2.5 lần cạnh lớn nhất của khuôn mặt
            size = max(w, h) * 2.5
            
            # Đẩy tâm y lên cao 20% để không bị mất đỉnh đầu
            cy -= h * 0.2
            
            new_x1 = max(0, int(cx - size/2))
            new_y1 = max(0, int(cy - size/2))
            new_x2 = min(user_pil.width, int(cx + size/2))
            new_y2 = min(user_pil.height, int(cy + size/2))
            
            crop = user_pil.crop((new_x1, new_y1, new_x2, new_y2))
            crop_filename = f"face_{i}.png"
            crop_path = os.path.join(session_dir, crop_filename)
            crop.save(crop_path)
            
            confidence = getattr(f, "det_score", 1.0)
            if hasattr(confidence, "item"):
                confidence = confidence.item()  # Unwrap tensor if needed
                
            results.append({
                "face_id": i,
                "bbox": [new_x1, new_y1, new_x2, new_y2],
                "confidence": float(confidence),
                "cropped_image_url": f"/static/output/{session_name}/{crop_filename}"
            })
            
        print(f"  ✅ Finished saving {len(results)} cropped faces.")
        return {
            "status": "SUCCESS",
            "faces": results
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "FAILURE", "error": str(e)}

