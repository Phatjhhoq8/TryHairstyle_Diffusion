
import sys
import os

# Add project root to path — MUST be before any backend imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# CRITICAL: Import torch patch BEFORE any diffusers/transformers imports!
print("Applying torch/diffusers/transformers patches...", flush=True)
from backend.app.utils import torch_patch  # noqa: F401

print("Importing libraries...", flush=True)
import random
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageOps

print("Importing backend services...", flush=True)
from datetime import datetime
from backend.app.config import settings, OUTPUT_DIR
from backend.app.services.face import FaceInfoService
from backend.app.services.mask import SegmentationService
from backend.app.services.diffusion import HairDiffusionService
from backend.app.services.hair_color_service import HairColorService, PRESET_COLORS
from backend.app.services.translate_service import translate_vi_to_en

# Global Services (lazy loaded)
face_service = None
mask_service = None
diffusion_service = None
depth_estimator = None
color_service = None


def load_services():
    """Load tất cả AI services (1 lần duy nhất). Dùng yield để Gradio không bị timeout."""
    global face_service, mask_service, diffusion_service, depth_estimator, color_service
    
    if face_service is not None and diffusion_service is not None and depth_estimator is not None:
        yield "✅ Services Already Loaded"
        return
    
    print(">>> Loading Services...", flush=True)
    try:
        # Reset first to ensure clean state
        face_service = None
        mask_service = None
        diffusion_service = None
        depth_estimator = None
        
        yield "⏳ Đang tải Face Service (1/5)..."
        face_service = FaceInfoService()
        print("  ✅ Face Service loaded", flush=True)
        
        yield "⏳ Đang tải Mask Service (2/5)..."
        mask_service = SegmentationService()
        print("  ✅ Mask Service loaded", flush=True)
        
        yield "⏳ Đang tải Diffusion Service (3/5 - Nặng nhất, chờ ~1-2 phút)..."
        diffusion_service = HairDiffusionService()
        print("  ✅ Diffusion Service loaded", flush=True)
        
        yield "⏳ Đang tải Depth Estimator (4/5)..."
        from transformers import pipeline
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
        print("  ✅ Depth Estimator loaded (Intel/dpt-large)", flush=True)
        
        yield "⏳ Đang tải Hair Color Service (5/5)..."
        color_service = HairColorService()
        print("  ✅ Hair Color Service loaded", flush=True)
        
        print(">>> All Services Loaded Successfully!", flush=True)
        yield "✅ Services Loaded — Ready to Run"
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Ensure we don't leave partial state
        face_service = None
        mask_service = None
        diffusion_service = None
        depth_estimator = None
        color_service = None
        yield f"❌ Error: {e}"


def get_random_ffhq_image():
    """Lấy 1 ảnh ngẫu nhiên từ dataset FFHQ."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_root = os.path.join(base_dir, "backend", "data", "dataset", "ffhq")
    
    if not os.path.exists(dataset_root):
        return None
    
    folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    if not folders:
        return None
    
    folder = random.choice(folders)
    folder_path = os.path.join(dataset_root, folder)
    files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        return None
    
    img_path = os.path.join(folder_path, random.choice(files))
    return Image.open(img_path).convert("RGB")


def process_pipeline(user_image, hair_image, prompt, language="English", ai_model="HairFusion", latent_injection_weight=0.3):
    """Chạy full pipeline: Face → Mask → Depth → Diffusion. Hoặc route sang TryOnHairstyle."""
    if user_image is None or hair_image is None:
        return None, "⚠️ Please select both images."
    
    # === Route to TryOnHairstyle (subprocess) ===
    if "TryOnHairstyle" in ai_model:
        return _run_tryonhairstyle_gradio(user_image, hair_image)
    
    # Dịch prompt nếu người dùng chọn tiếng Việt
    final_prompt = prompt
    if language == "Tiếng Việt":
        final_prompt = translate_vi_to_en(prompt)
        print(f"  🌐 Đã dịch prompt: '{prompt}' → '{final_prompt}'")
    # Auto-load services nếu chưa load
    global face_service, mask_service, diffusion_service, depth_estimator
    if face_service is None or diffusion_service is None or depth_estimator is None:
        load_result = list(load_services())[-1]
        if "Error" in load_result:
            return None, f"❌ Service Load Failed: {load_result}"
        if diffusion_service is None:
            return None, "❌ Critical: HairDiffusionService failed to load."
    
    try:
        # Tạo session folder cho lần inference này
        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = os.path.join(str(OUTPUT_DIR), session_name)
        os.makedirs(session_dir, exist_ok=True)
        print(f">>> Session folder: {session_dir}", flush=True)
        
        # Convert to CV2 for Face Analysis
        user_cv2 = cv2.cvtColor(np.array(user_image), cv2.COLOR_RGB2BGR)
        
        # 1. Face Analysis
        print("  → Face Analysis...", flush=True)
        face_info = face_service.analyze(user_cv2)
        status_msg = "✅ Success"
        if not face_info:
            print("  ⚠️ No face detected. Proceeding anyway...", flush=True)
            status_msg = "⚠️ No face detected — result may vary."
        
        # 2. Hair + Face Mask (lấy cả 2 mask trong 1 lần inference)
        print("  → Creating Hair & Face Mask...", flush=True)
        masks = mask_service.get_hair_and_face_mask(user_image)
        hair_mask = masks["hair_mask"]
        face_mask = masks["face_mask"]
        
        # Mask tóc từ ảnh reference
        ref_masks = mask_service.get_hair_and_face_mask(hair_image)
        ref_hair_mask = ref_masks["hair_mask"]
        
        # Lưu mask trung gian vào session folder
        hair_mask.save(os.path.join(session_dir, "hair_mask.png"))
        face_mask.save(os.path.join(session_dir, "face_mask.png"))
        ref_hair_mask.save(os.path.join(session_dir, "ref_hair_mask.png"))
        print(f"  ✅ Saved hair_mask, face_mask, ref_hair_mask → {session_dir}", flush=True)
        
        # Dynamic Mask: mở rộng mask nếu tóc mẫu lớn hơn tóc user
        hair_mask = mask_service.expand_hair_mask(hair_mask, face_mask, ref_hair_mask, face_info)
        hair_mask.save(os.path.join(session_dir, "hair_mask_expanded.png"))
        
        # 3. Depth Estimation (dùng cached depth_estimator)
        print("  → Estimating Depth...", flush=True)
        depth_map = depth_estimator(user_image)['depth']
        depth_map.save(os.path.join(session_dir, "depth_map.png"))
        print(f"  ✅ Saved depth_map → {session_dir}", flush=True)
        
        # 4. Diffusion
        original_size = user_image.size  # (w, h) — lưu kích thước gốc
        print(f"  → Running Generation (original: {original_size})...", flush=True)
        result = diffusion_service.generate(
            base_image=user_image,
            mask_image=hair_mask,
            control_image=depth_map,
            ref_hair_image=hair_image,
            prompt=final_prompt,
            latent_injection_weight=latent_injection_weight
        )
        
        # Resize kết quả về kích thước gốc để không bị méo
        if result.size != original_size:
            result = result.resize(original_size, Image.LANCZOS)
        
        # Lưu kết quả vào session folder
        result.save(os.path.join(session_dir, "result.png"))
        print(f"  ✅ Saved result ({original_size}) → {session_dir}", flush=True)
        
        # Cleanup GPU memory sau mỗi inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result, f"{status_msg}\n📁 Session: {session_name}"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Cleanup GPU memory ngay cả khi lỗi
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, f"❌ Error: {str(e)}"


def _run_tryonhairstyle_gradio(user_image, hair_image):
    """
    Chạy mô hình TryOnHairstyle qua subprocess (cách ly hoàn toàn).
    Dùng cho Gradio Test UI.
    """
    import subprocess, sys, tempfile
    from datetime import datetime as dt
    
    try:
        session_name = f"tryon_{dt.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = os.path.join(str(OUTPUT_DIR), session_name)
        os.makedirs(session_dir, exist_ok=True)
        
        # Lưu ảnh tạm
        face_path = os.path.join(session_dir, "input_face.png")
        hair_path = os.path.join(session_dir, "input_hair.png")
        user_image.save(face_path)
        hair_image.save(hair_path)
        
        # Tìm thư mục TryOnHairstyle
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        tryon_dir = os.path.join(base_dir, "TryOnHairstyle-master")
        run_script = os.path.join(tryon_dir, "run_custom.py")
        
        if not os.path.exists(run_script):
            return None, f"❌ TryOnHairstyle script not found: {run_script}"
        
        # Dùng Python từ venv riêng của TryOnHairstyle (tránh xung đột thư viện)
        tryon_python = os.path.join(tryon_dir, "hairfusion", "bin", "python")
        if not os.path.exists(tryon_python):
            return None, "❌ TryOnHairstyle venv not found. Cần tạo venv `hairfusion` riêng trước."
        
        print(f">>> Running TryOnHairstyle subprocess...")
        
        # Thiết lập LD_LIBRARY_PATH: libcuda.so (WSL) + libcudnn*.so (torch)
        site_packages = os.path.join(tryon_dir, "hairfusion", "lib", "python3.8", "site-packages")
        torch_lib = os.path.join(site_packages, "torch", "lib")
        vision_lib = os.path.join(site_packages, "torchvision.libs")
        ld_path = f"/usr/lib/wsl/lib:{torch_lib}:{vision_lib}"
        
        shell_cmd = (
            f'export LD_LIBRARY_PATH="{ld_path}:${{LD_LIBRARY_PATH:-}}" && '
            f'"{tryon_python}" "{run_script}" '
            f'--face "{face_path}" --hair "{hair_path}" --output "{session_dir}"'
        )
        print(f"  Shell cmd: {shell_cmd}")
        
        result = subprocess.run(
            shell_cmd, shell=True, executable="/bin/bash",
            cwd=tryon_dir, capture_output=True, text=True, timeout=900
        )
        
        # Debug: in stdout/stderr từ subprocess
        if result.stdout:
            print(f"  TryOnHairstyle stdout:\n{result.stdout[-1000:]}")
        if result.stderr:
            print(f"  TryOnHairstyle stderr:\n{result.stderr[-500:]}")
        
        if result.returncode != 0:
            error_detail = result.stderr[-300:] if result.stderr else (result.stdout[-300:] if result.stdout else 'Unknown')
            return None, f"❌ TryOnHairstyle failed (code {result.returncode}): {error_detail}"
        
        # Tìm kết quả — liệt kê tất cả file trong session_dir
        all_files = os.listdir(session_dir) if os.path.exists(session_dir) else []
        print(f"  Session dir files: {all_files}")
        
        result_path = os.path.join(session_dir, "result.png")
        if not os.path.exists(result_path):
            # Tìm bất kỳ file ảnh nào (không chỉ bắt đầu bằng 'result')
            for f in all_files:
                if f.endswith(('.png', '.jpg', '.jpeg')) and f not in ('input_face.png', 'input_hair.png'):
                    result_path = os.path.join(session_dir, f)
                    print(f"  Found result file: {f}")
                    break
        
        if not os.path.exists(result_path):
            stdout_hint = result.stdout[-300:] if result.stdout else '(empty)'
            return None, f"❌ TryOnHairstyle xong nhưng không tìm thấy ảnh kết quả.\nFiles: {all_files}\nStdout: {stdout_hint}"
        
        return Image.open(result_path).convert("RGB"), f"✅ TryOnHairstyle thành công\n📁 Session: {session_name}"
    
    except subprocess.TimeoutExpired:
        return None, "❌ TryOnHairstyle timeout (>15 phút)"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ TryOnHairstyle Error: {str(e)}"

def process_colorize_pipeline(user_image, color_name, intensity):
    """Chạy pipeline đổi màu tóc (không cần Diffusion)."""
    if user_image is None:
        return None, "⚠️ Vui lòng upload ảnh khuôn mặt."
    
    global mask_service, color_service
    # Chỉ cần mask + color service
    if mask_service is None:
        try:
            mask_service = SegmentationService()
        except Exception as e:
            return None, f"❌ Mask Service Load Failed: {e}"
    if color_service is None:
        color_service = HairColorService()
    
    try:
        session_name = f"color_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = os.path.join(str(OUTPUT_DIR), session_name)
        os.makedirs(session_dir, exist_ok=True)
        
        # 1. Tạo hair mask
        print("  → Creating Hair Mask...", flush=True)
        masks = mask_service.get_hair_and_face_mask(user_image)
        hair_mask = masks["hair_mask"]
        hair_mask.save(os.path.join(session_dir, "hair_mask.png"))
        
        # 2. Colorize
        print(f"  → Colorizing: {color_name} (intensity: {intensity})", flush=True)
        result = color_service.colorize(user_image, hair_mask, color_name, intensity)
        
        # 3. Save
        result.save(os.path.join(session_dir, "result.png"))
        print(f"  ✅ Saved color result → {session_dir}", flush=True)
        
        return result, f"✅ Đổi màu thành công: {color_name}\n📁 Session: {session_name}"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"


# ======================= GRADIO UI =======================

custom_css = """
/* Header */
.header-row { 
    background: linear-gradient(135deg, #1a7a6d, #2d9b8e); 
    padding: 16px 24px; 
    border-radius: 12px; 
    margin-bottom: 16px; 
}
.header-row h1 { color: white !important; margin: 0 !important; font-size: 1.6em !important; }
.header-row p { color: rgba(255,255,255,0.85) !important; margin: 4px 0 0 0 !important; font-size: 0.9em !important; }

/* Upload boxes */
.upload-box { 
    border: 2px dashed #2d9b8e !important; 
    border-radius: 12px !important; 
    min-height: 280px !important; 
    background: #f8fffe !important; 
}

/* VẼ TÓC button */
.draw-btn { 
    background: linear-gradient(135deg, #f5a623, #f7c948) !important; 
    color: white !important; 
    font-size: 1.4em !important; 
    font-weight: bold !important; 
    border-radius: 16px !important; 
    min-height: 120px !important; 
    border: none !important;
    box-shadow: 0 4px 15px rgba(245,166,35,0.4) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
.draw-btn:hover { 
    transform: translateY(-2px) !important; 
    box-shadow: 0 6px 20px rgba(245,166,35,0.6) !important; 
}

/* Result panel */
.result-panel { 
    background: #f0faf8 !important; 
    border: 1px solid #d0e8e4 !important; 
    border-radius: 12px !important; 
    padding: 12px !important; 
}

/* Popup overlay - base styles only, display managed by JS */
.popup-overlay { 
    position: fixed !important;
    inset: 0 !important;
    background: rgba(0,0,0,0.6) !important; 
    z-index: 9999 !important;
    align-items: center !important;
    justify-content: center !important;
}
/* popup-active is toggled by our JS MutationObserver */
.popup-overlay.popup-active {
    display: flex !important;
}

/* Inner popup box */
.popup-content {
    background: white !important; 
    border-radius: 16px !important; 
    padding: 24px !important; 
    box-shadow: 0 12px 40px rgba(0,0,0,0.2) !important; 
    max-width: 800px !important;
    width: 90% !important;
    max-height: 90vh !important;
    overflow-y: auto !important;
}

/* Color swatches */
.color-section { 
    background: #fafafa !important; 
    border-radius: 12px !important; 
    padding: 12px !important; 
    border: 1px solid #e0e0e0 !important; 
}

/* Status bar */
.status-bar { font-size: 0.85em !important; }

/* Loading spinner */
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
.loading-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
    gap: 16px;
}
.loading-spinner {
    width: 48px; height: 48px;
    border: 4px solid rgba(45,155,142,0.2);
    border-top-color: #2d9b8e;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}
.loading-text {
    color: #2d9b8e;
    font-size: 1.1em;
    font-weight: 600;
    animation: pulse 1.5s ease-in-out infinite;
}
.loading-sub {
    color: #888;
    font-size: 0.85em;
}

/* Nút ĐỔI MÀU NHANH */
.quick-color-btn {
    background: linear-gradient(135deg, #2d9b8e, #1a7a6d) !important;
    color: white !important;
    font-size: 0.95em !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    min-height: 50px !important;
    border: none !important;
    box-shadow: 0 3px 12px rgba(45,155,142,0.3) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    margin-top: 4px !important;
}
.quick-color-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 16px rgba(45,155,142,0.5) !important;
}
"""

# JavaScript: MutationObserver quản lý popup overlay display
popup_js = """
function() {
    // Chờ DOM sẵn sàng rồi gắn MutationObserver
    requestAnimationFrame(() => {
        const popup = document.querySelector('.popup-overlay');
        if (!popup) return;
        
        function updatePopupDisplay() {
            // Gradio ẩn component bằng cách thêm class 'hidden' hoặc 'hide' 
            // hoặc set style.display = 'none' trên element hoặc parent wrapper
            const isGradioHidden = popup.classList.contains('hide') || 
                                   popup.classList.contains('hidden') ||
                                   popup.style.display === 'none' ||
                                   popup.parentElement?.style?.display === 'none';
            
            if (isGradioHidden) {
                popup.classList.remove('popup-active');
            } else {
                popup.classList.add('popup-active');
            }
        }
        
        // Observer theo dõi class + style thay đổi trên popup và parent
        const observer = new MutationObserver(updatePopupDisplay);
        observer.observe(popup, { attributes: true, attributeFilter: ['class', 'style'] });
        if (popup.parentElement) {
            observer.observe(popup.parentElement, { attributes: true, attributeFilter: ['class', 'style'] });
        }
        
        // Khởi tạo lần đầu
        updatePopupDisplay();
    });
}
"""


def extract_faces_from_image(img):
    """Detect và crop tất cả khuôn mặt từ ảnh. Trả về list PIL Images."""
    if img is None:
        return []
    
    global face_service
    if face_service is None:
        load_result = list(load_services())[-1]
        if "Error" in load_result:
            return []
    
    if face_service is None:
        return []
    
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    faces = face_service.analyze_all(img_cv2)
    
    if not faces:
        return []
    
    cropped = []
    for f in faces:
        bbox = f.bbox
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        new_w = w * 1.8
        new_h = h * 1.8
        new_x1 = max(0, int(cx - new_w / 2))
        new_y1 = max(0, int(cy - new_h / 2))
        new_x2 = min(img.width, int(cx + new_w / 2))
        new_y2 = min(img.height, int(cy + new_h / 2))
        cropped.append(img.crop((new_x1, new_y1, new_x2, new_y2)))
    
    return cropped


with gr.Blocks(title="AI Hair Stylist", theme=gr.themes.Soft(), css=custom_css) as demo:
    
    # ===== States =====
    selected_face_state = gr.State(None)       # PIL Image — khuôn mặt đã chọn
    selected_hair_state = gr.State(None)       # PIL Image — tóc đã chọn
    face_crops_state = gr.State([])            # list[PIL] — các khuôn mặt crop được
    hair_crops_state = gr.State([])            # list[PIL] — các tóc crop được
    popup_source_state = gr.State("")          # "face" hoặc "hair" — popup đang chọn cho cái nào
    
    # ===== Header =====
    with gr.Row(elem_classes="header-row"):
        gr.Markdown("# AI HAIR STYLIST — HỆ THỐNG TẠO KIỂU TÓC\n*Tải ảnh chân dung & kiểu tóc tham khảo, nhấn Vẽ Tóc để tạo kết quả*")
    
    # ===== Status bar =====
    with gr.Row():
        status_box = gr.Textbox(
            label="Trạng thái hệ thống", value="Chưa khởi tạo — Bấm 'Khởi tạo' hoặc chạy lần đầu sẽ tự load",
            interactive=False, elem_classes="status-bar"
        )
        load_btn = gr.Button("Khởi tạo", variant="secondary", scale=0)
    
    # ===== Main Layout: 2 uploads + VẼ TÓC btn + Result =====
    with gr.Row():
        # --- Column 1: Ảnh chân dung ---
        with gr.Column(scale=3):
            gr.Markdown("### ẢNH ĐẦU VÀO 1: CHÂN DUNG")
            user_input = gr.Image(
                type="pil",
                height=300,
                elem_classes="upload-box"
            )
        
        # --- Column 2: Ảnh kiểu tóc ---
        with gr.Column(scale=3):
            gr.Markdown("### ẢNH ĐẦU VÀO 2: THAM KHẢO KIỂU TÓC")
            hair_input = gr.Image(
                type="pil",
                height=300,
                elem_classes="upload-box"
            )
        
        # --- Column 3: Nút VẼ TÓC + ĐỔI MÀU NHANH ---
        with gr.Column(scale=1, min_width=140):
            gr.Markdown("&nbsp;")  # spacer
            draw_btn = gr.Button(
                "VẼ TÓC",
                variant="primary",
                elem_classes="draw-btn"
            )
            gr.Markdown("<center>Nhấn để tạo kết quả</center>")
            quick_color_btn = gr.Button(
                "ĐỔI MÀU NHANH",
                variant="secondary",
                elem_classes="quick-color-btn"
            )
            gr.Markdown("<center><small>Đổi màu tóc trên ảnh kết quả</small></center>")
        
        # --- Column 4: Kết quả ---
        with gr.Column(scale=3, elem_classes="result-panel"):
            gr.Markdown("### ẢNH KẾT QUẢ")
            output_image = gr.Image(label="Kết quả", type="pil", height=300, interactive=False)
            loading_html = gr.HTML(
                value="",
                visible=False,
                elem_classes="loading-container"
            )
            log_output = gr.Textbox(label="Trạng thái", lines=2, interactive=False)
            with gr.Row():
                download_btn = gr.Button("Tải xuống kết quả", variant="primary", size="sm")
                clear_result_btn = gr.Button("❌ Xóa kết quả", variant="secondary", size="sm")
    
    # ===== Text Prompt + Language Selector (dưới 2 ảnh) =====
    with gr.Row():
        with gr.Column(scale=5):
            prompt_input = gr.Textbox(
                label="Text Prompt",
                value="high quality, realistic hairstyle",
                placeholder="Mô tả kiểu tóc mong muốn...",
                lines=2
            )
        with gr.Column(scale=1, min_width=150):
            language_radio = gr.Radio(
                choices=["English", "Tiếng Việt"],
                value="English",
                label="Ngôn ngữ Prompt",
                info="Nếu chọn Tiếng Việt, prompt sẽ tự động dịch sang Anh"
            )
        with gr.Column(scale=2, min_width=200):
            model_radio = gr.Radio(
                choices=["TryHairStyle", "TryOnHairstyle"],
                value="HairFusion",
                label="Mô hình AI",
                info="Chọn mô hình sử dụng để tạo kiểu tóc"
            )
        with gr.Column(scale=2):
            pass  # khoảng trống cân đối
    
    # ===== Color Picker (dưới prompt) =====
    with gr.Row(elem_classes="color-section"):
        with gr.Column(scale=2):
            gr.Markdown("### Tuỳ chỉnh màu tóc (tuỳ chọn)")
            preset_names = list(PRESET_COLORS.keys())
            color_dropdown = gr.Dropdown(
                label="Màu preset",
                choices=["none"] + preset_names,
                value="none",
                info="Chọn 'none' nếu không muốn đổi màu"
            )
        with gr.Column(scale=2):
            color_hex_input = gr.Textbox(
                label="Hex tuỳ chỉnh",
                placeholder="#FF0000",
                info="Ưu tiên hex nếu nhập (để trống = dùng preset)"
            )
        with gr.Column(scale=2):
            color_intensity_slider = gr.Slider(
                label="Cường độ màu",
                minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                info="0 = giữ nguyên, 1 = 100% màu mới"
            )
        with gr.Column(scale=1):
            random_btn = gr.Button("FFHQ ngẫu nhiên", variant="secondary", size="sm")
    
    # ===== Latent Injection Slider =====
    with gr.Row():
        with gr.Column(scale=3):
            injection_slider = gr.Slider(
                label="Latent Injection Weight",
                minimum=0.0, maximum=0.5, value=0.3, step=0.05,
                info="Truyền kiểu tóc mẫu trực tiếp vào latent space. 0 = tắt, 0.3 = mặc định, 0.5 = mạnh nhất"
            )
        with gr.Column(scale=3):
            pass  # khoảng trống cân đối
    
    # ===== POPUP: Chọn khuôn mặt / tóc (ẩn mặc định) =====
    with gr.Column(visible=False, elem_classes="popup-overlay") as popup_panel:
        with gr.Column(elem_classes="popup-content"):
            popup_title = gr.Markdown("### Phát hiện nhiều khuôn mặt — Vui lòng chọn một")
            popup_gallery = gr.Gallery(
                label="Các khuôn mặt phát hiện được",
                show_label=True, columns=4, height=300,
                object_fit="contain", allow_preview=False
            )
            with gr.Row():
                popup_confirm_btn = gr.Button("Xác nhận chọn", variant="primary")
                popup_cancel_btn = gr.Button("Huỷ", variant="secondary")
            popup_selected_preview = gr.Image(label="Đã chọn", type="pil", height=150)
    
    # ===============================================================
    # ======================== EVENT HANDLERS ========================
    # ===============================================================
    
    load_btn.click(fn=load_services, outputs=status_box)
    
    clear_result_btn.click(
        fn=lambda: (None, "Trạng thái: Đã xóa kết quả"),
        outputs=[output_image, log_output]
    )
    
    # --- Random FFHQ pair ---
    def random_pair():
        img1 = get_random_ffhq_image()
        img2 = get_random_ffhq_image()
        return img1, img2
    random_btn.click(fn=random_pair, outputs=[user_input, hair_input])
    
    # --- Click VẼ TÓC: validate → detect faces → popup hoặc chạy luôn ---
    def on_draw_click(user_img, hair_img, prompt, lang, ai_model, color_name, hex_input, intensity, injection_weight):
        """
        Bước 1: Validate input.
        Bước 2: Detect faces trong ảnh chân dung.
           - Nếu > 1 face → trả về list faces + hiện popup.
           - Nếu 1 face → lưu face đó.
        Bước 3: Detect faces trong ảnh tóc (tương tự).
        Bước 4: Nếu không cần popup → chạy pipeline luôn.
        
        Returns: (output_image, log, popup_visible, popup_title, popup_gallery, 
                  face_crops, hair_crops, selected_face, selected_hair, popup_source,
                  popup_preview)
        """
        # Validate
        if user_img is None:
            raise gr.Error("⚠️ Vui lòng tải lên ảnh chân dung!")
        if hair_img is None:
            raise gr.Error("⚠️ Vui lòng tải lên ảnh kiểu tóc tham khảo!")
        
        # Detect faces in user image
        user_faces = extract_faces_from_image(user_img)
        
        if len(user_faces) > 1:
            # Nhiều khuôn mặt → hiện popup chọn
            yield (
                gr.update(),            # output_image — giữ nguyên 
                "🔍 Phát hiện nhiều khuôn mặt trong ảnh chân dung. Vui lòng chọn 1.",
                gr.update(visible=True),  # popup_panel
                "### 🔍 Phát hiện **nhiều khuôn mặt** trong ảnh chân dung — Chọn 1 khuôn mặt để xử lý",
                user_faces,              # popup_gallery
                user_faces,              # face_crops_state
                [],                      # hair_crops_state (chưa detect)
                None,                    # selected_face_state
                None,                    # selected_hair_state
                "face",                  # popup_source_state
                None,                    # popup_selected_preview
                gr.update(),             # loading_html
            )
            return
        
        # 1 face hoặc 0 face trong ảnh chân dung → dùng luôn ảnh gốc (hoặc face crop nếu có 1)
        chosen_face = user_faces[0] if len(user_faces) == 1 else user_img
        
        # Detect faces in hair image
        hair_faces = extract_faces_from_image(hair_img)
        
        if len(hair_faces) > 1:
            # Nhiều khuôn mặt/tóc → hiện popup chọn
            yield (
                gr.update(),
                "🔍 Phát hiện nhiều khuôn mặt trong ảnh tóc tham khảo. Vui lòng chọn 1.",
                gr.update(visible=True),
                "### 🔍 Phát hiện **nhiều khuôn mặt** trong ảnh tóc — Chọn 1 kiểu tóc để tham khảo",
                hair_faces,
                [],                     # face_crops (đã chọn xong)
                hair_faces,             # hair_crops_state
                chosen_face,            # selected_face_state (đã OK)
                None,                   # selected_hair_state
                "hair",                 # popup_source_state
                None,
                gr.update(),            # loading_html
            )
            return
        
        # Không cần popup → chạy pipeline luôn
        chosen_hair = hair_faces[0] if len(hair_faces) == 1 else hair_img
        
        # Yield trạng thái loading: ẩn ảnh output, hiện spinner thay thế
        loading_content = '''
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <div class="loading-text">⏳ Đang tạo kiểu tóc...</div>
            <div class="loading-sub">Quá trình này mất khoảng 30-60 giây</div>
        </div>
        '''
        yield (
            gr.update(visible=False),    # ẩn output_image
            "⏳ Đang xử lý tạo kiểu tóc, vui lòng chờ...",
            gr.update(visible=False), "", [],
            [], [], chosen_face, chosen_hair, "", None,
            gr.update(value=loading_content, visible=True),  # hiện loading
        )
        
        result_img, status_msg = process_pipeline(chosen_face, chosen_hair, prompt, lang, ai_model, injection_weight)
        
        # Nếu có chọn color → colorize thêm
        if result_img is not None:
            actual_color = hex_input.strip() if hex_input and hex_input.strip() else color_name
            if actual_color and actual_color != "none":
                colored_img, color_msg = process_colorize_pipeline(result_img, actual_color, intensity)
                if colored_img is not None:
                    result_img = colored_img
                    status_msg += f"\n🎨 {color_msg}"
        
        
        yield (
            gr.update(value=result_img, visible=True),  # hiện lại output với kết quả
            status_msg,
            gr.update(visible=False),
            "",
            [],
            [], [],
            chosen_face, chosen_hair,
            "",
            None,
            gr.update(value="", visible=False),   # ẩn loading
        )
    
    draw_btn.click(
        fn=on_draw_click,
        inputs=[user_input, hair_input, prompt_input, language_radio, model_radio, color_dropdown, color_hex_input, color_intensity_slider, injection_slider],
        outputs=[
            output_image, log_output,
            popup_panel, popup_title, popup_gallery,
            face_crops_state, hair_crops_state,
            selected_face_state, selected_hair_state,
            popup_source_state, popup_selected_preview,
            loading_html
        ]
    )
    
    # --- Popup: select from gallery ---
    def on_popup_select(face_crops, hair_crops, source, evt: gr.SelectData):
        """User click vào 1 ảnh trong popup gallery."""
        if source == "face":
            chosen = face_crops[evt.index]
        else:
            chosen = hair_crops[evt.index]
        return chosen
    
    popup_gallery.select(
        fn=on_popup_select,
        inputs=[face_crops_state, hair_crops_state, popup_source_state],
        outputs=[popup_selected_preview]
    )
    
    # --- Popup: confirm selection ---
    def on_popup_confirm(
        preview_img, source,
        face_crops, hair_crops,
        sel_face, sel_hair,
        user_img, hair_img,
        prompt, color_name, hex_input, intensity
    ):
        """
        Xác nhận chọn từ popup.
        - Nếu đang chọn face → lưu face, tiếp tục check hair.
        - Nếu đang chọn hair → lưu hair, chạy pipeline.
        """
        if preview_img is None:
            raise gr.Error("⚠️ Vui lòng click chọn 1 ảnh trong danh sách trước!")
        
        if source == "face":
            # Đã chọn face xong → check hair
            chosen_face = preview_img
            
            hair_faces = extract_faces_from_image(hair_img)
            if len(hair_faces) > 1:
                return (
                    gr.update(),
                    "🔍 Phát hiện nhiều khuôn mặt trong ảnh tóc. Vui lòng chọn 1.",
                    gr.update(visible=True),
                    "### 🔍 Phát hiện **nhiều khuôn mặt** trong ảnh tóc — Chọn 1 kiểu tóc để tham khảo",
                    hair_faces,
                    [],             # face_crops — đã xong
                    hair_faces,     # hair_crops
                    chosen_face,    # selected_face
                    None,           # selected_hair (chưa chọn)
                    "hair",         # source
                    None,           # preview
                    chosen_face,    # user_input update
                    gr.update(),    # hair_input update
                )
            
            # Hair OK → Đóng popup, cập nhật ảnh, yêu cầu người dùng bấm VẼ TÓC
            chosen_hair = hair_faces[0] if len(hair_faces) == 1 else hair_img
            
            return (
                None, "✅ Đã chọn khuôn mặt xong. Vui lòng bấm 'VẼ TÓC' để tạo kiểu.",
                gr.update(visible=False), "", [],
                [], [],
                chosen_face, chosen_hair,
                "", None,
                gr.update(value=chosen_face), gr.update(value=chosen_hair),
            )
        
        else:
            # source == "hair" — đã chọn hair → cập nhật ảnh
            chosen_hair = preview_img
            chosen_face = sel_face if sel_face is not None else user_img
            
            return (
                None, "✅ Đã chọn kiểu tóc xong. Vui lòng bấm 'VẼ TÓC' để tạo kiểu.",
                gr.update(visible=False), "", [],
                [], [],
                chosen_face, chosen_hair,
                "", None,
                gr.update(value=chosen_face), gr.update(value=chosen_hair),
            )
    
    popup_confirm_btn.click(
        fn=on_popup_confirm,
        inputs=[
            popup_selected_preview, popup_source_state,
            face_crops_state, hair_crops_state,
            selected_face_state, selected_hair_state,
            user_input, hair_input,
            prompt_input, color_dropdown, color_hex_input, color_intensity_slider
        ],
        outputs=[
            output_image, log_output,
            popup_panel, popup_title, popup_gallery,
            face_crops_state, hair_crops_state,
            selected_face_state, selected_hair_state,
            popup_source_state, popup_selected_preview,
            user_input, hair_input
        ]
    )
    
    # --- Quick Color: đổi màu nhanh trên ảnh kết quả hoặc ảnh đầu vào ---
    def on_quick_color_click(result_img, user_img, color_name, hex_input, intensity):
        """Đổi màu tóc nhanh (không cần Diffusion). Ưu tiên ảnh kết quả, không thì dùng ảnh đầu vào."""
        # Chọn ảnh nguồn: kết quả > đầu vào
        source_img = result_img if result_img is not None else user_img
        if source_img is None:
            raise gr.Error("⚠️ Chưa có ảnh nào để đổi màu! Hãy tải ảnh chân dung lên hoặc Vẽ Tóc trước.")
        
        # Ưu tiên hex nếu có, ngược lại dùng preset
        actual_color = hex_input.strip() if hex_input and hex_input.strip() else color_name
        if not actual_color or actual_color == "none":
            raise gr.Error("⚠️ Vui lòng chọn màu tóc trước khi đổi màu!")
        
        label = "ảnh kết quả" if result_img is not None else "ảnh đầu vào"
        colored_img, status_msg = process_colorize_pipeline(source_img, actual_color, intensity)
        if colored_img is not None:
            return colored_img, f"{status_msg}\n🖼️ Đổi màu trên {label}"
        else:
            return gr.update(), status_msg
    
    quick_color_btn.click(
        fn=on_quick_color_click,
        inputs=[output_image, user_input, color_dropdown, color_hex_input, color_intensity_slider],
        outputs=[output_image, log_output]
    )
    
    # --- Popup: cancel ---
    def on_popup_cancel():
        return (
            gr.update(visible=False), "", [],
            [], [],
            None, None,
            "", None,
        )
    
    popup_cancel_btn.click(
        fn=on_popup_cancel,
        outputs=[
            popup_panel, popup_title, popup_gallery,
            face_crops_state, hair_crops_state,
            selected_face_state, selected_hair_state,
            popup_source_state, popup_selected_preview
        ]
    )
    
    # --- Download result ---
    def save_result(result_img):
        if result_img is None:
            raise gr.Error("Chưa có kết quả để tải!")
        save_path = os.path.join(str(OUTPUT_DIR), f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        result_img.save(save_path)
        return save_path
    
    download_btn.click(
        fn=save_result,
        inputs=[output_image],
        outputs=gr.File(visible=False)
    )

if __name__ == "__main__":
    print(">>> Launching Gradio UI...", flush=True)
    demo.launch(server_name="127.0.0.1", server_port=7862, share=False)
