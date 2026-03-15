
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

# Global Services (lazy loaded)
face_service = None
mask_service = None
diffusion_service = None
depth_estimator = None
color_service = None


def load_services():
    """Load tất cả AI services (1 lần duy nhất)."""
    global face_service, mask_service, diffusion_service, depth_estimator, color_service
    
    if face_service is not None and diffusion_service is not None and depth_estimator is not None:
        return "✅ Services Already Loaded"
    
    print(">>> Loading Services...", flush=True)
    try:
        # Reset first to ensure clean state
        face_service = None
        mask_service = None
        diffusion_service = None
        depth_estimator = None
        
        face_service = FaceInfoService()
        print("  ✅ Face Service loaded", flush=True)
        
        mask_service = SegmentationService()
        print("  ✅ Mask Service loaded", flush=True)
        
        diffusion_service = HairDiffusionService()
        print("  ✅ Diffusion Service loaded", flush=True)
        
        # Depth estimator — load 1 lần, cache global (tránh reload mỗi inference)
        from transformers import pipeline
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
        print("  ✅ Depth Estimator loaded (Intel/dpt-large)", flush=True)
        
        color_service = HairColorService()
        print("  ✅ Hair Color Service loaded", flush=True)
        
        print(">>> All Services Loaded Successfully!", flush=True)
        return "✅ Services Loaded — Ready to Run"
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Ensure we don't leave partial state
        face_service = None
        mask_service = None
        diffusion_service = None
        depth_estimator = None
        color_service = None
        return f"❌ Error: {e}"


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


def process_pipeline(user_image, hair_image, prompt):
    """Chạy full pipeline: Face → Mask → Depth → Diffusion."""
    if user_image is None or hair_image is None:
        return None, "⚠️ Please select both images."
    
    # Auto-load services nếu chưa load
    global face_service, mask_service, diffusion_service, depth_estimator
    if face_service is None or diffusion_service is None or depth_estimator is None:
        load_result = load_services()
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
            prompt=prompt
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

with gr.Blocks(title="TryHairStyle - FFHQ Test", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💇 TryHairStyle Review Tool (FFHQ)")
    gr.Markdown("*Pipeline: Face Detection → Hair+Hat Mask → Depth Estimation → SDXL Inpainting*")
    
    with gr.Row():
        status_box = gr.Textbox(label="System Status", value="⏳ Not Loaded", interactive=False)
        load_btn = gr.Button("🔧 Initialize System Services", variant="secondary")
    
    with gr.Tabs():
        # ===== Tab 1: Hair Transfer (original) =====
        with gr.TabItem("💇 Hair Transfer"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📸 Inputs")
                    user_input = gr.Image(label="User Face", type="pil")
                    hair_input = gr.Image(label="Hair Reference", type="pil")
                    random_btn = gr.Button("🎲 Load Random FFHQ Pair")
                
                with gr.Column():
                    gr.Markdown("### ⚙️ Settings")
                    prompt_input = gr.Textbox(
                        label="Prompt", 
                        value="high quality, realistic hairstyle",
                        placeholder="Describe the hairstyle..."
                    )
                    run_btn = gr.Button("🚀 Run Transfer", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### 🖼️ Result")
                    output_image = gr.Image(label="Result", type="pil")
                    log_output = gr.Textbox(label="Log", lines=3)
        
        # ===== Tab 2: Hair Color =====
        with gr.TabItem("🎨 Hair Color"):
            gr.Markdown("### 🎨 Đổi Màu Tóc")
            gr.Markdown("*Chỉ đổi màu tóc — không thay đổi kiểu tóc. Nhanh hơn Hair Transfer.*")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📸 Input")
                    color_user_input = gr.Image(label="User Face", type="pil")
                    color_random_btn = gr.Button("🎲 Load Random FFHQ")
                
                with gr.Column():
                    gr.Markdown("### ⚙️ Chọn Màu")
                    # Tạo danh sách tên preset cho dropdown
                    preset_labels = [f"{info['label']} ({name})" for name, info in PRESET_COLORS.items()]
                    preset_names = list(PRESET_COLORS.keys())
                    color_dropdown = gr.Dropdown(
                        label="Preset Color",
                        choices=preset_names,
                        value="blonde",
                        info="Chọn màu preset hoặc nhập hex bên dưới"
                    )
                    color_hex_input = gr.Textbox(
                        label="Custom Hex (tuỳ chọn)",
                        placeholder="#FF0000",
                        info="Để trống nếu dùng preset"
                    )
                    color_intensity_slider = gr.Slider(
                        label="Intensity", minimum=0.0, maximum=1.0,
                        value=0.7, step=0.05,
                        info="0 = giữ nguyên, 1 = 100% màu mới"
                    )
                    color_run_btn = gr.Button("🎨 Colorize", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### 🖼️ Result")
                    color_output_image = gr.Image(label="Result", type="pil")
                    color_log_output = gr.Textbox(label="Log", lines=3)
    
    # ===== Events =====
    load_btn.click(fn=load_services, outputs=status_box)
    
    def random_pair():
        img1 = get_random_ffhq_image()
        img2 = get_random_ffhq_image()
        return img1, img2
    
    random_btn.click(fn=random_pair, outputs=[user_input, hair_input])
    
    run_btn.click(
        fn=process_pipeline,
        inputs=[user_input, hair_input, prompt_input],
        outputs=[output_image, log_output]
    )
    
    # Hair Color events
    def random_single():
        return get_random_ffhq_image()
    
    color_random_btn.click(fn=random_single, outputs=[color_user_input])
    
    def run_colorize(image, preset, hex_input, intensity):
        # Ưu tiên hex input nếu có
        color = hex_input.strip() if hex_input and hex_input.strip() else preset
        return process_colorize_pipeline(image, color, intensity)
    
    color_run_btn.click(
        fn=run_colorize,
        inputs=[color_user_input, color_dropdown, color_hex_input, color_intensity_slider],
        outputs=[color_output_image, color_log_output]
    )

if __name__ == "__main__":
    print(">>> Launching Gradio UI...", flush=True)
    demo.launch(server_name="127.0.0.1", server_port=7862, share=False)
