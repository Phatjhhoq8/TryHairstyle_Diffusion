
import sys
import os
import random
import cv2
import numpy as np

# Monkeypatch for huggingface_hub > 0.23 compatibility with older Gradio/Libs
import os
try:
    import huggingface_hub.constants
    if not hasattr(huggingface_hub.constants, 'hf_cache_home'):
        huggingface_hub.constants.hf_cache_home = huggingface_hub.constants.HF_HOME
except ImportError:
    pass

import gradio as gr
from PIL import Image, ImageOps

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.app.config import settings
from backend.app.services.face import FaceInfoService
from backend.app.services.mask import SegmentationService
from backend.app.services.diffusion import HairDiffusionService

# Global Services
face_service = None
mask_service = None
diffusion_service = None

def load_services():
    global face_service, mask_service, diffusion_service
    # Reload if any service is missing
    if face_service is None or diffusion_service is None:
        print("Loading Services...", flush=True)
        try:
            # Reset first to ensure clean state
            face_service = None 
            mask_service = None 
            diffusion_service = None
            
            face_service = FaceInfoService()
            mask_service = SegmentationService()
            diffusion_service = HairDiffusionService()
            print("Services Loaded Successfully!", flush=True)
            return "Services Loaded Ready to Run"
        except Exception as e:
            print(f"Error loading services: {e}")
            # Ensure we don't leave partial state
            face_service = None 
            mask_service = None 
            diffusion_service = None
            return f"Error: {e}"
    return "Services Already Loaded"

def get_random_ffhq_image():
    # Use relative paths from project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_root = os.path.join(base_dir, "backend", "data", "dataset", "ffhq")
    
    if not os.path.exists(dataset_root):
        return None
    
    folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    if not folders: return None
    
    folder = random.choice(folders)
    folder_path = os.path.join(dataset_root, folder)
    files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not files: return None
    
    img_path = os.path.join(folder_path, random.choice(files))
    return Image.open(img_path).convert("RGB")

def process_pipeline(user_image, hair_image, prompt):
    if user_image is None or hair_image is None:
        return None, "Please select both images."
    
    # Check if services are actually loaded
    global diffusion_service
    if face_service is None or diffusion_service is None:
        load_result = load_services() # Auto load if not ready
        if "Error" in load_result:
             return None, f"Service Load Failed: {load_result}"
        if diffusion_service is None:
             return None, "Critical: HairDiffusionService failed to load."
        
    try:
        # Convert to CV2 for Face Analysis
        user_cv2 = cv2.cvtColor(np.array(user_image), cv2.COLOR_RGB2BGR)
        
        # 1. Face Analysis
        face_info = face_service.analyze(user_cv2)
        status_msg = "Success"
        if not face_info:
            print("Warning: No face detected. Proceeding anyway...", flush=True)
            status_msg = "Warning: No face detected by InsightFace, result accuracy may vary."
            # return None, "No face detected in User Image." # OLD BLOCKER
            
        # 2. Mask
        hair_mask = mask_service.get_mask(user_image, target_class=17)
        
        # 3. Depth (Dummy for now to avoid crash if transformers issues persist, 
        # or use simple grayscale as fallback which works for SD1.5 Inpaint pipe)
        depth_map = ImageOps.grayscale(user_image)
        
        # 4. Diffusion
        result = diffusion_service.generate(
            base_image=user_image,
            mask_image=hair_mask,
            control_image=depth_map,
            ref_hair_image=hair_image,
            prompt=prompt
        )
        
        return result, status_msg
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

# UI Construction
with gr.Blocks(title="TryHairStyle - FFHQ Test") as demo:
    gr.Markdown("# TryHairStyle Review Tool (FFHQ)")
    
    with gr.Row():
        status_box = gr.Textbox(label="System Status", value="Not Loaded", interactive=False)
        load_btn = gr.Button("Initialize System Services")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Inputs")
            user_input = gr.Image(label="User Face", type="pil")
            hair_input = gr.Image(label="Hair Reference", type="pil")
            random_btn = gr.Button("🎲 Load Random FFHQ Pair")
            
        with gr.Column():
            gr.Markdown("### 2. Settings")
            prompt_input = gr.Textbox(label="Prompt", value="high quality, realistic hairstyle")
            run_btn = gr.Button("🚀 Run Transfer", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 3. Result")
            output_image = gr.Image(label="Result", type="pil")
            log_output = gr.Textbox(label="Log")

    # Events
    load_btn.click(fn=load_services, outputs=status_box)
    
    def random_pair():
        return get_random_ffhq_image(), get_random_ffhq_image()
        
    random_btn.click(fn=random_pair, outputs=[user_input, hair_input])
    
    run_btn.click(
        fn=process_pipeline,
        inputs=[user_input, hair_input, prompt_input],
        outputs=[output_image, log_output]
    )

if __name__ == "__main__":
    # Launch on localhost
    print("Launching Gradio UI...", flush=True)
    # Using port 7861 to avoid conflict with stuck process
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)