
import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
    ControlNetModel,
    AutoencoderKL
)
from transformers import CLIPVisionModelWithProjection

from backend.app.config import model_paths, settings

class HairDiffusionService:
    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.dtype = torch.float16 if "cuda" in settings.DEVICE else torch.float32
        self.use_sdxl = False # Default to False due to issues with SDXL availability
        
        # Check if SDXL Base is available locally or we should trigger SD1.5 fallback
        # Given current state: SDXL is missing/slow. Force SD1.5.
        
        print(f"Initializing HairDiffusionService (Device: {self.device}, Dtype: {self.dtype})")
        
        try:
            # Try loading SDXL Config/Model if it existed, but we know it doesn't.
            # So let's intentionally use SD1.5 logic.
            # However, keeping a structure that *could* support SDXL if paths were valid.
            if os.path.exists(model_paths.SDXL_REPO) or "stabilityai" in model_paths.SDXL_REPO:
               # Simple check: If local folder doesn't exist and repo is remote -> Validation fail if offline/slow.
               # Let's assume we want SD1.5.
               pass

            self._load_sd15_pipeline()
            
        except Exception as e:
            print(f"Error loading SD1.5 Pipeline: {e}")
            raise e

    def _load_sd15_pipeline(self):
        print(">>> Loading SD1.5 Inpaint Pipeline (AutoPipeline)...")
        from diffusers import AutoPipelineForInpainting
        
        # 1. Load SD1.5 Inpaint Pipe (Auto detects structure)
        # We specify variant="fp16" if we are using fp16 to ensure we pick the right files.
        variant = "fp16" if self.dtype == torch.float16 else None
        
        try:
            # 1. Try loading Safetensors with variant
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                model_paths.SD15_BASE,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant=variant
            )
        except Exception as e_safe:
            print(f"Warning: Failed to load Safetensors variant ({e_safe}). Trying Bin/Default...")
            try:
                # 2. Try loading .bin (use_safetensors=False)
                self.pipe = AutoPipelineForInpainting.from_pretrained(
                    model_paths.SD15_BASE,
                    torch_dtype=self.dtype,
                    use_safetensors=False, 
                    variant=variant
                )
            except Exception as e_bin:
                 print(f"Warning: Failed to load Bin variant ({e_bin}). Trying absolute default...")
                 # 3. Absolute Default (no variant, agnostic)
                 self.pipe = AutoPipelineForInpainting.from_pretrained(
                    model_paths.SD15_BASE,
                    torch_dtype=self.dtype
                 )
        
        # 2. Load IP-Adapter
        print(f">>> Loading IP-Adapter from {model_paths.IP_ADAPTER_SD15_PATH}")
        # Note: AutoPipeline often returns a specific pipeline class. Verify it has load_ip_adapter.
        # Most modern Diffusers pipelines have it.
        
        try:
            self.pipe.load_ip_adapter(
                model_paths.IP_ADAPTER_PLUS_HAIR, 
                subfolder="", 
                weight_name="ip-adapter-plus_sd15.bin"
            )
        except Exception as e:
            print(f"Warning: Failed to load IP-Adapter ({e}). Proceeding without it.")
        
        self.pipe.to(self.device)
        self.use_sdxl = False
        print(">>> SD1.5 Pipeline Loaded Successfully.")

    def generate(
        self, 
        base_image: Image.Image, 
        mask_image: Image.Image, 
        control_image: Image.Image, # Depth Map (Ignored in SD1.5 fallback if no ControlNet)
        ref_hair_image: Image.Image,
        prompt: str = "high quality, realistic hairstyle",
        negative_prompt: str = "blur, low quality, distortion",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 0.5,
        ip_adapter_scale: float = 0.6
    ):
        """
        Thực hiện Inpainting thay tóc.
        """
        # Resize inputs
        target_size = (1024, 1024) if self.use_sdxl else (512, 512)
        
        image = base_image.resize(target_size, Image.LANCZOS)
        mask = mask_image.resize(target_size, Image.NEAREST)
        ref_hair = ref_hair_image.resize(target_size, Image.LANCZOS)
        
        # Set IP Adapter Scale
        if hasattr(self.pipe, "set_ip_adapter_scale"):
             self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        else:
             print("Warning: set_ip_adapter_scale not found on pipeline. Skipping scale set.", flush=True)
        
        # Generator seed
        generator = torch.Generator(self.device).manual_seed(42)
        
        if self.use_sdxl:
            control = control_image.resize(target_size, Image.BILINEAR)
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                control_image=control,
                ip_adapter_image=ref_hair,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
                strength=0.99,
                generator=generator
            ).images[0]
        else:
            # SD1.5 Inpaint (No ControlNet Depth in this fallback)
            # Just Inpaint + (Optional) IP-Adapter
            
            # Prepare args (some versions of diffusers don't support ip_adapter_image in __call__ even if loaded)
            inpainting_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "mask_image": mask,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "strength": 0.99,
                "generator": generator
            }
            
            # Try with IP-Adapter arg first
            try:
                result = self.pipe(
                    **inpainting_args,
                    ip_adapter_image=ref_hair
                ).images[0]
            except TypeError as e:
                # If argument is not supported
                if "ip_adapter_image" in str(e):
                    print(f"Warning: Pipeline doesn't accept 'ip_adapter_image' ({e}). Running standard Inpainting.", flush=True)
                    result = self.pipe(**inpainting_args).images[0]
                else:
                    raise e # Re-raise if it's a different error
        
        return result
