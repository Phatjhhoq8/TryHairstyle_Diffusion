
import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLImg2ImgPipeline,
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
        # Set to True for SDXL migration as requested
        self.use_sdxl = True 
        
        print(f"Initializing HairDiffusionService (Device: {self.device}, Dtype: {self.dtype})")
        
        try:
            if self.use_sdxl:
                self._load_sdxl_pipeline()
            else:
                self._load_sd15_pipeline()
            
        except Exception as e:
            print(f"Error loading Pipeline: {e}")
            # Fallback attempts could go here
            if self.use_sdxl:
                print("Falling back to SD1.5 due to SDXL error...")
                self.use_sdxl = False
                try:
                    self._load_sd15_pipeline()
                except Exception as e2:
                    print(f"Critical: Failed to load both SDXL and SD1.5: {e2}")
                    raise e2



    def _load_refiner(self):
        """Lazy load refiner only when needed"""
        if hasattr(self, 'refiner') and self.refiner is not None:
             return
             
        print(">>> Loading SDXL Refiner for High Quality Mode...")
        
        try:
             # Check if local exists
             if os.path.exists(model_paths.SDXL_REFINER):
                print(f"Loading Refiner from single file: {model_paths.SDXL_REFINER}")
                
                # Check Base Pipe components
                text_enc_2 = getattr(self.pipe, 'text_encoder_2', None)
                if text_enc_2 is None:
                    print("Warning: Base pipe has no text_encoder_2. Refiner might fail.")
                else:
                    print(f"DEBUG: Sharing text_encoder_2 from Base: {type(text_enc_2)}")

                # Refiner ONLY uses text_encoder_2 (CLIP ViT-G/14 with projection).
                # It does NOT use text_encoder (CLIP ViT-L/14).
                # Sharing text_encoder from Base causes dimension mismatch (1280 vs 768 pooled).
                # Let from_single_file handle text_encoder internally or leave it unused.
                
                self.refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
                     model_paths.SDXL_REFINER,
                     # Do NOT share text_encoder - Refiner doesn't use it and dimensions differ
                     text_encoder_2=text_enc_2,  # Only share text_encoder_2
                     tokenizer_2=getattr(self.pipe, 'tokenizer_2', None),
                     vae=self.pipe.vae,
                     torch_dtype=self.dtype,
                     use_safetensors=True
                )
                print("DEBUG: Refiner Pipeline Initialized.")
             else:
                print("Loading Refiner from Repo ID...")
                self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                     "diffusers/stable-diffusion-xl-refiner-1.0",
                     text_encoder_2=getattr(self.pipe, 'text_encoder_2', None),
                     tokenizer_2=getattr(self.pipe, 'tokenizer_2', None),
                     vae=self.pipe.vae,
                     torch_dtype=self.dtype,
                     use_safetensors=True
                )

             print("DEBUG: Checking Refiner Components before .to()...")
             for name, module in self.refiner.components.items():
                 if module is None:
                     print(f"   - {name}: None")
                 else:
                     print(f"   - {name}: {type(module)}")

             print("DEBUG: Moving Refiner UNet to Device (Skipping full pipeline .to() to protect shared components)...")
             if hasattr(self.refiner, 'unet') and self.refiner.unet:
                 self.refiner.unet.to(self.device, self.dtype)
             
             # Also ensure scheduler is compatible if needed, but it's not on device.
             print(">>> Refiner Loaded.")
        except Exception as e:
             print(f"Failed to load Refiner: {e}")
             import traceback
             traceback.print_exc()
             self.refiner = None

    def _load_sdxl_pipeline(self):
        print(">>> Loading SDXL Inpaint Pipeline (with ControlNet)...")
        
        # 1. Load ControlNet Depth (SDXL)
        try:
            controlnet = ControlNetModel.from_pretrained(
                model_paths.CONTROLNET_DEPTH,
                torch_dtype=self.dtype,
                use_safetensors=True
            )
            print(">>> ControlNet Depth (SDXL) loaded.")
        except Exception as e:
             print(f"Error loading ControlNet (trying remote if local fails): {e}")
             controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0",
                torch_dtype=self.dtype
             )

        # 2. Load SDXL Inpaint Pipeline
        # We need StableDiffusionXLControlNetInpaintPipeline
        try:
            if os.path.exists(model_paths.SDXL_BASE):
                print(f"Loading SDXL from local: {model_paths.SDXL_BASE}")
                self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                    model_paths.SDXL_BASE,
                    controlnet=controlnet,
                    torch_dtype=self.dtype,
                    use_safetensors=True
                )
            else:
                print(f"Local SDXL not found. Loading from Repo: {model_paths.SDXL_REPO}")
                self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                    model_paths.SDXL_REPO, # "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
                    controlnet=controlnet,
                    torch_dtype=self.dtype,
                    use_safetensors=True
                )
        except Exception as e:
            print(f"Failed to load SDXL Pipe: {e}")
            raise e

        # 3. Load IP-Adapter (SDXL)
        print(f">>> Loading IP-Adapter SDXL from {model_paths.IP_ADAPTER_SDXL_PATH}")
        self.ip_adapter_loaded = False
        
        try:
            # Note: For SDXL IP-Adapter, we need image encoder to encode reference images.
            # load_ip_adapter needs the image_encoder_folder to properly initialize.
            
            # Check if local image encoder exists
            if os.path.exists(model_paths.IMAGE_ENCODER_PATH):
                print(f">>> Using local image encoder: {model_paths.IMAGE_ENCODER_PATH}")
                image_encoder_folder = model_paths.IMAGE_ENCODER_PATH
            else:
                # Use HuggingFace repo for image encoder (CLIP ViT-H for SDXL IP-Adapter Plus)
                print(">>> Downloading image encoder from HuggingFace...")
                image_encoder_folder = "h94/IP-Adapter"  # Will auto-download
            
            self.pipe.load_ip_adapter(
                model_paths.IP_ADAPTER_PLUS_HAIR, 
                subfolder="", 
                weight_name="ip-adapter-plus_sdxl_vit-h.bin",
                image_encoder_folder=image_encoder_folder
            )
            self.ip_adapter_loaded = True
            print(">>> IP-Adapter SDXL loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load IP-Adapter SDXL ({e}).")

        self.pipe.to(self.device, self.dtype)
        print(">>> SDXL Pipeline Loaded Successfully.")

    def _load_sd15_pipeline(self):
        print(">>> Loading SD1.5 Inpaint Pipeline (AutoPipeline)...")
        from diffusers import AutoPipelineForInpainting
        
        # 1. Load SD1.5 Inpaint Pipe
        variant = "fp16" if self.dtype == torch.float16 else None
        
        # ... (simplified loading logic reused from before or kept minimal)
        self.pipe = AutoPipelineForInpainting.from_pretrained(
             model_paths.SD15_BASE,
             torch_dtype=self.dtype,
             variant=variant
        )
        
        # 2. Load IP-Adapter
        self.ip_adapter_loaded = False
        try:
            self.pipe.load_ip_adapter(
                model_paths.IP_ADAPTER_PLUS_HAIR, 
                subfolder="", 
                weight_name="ip-adapter-plus_sd15.bin"
            )
            self.ip_adapter_loaded = True
        except Exception:
            pass
            
        self.pipe.to(self.device, self.dtype)
        self.use_sdxl = False
        print(">>> SD1.5 Pipeline Loaded Successfully.")

    def generate(
        self, 
        base_image: Image.Image, 
        mask_image: Image.Image, 
        control_image: Image.Image, 
        ref_hair_image: Image.Image,
        prompt: str = "high quality, realistic hairstyle",
        negative_prompt: str = "blur, low quality, distortion",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 0.5,
        ip_adapter_scale: float = 0.6,
        use_refiner: bool = False
    ):
        """
        Thực hiện Inpainting thay tóc (SDXL 1024x1024 hoặc SD1.5 512x512).
        """
        # Resize inputs based on model
        target_size = (1024, 1024) if self.use_sdxl else (512, 512)
        
        image = base_image.resize(target_size, Image.LANCZOS)
        mask = mask_image.resize(target_size, Image.NEAREST)
        ref_hair = ref_hair_image.resize(target_size, Image.LANCZOS)
        
        # Generator seed
        generator = torch.Generator(self.device).manual_seed(42)
        
        # Set IP Adapter Scale
        try:
            if self.ip_adapter_loaded:
                self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        except:
             pass

        if self.use_sdxl:
            # ControlNet requires Control Image (Depth)
            # Resize control image to match target
            if control_image:
                 control = control_image.resize(target_size, Image.BILINEAR)
            else:
                 # Should not happen if tasks.py provides it, but as fallback creates blank
                 control = Image.new("RGB", target_size, (0, 0, 0))

            print(f"Running SDXL Inference (Refiner: {use_refiner})...")
            
            # 1. Run Base (High noise fraction if using refiner)
            extra_args = {}
            if use_refiner:
                 print("DEBUG: Calling _load_refiner()...")
                 self._load_refiner()
                 print("DEBUG: _load_refiner() returned.")
                 
                 if self.refiner:
                      print("DEBUG: Refiner object exists. Setting output_type=latent")
                      extra_args["output_type"] = "latent"
                      extra_args["denoising_end"] = 0.8
            
            # Prepare arguments dynamically
            input_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,       
                "mask_image": mask,   
                "control_image": control, 
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "controlnet_conditioning_scale": controlnet_scale, 
                "strength": 0.99,     
                "generator": generator
            }
            
            # Only add IP Adapter if loaded
            if self.ip_adapter_loaded:
                 input_args["ip_adapter_image"] = ref_hair
            
            # Add Refiner args
            input_args.update(extra_args)

            print(f"Debug: Arguments prepared. Keys: {list(input_args.keys())}")
            print("Debug: Starting Base Pipeline Inference...")
            
            try:
                latents = self.pipe(**input_args).images
                print("Debug: Base Inference Complete. Latents acquired.")
            except Exception as e:
                print(f"Error in SDXL Base generation: {e}")
                import traceback
                traceback.print_exc()
                raise e
            
            if use_refiner and self.refiner:
                 print(">>> Running Refiner...")
                 try:
                     result = self.refiner(
                         prompt=prompt,
                         negative_prompt=negative_prompt,
                         image=latents[0], # Pass latents from base
                         # mask_image=mask, # Refiner (Img2Img) doesn't use mask in this flow
                         num_inference_steps=num_inference_steps,
                         denoising_start=0.8,
                         strength=0.99, # Refiner strength
                         generator=generator
                     ).images[0]
                 except Exception as e:
                     print(f"Error in SDXL Refiner generation: {e}")
                     import traceback
                     traceback.print_exc()
                     # Fallback to base result (convert latent to image if needed)
                     # Note: If output_type was latent, we need to decode it manually to fallback
                     # But for now let's raise
                     raise e
            else:
                 # Standard output
                 if isinstance(latents, list):
                     result = latents[0]
                 else:
                     result = latents # Should handle if it returns image directly depending on version
        else:
            # SD1.5 Logic (Fallback)
            print("Running SD1.5 Inference...")
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
            if self.ip_adapter_loaded:
                result = self.pipe(**inpainting_args, ip_adapter_image=ref_hair).images[0]
            else:
                result = self.pipe(**inpainting_args).images[0]
        
        return result
