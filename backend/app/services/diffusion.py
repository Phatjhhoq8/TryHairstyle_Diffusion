
import os
import torch

import backend.app.utils.torch_patch

import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL
)
from transformers import CLIPVisionModelWithProjection

from backend.app.config import model_paths, settings

class HairDiffusionService:
    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.dtype = torch.float16 if "cuda" in settings.DEVICE else torch.float32
        
        print(f"Initializing HairDiffusionService (Device: {self.device}, Dtype: {self.dtype})")
        
        try:
            self._load_sdxl_pipeline()
            
        except Exception as e:
            print(f"Error loading Pipeline: {e}")
            raise e



    def _load_refiner(self):
        """Tải refiner chỉ khi cần (lazy load)"""
        if hasattr(self, 'refiner') and self.refiner is not None:
             return
             
        print(">>> Loading SDXL Refiner for High Quality Mode...")
        
        try:
             # Kiểm tra xem file nội bộ có tồn tại không
             if os.path.exists(model_paths.SDXL_REFINER):
                print(f"Loading Refiner from single file: {model_paths.SDXL_REFINER}")
                
                # Kiểm tra các thành phần của Base Pipe
                text_enc_2 = getattr(self.pipe, 'text_encoder_2', None)
                if text_enc_2 is None:
                    print("Warning: Base pipe has no text_encoder_2. Refiner might fail.")
                else:
                    print(f"DEBUG: Sharing text_encoder_2 from Base: {type(text_enc_2)}")

                # Refiner CHỈ sử dụng text_encoder_2 (CLIP ViT-G/14 với projection).
                # Nó KHÔNG sử dụng text_encoder (CLIP ViT-L/14).
                # Việc chia sẻ text_encoder từ Base gây ra lỗi sai lệch kích thước (1280 vs 768 pooled).
                # Để from_single_file tự xử lý text_encoder bên trong hoặc để trống.
                
                self.refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
                     model_paths.SDXL_REFINER,
                     text_encoder_2=text_enc_2,  # Only share text_encoder_2
                     tokenizer_2=getattr(self.pipe, 'tokenizer_2', None),
                     vae=self.pipe.vae,
                     torch_dtype=torch.float32,
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
                     torch_dtype=torch.float32,
                     use_safetensors=True
                )

             print("DEBUG: Checking Refiner Components before .to()...")
             for name, module in self.refiner.components.items():
                 if module is None:
                     print(f"   - {name}: None")
                 else:
                     print(f"   - {name}: {type(module)}")

             print("DEBUG: Moving Refiner Pipeline to Device...")
             self.refiner.to(self.device, self.dtype)
             
             # Cũng đảm bảo scheduler tương thích nếu cần, nhưng nó không nằm trên thiết bị.
             print(">>> Refiner Loaded.")
        except Exception as e:
             print(f"Failed to load Refiner: {e}")
             import traceback
             traceback.print_exc()
             self.refiner = None

    def _load_sdxl_pipeline(self):
        print(">>> Loading SDXL Inpaint Pipeline (with ControlNet)...")
        
        # 1. Tải ControlNet Depth (SDXL)
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

        # 2. Tải SDXL Inpaint Pipeline
        # Chúng ta cần StableDiffusionXLControlNetInpaintPipeline
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

        # 3. Tải IP-Adapter (SDXL)
        print(f">>> Loading IP-Adapter SDXL from {model_paths.IP_ADAPTER_SDXL_PATH}")
        self.ip_adapter_loaded = False
        
        try:
            # Lưu ý: Đối với SDXL IP-Adapter, ta cần image encoder để mã hóa ảnh tham chiếu.
            # load_ip_adapter cần image_encoder_folder để khởi tạo đúng cách.
            
            # Kiểm tra xem image encoder nội bộ có tồn tại không
            if os.path.exists(model_paths.IMAGE_ENCODER_PATH):
                print(f">>> Using local image encoder: {model_paths.IMAGE_ENCODER_PATH}")
                image_encoder_folder = model_paths.IMAGE_ENCODER_PATH
            else:
                # Sử dụng HuggingFace repo cho image encoder (CLIP ViT-H cho SDXL IP-Adapter Plus)
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
        Thực hiện Inpainting thay tóc (SDXL 1024x1024).
        """
        # Thay đổi kích thước đầu vào dựa trên model
        target_size = (1024, 1024)
        
        image = base_image.resize(target_size, Image.LANCZOS)
        mask = mask_image.resize(target_size, Image.NEAREST)
        ref_hair = ref_hair_image.resize(target_size, Image.LANCZOS)
        
        # Generator seed (hạt giống sinh ngẫu nhiên)
        generator = torch.Generator(self.device).manual_seed(42)
        
        # Thiết lập tỷ lệ IP Adapter
        try:
            if self.ip_adapter_loaded:
                self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        except:
             pass

        # ControlNet yêu cầu Control Image (Depth)
        # Thay đổi kích thước control image để khớp với mục tiêu
        if control_image:
             control = control_image.resize(target_size, Image.BILINEAR)
        else:
             # Không nên xảy ra nếu tasks.py đã cung cấp, nhưng tạo ảnh trắng để dự phòng
             control = Image.new("RGB", target_size, (0, 0, 0))

        print(f"Running SDXL Inference (Refiner: {use_refiner})...")
        
        # 1. Chạy Base (High noise fraction nếu dùng refiner)
        extra_args = {}
        if use_refiner:
             print("DEBUG: Calling _load_refiner()...")
             self._load_refiner()
             print("DEBUG: _load_refiner() returned.")
             
             if self.refiner:
                  print("DEBUG: Refiner object exists. Setting output_type=latent")
                  extra_args["output_type"] = "latent"
                  extra_args["denoising_end"] = 0.8
        
        # Chuẩn bị các tham số động
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
        
        # Chỉ thêm IP Adapter nếu đã tải
        if self.ip_adapter_loaded:
             input_args["ip_adapter_image"] = ref_hair
        
        # Thêm tham số Refiner
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
                    image=latents, # Truyền latents (4D) cho Refiner
                    # mask_image=mask, 
                    num_inference_steps=num_inference_steps,
                    denoising_start=0.8,
                    strength=0.99, 
                    generator=generator
                ).images[0]
             except Exception as e:
                 print(f"Error in SDXL Refiner generation: {e}")
                 import traceback
                 traceback.print_exc()
                 # Dự phòng về kết quả base (chuyển đổi latent sang ảnh nếu cần)
                 # Lưu ý: Nếu output_type là latent, ta cần giải mã thủ công để fallback
                 # Nhưng hiện tại hãy raise lỗi
                 raise e
        else:
             # Output chuẩn
             if isinstance(latents, list):
                 result = latents[0]
             else:
                 result = latents # Cần xử lý nếu nó trả về ảnh trực tiếp tùy thuộc vào phiên bản
        
        return result
