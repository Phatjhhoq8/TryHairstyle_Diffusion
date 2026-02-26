
import os
import torch

import backend.app.utils.torch_patch

import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    AutoencoderKL
)
from transformers import CLIPVisionModelWithProjection

from diffusers import DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from backend.app.config import model_paths, settings, BACKEND_DIR
from safetensors.torch import load_file
import torch.nn.functional as F
from tqdm import tqdm

class HairDiffusionService:
    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.dtype = torch.float16 if "cuda" in settings.DEVICE else torch.float32
        
        print(f"Initializing HairDiffusionService (Device: {self.device}, Dtype: {self.dtype})")
        self.is_custom_pipeline = False
        
        try:
            self._init_pipeline()
            
        except Exception as e:
            print(f"Error loading Pipeline: {e}")
            raise e

    def _init_pipeline(self):
        """Khởi tạo pipeline dựa trên file weights"""
        # Kiểm tra file weight custom
        best_ckpt = model_paths.CUSTOM_INPAINTING_MODEL
        
        if os.path.exists(best_ckpt):
            print(f">>> Found custom trained model at {best_ckpt}. Loading Custom Pipeline...")
            self.is_custom_pipeline = True
            self._load_custom_pipeline(best_ckpt)
        else:
            print(f">>> Custom model not found. Fallback to Standard SDXL ControlNet Pipeline...")
            self.is_custom_pipeline = False
            self._load_sdxl_pipeline()

    def _load_custom_pipeline(self, checkpoint_path):
        """Tải Custom 9-Channel UNet Pipeline"""
        print(">>> Loading Custom 9-Channel UNet Pipeline for Inpainting...")
        
        # We need the custom network definitions
        # Make sure they are available or imported correctly
        import sys
        
        try:
            from backend.training.models.stage2_unet import HairInpaintingUNet, CrossAttentionInjector
        except ImportError:
            # Fallback import path if running from deep inside
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "stage2_unet", 
                os.path.join(str(BACKEND_DIR), "training", "models", "stage2_unet.py")
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["backend.training.models.stage2_unet"] = module
            spec.loader.exec_module(module)
            from backend.training.models.stage2_unet import HairInpaintingUNet, CrossAttentionInjector
            
        # VAE
        print("  → Loading VAE (fp16)...")
        self.vae = AutoencoderKL.from_pretrained(
            model_paths.SDXL_BASE, subfolder="vae", torch_dtype=torch.float16
        ).to(self.device).eval()
        self.vae.enable_slicing()
        self.vae_scale_factor = self.vae.config.scaling_factor
        
        # Noise Scheduler
        print("  → Loading DDPMScheduler...")
        self.scheduler = DDPMScheduler.from_pretrained(model_paths.SDXL_BASE, subfolder="scheduler")
        
        # Text Encoders
        print("  → Loading CLIP Text Encoders...")
        self.tokenizer_1 = CLIPTokenizer.from_pretrained(model_paths.SDXL_BASE, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_paths.SDXL_BASE, subfolder="tokenizer_2")
        
        self.text_encoder_1 = CLIPTextModel.from_pretrained(
            model_paths.SDXL_BASE, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(self.device).eval()
        
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_paths.SDXL_BASE, subfolder="text_encoder_2", torch_dtype=torch.float16
        ).to(self.device).eval()
        
        # UNet
        print("  → Loading UNet 9-channel...")
        self.unet = HairInpaintingUNet().to(self.device)
        
        print(f"  → Loading checkpoint: {checkpoint_path}")
        state_dict = load_file(checkpoint_path)
        self.unet.load_state_dict(state_dict, strict=False)
        self.unet.eval()
        
        # Injector
        self.injector = CrossAttentionInjector(self.unet.unet).to(self.device)
        inj_path = checkpoint_path.replace("deep_hair_v1_best", "injector_best") \
                                  .replace("deep_hair_v1_latest", "injector_latest") \
                                  .replace("stage2_", "injector_")
        if os.path.exists(inj_path):
            print(f"  → Loading injector: {inj_path}")
            inj_dict = load_file(inj_path)
            self.injector.load_state_dict(inj_dict, strict=False)
        self.injector.eval()
        
        print("  ✅ Custom Model loaded successfully!")
        
        # CLIP model cho style embedding (cache 1 lần, không load mỗi lần generate)
        self.clip_model = None
        self.clip_preprocess = None
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
            self.clip_model.eval()
            print("  ✅ CLIP ViT-L/14 loaded cho style embedding")
        except Exception as e:
            print(f"  ⚠️ CLIP load failed (style embedding sẽ dùng zeros fallback): {e}")

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
        print(f">>> Loading IP-Adapter SDXL...")
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
        ip_adapter_scale: float = 0.6
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
        except Exception as e:
             print(f"Warning: Failed to set IP-adapter scale: {e}")

        # ControlNet yêu cầu Control Image (Depth)
        # Thay đổi kích thước control image để khớp với mục tiêu
        if control_image:
             control = control_image.resize(target_size, Image.BILINEAR)
        else:
             # Không nên xảy ra nếu tasks.py đã cung cấp, nhưng tạo ảnh trắng để dự phòng
             control = Image.new("RGB", target_size, (0, 0, 0))

        if self.is_custom_pipeline:
            print(f"Running Custom Inpainting Inference...")
            return self._generate_custom(
                image, mask, ref_hair, control, 
                prompt, negative_prompt, num_inference_steps, guidance_scale, target_size
            )
        else:
            print(f"Running SDXL Standard Inference...")
            return self._generate_standard(
                image, mask, ref_hair, control, 
                prompt, negative_prompt, num_inference_steps, guidance_scale, controlnet_scale, ip_adapter_scale, target_size
            )
            
    def _generate_custom(self, target_pil, mask_pil, ref_pil, control_pil, prompt, negative_prompt, num_steps, guidance_scale, target_size):
        from torchvision import transforms
        
        # Identity Embeddings via InsightFace (should be injected outside or just zeroed if not available)
        id_embed = np.zeros(512, dtype=np.float32) 
        try:
            from backend.app.services.embedder import TrainingEmbedder
            emb_service = TrainingEmbedder()
            faces = emb_service.insightApp.get(cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR))
            if faces:
                embedding = faces[0].embedding
                id_embed = embedding / (np.linalg.norm(embedding) + 1e-8)
        except Exception as e:
            print(f"Warning: Failed to extract ID embedding for Custom Pipeline: {e}")
            
        print("  → Tạo bald image (inpaint)...")
        img = np.array(target_pil)
        mask = np.array(mask_pil)
        if len(mask.shape) == 3:
             mask = mask[:, :, 0]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bald_bgr = cv2.inpaint(img_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        bald_pil = Image.fromarray(cv2.cvtColor(bald_bgr, cv2.COLOR_BGR2RGB))

        # Transform
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        gt_tensor = img_transform(target_pil).unsqueeze(0).to(self.device)
        bald_tensor = img_transform(bald_pil).unsqueeze(0).to(self.device)

        # Mask tensor
        mask_np = np.array(mask_pil)
        if len(mask_np.shape) == 3: mask_np = mask_np[:, :, 0]
        mask_float = (mask_np / 255.0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_float[np.newaxis, np.newaxis, ...]).to(self.device)

        # Encode text prompt
        print(f"  → Encoding text prompt: '{prompt}'")
        with torch.no_grad():
            tokens_1 = self.tokenizer_1(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(self.device)
            tokens_2 = self.tokenizer_2(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(self.device)

            enc_1 = self.text_encoder_1(tokens_1, output_hidden_states=True)
            enc_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)

            hidden_1 = enc_1.hidden_states[-2]
            hidden_2 = enc_2.hidden_states[-2]

            prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1).float()
            pooled_embeds = enc_2.text_embeds.float()

        # VAE encode
        print("  → VAE encoding images...")
        with torch.no_grad():
            gt_latents = self.vae.encode(gt_tensor.to(self.vae.dtype)).latent_dist.sample() * self.vae_scale_factor
            bald_latents = self.vae.encode(bald_tensor.to(self.vae.dtype)).latent_dist.sample() * self.vae_scale_factor
            gt_latents = gt_latents.float()
            bald_latents = bald_latents.float()

        mask_down = F.interpolate(mask_tensor, size=gt_latents.shape[-2:], mode='nearest')
        
        # Identity + Style embeddings
        id_embed_t = torch.from_numpy(id_embed).unsqueeze(0).float().to(self.device)
        style_embed_t = torch.zeros(1, 1024).to(self.device)  
        
        try:
             if self.clip_model is not None and self.clip_preprocess is not None:
                 img_ref = self.clip_preprocess(ref_pil).unsqueeze(0).to(self.device)
                 with torch.no_grad():
                     style_embed_t = self.clip_model.encode_image(img_ref).float()
        except Exception as e:
             print(f"Warning: CLIP style encoding failed: {e}")

        injected_conds = self.injector.inject_conditioning(style_embed_t, id_embed_t)
        encoder_hidden_states = torch.cat([prompt_embeds, injected_conds], dim=1)

        time_ids = torch.tensor([1024, 1024, 0, 0, 1024, 1024], dtype=torch.float32).unsqueeze(0).to(self.device)

        added_cond_kwargs = {
            "text_embeds": pooled_embeds,
            "time_ids": time_ids
        }

        print(f"  → Bắt đầu Denoising ({num_steps} steps)...")
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        latents = torch.randn_like(gt_latents)

        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            with torch.amp.autocast('cuda', dtype=torch.float16):
                with torch.no_grad():
                    noise_pred = self.unet(
                        noisy_latents=latents,
                        bald_latents=bald_latents,
                        mask=mask_down,
                        timestep=t.unsqueeze(0),
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs
                    )

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            latents = latents * mask_down + bald_latents * (1 - mask_down)

        print("  → VAE decoding...")
        with torch.no_grad():
            latents_decode = (latents / self.vae_scale_factor).to(self.vae.dtype)
            decoded = self.vae.decode(latents_decode).sample
            decoded = decoded.float()

        output_img = ((decoded.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255)
        output_img = np.clip(output_img, 0, 255).astype(np.uint8)

        target_np = np.array(target_pil.resize(target_size))
        mask_3ch = np.stack([mask_float] * 3, axis=-1)
        mask_3ch = cv2.resize(mask_3ch, target_size)

        mask_blur = cv2.GaussianBlur(mask_3ch, (21, 21), 10)
        final = (output_img * mask_blur + target_np * (1 - mask_blur)).astype(np.uint8)

        return Image.fromarray(final)


    def _generate_standard(self, image, mask, ref_hair, control, prompt, negative_prompt, num_inference_steps, guidance_scale, controlnet_scale, ip_adapter_scale, target_size):
        
        # Generator seed (hạt giống sinh ngẫu nhiên)
        generator = torch.Generator(self.device).manual_seed(42)
        
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

        print(f"Debug: Arguments prepared. Keys: {list(input_args.keys())}")
        print("Debug: Starting Base Pipeline Inference...")
        
        try:
            result = self.pipe(**input_args).images
            print("Debug: Base Inference Complete.")
        except Exception as e:
            print(f"Error in SDXL Base generation: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        # Output chuẩn
        if isinstance(result, list):
            return result[0]
        return result
