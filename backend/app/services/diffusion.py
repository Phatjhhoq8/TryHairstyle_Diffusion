
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
        
        # Tìm file injector cùng thư mục với UNet checkpoint
        # Ưu tiên: injector.safetensors (production) > injector_best > injector_latest > injector_backup
        ckpt_dir = os.path.dirname(checkpoint_path)
        inj_candidates = [
            os.path.join(ckpt_dir, "injector.safetensors"),
            os.path.join(ckpt_dir, "injector_best.safetensors"),
            os.path.join(ckpt_dir, "injector_latest.safetensors"),
            os.path.join(ckpt_dir, "injector_backup.safetensors"),
        ]
        inj_path = next((p for p in inj_candidates if os.path.exists(p)), None)
        
        if inj_path:
            print(f"  → Loading injector: {inj_path}")
            inj_dict = load_file(inj_path)
            self.injector.load_state_dict(inj_dict, strict=False)
        else:
            print(f"  ⚠️ Không tìm thấy injector trong {ckpt_dir}! Style/Identity conditioning sẽ không hoạt động.")
        self.injector.eval()
        
        print("  ✅ Custom Model loaded successfully!")
        
        # HairTextureEncoder (Stage 1) cho style embedding — CÙNG model đã dùng lúc training
        # Tránh dùng CLIP vì distribution khác hoàn toàn với ResNet50 2048-d đã train
        self.texture_encoder = None
        self.style_transform = None
        try:
            from backend.training.models.texture_encoder import HairTextureEncoder
            from torchvision import transforms as T
            
            self.texture_encoder = HairTextureEncoder(pretrained=False).to(self.device).eval()
            
            # Tìm checkpoint Stage 1 (ưu tiên best > latest)
            ckpt_dir = os.path.join(str(BACKEND_DIR), "training", "checkpoints")
            tex_best = os.path.join(ckpt_dir, "texture_encoder_best.safetensors")
            tex_latest = os.path.join(ckpt_dir, "texture_encoder_latest.safetensors")
            tex_ckpt = tex_best if os.path.exists(tex_best) else tex_latest
            
            if os.path.exists(tex_ckpt):
                tex_state = load_file(tex_ckpt)
                self.texture_encoder.load_state_dict(tex_state, strict=False)
                self.texture_encoder.requires_grad_(False)
                print(f"  ✅ HairTextureEncoder loaded từ {os.path.basename(tex_ckpt)}")
            else:
                print(f"  ⚠️ Không tìm thấy checkpoint Stage 1, style embedding sẽ dùng zeros fallback")
                self.texture_encoder = None
            
            # Transform chuẩn ImageNet (khớp với training Stage 1)
            self.style_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"  ⚠️ HairTextureEncoder load failed (style embedding sẽ dùng zeros fallback): {e}")

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
        # 512×512 để khớp với resolution training (train_stage2.py --resolution 512)
        # Output sẽ được upscale về kích thước gốc bởi tasks.py sau inference
        target_size = (512, 512)
        
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
            
        # Transform
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        gt_tensor = img_transform(target_pil).unsqueeze(0).to(self.device)

        # Mask tensor — chỉ khai báo 1 lần, tránh biến shadow
        mask_np = np.array(mask_pil)
        if len(mask_np.shape) == 3: mask_np = mask_np[:, :, 0]
        mask_float = (mask_np / 255.0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_float[np.newaxis, np.newaxis, ...]).to(self.device)

        # Tạo masked_image = gt × (1 - mask) — vùng tóc = 0, phần còn lại giữ nguyên
        # Đây là convention chuẩn SDXL Inpainting, khớp với training pipeline
        masks_pixel = mask_tensor.expand_as(gt_tensor)  # (1, 3, H, W)
        masked_tensor = gt_tensor * (1.0 - masks_pixel)

        # Encode text prompt (conditional)
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

        # Encode negative/unconditional prompt (cho CFG)
        # CFG: noise_pred = uncond + guidance_scale * (cond - uncond)
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            neg_prompt = negative_prompt if negative_prompt else ""
            print(f"  → Encoding negative prompt cho CFG (scale={guidance_scale}): '{neg_prompt}'")
            with torch.no_grad():
                neg_tokens_1 = self.tokenizer_1(neg_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(self.device)
                neg_tokens_2 = self.tokenizer_2(neg_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(self.device)

                neg_enc_1 = self.text_encoder_1(neg_tokens_1, output_hidden_states=True)
                neg_enc_2 = self.text_encoder_2(neg_tokens_2, output_hidden_states=True)

                neg_hidden_1 = neg_enc_1.hidden_states[-2]
                neg_hidden_2 = neg_enc_2.hidden_states[-2]

                neg_prompt_embeds = torch.cat([neg_hidden_1, neg_hidden_2], dim=-1).float()
                neg_pooled_embeds = neg_enc_2.text_embeds.float()

        # VAE encode
        print("  → VAE encoding images...")
        with torch.no_grad():
            gt_latents = self.vae.encode(gt_tensor.to(self.vae.dtype)).latent_dist.sample() * self.vae_scale_factor
            masked_latents = self.vae.encode(masked_tensor.to(self.vae.dtype)).latent_dist.sample() * self.vae_scale_factor
            gt_latents = gt_latents.float()
            masked_latents = masked_latents.float()

        mask_down = F.interpolate(mask_tensor, size=gt_latents.shape[-2:], mode='nearest')
        
        # Identity + Style embeddings
        id_embed_t = torch.from_numpy(id_embed).unsqueeze(0).float().to(self.device)
        style_embed_t = torch.zeros(1, 2048).to(self.device)  # 2048-d khớp Stage 1 Texture Encoder
        
        try:
             if self.texture_encoder is not None and self.style_transform is not None:
                 # Crop vùng tóc từ ảnh reference (khớp với training pipeline)
                 # Training dùng hair-only crop (polygon mask) → resize 128×128
                 # Inference cần tái tạo flow tương tự bằng SegFormer
                 ref_for_style = ref_pil
                 try:
                     from backend.app.services.mask import SegmentationService
                     seg = SegmentationService()
                     ref_cv2 = cv2.cvtColor(np.array(ref_pil), cv2.COLOR_RGB2BGR)
                     parsing = seg.get_parsing(ref_cv2)
                     if parsing is not None:
                         # Hair classes = {13, 14} (tóc + nón) — khớp với mask.py
                         hair_mask = np.zeros_like(parsing, dtype=np.uint8)
                         for cls in [13, 14]:
                             hair_mask[parsing == cls] = 255
                         
                         hair_pixels = np.count_nonzero(hair_mask)
                         if hair_pixels > 500:
                             # Crop bounding box vùng tóc
                             ys, xs = np.where(hair_mask > 0)
                             y1, y2 = np.min(ys), np.max(ys)
                             x1, x2 = np.min(xs), np.max(xs)
                             
                             # Crop hair-only từ ảnh RGB gốc (nền = đen)
                             ref_rgb = np.array(ref_pil)
                             hair_only = ref_rgb.copy()
                             hair_only[hair_mask == 0] = 0  # Xóa nền + mặt
                             hair_crop = hair_only[y1:y2+1, x1:x2+1]
                             ref_for_style = Image.fromarray(hair_crop)
                             print(f"  ✅ Hair cropped from reference: {hair_crop.shape[1]}×{hair_crop.shape[0]} ({hair_pixels} hair pixels)")
                         else:
                             print(f"  ⚠️ Hair mask quá nhỏ ({hair_pixels}px), dùng full ảnh reference")
                     del seg  # Free SegFormer ngay sau khi dùng xong
                     torch.cuda.empty_cache()
                 except Exception as e:
                     print(f"  ⚠️ SegFormer crop failed ({e}), dùng full ảnh reference")
                 
                 ref_np = np.array(ref_for_style.resize((128, 128)))
                 ref_tensor = self.style_transform(ref_np).unsqueeze(0).to(self.device)
                 with torch.no_grad():
                     embed, _, _ = self.texture_encoder(ref_tensor)  # (1, 2048) — khớp distribution training
                     style_embed_t = embed.float()
                 print(f"  ✅ Style embedding extracted via HairTextureEncoder (2048-d)")
        except Exception as e:
             print(f"Warning: Style encoding failed: {e}")

        # Conditional encoder hidden states
        injected_conds = self.injector.inject_conditioning(style_embed_t, id_embed_t)
        encoder_hidden_states = torch.cat([prompt_embeds, injected_conds], dim=1)

        # Unconditional encoder hidden states (cho CFG)
        if do_cfg:
            # Dùng zero conditioning cho style/identity trong unconditional pass
            uncond_style = torch.zeros_like(style_embed_t)
            uncond_id = torch.zeros_like(id_embed_t)
            uncond_injected = self.injector.inject_conditioning(uncond_style, uncond_id)
            uncond_encoder_hidden_states = torch.cat([neg_prompt_embeds, uncond_injected], dim=1)

        # time_ids = 1024 (SDXL native) để khớp với pretraining distribution của SDXL
        # Dù ảnh chạy ở 512×512, SDXL conditioning nên ở 1024 để nhất quán với training
        time_ids = torch.tensor([1024, 1024, 0, 0, 1024, 1024], dtype=torch.float32).unsqueeze(0).to(self.device)

        added_cond_kwargs = {
            "text_embeds": pooled_embeds,
            "time_ids": time_ids
        }
        if do_cfg:
            uncond_added_cond_kwargs = {
                "text_embeds": neg_pooled_embeds,
                "time_ids": time_ids
            }

        print(f"  → Bắt đầu Denoising ({num_steps} steps, CFG={'ON scale='+str(guidance_scale) if do_cfg else 'OFF'})...")
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        latents = torch.randn_like(gt_latents)

        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            with torch.amp.autocast('cuda', dtype=torch.float16):
                with torch.no_grad():
                    # Forward pass conditional (với prompt + style + identity)
                    noise_pred_cond = self.unet(
                        noisy_latents=latents,
                        masked_latents=masked_latents,
                        mask=mask_down,
                        timestep=t.unsqueeze(0),
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs
                    )
                    
                    if do_cfg:
                        # Forward pass unconditional (negative prompt + zero conditioning)
                        noise_pred_uncond = self.unet(
                            noisy_latents=latents,
                            masked_latents=masked_latents,
                            mask=mask_down,
                            timestep=t.unsqueeze(0),
                            encoder_hidden_states=uncond_encoder_hidden_states,
                            added_cond_kwargs=uncond_added_cond_kwargs
                        )
                        # Classifier-Free Guidance: đẩy mạnh đặc trưng theo prompt
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_cond

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            latents = latents * mask_down + masked_latents * (1 - mask_down)

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
