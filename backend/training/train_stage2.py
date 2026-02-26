import os
import sys
import json
import time
import random
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (tương thích server/Colab)
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from safetensors.torch import save_file

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from backend.app.services.training_utils import setupLogger, getDevice, ensureDir
from backend.training.models.stage2_unet import HairInpaintingUNet, CrossAttentionInjector
from backend.training.models.losses import MaskAwareLoss, IdentityCosineLoss, TextureConsistencyLoss

logger = setupLogger("TrainStage2_Inpainting")
DEVICE = getDevice()

# Đường dẫn SDXL local model
LOCAL_SDXL_PATH = str(PROJECT_DIR / "backend" / "models" / "stable-diffusion" / "sd_xl_inpainting")

# ==============================================================================
# TEXT PROMPT ENCODER — Pre-encode text prompts thành SDXL embeddings
# ==============================================================================

class SDXLTextEncoder:
    """
    Load CLIP Text Encoders (text_encoder + text_encoder_2) của SDXL để encode prompts.
    Sau khi encode xong, giải phóng encoders ra khỏi VRAM để nhường chỗ cho UNet.
    """
    def __init__(self, sdxl_path=LOCAL_SDXL_PATH, device=DEVICE):
        from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
        
        logger.info("Đang load SDXL Text Encoders (tạm thời, chỉ để encode prompts)...")
        
        self.tokenizer = CLIPTokenizer.from_pretrained(sdxl_path, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(sdxl_path, subfolder="tokenizer_2")
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            sdxl_path, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(device).eval()
        
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            sdxl_path, subfolder="text_encoder_2", torch_dtype=torch.float16
        ).to(device).eval()
        
        self.device = device
        logger.info("Text Encoders loaded thành công.")
    
    @torch.no_grad()
    def encode_prompt(self, prompt: str):
        """
        Encode 1 text prompt thành embeddings dùng cho SDXL UNet.
        Returns:
            prompt_embeds: (1, 77, 2048) — concat hidden states từ cả 2 encoders
            pooled_prompt_embeds: (1, 1280) — pooled output từ text_encoder_2
        """
        # Tokenize cho encoder 1 (CLIP ViT-L/14 — 768d hidden)
        tokens_1 = self.tokenizer(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Tokenize cho encoder 2 (CLIP ViT-bigG — 1280d hidden)
        tokens_2 = self.tokenizer_2(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Encode
        enc_out_1 = self.text_encoder(tokens_1, output_hidden_states=True)
        enc_out_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)
        
        # SDXL sử dụng hidden state thứ -2 (penultimate layer)
        hidden_1 = enc_out_1.hidden_states[-2]  # (1, 77, 768)
        hidden_2 = enc_out_2.hidden_states[-2]  # (1, 77, 1280)
        
        # Nối lại thành prompt_embeds (1, 77, 2048)  
        prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1)
        
        # Pooled output từ text_encoder_2 (1, 1280)
        pooled_prompt_embeds = enc_out_2.text_embeds
        
        return prompt_embeds.float().cpu(), pooled_prompt_embeds.float().cpu()
    
    def unload(self):
        """Giải phóng VRAM sau khi encode xong tất cả prompts."""
        del self.text_encoder, self.text_encoder_2
        del self.tokenizer, self.tokenizer_2
        torch.cuda.empty_cache()
        logger.info("Text Encoders đã được giải phóng khỏi VRAM.")

# ==============================================================================
# DATASET
# ==============================================================================

class HairInpaintingDataset(Dataset):
    def __init__(self, data_dir: Path, text_encoder: SDXLTextEncoder = None, 
                 texture_encoder=None, target_size=(512, 512), max_samples: int = 0):
        """
        Args:
            data_dir: Thư mục processed data
            text_encoder: SDXLTextEncoder để pre-encode prompts (None = load từ cache)
            texture_encoder: HairTextureEncoder đã train (None = dùng zeros fallback)
            target_size: Kích thước ảnh đầu vào (512x512 cho tiết kiệm VRAM)
            max_samples: Số lượng samples tối đa (0 = dùng tất cả)
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.metadata = []
        self.prompt_embeds_cache = {}
        
        meta_path = data_dir / "metadata.jsonl"
        if meta_path.exists():
            with open(str(meta_path), "r", encoding="utf-8") as f:
                for line in f:
                    self.metadata.append(json.loads(line.strip()))
        
        # Giới hạn dataset size nếu cần
        if max_samples > 0 and len(self.metadata) > max_samples:
            random.seed(42)  # Reproducible subset
            self.metadata = random.sample(self.metadata, max_samples)
            logger.info(f"📉 Dataset giảm xuống {max_samples} samples (từ {len(self.metadata)} gốc)")
        
        # Pre-encode text prompts nếu Text Encoder được cung cấp
        if text_encoder is not None and len(self.metadata) > 0:
            logger.info(f"Pre-encoding {len(self.metadata)} text prompts...")
            
            # Cache thư mục cho prompt embeddings
            cache_dir = data_dir / "prompt_embeddings"
            ensureDir(str(cache_dir))
            
            for item in tqdm(self.metadata, desc="Encoding Prompts"):
                img_id = item["id"]
                img_id_dashed = img_id.replace("_", "-")
                cache_file = cache_dir / f"{img_id}.pt"
                cache_file_dashed = cache_dir / f"{img_id_dashed}.pt"
                
                if cache_file.exists():
                    # Load từ cache
                    cached = torch.load(str(cache_file), map_location="cpu", weights_only=True)
                    self.prompt_embeds_cache[img_id] = cached
                elif cache_file_dashed.exists():
                    # Load fallback từ dashed cache
                    cached = torch.load(str(cache_file_dashed), map_location="cpu", weights_only=True)
                    self.prompt_embeds_cache[img_id] = cached
                else:
                    # Encode và cache
                    text_prompt = item.get("text_prompt", "hairstyle")
                    if not text_prompt.strip():
                        text_prompt = "hairstyle"
                    
                    # Thêm prefix chất lượng cho SDXL
                    full_prompt = f"high quality, realistic {text_prompt}, detailed hair texture"
                    
                    p_embeds, pooled_embeds = text_encoder.encode_prompt(full_prompt)
                    cached = {
                        "prompt_embeds": p_embeds.squeeze(0),      # (77, 2048)
                        "pooled_prompt_embeds": pooled_embeds.squeeze(0)  # (1280,)
                    }
                    torch.save(cached, str(cache_file))
                    self.prompt_embeds_cache[img_id] = cached
            
            logger.info("Pre-encoding prompts hoàn tất!")
        else:
            # Tải từ cache nếu đã encode trước đó
            cache_dir = data_dir / "prompt_embeddings"
            if cache_dir.exists():
                for item in self.metadata:
                    img_id = item["id"]
                    img_id_dashed = img_id.replace("_", "-")
                    cache_file = cache_dir / f"{img_id}.pt"
                    cache_file_dashed = cache_dir / f"{img_id_dashed}.pt"
                    if cache_file.exists():
                        self.prompt_embeds_cache[img_id] = torch.load(
                            str(cache_file), map_location="cpu", weights_only=True
                        )
                    elif cache_file_dashed.exists():
                        self.prompt_embeds_cache[img_id] = torch.load(
                            str(cache_file_dashed), map_location="cpu", weights_only=True
                        )
                    
        # Pre-extract style embeddings từ Texture Encoder (Stage 1)
        self.style_embeds_cache = {}
        style_cache_dir = data_dir / "style_embeddings_cache"
        
        if texture_encoder is not None:
            ensureDir(str(style_cache_dir))
            logger.info(f"Pre-extracting style embeddings cho {len(self.metadata)} samples...")
            
            style_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            with torch.no_grad():
                for item in tqdm(self.metadata, desc="Extracting Style"):
                    img_id = item["id"]
                    cache_file = style_cache_dir / f"{img_id}.npy"
                    
                    if cache_file.exists():
                        self.style_embeds_cache[img_id] = np.load(str(cache_file))
                    else:
                        style_img_path = data_dir / item["style"]
                        if style_img_path.exists():
                            style_img = cv2.cvtColor(cv2.imread(str(style_img_path)), cv2.COLOR_BGR2RGB)
                            style_img = cv2.resize(style_img, (128, 128))
                            style_tensor = style_transform(style_img).unsqueeze(0).to(DEVICE)
                            embed, _, _ = texture_encoder(style_tensor)
                            embed_np = embed.cpu().numpy().squeeze(0)  # (2048,)
                            np.save(str(cache_file), embed_np)
                            self.style_embeds_cache[img_id] = embed_np
                        else:
                            self.style_embeds_cache[img_id] = np.zeros(2048, dtype=np.float32)
            
            logger.info("✅ Style embeddings extraction hoàn tất!")
        else:
            # Load từ cache nếu đã extract trước đó
            if style_cache_dir.exists():
                for item in self.metadata:
                    img_id = item["id"]
                    cache_file = style_cache_dir / f"{img_id}.npy"
                    if cache_file.exists():
                        self.style_embeds_cache[img_id] = np.load(str(cache_file))
        
        # Transform tensor chuẩn SDXL (-1 to 1)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_id = item["id"]
        
        # Load Ground Truth image từ processed/ground_truth_images/
        gt_dir = self.data_dir / "ground_truth_images"
        img_candidates = list(gt_dir.glob(f"{img_id}.*"))
        if not img_candidates:
            img_candidates = list(gt_dir.glob(f"{img_id.replace('_', '-')}.*"))
        
        # Fallback: tìm trong thư mục dataset gốc (chỉ khi train trên máy local)
        if not img_candidates:
            for search_dir in [PROJECT_DIR / "dataset", 
                               PROJECT_DIR / "backend" / "data" / "dataset" / "khairstyle" / "training" / "images"]:
                if search_dir.exists():
                    img_candidates = list(search_dir.rglob(f"{img_id}.*"))
                    if img_candidates:
                        break
        
        if not img_candidates:
            raise FileNotFoundError(f"Không tìm thấy Ground Truth cho ID: {img_id}")
            
        gt_img = cv2.cvtColor(cv2.imread(str(img_candidates[0])), cv2.COLOR_BGR2RGB)
        bald_img = cv2.cvtColor(cv2.imread(str(self.data_dir / item["bald"])), cv2.COLOR_BGR2RGB)
        
        gt_img = cv2.resize(gt_img, self.target_size)
        bald_img = cv2.resize(bald_img, self.target_size)
        
        # Load Mask (1 Channel) từ kênh Alpha của ảnh hair_only
        hair_only_rgba = cv2.imread(str(self.data_dir / item["hair_only"]), cv2.IMREAD_UNCHANGED)
        hair_only_rgba = cv2.resize(hair_only_rgba, self.target_size)
        if hair_only_rgba.shape[2] == 4:
            mask = (hair_only_rgba[:, :, 3] / 255.0).astype(np.float32)
        else:
            mask = np.zeros((self.target_size[0], self.target_size[1]), dtype=np.float32)
        true_mask = mask[..., np.newaxis]
        
        # Identity (từ AdaFace 512d)
        id_embed = np.load(str(self.data_dir / item["identity"]))
        
        # Text Prompt Embeddings (đã pre-encoded)
        if img_id in self.prompt_embeds_cache:
            cached = self.prompt_embeds_cache[img_id]
            text_embeds = cached["prompt_embeds"]           # (77, 2048)
            pooled_text_embeds = cached["pooled_prompt_embeds"]  # (1280,)
        else:
            # Fallback nếu chưa encode — dùng zero vectors
            logger.warning(f"Prompt embeddings chưa có cho {img_id}, dùng zeros fallback.")
            text_embeds = torch.zeros(77, 2048)
            pooled_text_embeds = torch.zeros(1280)
        
        # Trả về Tensors
        return {
            "image": self.img_transform(gt_img),
            "bald_image": self.img_transform(bald_img),
            "mask": torch.from_numpy(true_mask).permute(2, 0, 1),
            "style_embed": torch.from_numpy(
                self.style_embeds_cache.get(img_id, np.zeros(2048, dtype=np.float32))
            ).float(),  # (2048,) — từ Stage 1 Texture Encoder
            "identity_embed": torch.from_numpy(id_embed).squeeze(0).float(),
            "text_embeds": text_embeds,
            "pooled_text_embeds": pooled_text_embeds,
            "time_ids": torch.tensor([
                self.target_size[0], self.target_size[1],
                0, 0,
                self.target_size[0], self.target_size[1]
            ], dtype=torch.float32)
        }

# ==============================================================================
# TRAINER
# ==============================================================================

class Stage2Trainer:
    """
    Kịch bản huấn luyện Stage 2: Mask-Conditioned Hair Inpainting
    Kết hợp 9-channel UNet, Custom Identity Loss, và Mask-aware Gradient Locking.
    Tối ưu cho GPU 12GB (RTX 3060): AMP + xformers + 8-bit optimizer + gradient accumulation.
    """
    def __init__(self):
        from diffusers import AutoencoderKL, DDPMScheduler
        
        logger.info("Khởi tạo Stage 2 Trainer: SDXL Mask-Conditioned Inpainting")
        
        # 1. Load VAE (frozen, chỉ encode/decode)
        logger.info("  → Loading VAE (fp16, frozen)...")
        self.vae = AutoencoderKL.from_pretrained(
            LOCAL_SDXL_PATH, subfolder="vae", torch_dtype=torch.float16
        ).to(DEVICE).eval()
        self.vae.requires_grad_(False)
        self.vae_scale_factor = self.vae.config.scaling_factor  # 0.13025 cho SDXL
        
        # VAE slicing để giảm peak VRAM khi encode/decode ảnh 1024×1024
        self.vae.enable_slicing()
        
        # 2. Load Noise Scheduler
        logger.info("  → Loading DDPMScheduler...")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            LOCAL_SDXL_PATH, subfolder="scheduler"
        )
        
        # 3. Load UNet 9-channel + CrossAttention Injector
        # Load FP16 trước, rồi convert sang FP32 cho training.
        # GradScaler (AMP) yêu cầu master weights ở FP32 — autocast sẽ tự cast FP16 trong forward pass.
        logger.info("  → Loading UNet 9-channel (fp16 → fp32 for training)...")
        self.unet = HairInpaintingUNet().to(DEVICE).float()  # FP16 → FP32
        self.injector = CrossAttentionInjector(self.unet.unet, style_dim=2048).to(DEVICE).float()  # style=2048 từ Stage 1
        
        # 4. Khởi tạo Loss Functions
        self.mask_aware_loss = MaskAwareLoss(loss_type='l2').to(DEVICE)
        self.identity_loss = IdentityCosineLoss().to(DEVICE)
        self.texture_loss = TextureConsistencyLoss().to(DEVICE)
        
        # 5. Optimizer — 8-bit AdamW để tiết kiệm ~2.5GB VRAM
        train_params = list(self.unet.parameters()) + list(self.injector.parameters())
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                train_params,
                lr=1e-5,
                weight_decay=1e-2
            )
            logger.info("  → Sử dụng 8-bit AdamW (bitsandbytes) — tiết kiệm ~2.5GB VRAM")
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                train_params,
                lr=1e-5,
                weight_decay=1e-2
            )
            logger.warning("  → bitsandbytes chưa cài, dùng AdamW 32-bit (tốn VRAM hơn). Chạy: pip install bitsandbytes")
        
        # 6. AMP Mixed Precision Scaler
        self.scaler = torch.amp.GradScaler('cuda')
        
        # 7. Tạo thư mục checkpoints
        self.checkpoints_dir = PROJECT_DIR / "backend" / "training" / "checkpoints"
        ensureDir(str(self.checkpoints_dir))
        
        logger.info("Trainer khởi tạo thành công. VRAM sử dụng:")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"  Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
        
    @torch.no_grad()
    def _encode_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ảnh RGB tensor sang latent space qua VAE."""
        # VAE cần fp16 input
        latents = self.vae.encode(images.to(self.vae.dtype)).latent_dist.sample()
        latents = latents * self.vae_scale_factor
        return latents.float()  # Trả về float32 cho training
    
    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents về RGB image qua VAE (cho Identity/Texture Loss)."""
        latents_input = (latents / self.vae_scale_factor).to(self.vae.dtype)
        decoded = self.vae.decode(latents_input).sample
        return decoded.float()
        
    def train_step(self, batch, global_step: int, accumulation_steps: int = 8):
        """
        Bước lặp huấn luyện duy nhất.
        batch bao gồm:
          - image: Ảnh thật (Ground Truth)
          - bald_image: Ảnh đã làm trọc
          - mask: Hair mask (1 kênh)
          - style_embed: Vector đặc trưng kiểu tóc (từ CLIP)
          - identity_embed: Vector khuôn mặt (từ AdaFace)
          - text_embeds: Prompt embeddings (đã pre-encode)
          - pooled_text_embeds: Pooled SDXL embeddings
          - time_ids: SDXL time conditioning
        """
        self.unet.train()
        self.injector.train()
        
        # Gradient Accumulation: chỉ zero_grad mỗi N steps
        if global_step % accumulation_steps == 0:
            self.optimizer.zero_grad()
        
        # Giải nén Batch
        gt_images = batch['image'].to(DEVICE)
        bald_images = batch['bald_image'].to(DEVICE)
        masks = batch['mask'].to(DEVICE)
        
        # ==========================================
        # VAE ENCODE (thực)
        # ==========================================
        latents = self._encode_to_latents(gt_images)
        bald_latents = self._encode_to_latents(bald_images)
        masks_downsampled = F.interpolate(masks, size=latents.shape[-2:], mode='nearest')
        
        # ==========================================
        # NOISE SCHEDULING (thực)
        # ==========================================
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=DEVICE
        ).long()
        
        # Thêm noise theo scheduler chuẩn diffusion
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # ==========================================
        # CONDITIONING (thực)
        # ==========================================
        style_embeds = batch['style_embed'].to(DEVICE)
        id_embeds = batch['identity_embed'].to(DEVICE)
        text_embeds = batch['text_embeds'].to(DEVICE)
        pooled_text_embeds = batch['pooled_text_embeds'].to(DEVICE)
        time_ids = batch['time_ids'].to(DEVICE)
        
        # Gộp Style + Identity conditioning qua IP-Adapter
        injected_conds = self.injector.inject_conditioning(style_embeds, id_embeds)
        encoder_hidden_states = torch.cat([text_embeds, injected_conds], dim=1)
        
        added_cond_kwargs = {
            "text_embeds": pooled_text_embeds,
            "time_ids": time_ids
        }
        
        # ==========================================
        # FORWARD PASS (AMP Mixed Precision)
        # ==========================================
        with torch.amp.autocast('cuda', dtype=torch.float16):
            noise_pred = self.unet(
                noisy_latents=noisy_latents,
                bald_latents=bald_latents,
                mask=masks_downsampled,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs
            )
            
            # ==========================================
            # LOSS COMPUTATION
            # ==========================================
            
            # 1. Mask-Aware Diffusion Loss (Core — mọi step)
            loss_diffusion = self.mask_aware_loss(noise_pred, noise, masks_downsampled)
            total_loss = loss_diffusion
        
        # 2. Identity & Texture Loss (mỗi N steps để tiết kiệm VRAM)
        # Decode latents → ảnh RGB → tính perceptual losses
        loss_id_val = 0.0
        loss_tex_val = 0.0
        
        if global_step % 50 == 0 and global_step > 0:
            try:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    # Ước tính denoised output (x0 prediction)
                    alpha_prod_t = self.noise_scheduler.alphas_cumprod[timesteps[0].cpu().item()]
                    sqrt_alpha = alpha_prod_t ** 0.5
                    sqrt_one_minus_alpha = (1 - alpha_prod_t) ** 0.5
                    pred_original = (noisy_latents - sqrt_one_minus_alpha * noise_pred) / (sqrt_alpha + 1e-8)
                    
                    # Decode để lấy ảnh tái tạo
                    decoded_img = self._decode_latents(pred_original.detach())
                    
                    # Identity Loss — so sánh khuôn mặt giữa ảnh gốc và ảnh tái tạo
                    # (Cần identity embedding từ decoded image, nhưng tạm dùng pixel-level proxy)
                    # Bùng nổ VRAM nếu decode mỗi step → chỉ tính mỗi 10 steps
                    loss_tex = self.texture_loss(decoded_img, gt_images)
                    total_loss = total_loss + 0.01 * loss_tex
                    loss_tex_val = loss_tex.item()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("OOM khi tính Texture Loss, skip step này.")
                    torch.cuda.empty_cache()
                else:
                    raise
        
        # ==========================================
        # BACKPROP (AMP + Gradient Accumulation)
        # ==========================================
        # Chia loss cho accumulation_steps để trung bình gradient
        scaled_loss = total_loss / accumulation_steps
        self.scaler.scale(scaled_loss).backward()
        
        # Chỉ cập nhật weights mỗi accumulation_steps
        if (global_step + 1) % accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.unet.parameters()) + list(self.injector.parameters()),
                max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        return {
            "total_loss": total_loss.item(),
            "diffusion_loss": loss_diffusion.item(),
            "texture_loss": loss_tex_val,
        }
        
    def _save_safetensors_safe(self, state_dict, path: str):
        """Lưu safetensors an toàn — ghi vào temp file rồi move để tránh corrupt."""
        import tempfile, shutil
        
        # Tạo temp file CÙNG thư mục để tránh cross-filesystem move trên WSL
        target_dir = os.path.dirname(path)
        fd, temp_path = tempfile.mkstemp(suffix=".safetensors", dir=target_dir)
        os.close(fd)
        try:
            save_file(state_dict, temp_path)
            shutil.move(temp_path, path)
            
            # Verify file đã được tạo thành công
            if not os.path.exists(path):
                logger.error(f"❌ File không tồn tại sau khi save: {path}")
            else:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                logger.info(f"  💾 Saved: {os.path.basename(path)} ({size_mb:.1f} MB)")
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu {path}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def _plot_training_charts(self, history: dict, epoch: int):
        """
        Tạo biểu đồ Loss Chart tự động sau mỗi epoch.
        Lưu vào checkpoints/ để dễ theo dõi.
        """
        charts_dir = self.checkpoints_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Training Stage 2 — After Epoch {epoch}", fontsize=16, fontweight='bold')
        
        steps = range(1, len(history['total_loss']) + 1)
        
        # --- 1. Total Loss (mỗi step) ---
        ax1 = axes[0, 0]
        ax1.plot(steps, history['total_loss'], alpha=0.3, color='#2196F3', linewidth=0.5, label='Per step')
        # Đường trung bình trượt (window = 100 steps)
        if len(history['total_loss']) > 100:
            window = 100
            smoothed = np.convolve(history['total_loss'], np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(history['total_loss']) + 1), smoothed, color='#1565C0', linewidth=2, label=f'MA-{window}')
        ax1.set_title('Total Loss', fontsize=13)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # --- 2. Diffusion Loss (mỗi step) ---
        ax2 = axes[0, 1]
        ax2.plot(steps, history['diffusion_loss'], alpha=0.3, color='#4CAF50', linewidth=0.5, label='Per step')
        if len(history['diffusion_loss']) > 100:
            smoothed = np.convolve(history['diffusion_loss'], np.ones(100)/100, mode='valid')
            ax2.plot(range(100, len(history['diffusion_loss']) + 1), smoothed, color='#2E7D32', linewidth=2, label='MA-100')
        ax2.set_title('Diffusion Loss (Noise Prediction)', fontsize=13)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # --- 3. Texture Loss (mỗi 50 steps, bỏ qua giá trị 0) ---
        ax3 = axes[1, 0]
        texSteps = [i+1 for i, v in enumerate(history['texture_loss']) if v > 0]
        texValues = [v for v in history['texture_loss'] if v > 0]
        if texValues:
            ax3.plot(texSteps, texValues, color='#FF9800', linewidth=1.5, marker='.', markersize=3, label='Texture Loss')
            ax3.set_title('Texture Consistency Loss (mỗi 50 steps)', fontsize=13)
        else:
            ax3.text(0.5, 0.5, 'Chưa có dữ liệu\n(tính sau step 50)', ha='center', va='center', fontsize=12, transform=ax3.transAxes)
            ax3.set_title('Texture Loss', fontsize=13)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # --- 4. Epoch Average Loss ---
        ax4 = axes[1, 1]
        if history['epoch_avg_loss']:
            epochs = range(1, len(history['epoch_avg_loss']) + 1)
            ax4.plot(epochs, history['epoch_avg_loss'], color='#E91E63', linewidth=2.5, marker='o', markersize=8, label='Avg Loss')
            # Đánh dấu best epoch
            bestIdx = np.argmin(history['epoch_avg_loss'])
            ax4.scatter([bestIdx + 1], [history['epoch_avg_loss'][bestIdx]], color='#FFD700', s=200, zorder=5, marker='★', label=f'Best (Epoch {bestIdx+1})')
            ax4.set_title('Epoch Average Loss', fontsize=13)
            ax4.set_xticks(list(epochs))
        else:
            ax4.text(0.5, 0.5, 'Chưa có dữ liệu', ha='center', va='center', fontsize=12, transform=ax4.transAxes)
            ax4.set_title('Epoch Average Loss', fontsize=13)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Avg Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Lưu file
        chartPath = charts_dir / f"loss_chart_epoch_{epoch}.png"
        latestPath = charts_dir / "loss_chart_latest.png"
        fig.savefig(str(chartPath), dpi=150, bbox_inches='tight')
        fig.savefig(str(latestPath), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"📊 Loss Chart đã lưu: {chartPath.name}")

    def _save_checkpoint(self, suffix: str, is_best: bool = False):
        """
        Lưu checkpoint UNet + Injector.
        Chiến lược tiết kiệm dung lượng: CHỈ GIỮ file BEST.
        File epoch/step chỉ lưu tạm, xóa ngay sau khi best được cập nhật.
        """
        if is_best:
            # Lưu trực tiếp vào file best (không lưu epoch/step riêng để tiết kiệm disk)
            best_unet = self.checkpoints_dir / "deep_hair_v1_best.safetensors"
            best_inj = self.checkpoints_dir / "injector_best.safetensors"
            logger.info(f"🏆 Saving BEST model ({suffix})...")
            self._save_safetensors_safe(self.unet.state_dict(), str(best_unet))
            self._save_safetensors_safe(self.injector.state_dict(), str(best_inj))
            
            # Verify
            if best_unet.exists() and best_inj.exists():
                logger.info(f"✅ BEST files verified: {best_unet.name} + {best_inj.name}")
            else:
                logger.error(f"❌ BEST files MISSING! Kiểm tra quyền ghi: {self.checkpoints_dir}")
        else:
            # Không phải best → lưu tạm 1 file backup (ghi đè mỗi lần)
            # Mục đích: nếu training crash, có thể resume từ backup này
            backup_unet = self.checkpoints_dir / "stage2_backup.safetensors"
            backup_inj = self.checkpoints_dir / "injector_backup.safetensors"
            self._save_safetensors_safe(self.unet.state_dict(), str(backup_unet))
            self._save_safetensors_safe(self.injector.state_dict(), str(backup_inj))
        
        # Dọn dẹp: xóa tất cả file step/epoch cũ (chỉ giữ best + backup)
        keep_names = {
            "deep_hair_v1_best.safetensors",
            "injector_best.safetensors", 
            "stage2_backup.safetensors",
            "injector_backup.safetensors",
            "deep_hair_v1_latest.safetensors",
            "injector_latest.safetensors",
            "deep_hair_v1.safetensors",  # file export đã có sẵn
        }
        for p in self.checkpoints_dir.glob("*.safetensors"):
            if p.name not in keep_names:
                try:
                    p.unlink()
                    logger.info(f"  🗑️ Xóa checkpoint cũ: {p.name}")
                except:
                    pass

    def train_loop(self, num_epochs=1, batch_size=1, max_samples=1000, target_size=(512, 512), accumulation_steps=8, resume=True):
        """
        Vòng lặp PyTorch Training Thực Kế.
        
        Args:
            num_epochs: Số epochs training
            batch_size: Batch size (1 cho RTX 3060 12GB)
            max_samples: Giới hạn số samples dataset (0 = tất cả)
            target_size: Kích thước ảnh (512x512 tiết kiệm VRAM, 1024x1024 chất lượng cao)
            accumulation_steps: Gradient accumulation steps (effective batch = batch_size * accumulation_steps)
            resume: Tiếp tục train từ checkpoint tốt nhất nếu tồn tại
        """
        logger.info(f"Khởi động vòng lặp Training Stage 2 - {num_epochs} Epochs")
        logger.info(f"  📐 Resolution: {target_size[0]}x{target_size[1]}")
        logger.info(f"  📊 Max samples: {max_samples if max_samples > 0 else 'ALL'}")
        logger.info(f"  🔄 Gradient Accumulation: {accumulation_steps} steps (effective batch = {batch_size * accumulation_steps})")
        logger.info(f"  💾 Checkpoint strategy: CHỈ GIỮ BEST + backup (tiết kiệm dung lượng)")
        
        if resume:
            # Ưu tiên load best, fallback sang backup
            best_unet = self.checkpoints_dir / "deep_hair_v1_best.safetensors"
            best_inj = self.checkpoints_dir / "injector_best.safetensors"
            backup_unet = self.checkpoints_dir / "stage2_backup.safetensors"
            backup_inj = self.checkpoints_dir / "injector_backup.safetensors"
            
            load_unet = best_unet if best_unet.exists() else (backup_unet if backup_unet.exists() else None)
            load_inj = best_inj if best_inj.exists() else (backup_inj if backup_inj.exists() else None)
            
            if load_unet and load_inj:
                from safetensors.torch import load_file as load_safetensors
                try:
                    self.unet.load_state_dict(load_safetensors(str(load_unet)))
                    self.injector.load_state_dict(load_safetensors(str(load_inj)))
                    logger.info(f"🔄 [RESUME] Đã tải trọng số từ {load_unet.name} + {load_inj.name}")
                except Exception as e:
                    logger.error(f"❌ Lỗi khi tải checkpoint: {e}")
            else:
                logger.warning("⚠️ Flag --resume được bật nhưng thiếu file checkpoint. Bắt đầu train từ đầu.")
                
        processed_dir = PROJECT_DIR / "backend" / "training" / "processed"
        
        # Pre-encode text prompts trước khi bắt đầu training
        # Text Encoders sẽ bị giải phóng sau khi encode xong
        text_encoder = None
        cache_dir = processed_dir / "prompt_embeddings"
        
        # Chỉ load text encoder nếu chưa có cache
        meta_path = processed_dir / "metadata.jsonl"
        needs_encoding = False
        if meta_path.exists():
            with open(str(meta_path), "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line.strip())
                    img_id = item['id']
                    cache_file = cache_dir / f"{img_id}.pt"
                    cache_file_dashed = cache_dir / f"{img_id.replace('_', '-')}.pt"
                    if not cache_file.exists() and not cache_file_dashed.exists():
                        needs_encoding = True
                        break
        
        if needs_encoding:
            logger.info("Prompt embeddings chưa được cache. Đang load Text Encoders...")
            text_encoder = SDXLTextEncoder()
        
        # Load Texture Encoder (Stage 1) để extract style vectors
        texture_encoder = None
        style_cache_dir = processed_dir / "style_embeddings_cache"
        
        # Chỉ load encoder nếu chưa có cache
        needs_style_extraction = False
        if meta_path.exists():
            with open(str(meta_path), "r", encoding="utf-8") as f2:
                for line in f2:
                    if not line.strip(): continue
                    item = json.loads(line.strip())
                    if not (style_cache_dir / f"{item['id']}.npy").exists():
                        needs_style_extraction = True
                        break
        
        if needs_style_extraction:
            logger.info("Style embeddings chưa cache. Đang load Texture Encoder (Stage 1)...")
            from backend.training.models.texture_encoder import HairTextureEncoder
            from safetensors.torch import load_file as load_safetensors
            
            texture_encoder = HairTextureEncoder(pretrained=False).to(DEVICE).eval()
            
            # Load checkpoint tốt nhất của Stage 1
            tex_ckpt = PROJECT_DIR / "backend" / "training" / "checkpoints" / "texture_encoder_latest.safetensors"
            if tex_ckpt.exists():
                state_dict = load_safetensors(str(tex_ckpt))
                texture_encoder.load_state_dict(state_dict, strict=False)
                logger.info(f"  → Loaded Texture Encoder từ {tex_ckpt.name}")
            else:
                logger.warning("  ⚠️ Không tìm thấy checkpoint Stage 1, dùng random weights!")
            texture_encoder.requires_grad_(False)
        
        dataset = HairInpaintingDataset(
            processed_dir, text_encoder=text_encoder, texture_encoder=texture_encoder,
            target_size=target_size, max_samples=max_samples
        )
        
        # Giải phóng Text Encoders sau khi encode xong
        if text_encoder is not None:
            text_encoder.unload()
            del text_encoder
        
        if len(dataset) == 0:
            logger.error("Dataset trống! Hãy chạy prepare_dataset_deephair.py trước.")
            return
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
        
        # =============================================
        # BEST MODEL TRACKING — theo dõi loss tốt nhất
        # =============================================
        best_epoch_loss = float('inf')
        best_epoch = -1
        
        global_step = 0
        step_times = []  # Theo dõi thời gian mỗi step
        
        # Thu thập lịch sử loss để vẽ chart
        loss_history = {
            'total_loss': [],
            'diffusion_loss': [],
            'texture_loss': [],
            'epoch_avg_loss': [],
        }
        
        for epoch in range(num_epochs):
            epoch_losses = []  # Thu thập loss mỗi step để tính trung bình
            epoch_start = time.time()
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for step, batch in enumerate(pbar):
                step_start = time.time()
                losses = self.train_step(batch, global_step, accumulation_steps=accumulation_steps)
                step_time = time.time() - step_start
                step_times.append(step_time)
                epoch_losses.append(losses['total_loss'])
                
                # Ghi lại loss history cho chart
                loss_history['total_loss'].append(losses['total_loss'])
                loss_history['diffusion_loss'].append(losses['diffusion_loss'])
                loss_history['texture_loss'].append(losses['texture_loss'])
                
                # Tính ETA
                avg_step_time = sum(step_times[-50:]) / len(step_times[-50:])  # Trung bình 50 steps gần nhất
                remaining_steps = len(dataloader) - step - 1
                eta_seconds = remaining_steps * avg_step_time
                eta_str = f"{int(eta_seconds//3600)}h{int((eta_seconds%3600)//60)}m"
                
                pbar.set_postfix({
                    "Loss": f"{losses['total_loss']:.4f}",
                    "Diff": f"{losses['diffusion_loss']:.4f}",
                    "Tex": f"{losses['texture_loss']:.4f}",
                    "Best": f"{best_epoch_loss:.4f}" if best_epoch_loss < float('inf') else "N/A",
                    "s/it": f"{step_time:.1f}s",
                    "ETA": eta_str,
                })
                global_step += 1
            
            # Tính average loss cho epoch này
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
            
            # Kiểm tra xem epoch này có tốt hơn best không
            is_new_best = avg_epoch_loss < best_epoch_loss
            
            if is_new_best:
                best_epoch_loss = avg_epoch_loss
                best_epoch = epoch + 1
                logger.info(f"🏆 NEW BEST! Epoch {epoch+1} — Avg Loss: {avg_epoch_loss:.6f} — Time: {epoch_time/60:.1f}min")
            else:
                logger.info(f"Epoch {epoch+1} — Avg Loss: {avg_epoch_loss:.6f} (Best vẫn là Epoch {best_epoch}: {best_epoch_loss:.6f}) — Time: {epoch_time/60:.1f}min")
            
            # Lưu checkpoint: best nếu đạt, backup nếu không
            self._save_checkpoint(f"epoch_{epoch+1}", is_best=is_new_best)
            
            # Ghi epoch avg loss và vẽ chart
            loss_history['epoch_avg_loss'].append(avg_epoch_loss)
            self._plot_training_charts(loss_history, epoch + 1)
                    
        # Lưu Final Model (latest = epoch cuối cùng)
        final_ckpt_path = self.checkpoints_dir / "deep_hair_v1_latest.safetensors"
        self._save_safetensors_safe(self.unet.state_dict(), str(final_ckpt_path))
        final_inj_path = self.checkpoints_dir / "injector_latest.safetensors"
        self._save_safetensors_safe(self.injector.state_dict(), str(final_inj_path))
        
        # Xóa file backup sau khi đã có latest
        for backup in ["stage2_backup.safetensors", "injector_backup.safetensors"]:
            bp = self.checkpoints_dir / backup
            if bp.exists():
                try: bp.unlink()
                except: pass
        
        # Thống kê dung lượng checkpoint cuối cùng
        total_size = 0
        for f in self.checkpoints_dir.glob("*.safetensors"):
            total_size += f.stat().st_size
        
        logger.info(f"{'='*60}")
        logger.info(f"✅ Hoàn thành Training Stage 2!")
        logger.info(f"  📁 Model cuối cùng: {final_ckpt_path}")
        logger.info(f"  🏆 Model tốt nhất: deep_hair_v1_best.safetensors (Epoch {best_epoch}, Loss: {best_epoch_loss:.6f})")
        logger.info(f"  💾 Tổng dung lượng checkpoints: {total_size / (1024**3):.2f} GB")
        logger.info(f"  💡 Dùng file BEST để deploy, không dùng file latest!")
        logger.info(f"{'='*60}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Stage 2 - Hair Inpainting")
    parser.add_argument("--epochs", type=int, default=1, help="Số epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=1000, help="Giới hạn samples (0=tất cả)")
    parser.add_argument("--resolution", type=int, default=512, help="Kích thước ảnh (512 hoặc 1024)")
    parser.add_argument("--accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--fresh", action="store_true", help="Train từ đầu, KHÔNG load checkpoint cũ")
    args = parser.parse_args()
    
    trainer = Stage2Trainer()
    trainer.train_loop(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        target_size=(args.resolution, args.resolution),
        accumulation_steps=args.accumulation,
        resume=not args.fresh
    )
