import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
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
    def __init__(self, data_dir: Path, text_encoder: SDXLTextEncoder = None, target_size=(1024, 1024)):
        self.data_dir = data_dir
        self.target_size = target_size
        self.metadata = []
        self.prompt_embeds_cache = {}
        
        meta_path = data_dir / "metadata.jsonl"
        if meta_path.exists():
            with open(str(meta_path), "r", encoding="utf-8") as f:
                for line in f:
                    self.metadata.append(json.loads(line.strip()))
        
        # Pre-encode text prompts nếu Text Encoder được cung cấp
        if text_encoder is not None and len(self.metadata) > 0:
            logger.info(f"Pre-encoding {len(self.metadata)} text prompts...")
            
            # Cache thư mục cho prompt embeddings
            cache_dir = data_dir / "prompt_embeddings"
            ensureDir(str(cache_dir))
            
            for item in tqdm(self.metadata, desc="Encoding Prompts"):
                img_id = item["id"]
                cache_file = cache_dir / f"{img_id}.pt"
                
                if cache_file.exists():
                    # Load từ cache
                    cached = torch.load(str(cache_file), map_location="cpu", weights_only=True)
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
                    cache_file = cache_dir / f"{img_id}.pt"
                    if cache_file.exists():
                        self.prompt_embeds_cache[img_id] = torch.load(
                            str(cache_file), map_location="cpu", weights_only=True
                        )
                    
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
        
        # Load Ground Truth image
        img_candidates = list((PROJECT_DIR / "dataset").rglob(f"{img_id}.*"))
        if not img_candidates:
            img_candidates = list((PROJECT_DIR / "dataset").rglob(f"{img_id.replace('_', '-')}.*"))
        
        # Fallback: tìm trong thư mục dataset K-Hairstyle
        if not img_candidates:
            khairstyle_dir = PROJECT_DIR / "backend" / "data" / "dataset" / "khairstyle" / "training" / "images"
            img_candidates = list(khairstyle_dir.rglob(f"{img_id}.*"))
            
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
            "style_embed": torch.zeros(1024),  # TODO: CLIP Vision vector khi có encoder riêng
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
    Sử dụng AMP Mixed Precision cho GPU 24GB.
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
        
        # 2. Load Noise Scheduler
        logger.info("  → Loading DDPMScheduler...")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            LOCAL_SDXL_PATH, subfolder="scheduler"
        )
        
        # 3. Load UNet 9-channel + CrossAttention Injector
        logger.info("  → Loading UNet 9-channel (fp16)...")
        self.unet = HairInpaintingUNet().to(DEVICE)
        self.injector = CrossAttentionInjector(self.unet.unet).to(DEVICE)
        
        # 4. Khởi tạo Loss Functions
        self.mask_aware_loss = MaskAwareLoss(loss_type='l2').to(DEVICE)
        self.identity_loss = IdentityCosineLoss().to(DEVICE)
        self.texture_loss = TextureConsistencyLoss().to(DEVICE)
        
        # 5. Optimizer (chỉ train UNet + Injector)
        self.optimizer = torch.optim.AdamW(
            list(self.unet.parameters()) + list(self.injector.parameters()),
            lr=1e-5,   # Learning rate thấp hơn cho fine-tuning SDXL
            weight_decay=1e-2
        )
        
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
        
    def train_step(self, batch, global_step: int):
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
        
        if global_step % 10 == 0 and global_step > 0:
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
        # BACKPROP (AMP)
        # ==========================================
        self.scaler.scale(total_loss).backward()
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
        
    def train_loop(self, num_epochs=1, batch_size=1):
        """ Vòng lặp PyTorch Training Thực Kế """
        logger.info(f"Khởi động vòng lặp Training Stage 2 - {num_epochs} Epochs")
        
        processed_dir = PROJECT_DIR / "backend" / "training" / "processed"
        
        # Pre-encode text prompts trước khi bắt đầu training
        # Text Encoders sẽ bị giải phóng sau khi encode xong
        text_encoder = None
        cache_dir = processed_dir / "prompt_embeddings"
        
        # Chỉ load text encoder nếu chưa có cache
        meta_path = processed_dir / "metadata.jsonl"
        needs_encoding = False
        if meta_path.exists():
            with open(str(meta_path), "r") as f:
                first_line = f.readline().strip()
                if first_line:
                    first_item = json.loads(first_line)
                    cache_file = cache_dir / f"{first_item['id']}.pt"
                    if not cache_file.exists():
                        needs_encoding = True
        
        if needs_encoding:
            logger.info("Prompt embeddings chưa được cache. Đang load Text Encoders...")
            text_encoder = SDXLTextEncoder()
        
        dataset = HairInpaintingDataset(processed_dir, text_encoder=text_encoder)
        
        # Giải phóng Text Encoders sau khi encode xong
        if text_encoder is not None:
            text_encoder.unload()
            del text_encoder
        
        if len(dataset) == 0:
            logger.error("Dataset trống! Hãy chạy prepare_dataset_deephair.py trước.")
            return
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        global_step = 0
        for epoch in range(num_epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for step, batch in enumerate(pbar):
                losses = self.train_step(batch, global_step)
                
                pbar.set_postfix({
                    "Loss": f"{losses['total_loss']:.4f}",
                    "Diff": f"{losses['diffusion_loss']:.4f}",
                    "Tex": f"{losses['texture_loss']:.4f}",
                })
                global_step += 1
                
                # Checkpointing (mỗi 500 steps)
                if global_step % 500 == 0:
                    ckpt_path = self.checkpoints_dir / f"stage2_step_{global_step}.safetensors"
                    save_file(self.unet.state_dict(), str(ckpt_path))
                    # Lưu thêm injector
                    inj_path = self.checkpoints_dir / f"injector_step_{global_step}.safetensors"
                    save_file(self.injector.state_dict(), str(inj_path))
                    logger.info(f"Đã lưu Checkpoint tại step {global_step}")
                    
            # Checkpoint cuối epoch
            ckpt_path = self.checkpoints_dir / f"stage2_epoch_{epoch+1}.safetensors"
            save_file(self.unet.state_dict(), str(ckpt_path))
            logger.info(f"Đã lưu Checkpoint cuối Epoch {epoch+1}")
                    
        # Lưu Final Model sau khi xong vòng lặp
        final_ckpt_path = self.checkpoints_dir / "deep_hair_v1_latest.safetensors"
        save_file(self.unet.state_dict(), str(final_ckpt_path))
        
        final_inj_path = self.checkpoints_dir / "injector_latest.safetensors"
        save_file(self.injector.state_dict(), str(final_inj_path))
        
        logger.info(f"Hoàn thành toàn bộ Training Stage 2! Lưu model cuối: {final_ckpt_path}")

if __name__ == "__main__":
    trainer = Stage2Trainer()
    trainer.train_loop(num_epochs=20, batch_size=1)
