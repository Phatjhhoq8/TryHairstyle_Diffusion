import os
import torch
from pathlib import Path
from tqdm import tqdm

from backend.app.services.training_utils import setupLogger, getDevice
from backend.training.models.stage2_unet import HairInpaintingUNet, CrossAttentionInjector
from backend.training.models.losses import MaskAwareLoss, IdentityCosineLoss, TextureConsistencyLoss

logger = setupLogger("TrainStage2_Inpainting")
DEVICE = getDevice()

class Stage2Trainer:
    """
    Kịch bản huấn luyện Stage 2: Mask-Conditioned Hair Inpainting
    Kết hợp 9-channel UNet, Custom Identity Loss, và Mask-aware Gradient Locking.
    """
    def __init__(self):
        logger.info("Khởi tạo Stage 2 Trainer: SDXL Mask-Conditioned Inpainting")
        
        # 1. Khởi tạo Models
        self.unet = HairInpaintingUNet().to(DEVICE)
        self.injector = CrossAttentionInjector(self.unet.unet).to(DEVICE)
        
        # TODO: Load VAE, Scheduler và TextEncoder của SDXL để hoàn thành Diffusion Pipeline
        # self.vae = AutoencoderKL.from_pretrained(...)
        # self.noise_scheduler = DDPMScheduler.from_pretrained(...)
        
        # 2. Khởi tạo Loss
        self.mask_aware_loss = MaskAwareLoss(loss_type='l2').to(DEVICE)
        self.identity_loss = IdentityCosineLoss().to(DEVICE)
        self.texture_loss = TextureConsistencyLoss().to(DEVICE)
        
        # 3. Optimizer
        # Sử dụng bitsandbytes 8-bit AdamW để tiết kiệm VRAM 24GB
        self.optimizer = torch.optim.AdamW(
            list(self.unet.parameters()) + list(self.injector.parameters()),
            lr=1e-4,
            weight_decay=1e-2
        )
        
        logger.info("Trainer khởi tạo thành công.")
        
    def train_step(self, batch):
        """
        Bước lặp huấn luyện duy nhất.
        batch bao gồm:
          - image: Ảnh thật (Ground Truth)
          - bald_image: Ảnh đã làm trọc
          - mask: Hair mask (1 kênh)
          - style_embed: Vector đặc trưng kiểu tóc (từ CLIP)
          - identity_embed: Vector khuôn mặt (từ AdaFace)
          - text_embeds: Prompt embeddings (Tiếng Anh dịch sẵn)
        """
        self.unet.train()
        self.injector.train()
        self.optimizer.zero_grad()
        
        # Giải nén Batch (Mock tensors cho kịch bản)
        gt_images = batch['image'].to(DEVICE)
        bald_images = batch['bald_image'].to(DEVICE)
        masks = batch['mask'].to(DEVICE)
        
        # VAE Encode
        # latents = self.vae.encode(gt_images).latent_dist.sample() * self.vae.config.scaling_factor
        # bald_latents = self.vae.encode(bald_images).latent_dist.sample() * self.vae.config.scaling_factor
        
        # Mock VAE output
        latents = torch.randn(gt_images.size(0), 4, gt_images.size(2)//8, gt_images.size(3)//8, device=DEVICE)
        bald_latents = torch.randn_like(latents)
        masks_downsampled = torch.nn.functional.interpolate(masks, size=latents.shape[-2:], mode='nearest')
        
        # Thêm Noise chuẩn bị cho mô hình đoán
        noise = torch.randn_like(latents)
        # timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()
        timesteps = torch.tensor([500] * latents.shape[0], device=DEVICE) 
        noisy_latents = latents + noise # Lấy ví dụ cộng tuyến tính (Thực tế là add_noise từ Scheduler)
        
        # Identity Injection (IP-Adapter Base)
        style_embeds = batch['style_embed'].to(DEVICE)
        id_embeds = batch['identity_embed'].to(DEVICE)
        text_embeds = batch['text_embeds'].to(DEVICE)
        
        # Gộp Style + Identity + Text Prompt lại
        injected_conds = self.injector.inject_conditioning(style_embeds, id_embeds)
        encoder_hidden_states = torch.cat([text_embeds, injected_conds], dim=1)
        
        # Forward qua UNet 9 Channels
        # added_cond_kwargs = {"text_embeds": ..., "time_ids": ...} (Của SDXL)
        noise_pred = self.unet(
            noisy_latents=noisy_latents,
            bald_latents=bald_latents,
            mask=masks_downsampled,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=None # Mock
        )
        
        # ==========================================
        # LOSS COMPUTATION
        # ==========================================
        
        # 1. Mask-Aware Loss (Chỉ tính Gradient theo Vùng Tóc)
        # Ép mô hình vẽ vào vùng tóc chứ không được làm nhiễu khuôn mặt.
        loss_diffusion = self.mask_aware_loss(noise_pred, noise, masks_downsampled)
        
        total_loss = loss_diffusion
        
        # 2. Identity & Texture Loss (Sẽ đòi hỏi decode Latents ra ảo ảnh RGB)
        # Xảy ra ở các step cuối (Denoised Output)
        # Do việc decode VAE liên tục tốn VRAM, thường ta chỉ tính các Loss này sau N steps (VD: N=10)
        # giả sử tính:
        # decoded_img = self.vae.decode(latents_pred)
        # gen_id = arcface(decoded_img)
        # loss_id = self.identity_loss(gen_id, target_id)
        # loss_texture = self.texture_loss(decoded_img, gt_images)
        # total_loss += (0.1 * loss_id) + (0.01 * loss_texture)
        
        # Backprop
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
        
    def test_run(self):
        """ Chạy thử thiết kế kịch bản với Random Tensors. """
        logger.info("Chạy Test Run Kịch Bản Stage 2...")
        
        B, H, W = 2, 1024, 1024 # Batch 2, 1024x1024 SDXL
        mock_batch = {
            'image': torch.randn(B, 3, H, W),
            'bald_image': torch.randn(B, 3, H, W),
            'mask': torch.ones(B, 1, H, W),  # Tóc trên toàn ảnh
            'style_embed': torch.randn(B, 1024),
            'identity_embed': torch.randn(B, 512),
            'text_embeds': torch.randn(B, 77, 1024) # SDXL 77 tokens context lengths
        }
        
        loss = self.train_step(mock_batch)
        logger.info(f"Hoàn tất Forward/Backward Test Run! Dummy Loss: {loss:.4f}")

if __name__ == "__main__":
    trainer = Stage2Trainer()
    trainer.test_run()
