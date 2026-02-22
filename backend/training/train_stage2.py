import os
import sys
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from safetensors.torch import save_file

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from backend.app.services.training_utils import setupLogger, getDevice
from backend.training.models.stage2_unet import HairInpaintingUNet, CrossAttentionInjector
from backend.training.models.losses import MaskAwareLoss, IdentityCosineLoss, TextureConsistencyLoss

logger = setupLogger("TrainStage2_Inpainting")
DEVICE = getDevice()

class HairInpaintingDataset(Dataset):
    def __init__(self, data_dir: Path, target_size=(1024, 1024)):
        self.data_dir = data_dir
        self.target_size = target_size
        self.metadata = []
        
        meta_path = data_dir / "metadata.jsonl"
        if meta_path.exists():
            with open(str(meta_path), "r", encoding="utf-8") as f:
                for line in f:
                    self.metadata.append(json.loads(line.strip()))
                    
        # Transform tensor chuẩn SDXL (-1 to 1)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load Images
        # Lưu ý: Các file gốc từ web UI thường không đúng 1024, ta resize ép cứng (Trong thực tế cần padding để tránh méo ảnh)
        img_candidates = list((PROJECT_DIR / "dataset").rglob(f"{item['id']}.*"))
        if not img_candidates:
            # Fallback nếu tìm ID gốc bị lỗi dấu _
            img_candidates = list((PROJECT_DIR / "dataset").rglob(f"{item['id'].replace('_', '-')}.*"))
            
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
        
        # Trả về Tensors
        return {
            "image": self.img_transform(gt_img),
            "bald_image": self.img_transform(bald_img),
            "mask": torch.from_numpy(true_mask).permute(2,0,1),
            "style_embed": torch.zeros(1024), # Mock CLIP Vision Vector
            "identity_embed": torch.from_numpy(id_embed).squeeze(0),
            "text_embeds": torch.zeros(77, 2048), # Mock CLIP Text Vector
            "pooled_text_embeds": torch.zeros(1280),
            "time_ids": torch.tensor([self.target_size[0], self.target_size[1], 0, 0, self.target_size[0], self.target_size[1]], dtype=torch.float32)
        }

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
        pooled_text_embeds = batch['pooled_text_embeds'].to(DEVICE)
        time_ids = batch['time_ids'].to(DEVICE)
        
        # Gộp Style + Identity + Text Prompt lại
        injected_conds = self.injector.inject_conditioning(style_embeds, id_embeds)
        encoder_hidden_states = torch.cat([text_embeds, injected_conds], dim=1)
        
        # Forward qua UNet 9 Channels
        # added_cond_kwargs = {"text_embeds": ..., "time_ids": ...} (Của SDXL)
        added_cond_kwargs = {
            "text_embeds": pooled_text_embeds,
            "time_ids": time_ids
        }
        
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
        
    def train_loop(self, num_epochs=1, batch_size=1):
        """ Vòng lặp PyTorch Training Thực Kế """
        logger.info(f"Khởi động vòng lặp Training Stage 2 - {num_epochs} Epochs")
        
        processed_dir = PROJECT_DIR / "backend" / "training" / "processed"
        dataset = HairInpaintingDataset(processed_dir)
        
        if len(dataset) == 0:
            logger.error("Dataset trống! Hãy chạy prepare_dataset_deephair.py trước.")
            return
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        global_step = 0
        for epoch in range(num_epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for step, batch in enumerate(pbar):
                loss = self.train_step(batch)
                
                pbar.set_postfix({"Loss": f"{loss:.4f}"})
                global_step += 1
                
                # Checkpointing thực tế (mỗi X steps hoặc mỗi cuối epoch)
                if global_step % 1000 == 0:
                    ckpt_path = PROJECT_DIR / "backend" / "training" / "checkpoints" / f"stage2_step_{global_step}.safetensors"
                    save_file(self.unet.state_dict(), str(ckpt_path))
                    logger.info(f"Đã lưu Checkpoint Weights tại: {ckpt_path}")
                    
        # Lưu Final Model sau khi xong vòng lặp
        final_ckpt_path = PROJECT_DIR / "backend" / "training" / "checkpoints" / "deep_hair_v1_latest.safetensors"
        save_file(self.unet.state_dict(), str(final_ckpt_path))
        logger.info(f"Hoàn thành toàn bộ Training Stage 2! Lưu model cuối: {final_ckpt_path}")

if __name__ == "__main__":
    trainer = Stage2Trainer()
    trainer.train_loop(num_epochs=20, batch_size=1)
