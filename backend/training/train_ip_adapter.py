"""
Train IP-Adapter for Hair Style Transfer
Based on official tencent-ailab/IP-Adapter implementation

Training Objective:
- Input: Hair image → CLIP embedding → IP-Adapter projection
- Task: Reconstruct the same image using diffusion model
- Loss: MSE(noise_pred, noise) - standard diffusion denoising loss
"""

import os
import json
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

from ip_adapter.projection import ImageProjModel
from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


class HairStyleDataset(Dataset):
    """
    Dataset for K-Hairstyle training
    
    Format: Folder structure with hair images
    Each image is used as both input (CLIP) and target (reconstruction)
    """
    
    def __init__(
        self, 
        data_root: str,
        tokenizer,
        size: int = 512,
        t_drop_rate: float = 0.05,  # Text drop rate for classifier-free guidance
        i_drop_rate: float = 0.05,  # Image drop rate
        ti_drop_rate: float = 0.05  # Both drop rate
    ):
        super().__init__()
        
        self.data_root = data_root
        self.size = size
        self.tokenizer = tokenizer
        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.ti_drop_rate = ti_drop_rate
        
        # Collect all image paths
        self.image_paths = self._collect_images()
        print(f"Found {len(self.image_paths)} images")
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1] range
        ])
        self.clip_image_processor = CLIPImageProcessor()
    
    def _collect_images(self):
        """Collect all image paths from data_root"""
        paths = []
        extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        
        for root, _, files in os.walk(self.data_root):
            for f in files:
                if Path(f).suffix.lower() in extensions:
                    paths.append(os.path.join(root, f))
        
        return paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        max_retries = 10
        for _ in range(max_retries):
            try:
                img_path = self.image_paths[idx]
                
                # Load image
                raw_image = Image.open(img_path).convert("RGB")
                
                # Transform for VAE
                image = self.transform(raw_image)
                
                # Transform for CLIP
                clip_image = self.clip_image_processor(
                    images=raw_image, 
                    return_tensors="pt"
                ).pixel_values[0]
                
                # Generate simple text description (can be improved with labels)
                text = "a photo of a person with beautiful hairstyle"
                
                # Classifier-free guidance: random dropout
                drop_image_embed = 0
                rand_num = random.random()
                if rand_num < self.i_drop_rate:
                    drop_image_embed = 1
                elif rand_num < (self.i_drop_rate + self.t_drop_rate):
                    text = ""
                elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
                    text = ""
                    drop_image_embed = 1
                
                # Tokenize text
                text_input_ids = self.tokenizer(
                    text,
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids[0]
                
                return {
                    "image": image,
                    "clip_image": clip_image,
                    "text_input_ids": text_input_ids,
                    "drop_image_embed": drop_image_embed,
                }
            except Exception as e:
                # Skip corrupted images
                idx = random.randint(0, len(self.image_paths) - 1)
        
        # Fallback
        return {
            "image": torch.zeros(3, self.size, self.size),
            "clip_image": torch.zeros(3, 224, 224),
            "text_input_ids": torch.zeros(77, dtype=torch.long),
            "drop_image_embed": 1,
        }


def collate_fn(data):
    """Custom collate function for DataLoader"""
    images = torch.stack([d["image"] for d in data])
    clip_images = torch.stack([d["clip_image"] for d in data])
    text_input_ids = torch.stack([d["text_input_ids"] for d in data])
    drop_image_embeds = [d["drop_image_embed"] for d in data]
    
    return {
        "images": images,
        "clip_images": clip_images,
        "text_input_ids": text_input_ids,
        "drop_image_embeds": drop_image_embeds,
    }


class IPAdapterTrainer(nn.Module):
    """
    IP-Adapter wrapper for training
    Combines: UNet + ImageProjModel + Adapter Modules
    """
    
    def __init__(self, unet, image_proj_model, adapter_modules):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
    
    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        """
        Forward pass with IP-Adapter
        
        Args:
            noisy_latents: Noisy latent images
            timesteps: Diffusion timesteps
            encoder_hidden_states: Text embeddings from CLIP text encoder
            image_embeds: CLIP image embeddings
            
        Returns:
            noise_pred: Predicted noise
        """
        # Project image embeddings to IP tokens
        ip_tokens = self.image_proj_model(image_embeds)
        
        # Concatenate text embeddings with IP tokens
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        
        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        return noise_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Train IP-Adapter for Hair Style Transfer")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Pretrained SD model"
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="h94/IP-Adapter",
        help="Path to CLIP image encoder or HuggingFace model ID"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="backend/dataset/khairstyle/training/images",
        help="Path to training images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="backend/output/ip_adapter_hair",
        help="Output directory for checkpoints"
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup accelerator
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # ==== Load Models ====
    print("Loading models...")
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model, 
        subfolder="scheduler"
    )
    
    # Tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model, 
        subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model, 
        subfolder="text_encoder"
    )
    
    # VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, 
        subfolder="vae"
    )
    
    # UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model, 
        subfolder="unet"
    )
    
    # CLIP Image Encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.image_encoder_path,
        subfolder="models/image_encoder"
    )
    
    # Freeze base models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    # ==== Setup IP-Adapter ====
    print("Setting up IP-Adapter...")
    
    # Image projection model
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )
    
    # Setup attention processors
    attn_procs = {}
    unet_sd = unet.state_dict()
    
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim
            )
            attn_procs[name].load_state_dict(weights)
    
    unet.set_attn_processor(attn_procs)
    adapter_modules = nn.ModuleList(unet.attn_processors.values())
    
    # Create trainer wrapper
    ip_adapter = IPAdapterTrainer(unet, image_proj_model, adapter_modules)
    
    # ==== Setup Training ====
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # Optimizer (only train IP-Adapter modules)
    params_to_opt = list(ip_adapter.image_proj_model.parameters()) + \
                    list(ip_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(
        params_to_opt,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Dataset
    print("Loading dataset...")
    train_dataset = HairStyleDataset(
        data_root=args.data_root,
        tokenizer=tokenizer,
        size=args.resolution,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare with accelerator
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(
        ip_adapter, optimizer, train_dataloader
    )
    
    # ==== Training Loop ====
    global_step = 0
    
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_main_process
        )
        
        for batch in progress_bar:
            with accelerator.accumulate(ip_adapter):
                # Encode images to latents
                with torch.no_grad():
                    latents = vae.encode(
                        batch["images"].to(accelerator.device, dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()
                
                # Add noise to latents (forward diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get CLIP image embeddings
                with torch.no_grad():
                    image_embeds = image_encoder(
                        batch["clip_images"].to(accelerator.device, dtype=weight_dtype)
                    ).image_embeds
                
                # Apply dropout for classifier-free guidance
                image_embeds_list = []
                for img_emb, drop in zip(image_embeds, batch["drop_image_embeds"]):
                    if drop == 1:
                        image_embeds_list.append(torch.zeros_like(img_emb))
                    else:
                        image_embeds_list.append(img_emb)
                image_embeds = torch.stack(image_embeds_list)
                
                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(
                        batch["text_input_ids"].to(accelerator.device)
                    )[0]
                
                # Predict noise
                noise_pred = ip_adapter(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states, 
                    image_embeds
                )
                
                # MSE Loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                # Backprop
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            # Log
            if accelerator.is_main_process:
                progress_bar.set_postfix({"loss": loss.item(), "step": global_step})
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    
                    # Save IP-Adapter weights
                    ip_proj_sd = ip_adapter.image_proj_model.state_dict()
                    adapter_sd = ip_adapter.adapter_modules.state_dict()
                    
                    torch.save({
                        "image_proj": ip_proj_sd,
                        "ip_adapter": adapter_sd,
                    }, os.path.join(save_path, "ip_adapter.bin"))
                    
                    # Also save full checkpoint for resume
                    accelerator.save_state(save_path)
                    
                    print(f"\nSaved checkpoint to {save_path}")
    
    # Final save
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "ip_adapter_hair_final.bin")
        ip_proj_sd = ip_adapter.image_proj_model.state_dict()
        adapter_sd = ip_adapter.adapter_modules.state_dict()
        
        torch.save({
            "image_proj": ip_proj_sd,
            "ip_adapter": adapter_sd,
        }, final_path)
        
        print(f"\nTraining complete! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
