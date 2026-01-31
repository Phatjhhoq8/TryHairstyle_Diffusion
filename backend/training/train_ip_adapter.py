"""
Train IP-Adapter for Hair Style Transfer
Phase 1: Finetune IP-Adapter to understand hair textures and styles

Based on QUY_TRINH.md:
- Input: Ảnh tóc đã mask che mặt
- Task: Reconstruct lại chính bức ảnh đó dựa trên embedding của tóc
- Mục tiêu: Giúp model hiểu sâu sắc về texture tóc (xoăn, thẳng, nhuộm, highlight)
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from PIL import Image
import wandb
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train IP-Adapter for Hair Style")
    parser.add_argument("--data_dir", type=str, default="backend/dataset/khairstyle/training",
                        help="Path to K-Hairstyle dataset")
    parser.add_argument("--output_dir", type=str, default="backend/output/checkpoints",
                        help="Path to save checkpoints")
    parser.add_argument("--pretrained_model", type=str, 
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Pretrained SDXL model")
    parser.add_argument("--ip_adapter_path", type=str,
                        default="h94/IP-Adapter",
                        help="Pretrained IP-Adapter path")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (small due to 12GB VRAM)")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Accumulate gradients to simulate larger batch")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log to Weights & Biases")
    return parser.parse_args()


class HairStyleDataset(torch.utils.data.Dataset):
    """
    Dataset for hair style training
    Returns pairs of (masked_image, original_image) for reconstruction task
    """
    def __init__(self, data_dir, image_size=1024):  # SDXL uses 1024x1024
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_paths = self._load_image_paths()
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        
    def _load_image_paths(self):
        """Load all image paths from dataset directory"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        paths = []
        if os.path.exists(self.data_dir):
            for root, _, files in os.walk(self.data_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_extensions:
                        paths.append(os.path.join(root, f))
        return paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Process for CLIP (IP-Adapter input)
        clip_image = self.clip_processor(
            images=image, 
            return_tensors="pt"
        ).pixel_values[0]
        
        # Convert to tensor for VAE
        image_tensor = torch.from_numpy(
            np.array(image).astype(np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1)
        
        return {
            "pixel_values": image_tensor,
            "clip_image": clip_image,
        }


def train(args):
    """Main training loop"""
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    
    # Initialize wandb
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(project="tryhairly-ip-adapter", config=vars(args))
    
    # Load models
    print("Loading models...")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, 
        subfolder="vae",
        torch_dtype=torch.float16
    )
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16
    )
    
    # Freeze VAE and image encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    # Load dataset
    print("Loading dataset...")
    dataset = HairStyleDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    if len(dataset) == 0:
        print(f"WARNING: No images found in {args.data_dir}")
        print("Please add training images to the dataset directory.")
        return
    
    print(f"Found {len(dataset)} images for training")
    
    # TODO: Load IP-Adapter and create trainable projection layers
    # This is a simplified version - full implementation requires
    # IP-Adapter architecture from the official repo
    
    # Prepare with accelerator
    vae, image_encoder, dataloader = accelerator.prepare(
        vae, image_encoder, dataloader
    )
    
    # Training loop
    global_step = 0
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_main_process
        )
        
        for batch in progress_bar:
            with accelerator.accumulate():
                # Get CLIP embeddings
                clip_embeds = image_encoder(batch["clip_image"]).image_embeds
                
                # Encode images to latents
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # TODO: Add noise and predict noise (standard diffusion training)
                # TODO: Compute loss and backprop
                
                # Placeholder loss for structure
                loss = torch.tensor(0.0, requires_grad=True)
                
                accelerator.backward(loss)
                
            global_step += 1
            
            # Log metrics
            if args.use_wandb and accelerator.is_main_process:
                wandb.log({"loss": loss.item(), "step": global_step})
            
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(
                        args.output_dir, 
                        f"checkpoint-{global_step}"
                    )
                    os.makedirs(save_path, exist_ok=True)
                    # TODO: Save IP-Adapter weights
                    print(f"Saved checkpoint to {save_path}")
    
    print("Training complete!")


if __name__ == "__main__":
    import numpy as np
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
