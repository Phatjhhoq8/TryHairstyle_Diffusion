"""
HairFusion Diffusion Service.
Encapsulates model loading, inference, and sampling into a service class.
Extracted from app.py / run_custom.py logic.
"""
import os
import argparse
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from einops import rearrange
from torchvision.transforms.functional import resize
from omegaconf import OmegaConf

# Library imports (via sys.path)
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict

# App imports
from backend.app.utils.image_utils import tensor2img
from backend.app.config import (
    CHECKPOINT_PATH, VAE_PATH, CONFIG_YAML_PATH,
    DEVICE, IMG_H, IMG_W, DATA_DIR
)


class HairDiffusionService:
    """Handles HairFusion model loading and inference."""

    def __init__(self, device=None):
        self.device = device or DEVICE

        print(f"Loading config from {CONFIG_YAML_PATH}...")
        self.config = OmegaConf.load(CONFIG_YAML_PATH)
        self.config.model.params.setdefault("use_VAEdownsample", False)
        self.config.model.params.setdefault("use_imageCLIP", False)
        self.config.model.params.setdefault("use_lastzc", False)
        self.config.model.params.setdefault("use_regdecoder", False)
        self.config.model.params.setdefault("use_pbe_weight", False)
        self.config.model.params.setdefault("u_cond_percent", 0.0)
        self.config.model.params.img_H = IMG_H
        self.config.model.params.img_W = IMG_W

        print(f"Loading model from {CHECKPOINT_PATH}...")
        self.model = create_model(config_path=CONFIG_YAML_PATH, config=self.config)
        self.model.load_state_dict(
            load_state_dict(CHECKPOINT_PATH, location="cpu"),
            strict=False
        )

        # Load VAE
        if os.path.exists(VAE_PATH):
            print(f"Loading VAE from {VAE_PATH}...")
            state_dict = load_state_dict(VAE_PATH, location="cpu")
            new_state_dict = {}
            for k, v in state_dict.items():
                if "first_stage_model." in k and "loss." not in k:
                    new_k = k.replace("first_stage_model.", "")
                    new_state_dict[new_k] = v.clone()
                elif "loss." not in k:
                    new_state_dict[k] = v.clone()
            self.model.first_stage_model.load_state_dict(new_state_dict, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.ddim_sampler = DDIMSampler(self.model)
        print("Model loaded successfully!")

    def run_inference(self, data_root_dir, steps=50, scale=5.0, batch_size=1):
        """
        Run full inference on a prepared dataset.
        Returns list of result images (numpy arrays, RGB).
        """
        from backend.app.services.dataset import CustomDataset

        args = argparse.Namespace()
        args.img_H = IMG_H
        args.img_W = IMG_W
        args.crop_scale = 4.0

        dataset = CustomDataset(
            args=args,
            data_root_dir=data_root_dir,
            img_H=IMG_H,
            img_W=IMG_W,
            default_prompt="",
            is_test=True,
        )
        dataloader = DataLoader(dataset, num_workers=0, shuffle=False, batch_size=batch_size)

        results = []

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                self.model.batch = batch

                z, c = self.model.get_input(
                    batch, self.config.model.params.first_stage_key
                )

                bs = z.shape[0]
                c_crossattn = c["c_crossattn"][0][:bs]
                if c_crossattn.ndim == 4:
                    c_crossattn = self.model.get_learned_conditioning(c_crossattn)
                    c["c_crossattn"] = [c_crossattn]

                # Setup Mask
                mask = batch["agn_mask"]
                if len(mask.shape) == 3:
                    mask = mask[..., None]
                mask = rearrange(mask, 'b h w c -> b c h w')
                mask = mask.to(memory_format=torch.contiguous_format).float()
                mask = resize(mask, (IMG_H // 8, IMG_W // 8))

                x0 = z

                # Unconditional Conditioning
                uc_cross = self.model.get_unconditional_conditioning(bs)
                uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
                if "first_stage_cond" in c:
                    uc_full["first_stage_cond"] = c["first_stage_cond"]

                # Sampling
                shape = (4, IMG_H // 8, IMG_W // 8)
                samples, _, _ = self.ddim_sampler.sample(
                    steps,
                    bs,
                    shape,
                    c,
                    eta=0.0,
                    mask=mask,
                    x0=x0,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_full,
                    verbose=False
                )

                x_samples = self.model.decode_first_stage(samples)

                for x_sample in x_samples:
                    x_sample_img = tensor2img(x_sample)
                    results.append(x_sample_img)

        return results
