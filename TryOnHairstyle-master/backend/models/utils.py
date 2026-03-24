"""
Utility functions for HairFusion models.
These are referenced by cldm/cldm.py but only used in commented-out / training code.
"""
import torch
import numpy as np
from PIL import Image


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """Convert a torch Tensor into a numpy image array (H, W, C) [0, 255]."""
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to [0, 1]
    if tensor.dim() == 3:
        img_np = tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC
    elif tensor.dim() == 2:
        img_np = tensor.numpy()
    else:
        raise ValueError(f"Unexpected tensor dim: {tensor.dim()}")
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round().astype(np.uint8)
    return img_np


def resize_mask(mask, h, w):
    """Resize a binary mask to (h, w) using nearest interpolation."""
    if isinstance(mask, torch.Tensor):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
        mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
        return mask.squeeze()
    else:
        # numpy / PIL
        img = Image.fromarray(mask.astype(np.uint8))
        img = img.resize((w, h), Image.NEAREST)
        return np.array(img)
