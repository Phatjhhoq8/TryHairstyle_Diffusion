
import os
from safetensors.torch import load_file
import torch

path_bin = r"c:\Users\Admin\Desktop\TryHairStyle\backend\models\stable-diffusion\sd15\unet\diffusion_pytorch_model.bin"
path_bin_fp16 = r"c:\Users\Admin\Desktop\TryHairStyle\backend\models\stable-diffusion\sd15\unet\diffusion_pytorch_model.fp16.bin"

print(f"Checking Bin FP32: {path_bin}")
try:
    # Use map_location to cpu to save gpu memory during check
    state_dict = torch.load(path_bin, map_location="cpu")
    print("Bin FP32 Loaded Successfully keys:", len(state_dict))
except Exception as e:
    print(f"Bin FP32 Failed: {e}")

print(f"Checking Bin FP16: {path_bin_fp16}")
try:
    state_dict = torch.load(path_bin_fp16, map_location="cpu")
    print("Bin FP16 Loaded Successfully keys:", len(state_dict))
except Exception as e:
    print(f"Bin FP16 Failed: {e}")

