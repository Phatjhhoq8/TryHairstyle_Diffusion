"""
📦 So sánh & Export Checkpoint
Dùng khi muốn export ngay mà không cần chờ hết epoch.
Tự động chọn checkpoint 13-ch tốt nhất.
"""
import os, sys, shutil
from pathlib import Path

PROJECT_DIR = Path("/content/TryHairStyle")
sys.path.append(str(PROJECT_DIR))

import torch
from safetensors.torch import load_file as load_safetensors

from backend.app.services.training_utils import setupLogger

logger = setupLogger("ExportCompare")

# ============================================================
# 1. Download cả 2 checkpoint từ HF Hub
# ============================================================
HF_TOKEN = os.environ.get("HUGFACE_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
CKPT_DIR = Path("/tmp/training_checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def download_from_hf():
    from huggingface_hub import hf_hub_download
    files = ["lora_best.safetensors", "lora_latest.safetensors",
             "injector_best.safetensors", "injector_latest.safetensors"]
    downloaded = {}
    for fname in files:
        local_path = CKPT_DIR / fname
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID, repo_type="dataset",
                filename=f"checkpoints/{fname}", token=HF_TOKEN,
                local_dir=str(CKPT_DIR), local_dir_use_symlinks=False,
            )
            hf_path = CKPT_DIR / "checkpoints" / fname
            if hf_path.exists():
                shutil.move(str(hf_path), str(local_path))
            if local_path.exists():
                downloaded[fname] = local_path
        except Exception:
            pass
    return downloaded

print("⏳ Downloading checkpoints từ HF Hub...")
files = download_from_hf()
print(f"✅ Downloaded: {list(files.keys())}")

# ============================================================
# 2. So sánh 2 checkpoint
# ============================================================
def analyze_checkpoint(path):
    """Phân tích checkpoint: channels, params, có NaN không."""
    sd = load_safetensors(str(path))
    
    # Detect conv_in channels
    conv_weight = sd.get("conv_in.weight")
    channels = conv_weight.shape[1] if conv_weight is not None else "?"
    
    # Đếm params
    total_params = sum(t.numel() for t in sd.values())
    lora_keys = sum(1 for k in sd if "lora" in k.lower())
    has_nan = any(torch.isnan(t).any() for t in sd.values())
    
    # Size file
    size_mb = os.path.getsize(str(path)) / (1024 * 1024)
    
    return {
        "channels": channels,
        "params_M": round(total_params / 1e6, 1),
        "lora_keys": lora_keys,
        "has_nan": has_nan,
        "size_mb": round(size_mb, 1),
    }

print("\n" + "=" * 60)
print("📊 SO SÁNH CHECKPOINTS")
print("=" * 60)

results = {}
for name in ["lora_best.safetensors", "lora_latest.safetensors"]:
    if name in files:
        info = analyze_checkpoint(files[name])
        results[name] = info
        tag = "⭐ BEST" if "best" in name else "🔄 LATEST"
        print(f"\n{tag}: {name}")
        print(f"  → Channels:  {info['channels']}-ch")
        print(f"  → Params:    {info['params_M']}M")
        print(f"  → LoRA keys: {info['lora_keys']}")
        print(f"  → Size:      {info['size_mb']} MB")
        print(f"  → NaN:       {'❌ CÓ' if info['has_nan'] else '✅ Không'}")

# ============================================================
# 3. Chọn checkpoint phù hợp để export
# ============================================================
print("\n" + "=" * 60)
print("🎯 QUYẾT ĐỊNH EXPORT")
print("=" * 60)

chosen = None
chosen_name = None

best_info = results.get("lora_best.safetensors")
latest_info = results.get("lora_latest.safetensors")

if best_info and latest_info:
    if best_info["channels"] == 13:
        # Cả 2 đều 13-ch → chọn best
        chosen_name = "lora_best.safetensors"
        reason = "lora_best đã là 13-ch (val_loss tốt nhất)"
    elif latest_info["channels"] == 13:
        # Best = 9-ch, Latest = 13-ch → chọn latest
        chosen_name = "lora_latest.safetensors"
        reason = "lora_best vẫn 9-ch (epoch cũ), chọn lora_latest (13-ch, mới nhất)"
    else:
        # Cả 2 đều 9-ch??
        chosen_name = "lora_best.safetensors"
        reason = "Cả 2 đều 9-ch, chọn best (val_loss tốt hơn)"
elif best_info:
    chosen_name = "lora_best.safetensors"
    reason = "Chỉ có lora_best"
elif latest_info:
    chosen_name = "lora_latest.safetensors"
    reason = "Chỉ có lora_latest"
else:
    print("❌ Không tìm thấy checkpoint nào!")
    sys.exit(1)

chosen = files[chosen_name]
print(f"\n→ Chọn: {chosen_name}")
print(f"→ Lý do: {reason}")

# ============================================================
# 4. Export
# ============================================================
print("\n" + "=" * 60)
print("🚀 EXPORT MODEL")
print("=" * 60)

from backend.training.export_model import CheckpointManager

manager = CheckpointManager()

# Chạy test trước
passed, metrics = manager.test_checkpoint(str(chosen))

if passed:
    print(f"\n✅ Checkpoint hợp lệ → Bắt đầu export...")
    success = manager.export_to_production(str(chosen))
    if success:
        print(f"\n🎉 Export thành công! Model đã sẵn sàng cho inference.")
    else:
        print(f"\n❌ Export thất bại!")
else:
    print(f"\n❌ Checkpoint không hợp lệ: {metrics}")
