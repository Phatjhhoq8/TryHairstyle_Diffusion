"""
So sánh & Export Checkpoint — Phiên bản nâng cấp.

Ưu tiên chọn checkpoint theo thứ tự:
  1. Đọc leaderboard.json (nếu đã chạy benchmark) → chọn Top 1 final_score
  2. Fallback: so sánh lora_best vs lora_latest theo val_loss và cấu trúc kênh

Cách dùng:
    python -m backend.training.export_compare
"""
import os, sys, shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

import torch
from safetensors.torch import load_file as load_safetensors

from backend.app.services.training_utils import setupLogger

logger = setupLogger("ExportCompare")

# ============================================================
# CONFIG
# ============================================================
HF_TOKEN = os.environ.get("HUGFACE_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
CKPT_DIR = Path("/tmp/training_checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PROJECT_DIR / "backend" / "training" / "results"
LOCAL_MODELS_DIR = PROJECT_DIR / "backend" / "training" / "models"

# ============================================================
# 1. Download checkpoints từ HF Hub
# ============================================================
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

print("Downloading checkpoints from HF Hub...")
files = download_from_hf()
print(f"Downloaded: {list(files.keys())}")

# Thêm checkpoint local nếu có (dùng khi benchmark chạy trên máy local)
for local_ckpt in LOCAL_MODELS_DIR.glob("*.safetensors"):
    if "injector" in local_ckpt.name.lower():
        continue
    files.setdefault(local_ckpt.name, local_ckpt)

# ============================================================
# 2. Phân tích checkpoint
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
print("CHECKPOINT COMPARISON")
print("=" * 60)

results = {}
for name in ["lora_best.safetensors", "lora_latest.safetensors"]:
    if name in files:
        info = analyze_checkpoint(files[name])
        results[name] = info
        tag = "BEST" if "best" in name else "LATEST"
        print(f"\n{tag}: {name}")
        print(f"  Channels:  {info['channels']}-ch")
        print(f"  Params:    {info['params_M']}M")
        print(f"  LoRA keys: {info['lora_keys']}")
        print(f"  Size:      {info['size_mb']} MB")
        print(f"  NaN:       {'YES' if info['has_nan'] else 'NO'}")

# ============================================================
# 3. Chọn checkpoint: ưu tiên Benchmark > Fallback val_loss
# ============================================================
print("\n" + "=" * 60)
print("EXPORT DECISION")
print("=" * 60)

chosen = None
chosen_name = None
reason = ""

# --- Ưu tiên 1: Đọc leaderboard.json từ benchmark offline ---
leaderboard_path = RESULTS_DIR / "leaderboard.json"
if leaderboard_path.exists():
    import json
    with open(leaderboard_path, "r", encoding="utf-8") as f:
        leaderboard = json.load(f)

    if leaderboard:
        top = leaderboard[0]  # Đã sort theo final_score giảm dần
        top_name = top.get('checkpoint', '')
        top_score = top.get('final_score', 0)
        top_file = top.get('checkpoint_file', '')

        # Tìm file checkpoint tương ứng
        # Tên trong leaderboard thường là "lora_best" hoặc "lora_latest"
        matching_file = f"{top_name}.safetensors"
        if matching_file in files:
            chosen = files[matching_file]
            chosen_name = matching_file
            reason = f"Benchmark leaderboard Top 1: {top_name} (Final Score = {top_score:.4f})"

            # In chi tiết
            print(f"\nBenchmark data available!")
            print(f"   Leaderboard Top 1: {top_name}")
            print(f"   Final Score: {top_score:.4f}")
            if 'per_group_score' in top:
                for group, score in top['per_group_score'].items():
                    print(f"     {group}: {score:.4f}")
        else:
            print(f"\nWarning: benchmark points to {top_name} but file is not available locally.")
            print(f"   Fallback về logic so sánh kênh...")

# --- Ưu tiên 2: Fallback — so sánh theo channels + val_loss ---
if chosen is None:
    print("\nNo benchmark data -> fallback to channels + val_loss")

    best_info = results.get("lora_best.safetensors")
    latest_info = results.get("lora_latest.safetensors")

    if best_info and latest_info:
        if best_info["channels"] == 13:
            chosen_name = "lora_best.safetensors"
            reason = "lora_best đã là 13-ch (val_loss tốt nhất)"
        elif latest_info["channels"] == 13:
            chosen_name = "lora_latest.safetensors"
            reason = "lora_best vẫn 9-ch (epoch cũ), chọn lora_latest (13-ch, mới nhất)"
        else:
            chosen_name = "lora_best.safetensors"
            reason = "Cả 2 đều 9-ch, chọn best (val_loss tốt hơn)"
    elif best_info:
        chosen_name = "lora_best.safetensors"
        reason = "Chỉ có lora_best"
    elif latest_info:
        chosen_name = "lora_latest.safetensors"
        reason = "Chỉ có lora_latest"
    else:
        print("No checkpoint found!")
        sys.exit(1)

    chosen = files[chosen_name]

print(f"\nChosen: {chosen_name}")
print(f"Reason: {reason}")

# ============================================================
# 4. Export
# ============================================================
print("\n" + "=" * 60)
print("EXPORT MODEL")
print("=" * 60)

from backend.training.export_model import CheckpointManager

manager = CheckpointManager()

# Chạy test trước
passed, metrics = manager.test_checkpoint(str(chosen))

if passed:
    print(f"\nCheckpoint is valid -> starting export...")
    success = manager.export_to_production(str(chosen))
    if success:
        print(f"\nExport completed successfully. Model is ready for inference.")
        print(f"   Selection reason: {reason}")
    else:
        print(f"\nExport failed!")
else:
    print(f"\nCheckpoint is invalid: {metrics}")
