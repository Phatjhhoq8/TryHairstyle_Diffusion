"""
📊 Offline Benchmark — Đánh giá nhiều checkpoint song song, ghi leaderboard.

Cách dùng:
    python -m backend.training.evaluate_checkpoints \
        --checkpoints_dir /path/to/checkpoints \
        --eval_set_dir   backend/training/eval_set \
        --results_dir    backend/training/results

Hoặc chạy mặc định (sẽ đọc đường dẫn từ eval_config.json).
Output:
    results/<checkpoint_name>/per_sample.json
    results/<checkpoint_name>/summary.json
    results/leaderboard.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from backend.app.services.training_utils import setupLogger, getDevice
from backend.training.evaluate import HairEvaluator, load_eval_config

logger = setupLogger("EvalCheckpoints")

# ============================================================
# 1. Đọc manifest.json (danh sách ảnh eval cố định)
# ============================================================
def load_eval_manifest(eval_set_dir: Path):
    """Đọc file manifest.json mô tả các mẫu đánh giá.

    Cấu trúc manifest.json:
    {
      "samples": [
        {
          "id": "easy_001",
          "original_image": "original/001.png",
          "reference_hair_image": "reference/001.png",
          "mask_image": "optional_masks/001.png",  // tùy chọn
          "group": "easy",
          "bbox": [y1, y2, x1, x2],               // tùy chọn
          "notes": ""
        },
        ...
      ]
    }
    """
    manifest_path = eval_set_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"❌ Không tìm thấy {manifest_path}")
        logger.info("💡 Hãy tạo bộ eval set theo cấu trúc:")
        logger.info("   eval_set/manifest.json")
        logger.info("   eval_set/original/   (ảnh gốc)")
        logger.info("   eval_set/reference/  (ảnh mẫu tóc)")
        logger.info("   eval_set/optional_masks/ (hair mask, tùy chọn)")
        return []

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest.get("samples", [])
    logger.info(f"📋 Đọc manifest: {len(samples)} mẫu eval")
    return samples


def load_image_tensor(image_path: Path, target_size=(512, 512)):
    """Load ảnh thành tensor [-1, 1], shape (1, 3, H, W)."""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(target_size, antialias=True),
        transforms.ToTensor(),        # → [0, 1]
        transforms.Normalize([0.5], [0.5]),  # → [-1, 1]
    ])
    return transform(img).unsqueeze(0)  # (1, 3, H, W)


def load_mask_tensor(mask_path: Path, target_size=(512, 512)):
    """Load mask thành tensor [0, 1], shape (1, 1, H, W)."""
    img = Image.open(mask_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize(target_size, antialias=True),
        transforms.ToTensor(),  # → [0, 1]
    ])
    return transform(img).unsqueeze(0)  # (1, 1, H, W)


# ============================================================
# 2. Chạy inference cho 1 checkpoint (mock hoặc thật)
# ============================================================
def run_inference_for_checkpoint(checkpoint_path: str, sample: dict,
                                  eval_set_dir: Path, device: str):
    """
    Chạy inference cho 1 mẫu bằng checkpoint cụ thể.

    Trong phiên bản hiện tại, nếu đã có ảnh kết quả pre-generated
    (lưu sẵn trong results/<checkpoint>/generated/), sẽ đọc trực tiếp.
    Nếu chưa, sẽ cần tích hợp pipeline inference thực tế.

    Returns:
        dict: {
            'original_img': Tensor, 'result_img': Tensor,
            'reference_hair_img': Tensor, 'hair_mask': Tensor or None,
            'face_bbox': tuple or None, 'runtime_seconds': float
        }
        hoặc None nếu lỗi.
    """
    try:
        # Load original image
        orig_path = eval_set_dir / sample['original_image']
        if not orig_path.exists():
            logger.warning(f"  ⚠️ Không tìm thấy: {orig_path}")
            return None
        original_img = load_image_tensor(orig_path).to(device)

        # Load reference hair image
        ref_path = eval_set_dir / sample['reference_hair_image']
        if not ref_path.exists():
            logger.warning(f"  ⚠️ Không tìm thấy: {ref_path}")
            return None
        reference_img = load_image_tensor(ref_path).to(device)

        # Load mask (tùy chọn)
        hair_mask = None
        mask_rel = sample.get('mask_image')
        if mask_rel:
            mask_path = eval_set_dir / mask_rel
            if mask_path.exists():
                hair_mask = load_mask_tensor(mask_path).to(device)

        # Face bbox (tùy chọn)
        face_bbox = sample.get('bbox')
        if face_bbox:
            face_bbox = tuple(face_bbox)

        # --- Kiểm tra kết quả pre-generated ---
        # Nếu đã có ảnh result sẵn, đọc thay vì chạy inference
        ckpt_name = Path(checkpoint_path).stem
        pre_gen_dir = eval_set_dir.parent / "results" / ckpt_name / "generated"
        pre_gen_path = pre_gen_dir / f"{sample['id']}.png"

        start_time = time.time()

        if pre_gen_path.exists():
            result_img = load_image_tensor(pre_gen_path).to(device)
            runtime = 0.0  # Ảnh đã sinh sẵn, runtime = 0
        else:
            # TODO: Tích hợp pipeline inference thực tế ở đây
            # Hiện tại sử dụng placeholder: result = original (self-reconstruction test)
            logger.warning(f"  ⚠️ Chưa có ảnh pre-gen cho {sample['id']}, dùng original làm placeholder")
            result_img = original_img.clone()
            runtime = time.time() - start_time

        return {
            'original_img': original_img,
            'result_img': result_img,
            'reference_hair_img': reference_img,
            'hair_mask': hair_mask,
            'face_bbox': face_bbox,
            'runtime_seconds': runtime,
        }

    except Exception as e:
        logger.error(f"  ❌ Lỗi inference sample {sample.get('id', '?')}: {e}")
        return None


# ============================================================
# 3. Pipeline chính: Benchmark nhiều checkpoint
# ============================================================
def benchmark_checkpoints(checkpoints_dir: Path, eval_set_dir: Path,
                           results_dir: Path, device: str = 'cuda'):
    """
    Quét tất cả .safetensors trong checkpoints_dir,
    chạy đánh giá trên eval set cố định, ghi leaderboard.
    """
    # Tìm checkpoint files. Ưu tiên LoRA, fallback sang full model.
    ckpt_files = sorted(checkpoints_dir.glob("lora_*.safetensors"))
    if not ckpt_files:
        ckpt_files = sorted(
            p for p in checkpoints_dir.glob("*.safetensors")
            if "injector" not in p.name.lower()
        )
    if not ckpt_files:
        logger.error(f"❌ Không tìm thấy checkpoint .safetensors hợp lệ trong {checkpoints_dir}")
        return

    logger.info(f"🔍 Tìm thấy {len(ckpt_files)} checkpoint(s):")
    for f in ckpt_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  → {f.name} ({size_mb:.1f} MB)")

    # Load manifest
    samples = load_eval_manifest(eval_set_dir)
    if not samples:
        logger.error("❌ Eval set rỗng. Cần chuẩn bị manifest.json trước.")
        return

    # Khởi tạo evaluator
    evaluator = HairEvaluator(device=device)

    # Kết quả tổng hợp
    leaderboard = []

    for ckpt_path in ckpt_files:
        ckpt_name = ckpt_path.stem
        logger.info(f"\n{'='*60}")
        logger.info(f"📦 Đánh giá: {ckpt_name}")
        logger.info(f"{'='*60}")

        ckpt_results_dir = results_dir / ckpt_name
        ckpt_results_dir.mkdir(parents=True, exist_ok=True)

        per_sample_results = []
        success_count = 0

        for idx, sample in enumerate(samples):
            sample_id = sample.get('id', f'sample_{idx}')
            logger.info(f"  [{idx+1}/{len(samples)}] {sample_id} (group: {sample.get('group', 'N/A')})")

            # Chạy inference
            inference_result = run_inference_for_checkpoint(
                str(ckpt_path), sample, eval_set_dir, device
            )

            if inference_result is None:
                per_sample_results.append({
                    'id': sample_id, 'group': sample.get('group', 'unknown'),
                    'success': False, 'error': 'Inference failed'
                })
                continue

            # Chạy đánh giá đầy đủ
            try:
                metrics = evaluator.run_full_evaluation(inference_result)
                metrics['id'] = sample_id
                metrics['group'] = sample.get('group', 'unknown')
                per_sample_results.append(metrics)
                success_count += 1

                # Log nhanh
                id_str = f"{metrics.get('identity_similarity', -1):.4f}"
                lpips_str = f"{metrics.get('hair_lpips', -1):.4f}"
                bg_str = f"{metrics.get('background_psnr', metrics.get('bg_psnr', -1)):.1f}dB"
                logger.info(f"    → ID={id_str} | LPIPS={lpips_str} | BG={bg_str}")

            except Exception as e:
                logger.error(f"    ❌ Evaluation lỗi: {e}")
                per_sample_results.append({
                    'id': sample_id, 'group': sample.get('group', 'unknown'),
                    'success': False, 'error': str(e)
                })

            # Cleanup VRAM sau mỗi sample
            if inference_result:
                for k, v in inference_result.items():
                    if isinstance(v, torch.Tensor):
                        del v
                torch.cuda.empty_cache()

        # Lưu per_sample.json
        per_sample_path = ckpt_results_dir / "per_sample.json"
        _save_json(per_sample_results, per_sample_path)

        # Tổng hợp metrics cho checkpoint này
        successful_results = [r for r in per_sample_results if r.get('success', False)]
        summary = evaluator.aggregate_metrics(successful_results)
        summary['checkpoint'] = ckpt_name
        summary['checkpoint_file'] = str(ckpt_path)

        # Lưu summary.json
        summary_path = ckpt_results_dir / "summary.json"
        _save_json(summary, summary_path)

        logger.info(f"\n  📊 {ckpt_name} Summary:")
        logger.info(f"     Final Score: {summary['final_score']}")
        logger.info(f"     Success Rate: {summary['success_rate']}")
        if 'per_group_score' in summary:
            for group, score in summary['per_group_score'].items():
                logger.info(f"     {group}: {score:.4f}")

        leaderboard.append(summary)

    # Giải phóng VRAM
    evaluator.unload_heavy_models()

    # Sắp xếp leaderboard theo final_score giảm dần
    leaderboard.sort(key=lambda x: x.get('final_score', 0), reverse=True)

    # Lưu leaderboard.json
    leaderboard_path = results_dir / "leaderboard.json"
    _save_json(leaderboard, leaderboard_path)

    # In bảng xếp hạng
    logger.info(f"\n{'='*60}")
    logger.info(f"🏆 LEADERBOARD")
    logger.info(f"{'='*60}")
    logger.info(f"{'Rank':<6}{'Checkpoint':<30}{'Score':<10}{'Success':<10}")
    logger.info(f"{'-'*56}")
    for rank, entry in enumerate(leaderboard, 1):
        name = entry.get('checkpoint', '?')
        score = entry.get('final_score', 0)
        success = entry.get('success_rate', 0)
        marker = " ⭐" if rank == 1 else ""
        logger.info(f"{rank:<6}{name:<30}{score:<10.4f}{success:<10.4f}{marker}")

    logger.info(f"\n✅ Leaderboard đã lưu: {leaderboard_path}")
    return leaderboard


def _save_json(data, path):
    """Lưu dict/list ra file JSON, tự convert numpy types."""
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return obj

    # Deep convert
    if isinstance(data, list):
        clean = [{k: _convert(v) for k, v in item.items()} if isinstance(item, dict) else item
                 for item in data]
    elif isinstance(data, dict):
        clean = {k: _convert(v) if not isinstance(v, dict)
                 else {kk: _convert(vv) for kk, vv in v.items()}
                 for k, v in data.items()}
    else:
        clean = data

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False, default=str)


# ============================================================
# 4. CLI Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Offline Benchmark: đánh giá nhiều checkpoint, xuất leaderboard."
    )
    parser.add_argument("--checkpoints_dir", type=str, default=None,
                        help="Thư mục chứa file lora_*.safetensors")
    parser.add_argument("--eval_set_dir", type=str, default=None,
                        help="Thư mục chứa manifest.json và ảnh eval")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Thư mục lưu kết quả benchmark")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda hoặc cpu (mặc định: tự detect)")

    args = parser.parse_args()

    # Đọc config
    config = load_eval_config()

    # Resolve paths
    checkpoints_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else \
                      Path(__file__).parent / "checkpoints"
    eval_set_dir = Path(args.eval_set_dir) if args.eval_set_dir else \
                   PROJECT_DIR / config.get("eval_set_dir", "backend/training/eval_set")
    results_dir = Path(args.results_dir) if args.results_dir else \
                  PROJECT_DIR / config.get("results_dir", "backend/training/results")

    results_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or str(getDevice())
    logger.info(f"📍 Checkpoints: {checkpoints_dir}")
    logger.info(f"📍 Eval Set:    {eval_set_dir}")
    logger.info(f"📍 Results:     {results_dir}")
    logger.info(f"📍 Device:      {device}")

    benchmark_checkpoints(checkpoints_dir, eval_set_dir, results_dir, device)


if __name__ == "__main__":
    main()
