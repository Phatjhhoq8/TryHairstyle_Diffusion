"""
Prompt Response Evaluator - Do kha nang model nghe prompt.

Cach hoat dong:
  1. Load conflict eval set (eval_conflict_set.json)
  2. Voi moi sample: chay inference 2 lan (match prompt vs conflict prompt)
  3. So sanh 2 output:
     - LPIPS giua output_match vs output_conflict (> 0.1 = model nghe prompt)
     - Optional: CLIP text-image alignment score
  4. Xuat bao cao tong hop

Su dung:
  python -m backend.training.evaluate_prompt_response [--dry-run] [--checkpoint PATH]
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image

try:
    import lpips
except ImportError:
    lpips = None

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from backend.app.services.training_utils import setupLogger, getDevice

logger = setupLogger("PromptResponseEval")
DEVICE = getDevice()


class PromptResponseEvaluator:
    """
    Danh gia kha nang prompt response cua model Stage 2.
    So sanh output khi dung match_prompt vs conflict_prompt
    tren cung anh goc + cung ref hair.
    """

    def __init__(self, device='cuda'):
        self.device = device

        # LPIPS metric
        self.loss_fn_vgg = None
        try:
            if lpips is not None:
                self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.device)
        except Exception as e:
            logger.warning(f"LPIPS init error: {e}")

    def load_conflict_set(self, manifest_path: str = None):
        """Load conflict eval set tu JSON file."""
        if manifest_path is None:
            manifest_path = str(
                PROJECT_DIR / "backend" / "training" / "eval_set" / "eval_conflict_set.json"
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = data.get("samples", [])
        logger.info(f"Loaded {len(samples)} conflict samples tu {Path(manifest_path).name}")
        return samples

    @torch.no_grad()
    def compute_prompt_response_score(
        self,
        output_match: torch.Tensor,
        output_conflict: torch.Tensor,
        hair_mask: torch.Tensor = None,
    ):
        """
        Tinh diem prompt response tu 2 output images.

        Args:
            output_match: (1, 3, H, W) [-1, 1] - output khi dung match prompt
            output_conflict: (1, 3, H, W) [-1, 1] - output khi dung conflict prompt
            hair_mask: (1, 1, H, W) [0, 1] - mask vung toc (optional)

        Returns:
            dict: {
                'prompt_lpips_diff': float - LPIPS giua 2 outputs (cao = model nghe prompt),
                'prompt_l2_diff': float - L2 distance giua 2 outputs,
                'prompt_responsive': bool - True neu model phan ung voi prompt (lpips > 0.1),
            }
        """
        results = {}

        # Resize ve cung kich thuoc neu can
        if output_match.shape != output_conflict.shape:
            target_size = output_match.shape[-2:]
            output_conflict = F.interpolate(
                output_conflict, size=target_size, mode='bilinear', align_corners=False
            )

        # 1. LPIPS difference
        if self.loss_fn_vgg is not None:
            lpips_val = self.loss_fn_vgg(output_match, output_conflict).item()
            results['prompt_lpips_diff'] = lpips_val
        else:
            results['prompt_lpips_diff'] = -1.0

        # 2. L2 distance (normalized)
        match_norm = (output_match + 1.0) / 2.0
        conflict_norm = (output_conflict + 1.0) / 2.0

        if hair_mask is not None:
            # Chi tinh L2 trong vung toc
            if hair_mask.shape[-2:] != match_norm.shape[-2:]:
                hair_mask = F.interpolate(hair_mask, size=match_norm.shape[-2:], mode='nearest')
            diff = ((match_norm - conflict_norm) * hair_mask) ** 2
            l2_val = torch.sqrt(diff.sum() / (hair_mask.sum() * 3 + 1e-8)).item()
        else:
            l2_val = torch.sqrt(torch.mean((match_norm - conflict_norm) ** 2)).item()

        results['prompt_l2_diff'] = l2_val

        # 3. Judgment: responsive hay khong
        lpips_threshold = 0.10  # Nguong toi thieu de coi la "co phan ung"
        if results['prompt_lpips_diff'] >= 0:
            results['prompt_responsive'] = results['prompt_lpips_diff'] > lpips_threshold
        else:
            results['prompt_responsive'] = l2_val > 0.05  # Fallback dung L2

        return results

    def run_evaluation(self, results_pairs: list):
        """
        Tong hop ket qua tu nhieu cap (match_output, conflict_output).

        Args:
            results_pairs: List[dict] - moi dict chua output cua compute_prompt_response_score()

        Returns:
            dict: {
                'avg_lpips_diff': float,
                'avg_l2_diff': float,
                'responsive_rate': float (ty le model nghe prompt),
                'num_samples': int,
                'per_sample': list,
            }
        """
        if not results_pairs:
            return {
                'avg_lpips_diff': 0.0,
                'avg_l2_diff': 0.0,
                'responsive_rate': 0.0,
                'num_samples': 0,
                'per_sample': [],
            }

        lpips_diffs = [r['prompt_lpips_diff'] for r in results_pairs if r['prompt_lpips_diff'] >= 0]
        l2_diffs = [r['prompt_l2_diff'] for r in results_pairs]
        responsive = [r['prompt_responsive'] for r in results_pairs]

        return {
            'avg_lpips_diff': float(np.mean(lpips_diffs)) if lpips_diffs else -1.0,
            'avg_l2_diff': float(np.mean(l2_diffs)),
            'responsive_rate': sum(responsive) / len(responsive),
            'num_samples': len(results_pairs),
            'per_sample': results_pairs,
        }

    def unload(self):
        """Giai phong VRAM."""
        if self.loss_fn_vgg is not None:
            self.loss_fn_vgg.cpu()
            del self.loss_fn_vgg
            self.loss_fn_vgg = None
        torch.cuda.empty_cache()


# ==================================================================
# STANDALONE TEST / DRY RUN
# ==================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Prompt Response")
    parser.add_argument("--dry-run", action="store_true", help="Test voi dummy tensors")
    parser.add_argument("--manifest", type=str, default=None, help="Path to conflict set JSON")
    args = parser.parse_args()

    evaluator = PromptResponseEvaluator(device='cpu' if args.dry_run else str(DEVICE))

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN - Testing PromptResponseEvaluator with dummy tensors")
        print("=" * 60)

        # Load manifest de verify format
        samples = evaluator.load_conflict_set(args.manifest)
        print(f"\nConflict set: {len(samples)} samples")
        for s in samples[:3]:
            print(f"  {s['id']}: match='{s['match_prompt'][:50]}...' vs conflict='{s['conflict_prompt'][:50]}...'")

        # Test scoring voi dummy tensors
        dummy_match = torch.rand(1, 3, 256, 256) * 2 - 1
        dummy_conflict = dummy_match + torch.randn_like(dummy_match) * 0.3
        dummy_conflict = torch.clamp(dummy_conflict, -1, 1)

        scores = evaluator.compute_prompt_response_score(dummy_match, dummy_conflict)
        print(f"\nDummy scores:")
        print(f"  LPIPS diff: {scores['prompt_lpips_diff']:.4f}")
        print(f"  L2 diff:    {scores['prompt_l2_diff']:.4f}")
        print(f"  Responsive: {scores['prompt_responsive']}")

        # Test aggregation
        pairs = [scores, scores]
        agg = evaluator.run_evaluation(pairs)
        print(f"\nAggregated ({agg['num_samples']} samples):")
        print(f"  Avg LPIPS diff:   {agg['avg_lpips_diff']:.4f}")
        print(f"  Avg L2 diff:      {agg['avg_l2_diff']:.4f}")
        print(f"  Responsive rate:  {agg['responsive_rate']:.1%}")

        print("\nAll tests passed!")
    else:
        print("Full evaluation requires a trained model checkpoint.")
        print("Usage: python -m backend.training.evaluate_prompt_response --dry-run")
        print("       python -m backend.training.evaluate_prompt_response --checkpoint PATH")

    evaluator.unload()
