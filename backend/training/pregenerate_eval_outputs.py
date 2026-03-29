import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import pipeline

from backend.app.config import model_paths, settings
from backend.app.services.face import FaceInfoService
from backend.app.services.mask import SegmentationService
from backend.app.services.diffusion import HairDiffusionService


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_manifest(eval_set_dir: Path):
    manifest_path = eval_set_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f).get("samples", [])


def ensure_rgb(image_path: Path):
    return Image.open(image_path).convert("RGB")


def generate_sample(sample, eval_set_dir: Path, output_dir: Path, face_service, mask_service, depth_estimator, diffusion_service, prompt: str, overwrite: bool):
    sample_id = sample["id"]
    out_path = output_dir / f"{sample_id}.png"
    if out_path.exists() and not overwrite:
        return {"id": sample_id, "status": "skipped", "output": str(out_path)}

    orig_path = eval_set_dir / sample["original_image"]
    ref_path = eval_set_dir / sample["reference_hair_image"]

    user_image = ensure_rgb(orig_path)
    hair_image = ensure_rgb(ref_path)
    user_cv2 = cv2.cvtColor(cv2.imread(str(orig_path)), cv2.COLOR_BGR2RGB)
    user_cv2 = cv2.cvtColor(user_cv2, cv2.COLOR_RGB2BGR)

    start = time.time()
    face_info = face_service.analyze(user_cv2)

    masks = mask_service.get_hair_and_face_mask(user_image)
    hair_mask = masks["hair_mask"]
    face_mask = masks["face_mask"]
    ref_masks = mask_service.get_hair_and_face_mask(hair_image)
    ref_hair_mask = ref_masks["hair_mask"]
    expanded_mask = mask_service.expand_hair_mask(hair_mask, face_mask, ref_hair_mask, face_info)

    depth_map = depth_estimator(user_image)["depth"]
    result = diffusion_service.generate(
        base_image=user_image,
        mask_image=expanded_mask,
        control_image=depth_map,
        ref_hair_image=hair_image,
        prompt=prompt,
    )
    if result.size != user_image.size:
        result = result.resize(user_image.size, Image.LANCZOS)

    output_dir.mkdir(parents=True, exist_ok=True)
    result.save(out_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "id": sample_id,
        "status": "generated",
        "output": str(out_path),
        "runtime_seconds": round(time.time() - start, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-generate eval outputs for benchmark")
    parser.add_argument("--checkpoint", type=str, default="backend/training/models/deep_hair_v1.safetensors")
    parser.add_argument("--eval_set_dir", type=str, default="backend/training/eval_set")
    parser.add_argument("--output_dir", type=str, default="backend/training/results/deep_hair_v1/generated")
    parser.add_argument("--prompt", type=str, default="high quality realistic hairstyle")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    eval_set_dir = Path(args.eval_set_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    model_paths.CUSTOM_INPAINTING_MODEL = str(checkpoint_path)
    settings.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Eval set:   {eval_set_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device:     {settings.DEVICE}")

    samples = load_manifest(eval_set_dir)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    print(f"Samples:    {len(samples)}")

    face_service = FaceInfoService()
    mask_service = SegmentationService()
    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
    diffusion_service = HairDiffusionService()

    results = []
    failures = []
    for idx, sample in enumerate(samples, 1):
        print(f"[{idx}/{len(samples)}] {sample['id']}")
        try:
            info = generate_sample(
                sample,
                eval_set_dir,
                output_dir,
                face_service,
                mask_service,
                depth_estimator,
                diffusion_service,
                args.prompt,
                args.overwrite,
            )
            print(f"  -> {info['status']}: {info['output']}")
            results.append(info)
        except Exception as e:
            msg = {"id": sample["id"], "error": str(e)}
            print(f"  -> failed: {e}")
            failures.append(msg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "generated": len([r for r in results if r["status"] == "generated"]),
        "skipped": len([r for r in results if r["status"] == "skipped"]),
        "failed": len(failures),
        "failures": failures,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir.parent / "generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
