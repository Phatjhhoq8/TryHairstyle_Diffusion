"""
Test CLI — Chạy inference từ dòng lệnh để kiểm tra model đã train.

Cách sử dụng:
    # Chọn ngẫu nhiên từ FFHQ (target) + K-Hairstyle (reference):
    python backend/training/test_cli.py --random

    # Chỉ định ảnh cụ thể:
    python backend/training/test_cli.py --target face.jpg --reference hair.jpg

    # Kết hợp: random target, chỉ định reference:
    python backend/training/test_cli.py --random --reference hair.jpg

    # Chỉ định checkpoint cụ thể:
    python backend/training/test_cli.py --random \\
        --checkpoint backend/training/checkpoints/stage2_epoch_5.safetensors

Nếu không chỉ định --checkpoint, tự tìm file best > latest > mới nhất.
"""

import os
import sys
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from backend.app.services.training_utils import setupLogger, getDevice, ensureDir

logger = setupLogger("TestCLI")
DEVICE = getDevice()

# Đường dẫn mặc định
LOCAL_SDXL_PATH = str(PROJECT_DIR / "backend" / "models" / "stable-diffusion" / "sd_xl_inpainting")
CHECKPOINTS_DIR = PROJECT_DIR / "backend" / "training" / "checkpoints"

# Dataset paths
FFHQ_DIR = PROJECT_DIR / "backend" / "data" / "dataset" / "ffhq"
KHAIRSTYLE_DIR = PROJECT_DIR / "backend" / "data" / "dataset" / "khairstyle" / "training" / "images"


def find_best_checkpoint():
    """Tìm checkpoint tốt nhất theo thứ tự ưu tiên: best > latest > mới nhất."""
    # 1. Best model
    best = CHECKPOINTS_DIR / "deep_hair_v1_best.safetensors"
    if best.exists():
        logger.info(f"Dùng BEST model: {best.name}")
        return str(best)

    # 2. Latest model
    latest = CHECKPOINTS_DIR / "deep_hair_v1_latest.safetensors"
    if latest.exists():
        logger.info(f"Dùng LATEST model: {latest.name}")
        return str(latest)

    # 3. Mới nhất theo thời gian
    import glob
    files = glob.glob(str(CHECKPOINTS_DIR / "*stage2*.safetensors"))
    if files:
        files.sort(key=os.path.getmtime, reverse=True)
        logger.info(f"Dùng checkpoint mới nhất: {Path(files[0]).name}")
        return files[0]

    return None


def load_model(checkpoint_path):
    """Load trained UNet + Injector + VAE."""
    from diffusers import AutoencoderKL, DDPMScheduler
    from safetensors.torch import load_file
    from backend.training.models.stage2_unet import HairInpaintingUNet, CrossAttentionInjector

    logger.info("=" * 50)
    logger.info("LOADING MODEL")
    logger.info("=" * 50)

    # VAE
    logger.info("  → Loading VAE (fp16)...")
    vae = AutoencoderKL.from_pretrained(
        LOCAL_SDXL_PATH, subfolder="vae", torch_dtype=torch.float16
    ).to(DEVICE).eval()
    vae.enable_slicing()

    # Noise Scheduler
    logger.info("  → Loading DDPMScheduler...")
    scheduler = DDPMScheduler.from_pretrained(LOCAL_SDXL_PATH, subfolder="scheduler")

    # UNet
    logger.info("  → Loading UNet 9-channel...")
    unet = HairInpaintingUNet().to(DEVICE)

    # Load trained weights
    logger.info(f"  → Loading checkpoint: {Path(checkpoint_path).name}")
    state_dict = load_file(checkpoint_path)
    unet.load_state_dict(state_dict, strict=False)
    unet.eval()

    # Injector
    injector = CrossAttentionInjector(unet.unet).to(DEVICE)
    inj_path = checkpoint_path.replace("deep_hair_v1_best", "injector_best") \
                              .replace("deep_hair_v1_latest", "injector_latest") \
                              .replace("stage2_", "injector_")
    if os.path.exists(inj_path):
        logger.info(f"  → Loading injector: {Path(inj_path).name}")
        inj_dict = load_file(inj_path)
        injector.load_state_dict(inj_dict, strict=False)
    injector.eval()

    logger.info("  ✅ Model loaded!")

    return {
        "unet": unet,
        "injector": injector,
        "vae": vae,
        "scheduler": scheduler,
        "vae_scale_factor": vae.config.scaling_factor,
    }


def get_hair_mask(image_pil):
    """Tạo hair mask bằng SegFormer."""
    logger.info("  → Tạo hair mask (SegFormer)...")
    try:
        from backend.app.services.mask import SegmentationService
        seg = SegmentationService()
        mask_pil = seg.get_mask(image_pil, target_class=17)  # 17 = hair
        return mask_pil
    except Exception as e:
        logger.warning(f"SegFormer lỗi: {e}. Tạo mask đơn giản (nửa trên ảnh)...")
        # Fallback: mask nửa trên ảnh (vùng tóc phổ biến)
        w, h = image_pil.size
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:h // 2, :] = 255  # Nửa trên
        return Image.fromarray(mask)


def get_identity_embedding(image_pil):
    """Lấy identity embedding bằng InsightFace."""
    logger.info("  → Trích xuất identity embedding...")
    try:
        import insightface
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="antelopev2",
            root=str(PROJECT_DIR / "backend"),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        app.prepare(ctx_id=0, det_size=(640, 640))

        img_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        faces = app.get(img_cv2)

        if faces:
            embedding = faces[0].embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            logger.info(f"  ✅ Embedding shape: {embedding.shape}")
            return embedding

    except Exception as e:
        logger.warning(f"InsightFace lỗi: {e}")

    # Fallback: zero vector
    logger.warning("  ⚠️ Không extract được embedding, dùng zero vector")
    return np.zeros(512, dtype=np.float32)


def create_bald_image(image_pil, mask_pil):
    """
    Tạo bald image bằng cách inpaint vùng tóc.
    Thay vùng tóc bằng skin color trung bình.
    """
    logger.info("  → Tạo bald image (inpaint)...")
    img = np.array(image_pil)
    mask = np.array(mask_pil)

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    # OpenCV inpaint
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bald_bgr = cv2.inpaint(img_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    bald_rgb = cv2.cvtColor(bald_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(bald_rgb)


def encode_text_prompt(prompt):
    """Encode text prompt bằng SDXL CLIP encoders."""
    logger.info(f"  → Encoding text prompt: '{prompt}'")

    from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

    tokenizer_1 = CLIPTokenizer.from_pretrained(LOCAL_SDXL_PATH, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(LOCAL_SDXL_PATH, subfolder="tokenizer_2")

    text_encoder_1 = CLIPTextModel.from_pretrained(
        LOCAL_SDXL_PATH, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(DEVICE).eval()

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        LOCAL_SDXL_PATH, subfolder="text_encoder_2", torch_dtype=torch.float16
    ).to(DEVICE).eval()

    with torch.no_grad():
        tokens_1 = tokenizer_1(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(DEVICE)

        tokens_2 = tokenizer_2(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(DEVICE)

        enc_1 = text_encoder_1(tokens_1, output_hidden_states=True)
        enc_2 = text_encoder_2(tokens_2, output_hidden_states=True)

        hidden_1 = enc_1.hidden_states[-2]  # (1, 77, 768)
        hidden_2 = enc_2.hidden_states[-2]  # (1, 77, 1280)

        prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1).float()  # (1, 77, 2048)
        pooled_embeds = enc_2.text_embeds.float()  # (1, 1280)

    # Giải phóng VRAM
    del text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2
    torch.cuda.empty_cache()

    return prompt_embeds, pooled_embeds


@torch.no_grad()
def run_inference(model, target_pil, mask_pil, bald_pil, id_embedding, prompt,
                  num_steps=30, guidance_scale=7.5):
    """
    Chạy full denoising loop.
    Input: target (GT), bald, mask, identity embedding, prompt
    Output: PIL Image — ảnh kết quả với tóc mới
    """
    from torchvision import transforms

    unet = model["unet"]
    vae = model["vae"]
    scheduler = model["scheduler"]
    injector = model["injector"]
    scale_factor = model["vae_scale_factor"]

    target_size = (1024, 1024)

    # Resize tất cả về 1024x1024
    target_pil = target_pil.resize(target_size, Image.LANCZOS)
    bald_pil = bald_pil.resize(target_size, Image.LANCZOS)
    mask_pil = mask_pil.resize(target_size, Image.NEAREST)

    # Transform
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    gt_tensor = img_transform(target_pil).unsqueeze(0).to(DEVICE)
    bald_tensor = img_transform(bald_pil).unsqueeze(0).to(DEVICE)

    # Mask tensor
    mask_np = np.array(mask_pil)
    if len(mask_np.shape) == 3:
        mask_np = mask_np[:, :, 0]
    mask_float = (mask_np / 255.0).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_float[np.newaxis, np.newaxis, ...]).to(DEVICE)

    # Encode text prompt
    prompt_embeds, pooled_embeds = encode_text_prompt(prompt)

    # VAE encode
    logger.info("  → VAE encoding images...")
    gt_latents = vae.encode(gt_tensor.to(vae.dtype)).latent_dist.sample() * scale_factor
    bald_latents = vae.encode(bald_tensor.to(vae.dtype)).latent_dist.sample() * scale_factor
    gt_latents = gt_latents.float()
    bald_latents = bald_latents.float()

    # Downsample mask
    mask_down = F.interpolate(mask_tensor, size=gt_latents.shape[-2:], mode='nearest')

    # Identity + Style embeddings
    id_embed = torch.from_numpy(id_embedding).unsqueeze(0).float().to(DEVICE)
    style_embed = torch.zeros(1, 1024).to(DEVICE)  # TODO: CLIP Vision encoder khi có

    # Inject conditioning
    injected_conds = injector.inject_conditioning(style_embed, id_embed)
    encoder_hidden_states = torch.cat([prompt_embeds, injected_conds], dim=1)

    # Time IDs (SDXL)
    time_ids = torch.tensor([
        1024, 1024, 0, 0, 1024, 1024
    ], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    added_cond_kwargs = {
        "text_embeds": pooled_embeds,
        "time_ids": time_ids
    }

    # =============================================
    # DENOISING LOOP
    # =============================================
    logger.info(f"  → Bắt đầu Denoising ({num_steps} steps, guidance_scale={guidance_scale})...")

    # Khởi tạo latents từ noise
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    # Bắt đầu từ random noise
    latents = torch.randn_like(gt_latents)

    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        # Classifier-Free Guidance: uncond + cond
        # Simplified: chỉ dùng conditional (guidance_scale=1)
        # Nếu muốn CFG đầy đủ cần encode empty prompt → gộp batch uncond+cond

        with torch.amp.autocast('cuda', dtype=torch.float16):
            noise_pred = unet(
                noisy_latents=latents,
                bald_latents=bald_latents,
                mask=mask_down,
                timestep=t.unsqueeze(0),
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs
            )

        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Mask blending: giữ nguyên vùng không phải tóc
        # Trộn latents sinh ra (vùng tóc) với bald_latents (vùng mặt)
        latents = latents * mask_down + bald_latents * (1 - mask_down)

    # VAE Decode
    logger.info("  → VAE decoding...")
    latents_decode = (latents / scale_factor).to(vae.dtype)
    decoded = vae.decode(latents_decode).sample
    decoded = decoded.float()

    # Denormalize: [-1, 1] → [0, 1] → [0, 255]
    output_img = ((decoded.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255)
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)

    # Mask blending cuối cùng: giữ pixel gốc ở vùng không phải tóc
    target_np = np.array(target_pil.resize(target_size))
    mask_3ch = np.stack([mask_float] * 3, axis=-1)
    mask_3ch = cv2.resize(mask_3ch, target_size)

    # Feathering (blur mask) để transition mượt
    mask_blur = cv2.GaussianBlur(mask_3ch, (21, 21), 10)
    final = (output_img * mask_blur + target_np * (1 - mask_blur)).astype(np.uint8)

    return Image.fromarray(final)


def pick_random_images(target_path=None, ref_path=None):
    """
    Chọn ngẫu nhiên ảnh từ dataset.
    - Target (mặt người): lấy từ FFHQ
    - Reference (kiểu tóc): lấy từ K-Hairstyle
    """
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # --- Target: FFHQ ---
    if target_path is None:
        if not FFHQ_DIR.exists():
            logger.error(f"FFHQ dataset không tìm thấy: {FFHQ_DIR}")
            sys.exit(1)
        
        # Chọn random subfolder rồi random file
        subfolders = [d for d in FFHQ_DIR.iterdir() if d.is_dir()]
        if not subfolders:
            logger.error("FFHQ không có subfolder nào!")
            sys.exit(1)
        
        chosen_folder = random.choice(subfolders)
        images = [f for f in chosen_folder.iterdir() if f.suffix.lower() in IMG_EXTS]
        if not images:
            logger.error(f"Thư mục FFHQ/{chosen_folder.name} trống!")
            sys.exit(1)
        
        target_path = random.choice(images)
        logger.info(f"  🎲 Random Target (FFHQ): {target_path.name}")

    # --- Reference: K-Hairstyle ---
    if ref_path is None:
        if not KHAIRSTYLE_DIR.exists():
            # Fallback: dùng ảnh khác từ FFHQ làm reference
            logger.warning(f"K-Hairstyle không tìm thấy: {KHAIRSTYLE_DIR}")
            logger.warning("Fallback: dùng ảnh FFHQ khác làm reference")
            
            subfolders = [d for d in FFHQ_DIR.iterdir() if d.is_dir()]
            chosen_folder = random.choice(subfolders)
            images = [f for f in chosen_folder.iterdir() 
                      if f.suffix.lower() in IMG_EXTS and f != target_path]
            ref_path = random.choice(images) if images else target_path
        else:
            # Quét tất cả ảnh trong K-Hairstyle (có thể có subfolder)
            all_images = []
            for ext in IMG_EXTS:
                all_images.extend(KHAIRSTYLE_DIR.rglob(f"*{ext}"))
            
            if not all_images:
                logger.error("K-Hairstyle images/ trống!")
                sys.exit(1)
            
            ref_path = random.choice(all_images)
        
        logger.info(f"  🎲 Random Reference (K-Hairstyle): {ref_path.name}")

    return Path(target_path), Path(ref_path)


def main():
    parser = argparse.ArgumentParser(
        description="Test CLI — Kiểm tra model HairInpainting đã train",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Random từ FFHQ + K-Hairstyle:
  python backend/training/test_cli.py --random

  # Chỉ định ảnh cụ thể:
  python backend/training/test_cli.py --target face.jpg --reference hair.jpg

  # Kết hợp random + chỉ định:
  python backend/training/test_cli.py --random --reference hair.jpg

  # Tuỳ chỉnh prompt và steps:
  python backend/training/test_cli.py --random --prompt "curly hair" --steps 50
        """
    )
    parser.add_argument("--random", action="store_true",
                        help="Chọn ngẫu nhiên target (FFHQ) + reference (K-Hairstyle)")
    parser.add_argument("--target", default=None, help="Ảnh mặt người (target) — bỏ qua nếu dùng --random")
    parser.add_argument("--reference", default=None, help="Ảnh kiểu tóc mẫu (reference) — bỏ qua nếu dùng --random")
    parser.add_argument("--checkpoint", default=None, help="Đường dẫn file .safetensors (mặc định: tự tìm best)")
    parser.add_argument("--output", default=None, help="Đường dẫn ảnh kết quả (mặc định: results/output_<timestamp>.png)")
    parser.add_argument("--prompt", default="high quality, realistic hairstyle, detailed hair texture",
                        help="Text prompt mô tả tóc")
    parser.add_argument("--steps", type=int, default=30, help="Số bước denoising (mặc định: 30)")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (mặc định: 7.5)")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  TEST CLI — Hair Inpainting Inference")
    logger.info("=" * 60)

    # 1. Resolve input images (random hoặc chỉ định)
    if args.random:
        target_path, ref_path = pick_random_images(
            target_path=args.target,
            ref_path=args.reference
        )
    else:
        if not args.target or not args.reference:
            logger.error("Phải chỉ định --target và --reference, hoặc dùng --random")
            sys.exit(1)
        target_path = Path(args.target)
        ref_path = Path(args.reference)

    if not target_path.exists():
        logger.error(f"Không tìm thấy ảnh target: {target_path}")
        sys.exit(1)
    if not ref_path.exists():
        logger.error(f"Không tìm thấy ảnh reference: {ref_path}")
        sys.exit(1)

    # 2. Tìm checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = find_best_checkpoint()
        if ckpt_path is None:
            logger.error("Không tìm thấy checkpoint nào! Hãy train model trước.")
            sys.exit(1)
    elif not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint không tồn tại: {ckpt_path}")
        sys.exit(1)

    logger.info(f"  📸 Target: {target_path}")
    logger.info(f"  💇 Reference: {ref_path}")
    logger.info(f"  🏋️ Checkpoint: {Path(ckpt_path).name}")
    logger.info(f"  📝 Prompt: {args.prompt}")
    logger.info(f"  🔄 Steps: {args.steps}")

    # 3. Load images
    target_pil = Image.open(str(target_path)).convert("RGB")
    ref_pil = Image.open(str(ref_path)).convert("RGB")

    # 4. Preprocessing
    logger.info("\n--- PREPROCESSING ---")
    mask_pil = get_hair_mask(target_pil)
    bald_pil = create_bald_image(target_pil, mask_pil)
    id_embedding = get_identity_embedding(target_pil)

    # 5. Load model
    logger.info("\n--- LOADING MODEL ---")
    model = load_model(ckpt_path)

    # 6. Inference
    logger.info("\n--- INFERENCE ---")
    result = run_inference(
        model, target_pil, mask_pil, bald_pil, id_embedding,
        prompt=args.prompt, num_steps=args.steps, guidance_scale=args.guidance
    )

    # 7. Save output
    if args.output:
        output_path = Path(args.output)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_DIR / "backend" / "training" / "results"
        ensureDir(str(output_dir))
        output_path = output_dir / f"output_{timestamp}.png"

    ensureDir(str(output_path.parent))
    result.save(str(output_path))

    # 8. Lưu thêm ảnh debug (mask, bald, so sánh)
    debug_dir = output_path.parent / "debug"
    ensureDir(str(debug_dir))
    stem = output_path.stem

    mask_pil.save(str(debug_dir / f"{stem}_mask.png"))
    bald_pil.save(str(debug_dir / f"{stem}_bald.png"))
    target_pil.resize((1024, 1024)).save(str(debug_dir / f"{stem}_target.png"))
    ref_pil.resize((1024, 1024)).save(str(debug_dir / f"{stem}_reference.png"))

    # Tạo ảnh so sánh side-by-side: Target | Reference | Bald | Result
    comparison = Image.new("RGB", (1024 * 4, 1024))
    comparison.paste(target_pil.resize((1024, 1024)), (0, 0))
    comparison.paste(ref_pil.resize((1024, 1024)), (1024, 0))
    comparison.paste(bald_pil.resize((1024, 1024)), (2048, 0))
    comparison.paste(result.resize((1024, 1024)), (3072, 0))
    comparison.save(str(debug_dir / f"{stem}_comparison.png"))

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  ✅ KẾT QUẢ ĐÃ LƯU!")
    logger.info(f"  📁 Output: {output_path}")
    logger.info(f"  📁 Debug: {debug_dir}/")
    logger.info(f"     - {stem}_comparison.png (Target | Reference | Bald | Result)")
    logger.info(f"     - {stem}_mask.png")
    logger.info(f"     - {stem}_bald.png")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
