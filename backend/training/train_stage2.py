import os

# Giảm memory fragmentation trên T4/Colab — PyTorch sẽ dùng expandable segments
# thay vì allocate block cố định, tránh tình trạng "reserved but unallocated"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import json
import time
import random
import shutil
import signal
import atexit

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (tương thích server/Colab)
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from safetensors.torch import save_file

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from backend.app.services.training_utils import setupLogger, getDevice, ensureDir
from backend.training.models.stage2_unet import HairInpaintingUNet, CrossAttentionInjector
from backend.training.models.losses import MaskAwareLoss, IdentityCosineLoss, TextureConsistencyLoss, FaceFeatureExtractor

logger = setupLogger("TrainStage2_Inpainting")
DEVICE = getDevice()

# Auto-detect Google Colab environment
IS_COLAB = os.path.exists("/content") and "COLAB_GPU" in os.environ

# CHỈ save cuối cùng khi training hoàn tất (lora_latest + injector_latest)
# Không save mid-chunk, end-of-chunk, hay end-of-epoch → tránh Colab SIGINT

# HuggingFace Hub — lưu checkpoint ngoài Drive (reliable, không phụ thuộc FUSE)
# Đọc từ environment variable (.env hoặc Colab secret)
HF_TOKEN = os.environ.get("HUGFACE_TOKEN", "")   # token write permission
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")    # vd: halogenbr/tryhairstyle
HF_REPO_TYPE = "dataset"                           # loại repo trên HF Hub
HF_SUBFOLDER = "checkpoints"                       # subfolder trong repo

# Đường dẫn SDXL local model
LOCAL_SDXL_PATH = str(PROJECT_DIR / "backend" / "models" / "stable-diffusion" / "sd_xl_inpainting")

# ==============================================================================
# TEXT PROMPT ENCODER — Pre-encode text prompts thành SDXL embeddings
# ==============================================================================

class SDXLTextEncoder:
    """
    Load CLIP Text Encoders (text_encoder + text_encoder_2) của SDXL để encode prompts.
    Sau khi encode xong, giải phóng encoders ra khỏi VRAM để nhường chỗ cho UNet.
    """
    def __init__(self, sdxl_path=LOCAL_SDXL_PATH, device=DEVICE):
        from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
        
        logger.info("Đang load SDXL Text Encoders (tạm thời, chỉ để encode prompts)...")
        
        self.tokenizer = CLIPTokenizer.from_pretrained(sdxl_path, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(sdxl_path, subfolder="tokenizer_2")
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            sdxl_path, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(device).eval()
        
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            sdxl_path, subfolder="text_encoder_2", torch_dtype=torch.float16
        ).to(device).eval()
        
        self.device = device
        logger.info("Text Encoders loaded thành công.")
    
    @torch.no_grad()
    def encode_prompt(self, prompt: str):
        """
        Encode 1 text prompt thành embeddings dùng cho SDXL UNet.
        Returns:
            prompt_embeds: (1, 77, 2048) — concat hidden states từ cả 2 encoders
            pooled_prompt_embeds: (1, 1280) — pooled output từ text_encoder_2
        """
        # Tokenize cho encoder 1 (CLIP ViT-L/14 — 768d hidden)
        tokens_1 = self.tokenizer(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Tokenize cho encoder 2 (CLIP ViT-bigG — 1280d hidden)
        tokens_2 = self.tokenizer_2(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Encode
        enc_out_1 = self.text_encoder(tokens_1, output_hidden_states=True)
        enc_out_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)
        
        # SDXL sử dụng hidden state thứ -2 (penultimate layer)
        hidden_1 = enc_out_1.hidden_states[-2]  # (1, 77, 768)
        hidden_2 = enc_out_2.hidden_states[-2]  # (1, 77, 1280)
        
        # Nối lại thành prompt_embeds (1, 77, 2048)  
        prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1)
        
        # Pooled output từ text_encoder_2 (1, 1280)
        pooled_prompt_embeds = enc_out_2.text_embeds
        
        return prompt_embeds.float().cpu(), pooled_prompt_embeds.float().cpu()
    
    @torch.no_grad()
    def encode_prompts_batch(self, prompts: list, batch_size: int = 16):
        """
        Batch encode nhiều prompts cùng lúc — nhanh hơn 5-10x so với encode từng cái.
        Returns:
            dict[str, dict]: mapping prompt_text → {"prompt_embeds": (77, 2048), "pooled_prompt_embeds": (1280,)}
        """
        results = {}
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            
            # Batch tokenize cho encoder 1
            tokens_1 = self.tokenizer(
                batch, padding="max_length", max_length=77,
                truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            
            # Batch tokenize cho encoder 2
            tokens_2 = self.tokenizer_2(
                batch, padding="max_length", max_length=77,
                truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            
            # Batch encode
            enc_out_1 = self.text_encoder(tokens_1, output_hidden_states=True)
            enc_out_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)
            
            hidden_1 = enc_out_1.hidden_states[-2]  # (B, 77, 768)
            hidden_2 = enc_out_2.hidden_states[-2]  # (B, 77, 1280)
            
            prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1)  # (B, 77, 2048)
            pooled_embeds = enc_out_2.text_embeds  # (B, 1280)
            
            # Tách kết quả cho từng prompt
            for j, prompt_text in enumerate(batch):
                results[prompt_text] = {
                    "prompt_embeds": prompt_embeds[j].float().cpu(),       # (77, 2048)
                    "pooled_prompt_embeds": pooled_embeds[j].float().cpu() # (1280,)
                }
        
        return results
    
    def unload(self):
        """Giải phóng VRAM sau khi encode xong tất cả prompts."""
        del self.text_encoder, self.text_encoder_2
        del self.tokenizer, self.tokenizer_2
        torch.cuda.empty_cache()
        logger.info("Text Encoders đã được giải phóng khỏi VRAM.")

# ==============================================================================
# DATASET
# ==============================================================================

class HairInpaintingDataset(Dataset):
    def __init__(self, data_dir: Path, text_encoder: SDXLTextEncoder = None, 
                 texture_encoder=None, target_size=(512, 512), max_samples: int = 0):
        """
        Args:
            data_dir: Thư mục processed data
            text_encoder: SDXLTextEncoder để pre-encode prompts (None = load từ cache)
            texture_encoder: HairTextureEncoder đã train (None = dùng zeros fallback)
            target_size: Kích thước ảnh đầu vào (512x512 cho tiết kiệm VRAM)
            max_samples: Số lượng samples tối đa (0 = dùng tất cả)
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.metadata = []
        self.prompt_embeds_cache = {}
        
        meta_path = data_dir / "metadata.jsonl"
        if meta_path.exists():
            with open(str(meta_path), "r", encoding="utf-8") as f:
                for line in f:
                    self.metadata.append(json.loads(line.strip()))
        
        # ================================================================
        # SKIP VALIDATION — tin tưởng metadata.jsonl (đã validate ở Stage 1)
        # Nếu file thiếu → __getitem__ sẽ catch + retry sample khác
        # Lý do: os.path.exists() trên Google Drive FUSE mount cực chậm
        #         (~14 phút / 5000 samples). Bỏ bước này tiết kiệm rất nhiều.
        # ================================================================
        logger.info(f"📋 Loaded {len(self.metadata)} samples từ metadata (skip file validation)")
        
        # Giới hạn dataset size nếu cần
        if max_samples > 0 and len(self.metadata) > max_samples:
            before_limit = len(self.metadata)
            rng = random.Random(42)  # Local RNG — không ảnh hưởng global random state
            self.metadata = rng.sample(self.metadata, max_samples)
            logger.info(f"📉 Dataset giảm xuống {max_samples} samples (từ {before_limit} hợp lệ)")
        
        # Pre-encode text prompts nếu Text Encoder được cung cấp
        if text_encoder is not None and len(self.metadata) > 0:
            # Cache thư mục cho prompt embeddings
            cache_dir = data_dir / "prompt_embeddings"
            ensureDir(str(cache_dir))
            
            # ============================================================
            # STEP 1: Tìm samples chưa có cache, nhóm theo unique prompt
            # ============================================================
            uncached_by_prompt = {}  # {full_prompt: [img_id, ...]}
            cached_count = 0
            
            for item in self.metadata:
                img_id = item["id"]
                img_id_dashed = img_id.replace("_", "-")
                cache_file = cache_dir / f"{img_id}.pt"
                cache_file_dashed = cache_dir / f"{img_id_dashed}.pt"
                
                if cache_file.exists() or cache_file_dashed.exists():
                    cached_count += 1
                else:
                    text_prompt = item.get("text_prompt", "hairstyle")
                    if not text_prompt.strip():
                        text_prompt = "hairstyle"
                    full_prompt = f"high quality, realistic {text_prompt}, detailed hair texture"
                    uncached_by_prompt.setdefault(full_prompt, []).append(img_id)
            
            # ============================================================
            # STEP 2: Batch encode các unique prompts chưa cache
            # ============================================================
            if uncached_by_prompt:
                unique_prompts = list(uncached_by_prompt.keys())
                total_uncached = sum(len(ids) for ids in uncached_by_prompt.values())
                logger.info(
                    f"Pre-encoding {total_uncached} samples "
                    f"({len(unique_prompts)} unique prompts, {cached_count} đã cache)"
                )
                
                # Batch encode tất cả unique prompts cùng lúc
                encoded_map = text_encoder.encode_prompts_batch(unique_prompts, batch_size=16)
                
                # Lưu cache cho tất cả samples
                for full_prompt, img_ids in tqdm(
                    uncached_by_prompt.items(), desc="Saving Prompt Cache"
                ):
                    cached_data = encoded_map[full_prompt]
                    for img_id in img_ids:
                        cache_file = cache_dir / f"{img_id}.pt"
                        torch.save(cached_data, str(cache_file))
                
                logger.info(f"✅ Pre-encoding hoàn tất! ({len(unique_prompts)} unique → {total_uncached} files)")
            else:
                logger.info(f"✅ Tất cả {cached_count} prompt embeddings đã cache — skip encoding")
        else:
            # Lazy loading: prompt embeddings load on-demand trong _load_sample()
            # Không bulk-load vào RAM → tiết kiệm ~3GB/chunk cho Colab
            pass
                    
        # Pre-extract style embeddings từ Texture Encoder (Stage 1)
        self.style_embeds_cache = {}
        style_cache_dir = data_dir / "style_embeddings_cache"
        
        if texture_encoder is not None:
            try:
                ensureDir(str(style_cache_dir))
            except OSError as e:
                logger.error(f"❌ Không thể tạo style cache dir: {e}")
                raise RuntimeError(
                    f"Không thể tạo thư mục cache '{style_cache_dir}'. "
                    f"Kiểm tra quyền ghi trên Drive hoặc symlink."
                ) from e
            
        if texture_encoder is not None:
            # ============================================================
            # PRE-CHECK: Đếm xem bao nhiêu style cache đã tồn tại
            # Nếu tất cả đã cache → skip hoàn toàn (không loop 5000 items)
            # ============================================================
            uncached_items = []
            cached_style_count = 0
            for item in self.metadata:
                img_id = item["id"]
                cache_file = style_cache_dir / f"{img_id}.npy"
                if cache_file.exists():
                    cached_style_count += 1
                else:
                    uncached_items.append(item)

            if not uncached_items:
                # Tất cả đã cache → lazy load on-demand, không load vào RAM trước
                logger.info(f"✅ Tất cả {cached_style_count} style embeddings đã cache — skip extraction")
            else:
                logger.info(
                    f"Pre-extracting style embeddings: {len(uncached_items)} chưa cache "
                    f"({cached_style_count} đã cache / {len(self.metadata)} tổng)"
                )

                style_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                STYLE_BATCH = 32  # Số samples xử lý mỗi batch → ~20-30x nhanh hơn 1-by-1
                with torch.no_grad():
                    for batch_start in tqdm(range(0, len(uncached_items), STYLE_BATCH), desc="Extracting Style"):
                        batch_items = uncached_items[batch_start:batch_start + STYLE_BATCH]

                        batch_tensors = []
                        batch_ids = []
                        zero_ids = []  # items không có file ảnh → lưu zeros

                        for item in batch_items:
                            img_id = item["id"]
                            style_img_path = data_dir / item["style"]
                            if style_img_path.exists():
                                style_img = cv2.cvtColor(cv2.imread(str(style_img_path)), cv2.COLOR_BGR2RGB)
                                style_img = cv2.resize(style_img, (128, 128))
                                batch_tensors.append(style_transform(style_img))
                                batch_ids.append(img_id)
                            else:
                                zero_ids.append(img_id)

                        # Lưu zeros cho ảnh bị thiếu
                        for img_id in zero_ids:
                            np.save(str(style_cache_dir / f"{img_id}.npy"), np.zeros(2048, dtype=np.float32))

                        # Batch forward pass
                        if batch_tensors:
                            batch_tensor = torch.stack(batch_tensors).to(DEVICE)  # (B, 3, 128, 128)
                            embeds, _, _ = texture_encoder(batch_tensor)           # (B, 2048)
                            embeds_np = embeds.cpu().numpy()                       # (B, 2048)

                            for img_id, embed_np in zip(batch_ids, embeds_np):
                                np.save(str(style_cache_dir / f"{img_id}.npy"), embed_np)

                logger.info(f"✅ Style embeddings extraction hoàn tất! ({len(uncached_items)} mới cache)")
        else:
            # Lazy loading: style embeddings load on-demand trong _load_sample()
            pass
        
        # Transform tensor chuẩn SDXL (-1 to 1)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        max_retries = 10  # Thử tối đa 10 sample khác nhau
        for attempt in range(max_retries + 1):
            try:
                target_idx = idx if attempt == 0 else random.randint(0, len(self.metadata) - 1)
                return self._load_sample(target_idx)
            except Exception as e:
                if attempt < 3 or attempt == max_retries:
                    # Log 3 lần đầu + lần cuối, tránh spam log
                    logger.warning(f"⚠️ Skip sample idx={target_idx}: {e}")
                if attempt == max_retries:
                    raise RuntimeError(
                        f"❌ Đã thử {max_retries+1} samples ngẫu nhiên, tất cả đều lỗi. "
                        f"Dataset có thể bị hỏng hoặc thiếu file."
                    )
    
    def _load_sample(self, idx):
        item = self.metadata[idx]
        img_id = item["id"]
        
        # Load Ground Truth image từ processed/ground_truth_images/
        gt_dir = self.data_dir / "ground_truth_images"
        img_candidates = list(gt_dir.glob(f"{img_id}.*"))
        if not img_candidates:
            img_candidates = list(gt_dir.glob(f"{img_id.replace('_', '-')}.*"))
        
        # Fallback: tìm trong thư mục dataset gốc (chỉ khi train trên máy local)
        if not img_candidates:
            for search_dir in [PROJECT_DIR / "dataset", 
                               PROJECT_DIR / "backend" / "data" / "dataset" / "khairstyle" / "training" / "images"]:
                if search_dir.exists():
                    img_candidates = list(search_dir.rglob(f"{img_id}.*"))
                    if img_candidates:
                        break
        
        if not img_candidates:
            raise FileNotFoundError(f"Không tìm thấy Ground Truth cho ID: {img_id}")
            
        gt_img = cv2.cvtColor(cv2.imread(str(img_candidates[0])), cv2.COLOR_BGR2RGB)
        gt_img = cv2.resize(gt_img, self.target_size)
        
        # Load Mask (1 Channel) từ kênh Alpha của ảnh hair_only
        hair_only_rgba = cv2.imread(str(self.data_dir / item["hair_only"]), cv2.IMREAD_UNCHANGED)
        hair_only_rgba = cv2.resize(hair_only_rgba, self.target_size)
        if hair_only_rgba.shape[2] == 4:
            mask = (hair_only_rgba[:, :, 3] / 255.0).astype(np.float32)
        else:
            mask = np.zeros((self.target_size[0], self.target_size[1]), dtype=np.float32)
        true_mask = mask[..., np.newaxis]
        
        # Identity (từ AdaFace 512d)
        id_embed = np.load(str(self.data_dir / item["identity"]))
        
        # Text Prompt Embeddings — lazy load từ disk cache
        if img_id in self.prompt_embeds_cache:
            cached = self.prompt_embeds_cache[img_id]
            text_embeds = cached["prompt_embeds"]
            pooled_text_embeds = cached["pooled_prompt_embeds"]
        else:
            cache_dir = self.data_dir / "prompt_embeddings"
            img_id_dashed = img_id.replace("_", "-")
            cache_file = cache_dir / f"{img_id}.pt"
            cache_file_dashed = cache_dir / f"{img_id_dashed}.pt"
            if cache_file.exists():
                cached = torch.load(str(cache_file), map_location="cpu", weights_only=True)
                text_embeds = cached["prompt_embeds"]
                pooled_text_embeds = cached["pooled_prompt_embeds"]
            elif cache_file_dashed.exists():
                cached = torch.load(str(cache_file_dashed), map_location="cpu", weights_only=True)
                text_embeds = cached["prompt_embeds"]
                pooled_text_embeds = cached["pooled_prompt_embeds"]
            else:
                text_embeds = torch.zeros(77, 2048)
                pooled_text_embeds = torch.zeros(1280)
        
        # Style Embedding — lazy load từ disk cache
        if img_id in self.style_embeds_cache:
            style_embed = self.style_embeds_cache[img_id]
        else:
            style_cache_file = self.data_dir / "style_embeddings_cache" / f"{img_id}.npy"
            if style_cache_file.exists():
                style_embed = np.load(str(style_cache_file))
            else:
                style_embed = np.zeros(2048, dtype=np.float32)
        
        # Trả về Tensors (bald_image không load vì train_step tự tạo masked_images)
        return {
            "image": self.img_transform(gt_img),
            "mask": torch.from_numpy(true_mask).permute(2, 0, 1),
            "style_embed": torch.from_numpy(style_embed).float(),
            "identity_embed": torch.from_numpy(id_embed).squeeze(0).float(),
            "text_embeds": text_embeds,
            "pooled_text_embeds": pooled_text_embeds,
            "time_ids": torch.tensor([
                1024, 1024,   # SDXL native resolution — nhất quán với inference
                0, 0,
                1024, 1024
            ], dtype=torch.float32)
        }

# ==============================================================================
# TRAINER
# ==============================================================================

class Stage2Trainer:
    """
    Kịch bản huấn luyện Stage 2: Mask-Conditioned Hair Inpainting
    Kết hợp 9-channel UNet, Custom Identity Loss, và Mask-aware Gradient Locking.
    Tối ưu cho GPU 12GB (RTX 3060): AMP + xformers + 8-bit optimizer + gradient accumulation.
    """
    def __init__(self):
        from diffusers import AutoencoderKL, DDPMScheduler
        from peft import LoraConfig, get_peft_model
        
        logger.info("Khởi tạo Stage 2 Trainer: SDXL LoRA Inpainting ")
        
        # 1. Load VAE (frozen, FP16) → CPU offload để tiết kiệm ~0.3GB VRAM
        logger.info("  → Loading VAE (fp16, frozen, CPU offload)...")
        self.vae = AutoencoderKL.from_pretrained(
            LOCAL_SDXL_PATH, subfolder="vae", torch_dtype=torch.float16
        ).eval()
        self.vae.requires_grad_(False)
        self.vae_scale_factor = self.vae.config.scaling_factor  # 0.13025 cho SDXL
        self.vae.enable_slicing()
        self.vae.enable_tiling()
        self.vae.to('cpu')  # CPU offload — move lên GPU khi cần encode/decode
        logger.info("  → VAE loaded → CPU (sẽ move lên GPU on-demand)")
        
        # 2. Load Noise Scheduler
        logger.info("  → Loading DDPMScheduler...")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            LOCAL_SDXL_PATH, subfolder="scheduler"
        )
        
        # 3. Load UNet 9-channel (FP16 frozen) + LoRA adapters
        # KHÔNG gọi .float() — giữ FP16 cho frozen weights (~5GB thay vì ~10GB)
        logger.info("  → Loading UNet 9-channel (fp16 frozen + LoRA r=16)...")
        self.unet = HairInpaintingUNet().to(DEVICE)  # FP16 trên GPU
        
        # 3a. Apply LoRA — chỉ train tiny adapters thay vì toàn bộ 2.6B params
        lora_config = LoraConfig(
            r=16,                    # Rank 16 — cân bằng quality / VRAM
            lora_alpha=16,           # Alpha = rank → scaling = 1.0
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.05,
            bias="none",
        )
        self.unet.unet = get_peft_model(self.unet.unet, lora_config)
        
        # 3b. Unfreeze conv_in (9-channel layer mới, cần train)
        self.unet.unet.base_model.model.conv_in.requires_grad_(True)
        self.unet.unet.base_model.model.conv_in.float()  # FP32 cho training precision
        
        # Log trainable params
        trainable_n = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total_n = sum(p.numel() for p in self.unet.parameters())
        logger.info(f"  → LoRA: {trainable_n:,} trainable / {total_n:,} total ({100*trainable_n/total_n:.2f}%)")
        
        # 3c. CrossAttention Injector (FP32, train toàn bộ — rất nhỏ ~50MB)
        self.injector = CrossAttentionInjector(
            self.unet.unet.base_model.model, style_dim=2048
        ).to(DEVICE).float()
        
        # 4. Khởi tạo Loss Functions
        self.mask_aware_loss = MaskAwareLoss(loss_type='l2').to(DEVICE)
        self.identity_loss = IdentityCosineLoss().to(DEVICE)
        self.texture_loss = None  # LAZY LOAD mỗi 50 steps
        logger.info("  → TextureConsistencyLoss: LAZY mode (load mỗi 50 steps)")
        self.face_extractor = None  # LAZY LOAD
        logger.info("  → Face Feature Extractor: LAZY mode (load mỗi 50 steps)")
        
        # 5. Optimizer — CHỈ train LoRA + conv_in + Injector
        self._trainable_params = (
            [p for p in self.unet.parameters() if p.requires_grad]
            + list(self.injector.parameters())
        )
        n_train = sum(p.numel() for p in self._trainable_params)
        logger.info(f"  → Total trainable (LoRA+conv_in+Injector): {n_train:,} params")
        
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                self._trainable_params, lr=1e-5, weight_decay=1e-2
            )
            logger.info("  → 8-bit AdamW cho LoRA + Injector")
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                self._trainable_params, lr=1e-5, weight_decay=1e-2
            )
            logger.warning("  → bitsandbytes chưa cài, dùng AdamW 32-bit")
        
        # 6. AMP Mixed Precision Scaler
        self.scaler = torch.amp.GradScaler('cuda')
        
        # 7. Learning Rate Scheduler — Warmup + Cosine Decay
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        self._warmup_steps = 100
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=self._warmup_steps)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=2000, eta_min=1e-7)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self._warmup_steps]
        )
        
        # 8. Tạo thư mục checkpoints
        #    Colab: save vào /tmp/ (SSD local, instant) — KHÔNG dùng Drive FUSE (chậm, bị ^C kill)
        #    HF Hub upload xử lý persistence
        if IS_COLAB:
            self.checkpoints_dir = Path("/tmp/training_checkpoints")
        else:
            self.checkpoints_dir = PROJECT_DIR / "backend" / "training" / "checkpoints"
        ensureDir(str(self.checkpoints_dir))
        
        # 11. Setup SIGINT handler cho graceful shutdown
        self._interrupted = False
        self._last_save_state = None  # Cho atexit handler
        self._setup_signal_handler()
        
        # 12. Pre-download VGG16 + InceptionResnet — tránh download 635MB giữa lúc train
        #     Chỉ cần download 1 lần, PyTorch cache vào ~/.cache/torch/hub/
        try:
            from torchvision import models as tv_models
            logger.info("  → Pre-downloading VGG16 weights (nếu chưa cache)...")
            _vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
            del _vgg
            logger.info("  → VGG16 weights cached ✅")
        except Exception as e:
            logger.warning(f"  ⚠️ VGG16 pre-download failed: {e} (sẽ download khi cần)")
        try:
            from facenet_pytorch import InceptionResnetV1
            logger.info("  → Pre-downloading InceptionResnetV1 weights (nếu chưa cache)...")
            _face = InceptionResnetV1(pretrained='vggface2')
            del _face
            logger.info("  → InceptionResnetV1 weights cached ✅")
        except Exception as e:
            logger.warning(f"  ⚠️ InceptionResnetV1 pre-download failed: {e} (sẽ download khi cần)")
        
        logger.info("Trainer khởi tạo thành công (LoRA mode). VRAM sử dụng:")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"  Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
        
    def _setup_signal_handler(self):
        """Register SIGINT + SIGTERM + atexit handler cho graceful shutdown.
        Colab gửi SIGINT/SIGTERM khi runtime idle quá lâu hoặc user ngắt kết nối.
        Handler bắt signal, set flag để training loop save checkpoint trước khi thoát.
        atexit đảm bảo save ngay cả khi signal bị miss.
        """
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        
        def _handler(signum, frame):
            sig_name = 'SIGINT' if signum == signal.SIGINT.value else 'SIGTERM'
            if self._interrupted:
                # Double signal → force quit
                logger.warning(f"\n⚠️ Double {sig_name} — force quit!")
                signal.signal(signal.SIGINT, self._original_sigint)
                raise KeyboardInterrupt
            self._interrupted = True
            logger.warning(f"\n⚠️ {sig_name} received — sẽ save checkpoint và thoát sau step hiện tại...")
        
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        
        logger.info("  ⚒️ Signal handlers registered (SIGINT + SIGTERM)")
    

    
    @torch.no_grad()
    def _encode_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ảnh RGB tensor sang latent space qua VAE (CPU↔GPU offload)."""
        self.vae.to(DEVICE)
        latents = self.vae.encode(images.to(self.vae.dtype)).latent_dist.sample()
        latents = latents * self.vae_scale_factor
        self.vae.to('cpu')
        torch.cuda.empty_cache()
        return latents.float()
    
    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents về RGB image qua VAE (CPU↔GPU offload, no_grad).
        
        Dùng no_grad để tiết kiệm ~1-2GB VRAM (không lưu VAE activation graph).
        Auxiliary losses (Texture/Identity) chỉ dùng để monitor, không cần gradient.
        """
        self.vae.to(DEVICE)
        latents_input = (latents / self.vae_scale_factor).to(self.vae.dtype)
        decoded = self.vae.decode(latents_input).sample.float()
        self.vae.to('cpu')
        torch.cuda.empty_cache()
        return decoded
    
    @torch.no_grad()
    def _decode_latents_no_grad(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents về RGB (validation, CPU↔GPU offload)."""
        self.vae.to(DEVICE)
        latents_input = (latents / self.vae_scale_factor).to(self.vae.dtype)
        decoded = self.vae.decode(latents_input).sample
        self.vae.to('cpu')
        torch.cuda.empty_cache()
        return decoded.float()
    
    def _get_lora_state_dict(self):
        """Lấy LoRA adapter weights + conv_in weights để save checkpoint."""
        from peft import get_peft_model_state_dict
        lora_weights = dict(get_peft_model_state_dict(self.unet.unet))
        # Thêm conv_in weights (9-channel layer, train riêng)
        for k, v in self.unet.unet.base_model.model.conv_in.state_dict().items():
            lora_weights[f"conv_in.{k}"] = v
        return lora_weights
    
    def _load_lora_weights(self, path):
        """Load LoRA + conv_in weights từ checkpoint."""
        from safetensors.torch import load_file as load_safetensors
        from peft import set_peft_model_state_dict
        state_dict = load_safetensors(str(path))
        conv_in_state = {}
        lora_state = {}
        for k, v in state_dict.items():
            if k.startswith("conv_in."):
                conv_in_state[k[len("conv_in."):]] = v
            else:
                lora_state[k] = v
        set_peft_model_state_dict(self.unet.unet, lora_state)
        if conv_in_state:
            self.unet.unet.base_model.model.conv_in.load_state_dict(conv_in_state)
        logger.info(f"  → LoRA + conv_in loaded từ {Path(path).name}")
        
    def train_step(self, batch, global_step: int, accumulation_steps: int = 8):
        """
        Bước lặp huấn luyện duy nhất.
        batch bao gồm:
          - image: Ảnh thật (Ground Truth)
          - mask: Hair mask (1 kênh)
          - style_embed: Vector đặc trưng kiểu tóc (từ CLIP)
          - identity_embed: Vector khuôn mặt (từ AdaFace)
          - text_embeds: Prompt embeddings (đã pre-encode)
          - pooled_text_embeds: Pooled SDXL embeddings
          - time_ids: SDXL time conditioning
        """
        self.unet.train()
        self.injector.train()
        
        # Gradient Accumulation: chỉ zero_grad mỗi N steps
        if global_step % accumulation_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)
        
        # Giải nén Batch
        gt_images = batch['image'].to(DEVICE)
        masks = batch['mask'].to(DEVICE)
        
        # ==========================================
        # VAE ENCODE (thực) — sequential + aggressive cleanup
        # ==========================================
        latents = self._encode_to_latents(gt_images)
        
        # Tạo masked_image = ảnh gốc × (1 - mask) → vùng tóc bị zero, phần còn lại giữ nguyên
        # Đây là convention chuẩn của SDXL Inpainting (channel 5-8 = masked_image_latents)
        masks_pixel = F.interpolate(masks, size=gt_images.shape[-2:], mode='nearest')
        masked_images = gt_images * (1.0 - masks_pixel)
        masked_latents = self._encode_to_latents(masked_images)
        del masked_images  # Free ngay sau khi encode xong
        masks_latent = F.interpolate(masks, size=latents.shape[-2:], mode='nearest')
        # Free pixel-space tensors — chỉ cần latent-space cho UNet forward
        # gt_images/masks/masks_pixel sẽ re-load từ batch nếu cần ở step monitor
        del gt_images, masks, masks_pixel
        torch.cuda.empty_cache()  # Reclaim fragmented memory trước UNet forward
        
        # ==========================================
        # NOISE SCHEDULING (thực)
        # ==========================================
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=DEVICE
        ).long()
        
        # Thêm noise theo scheduler chuẩn diffusion
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        del latents  # Free — noise target là `noise`, không cần `latents` nữa
        
        # ==========================================
        # CONDITIONING (thực)
        # ==========================================
        style_embeds = batch['style_embed'].to(DEVICE)
        id_embeds = batch['identity_embed'].to(DEVICE)
        text_embeds = batch['text_embeds'].to(DEVICE)
        pooled_text_embeds = batch['pooled_text_embeds'].to(DEVICE)
        time_ids = batch['time_ids'].to(DEVICE)
        
        # Gộp Style + Identity conditioning qua IP-Adapter
        injected_conds = self.injector.inject_conditioning(style_embeds, id_embeds)
        encoder_hidden_states = torch.cat([text_embeds, injected_conds], dim=1)
        
        added_cond_kwargs = {
            "text_embeds": pooled_text_embeds,
            "time_ids": time_ids
        }
        
        # ==========================================
        # FORWARD PASS (AMP Mixed Precision)
        # ==========================================
        with torch.amp.autocast('cuda', dtype=torch.float16):
            noise_pred = self.unet(
                noisy_latents=noisy_latents,
                masked_latents=masked_latents,
                mask=masks_latent,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs
            )
            
            # Free conditioning tensors ngay sau UNet forward — không cần cho backward
            del encoder_hidden_states, added_cond_kwargs
            del style_embeds, id_embeds, injected_conds
            del text_embeds, pooled_text_embeds, time_ids
            del masked_latents  # Không cần sau forward
            
            # ==========================================
            # LOSS COMPUTATION
            # ==========================================
            
            # 1. Mask-Aware Diffusion Loss (Core — mọi step)
            loss_diffusion = self.mask_aware_loss(noise_pred, noise, masks_latent)
            total_loss = loss_diffusion
        
        # Cast sang fp32 TRƯỚC khi backprop
        total_loss = total_loss.float()
        
        # 2. Texture & Identity Loss — MONITOR ONLY (mỗi 50 steps)
        # Dùng no_grad: chỉ log giá trị lên chart, KHÔNG cộng vào total_loss
        # → Tiết kiệm ~2GB VRAM, cho phép batch_size=2 trên T4 15GB
        loss_id_val = 0.0
        loss_tex_val = 0.0
        
        if global_step % 50 == 0 and global_step > 0:
            try:
                with torch.no_grad():
                    # Re-load pixel tensors từ batch (đã free ở trên)
                    gt_images = batch['image'].to(DEVICE)
                    masks = batch['mask'].to(DEVICE)
                    masks_pixel = F.interpolate(masks, size=gt_images.shape[-2:], mode='nearest')
                    
                    # Ước tính denoised output (x0 prediction)
                    alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(DEVICE)
                    alpha_prod_t = alphas_cumprod[timesteps]  # (B,)
                    sqrt_alpha = (alpha_prod_t ** 0.5).view(-1, 1, 1, 1)
                    sqrt_one_minus_alpha = ((1 - alpha_prod_t) ** 0.5).view(-1, 1, 1, 1)
                    pred_original = (noisy_latents - sqrt_one_minus_alpha * noise_pred) / (sqrt_alpha + 1e-8)
                    
                    # Decode no_grad — tiết kiệm VRAM
                    decoded_img = self._decode_latents(pred_original)
                    
                    # Texture Loss — LAZY LOAD VGG16 rồi xóa ngay
                    texture_loss_fn = TextureConsistencyLoss().to(DEVICE)
                    texture_loss_fn.eval()
                    masks_decoded = F.interpolate(masks_pixel, size=decoded_img.shape[-2:], mode='nearest')
                    loss_tex_val = texture_loss_fn(decoded_img, gt_images, masks_decoded).item()
                    del texture_loss_fn
                    
                    # Identity Loss — Lazy load FaceFeatureExtractor (InceptionResnetV1)
                    face_extractor = FaceFeatureExtractor(device=str(DEVICE)).to(DEVICE)
                    face_extractor.eval()
                    
                    gt_images_resized = F.interpolate(gt_images, size=decoded_img.shape[-2:], mode='bilinear', align_corners=False) if gt_images.shape[-2:] != decoded_img.shape[-2:] else gt_images
                    masks_for_face = F.interpolate(masks_pixel[:, :1], size=decoded_img.shape[-2:], mode='nearest') if masks_pixel.shape[-2:] != decoded_img.shape[-2:] else masks_pixel[:, :1]
                    
                    gen_face_embeds = face_extractor(decoded_img, masks_for_face)
                    target_face_embeds = face_extractor(gt_images_resized, masks_for_face)
                    loss_id_val = self.identity_loss(gen_face_embeds.float(), target_face_embeds.float()).item()
                    
                    # Cleanup
                    del face_extractor, gen_face_embeds, target_face_embeds
                    del gt_images_resized, masks_for_face, decoded_img, pred_original
                    del gt_images, masks, masks_pixel, masks_decoded
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("OOM khi tính Texture/Identity monitor, skip step này.")
                    torch.cuda.empty_cache()
                else:
                    raise
        
        # ==========================================
        # BACKPROP (AMP + Gradient Accumulation)
        # ==========================================
        # Chia loss cho accumulation_steps để trung bình gradient
        scaled_loss = total_loss / accumulation_steps
        self.scaler.scale(scaled_loss).backward()
        
        # Chỉ cập nhật weights mỗi accumulation_steps
        if (global_step + 1) % accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self._trainable_params,  # Chỉ LoRA + conv_in + Injector
                max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
        
        return {
            "total_loss": total_loss.item(),
            "diffusion_loss": loss_diffusion.item(),
            "texture_loss": loss_tex_val,
            "identity_loss": loss_id_val,
        }
        
    @torch.no_grad()
    def validate_epoch(self, val_dataloader, global_step: int):
        """
        Đánh giá model trên validation set.
        Tính Validation Diffusion Loss (noise prediction MSE) — KHÔNG backprop.
        Tùy chọn: tính LPIPS trên vài samples đầu tiên.
        
        Returns:
            dict: {
                'val_loss': float — trung bình diffusion loss trên val set,
                'val_lpips': float — trung bình LPIPS (nếu có), -1.0 nếu không
            }
        """
        self.unet.eval()
        self.injector.eval()
        
        val_losses = []
        lpips_scores = []
        
        # Khởi tạo LPIPS evaluator (lazy, chỉ lần đầu)
        lpips_evaluator = None
        try:
            from backend.training.evaluate import HairEvaluator
            lpips_evaluator = HairEvaluator(device=str(DEVICE))
            if lpips_evaluator.loss_fn_vgg is None:
                lpips_evaluator = None
        except Exception:
            pass  # LPIPS không bắt buộc
        
        max_lpips_samples = 4  # Chỉ tính LPIPS trên vài samples để tiết kiệm VRAM
        lpips_count = 0
        
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            gt_images = batch['image'].to(DEVICE)
            masks = batch['mask'].to(DEVICE)
            
            # VAE Encode
            latents = self._encode_to_latents(gt_images)
            masks_pixel = F.interpolate(masks, size=gt_images.shape[-2:], mode='nearest')
            masked_images = gt_images * (1.0 - masks_pixel)
            masked_latents = self._encode_to_latents(masked_images)
            masks_latent = F.interpolate(masks, size=latents.shape[-2:], mode='nearest')
            
            # Noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (bsz,), device=DEVICE
            ).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Conditioning
            style_embeds = batch['style_embed'].to(DEVICE)
            id_embeds = batch['identity_embed'].to(DEVICE)
            text_embeds = batch['text_embeds'].to(DEVICE)
            pooled_text_embeds = batch['pooled_text_embeds'].to(DEVICE)
            time_ids = batch['time_ids'].to(DEVICE)
            
            injected_conds = self.injector.inject_conditioning(style_embeds, id_embeds)
            encoder_hidden_states = torch.cat([text_embeds, injected_conds], dim=1)
            
            added_cond_kwargs = {
                "text_embeds": pooled_text_embeds,
                "time_ids": time_ids
            }
            
            # Forward (no grad, AMP)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                noise_pred = self.unet(
                    noisy_latents=noisy_latents,
                    masked_latents=masked_latents,
                    mask=masks_latent,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs
                )
                
                val_diff_loss = self.mask_aware_loss(noise_pred, noise, masks_latent)
                val_losses.append(val_diff_loss.item())
            
            # LPIPS trên vài samples đầu (tùy chọn, tốn VRAM)
            if lpips_evaluator is not None and lpips_count < max_lpips_samples:
                try:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        # Ước tính denoised output (x0 prediction)
                        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(DEVICE)
                        alpha_prod_t = alphas_cumprod[timesteps]
                        sqrt_alpha = (alpha_prod_t ** 0.5).view(-1, 1, 1, 1)
                        sqrt_one_minus_alpha = ((1 - alpha_prod_t) ** 0.5).view(-1, 1, 1, 1)
                        pred_original = (noisy_latents - sqrt_one_minus_alpha * noise_pred) / (sqrt_alpha + 1e-8)
                        
                        decoded_img = self._decode_latents_no_grad(pred_original)
                    
                    # LPIPS cần input [-1, 1] — cả decoded_img và gt_images đều ở range này
                    lpips_val = lpips_evaluator.evaluate_lpips(
                        decoded_img[:1], gt_images[:1], masks[:1]  # Chỉ lấy 1 sample
                    )
                    if lpips_val >= 0:
                        lpips_scores.append(lpips_val)
                    lpips_count += 1
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                    lpips_evaluator = None  # Tắt LPIPS nếu OOM
        
        # Tổng hợp
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        avg_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else -1.0
        
        # Trả UNet về train mode
        self.unet.train()
        self.injector.train()
        
        return {
            'val_loss': avg_val_loss,
            'val_lpips': avg_lpips
        }

    def _save_safetensors_safe(self, state_dict, path: str):
        """Lưu safetensors an toàn — ghi vào /tmp/ (SSD) rồi upload HF Hub ngay.
        Giống hệt cơ chế code cũ đã chạy thành công trên Colab.
        Không dùng background thread, không ghi Drive.
        """
        import tempfile
        filename = os.path.basename(path)
        
        # === BƯỚC 1: Save vào /tmp/ (SSD local, cực nhanh) ===
        if IS_COLAB:
            save_dir = "/tmp/training_saves"
        else:
            save_dir = os.path.dirname(path)
        os.makedirs(save_dir, exist_ok=True)
        
        fd, temp_path = tempfile.mkstemp(suffix=".safetensors", dir=save_dir)
        os.close(fd)
        try:
            save_file(state_dict, temp_path)
            final_path = os.path.join(save_dir, filename)
            shutil.move(temp_path, final_path)
            size_mb = os.path.getsize(final_path) / (1024 * 1024)
            logger.info(f"  💾 Saved: {filename} ({size_mb:.1f} MB) [local]")
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu {filename}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
        
        # Cập nhật path để trỏ đến file thật (có thể khác path ban đầu trên Colab)
        actual_path = final_path
        
        # === BƯỚC 2: Upload lên HF Hub ngay lập tức (đồng bộ) ===
        if HF_TOKEN and HF_REPO_ID:
            try:
                from huggingface_hub import upload_file, create_repo
                try:
                    create_repo(HF_REPO_ID, token=HF_TOKEN, repo_type=HF_REPO_TYPE, exist_ok=True, private=True)
                except Exception:
                    pass
                upload_file(
                    path_or_fileobj=actual_path,
                    path_in_repo=f"{HF_SUBFOLDER}/{filename}",
                    repo_id=HF_REPO_ID,
                    repo_type=HF_REPO_TYPE,
                    token=HF_TOKEN,
                    commit_message=f"checkpoint: {filename}",
                )
                logger.info(f"  ☁️ HF: {filename} → {HF_REPO_ID}/{HF_SUBFOLDER}/")
            except Exception as hf_err:
                logger.warning(f"  ⚠️ HF upload failed ({filename}): {hf_err}")

    def _plot_training_charts(self, history: dict, epoch: int):
        """
        Tạo biểu đồ Loss Chart tự động sau mỗi epoch.
        Lưu vào checkpoints/ để dễ theo dõi.
        """
        charts_dir = self.checkpoints_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f"Training Stage 2 — After Epoch {epoch}", fontsize=16, fontweight='bold')
        
        steps = range(1, len(history['total_loss']) + 1)
        
        # --- 1. Total Loss (mỗi step) ---
        ax1 = axes[0, 0]
        ax1.plot(steps, history['total_loss'], alpha=0.3, color='#2196F3', linewidth=0.5, label='Per step')
        if len(history['total_loss']) > 100:
            window = 100
            smoothed = np.convolve(history['total_loss'], np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(history['total_loss']) + 1), smoothed, color='#1565C0', linewidth=2, label=f'MA-{window}')
        ax1.set_title('Total Loss', fontsize=13)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # --- 2. Diffusion Loss (mỗi step) ---
        ax2 = axes[0, 1]
        ax2.plot(steps, history['diffusion_loss'], alpha=0.3, color='#4CAF50', linewidth=0.5, label='Per step')
        if len(history['diffusion_loss']) > 100:
            smoothed = np.convolve(history['diffusion_loss'], np.ones(100)/100, mode='valid')
            ax2.plot(range(100, len(history['diffusion_loss']) + 1), smoothed, color='#2E7D32', linewidth=2, label='MA-100')
        ax2.set_title('Diffusion Loss (Noise Prediction)', fontsize=13)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # --- 3. Texture + Identity Loss (monitor mỗi 50 steps, bỏ qua giá trị 0) ---
        ax3 = axes[1, 0]
        texSteps = [i+1 for i, v in enumerate(history['texture_loss']) if v > 0]
        texValues = [v for v in history['texture_loss'] if v > 0]
        idSteps = [i+1 for i, v in enumerate(history.get('identity_loss', [])) if v > 0]
        idValues = [v for v in history.get('identity_loss', []) if v > 0]
        
        has_data = texValues or idValues
        if has_data:
            if texValues:
                ax3.plot(texSteps, texValues, color='#FF9800', linewidth=1.5, marker='.', markersize=3, label='Texture Loss')
            if idValues:
                ax3_twin = ax3.twinx()
                ax3_twin.plot(idSteps, idValues, color='#9C27B0', linewidth=1.5, marker='.', markersize=3, label='Identity Loss')
                ax3_twin.set_ylabel('Identity Loss', color='#9C27B0')
                ax3_twin.legend(loc='upper right')
            ax3.set_title('Texture + Identity Loss (monitor mỗi 50 steps)', fontsize=13)
        else:
            ax3.text(0.5, 0.5, 'Chưa có dữ liệu\n(tính sau step 50)', ha='center', va='center', fontsize=12, transform=ax3.transAxes)
            ax3.set_title('Texture + Identity Loss', fontsize=13)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Texture Loss', color='#FF9800')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # --- 4. Train vs Validation Loss (mỗi epoch) ---
        ax4 = axes[1, 1]
        if history['epoch_avg_loss']:
            epochs = range(1, len(history['epoch_avg_loss']) + 1)
            ax4.plot(epochs, history['epoch_avg_loss'], color='#E91E63', linewidth=2.5, marker='o', markersize=8, label='Train Loss')
            
            # Vẽ Validation Loss nếu có (guard: chỉ plot nếu cùng length)
            val_losses = history.get('val_loss', [])
            val_losses = val_losses[:len(history['epoch_avg_loss'])]  # Guard: trim to same length
            if val_losses:
                ax4.plot(epochs, val_losses, color='#2196F3', linewidth=2.5, marker='s', markersize=8, label='Val Loss')
                # Đánh dấu best epoch (dựa trên val loss)
                bestIdx = np.argmin(val_losses)
                ax4.scatter([bestIdx + 1], [val_losses[bestIdx]], color='#FFD700', s=200, zorder=5, marker='★', label=f'Best Val (Epoch {bestIdx+1})')
            else:
                # Fallback: đánh dấu best trên train loss
                bestIdx = np.argmin(history['epoch_avg_loss'])
                ax4.scatter([bestIdx + 1], [history['epoch_avg_loss'][bestIdx]], color='#FFD700', s=200, zorder=5, marker='★', label=f'Best (Epoch {bestIdx+1})')
            
            ax4.set_title('Train vs Validation Loss', fontsize=13)
            ax4.set_xticks(list(epochs))
        else:
            ax4.text(0.5, 0.5, 'Chưa có dữ liệu', ha='center', va='center', fontsize=12, transform=ax4.transAxes)
            ax4.set_title('Train vs Validation Loss', fontsize=13)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # --- 5. Validation LPIPS (mỗi epoch) ---
        ax5 = axes[2, 0]
        val_lpips = history.get('val_lpips', [])
        if val_lpips and any(v > 0 for v in val_lpips):
            epochs = range(1, len(val_lpips) + 1)
            ax5.plot(epochs, val_lpips, color='#00BCD4', linewidth=2.5, marker='D', markersize=8, label='Val LPIPS')
            # LPIPS thấp hơn = tốt hơn
            bestIdx = np.argmin([v if v > 0 else float('inf') for v in val_lpips])
            ax5.scatter([bestIdx + 1], [val_lpips[bestIdx]], color='#FFD700', s=200, zorder=5, marker='★', label=f'Best LPIPS (Epoch {bestIdx+1})')
            ax5.set_title('Validation LPIPS (↓ thấp = tốt)', fontsize=13)
            ax5.set_xticks(list(epochs))
        else:
            ax5.text(0.5, 0.5, 'LPIPS chưa có\n(cần pip install lpips)', ha='center', va='center', fontsize=12, transform=ax5.transAxes)
            ax5.set_title('Validation LPIPS', fontsize=13)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('LPIPS Score')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # --- 6. Epoch Summary Table ---
        ax6 = axes[2, 1]
        ax6.axis('off')
        if history['epoch_avg_loss']:
            table_data = []
            for i in range(len(history['epoch_avg_loss'])):
                train_l = f"{history['epoch_avg_loss'][i]:.5f}"
                val_l = f"{history.get('val_loss', [])[i]:.5f}" if i < len(history.get('val_loss', [])) else "N/A"
                lpips_l = f"{history.get('val_lpips', [])[i]:.4f}" if i < len(history.get('val_lpips', [])) and history.get('val_lpips', [])[i] > 0 else "N/A"
                table_data.append([f"Epoch {i+1}", train_l, val_l, lpips_l])
            
            table = ax6.table(
                cellText=table_data,
                colLabels=['Epoch', 'Train Loss', 'Val Loss', 'LPIPS'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.0, 1.5)
            ax6.set_title('Epoch Summary', fontsize=13, pad=20)
        
        plt.tight_layout()
        
        # Lưu file
        chartPath = charts_dir / f"loss_chart_epoch_{epoch}.png"
        latestPath = charts_dir / "loss_chart_latest.png"
        fig.savefig(str(chartPath), dpi=150, bbox_inches='tight')
        fig.savefig(str(latestPath), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"📊 Loss Chart đã lưu: {chartPath.name}")
        
        # Upload chart lên HF Hub
        if HF_TOKEN and HF_REPO_ID:
            try:
                from huggingface_hub import upload_file
                upload_file(
                    path_or_fileobj=str(latestPath),
                    path_in_repo=f"{HF_SUBFOLDER}/charts/loss_chart_latest.png",
                    repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, token=HF_TOKEN,
                    commit_message=f"chart: epoch {epoch}",
                )
                logger.info(f"  ☁️ HF: loss_chart_latest.png → {HF_REPO_ID}/{HF_SUBFOLDER}/charts/")
            except Exception as e:
                logger.warning(f"  ⚠️ HF upload chart failed: {e}")

    def _save_checkpoint(self, suffix: str, is_best: bool = False):
        """
        Lưu LoRA checkpoint (tiny ~50MB thay vì ~5GB full UNet).
        Chiến lược: CHỈ GIỮ file BEST + backup.
        """
        if is_best:
            best_lora = self.checkpoints_dir / "lora_best.safetensors"
            best_inj = self.checkpoints_dir / "injector_best.safetensors"
            logger.info(f"🏆 Saving BEST LoRA model ({suffix})...")
            self._save_safetensors_safe(self._get_lora_state_dict(), str(best_lora))
            self._save_safetensors_safe(self.injector.state_dict(), str(best_inj))
            
            if best_lora.exists() and best_inj.exists():
                logger.info(f"✅ BEST files verified: {best_lora.name} + {best_inj.name}")
            else:
                logger.error(f"❌ BEST files MISSING! Kiểm tra quyền ghi: {self.checkpoints_dir}")
        else:
            backup_lora = self.checkpoints_dir / "lora_backup.safetensors"
            backup_inj = self.checkpoints_dir / "injector_backup.safetensors"
            self._save_safetensors_safe(self._get_lora_state_dict(), str(backup_lora))
            self._save_safetensors_safe(self.injector.state_dict(), str(backup_inj))
        
        # Dọn dẹp: chỉ giữ LoRA files + legacy files
        keep_names = {
            "lora_best.safetensors",
            "injector_best.safetensors",
            "lora_backup.safetensors",
            "injector_backup.safetensors",
            "lora_latest.safetensors",
            "injector_latest.safetensors",
            # Legacy (backward compat)
            "deep_hair_v1_best.safetensors",
            "deep_hair_v1_latest.safetensors",
            "deep_hair_v1.safetensors",
            "texture_encoder_best.safetensors",
            "texture_encoder_latest.safetensors",
            "training_history.json",
        }
        for p in self.checkpoints_dir.glob("*.safetensors"):
            if p.name in keep_names:
                continue
            if IS_COLAB and ("_epoch_" in p.name):
                continue
            try:
                p.unlink()
                logger.info(f"  🗑️ Xóa checkpoint cũ: {p.name}")
            except:
                pass
    


    # ==============================================================================
    # CHUNKED LOADING — GLOBAL EPOCH TRAINING
    # ==============================================================================
    
    def _discover_chunk_dirs(self):
        """Tìm tất cả processed_NNN directories."""
        import re as _re
        training_dir = PROJECT_DIR / "backend" / "training"
        chunks = sorted([
            d for d in training_dir.iterdir()
            if d.is_dir() and _re.match(r'^processed_\d+$', d.name)
            and (d / "metadata.jsonl").exists()
        ])
        return chunks
    
    def _precache_all_chunks(self, chunk_dirs, target_size, max_samples_per_chunk):
        """Pre-encode text + style embeddings cho tất cả chunks chưa cache.

        Đếm đầy đủ file cache toàn bộ chunks, in báo cáo chi tiết, rồi chỉ
        encode/extract những chunks thực sự còn thiếu (không dò lại từ đầu).
        """
        # chunks_need_text / chunks_need_style: chỉ chứa chunks thực sự thiếu
        chunks_need_text  = []
        chunks_need_style = []

        logger.info("🔍 Kiểm tra trạng thái cache toàn bộ chunks...")
        for chunk_dir in chunk_dirs:
            meta_path = chunk_dir / "metadata.jsonl"
            if not meta_path.exists():
                continue

            cache_dir = chunk_dir / "prompt_embeddings"
            style_cache_dir = chunk_dir / "style_embeddings_cache"

            with open(str(meta_path), "r", encoding="utf-8") as f:
                all_items = [json.loads(line.strip()) for line in f if line.strip()]

            # Khi max_samples > 0 (smoke test): chỉ kiểm tra samples sẽ thực sự dùng
            # Dùng cùng seed 42 và logic random.sample giống HairInpaintingDataset
            if max_samples_per_chunk > 0 and len(all_items) > max_samples_per_chunk:
                rng = random.Random(42)
                check_items = rng.sample(all_items, max_samples_per_chunk)
            else:
                check_items = all_items

            n_expected = len(check_items)

            # ── Đếm file trong từng subdir cache ──────────────────────────────
            prompt_count = len(list(cache_dir.iterdir()))       if cache_dir.exists()       else 0
            style_count  = len(list(style_cache_dir.iterdir())) if style_cache_dir.exists() else 0

            # ── In báo cáo dạng "📦 chunk  →  📁 subdir → N files" ───────────
            logger.info(f"\n📦 {chunk_dir.name}  (expected: {n_expected} samples)")
            for subdir, count in [
                (cache_dir,       prompt_count),
                (style_cache_dir, style_count),
            ]:
                status = "✅" if count >= n_expected else "⚠️ "
                logger.info(f"  📁 {subdir.name:45s} → {count} files  {status}")

            # ── Track từng chunk: chỉ thêm vào list nếu thực sự thiếu ─────────
            if prompt_count < n_expected:
                chunks_need_text.append(chunk_dir)
            if style_count < n_expected:
                chunks_need_style.append(chunk_dir)

        needs_text  = len(chunks_need_text)  > 0
        needs_style = len(chunks_need_style) > 0

        text_encoder = None
        if needs_text:
            logger.info(f"📝 {len(chunks_need_text)} chunks cần encode prompt. Đang load Text Encoders...")
            text_encoder = SDXLTextEncoder()

        texture_encoder = None
        if needs_style:
            logger.info(f"🎨 {len(chunks_need_style)} chunks cần extract style. Đang load Texture Encoder...")
            from backend.training.models.texture_encoder import HairTextureEncoder
            from safetensors.torch import load_file as load_safetensors
            texture_encoder = HairTextureEncoder(pretrained=False).to(DEVICE).eval()
            tex_best = PROJECT_DIR / "backend" / "training" / "checkpoints" / "texture_encoder_best.safetensors"
            tex_latest = PROJECT_DIR / "backend" / "training" / "checkpoints" / "texture_encoder_latest.safetensors"
            tex_ckpt = tex_best if tex_best.exists() else tex_latest
            if tex_ckpt.exists():
                texture_encoder.load_state_dict(load_safetensors(str(tex_ckpt)), strict=False)
                logger.info(f"  → Loaded Texture Encoder từ {tex_ckpt.name}")
            else:
                logger.warning("  ⚠️ Không tìm thấy checkpoint Stage 1!")
            texture_encoder.requires_grad_(False)

        if text_encoder or texture_encoder:
            # Chỉ xử lý chunks thực sự thiếu (union của 2 list)
            chunks_to_process = sorted(
                set(chunks_need_text) | set(chunks_need_style),
                key=lambda d: d.name
            )
            # Truyền encoder tương ứng cho từng chunk
            chunk_pbar = tqdm(enumerate(chunks_to_process), total=len(chunks_to_process), desc="Pre-caching chunks")
            for i, chunk_dir in chunk_pbar:
                chunk_pbar.set_postfix({"chunk": chunk_dir.name})
                logger.info(f"  📂 Caching chunk {i+1}/{len(chunks_to_process)}: {chunk_dir.name}")
                # Chỉ truyền encoder nếu chunk đó thực sự thiếu loại đó
                _te  = text_encoder    if chunk_dir in chunks_need_text  else None
                _ste = texture_encoder if chunk_dir in chunks_need_style else None
                _ = HairInpaintingDataset(
                    chunk_dir, text_encoder=_te, texture_encoder=_ste,
                    target_size=target_size, max_samples=max_samples_per_chunk
                )
                del _
            chunk_pbar.close()
            
            if text_encoder:
                text_encoder.unload()
                del text_encoder
            if texture_encoder:
                del texture_encoder
            torch.cuda.empty_cache()
            logger.info("✅ Pre-caching hoàn tất, encoders đã giải phóng.")
        else:
            logger.info("✅ Tất cả embeddings đã cache đầy đủ — skip pre-caching")
    

    
    def _validate_across_chunks(self, chunk_dirs, batch_size, target_size, global_step):
        """Validate qua tối đa 3 chunks CỐ ĐỊNH, lấy mỗi chunk ~30 samples.
        Seed cố định để val set nhất quán giữa các epoch → metric so sánh được."""
        all_val_losses = []
        all_val_lpips = []
        
        # Chọn tối đa 3 chunks CỐ ĐỊNH (sorted + lấy đầu) để val set ổn định
        # Không random shuffle — đảm bảo cùng val set giữa các epoch
        val_chunks = sorted(chunk_dirs, key=lambda d: d.name)[:min(3, len(chunk_dirs))]
        
        for chunk_dir in val_chunks:
            try:
                dataset = HairInpaintingDataset(
                    chunk_dir, target_size=target_size, max_samples=60
                )
                if len(dataset) < 2:
                    continue
                
                # Split 80/20 (thay vì 50/50) với seed cố định
                val_size = max(1, len(dataset) // 5)  # 20% cho validation
                train_size = len(dataset) - val_size
                _, val_subset = random_split(
                    dataset, [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
                
                val_loader = DataLoader(
                    val_subset, batch_size=batch_size,
                    shuffle=False, drop_last=False, num_workers=0, pin_memory=True
                )
                
                metrics = self.validate_epoch(val_loader, global_step)
                all_val_losses.append(metrics['val_loss'])
                if metrics['val_lpips'] >= 0:
                    all_val_lpips.append(metrics['val_lpips'])
                
                del dataset, val_subset, val_loader
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"  ⚠️ Validation lỗi cho {chunk_dir.name}: {e}")
                continue
        
        avg_val = sum(all_val_losses) / len(all_val_losses) if all_val_losses else float('inf')
        avg_lpips = sum(all_val_lpips) / len(all_val_lpips) if all_val_lpips else -1.0
        return {'val_loss': avg_val, 'val_lpips': avg_lpips}
    
    # ==============================================================================
    # MAIN TRAINING LOOP — Chunked Loading, Global Epoch
    # ==============================================================================
    
    def train_loop(self, num_epochs=1, batch_size=1, max_samples_per_chunk=0, target_size=(512, 512), accumulation_steps=8, resume=True, chunk_names=None):
        """
        Chunked Loading – Global Epoch Training.
        1 epoch = model nhìn thấy TẤT CẢ chunks đúng 1 lần.
        Model, optimizer, scheduler KHÔNG reset giữa các chunks.
        
        Args:
            num_epochs: Số epochs (1 epoch = tất cả chunks)
            batch_size: Batch size (1 cho RTX 3060 12GB)
            max_samples_per_chunk: Giới hạn samples mỗi chunk (0 = tất cả)
            target_size: Kích thước ảnh (512x512)
            accumulation_steps: Gradient accumulation steps
            resume: Resume từ checkpoint nếu có
            chunk_names: List tên các chunk cụ thể để train (vd: ['processed_001', 'processed_002']). 
                         Nếu None, train trên TẤT CẢ chunks tìm thấy.
        """
        logger.info(f"Khởi động Chunked Loading – Global Epoch Training")
        logger.info(f"  📐 Resolution: {target_size[0]}x{target_size[1]}")
        logger.info(f"  🔄 Gradient Accumulation: {accumulation_steps} steps (effective batch = {batch_size * accumulation_steps})")
        logger.info(f"  💾 Checkpoint: Local=BEST+backup | Drive={'LƯU TẤT CẢ epoch' if IS_COLAB else 'N/A'}")
        
        # ==================================================
        # 1. DISCOVER CHUNKS
        # ==================================================
        chunk_dirs = self._discover_chunk_dirs()
        if chunk_names:
            # Lọc chỉ lấy các chunks có tên trong chunk_names
            chunk_dirs = [d for d in chunk_dirs if d.name in chunk_names]
            if not chunk_dirs:
                logger.error(f"❌ Không tìm thấy chunk nào khớp với list: {chunk_names}")
                return
        
        if not chunk_dirs:
            logger.error("❌ Không tìm thấy bất kỳ thư mục processed_NNN nào! Hãy chạy prepare_dataset_deephair.py trước.")
            return
        
        logger.info(f"  📂 Found {len(chunk_dirs)} chunk(s): {[d.name for d in chunk_dirs]}")
        
        # ==================================================
        # 2. PRE-CACHE EMBEDDINGS cho tất cả chunks
        # ==================================================
        self._precache_all_chunks(chunk_dirs, target_size, max_samples_per_chunk)
        
        # ==================================================
        # 2b. RESUME FROM CHECKPOINT (TRƯỚC KHI reconfigure scheduler)
        # ==================================================
        start_epoch = 0
        global_step = 0
        best_val_loss = float('inf')
        best_epoch = -1
        loss_history = {
            'total_loss': [], 'diffusion_loss': [], 'texture_loss': [],
            'identity_loss': [], 'epoch_avg_loss': [], 'val_loss': [], 'val_lpips': [],
        }
        
        resume_chunk_index = -1  # -1 = bắt đầu epoch mới
        resume_chunk_names = []   # thứ tự chunks đã shuffle của epoch bị interrupt
        resume_step_in_chunk = 0  # số batches đã train trong chunk bị interrupt
        
        # === LOAD PREVIOUS WEIGHTS (nếu có) ===
        # Luồng: HF Hub download → local check → load weights
        # Trên Colab, /tmp/ bị xóa mỗi lần reset → cần tải lại từ HF Hub
        if resume:
            # Bước 1: Download từ HF Hub nếu local chưa có
            if IS_COLAB and HF_TOKEN and HF_REPO_ID:
                try:
                    from huggingface_hub import hf_hub_download
                    hf_files = [
                        "lora_best.safetensors", "injector_best.safetensors",
                        "lora_latest.safetensors", "injector_latest.safetensors",
                    ]
                    downloaded = []
                    for fname in hf_files:
                        local_path = self.checkpoints_dir / fname
                        if not local_path.exists():
                            try:
                                hf_hub_download(
                                    repo_id=HF_REPO_ID,
                                    filename=f"{HF_SUBFOLDER}/{fname}",
                                    repo_type=HF_REPO_TYPE,
                                    token=HF_TOKEN,
                                    local_dir=str(self.checkpoints_dir),
                                    local_dir_use_symlinks=False,
                                )
                                # hf_hub_download lưu vào subfolder, move ra ngoài
                                subfolder_path = self.checkpoints_dir / HF_SUBFOLDER / fname
                                if subfolder_path.exists() and not local_path.exists():
                                    shutil.move(str(subfolder_path), str(local_path))
                                if local_path.exists():
                                    downloaded.append(fname)
                            except Exception:
                                pass  # File chưa có trên Hub → bỏ qua
                    if downloaded:
                        logger.info(f"  ☁️ Downloaded từ HF Hub: {downloaded}")
                    else:
                        logger.info(f"  ☁️ Không tìm thấy checkpoint trên HF Hub — train mới")
                except Exception as e:
                    logger.warning(f"  ⚠️ HF Hub download failed: {e}")
            
            # Bước 2: Load weights từ local (đã có sẵn hoặc vừa download)
            best_lora = self.checkpoints_dir / "lora_best.safetensors"
            best_inj = self.checkpoints_dir / "injector_best.safetensors"
            latest_lora = self.checkpoints_dir / "lora_latest.safetensors"
            latest_inj = self.checkpoints_dir / "injector_latest.safetensors"
            
            loaded_from = None
            if best_lora.exists() and best_inj.exists():
                loaded_from = "best"
                load_lora, load_inj = best_lora, best_inj
            elif latest_lora.exists() and latest_inj.exists():
                loaded_from = "latest"
                load_lora, load_inj = latest_lora, latest_inj
            
            if loaded_from:
                try:
                    from safetensors.torch import load_file as load_safetensors
                    self._load_lora_weights(str(load_lora))
                    self.injector.load_state_dict(load_safetensors(str(load_inj)))
                    lora_size = load_lora.stat().st_size / (1024**2)
                    logger.info(f"🔄 [RESUME] Loaded {loaded_from} weights: {load_lora.name} ({lora_size:.1f} MB) + {load_inj.name}")
                    logger.info(f"  ⚠️ Optimizer/Scheduler bắt đầu fresh (chỉ load model weights)")
                except Exception as e:
                    logger.error(f"❌ Load weights failed: {e} — train từ đầu")
            else:
                logger.info("🆕 Không tìm thấy checkpoint — train từ đầu")
        else:
            logger.info("🆕 --fresh flag — train từ đầu (bỏ qua checkpoint cũ)")
        
        # ==================================================
        # 3. RECONFIGURE LR SCHEDULER dựa trên dataset size thực tế
        #    (SAU resume để không bị state cũ ghi đè T_max)
        # ==================================================
        # Ước tính tổng steps: đếm samples từ metadata → tính steps/epoch
        total_samples = 0
        for chunk_dir in chunk_dirs:
            meta_path = chunk_dir / "metadata.jsonl"
            if meta_path.exists():
                with open(str(meta_path), "r", encoding="utf-8") as f:
                    n = sum(1 for line in f if line.strip())
                if max_samples_per_chunk > 0:
                    n = min(n, max_samples_per_chunk)
                total_samples += n
        
        if total_samples > 0:
            steps_per_epoch = total_samples // (batch_size * accumulation_steps)
            estimated_total_steps = steps_per_epoch * num_epochs
            new_t_max = max(estimated_total_steps - self._warmup_steps, 500)  # minimum 500
            
            # Reconfigure cosine scheduler T_max (ghi đè giá trị từ checkpoint nếu có)
            self.scheduler._schedulers[1].T_max = new_t_max
            logger.info(f"  📐 LR Scheduler: warmup={self._warmup_steps} → cosine T_max={new_t_max} (est. {estimated_total_steps} total optimizer steps)")
        
        logger.info(f"  📊 Training: {num_epochs} epochs × {len(chunk_dirs)} chunk(s)")
        logger.info(f"  📊 Max samples/chunk: {max_samples_per_chunk if max_samples_per_chunk > 0 else 'ALL'}")
        
        step_times = []
        
        # ==================================================
        # 4. TRAINING LOOP — 1 epoch = TẤT CẢ chunks
        # ==================================================
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            epoch_losses = []
            
            # Shuffle thứ tự chunks mỗi epoch
            # Nếu resume giữa epoch → dùng lại thứ tự chunks cũ
            if resume_chunk_index >= 0 and resume_chunk_names and epoch == start_epoch:
                # Khôi phục thứ tự chunks từ state đã lưu
                name_to_dir = {d.name: d for d in chunk_dirs}
                shuffled_chunks = [name_to_dir[n] for n in resume_chunk_names if n in name_to_dir]
                if resume_step_in_chunk > 0:
                    # Chunk bị interrupt giữa chừng → quay lại chunk đó
                    skip_until = resume_chunk_index
                    logger.info(f"\n🔄 RESUME Epoch {epoch+1} — skip {skip_until} chunks, resume chunk {resume_chunk_index+1} từ batch {resume_step_in_chunk}")
                else:
                    skip_until = resume_chunk_index + 1  # chunk đã xong → skip
                    logger.info(f"\n🔄 RESUME Epoch {epoch+1} — skip {skip_until} chunks đã train")
            else:
                shuffled_chunks = chunk_dirs.copy()
                random.shuffle(shuffled_chunks)
                skip_until = 0
            
            chunk_names = [d.name for d in shuffled_chunks]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch+1}/{num_epochs} — {len(shuffled_chunks)} chunk(s)")
            logger.info(f"  Chunk order: {chunk_names}")
            if skip_until > 0:
                logger.info(f"  ⏭️ Skipping chunks 1-{skip_until} (đã train)")
            logger.info(f"{'='*60}")
            
            for chunk_idx, chunk_dir in enumerate(shuffled_chunks):
                # Skip chunks đã train xong (resume)
                if chunk_idx < skip_until:
                    logger.info(f"  ⏭️ Skip chunk {chunk_idx+1}/{len(shuffled_chunks)}: {chunk_dir.name} (đã train)")
                    continue
                chunk_start = time.time()
                
                # Load dataset cho chunk hiện tại (lazy loading — không tốn RAM cho embeddings)
                dataset = HairInpaintingDataset(
                    chunk_dir, target_size=target_size, max_samples=max_samples_per_chunk
                )
                
                if len(dataset) == 0:
                    logger.warning(f"  ⚠️ Chunk {chunk_dir.name} trống, bỏ qua.")
                    continue
                
                num_workers = 0 if os.name == 'nt' else 2
                # Deterministic DataLoader: cùng seed → cùng thứ tự batch → resume chính xác
                dl_seed = epoch * 10000 + chunk_idx
                dl_generator = torch.Generator().manual_seed(dl_seed)
                dataloader = DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, generator=dl_generator,
                    drop_last=True, num_workers=num_workers, pin_memory=True
                )
                
                # Mid-chunk resume: skip batches đã train
                skip_steps = 0
                if chunk_idx == resume_chunk_index and resume_step_in_chunk > 0 and epoch == start_epoch:
                    skip_steps = resume_step_in_chunk
                
                logger.info(f"\n  📂 Chunk {chunk_idx+1}/{len(shuffled_chunks)}: {chunk_dir.name} ({len(dataset)} samples)")
                if skip_steps > 0:
                    logger.info(f"  ⏩ Skipping {skip_steps} batches đã train (mid-chunk resume)...")
                
                pbar = tqdm(dataloader, desc=f"E{epoch+1} C{chunk_idx+1}/{len(shuffled_chunks)}")
                for step, batch in enumerate(pbar):
                    # Skip batches đã train (mid-chunk resume)
                    if step < skip_steps:
                        if step == 0 or (step + 1) % 100 == 0:
                            pbar.set_postfix({"skip": f"{step+1}/{skip_steps}"})
                        continue
                    
                    step_start = time.time()
                    losses = self.train_step(batch, global_step, accumulation_steps=accumulation_steps)
                    step_time = time.time() - step_start
                    step_times.append(step_time)
                    epoch_losses.append(losses['total_loss'])
                    
                    # Ghi loss history
                    loss_history['total_loss'].append(losses['total_loss'])
                    loss_history['diffusion_loss'].append(losses['diffusion_loss'])
                    loss_history['texture_loss'].append(losses['texture_loss'])
                    loss_history['identity_loss'].append(losses.get('identity_loss', 0.0))
                    
                    # ETA
                    avg_step_time = sum(step_times[-50:]) / len(step_times[-50:])
                    remaining_steps = len(dataloader) - step - 1
                    eta_seconds = remaining_steps * avg_step_time
                    eta_str = f"{int(eta_seconds//3600)}h{int((eta_seconds%3600)//60)}m"
                    
                    pbar.set_postfix({
                        "Loss": f"{losses['total_loss']:.4f}",
                        "Diff": f"{losses['diffusion_loss']:.4f}",
                        "Tex": f"{losses['texture_loss']:.4f}",
                        "ID": f"{losses.get('identity_loss', 0.0):.4f}",
                        "Best": f"{best_val_loss:.4f}" if best_val_loss < float('inf') else "N/A",
                        "s/it": f"{step_time:.1f}s",
                        "ETA": eta_str,
                    })
                    global_step += 1
                    
                    # SIGINT/SIGTERM check: graceful shutdown
                    if self._interrupted:
                        logger.warning(f"🛑 Graceful shutdown — thoát training...")
                        logger.info(f"{'='*60}")
                        return  # Thoát sạch (không raise exception)
                
                chunk_time = time.time() - chunk_start
                chunk_avg = sum(epoch_losses[-len(dataloader):]) / max(1, len(dataloader))
                logger.info(f"  ✅ Chunk {chunk_dir.name} done — Avg Loss: {chunk_avg:.5f} — Time: {chunk_time/60:.1f}min")
                
                # Giải phóng memory sau mỗi chunk
                del dataset, dataloader
                torch.cuda.empty_cache()
                
                # Không save giữa chừng — chỉ save cuối cùng khi training hoàn tất
                
                # Reset mid-chunk resume flag sau khi chunk resume xong
                if resume_step_in_chunk > 0:
                    resume_step_in_chunk = 0
            
            # ==================================================
            # VALIDATION sau mỗi epoch đầy đủ
            # ==================================================
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
            
            logger.info(f"\nĐang validate Epoch {epoch+1} qua {len(chunk_dirs)} chunk(s)...")
            val_metrics = self._validate_across_chunks(
                chunk_dirs, batch_size, target_size, global_step
            )
            val_loss = val_metrics['val_loss']
            val_lpips = val_metrics['val_lpips']
            
            is_new_best = val_loss < best_val_loss
            val_lpips_str = f"{val_lpips:.4f}" if val_lpips >= 0 else "N/A"
            
            if is_new_best:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                logger.info(f"🏆 NEW BEST! Epoch {epoch+1} — Train: {avg_epoch_loss:.6f} — Val: {val_loss:.6f} — LPIPS: {val_lpips_str} — Time: {epoch_time/60:.1f}min")
            else:
                logger.info(f"Epoch {epoch+1} — Train: {avg_epoch_loss:.6f} — Val: {val_loss:.6f} — LPIPS: {val_lpips_str} (Best: Ep{best_epoch}, {best_val_loss:.6f}) — Time: {epoch_time/60:.1f}min")
            
            # Ghi epoch loss history
            loss_history['epoch_avg_loss'].append(avg_epoch_loss)
            loss_history['val_loss'].append(val_loss)
            loss_history['val_lpips'].append(val_lpips if val_lpips >= 0 else 0.0)
            
            # === SAVE WEIGHTS SAU MỖI EPOCH ===
            # An toàn vì training loop đã dừng, không có GPU conflict
            # ⚠️ QUAN TRỌNG: Giải phóng RAM trước khi save!
            # Sau validation + LPIPS, RAM hệ thống gần hết.
            # Nếu không dọn → OOM Killer gửi ^C giết process ngay lúc save_file().
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"  💾 Saving epoch {epoch+1} weights...")
            
            # Save latest (luôn luôn) — injector trước (nhỏ, nhanh), lora sau (lớn, lâu)
            self._save_safetensors_safe(self.injector.state_dict(),
                                       str(self.checkpoints_dir / "injector_latest.safetensors"))
            self._save_safetensors_safe(self._get_lora_state_dict(),
                                       str(self.checkpoints_dir / "lora_latest.safetensors"))
            
            # Save best (chỉ khi val_loss tốt hơn)
            if is_new_best:
                self._save_safetensors_safe(self.injector.state_dict(),
                                           str(self.checkpoints_dir / "injector_best.safetensors"))
                self._save_safetensors_safe(self._get_lora_state_dict(),
                                           str(self.checkpoints_dir / "lora_best.safetensors"))
                logger.info(f"  🏆 Best weights saved (Val: {val_loss:.6f})")
            
            # Save training history JSON (upload riêng, không qua _save_safetensors_safe)
            history_path = self.checkpoints_dir / "training_history.json"
            with open(str(history_path), "w", encoding="utf-8") as f:
                json.dump(loss_history, f, indent=2)
            if HF_TOKEN and HF_REPO_ID:
                try:
                    from huggingface_hub import upload_file
                    upload_file(
                        path_or_fileobj=str(history_path),
                        path_in_repo=f"{HF_SUBFOLDER}/training_history.json",
                        repo_id=HF_REPO_ID,
                        repo_type=HF_REPO_TYPE,
                        token=HF_TOKEN,
                        commit_message=f"epoch {epoch+1}: training_history.json",
                    )
                except Exception:
                    pass
            
            logger.info(f"  ✅ Epoch {epoch+1} saved: injector + lora → /tmp/ + HF Hub")
            
            # Reset resume flags sau epoch đầu tiên
            resume_chunk_index = -1
            resume_chunk_names = []
            skip_until = 0
            
            # Plot training charts
            self._plot_training_charts(loss_history, epoch + 1)
        
        # ==================================================
        # FINAL: Save latest LoRA model
        # ==================================================
        final_lora_path = self.checkpoints_dir / "lora_latest.safetensors"
        self._save_safetensors_safe(self._get_lora_state_dict(), str(final_lora_path))
        final_inj_path = self.checkpoints_dir / "injector_latest.safetensors"
        self._save_safetensors_safe(self.injector.state_dict(), str(final_inj_path))
        
        # Xóa backup
        for backup in ["lora_backup.safetensors", "injector_backup.safetensors"]:
            bp = self.checkpoints_dir / backup
            if bp.exists():
                try: bp.unlink()
                except: pass
        
        total_size = sum(f.stat().st_size for f in self.checkpoints_dir.glob("*.safetensors"))
        
        logger.info(f"{'='*60}")
        logger.info(f"✅ Hoàn thành LoRA Training – Global Epoch!")
        logger.info(f"  📁 LoRA cuối: {final_lora_path}")
        logger.info(f"  🏆 LoRA tốt nhất: lora_best.safetensors (Epoch {best_epoch}, Val: {best_val_loss:.6f})")
        logger.info(f"  💾 Tổng checkpoints: {total_size / (1024**2):.1f} MB")
        logger.info(f"  📂 Chunks trained: {len(chunk_dirs)}")
        logger.info(f"{'='*60}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Stage 2 - Hair Inpainting (Chunked Loading – Global Epoch)")
    parser.add_argument("--epochs", type=int, default=1, help="Số epochs (1 epoch = tất cả chunks)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--max-samples-per-chunk", type=int, default=0, help="Giới hạn samples mỗi chunk (0=tất cả)")
    parser.add_argument("--resolution", type=int, default=512, help="Kích thước ảnh (512 hoặc 1024)")
    parser.add_argument("--accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--fresh", action="store_true", help="Train từ đầu, KHÔNG load checkpoint cũ")
    parser.add_argument("--chunk-names", type=str, default="", help="Chỉ định cụ thể chunks để train, cách nhau bằng dấu phẩy (vd: processed_001,processed_002)")
    args = parser.parse_args()
    
    chunk_names_list = [name.strip() for name in args.chunk_names.split(",")] if args.chunk_names else None
    
    trainer = Stage2Trainer()
    trainer.train_loop(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples_per_chunk=args.max_samples_per_chunk,
        target_size=(args.resolution, args.resolution),
        accumulation_steps=args.accumulation,
        resume=not args.fresh,
        chunk_names=chunk_names_list
    )
