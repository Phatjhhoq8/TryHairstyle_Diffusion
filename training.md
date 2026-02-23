# Hoạt Đồ Hệ Thống Huấn Luyện (Training Pipeline)

## I. Tổng Quan Pipeline

Pipeline huấn luyện gồm **4 Giai đoạn** (Stages) chạy tuần tự, mỗi giai đoạn phụ thuộc đầu ra của giai đoạn trước.

```mermaid
flowchart LR
    S0["Stage 0\nChuẩn Bị Dataset"]
    S1["Stage 1\nTexture Encoder"]
    S2["Stage 2\nUNet Inpainting"]
    S3["Stage 3\nEvaluate & Export"]

    S0 -->|"processed/"| S1
    S1 -->|"texture_encoder.safetensors"| S2
    S2 -->|"deep_hair_v1.safetensors"| S3
    S3 -->|"Deploy"| PROD["Production\nbackend/training/models/"]

    classDef stage fill:#4a9eff,color:white,stroke:#2668b5,stroke-width:2px;
    classDef prod fill:#2ecc71,color:white,stroke:#27ae60,stroke-width:3px;
    class S0,S1,S2,S3 stage;
    class PROD prod;
```

**File điều phối:** `run_training_pipeline.sh` — chạy lần lượt 4 lệnh Python.

---

## II. Stage 0 — Chuẩn Bị Dataset

**File:** `prepare_dataset_deephair.py`

### Mục tiêu
Chuyển đổi ảnh raw + JSON labels (K-Hairstyle) thành các tensor/ảnh sẵn sàng cho training.

### Hoạt đồ chi tiết

```mermaid
flowchart TD
    subgraph INPUT["📂 Input"]
        IMG["Ảnh JPG/PNG\n(K-Hairstyle)"]
        JSON["JSON Labels\n(polygon, color, curl,\nlength, bang...)"]
        DICT["mapping_dict.json\n(Hàn → Anh)"]
    end

    IMG --> WORKER["process_single_image()"]
    JSON --> WORKER
    DICT --> PROMPT["generate_text_prompt()"]
    JSON --> PROMPT

    WORKER --> STEP1["1. Đọc JSON polygon\n→ cv2.fillPoly()\n→ Hair Mask"]
    STEP1 --> STEP2["2. Detect Face\n(YOLO)\n→ Bbox lớn nhất"]
    STEP2 --> STEP3["3. cv2.inpaint()\n→ Bald Image\n(Xóa tóc)"]
    STEP3 --> STEP4["4. Merge RGBA\n→ Hair Only\n(Tóc + Alpha)"]
    STEP4 --> STEP5["5. AdaFace Embedding\n→ Identity .npy\n(512-dim vector)"]
    STEP5 --> STEP6["6. Crop + Resize\n→ Style Vector\n(224×224 image)"]
    STEP6 --> STEP7["7. Sliding Window\n→ Hair Patches\n(128×128, ratio≥85%)"]
    PROMPT --> STEP8["8. Text Prompt\n(Tiếng Anh)"]

    subgraph OUTPUT["📁 processed/"]
        OUT1["bald_images/*.png"]
        OUT2["hair_only_images/*.png"]
        OUT3["hair_patches/*.png"]
        OUT4["style_vectors/*.png"]
        OUT5["identity_embeddings/*.npy"]
        OUT6["metadata.jsonl"]
    end

    STEP3 --> OUT1
    STEP4 --> OUT2
    STEP7 --> OUT3
    STEP6 --> OUT4
    STEP5 --> OUT5
    STEP8 --> OUT6

    classDef input fill:#ffeaa7,stroke:#fdcb6e,stroke-width:2px;
    classDef process fill:#74b9ff,stroke:#0984e3,stroke-width:1px;
    classDef output fill:#55efc4,stroke:#00b894,stroke-width:2px;
    class IMG,JSON,DICT input;
    class STEP1,STEP2,STEP3,STEP4,STEP5,STEP6,STEP7,STEP8,WORKER,PROMPT process;
    class OUT1,OUT2,OUT3,OUT4,OUT5,OUT6 output;
```

### Đa luồng
`ProcessPoolExecutor` (tối đa 8 workers) xử lý song song nhiều ảnh, mỗi worker khởi tạo riêng YOLO detector + AdaFace embedder.

---

## III. Stage 1 — Hair Texture Encoder

**File:** `models/texture_encoder.py`

### Mục tiêu
Huấn luyện mạng **ResNet50** để "hiểu" texture tóc: xoăn/thẳng, dày/mỏng, lọn/sợi.

### Kiến trúc Model

```mermaid
flowchart LR
    INPUT["Hair Patch\n128×128×3"]
    BB["ResNet50\nBackbone\n(ImageNet pretrained)"]
    EMBED["Embedding\n2048-dim"]

    INPUT --> BB --> EMBED

    EMBED --> PROJ["Projection Head\nMLP 2-layer\n→ 128-dim"]
    EMBED --> CLS1["Curl Classifier\nDropout + Linear\n→ 4 classes"]
    EMBED --> CLS2["Volume Classifier\nDropout + Linear\n→ 3 classes"]

    classDef model fill:#a29bfe,color:white,stroke:#6c5ce7,stroke-width:2px;
    classDef head fill:#fd79a8,color:white,stroke:#e84393,stroke-width:1px;
    class BB model;
    class PROJ,CLS1,CLS2 head;
```

### Quá trình Training

```mermaid
flowchart TD
    DATA["HairTextureDataset\n(Patches 128×128\n+ nhãn curl/volume\ntừ metadata)"]

    DATA --> FWD["Forward Pass\nResNet50 → embed, proj, cls"]
    FWD --> L1["Loss 1: CrossEntropyLoss\n(curl_logits vs curl_label)\n(volume_logits vs volume_label)"]
    FWD --> L2["Loss 2: SupConLoss\n(Supervised Contrastive)\nKéo patches cùng nhãn lại gần\nĐẩy patches khác nhãn ra xa"]

    L1 --> TOTAL["total_loss = L1 + 0.5 × L2"]
    L2 --> TOTAL
    TOTAL --> BACK["Backprop + AdamW\n(lr=5e-4)"]
    BACK --> CKPT["Checkpoint mỗi 500 steps\n→ stage1_step_N.safetensors"]

    classDef loss fill:#e17055,color:white,stroke:#d63031,stroke-width:1px;
    class L1,L2,TOTAL loss;
```

### Nhãn phân loại (từ text_prompt metadata)

| Curl (4 classes) | Volume (3 classes) |
|---|---|
| 0 = straight (thẳng) | 0 = low (ít) |
| 1 = wavy (vểnh) | 1 = normal (bình thường) |
| 2 = curly (xoăn) | 2 = high (nhiều) |
| 3 = tightly curly (xoăn tít) | |

---

## IV. Stage 2 — Mask-Conditioned Hair Inpainting (Core)

**Files:** `train_stage2.py`, `models/stage2_unet.py`

### Mục tiêu
Fine-tune UNet SDXL để vẽ tóc mới vào vùng mask, giữ nguyên khuôn mặt.

### Kiến trúc UNet 9-Channel

```mermaid
flowchart TD
    subgraph ENCODE["VAE Encode (Frozen, fp16)"]
        GT["Ground Truth Image"] --> VAE1["VAE\nEncoder"] --> LAT["Latents\n4 channels"]
        BALD["Bald Image\n(Đã xóa tóc)"] --> VAE2["VAE\nEncoder"] --> BLAT["Bald Latents\n4 channels"]
    end

    MASK["Hair Mask\n(Downsampled)\n1 channel"]

    LAT --> NOISE["Add Noise\n(DDPMScheduler)"]
    NOISE --> CONCAT

    subgraph CONCAT["Concat → 9 Channels"]
        direction LR
        C1["Noisy Latents (4ch)"]
        C2["Mask (1ch)"]
        C3["Bald Latents (4ch)"]
    end

    BLAT --> CONCAT
    MASK --> CONCAT

    CONCAT --> UNET["SDXL UNet\n(9-ch conv_in patched)\nGradient Checkpointing"]

    subgraph COND["Cross-Attention Conditioning"]
        TEXT["Text Prompt\n→ CLIP Encode\n→ (77, 2048)"]
        STYLE["Style Embed\n→ Linear+LayerNorm\n→ (1, 2048)"]
        ID["Identity Embed\n→ Linear+GELU+Linear\n→ (1, 2048)"]
    end

    TEXT --> CAT_COND["Concat Sequence"]
    STYLE --> CAT_COND
    ID --> CAT_COND
    CAT_COND -->|"encoder_hidden_states"| UNET

    UNET --> PRED["Noise Prediction"]

    classDef vae fill:#00cec9,color:white,stroke:#00b894;
    classDef unet fill:#6c5ce7,color:white,stroke:#5f3dc4,stroke-width:3px;
    classDef cond fill:#fdcb6e,stroke:#f39c12;
    class VAE1,VAE2 vae;
    class UNET unet;
    class TEXT,STYLE,ID,CAT_COND cond;
```

### Luồng Training 1 Step

```mermaid
flowchart TD
    BATCH["DataLoader\n(batch_size=1)"]
    BATCH --> VAE_ENC["VAE Encode\n(frozen, fp16)\nGT → latents\nBald → bald_latents"]

    VAE_ENC --> ADD_NOISE["DDPMScheduler\nadd_noise(latents, noise,\nrandom timestep 0~999)"]

    ADD_NOISE --> FWD["UNet Forward\n(AMP fp16)\n9-ch input + conditions\n→ noise_pred"]

    FWD --> LOSS1["Loss 1: MaskAwareLoss\n|(noise_pred - noise)² × mask|\nChỉ vùng tóc, khóa mặt"]

    FWD -->|"Mỗi 10 steps"| LOSS2["Loss 2: TextureConsistencyLoss\nDecode latents → RGB\nVGG16 Gram Matrix\nSo sánh texture"]

    LOSS1 --> TOTAL["total = L1 + 0.01×L2"]
    LOSS2 --> TOTAL

    TOTAL --> BACK["AMP Backprop\n+ Gradient Clipping (norm=1.0)\n+ AdamW (lr=1e-5)"]

    BACK --> SAVE["Checkpoint:\n- Mỗi 500 steps\n- Cuối mỗi epoch\n→ .safetensors"]

    classDef loss fill:#e17055,color:white;
    classDef save fill:#2ecc71,color:white;
    class LOSS1,LOSS2,TOTAL loss;
    class SAVE save;
```

### VRAM Budget (GPU 12GB — RTX 3060)

| Component | VRAM | Ghi chú |
|---|---|---|
| UNet (fp16) | ~5 GB | Gradient checkpointing + xformers ON |
| VAE (fp16, frozen) | ~0.5 GB | Chỉ encode, enable_slicing() |
| Text Encoders | **0 GB** | Encode xong → giải phóng |
| Optimizer states | ~2.5 GB | 8-bit AdamW (bitsandbytes) |
| Activations + Gradients | ~3-4 GB | AMP fp16 + xformers + grad accumulation |
| **Tổng ước tính** | **~10-11 GB** | Fit trong 12GB ✅ |

> **Kỹ thuật tối ưu đã áp dụng:**
> - xformers memory-efficient attention
> - 8-bit AdamW optimizer (bitsandbytes)
> - VAE slicing (encode/decode từng slice)
> - Gradient Accumulation (4 steps)
> - Texture Loss giảm tần suất (mỗi 50 steps)

### Best Model Tracking

Sau mỗi epoch, hệ thống **tự so sánh avg loss** và lưu model tốt nhất:

| File | Ý nghĩa |
|---|---|
| `deep_hair_v1_best.safetensors` | ⭐ Model có **avg epoch loss thấp nhất** — dùng để deploy |
| `deep_hair_v1_latest.safetensors` | Model epoch cuối cùng (có thể overfit) |
| `stage2_step_N.safetensors` | Checkpoint mỗi 500 steps |
| `stage2_epoch_N.safetensors` | Checkpoint cuối mỗi epoch |

> **Export (`export_model.py`) ưu tiên:** best → latest → mới nhất theo thời gian

### Pre-encode Workflow (Text Prompts)

```mermaid
sequenceDiagram
    participant TE as Text Encoders (CLIP)
    participant DS as Dataset
    participant GPU as GPU VRAM

    Note over TE,GPU: Giai đoạn 1: Encode (Chỉ 1 lần)
    DS->>TE: Load metadata.jsonl
    TE->>GPU: Load CLIPTextModel + CLIPTextModelWithProjection (~3GB)
    loop Mỗi sample
        TE->>TE: tokenize(prompt) → encode → (77, 2048) + (1280,)
        TE->>DS: Save cache .pt vào processed/prompt_embeddings/
    end
    TE->>GPU: Giải phóng Text Encoders (del + empty_cache)

    Note over DS,GPU: Giai đoạn 2: Training (Nhiều epochs)
    DS->>DS: Load .pt cache trực tiếp (không cần Text Encoders)
    DS->>GPU: Chỉ UNet + VAE trên VRAM
```

---

## V. Stage 3 — Evaluate & Export

**Files:** `evaluate.py`, `export_model.py`

### Hoạt đồ

```mermaid
flowchart TD
    FIND["Tìm checkpoint mới nhất\n(*stage2*.safetensors)"]
    FIND --> LOAD["Load state_dict\n(safetensors)"]
    LOAD --> VALID{"Checkpoint hợp lệ?\n(>10 params, có conv_in)"}

    VALID -->|"Không"| FAIL["❌ Model chưa đủ\nchất lượng"]
    VALID -->|"Có"| EVAL["Chạy Evaluation\ntrên 5 validation samples"]

    EVAL --> LPIPS["LPIPS Score\n(VGG Perceptual)\nCàng thấp càng tốt"]
    EVAL --> PSNR["Masked PSNR\nChỉ tính vùng tóc\nCàng cao càng tốt"]

    LPIPS --> DECIDE{"Đạt threshold?"}
    PSNR --> DECIDE
    DECIDE -->|"Đạt"| COPY["shutil.copy2()\ncheckpoint → models/\ndeep_hair_v1.safetensors"]
    DECIDE -->|"Không đạt"| FAIL

    COPY --> DONE["✅ Deploy xong!\nWeb App load model mới"]

    classDef pass fill:#2ecc71,color:white;
    classDef fail fill:#e74c3c,color:white;
    class DONE pass;
    class FAIL fail;
```

### Metrics giải thích

| Metric | Công thức | Ý nghĩa |
|---|---|---|
| **LPIPS** | VGG features distance (crop vùng tóc) | Đo sự khác biệt thị giác giữa ảnh sinh ra và ảnh gốc (≤0.20 = tốt) |
| **Masked PSNR** | 10×log₁₀(1/MSE) chỉ trong mask | Đo chất lượng pixel vùng tóc (≥25 = tốt) |

---

## VI. Cấu Trúc File Hệ Thống

```
backend/training/
├── run_training_pipeline.sh      ← Script chạy toàn bộ 4 stages
├── prepare_dataset_deephair.py   ← Stage 0: Tạo dataset
├── train_stage2.py               ← Stage 2: Training UNet chính (best model tracking)
├── evaluate.py                   ← Metrics (LPIPS, PSNR)
├── export_model.py               ← Stage 3: Validate + Deploy (ưu tiên best)
├── test_cli.py                   ← 🆕 Test inference từ CLI (--random)
├── training_face.py              ← Face processing pipeline
├── models/
│   ├── texture_encoder.py        ← Stage 1: ResNet50 Texture Encoder
│   ├── stage2_unet.py            ← UNet 9-channel + IP-Adapter Injector
│   └── losses.py                 ← Loss functions (SupCon, MaskAware, Identity, Texture)
├── data_processing/
│   ├── mapping_dict.json         ← Bảng dịch Hàn → Anh (K-Hairstyle)
│   └── normalize_khairstyle.py   ← Chuẩn hóa dataset
├── processed/                    ← Output của Stage 0 (tự tạo khi chạy)
│   ├── bald_images/              (89,509 ảnh)
│   ├── hair_only_images/         (89,509 ảnh)
│   ├── hair_patches/             (38,192 ảnh)
│   ├── style_vectors/            (89,509 ảnh)
│   ├── identity_embeddings/      (89,509 file .npy)
│   ├── prompt_embeddings/        ← Cache CLIP embeddings (.pt)
│   └── metadata.jsonl            (29MB)
├── checkpoints/                  ← Weights lưu trong quá trình training
│   ├── stage1_step_*.safetensors (46 checkpoints — Stage 1 xong)
│   ├── stage2_step_*.safetensors (mỗi 500 steps)
│   ├── stage2_epoch_*.safetensors (mỗi epoch)
│   ├── deep_hair_v1_best.safetensors  ← ⭐ MODEL TỐT NHẤT
│   └── deep_hair_v1_latest.safetensors
└── results/                      ← Output của test_cli.py
    ├── output_*.png
    └── debug/                    (comparison, mask, bald, target, reference)
```

---

## VII. Cách Chạy

```bash
# Trên WSL (Linux)
cd /mnt/c/Users/Admin/Desktop/TryHairStyle
source venv_wsl/bin/activate

# Chạy toàn bộ pipeline tự động (4 stages)
bash backend/training/run_training_pipeline.sh

# Hoặc chạy từng stage riêng:
python backend/training/prepare_dataset_deephair.py   # Stage 0
python backend/training/models/texture_encoder.py      # Stage 1
python backend/training/train_stage2.py                # Stage 2
python backend/training/export_model.py                # Stage 3
```

### Test Inference (sau khi train)

```bash
# Random ảnh từ FFHQ (target) + K-Hairstyle (reference):
python backend/training/test_cli.py --random

# Chỉ định ảnh cụ thể:
python backend/training/test_cli.py --target face.jpg --reference hair.jpg

# Tuỳ chỉnh:
python backend/training/test_cli.py --random \
    --prompt "curly blonde hair" --steps 50 \
    --checkpoint backend/training/checkpoints/stage2_epoch_10.safetensors
```

Output lưu tại `backend/training/results/` kèm ảnh debug (comparison 4 cột: Target | Reference | Bald | Result).

### Yêu cầu trước khi chạy

| Yêu cầu | Đường dẫn |
|---|---|
| Dataset K-Hairstyle (images) | `backend/data/dataset/khairstyle/training/images/` |
| Dataset K-Hairstyle (labels) | `backend/data/dataset/khairstyle/training/labels/` |
| Dataset FFHQ (cho test_cli) | `backend/data/dataset/ffhq/` (~26,000 ảnh) |
| SDXL Inpainting Model | `backend/models/stable-diffusion/sd_xl_inpainting/` |
| GPU VRAM | ≥ 12 GB (RTX 3060 trở lên, đã tối ưu) |

---

## VIII. Mối Quan Hệ Training ↔ Production

```mermaid
flowchart LR
    subgraph TRAINING["🎓 Training Pipeline"]
        direction TB
        T0["prepare_dataset"] --> T1["texture_encoder"]
        T1 --> T2["train_stage2"]
        T2 --> T3["export_model"]
        T3 --> TEST["test_cli.py\n(Random FFHQ + K-Hairstyle)"]
    end

    subgraph PRODUCTION["🚀 Production Pipeline"]
        direction TB
        P1["User Upload\nTarget + Reference"]
        P2["SegFormer\nFace Mask"]
        P3["InsightFace\nID Embedding"]
        P4["SDXL UNet\n+ IP-Adapter\n+ ControlNet"]
        P5["VAE Decode\n→ Output Image"]
        P1 --> P2 --> P4
        P1 --> P3 --> P4
        P4 --> P5
    end

    T3 -->|"deep_hair_v1_best.safetensors\n(Model tốt nhất)"| P4

    classDef training fill:#74b9ff,color:white,stroke:#0984e3;
    classDef prod fill:#55efc4,stroke:#00b894;
    classDef test fill:#fdcb6e,stroke:#f39c12;
    class T0,T1,T2,T3 training;
    class P1,P2,P3,P4,P5 prod;
    class TEST test;
```

Model sau khi train xong, `export_model.py` ưu tiên file `deep_hair_v1_best.safetensors` (model có avg loss thấp nhất) và copy vào `backend/training/models/`. Web App (FastAPI) sẽ load weights mới khi khởi động lại server.
