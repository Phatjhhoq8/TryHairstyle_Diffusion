# KẾ HOẠCH HUẤN LUYỆN: DEEP HAIR TEXTURE & INPAINTING MODEL

Tài liệu này mô tả chi tiết quy trình, kiến trúc và chiến lược huấn luyện mô hình sinh tóc (Hair Inpainting) kết hợp hiểu sâu về cấu trúc hạt (Texture) dành cho hệ thống Virtual Hair Try-on.

Nguyên tắc cốt lõi: **Không can thiệp vào pipeline `training_face` hiện hành. Kế thừa ID Embedding, Face Bounding Box và Hair Mask từ `training_face`.**

---

## 1. DATASET PROCESSING (CHUẨN BỊ DỮ LIỆU)

Để mô hình phân biệt được cấu trúc tóc (Straight, Wavy, Density...) độc lập với khuôn mặt, chúng ta cần chia cắt hình ảnh từ dataset K-Hairstyle thành các thành phần cụ thể. Scripts sinh dữ liệu sẽ ghi vào `backend/training/processed/`.

### Cấu trúc thư mục đầu ra
```text
backend/training/processed/
├── bald_images/          # Ảnh mất tóc (Input gốc + mask để xóa tóc, fill bằng màu xám/blur)
├── hair_only_images/     # Ảnh chỉ chứa tóc (Nền đen/trong suốt) để tập trung học texture tổng thể
├── hair_patches/         # Các mảnh cắt nhỏ (64x64, 128x128) từ vùng tóc cực cận để học chi tiết (độ bóng, nếp uốn)
├── style_vectors/        # Vector embedding đại diện cho kiểu dáng (trích xuất qua CLIP / Hair-Classifier)
└── identity_embeddings/  # ID vector (Từ training_face pipeline)
```

### Tiền xử lý (Preprocessing Pipeline)
1. **Tạo `bald_images`**: Dùng `hair_mask` + thuật toán inpainting truyền thống (Navier-Stokes/Telea) hoặc làm mờ mạnh (Gaussian Blur) lên ảnh gốc để che mờ hoàn toàn vùng tóc cũ, tránh model bị "rò rỉ" (data leakage) hình dạng tóc nguyên bản.
2. **Tạo `hair_only` & `hair_patches`**: Dùng `hair_mask` cắt lấy vùng có tóc. Sau đó crop ngẫu nhiên các patch nhỏ nằm hoàn toàn trong vùng mask để tạo ngân hàng học texture.
3. **Đồng bộ Resolution**: Cố định mọi ảnh học ở **512x512** (giữ chi tiết tốt, vừa đủ cho VRAM 24GB). Các patches là 128x128.

---

## 2. CHIẾN LƯỢC HUẤN LUYỆN (TRAINING STRATEGY)

Thay vì huấn luyện model end-to-end (dễ rủi ro, hội tụ chậm, mô hình hay lười và chỉ học "shape" chứ không học "texture"), chúng ta chia làm **2 Stage**.

### Quyết định kiến trúc cơ sở:
Sử dụng mô hình **Diffusion** với **Inpainting UNet** kết hợp **IP-Adapter (cho Identity)** và **LoRA (cho Texture)**.

### Giải thích 2 Stage:
* **Stage 1 (Texture & Style Deep Learning)**:
  * **Mục tiêu:** Ép mô hình học vật liệu (material) của tóc (mật độ, độ bóng, chiều tóc).
  * **Cách làm:** Chỉ train mô hình auto-encoder sinh lại `hair_only_images` và `hair_patches` từ nhiễu.
  * **Ưu điểm:** Loại bỏ hoàn toàn nhiễu từ khuôn mặt và quần áo. Model học cách làm cho từng sợi tóc chân thực nhất có thể.

* **Stage 2 (Conditioned Inpainting & Blending)**:
  * **Mục tiêu:** Gắn khối tóc (đã biết cách sinh đẹp từ Stage 1) vào chuẩn xác vùng mask trên đầu người, khớp khuôn mặt, khớp ánh sáng.
  * **Cách làm:** Input 9-channel (4 latent + 1 mask + 4 masked_latent). Khóa Identity. Mô hình sinh tóc lấp đầy vùng mask.
  * **Ưu điểm:** Giảm tải nhiệm vụ cho UNet. Stage 2 chỉ lo việc "xếp đặt", Stage 1 đã lo việc "vẽ chi tiết".

---

## 3. KIẾN TRÚC MODEL

Đây là một biến thể của Stable Diffusion Inpainting có kết hợp inject embeddings.

### Sơ đồ Logic (Forward Pass)
```text
[Bald Image] --(VAE Encode)--> (4) Bald Latent ------------+
[Hair Mask]  --(Resize)------> (1) Mask Latent ------------+---> Concatenate (9 channels) ---> [Inpainting UNet]
[Noised Latent] -------------------------------------------+                                        |
                                                                                                    |
[Style Embedding] --------(Cross-Attention)---------------------------------------------------------+
                                                                                                    |
[Identity Embedding] -----(IP-Adapter Cross-Attention) ---> [Face Region Masked] -------------------+
(Sinh từ training_face)                                                                             |
                                                                                                    V
                                                                                          [Denoised Latent]
                                                                                                    |
                                                                                                (VAE Decode)
                                                                                                    |
                                                                                          [Final Hairstyle Image]
```

### Trả lời các thiết kế cốt lõi:
1. **Style embedding inject vào đâu?**
   Đưa vào các **Cross-Attention layers** tiêu chuẩn của UNet (thay thế cho Text Prompt hoặc Add trực tiếp vào Text Embedding). Nó dẫn đường cho hình khối tổng quan (ví dụ: cúp, xõa, xoăn).
2. **Identity embedding inject vào đâu?**
   Đưa vào mạng thông qua module **IP-Adapter** (Image Prompt-Adapter). Module này thêm một nhánh Cross-Attention song song đặc thù chỉ cho Image Prompt. Để bảo vệ không bị biến dạng phần tóc đang sinh, ID Embedding attention map bị nhân với `Face_Mask` (chỉ tác động vùng mặt).
3. **Cách khóa gradient vùng face (Giữ nguyên mặt)?**
   Sử dụng cơ chế **Masked Loss** và **Latent Blending**:
   * *Trong lúc Forward*: Phần mặt của ảnh sinh ra bị cưỡng ép (replace) bằng phần mặt của latent gốc theo mỗi step khử nhiễu.
   * *Trong lúc Backward (Tính Loss)*: Gradient sinh ra ở vùng `Face_Mask` được gán = 0 hoặc nhân Loss với `(1 - Face_Mask)`. Loss chỉ lan truyền cập nhật trọng số nhờ đạo hàm trên vùng tóc.
4. **Cách đảm bảo model chỉ vẽ lại hair region?**
   Đây là bản chất của kiến trúc **Inpainting UNet (9-channels)**. Kênh Mask Input quy định chặt chẽ ranh giới mà UNet được phép biến đổi latent.

---

## 4. LOSS FUNCTION

Total Loss là sự kết hợp có trọng số của các hàm mất mát sau:

1. **Mask-aware L2/Reconstruction Loss**: `$ \mathcal{L}_{recon} = || M_{hair} \odot (\epsilon - \epsilon_\theta(z_t, t, c)) ||_2^2 $`
   * *Vai trò*: Hàm cơ bản của Diffusion, ép model phục hồi lại đúng vùng tóc thực tế. Mask để bỏ qua nhiễu từ background.
2. **Identity Loss (Cosine Similarity)**: `$ \mathcal{L}_{id} = 1 - \cos(\text{ArcFace}(I_{out}), \text{ArcFace}(I_{orig})) $`
   * *Vai trò*: Phạt hình phạt rất nặng nếu ảnh xuất ra bị méo mặt hoặc đổi người (kể cả khi đã khóa gradient vùng lõi, ID loss giữ cho vùng trán, tóc mai khớp với gương mặt).
3. **Perceptual Loss (LPIPS)**:
   * *Vai trò*: Đánh giá bằng mạng VGG. Giúp chất lượng tóc sinh ra không bị mờ (nhược điểm L1/L2 loss), sợi tóc có độ nét (sharpness) giống ảnh thật.
4. **Texture Consistency Loss (Patch-based Gram Matrix)**:
   * *Vai trò*: Tính ma trận Gram trên các feature maps. Ép các vùng sáng tối, nếp uốn cục bộ của tóc sinh ra phải giống kiểu phân bổ ánh sáng của tóc chuyên gia (giải quyết được độ bóng - shine và direction).
5. **Style Classification Loss (Auxiliary)**:
   * *Vai trò*: Chạy ảnh đầu ra qua mô hình phân loại (VD: Bangs/No Bangs, Wavy/Straight) và tính Cross-Entropy. Ép model bắt buộc sinh đúng kiểu yêu cầu.

---

## 5. CƠ CHẾ HỌC TEXTURE SÂU

Để model "hiểu" độ phồng, mật độ tóc, thiết kế 1 nhánh phụ trợ (Auxiliary Branch) chỉ active trong lúc train:

* **Hair Patch Encoder (Contrastive Learning)**:
  Sử dụng tập `hair_patches`. Xây dựng một Encoder nhỏ (ResNet/ViT) nhúng patch thành vector. Dùng hàm loss InfoNCE để kéo gần khoảng cách cosine giữa các patch tóc của *cùng một người/cùng độ xoăn*, đẩy xa patch của các loại tóc khác nhau.
* **Tích hợp**: Trọng số của Encoder này sau đó được freeze và đóng vai trò làm module trích xuất `Style_Vectors` siêu việt, cấp thẳng vào bộ Cross-Attention của UNet. Nhờ đó UNet hiểu rất rõ: *"À, embedding này ứng với cấu trúc sợi dệt chặt, độ bóng cao"*.

---

## 6. TRAINING PLAN (Cấu hình hệ thống 24GB VRAM)

Cấu hình cho RTX 3060/4090/A5000 24GB.

**Môi trường & Siêu tham số:**
* Học máy: `PyTorch 2.x` + `xformers` (hoặc `F.scaled_dot_product_attention`) tiết kiệm tối đa bộ nhớ.
* **Resolution**: `512x512`. Khuyến cáo không dùng 768x768 lúc đầu vì Batch Size sẽ bị tụt cực đoan gây nhiễu gradient.
* **Batch Size**: `4` (VRAM dùng cỡ 18-21GB). Dùng `gradient_accumulation_steps = 4` để có **effective batch size = 16**.
* **Epochs**:
  * Stage 1 (Texture & Style): `50 - 80 epochs` (Vì task patch-learning hẹp, dễ hội tụ).
  * Stage 2 (Inpainting): `100 - 150 epochs`.
* **Learning Rate**:
  * `1e-4` cho các tầng Cross-Attention/ControlNet mới thêm.
  * `1e-5` nếu fine-tune các block sâu của UNet.
* **Scheduler**: `Cosine Annealing with Warmup` (Warmup 5% tổng steps đầu).
* **Precision**: `fp16` hoặc `bf16` (bắt buộc, tiết kiệm 50% VRAM).
* **Gradient Checkpointing**: `Bật (True)` trên toàn bộ UNet. Trao đổi tốc độ (-20%) để lấy thêm 30% VRAM.
* **Optimizer**: `AdamW` (hoặc `8-bit Adam` từ thư viện `bitsandbytes` nếu VRAM bị nghẽn).

---

## 7. EVALUATION (ĐÁNH GIÁ MÔ HÌNH)

Pipeline test tự động sử dụng tập Validation Testset (5% chưa bao giờ gặp trong lúc train). Các metric phải được log lại:

1. **Identity Similarity Score (Bảo toàn ID)**: Đo Cosine Similarity của AdaFace/InsightFace giữa ảnh gốc và ảnh Output. Yêu cầu `>= 0.70`.
2. **LPIPS (Hair Region)**: Cắt riêng vùng tóc của kết quả và Ground Truth để đo. Càng thấp càng sinh động, mượt mà (Yêu cầu đánh giá theo từng epoch).
3. **FID (Fréchet Inception Distance)**: Đo ranh giới phân phối ảnh thực và ảo ở vùng biên tóc-da.
4. **Texture Classifier Accuracy**: Đo xem tóc sinh ra đi qua mạng test style có chuẩn không (VD: Hỏi tóc xoăn, mạng có output xoăn không).
5. **Visual Realism (Đánh giá người)**: Quan sát vùng giao cắt: Trán, tai, cổ rễ tóc. Nếu có viền mờ (halo effect) -> Cần tinh chỉnh Mask-aware L2 loss.

---

## 8. QUY TRÌNH THAY THẾ PRODUCTION

Khi hoàn tất huấn luyện:

1. **Export Weights**: Model sẽ xuất 2 thành phần chính đặt vào `backend/training/checkpoints/`:
   * `unet_hair_inpainting.safetensors` (Hoặc LoRA weights siêu nhẹ).
   * `style_patch_encoder.pt` .
2. **A/B Testing**:
   * Viết script độc lập load folder 100 ảnh khách hàng thực tế (ảnh bald + mặt).
   * Chạy hệ thống qua Model Cũ và Model Mới.
   * Căn cứ vào điểm LPIPS + ID Similarity, nếu model Mới vượt > 5% hiệu năng và mắt thường thấy hết artifact, duyệt.
3. **Deploy thay thế**:
   * Cập nhật file config/registry của server.
   * Model không phá vỡ pipeline cũ do đầu vào hoàn toàn tận dụng output của hệ thống `training_face` (Embedding, Mask, Bounding Box còn y nguyên). Khởi động lại `Celery worker` và API backend.
