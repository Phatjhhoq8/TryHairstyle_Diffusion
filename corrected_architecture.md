# Xây dựng hệ thống thay đổi kiểu tóc bằng Diffusion

**Stable Diffusion Inpainting**
Stable Diffusion Inpainting là mô hình sinh ảnh dựa trên diffusion, học cách đảo ngược quá trình thêm nhiễu (denoising) trong không gian tiềm ẩn (latent space), dưới sự dẫn hướng ngữ nghĩa của CLIP.
Trong quá trình này, U-Net đóng vai trò trung tâm, thực hiện việc dự đoán và loại bỏ nhiễu theo từng bước thời gian.
CLIP (Contrastive Language–Image Pretraining) là mô hình liên kết ảnh và văn bản. CLIP học cách ánh xạ ảnh và văn bản vào cùng một không gian embedding, sao cho các cặp ảnh–văn bản có cùng ngữ nghĩa sẽ có vector gần nhau.
Stable Diffusion không học ảnh trực tiếp ở không gian pixel, mà học phân phối xác suất của ảnh thông qua quá trình ngược của việc phá hủy ảnh bằng nhiễu Gaussian. Điều này cho phép mô hình sinh ảnh mới bằng cách dần dần khôi phục cấu trúc ảnh từ nhiễu thuần.
Cơ chế attention trong mô hình tự động học các vùng quan trọng trong ảnh thông qua thống kê lặp lại trên dữ liệu lớn, mà không cần nhãn vùng.

Mô hình:
- Không sao chép ảnh từ dataset
- Không ghép ảnh
- Không ghi nhớ ảnh cụ thể
- Mà học quy luật thống kê trong không gian tiềm ẩn

**VAE - Variational AutoEncoder** là mô hình học cách nén ảnh rồi giải nén lại sao cho ảnh nén vẫn giữ đủ thông tin quan trọng nhưng không giữ từng pixel cụ thể. AE mã hóa ảnh thành điểm, VAE mã hóa ảnh thành phân phối -> tạo không gian tiềm ẩn liên tục, nên VAE sinh ảnh được còn AE thì không. Không gian tiềm ẩn (latent space) là không gian đặc trưng trừu tượng, nơi mô hình học cấu trúc, màu sắc và ngữ nghĩa, thay vì từng pixel.

**U-net**
Học noise nào cần được loại bỏ tại bước t nếu muốn sinh ra ảnh đúng mô tả. U-net nhận noise hiện tại, Prompt embedding, depth map, mask tóc -> u-net trả về nhiễu cần bỏ -> lặp lại đến khi ảnh rõ.
Mục tiêu: Vẽ lại chỉ vùng tóc, giữ nguyên khuôn mặt, ánh sáng, background.
Cách làm: Inpainting với mask tóc (sử dụng **SegFormer**).

**ControlNet**
ControlNet là mạng phụ gắn song song vào U-Net của SD, nhằm ép mô hình sinh ảnh tuân theo một tín hiệu cấu trúc có sẵn. Quá trình sinh ảnh được kiểm soát chặt chẽ hơn về hình học và bố cục.
*(Trong hệ thống này, **ControlNet Depth SDXL** được sử dụng làm phương pháp dẫn hướng chính).*

- **ControlNet Depth (Được dùng chính thức)**
Sử dụng bản đồ độ sâu làm điều kiện dẫn hướng, mô tả quan hệ gần - xa trong không gian 3D.
Phù hợp để: Giữ hình dạng đầu, duy trì cấu trúc không gian tự nhiên, đồng thời giúp tóc mới có thể phồng/xẹp khác tóc cũ một cách hợp lý (với weight điều chỉnh).
- **ControlNet Normal (Lý thuyết)**
Sử dụng bản đồ pháp tuyến bề mặt để mô tả hướng bề mặt tại mỗi điểm. Giữ độ cong của mặt tinh tế hơn so với depth.
- **ControlNet Canny (Lý thuyết / Tùy chọn)**
Sử dụng bản đồ biên bằng thuật toán Canny. Dùng để giữ outline tổng thể đường viền.

**IP-Adapter**
IP-Adapter là cơ chế cho phép SD tiếp nhận ảnh tham chiếu nhằm ép mô hình sinh ảnh theo các đặc trưng mong muốn.

**InstantID**
Giữ nguyên danh tính khuôn mặt. Mô hình khai thác embedding khuôn mặt từ **InsightFace** (buffalo_l), đảm bảo người trong ảnh sinh ra là cùng một cá nhân, bất kể tư thế hay biểu cảm.

**IP-Adapter Plus / Style**
Cho phép sao chép phong cách (hình dạng, kết cấu, màu sắc) từ ảnh tham chiếu. Không giữ danh tính khuôn mặt mẫu mà tập trung tái hiện đặc trưng thị giác (kiểu tóc) sang ảnh đích.

**SegFormer (Thay thế cho BiSeNet)**
Mô hình phân đoạn ngữ nghĩa mạnh mẽ dựa trên Transformer (jonathandinu/face-parsing). Được hệ thống sử dụng chính thức thay cho BiSeNet vì độ chính xác phân mảnh (19 classes) cao hơn nhiều, đặc biệt tốt trên các ảnh góc nghiêng (profile views). Giúp tách các mask: face, hair, neck, background một cách hoàn hảo.

**Dataset**
- FFHQ → học khuôn mặt + ánh sáng thật
- K-Hairstyle → học đa dạng kiểu tóc
- Kết hợp → học ghép tóc vào mặt thật

---

## Xây dựng hệ thống
### I. Mục tiêu hệ thống
**Đầu vào:**
- Ảnh Target (người dùng)
- Ảnh Reference (kiểu tóc mẫu)

**Đầu ra:**
- Ảnh người dùng với kiểu tóc của ảnh mẫu. Giữ nguyên: Identity khuôn mặt, Hướng nhìn, Ánh sáng, Background.

### II. Tech Stack & Môi trường
- **Base Model:** SDXL (Stable Diffusion XL)
- **Face Analysis:** InsightFace (buffalo_l) → trích xuất embedding ID
- **Segmentation:** SegFormer (jonathandinu/face-parsing)
- **Profile Alignment (>45°):** 3DDFA_V2 (căn chỉnh 3D góc nghiêng)
- **Adapter (Identity):** InstantID
- **Adapter (Style):** IP-Adapter Plus (Style / Reference)
- **ControlNet:** ControlNet Depth SDXL

### III. Preprocessing
1. **Align khuôn mặt:** Theo mắt (góc thẳng <45°) hoặc Dùng 3DDFA_V2 (góc nghiêng lệch >45°).
2. Chuẩn hóa kích thước (1024x1024 cho SDXL).
3. **Tách mask (SegFormer):** hair, face, neck, background.

### IV. Kiến trúc mô hình (Dual-Conditioning)
1. **Nhánh giữ Identity:** InsightFace → id_embedding → InstantID. Kết hợp ControlNet Depth để giữ hình khối đầu, cổ, vai.
2. **Nhánh truyền Style:** IP-Adapter Plus (Reference Image) → Học kiểu tóc, độ phồng, texture.
3. **Inpainting Mask (Dynamic Expansion):** 
   - Mask ban đầu = vùng tóc gốc (từ SegFormer).
   - Nếu User có **Đầu Trọc**, sử dụng lưới 3D từ **3DDFA_V2** để ngoại suy một "Nón Sọ" (Scalp Mask) bo tròn trên đỉnh trán theo đúng góc nghiêng khuôn mặt.
   - Khi tóc mẫu lớn hơn tóc User (Tóc tém → Tóc dài/phồng), hệ thống tự động **Dilate (Kéo giãn)** vùng mask theo Aspect Ratio của tóc mẫu (phình ngang hoặc chảy dọc xuống gáy).
   - Mọi vùng giãn nở đều bị loại trừ vùng da mặt (`face_mask`) để tạo "khiên bảo vệ", không cho tóc mọc đè lên mắt, mũi, miệng.

### V. Chiến lược Huấn luyện
Tập trung train/finetune Adapter:
- **Giai đoạn 1 (IP-Adapter cho Tóc):** Data K-Hairstyle. Input ảnh tóc đã mask mặt. Reconstruct ảnh để model hiểu sâu về texture tóc.
- **Giai đoạn 2 (Inpainting Identity):** Data FFHQ. Xóa vùng tóc, truyền Embedding mặt + tóc vào để model vẽ lại. Dùng L1/L2 Loss, ID Loss, Style Loss.

### VI. Inference Pipeline
1. **Preprocessing:** Target Image -> SegFormer -> Lấy Mask Tóc cũ + Mask khuôn mặt (`face_mask`). Lấy Depth Map, Lấy id_embedding. Lấy lưới 3D (`vertices3D`) từ 3DDFA_V2.
2. **Encoding:** Reference Image -> CLIP Image Encoder -> style_embedding. Đo đạc Aspect Ratio của mask tóc mẫu.
3. **Dynamic Masking:** Điều chỉnh mask tóc User thành Mask Đầu Vào cho SDXL:
   - User Trọc → Dựng nón 3D Scalp từ lưới 3D.
   - Tóc rỗng → Phình mask theo tỷ lệ tóc mẫu.
   - Khóa chặt vùng da mặt (Face Protection).
4. **Generation (SDXL):** 
   - Prompt miêu tả tóc, Negative prompt tránh lỗi.
   - Nhúng InstantID (w=0.7), IP-Adapter Plus (w=1.0), ControlNet Depth (w=0.6).
   - Inpainting vào vùng Mask Động vừa tạo.
5. **Post-processing:** Quét CodeFormer/GFPGAN nét mặt, blend vùng rìa tóc.

### VII. Triển khai
- **Backend:** FastAPI, dùng Celery + Redis cho hàng đợi (AI tốn vài giây xử lý, tránh timeout HTTP).
- **Frontend:** React/Vue (có tool vẽ mask thủ công để sửa lỗi AI).

---

*(Phần đánh giá nghiên cứu HairFastGAN vs Mask R-CNN và bảng so sánh nhanh giữ nguyên như ý tưởng gốc của người dùng vì thuộc dạng so sánh học thuật)*

---

# Hướng dẫn chi tiết: Cách thức hoạt động của các modules cốt lõi trong hệ thống TryHairStyle

Tài liệu này đi sâu vào logic code của các module quan trọng nhất trong hệ thống, giúp bạn hiểu rõ từng class và hàm hoạt động ra sao ở cấp độ mã nguồn.

## 1. Module Suy luận (Inference Pipeline) - [diffusion.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/diffusion.py)

File [backend/app/services/diffusion.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/diffusion.py) chứa class [HairDiffusionService](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/diffusion.py#26-511), là trái tim của quá trình sinh ảnh lúc người dùng sử dụng thực tế.

### Các hàm quan trọng:
*   [__init__()](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/adaface_ir.py#62-110) và [_init_pipeline()](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/diffusion.py#41-54): 
    - Khởi tạo Service và xác định cấu hình phần cứng (FP16 nếu có CUDA, FP32 nếu chạy CPU). 
    - Kiểm tra xem hệ thống có model Custom UNet 9-channel đã train không. Nếu có, nó chuyển sang [_load_custom_pipeline()](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/diffusion.py#55-167). Nếu không, nó fallback về mô hình SDXL tiêu chuẩn [_load_sdxl_pipeline()](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/diffusion.py#168-239).
*   [_load_custom_pipeline(checkpoint_path)](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/diffusion.py#55-167): 
    - Tải **Custom 9-channel UNet** (đã được fine-tune) thay vì SDXL UNet 4-channel thông thường. Tại sao lại là 9-channels? Vì UNet tiêu chuẩn nhận 4 kênh latent (từ VAE), nhưng mô hình inpainting cần thêm 4 kênh của ảnh gốc (masked) và 1 kênh mask. 
    - Tải **CrossAttentionInjector**: Một mạng siêu nhỏ được nạp vào U-Net để tiêm (inject) các embeddings về Identity (Khuôn mặt) và Style (Kiểu tóc) vào quá trình sinh ảnh một cách trực tiếp mà không phải đi qua CLIP Vision.
    - Cố gắng tải **HairTextureEncoder** (từ [texture_encoder.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/texture_encoder.py)) để lấy vectơ 2048-chiều mô tả chính xác hình thái, độ xoăn, đặc tính của tóc.
*   [_generate_custom(...)](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/diffusion.py#297-470): 
    - Đây là hàm thực thi việc tạo ảnh theo phong cách custom.
    - Ban đầu, nó lấy `id_embed` (Identity) thông qua thư viện InsightFace (đóng gói trong [embedder.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/embedder.py)).
    - Nó mã hóa ảnh đầu vào qua VAE (`vae.encode`) để đưa về không gian Latent. Đồng thời xử lý Mask để che đi vùng tóc cũ.
    - Sau đó, tạo Classifier-Free Guidance (CFG) bằng cách chạy song song một luồng có điều kiện (Prompt + Injector(ID + Hair)) và một luồng không điều kiện (Negative Prompt). Sự khác biệt giữa 2 luồng này được nhân với `guidance_scale` để ép ảnh sinh ra sát với văn bản và ảnh tham chiếu nhất có thể.
    - Cuối cùng, hàm gọi `vae.decode()` từ latent ra ảnh pixel, và ghép lại vùng mask với background gốc bằng kỹ thuật chập mờ (GaussianBlur) để không lộ viền ghép.

---

## 2. Module Tiền Xử Lý (Visualization & Segmentation) - [visualizer.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/visualizer.py)

Class [TrainingVisualizer](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/visualizer.py#45-639) chuyên xử lý tạo mask, tìm kiếm vùng mặt và tóc.

### Các hàm quan trọng:
*   [_loadSegFormer()](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/visualizer.py#63-89):
    - Khởi tạo mô hình mạng nơ-ron chuyên dụng tên là **SegFormer** (`jonathandinu/face-parsing` - bản model 19 class). Đây là Transformer có khả năng nhận diện rất chi tiết các vùng (mắt, mũi, môi, tóc, cổ, nền...).
*   [_runSegFormer()](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/visualizer.py#148-185) và [_runSegFormerForFace(image, bbox)](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/visualizer.py#186-247):
    - Khi có một ảnh chụp rộng, thay vì quét toàn bộ ảnh để lấy mask (dễ gây lỗi nhận diện nhầm tóc ở xa), hệ thống sẽ tính khung bounding box ([bbox](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/evaluate.py#33-59)) bao quanh khuôn mặt, sau đó crop vùng đó mở rộng ra một chút rồi đưa cho SegFormer.
    - Kết quả trả ra được upscale về kích thước 512x512. Các nhãn gồm: 13, 14 (tóc, nón) được gộp chung thành biến `HAIR_CLASSES`, còn mắt, mũi, da được gộp thành `FACE_CLASSES`.
*   [_filterParsingForFace(parsing, targetBbox, allBboxes)](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/visualizer.py#248-370):
    - Đây là hàm hậu xử lý cực kỳ thông minh. Giả sử trong khung hình có 2 người đứng sát nhau, tóc người này che khuất người kia. Hàm này dùng thuật toán `connectedComponents` để tìm các cụm (blobs) tóc. Cụm nào nằm xa tâm khuôn mặt mục tiêu sẽ bị loại bỏ (chuyển thành background 0). Nó tạo ra "Exclusion Zones" chặn không cho nhận diện nhầm biểu cảm của người khác.
*   [_enhanceFaceMaskWith3D(parsing, vertices3D)](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/visualizer.py#582-635):
    - *Công nghệ 3DDFA_V2 được áp dụng tại đây.* Khi người dùng quay mặt góc máy profile ngang 90 độ, SegFormer thường bỏ sót phân nửa khuôn mặt bị khuất. Hệ thống đưa điểm lưới 3D (3D mesh vertices) vào ảnh, tìm các đường viền bao quanh lưới (Convex Hull), bù lại vào nhãn `FACE` để tránh đắp tóc lên phần má.

---

## 2.5 Tiền Xử Lý Dữ Liệu Training (Pre-compute Masks) - [precompute_face_masks.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/precompute_face_masks.py)
*   **Mục đích:** Để AI (Stage 2) có thể học cách "vẽ tóc vào vùng mask lớn nhưng không chèn lên mặt", ta cần chuẩn bị trước các mask bảo vệ khuôn mặt.
*   **Hoạt động Offline:** Script chạy quét toàn bộ dataset (18 chunks) một lần duy nhất.
    *   Sử dụng **SegFormer** để tìm vùng da mặt chính xác.
    *   Tích hợp kỹ thuật **Dự phòng 3DDFA V2 (Convex Hull)** khi SegFormer thất bại (ở các góc nghiêng).
    *   Lưu các `face_mask.png` này vào cùng cụm dữ liệu. Lớp khiên này sẽ được Stage 2 lấy ra dùng mỗi khi nó cần mô phỏng quá trình "nới rộng mask giả lập" trong quá trình training.

---

## 2.8 Module Sinh Mask Động (Dynamic Masking) - [mask.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/mask.py)
Class `MaskService` là "Bộ Não Mở Rộng" chịu trách nhiệm biến đổi mask của người dùng thành vùng không gian lý tưởng nhất cho AI sinh ảnh.
*   [expand_hair_mask()](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/mask.py): Hàm trung tâm, nhận 3 ảnh (Tóc_User, Tóc_Mẫu, Da_User) và Lưới_3D từ 3DDFA.
    *   **Ông Trọc (`area < 100`)**: Kích hoạt `_create_scalp_mask_from_3d()`. Nó lấy lưới 3D, từ vùng trán nó phóng vector lên đỉnh đầu để tạo ra một "Nón Sọ" (Scalp Mask) nghiêng chuẩn theo góc mặt.
    *   **Tóc Tém lùi Tóc Dài (`ref_area > user_area * 1.5`)**: Đo Aspect Ratio của rập tóc mẫu. Nếu tóc mẫu là dáng rũ, hàm sử dụng *Kernel Dọc* để kéo tuột mask Tóc_User xuống tận cổ/lưng. Nếu tóc mẫu phồng bồng bềnh, dùng *Kernel Ngang* để thổi phình 2 bên.
    *   **Nguyên tắc Bất Bạo Phạm**: Cuối cùng, vùng da mặt (`face_buffer`) luôn được chèn lên cắt lẹm mọi vùng do Mask giãn nở đụng chạm tới. Đảm bảo SDXL vẽ tóc dài nhưng không phủ nhận mắt mũi.

---

## 3. Module Huấn luyện (Training Stage 2) - [train_stage2.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/train_stage2.py)

Đây là tệp quy định cách chiếc AI được "học". 

### A. Dataset Loading ([HairInpaintingDataset](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/train_stage2.py#170-454))
*   [_load_sample()](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/train_stage2.py#371-454): 
    - Hàm tải 1 batch dữ liệu gồm: Ảnh nguyên bản (Ground truth), Ảnh Mask (vùng tóc đen, mặt trắng), ID Vector 512-chiều (từ AdaFace), Style Vector 2048-chiều (từ Texture Encoder), Text Embeddings. Đáng chú ý: Tải thêm `face_mask` được pre-compute từ trước.
    - Cấu hình chạy **Mask Augmentation (`_augment_mask`)**: Với xác suất 30%, hệ thống cố tình kéo giãn mask tóc thật ra to hơn. Sau khi bù trừ phần `face_mask`, hệ thống nạp vào hàm loss. Điều này giúp AI học kỹ năng *"Tự bịa thêm cọng tóc để lấp đầy vùng thừa"*, cực kỳ quan trọng cho Inpainting tóc phồng/dài.
    - Đáng chú ý là kỹ thuật **Lazy Loading** và **Disk Caching**: Để tiết kiệm VRAM trên Google Colab (vốn hay sập RAM), hệ thống tính toán trước (`pre-encode`) tất cả các CLIP Text Prompts và lưu ra ổ đĩa dưới dạng file `.pt`. Khi cần, nó mới bốc file từ đĩa thay vì bắt card đồ họa dịch text thành vector hàng nghìn lần.

### B. Huấn luyện thực tế ([Stage2Trainer](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/train_stage2.py#459-1908))
*   [__init__()](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/adaface_ir.py#62-110): Thiết lập **LoRA** (Low-Rank Adaptation) với hạng `r=16`. Việc này đóng băng 99% trọng số khổng lồ 2.6 tỷ tham số của SDXL, chỉ mở khóa huấn luyện ~0.3% thông qua LoRA. Trainer tải VAE và đẩy sang CPU để giải phóng bộ nhớ. Tích hợp `AdamW8bit` giúp giảm tải cho vRAM từ 16GB xuống chỉ còn khoảng ~8GB.
*   [train_step()](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/texture_encoder.py#220-253): Chu trình back-propagation trái tim của mô hình:
    1. **VAE Encode**: Lấy ảnh Ground Truth chuyển sang Latent [latents](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/train_stage2.py#631-644). Tạo ảnh khuyết vùng tóc `masked_latents`.
    2. **Thêm Denoising Noise**: Bỏ nhiễu Gaussian ngẫu nhiên vào [latents](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/train_stage2.py#631-644) theo timestep `t` bằng mảng Pytorch DDPMScheduler.
    3. **Forward**: Ném `noisy_latents` (nhiễu), `masked_latents` (ảnh đã khoét), `masks_latent`, và Encoder features vào UNet.
    4. **Loss**: 
       Hỗ trợ 3 hàm mất mát (Loss):
       - [MaskAwareLoss](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/losses.py#82-108): Hàm Loss quan trọng nhất. Yêu cầu U-Net loại bỏ chuẩn xác nhiễu Gaussian chỉ tại các điểm thuộc vùng Mask (tức là tóc), bỏ qua các điểm ngoài Mask (background, mặt).
       - *[Monitor Loss]* [IdentityCosineLoss](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/losses.py#109-129) & [TextureConsistencyLoss](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/losses.py#34-81) đo lường bằng VGG16/Inception. Mã hoá ảnh đầu ra no_grad cứ mỗi 50 step chỉ để vẽ biểu đồ log theo dõi việc giữ danh tính và kết cấu tóc mà không cộng dồn vào quá trình tính đạo hàm ngược, giúp tối giản tới cực bộ VRAM.

---

## 4. Module Hỗ Trợ Tiền & Hậu Xử Lý (Pre/Post-Processing)

*   **Face Detector ([face_detector.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/face_detector.py) - YOLOv8-Face)**: 
    - Khác với SegFormer dùng để tách pixel, YOLOv8-Face (mô hình siêu nhẹ `yolov8n-face.pt`) được dùng ở đầu phễu để khoanh vùng (bounding box) tất cả các khuôn mặt có trong ảnh.
    - Trong file [tasks.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/tasks.py), YOLO kiểm tra xem người dùng có nộp ảnh bị xoay lộn ngược hay không. Nếu không tìm thấy mặt, hệ thống sẽ tự động xoay ảnh (90, 180, 270 độ) cho đến khi bắt được khuôn mặt nhờ YOLO, giúp trải nghiệm người dùng không bị gián đoạn vì lỗi hướng ảnh.
*   **Hair Color Service ([hair_color_service.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/hair_color_service.py))**:
    - Đây là module đổi màu tóc siêu tốc bằng kỹ thuật **HSV Color Transfer**, hoạt động độc lập với Diffusion.
    - AI chuyển màu tóc RGB sang hệ màu Hue-Saturation-Value (HSV). Nó gán mã màu (Hue/Saturation) mới từ các preset (Ví dụ: Blonde, Auburn, Pink), nhưng **giữ nguyên hoàn toàn độ sáng (Value)** của ảnh gốc. Kết hợp với mask làm mờ viền (Gaussian Blur), tóc sẽ đổi màu rõ rệt nhưng vẫn giữ 100% nếp lượn sóng, độ bóng tự nhiên mà không cần chạy lại mô hình sinh ảnh nặng nề.
*   **Depth Estimator (`Intel/dpt-large`)**:
    - Dù ControlNet Depth SDXL phụ trách việc dẫn hướng không gian, nhưng để tạo ra bản đồ độ sâu (Depth Map) từ ảnh 2D RGB ban đầu, hệ thống sử dụng một mô hình Transformer chuyên biệt là `Intel/dpt-large` chạy ngầm trong [tasks.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/tasks.py).

---

## 5. Hàng Đợi Bất Đồng Bộ (Celery & Redis) - [tasks.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/tasks.py)

*   **Tại sao cần Celery?** Quá trình chạy Diffusion cho 1 bức ảnh mất từ 5-10 giây trên GPU RTX 3060. Nếu để API HTTP chờ sinh ảnh xong mới phản hồi, web sẽ bị treo (timeout) khi có nhiều người dùng.
*   **Cách hoạt động**: [tasks.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/tasks.py) chứa đối tượng `celery_app`. Khi người dùng bấm "Tạo ảnh", API chỉ gửi một thông điệp (Message) vào **Redis** và báo "Đang xử lý". Ở dưới nền (Background Worker), một tiến trình Celery đang chờ sẵn sẽ nhặt thông điệp đó ra, nạp ảnh vào VRAM, chạy lần lượt: Face Detector -> Masking -> Depth Map -> Diffusion -> Hair Colorization. Cuối cùng lưu kết quả ra thư mục `session_dir`. Đây là chuẩn công nghiệp để chịu tải lớn (Scalability).

---

## 6. Phân Tích Khuôn Mặt Sâu (Deep Face Analysis)

*   **AdaFace (IR-100 Backbone) - [adaface_ir.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/adaface_ir.py)**:
    - Đây là một mạng nhận diện khuôn mặt ResNet-100 được tinh chỉnh riêng. AdaFace cực kỳ xuất sắc trong việc nội suy các đặc trưng khuôn mặt từ ảnh chất lượng thấp hoặc bị che khuất mạnh (ví dụ: bị tóc mái che nửa mặt). 
    - Trước khi đưa vào mạng IR-100, hệ thống dùng thêm phân hệ **MTCNN** để dò đúng 5 điểm nhạy cảm (mắt/mũi/miệng) nhằm căn chỉnh (Align) và crop chặt khuôn mặt về đúng tỷ lệ vàng 112x112.
    - Đầu ra của nó là 1 vector 512-chiều mang đặc trưng không thể nhầm lẫn của người dùng, làm đầu vào cho InstantID giữ danh tính mạnh mẽ nhất.
*   **Pose Estimator (InsightFace 106 & 3DDFA) - [pose_estimator.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/pose_estimator.py)**:
    - Để AI biết chính xác người dùng đang nhìn thẳng hay cúi đầu, hệ thống chạy **InsightFace AntelopeV2** để dò tìm **106 điểm mốc (landmarks)** trên mặt (mắt, mũi, cong môi, viền hàm...).
    - Nếu InsightFace thất bại do ảnh quá nghiêng, hệ thống lập tức gọi **3DDFA V2** dự phòng để tính ra góc quay đầu 3 chiều: `Yaw` (lắc đầu), `Pitch` (gật đầu), `Roll` (nghiêng đầu).
*   **Reconstructor 3D - [reconstructor_3d.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/reconstructor_3d.py)**:
    - 3DDFA V2 không chỉ tìm góc quay mà còn dựng lại toàn bộ một lưới mặt nạ 3D (3D Mesh) áp sát gò má người dùng. 
    - Hệ thống dùng lưới 3D này để bù đắp vào những vùng da mặt mà SegFormer nhầm thành tóc hoặc background ở các góc khuất, ngăn chặn hiện tượng tóc mới "dính" vào má của người dùng.

---

## 7. Đóng Gói Tiền Xử Lý Dữ Liệu Huấn Luyện

*   **Auto-Pipeline đóng gói hàng loạt ([training_face.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/training_face.py))**: Quét qua toàn bộ ảnh thô trong dataset, kích hoạt dây chuyền 5 bước khép kín: YOLO Detect -> Pose Estimator -> Embedder -> 3DDFA -> Visualizer. Kết xuất file `.npy` và `.json` lưu sẵn vào ổ cứng, giúp Stage 2 chỉ cần đọc ma trận từ đĩa với độ trễ 0s.
*   **Phân mảnh dữ liệu siêu lớn ([split_dataset.py](file:///c:/Users/Admin/Desktop/TryHairStyle/split_dataset.py))**: Chia nhỏ kho dữ liệu thành các "Container" 5000 mẫu/cục, bao gồm kiểm tra chéo (Verify Integrity) xem mỗi mẫu đã đủ 5 thành phần chưa. Mẫu thiếu file tự động bị Drop.
*   **Chuẩn bị Dataset Deep Hair ([prepare_dataset_deephair.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/prepare_dataset_deephair.py))**: Pipeline ĐA LUỒNG (`ThreadPoolExecutor`) xử lý K-Hairstyle dataset:
    1. Đọc JSON label polygon → vẽ mask tóc chính xác (KHÔNG dùng SegFormer ở bước này — dùng polygon thật từ dataset).
    2. Tạo ảnh trọc (`bald_image`) bằng `cv2.inpaint(TELEA)` — xóa toàn bộ tóc khỏi ảnh gốc.
    3. Tách tóc riêng biệt (`hair_only_rgba`) — 4 kênh RGBA để giữ viền trong suốt.
    4. Cắt **Hair Patches 128×128** (các mảnh tóc thuần) với điều kiện ≥85% pixel là tóc — làm đầu vào cho Stage 1.
    5. Trích xuất Identity Embedding (AdaFace) và Style Vector cho mỗi ảnh.
    6. Ghi `metadata.jsonl` an toàn bằng Thread Lock chống corrupt.
*   **Chuẩn hóa nhãn tiếng Hàn ([normalize_khairstyle.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/data_processing/normalize_khairstyle.py))**: K-Hairstyle dataset gốc chứa nhãn tiếng Hàn (ví dụ: "긴 머리" = "long hair"). Script này quét toàn bộ file JSON, trích xuất tất cả từ tiếng Hàn, và tạo `mapping_dict.json` để dịch sang tiếng Anh — giúp Text Prompt Encoder (CLIP) hiểu được mô tả tóc.
*   **Trích xuất ảnh Ground Truth ([extract_eval_images.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/extract_eval_images.py))**: Copy ảnh gốc từ K-Hairstyle dataset vào thư mục `ground_truth_images/` dựa trên ID trong `metadata.jsonl`, dùng để so sánh kết quả sinh ảnh với bản gốc khi đánh giá chất lượng.

---

## 8. Huấn Luyện Giai Đoạn 1 — Texture Encoder ([texture_encoder.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/texture_encoder.py))

*   **Mục đích**: Dạy AI phân biệt các loại texture tóc (thẳng/xoăn/lượn sóng, ít/nhiều volume) TRƯỚC KHI bước vào Stage 2.
*   **Kiến trúc [HairTextureEncoder](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/texture_encoder.py#27-99)**: Backbone **ResNet-50** (pretrained ImageNet) + 3 đầu ra:
    1. [embedding](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/face.py#186-188) (2048-d): Đặc trưng sâu — cấp cho UNet (Style Injection) ở Stage 2.
    2. `proj` (128-d): Vector cho **Supervised Contrastive Loss (SupConLoss)** — kéo patch cùng loại tóc lại gần, đẩy patch khác loại ra xa trong không gian embedding.
    3. `cls_logits`: Phân loại phụ (Auxiliary) — ép mô hình học ranh giới vật lý rõ ràng: [curl](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/texture_encoder.py#104-112) (4 classes: thẳng/vểnh/xoăn/xoăn tít), [volume](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/texture_encoder.py#113-119) (3 classes: ít/bình thường/nhiều).
*   **Loss**: `CrossEntropyLoss` (phân loại) + `SupConLoss` (contrastive, temperature=0.07).
*   **Checkpointing**: Lưu mỗi 50 steps + cuối epoch. Hỗ trợ resume mid-epoch trên Colab. File `.safetensors` để an toàn và nhanh.

---

## 9. Kiến Trúc UNet 9-Channel & Cross-Attention Injector ([stage2_unet.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/stage2_unet.py))

*   **[HairInpaintingUNet](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/stage2_unet.py#8-99)**: "Phẫu thuật" lớp `conv_in` của SDXL UNet, mở rộng từ 4 kênh gốc → **9 kênh**:
    - 4 kênh: `noisy_latents` (latent nhiễu từ VAE Encode + Gaussian Noise)
    - 1 kênh: [mask](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/services/mask.py#119-162) (vùng tóc cần vẽ lại)
    - 4 kênh: `masked_latents` (ảnh gốc đã khoét mất vùng tóc, qua VAE).
    - Trọng số cũ được giữ nguyên ở 4 kênh đầu, 5 kênh mới khởi tạo `zeros` để không phá hỏng weights SDXL gốc.
    - Kích hoạt **Gradient Checkpointing** + **xFormers Memory-Efficient Attention** để tiết kiệm VRAM.
*   **[CrossAttentionInjector](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/stage2_unet.py#104-144)**: Module siêu nhẹ nằm cạnh UNet, chiếu (project) 2 vector:
    - `identity_proj`: Identity Embedding 512D → 2048D (cùng chiều với text prompt).
    - `style_proj`: Style Embedding → 2048D + LayerNorm.
    - Ghép 2 vector này vào chuỗi Cross-Attention của UNet (IP-Adapter concept): `[style_token, identity_token]`, giúp UNet "nhìn thấy" cả danh tính lẫn kiểu tóc mong muốn mà không đi qua CLIP Vision.

---

## 10. Hệ Thống Hàm Mất Mát (Loss Functions) — [losses.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/losses.py)

*   **[MaskAwareLoss](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/losses.py#82-108)** (Loss chính): Chỉ phạt sai số bên trong vùng mask tóc (weight=1.0), vùng nền/mặt chỉ phạt nhẹ (weight=0.1). Điều này "khóa gradient" vùng mặt, ép UNet tập trung 100% sức lực vẽ tóc.
*   **[TextureConsistencyLoss](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/losses.py#34-81)** (Monitor): Dùng **VGG16 Gram Matrix** (Style Loss) — so sánh "cảm giác" bề mặt tóc (lọn xoăn, độ bóng) giữa ảnh sinh ra và ảnh gốc. Chuẩn hóa input từ `[-1,1]` → ImageNet normalize trước khi đưa vào VGG.
*   **[IdentityCosineLoss](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/losses.py#109-129)** (Monitor): Cosine Similarity giữa embedding khuôn mặt gốc vs ảnh sinh ra. Target = 1.0 (hoàn toàn giống nhau). Mục tiêu đạt ≥ 0.90.
*   **[FaceFeatureExtractor](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/models/losses.py#130-241)**: Dùng **InceptionResnetV1** (pretrained VGGFace2, 3.3M ảnh mặt) để trích xuất face embedding 512-D trong quá trình training. Hoàn toàn **frozen** (không train), chỉ dùng để đo đạc.

---

## 11. Xuất Model & Triển Khai ([export_model.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/export_model.py) & [evaluate.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/evaluate.py))

*   **[CheckpointManager](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/export_model.py#25-369)** ([export_model.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/export_model.py)):
    - Tự động tìm checkpoint tốt nhất (ưu tiên: `lora_best` > `lora_latest` > legacy `deep_hair_v1_best`).
    - **Kiểm tra chất lượng**: Phát hiện NaN/Inf trong weights, đếm số params, phân biệt LoRA vs Full UNet.
    - **Merge LoRA → Full UNet**: Dùng `peft.merge_and_unload()` để hợp nhất adapter LoRA vào SDXL UNet gốc → xuất ra 1 file `.safetensors` duy nhất. Production không cần thư viện `peft`.
    - **Upload HuggingFace Hub**: Tự động push checkpoint + model export lên HF Hub (primary) sau khi merge.
*   **[HairEvaluator](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/evaluate.py#11-131)** ([evaluate.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/training/evaluate.py)): Đánh giá chất lượng tóc sinh ra bằng 2 metric:
    - **LPIPS** (VGG backbone): Đo khoảng cách thị giác — càng thấp càng giống ảnh thật.
    - **Masked PSNR**: Đo sai số pixel chỉ trong vùng tóc (mask-aware), tránh nhiễu từ background.

---

## 12. Hạ Tầng Kỹ Thuật (Infrastructure)

*   **[config.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/config.py)**: Quản lý tất cả đường dẫn model (SDXL, ControlNet, IP-Adapter, InsightFace) và cấu hình server (HOST, PORT, REDIS_URL, DEVICE).
*   **[schemas.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/schemas.py)**: Pydantic schemas cho FastAPI — validate request/response cho `/generate`, `/colorize`, `/status`.
*   **[torch_patch.py](file:///c:/Users/Admin/Desktop/TryHairStyle/backend/app/utils/torch_patch.py)**: Module "vá lỗi tương thích" — giả lập `torch.xpu` (Intel GPU), patch `flash_attention`, `accelerate`, và `huggingface_hub` để tránh crash trên các phiên bản thư viện khác nhau giữa local và Colab.
*   **[download_models.py](file:///c:/Users/Admin/Desktop/TryHairStyle/download_models.py)**: Script tải tất cả models (SDXL, ControlNet, SegFormer, YOLOv8-Face, AdaFace, 3DDFA V2) từ HuggingFace Hub và Google Drive về thư mục `backend/models/`.
*   **`docker-compose.yml`**: Đóng gói toàn bộ: FastAPI + Celery Worker + Redis vào Docker containers cho triển khai production.

