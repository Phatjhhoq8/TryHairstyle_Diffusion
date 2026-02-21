# Kế Hoạch Vận Hành Kịch Bản: Deep Texture Hair Model (Training Pipeline)

Tài liệu này hướng dẫn chi tiết luồng thực thi (Execution Flow) từ việc chuẩn bị hình ảnh đến lúc ra đời Model có khả năng vẽ tóc theo yêu cầu, giữ nguyên khuôn mặt gốc. Toàn bộ nằm trong thư mục `backend/training/`.

---

## Giai Đoạn 1: Prepare Dataset (Tạo Dữ liệu Đầu Vào)

**Script chạy:** `python backend/training/prepare_dataset_deephair.py`

*   **Đầu vào (Input)**: Ảnh gốc Hàn Quốc từ K-Hairstyle Dataset và nhãn Polygon JSON.
*   **Xử lý (Process)**:
    1.  Tự động dịch các từ khóa miêu tả tóc tiếng Hàn thành tiếng Anh bằng `mapping_dict.json` để tạo Text Prompt.
    2.  Đọc tọa độ Polygon để tạo mask trắng/đen vùng tóc hoàn hảo (`hair_mask`).
    3.  Cắt 5 bộ dữ liệu con cần thiết cho mạng SDXL:
        *   `bald_images/`: Ảnh mặt bị cạo trọc (sẽ dùng làm ảnh đầu vào cho UNet - Bald Latent).
        *   `hair_only_images/`: Ảnh chỉ chứa tóc, còn lại trong suốt.
        *   `style_vectors/`: Patch tóc 224x224 chuyên châm vào bộ giải mã Style CLIP.
        *   `identity_embeddings/`: File `.npy` chứa vector sinh lý khung xương mặt trích từ mạng AdaFace.
        *   `hair_patches/`: Các miếng mảnh tóc cực nhỏ để dạy model vân tóc.
*   **Đầu ra (Output)**: Thư mục `training/processed/` chứa 5 thư mục ảnh nhỏ kèm file `metadata.jsonl`.

---

## Giai Đoạn 2: Stage 1 - Texture Reconstruction (Học Chất Liệu Tóc)

**Script chạy:** `python backend/training/models/texture_encoder.py`

*   **Đầu vào (Input)**: Bức ảnh từ thư mục `hair_patches/` và `hair_only_images/`.
*   **Xử lý (Process)**:
    1.  Mạng Autoencoder bóc tách từng sợi tóc, ép chúng thành một khối nhỏ (Latent) rồi phóng to ra.
    2.  Dùng thuật toán **Supervised Contrastive Learning (SupConLoss)** trong `losses.py` để kéo các patch có cùng tính chất vật lý như "tóc xoăn" (Curly) lại thành một chùm chuẩn tắc.
*   **Đầu ra (Output)**: Một model có khả năng giải phẫu và thấu hiểu "vân sợi tóc" siêu thực.

---

## Giai Đoạn 3: Stage 2 - Mask-Conditioned Inpainting (Huấn Luyện Model Chính)

**Script chạy:** `python backend/training/train_stage2.py`

*   **Đầu vào (Input)**: `bald_image`, `hair_mask`, và các vector `style` + `identity`.
*   **Xử lý (Process)**: 
    1.  **UNet 9-Channel**: Mạng được nới rộng từ 4 kênh (SDXL Mặc định) lên 9 kênh để nhận cùng lúc [Latent + Bald_Latent + Mask]. Nó nhận biết đâu là vùng được quyền vẽ (Vùng trắng của Mask), đâu là vùng cấm chép đè.
    2.  **IP-Adapter Module**: (Cross-Attention) Tiêm vector nhận diện khuôn mặt (`identity`) và kiểu tóc (`style`) vào mạch tư duy của UNet.
    3.  **Hàm Loss Gradient Locking**: 
        *   Sử dụng `MaskAwareLoss` để ép mô hình chỉ tính lỗi khi lỡ vẽ sai trong ranh giới tóc, gradient bên ngoài (phông nền) $= 0$.
        *   Sử dụng `IdentityCosineLoss` tính độ vênh lệch khuôn mặt sinh ra với ảnh thật. Ép Cosine Similarity lên ngưỡng $>0.90$ (Chắn biến dạng mặt).
        *   Sử dụng `TextureConsistencyLoss` gọi mạng VGG16 ra chấm điểm mật độ sắc nét hạt màu của tóc.
*   **Đầu ra (Output)**: Một file trọng số (Weights `.safetensors`) chứa kiến thức vẽ tóc đỉnh cao, lưu định kỳ tại `backend/training/checkpoints/`.

---

## Giai Đoạn 4: Evaluation & Model Export (Thẩm Định và Đẩy Lênh Production)

**Script chạy:** `python backend/training/export_model.py` (Kích hoạt ngầm `evaluate.py`)

*   **Đầu vào (Input)**: File weights `.safetensors` mới nhất ở thư mục `checkpoints/`.
*   **Xử lý (Process)**:
    1.  **Bóc tách vùng tóc**: Script `evaluate.py` dùng tọa độ của Mask để cắt riêng hình bóng của mái tóc sinh ra.
    2.  **Chấm điểm LPIPS & PSNR**: Chỉ tính độ lệch màu sắc, tính thẩm mỹ của con người đánh giá trên viền bao đúng vùng tóc (loại trừ các thành phần khuôn mặt xuất sắc).
    3.  Kiểm tra điều kiện: Nếu `Identity > 0.90` (mặt 90% giống Gốc) và `LPIPS < 0.20` (tóc sinh ra đẹp)...
*   **Đầu ra (Output)**: Tự động sao chép file cấu hình hoàn chỉnh vào `backend/models/deep_hair_v1.safetensors`, báo hiệu Web App đã có thể tích hợp AI vẽ tóc thế hệ mới.

---

## Hướng Dẫn Kích Hoạt (One-Click Training)

Thay vì phải gõ lệnh chạy lẻ tẻ từng giai đoạn phía trên, mình đã gom gọn toàn bộ 4 bước đào tạo này vào trong 1 file chạy tự động duy nhất (Shell Script). 

Phần mềm sẽ tự liên kết: Bóc tách Dữ liệu -> Dạy Texture sợi cáp mảnh -> Dạy Bố cục inpaint 9-kênh -> Mở bài Test chấm điểm LPIPS -> Đẩy ra Website.

Để quá trình bắt đầu, mở Terminal (Ubuntu/WSL) và chạy lệnh:

```bash
cd backend/training/
bash run_training_pipeline.sh
```

**Lưu ý:**
* Đảm bảo môi trường `venv_wsl` Python đã được cài đặt các thư viện cần thiết (`torch`, `diffusers`, `lpips`...).
* Máy tính sẽ yêu cầu tối thiểu **24GB VRAM** GPU (như RTX 3090/4090) do kiến trúc của mạng SDXL.
* Logs (Nhật ký) của quá trình học sẽ được xuất thẳng ra màn hình Console.

---
**Tóm Lược Tác Giả & Vai Trò (The Full Pipeline):**
> Prepare Dataset -> [1] Texture Extractor -> [2] SDXL UNet Inpainter (+ Injection/Losses) -> Evaluator -> Production Web App Deployment.
