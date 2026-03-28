# Tổng hợp Các Công Nghệ & Kỹ Thuật trong Hệ Thống TryHairStyle Diffusion

Tài liệu này tóm tắt toàn bộ các công nghệ, thuật toán và kỹ thuật được sử dụng trong hệ thống **TryHairStyle** để bạn ôn tập và chuẩn bị bảo vệ đồ án/chuyên đề.

---

## 1. Mô hình Sinh Ảnh (Generative AI)
Đây là cốt lõi của bài toán biến đổi hoặc chuyển giao kiểu tóc (Hair Transfer).
* **Stable Diffusion:** Kiến trúc nền tảng sinh ảnh từ không gian tiềm ẩn (Latent Space). Hệ thống dùng mã hóa ảnh từ cấu trúc Ground Truth (GT) thành chuỗi latent qua bộ mã hóa VAE, sau đó dùng Diffusion Model để giải nhiễu.
* **UNet (13-channel input):** UNet được thiết kế và tinh chỉnh lại (Fine-tuned) với đầu vào 13 kênh để có thể tiếp nhận được nhiều điều kiện ảnh đầu vào (ảnh gốc, mask, ...). Quá trình train dùng LoRA (Low-Rank Adaptation) để tối ưu hóa trọng số.
* **IP-Adapter Plus & HairTextureEncoder (ResNet-50):** Trích xuất vector đặc trưng về kiểu dáng, cấu trúc và kết cấu (texture) của tóc từ ảnh tham chiếu, phục vụ "Nhánh truyền tải phong cách tóc".
* **ControlNet Depth:** Hỗ trợ duy trì cấu trúc hình học tổng thể của khuôn mặt, tránh dị dạng.

## 2. Thị giác Máy tính & Phân tích Khuôn mặt (Computer Vision)
Các mô hình chuyên biệt xử lý dữ liệu đầu vào để bảo tồn danh tính người dùng và phát hiện đặc trưng khuôn mặt.
* **YOLOv8-Face (YOLOv8n):** Mô hình siêu nhẹ giúp phát hiện khuôn mặt nhanh chóng trong ảnh, tạo Bounding Box (vùng chứa mặt ngang/dọc). Sử dụng thuật toán NMS (Non-Maximum Suppression) để lọc các box trùng lấn.
* **InsightFace (antelopev2):** Phân tích chi tiết khuôn mặt, trích xuất 106 điểm landmarks và vector đặc trưng (ArcFace 512 chiều) để bảo toàn danh tính (kết hợp với InstantID).
* **AdaFace IR-100 & MTCNN:** Là cơ chế dự phòng (fallback). Khi khuôn mặt nghiêng góc quá lớn ($|yaw| \geq 45^\circ$) làm InsightFace thất bại, đối tượng sẽ chạy qua căn chỉnh bằng MTCNN rồi nhúng bởi AdaFace.
* **3DDFA V2:** Tái dựng lưới khuôn mặt 3D (dense vertices) từ ảnh 2D tĩnh. Giúp dự đoán "da đầu" (scalp) cho các trường hợp người dùng bị hói/ít tóc để sinh tóc thật chính xác.

## 3. Phân đoạn Ngữ nghĩa (Image Segmentation)
* **SegFormer (jonathandinu/face-parsing):** Phân chia ảnh thành 19 lớp (classes) khác nhau (da, tóc, mắt, mũi, cổ, nền...).
* **Hair Mask & Face Mask:** Từ 19 lớp này hệ thống sẽ hợp nhất để ra `hair_mask` (vùng tóc, mũ) và `face_mask` (vùng cần bảo vệ). Hệ thống áp dụng thêm kỹ thuật **Forehead Unmasking** (Bỏ mặt nạ vùng trán để sinh tóc mái tự nhiên).

## 4. Xử lý Ảnh & Thuật toán Cơ bản (Image Processing)
* **HSV Color Transfer:** Giải pháp "Đổi màu tóc cấp tốc" (Quick Color Change). Chuyển đổi kênh màu RGB sang HSV, thay đổi thành phần Hue kết hợp nội suy 80/20 Saturation để đổi màu tóc mà không cần chạy AI phức tạp, tiết kiệm tài nguyên.
* **Paste-back (Hậu xử lý Bbox):** Cơ chế quan trọng nhất để **giữ nguyên hậu cảnh**. Quá trình inpainting chỉ diễn ra trong Bbox đã chuẩn hóa (512x512). Sau khi sinh xong, bức ảnh được khớp và cắt dán (soft blend bằng Gaussian Blur) trả ngược về tọa độ tỉ lệ tuyệt đối ban đầu.

## 5. Kiến trúc Backend & Xử lý Hệ thống
Do tác vụ Deep Learning ngốn rất nhiều RAM/VRAM và thời gian thực thi, hệ thống sử dụng kiến trúc bất đồng bộ (Asynchronous Workflow):
* **FastAPI:** Framework viết RESTful API cực nhanh, nhận hình, validate và trả về `task_id` cho người dùng ngay lập tức (tránh HTTP timeout).
* **Redis:** Hoạt động như Message Broker. Làm hàng đợi (queue) trung gian chuyển tiếp nhiệm vụ và lưu trữ trạng thái.
* **Celery:** Các Worker ngầm trong hệ thống để thực thi tiến trình xử lý đồ họa cường độ cao. Cập nhật các tiến độ như *PENDING, PROCESSING, SUCCESS* về cho frontend.
* **Frontend (React/Vite & Gradio):** Bắt `task_id` và dùng kỹ thuật Polling (gọi lại API API `/status/{task_id}` mỗi 2 giây) để cập nhật giao diện người dùng theo tiến trình thực.

## 6. Huấn luyện (Training) & Tối ưu hóa
Quá trình huấn luyện hệ thống 13-channel UNet được ghi chú với các công nghệ:
* **Framework:** PyTorch, HuggingFace Diffusers, Accelerate.
* **Môi trường & Phần cứng:** Google Colab với GPU T4, sử dụng quy trình đào tạo kết hợp Mixed Precision (FP16) - Float điểm ảnh bán xác suất giúp tăng tốc tính toán và tiết kiệm VRAM. Chống lỗi OOM (Out Of Memory) bằng Auto-caching system.
* **Hàm mất mát (Loss Functions):**
  * *Mask-Aware Diffusion Loss:* Phục vụ lan truyền ngược (backpropagation).
  * *LPIPS:* Đo lường độ chân thực cấu trúc cảm giác.
  * *Texture Consistency / Identity Preservation:* Đo cường độ khớp/bảo toàn bề mặt tóc và danh tính mặt người.

---
### 💡 Mẹo khi trả lời Ban Giám Khảo:
- Nếu bị hỏi **"Làm sao giữ nguyên được hình nền phía sau khi người dùng tạo tóc?"**: Trả lời ngay là dùng **YOLOv8-Face** cắt `bbox` vùng đầu, crop chuẩn ra `512x512`, xử lý AI xong thì dùng **Toán tử Paste-Back (Dán đè)** pha trộn Gaussian blur vào biên, áp nguyên vùng `bbox` đo về đúng vị trí pixel của ảnh gốc.
- Nếu bị hỏi **"Sao không dùng một mô hình để sinh luôn mọi thứ?"**: Trả lời là kết hợp nhiều mô hình nhỏ lẻ theo luồng (*InsightFace -> Segment -> Diffusion*) giúp điều hướng **Condition (Điều kiện chéo)** tốt hơn, tránh AI tự "ảo giác" (hallucination) thay đổi danh tính/vẻ mặt người dùng.
- Trình bày về **cách giảm tải máy chủ**: Đề cập đến API Queue với **Celery + Redis** và việc đổi màu tóc nhanh gọn nhẹ bằng kỹ thuật **HSV nội suy**.
