# Luồng Xử Lý Chi Tiết (Deep Dive Execution Flow)

Tài liệu này mô tả chi tiết luồng dữ liệu (Data Flow) và logic xử lý từng bước trong các hàm Backend.

## 1. API Endpoint (Nhận Request)
*   **File:** `backend/app/main.py`
*   **Hàm:** `generate_hair`
*   **Mục đích:** Cổng giao tiếp nhận dữ liệu từ Frontend.

### Input Data
| Tên biến | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `face_image` | `UploadFile` | File ảnh khuôn mặt người dùng upload. |
| `hair_image` | `UploadFile` | File ảnh mẫu tóc upload. |
| `description` | `str` | Prompt mô tả tóc (Mặc định: "high quality..."). |

### Logic Xử Lý
1.  **Generate UUID:** Tạo tên file ngẫu nhiên (VD: `a1b2..._face.png`) để tránh trùng lặp.
2.  **Save Files:**  Lưu 2 file ảnh vào thư mục `backend/uploads/`.
3.  **Trigger Celery:** Gọi hàm `process_hair_transfer.delay(...)` để đẩy task vào hàng đợi.

### Output Data
| Tên biến | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `task.id` | `str` | ID định danh duy nhất của Task (gửi về cho Frontend polling). |

---

## 2. Task Orchestrator (Điều Phối Quy Trình)
*   **File:** `backend/app/tasks.py`
*   **Hàm:** `process_hair_transfer`
*   **Mục đích:** Worker chạy ngầm, quản lý toàn bộ quy trình AI.

### Input Data
| Tên biến | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `user_img_path` | `str` | Đường dẫn tuyệt đối tới ảnh mặt đã lưu. |
| `hair_img_path` | `str` | Đường dẫn tuyệt đối tới ảnh tóc đã lưu. |
| `prompt` | `str` | Text mô tả kiểu tóc. |

### Logic Xử Lý
1.  **Load Image:**
    *   `user_cv2` = `cv2.imread(...)` (Dùng cho bước Phân tích mặt).
    *   `user_pil` = `Image.open(...)` (Dùng cho bước AI tạo ảnh).
    *   `hair_pil` = `Image.open(...)` (Dùng làm mẫu tóc).

2.  **Bước 2a: Phân tích mặt (Face Analysis)**
    *   **Gọi:** `face_service.analyze(user_cv2)`
    *   **File:** `backend/app/services/face.py`
    *   **Chức năng:** Dùng InsightFace check xem có mặt người không. Nếu `None` -> Báo lỗi ngay.

3.  **Bước 2b: Tạo Mask Tóc (Segmentation)**
    *   **Gọi:** `mask_service.get_mask(user_pil)`
    *   **File:** `backend/app/services/mask.py`
    *   **Chức năng:**
        *   Resize ảnh về 512x512.
        *   Chạy qua model **BiSeNet**.
        *   Lấy class ID 17 (Hair).
        *   Trả về ảnh Mask đen trắng (`hair_mask`).

4.  **Bước 2c: Tạo Depth Map (ControlNet Input)**
    *   **Gọi:** `depth_estimator(user_pil)`
    *   **Thư viện:** `transformers`
    *   **Chức năng:** Tạo ảnh bản đồ độ sâu (`depth_map`) để AI hiểu khối mũi, gò má, trán của khuôn mặt.

5.  **Bước 2d: Sinh ảnh (Generation)**
    *   **Gọi:** `diffusion_service.generate(...)` (Xem chi tiết mục 3).

### Output Data
| Tên biến | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `result_path` | `str` | Đường dẫn file ảnh kết quả (`backend/output/...`). |
| `url` | `str` | Link tĩnh để Frontend hiển thị (`/static/output/...`). |

---

## 3. Core AI Engine (Sinh Ảnh)
*   **File:** `backend/app/services/diffusion.py`
*   **Hàm:** `generate`
*   **Mục đích:** Chạy model Stable Diffusion để vẽ tóc mới.

### Input Data
| Tên biến | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `base_image` | `PIL.Image` | Ảnh gốc của người dùng. |
| `mask_image` | `PIL.Image` | Ảnh Mask (Vùng trắng = Vùng sẽ bị vẽ lại). |
| `control_image` | `PIL.Image` | Depth Map (để giữ cấu trúc khuôn mặt). |
| `ref_hair_image`| `PIL.Image` | Ảnh mẫu tóc (để copy style/màu sắc). |
| `prompt` | `str` | Lời nhắc bổ sung (VD: "blonde, curly"). |

### Logic Xử Lý
1.  **Preprocessing:**
    *   Resize tất cả (`base`, `mask`, `ref`) về `512x512`.
2.  **IP-Adapter Configuration:**
    *   `set_ip_adapter_scale(0.6)`: Trọng số ảnh tham chiếu (60% giống ảnh mẫu).
3.  **Inference Loop (Stable Diffusion Inpaint):**
    *   **Model:** `runwayml/stable-diffusion-inpainting` (SD1.5).
    *   **Input vào Model:**
        *   `image`: Ảnh gốc.
        *   `mask_image`: Mask tóc.
        *   `ip_adapter_image`: Ảnh tóc mẫu.
    *   **Quá trình:** Model chạy 30 bước (num_inference_steps=30) khử nhiễu. Tại vùng mask màu trắng, model sẽ xem xét `depth_map` (giữ khối mặt), `ref_hair` (kiểu tóc), và `prompt` để vẽ lại pixel tóc mới.
4.  **Postprocessing:** Output là danh sách ảnh, lấy ảnh đầu tiên `[0]`.

### Output Data
| Tên biến | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `result` | `PIL.Image` | Ảnh hoàn chỉnh 512x512 đã ghép tóc mới. |
