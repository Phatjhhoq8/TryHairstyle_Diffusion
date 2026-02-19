# 🔄 TryHairStyle - Luồng Thực Thi Chi Tiết

---

## 📊 Sơ Đồ Tổng Quan

```
[User Upload] → [API Endpoint] → [Celery Task] → [AI Pipeline] → [Output Image]
```

---

# 🟢 PHASE 1: FRONTEND → API

## Bước 1.1: User Upload Ảnh
| Chi tiết | Giá trị |
|:---|:---|
| **File** | `frontend/src/components/HairSwapper.jsx` |
| **Hàm** | `handleGenerate()` |
| **Dòng** | 82-116 |

**Thực hiện:**
```javascript
// Dòng 92-96: Tạo FormData chứa 2 ảnh
const formData = new FormData();
formData.append('face_image', targetFile);    // Ảnh khuôn mặt
formData.append('hair_image', referenceFile); // Ảnh tóc mẫu
formData.append('description', prompt);        // Prompt mô tả
formData.append('use_refiner', useRefiner);    // Bật/tắt refiner
```

---

## Bước 1.2: API Nhận Request
| Chi tiết | Giá trị |
|:---|:---|
| **File** | `backend/app/main.py` |
| **Hàm** | `generate_hair()` |
| **Dòng** | 48-79 |
| **Endpoint** | `POST /generate` |

**Thực hiện:**

### 1.2.1: Lưu file vào thư mục uploads
```python
# Dòng 60-61: Tạo tên file ngẫu nhiên
face_filename = f"{uuid.uuid4()}_face.{ext}"
hair_filename = f"{uuid.uuid4()}_hair.{ext}"

# Dòng 63-64: Xác định đường dẫn lưu
face_path = UPLOAD_DIR / face_filename
hair_path = UPLOAD_DIR / hair_filename

# Dòng 66-70: Ghi file vào disk
with open(face_path, "wb") as f:
    shutil.copyfileobj(face_image.file, f)
with open(hair_path, "wb") as f:
    shutil.copyfileobj(hair_image.file, f)
```

### 1.2.2: Trigger Celery Task
```python
# Dòng 73: Gọi Celery task bất đồng bộ
task = process_hair_transfer.delay(
    str(face_path),   # Đường dẫn ảnh mặt
    str(hair_path),   # Đường dẫn ảnh tóc
    description,      # Prompt
    use_refiner       # Có dùng refiner không
)
```

### 1.2.3: Trả về Task ID
```python
# Dòng 75-79: Response cho Frontend
return {
    "task_id": task.id,        # ID để polling
    "status": "QUEUED",
    "message": "Task started successfully"
}
```

---

# 🟡 PHASE 2: CELERY WORKER KHỞI TẠO

## Bước 2.1: Load AI Services (Lazy Loading)
| Chi tiết | Giá trị |
|:---|:---|
| **File** | `backend/app/tasks.py` |
| **Hàm** | `get_services()` |
| **Dòng** | 29-50 |

**Thực hiện:**

### 2.1.1: Khởi tạo FaceInfoService
```python
# File: backend/app/services/face.py
# Hàm: FaceInfoService.__init__()
# Dòng: 7-20

self.app = FaceAnalysis(
    name='antelopev2',                    # Model InsightFace
    root=model_paths.INSIGHTFACE_ROOT,    # Thư mục chứa model
    providers=['CUDAExecutionProvider']   # Ưu tiên GPU
)
self.app.prepare(ctx_id=0, det_size=(640, 640))  # Chuẩn bị detect
```

### 2.1.2: Khởi tạo SegmentationService
```python
# File: backend/app/services/mask.py
# Hàm: SegmentationService.__init__()
# Dòng: 56-68

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
self.processor = SegformerImageProcessor.from_pretrained(model_paths.SEGFORMER_LOCAL_PATH)
self.model = SegformerForSemanticSegmentation.from_pretrained(model_paths.SEGFORMER_LOCAL_PATH)
self.model.to(self.device).eval()
```

### 2.1.3: Khởi tạo HairDiffusionService
```python
# File: backend/app/services/diffusion.py
# Hàm: HairDiffusionService.__init__()
# Dòng: 20-44

if self.use_sdxl:
    self._load_sdxl_pipeline()  # Load SDXL + ControlNet + IP-Adapter
```

### 2.1.4: Load SDXL Pipeline chi tiết
```python
# File: backend/app/services/diffusion.py
# Hàm: _load_sdxl_pipeline()
# Dòng: 112-182

# a) Load ControlNet Depth (Dòng 117-121)
controlnet = ControlNetModel.from_pretrained(
    model_paths.CONTROLNET_DEPTH,
    torch_dtype=torch.float16
)

# b) Load SDXL Inpaint Pipeline (Dòng 135-140)
self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    model_paths.SDXL_BASE,
    controlnet=controlnet,
    torch_dtype=torch.float16
)

# c) Load IP-Adapter (Dòng 170-175)
self.pipe.load_ip_adapter(
    model_paths.IP_ADAPTER_PLUS_HAIR,
    weight_name="ip-adapter-plus_sdxl_vit-h.bin"
)

# d) Chuyển sang GPU (Dòng 181)
self.pipe.to(self.device, self.dtype)
```

---

# 🔵 PHASE 3: XỬ LÝ AI PIPELINE

## Bước 3.1: Load Ảnh Đầu Vào
| Chi tiết | Giá trị |
|:---|:---|
| **File** | `backend/app/tasks.py` |
| **Hàm** | `process_hair_transfer()` |
| **Dòng** | 64-72 |

```python
# Dòng 64-66: Load ảnh user (OpenCV + PIL)
user_cv2 = cv2.imread(user_img_path)
user_cv2 = cv2.cvtColor(user_cv2, cv2.COLOR_BGR2RGB)
user_pil = Image.fromarray(user_cv2)

# Dòng 70-72: Load ảnh tóc mẫu
hair_pil = Image.open(hair_img_path).convert("RGB")
```

---

## Bước 3.2: Phân Tích Khuôn Mặt (Face Analysis)
| Chi tiết | Giá trị |
|:---|:---|
| **File** | `backend/app/services/face.py` |
| **Hàm** | `FaceInfoService.analyze()` |
| **Dòng** | 22-37 |

**Thực hiện:**

### 3.2.1: Detect tất cả khuôn mặt
```python
# Dòng 27: Sử dụng InsightFace detect
faces = self.app.get(image_cv2)
```

### 3.2.2: Xử lý Profile Face (Góc nghiêng > 45°)
```python
# Kiểm tra góc Yaw từ Pose
if abs(yaw) > 45:
    # Sử dụng 3DDFA_V2 để dựng 3D Pose & Landmarks
    # Thực hiện Roll Correction (Xoay thẳng đầu)
    # Align & Crop 112x112
    # Trích xuất Embedding bằng AdaFace
else:
    # Sử dụng InsightFace/AdaFace 2D alignment thông thường
```

### 3.2.3: Kiểm tra và xoay ảnh nếu cần
```python
# Dòng 29-31: Nếu không tìm thấy mặt, thử xoay 90°
if len(faces) == 0:
    rotated = cv2.rotate(image_cv2, cv2.ROTATE_90_CLOCKWISE)
    faces = self.app.get(rotated)
```

### 3.2.4: Chọn khuôn mặt lớn nhất
```python
# Dòng 32-36: Sort theo diện tích bbox
faces = sorted(
    faces, 
    key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), 
    reverse=True
)
return faces[0]  # Trả về mặt lớn nhất
```

**Output:** Object `face_info` chứa:
- `.embedding` - Vector đặc trưng khuôn mặt (512D)
- `.kps` - 5 keypoints (mắt, mũi, miệng)
- `.bbox` - Bounding box [x1, y1, x2, y2]

---

## Bước 3.3: Tạo Hair Mask (Segmentation)
| Chi tiết | Giá trị |
|:---|:---|
| **File** | `backend/app/services/mask.py` |
| **Hàm** | `SegmentationService.get_mask()` |
| **Dòng** | 70-98 |

**Thực hiện:**

### 3.3.1: Resize ảnh về 512x512
```python
# Dòng 74-75: Lưu kích thước gốc
w, h = image_pil.size

# Dòng 77: Resize cho SegFormer
img_resized = image_pil.resize((512, 512), Image.BILINEAR)
```

### 3.3.2: Transform sang Tensor
```python
# Dòng 79: Chuyển sang tensor GPU
img_tensor = self.to_tensor(img_resized).unsqueeze(0).to(self.device)
```

### 3.3.3: Chạy SegFormer inference
```python
# Dòng 81-83: Forward pass
with torch.no_grad():
    out = self.net(img_tensor)[0]
    parsing = out.squeeze(0).cpu().numpy().argmax(0)
```

### 3.3.4: Tạo binary mask cho class "hair" (17)
```python
# Dòng 87-88: Class 17 = Hair trong CelebAMask-HQ
mask = np.zeros_like(parsing).astype(np.uint8)
mask[parsing == 17] = 255  # Vùng tóc = trắng (255)
```

### 3.3.5: Resize về kích thước gốc
```python
# Dòng 91: Resize mask về kích thước ảnh gốc
mask_cv2 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
```

### 3.3.6: Dilate mask (mở rộng vùng)
```python
# Dòng 94-95: Mở rộng mask để inpaint tốt hơn
kernel = np.ones((5, 5), np.uint8)
mask_dilated = cv2.dilate(mask_cv2, kernel, iterations=2)
```

**Output:** `PIL.Image` - Binary mask (0 = không tóc, 255 = vùng tóc)

---

## Bước 3.4: Ước Tính Depth Map
| Chi tiết | Giá trị |
|:---|:---|
| **File** | `backend/app/tasks.py` |
| **Dòng** | 106-109 |
| **Model** | `Intel/dpt-large` (HuggingFace) |

```python
# Dòng 106-107: Load depth estimator
from transformers import pipeline
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")

# Dòng 109: Chạy inference
depth_result = depth_estimator(user_pil)
depth_map = depth_result['depth']  # PIL Image grayscale
```

**Output:** `PIL.Image` - Grayscale depth map (gần = sáng, xa = tối)

---

## Bước 3.5: Sinh Ảnh AI (SDXL Inpainting)
| Chi tiết | Giá trị |
|:---|:---|
| **File** | `backend/app/services/diffusion.py` |
| **Hàm** | `HairDiffusionService.generate()` |
| **Dòng** | 214-349 |

**Thực hiện:**

### 3.5.1: Resize tất cả input về 1024x1024
```python
# Dòng 232-236: SDXL yêu cầu 1024x1024
image = base_image.resize((1024, 1024), Image.LANCZOS)
mask = mask_image.resize((1024, 1024), Image.NEAREST)
control = control_image.resize((1024, 1024), Image.LANCZOS)
ref_hair = ref_hair_image.resize((1024, 1024), Image.LANCZOS)
```

### 3.5.2: Set IP-Adapter scale
```python
# Dòng 243-244: Độ mạnh của style transfer
self.pipe.set_ip_adapter_scale(0.6)  # 0.6 = vừa phải
```

### 3.5.3: Chuẩn bị Generator cho reproducibility
```python
# Dòng 255-256: Random seed
generator = torch.Generator(device=self.device)
generator.manual_seed(42)  # Seed cố định để kết quả ổn định
```

### 3.5.4: Chuẩn bị arguments cho pipeline
```python
# Dòng 272-290: Tất cả tham số
input_args = {
    "prompt": prompt,                          # "high quality hair..."
    "negative_prompt": negative_prompt,        # "blurry, bad quality..."
    "image": image,                            # Ảnh gốc 1024x1024
    "mask_image": mask,                        # Hair mask
    "control_image": control,                  # Depth map
    "ip_adapter_image": ref_hair,              # Ảnh tóc mẫu
    "num_inference_steps": 30,                 # Số bước diffusion
    "guidance_scale": 7.5,                     # CFG scale
    "controlnet_conditioning_scale": 0.5,      # Độ mạnh ControlNet
    "strength": 0.99,                          # Inpaint strength
    "generator": generator
}
```

### 3.5.5: Chạy SDXL Pipeline
```python
# Dòng 296: Forward pass chính
result = self.pipe(**input_args).images[0]
```

### 3.5.6: (Optional) Chạy Refiner
```python
# Dòng 304-316: Nếu use_refiner=True
if use_refiner and self.refiner:
    result = self.refiner(
        prompt=prompt,
        image=result,
        num_inference_steps=20,
        denoising_start=0.8,       # Chỉ refine 20% cuối
        generator=generator
    ).images[0]
```

**Output:** `PIL.Image` - Ảnh kết quả 1024x1024

---

# 🟣 PHASE 4: LƯU KẾT QUẢ VÀ TRẢ VỀ

## Bước 4.1: Lưu Ảnh Output
| Chi tiết | Giá trị |
|:---|:---|
| **File** | `backend/app/tasks.py` |
| **Dòng** | 123-125 |

```python
# Dòng 123: Tạo tên file với task ID
filename = f"result_{self.request.id}.png"

# Dòng 124: Đường dẫn đầy đủ
output_path = os.path.join(OUTPUT_DIR, filename)

# Dòng 125: Lưu file
result_image.save(output_path)
```

---

## Bước 4.2: Trả Về Kết Quả
| Chi tiết | Giá trị |
|:---|:---|
| **File** | `backend/app/tasks.py` |
| **Dòng** | 127-134 |

```python
# Dòng 127-134: Return dict cho Celery
return {
    "status": "SUCCESS",
    "url": f"/static/output/{filename}",  # URL để Frontend download
    "filename": filename
}
```

---

## Bước 4.3: Frontend Polling Status
| Chi tiết | Giá trị |
|:---|:---|
| **File** | `frontend/src/components/HairSwapper.jsx` |
| **Hàm** | `useEffect` (polling) |
| **Dòng** | 17-48 |

```javascript
// Dòng 22-42: Poll mỗi 2 giây
intervalId = setInterval(async () => {
    const response = await fetch(`/status/${taskId}`);
    const data = await response.json();
    
    if (data.status === 'SUCCESS') {
        setResultUrl(data.result_url);  // Hiển thị ảnh
        setIsLoading(false);
    }
}, 2000);
```

---

## Bước 4.4: API Trả Status
| Chi tiết | Giá trị |
|:---|:---|
| **File** | `backend/app/main.py` |
| **Hàm** | `get_task_status()` |
| **Dòng** | 107-135 |

```python
# Dòng 112: Lấy kết quả từ Celery
task_result = AsyncResult(task_id, app=celery_app)

# Dòng 119-124: Nếu SUCCESS, trả về URL
if task_result.status == 'SUCCESS':
    result_data = task_result.result
    response["result_url"] = result_data.get("url")
```

---

# 📋 BẢNG TÓM TẮT

| Phase | Bước | File | Hàm | Mô tả |
|:---:|:---:|:---|:---|:---|
| 1 | 1.1 | `HairSwapper.jsx` | `handleGenerate()` | User upload ảnh |
| 1 | 1.2 | `main.py` | `generate_hair()` | API nhận + lưu file |
| 2 | 2.1 | `tasks.py` | `get_services()` | Load AI models |
| 2 | 2.1.1 | `face.py` | `FaceInfoService.__init__()` | Load InsightFace |
| 2 | 2.1.2 | `mask.py` | `SegmentationService.__init__()` | Load SegFormer |
| 2 | 2.1.3 | `diffusion.py` | `_load_sdxl_pipeline()` | Load SDXL |
| 3 | 3.1 | `tasks.py` | `process_hair_transfer()` | Load ảnh |
| 3 | 3.2 | `face.py` | `analyze()` | Detect face |
| 3 | 3.3 | `mask.py` | `get_mask()` | Tạo hair mask |
| 3 | 3.4 | `tasks.py` | `depth_estimator()` | Tạo depth map |
| 3 | 3.5 | `diffusion.py` | `generate()` | Sinh ảnh SDXL |
| 4 | 4.1 | `tasks.py` | `process_hair_transfer()` | Lưu output |
| 4 | 4.2 | `main.py` | `get_task_status()` | Trả về URL |
