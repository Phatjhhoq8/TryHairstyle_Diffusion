
import os
import shutil
import uuid
import random
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.app.config import settings, OUTPUT_DIR, UPLOAD_DIR, BACKEND_DIR
from pathlib import Path
from backend.app.schemas import HairTransferResponse, TaskStatusResponse, HairColorResponse, DetectFacesResponse
from backend.app.tasks import process_hair_transfer, process_hair_colorize, process_detect_faces, celery_app
from backend.app.services.hair_color_service import HairColorService
from backend.app.services.translate_service import translate_vi_to_en
from celery.result import AsyncResult

app = FastAPI(title="TryHairStyle API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories (Now managed in config.py)

# Mount Static Files (to serve results)
# Specific mount for Output (data/output)
app.mount("/static/output", StaticFiles(directory=OUTPUT_DIR), name="static_output")
# Specific mount for Uploads (data/uploads) - Needed if we want to serve uploaded images back
app.mount("/static/uploads", StaticFiles(directory=UPLOAD_DIR), name="static_uploads")

# Mount dataset riêng cho /random-pair (CHỈ dataset, KHÔNG expose toàn bộ backend)
DATASET_DIR = BACKEND_DIR / "data" / "dataset"
if DATASET_DIR.exists():
    app.mount("/static/dataset", StaticFiles(directory=str(DATASET_DIR)), name="static_dataset")

# BẢO MẬT: KHÔNG mount toàn bộ BACKEND_DIR — sẽ expose .env, config.py, source code

@app.post("/upload", tags=["Upload"])
async def upload_image(file: UploadFile = File(...)):
    """
    Upload ảnh (người dùng hoặc mẫu tóc) lên server.
    """
    file_ext = file.filename.split(".")[-1].lower()
    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = UPLOAD_DIR / filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"filename": filename}

@app.post("/generate", response_model=HairTransferResponse, tags=["Core"])
async def generate_hair(
    face_image: UploadFile = File(...),
    hair_image: UploadFile = File(...),
    original_face_image: UploadFile = File(None),
    bbox: str = Form(None),
    description: str = Form("high quality realistic hair"),
    hair_color: str = Form(None),
    color_intensity: float = Form(0.7),
    language: str = Form("en"),
    ai_model: str = Form("HairFusion")
):
    """
    Endpoint tương thích với React Frontend.
    Nhận 2 file ảnh và prompt, tự động upload rồi gửi task.
    Hỗ trợ thêm hair_color (tên preset hoặc hex) và color_intensity (0.0–1.0).
    Nếu language="vi", prompt sẽ được dịch sang tiếng Anh trước khi xử lý.
    """
    # Validate file type
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
    face_ext = face_image.filename.split('.')[-1].lower()
    hair_ext = hair_image.filename.split('.')[-1].lower()
    if face_ext not in ALLOWED_EXTENSIONS or hair_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}")
    
    # Dịch prompt nếu ngôn ngữ là tiếng Việt
    final_description = description
    if language == "vi":
        final_description = translate_vi_to_en(description)
    
    # 1. Save uploaded files
    face_filename = f"{uuid.uuid4()}_face.{face_ext}"
    hair_filename = f"{uuid.uuid4()}_hair.{hair_ext}"
    
    face_path = UPLOAD_DIR / face_filename
    hair_path = UPLOAD_DIR / hair_filename
    
    with open(face_path, "wb") as f:
        shutil.copyfileobj(face_image.file, f)
        
    with open(hair_path, "wb") as f:
        shutil.copyfileobj(hair_image.file, f)
        
    original_face_path = None
    if original_face_image:
        orig_ext = original_face_image.filename.split('.')[-1].lower()
        orig_filename = f"{uuid.uuid4()}_orig.{orig_ext}"
        original_face_path = UPLOAD_DIR / orig_filename
        with open(original_face_path, "wb") as f:
            shutil.copyfileobj(original_face_image.file, f)
            
    import json
    parsed_bbox = None
    if bbox:
        try:
            parsed_bbox = json.loads(bbox)
        except Exception as e:
            print(f"Error parsing bbox: {bbox}, {e}")
            pass
        
    # 2. Trigger Celery Task (truyền thêm hair_color, original_face_path, bbox)
    task = process_hair_transfer.delay(
        str(face_path), str(hair_path), final_description,
        hair_color=hair_color, color_intensity=color_intensity,
        ai_model=ai_model,
        original_face_path=str(original_face_path) if original_face_path else None,
        bbox=parsed_bbox
    )
    
    return {
        "task_id": task.id,
        "status": "QUEUED",
        "message": "Task started successfully"
    }

@app.post("/detect-faces", response_model=DetectFacesResponse, tags=["Core"])
async def detect_faces_api(image: UploadFile = File(...)):
    """
    Quét và cắt chân dung toàn bộ khuôn mặt có trong một bức ảnh chung.
    Hoạt động qua Celery Task để chống treo server và cạn kiệt VRAM.
    """
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
    file_ext = image.filename.split('.')[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}")
    
    filename = f"{uuid.uuid4()}_detect.{file_ext}"
    file_path = UPLOAD_DIR / filename
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(image.file, f)
        
    task = process_detect_faces.delay(str(file_path))
    
    return {
        "task_id": task.id,
        "status": "QUEUED",
        "message": "Face detection task started"
    }

@app.post("/transfer", response_model=HairTransferResponse, tags=["Core"])
async def transfer_hair(user_img: str, hair_img: str, prompt: str = "high quality realistic hair"):
    """
    API cũ (giữ lại để tương thích CLI cũ).
    """
    user_path = str(UPLOAD_DIR / user_img)
    hair_path = str(UPLOAD_DIR / hair_img)
    
    if not os.path.exists(user_path) or not os.path.exists(hair_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    task = process_hair_transfer.delay(user_path, hair_path, prompt)
    
    return {
        "task_id": task.id,
        "status": "QUEUED",
        "message": "Task submitted successfully"
    }

@app.get("/status/{task_id}", tags=["Core"])
async def check_status_frontend(task_id: str):
    """
    Endpoint tương thích với React Frontend (/status/{id}).
    """
    return await get_task_status(task_id)

@app.get("/task/{task_id}", response_model=TaskStatusResponse, tags=["Core"])
async def get_task_status(task_id: str):
    """
    Kiểm tra trạng thái task (API gốc).
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
    }
    
    if task_result.status == 'SUCCESS':
        result_data = task_result.result
        # Handle case where result is just text or dict
        if isinstance(result_data, dict):
            if result_data.get("status") == "SUCCESS":
                response["result_url"] = result_data.get("url")
                if "faces" in result_data:
                    response["faces"] = result_data["faces"]
            else:
                response["status"] = "FAILURE"
                response["error"] = result_data.get("error")
        else:
             response["status"] = "FAILURE"
             response["error"] = str(result_data)
            
    elif task_result.status == 'FAILURE':
        response["error"] = str(task_result.result)
        
    return response

@app.get("/random-pair", tags=["General"])
async def get_random_pair():
    """
    Lấy 2 ảnh ngẫu nhiên từ dataset FFHQ để test.
    Trả về URL tĩnh của ảnh.
    """
    dataset_full_path = BACKEND_DIR / "data" / "dataset" / "ffhq"
    
    if not dataset_full_path.exists():
        return {"error": "Dataset not found"}
        
    # Get all subfolders
    try:
        subfolders = [f for f in dataset_full_path.iterdir() if f.is_dir()]
        if not subfolders:
             pass 
    except Exception as e:
        return {"error": str(e)}

    # Helper to pick random image
    def pick_random_image():
        # Try max 5 times to find a valid image
        for _ in range(5):
            folder = random.choice(subfolders) if subfolders else dataset_full_path
            
            if folder.is_dir(): # Should be true
                images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
            else:
                images = list(dataset_full_path.glob("*.png")) + list(dataset_full_path.glob("*.jpg"))
                
            if images:
                img = random.choice(images)
                return f"/static/dataset/{img.relative_to(DATASET_DIR).as_posix()}"
        return None

    target_url = pick_random_image()
    hair_url = pick_random_image()
    
    return {
        "target_url": target_url,
        "hair_url": hair_url
    }

@app.get("/colors", tags=["Hair Color"])
async def get_colors():
    """
    Trả về danh sách preset màu tóc.
    """
    return HairColorService.getPresetColors()

@app.post("/colorize", response_model=HairColorResponse, tags=["Hair Color"])
async def colorize_hair(
    face_image: UploadFile = File(...),
    hair_color: str = Form(...),
    intensity: float = Form(0.7)
):
    """
    Chỉ đổi màu tóc trên ảnh gốc (KHÔNG thay kiểu tóc).
    Nhanh hơn /generate vì không cần Diffusion Model.
    """
    # Validate file type
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
    face_ext = face_image.filename.split('.')[-1].lower()
    if face_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}")
    
    # Save uploaded file
    face_filename = f"{uuid.uuid4()}_face.{face_ext}"
    face_path = UPLOAD_DIR / face_filename
    with open(face_path, "wb") as f:
        shutil.copyfileobj(face_image.file, f)
    
    # Trigger Celery Task
    task = process_hair_colorize.delay(
        str(face_path), hair_color, intensity
    )
    
    return {
        "task_id": task.id,
        "status": "QUEUED",
        "message": "Hair color task started"
    }

@app.get("/", tags=["General"])
async def root():
    return {"message": "TryHairStyle API is running with CORS Enabled"}
