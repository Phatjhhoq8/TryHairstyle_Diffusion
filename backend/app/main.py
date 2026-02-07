
import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.app.config import settings, OUTPUT_DIR, UPLOAD_DIR, BACKEND_DIR
from backend.app.schemas import HairTransferResponse, TaskStatusResponse
from backend.app.tasks import process_hair_transfer, celery_app
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
# General mount for other backend files (like default uploads/dataset if in backend)
app.mount("/static", StaticFiles(directory=BACKEND_DIR), name="static")

@app.post("/upload", tags=["Upload"])
async def upload_image(file: UploadFile = File(...)):
    """
    Upload ảnh (người dùng hoặc mẫu tóc) lên server.
    """
    file_ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = UPLOAD_DIR / filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"filename": filename, "path": str(file_path)}

@app.post("/generate", response_model=HairTransferResponse, tags=["Core"])
async def generate_hair(
    face_image: UploadFile = File(...),
    hair_image: UploadFile = File(...),
    description: str = Form("high quality realistic hair"),
    use_refiner: bool = Form(False)
):
    """
    Endpoint tương thích với React Frontend.
    Nhận 2 file ảnh và prompt, tự động upload rồi gửi task.
    """
    # 1. Save uploaded files
    face_filename = f"{uuid.uuid4()}_face.{face_image.filename.split('.')[-1]}"
    hair_filename = f"{uuid.uuid4()}_hair.{hair_image.filename.split('.')[-1]}"
    
    face_path = UPLOAD_DIR / face_filename
    hair_path = UPLOAD_DIR / hair_filename
    
    with open(face_path, "wb") as f:
        shutil.copyfileobj(face_image.file, f)
        
    with open(hair_path, "wb") as f:
        shutil.copyfileobj(hair_image.file, f)
        
    # 2. Trigger Celery Task
    task = process_hair_transfer.delay(str(face_path), str(hair_path), description, use_refiner)
    
    return {
        "task_id": task.id,
        "status": "QUEUED",
        "message": "Task started successfully"
    }

@app.post("/transfer", response_model=HairTransferResponse, tags=["Core"])
async def transfer_hair(user_img: str, hair_img: str, prompt: str = "high quality realistic hair", use_refiner: bool = False):
    """
    API cũ (giữ lại để tương thích CLI cũ).
    """
    user_path = str(UPLOAD_DIR / user_img)
    hair_path = str(UPLOAD_DIR / hair_img)
    
    if not os.path.exists(user_path) or not os.path.exists(hair_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    task = process_hair_transfer.delay(user_path, hair_path, prompt, use_refiner)
    
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
    import random
    from backend.app.config import BASE_DIR
    
    # Path to dataset (Adjust based on actual structure)
    # Expected: backend/data/dataset/ffhq
    dataset_rel_path = "data/dataset/ffhq"
    dataset_full_path = BACKEND_DIR / "data" / "dataset" / "ffhq"
    
    if not dataset_full_path.exists():
        return {"error": "Dataset not found"}
        
    # Get all subfolders
    try:
        subfolders = [f for f in dataset_full_path.iterdir() if f.is_dir()]
        if not subfolders:
             # Maybe flat structure?
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
                # Convert to static URL: /static/data/dataset/ffhq/...
                # BACKEND_DIR is mounted at /static
                # img path: .../backend/data/dataset/ffhq/00000/12345.png
                # rel path to backend: data/dataset/ffhq/00000/12345.png
                rel_path = img.relative_to(BACKEND_DIR)
                return f"/static/{rel_path.as_posix()}"
        return None

    target_url = pick_random_image()
    hair_url = pick_random_image()
    
    return {
        "target_url": target_url,
        "hair_url": hair_url
    }

@app.get("/", tags=["General"])
async def root():
    return {"message": "TryHairStyle API is running with CORS Enabled"}
