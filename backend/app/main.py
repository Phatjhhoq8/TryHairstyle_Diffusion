
import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from backend.app.config import settings, OUTPUT_DIR, BACKEND_DIR
from backend.app.schemas import HairTransferResponse, TaskStatusResponse
from backend.app.tasks import process_hair_transfer, celery_app
from celery.result import AsyncResult

app = FastAPI(title="TryHairStyle API")

# Directories
UPLOAD_DIR = BACKEND_DIR / "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount Static Files (to serve results)
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

@app.post("/transfer", response_model=HairTransferResponse, tags=["Core"])
async def transfer_hair(user_img: str, hair_img: str, prompt: str = "high quality realistic hair"):
    """
    Gửi task thay đổi kiểu tóc.
    Input: filename của ảnh đã upload.
    """
    user_path = str(UPLOAD_DIR / user_img)
    hair_path = str(UPLOAD_DIR / hair_img)
    
    if not os.path.exists(user_path) or not os.path.exists(hair_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    # Push to Celery
    task = process_hair_transfer.delay(user_path, hair_path, prompt)
    
    return {
        "task_id": task.id,
        "status": "QUEUED",
        "message": "Task submitted successfully"
    }

@app.get("/task/{task_id}", response_model=TaskStatusResponse, tags=["Core"])
async def get_task_status(task_id: str):
    """
    Kiểm tra trạng thái task.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
    }
    
    if task_result.status == 'SUCCESS':
        result_data = task_result.result
        if result_data.get("status") == "SUCCESS":
            response["result_url"] = result_data.get("url")
        else:
            response["status"] = "FAILURE"
            response["error"] = result_data.get("error")
            
    elif task_result.status == 'FAILURE':
        response["error"] = str(task_result.result)
        
    return response

@app.get("/", tags=["General"])
async def root():
    return {"message": "TryHairStyle API is running with Celery + Redis + SDXL"}
