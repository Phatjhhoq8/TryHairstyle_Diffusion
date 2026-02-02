from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uuid
import shutil
from pathlib import Path
import base64
import os

from .schemas import GenerateRequest, TaskResponse, StatusResponse
from .tasks import generate_hairstyle_task, celery_app

router = APIRouter()

# Directories for temp storage
UPLOAD_DIR = Path("backend/data/uploads")
RESULT_DIR = Path("backend/data/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

@router.get("/")
def health_check():
    return {"status": "ok", "message": "TryHairStyle API is running"}

@router.post("/generate", response_model=TaskResponse)
async def generate_hairstyle_endpoint(
    target_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    description: str = Form("a hairstyle transfer")
):
    """
    Upload target and reference images to start a background generation task.
    """
    task_id = str(uuid.uuid4())
    
    # Save uploaded files
    target_ext = target_image.filename.split('.')[-1]
    reference_ext = reference_image.filename.split('.')[-1]
    
    target_path = UPLOAD_DIR / f"{task_id}_target.{target_ext}"
    reference_path = UPLOAD_DIR / f"{task_id}_reference.{reference_ext}"
    output_path = RESULT_DIR / f"{task_id}_result.png"

    try:
        with open(target_path, "wb") as f:
            shutil.copyfileobj(target_image.file, f)
        
        with open(reference_path, "wb") as f:
            shutil.copyfileobj(reference_image.file, f)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save images: {str(e)}")

    # Enqueue Celery Task
    # We pass absolute paths or relative paths that the worker can access
    # Since worker and API share the filesystem in this setup (WSL), relative paths from project root are fine
    # But let's use string paths.
    
    # task = generate_hairstyle_task.delay(str(target_path), str(reference_path), str(output_path))
    # We should update the task signature if we want to use the description?
    # For now, let's keep the task signature simple or update it too if needed.
    # Let's update the tasks.py to accept description as well to be consistent.
    task = generate_hairstyle_task.delay(str(target_path), str(reference_path), str(output_path), description)
    
    return TaskResponse(
        task_id=task.id,
        status="queued",
        message="Task submitted successfully"
    )

@router.get("/status/{task_id}", response_model=StatusResponse)
def get_task_status(task_id: str):
    """
    Check the status of a generation task.
    """
    task_result = celery_app.AsyncResult(task_id)
    
    response = StatusResponse(task_id=task_id, status=task_result.status)

    if task_result.status == 'SUCCESS':
        result_data = task_result.result
        response.result_url = result_data.get("result_url")
    elif task_result.status == 'FAILURE':
        response.error = str(task_result.result)
    
    return response
