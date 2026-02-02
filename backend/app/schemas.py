from pydantic import BaseModel
from typing import Optional, List

class GenerateRequest(BaseModel):
    target_image_base64: str
    reference_image_base64: str
    description: Optional[str] = "a hairstyle transfer"

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str = "Task queued successfully"

class StatusResponse(BaseModel):
    task_id: str
    status: str
    result_url: Optional[str] = None
    error: Optional[str] = None
