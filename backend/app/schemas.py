
from pydantic import BaseModel
from typing import Optional

class HairTransferRequest(BaseModel):
    user_image_path: str # Path to uploaded user image on server
    hair_image_path: str # Path to uploaded/selected hair image
    prompt: Optional[str] = "best quality, realistic hair, 8k"
    hair_color: Optional[str] = None  # Tên màu preset hoặc hex code (vd: 'blonde', '#FF0000')
    color_intensity: Optional[float] = 0.7  # Mức độ đậm/nhạt (0.0–1.0)
    
class HairTransferResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str # PENDING, PROCESSING, SUCCESS, FAILURE
    result_url: Optional[str] = None
    error: Optional[str] = None

class HairColorRequest(BaseModel):
    """Schema cho endpoint /colorize — chỉ đổi màu tóc (không thay kiểu)"""
    hair_color: str  # Tên preset hoặc hex code
    intensity: Optional[float] = 0.7

class HairColorResponse(BaseModel):
    task_id: str
    status: str
    message: str
