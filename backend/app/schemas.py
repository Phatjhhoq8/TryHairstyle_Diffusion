
from pydantic import BaseModel
from typing import Optional

class HairTransferRequest(BaseModel):
    user_image_path: str # Path to uploaded user image on server
    hair_image_path: str # Path to uploaded/selected hair image
    prompt: Optional[str] = "best quality, realistic hair, 8k"
    
class HairTransferResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str # PENDING, PROCESSING, SUCCESS, FAILURE
    result_url: Optional[str] = None
    error: Optional[str] = None
