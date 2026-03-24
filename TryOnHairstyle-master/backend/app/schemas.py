"""
Pydantic schemas for HairFusion API.
"""
from pydantic import BaseModel
from typing import Optional


class GenerateRequest(BaseModel):
    """Request model for hair generation."""
    steps: int = 50
    scale: float = 5.0
    prompt: str = ""


class GenerateResponse(BaseModel):
    """Response model for generation result."""
    status: str
    message: str
    result_path: Optional[str] = None


class StatusResponse(BaseModel):
    """Response model for task status."""
    task_id: str
    status: str
    result_url: Optional[str] = None
