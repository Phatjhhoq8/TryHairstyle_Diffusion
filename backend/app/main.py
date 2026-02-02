from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os

from .api import router

app = FastAPI(title="TryHairStyle API", version="1.0.0")

# CORS Configuration
origins = [
    "http://localhost:5173", # Vite Frontend
    "http://127.0.0.1:5173",
    "*" # For dev, allow all
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for serving results
# backend/data/results -> /static/results
RESULT_DIR = Path("backend/data/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static/results", StaticFiles(directory=RESULT_DIR), name="results")

# Include Router
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
