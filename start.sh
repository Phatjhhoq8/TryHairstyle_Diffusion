#!/bin/bash
# ============================================================
# TryHairStyle — Khởi động tất cả services
# Chạy: wsl -e bash start.sh  (từ thư mục project)
# ============================================================

PROJECT_DIR="/mnt/c/Users/Admin/Desktop/TryHairStyle"
VENV="$PROJECT_DIR/venv_wsl/bin/activate"

cd "$PROJECT_DIR"
source "$VENV"
export PYTHONPATH="$PROJECT_DIR"
export REDIS_URL="redis://localhost:6379/0"

# CUDA/cuDNN — cần thiết cho WSL2 (libcuda.so) và PyTorch trong venv TryOnHairstyle
TRYON_SITE="$PROJECT_DIR/TryOnHairstyle-master/hairfusion/lib/python3.8/site-packages"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$TRYON_SITE/torch/lib:$TRYON_SITE/torchvision.libs:${LD_LIBRARY_PATH:-}"

echo "============================================"
echo "  TryHairStyle — Starting All Services"
echo "============================================"

# 1. Redis
echo "[1/4] Starting Redis..."
redis-server --daemonize yes 2>/dev/null
if redis-cli ping | grep -q PONG; then
    echo "  ✅ Redis: OK (port 6379)"
else
    echo "  ❌ Redis: Failed to start!"
    exit 1
fi

# 2. Backend FastAPI
echo "[2/4] Starting Backend (FastAPI)..."
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "  ✅ Backend: PID=$BACKEND_PID (port 8000)"

# 3. Celery Worker
echo "[3/4] Starting Celery Worker..."
celery -A backend.app.tasks.celery_app worker --loglevel=info --pool=solo &
CELERY_PID=$!
echo "  ✅ Celery: PID=$CELERY_PID"

# 4. Frontend (chạy trên Windows nên chỉ nhắc)
echo "[4/4] Frontend: Chạy riêng trên Windows terminal:"
echo "       cd frontend && npm run dev"

echo ""
echo "============================================"
echo "  All services started!"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173 (chạy riêng)"
echo ""
echo "  Nhấn Ctrl+C để dừng tất cả"
echo "============================================"

# Trap Ctrl+C → dừng tất cả
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $CELERY_PID 2>/dev/null
    redis-cli shutdown 2>/dev/null
    echo "All services stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Chờ (giữ script sống)
wait
