#!/bin/bash

# ==============================================================================
# PIPELINE HUẤN LUYỆN: DEEP TEXTURE HAIR INPAINTING MODEL
# Kịch bản thực thi toàn bộ 4 Giai đoạn tự động
# ==============================================================================

# Dừng Script ngay lập tức nếu có bất kỳ lệnh nào bị lỗi (Exit Code > 0)
set -e

# Khai báo đường dẫn gốc
PROJECT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/../.. && pwd)
VENV_ACTIVATE="${PROJECT_DIR}/venv_wsl/bin/activate"

echo "=========================================================="
echo " KHỞI ĐỘNG PIPELINE HUẤN LUYỆN Y AI (THỬ NGHIỆM)"
echo " PROJECT ROOT: ${PROJECT_DIR}"
echo "=========================================================="

# Kiểm tra Virtual Environment
if [ -f "$VENV_ACTIVATE" ]; then
    echo "[!] Đang kích hoạt môi trường WSL Python..."
    source "$VENV_ACTIVATE"
else
    echo "[LỖI] Không tìm thấy môi trường venv_wsl tại: $VENV_ACTIVATE"
    echo "Vui lòng 'python -m venv venv_wsl' trước khi chạy script này."
    exit 1
fi

echo ""
echo "----------------------------------------------------------"
echo " STAGE 0: CHUẨN BỊ DỮ LIỆU ĐẦU VÀO (PREPARE DATASET)"
echo "----------------------------------------------------------"
# Lệnh này sẽ quét K-Hairstyle, bóc tách Polygon và nạp vào thư mục processed/
python "${PROJECT_DIR}/backend/training/prepare_dataset_deephair.py"

echo ""
echo "----------------------------------------------------------"
echo " STAGE 1: HỌC CHẤT LIỆU TÓC (TEXTURE ENCODER)"
echo "----------------------------------------------------------"
# Lệnh này chạy mạng Autoencoder để thấu hiểu không gian sợi tóc bằng SupCon Loss
python "${PROJECT_DIR}/backend/training/models/texture_encoder.py"

echo ""
echo "----------------------------------------------------------"
echo " STAGE 2: MASK-CONDITIONED INPAINTING (UNET MAIN MODEL)"
echo "----------------------------------------------------------"
# Lệnh này gọi mạng SDXL nới rộng 9-Channels, áp dụng IdentityLoss để khóa mặt
python "${PROJECT_DIR}/backend/training/train_stage2.py"

echo ""
echo "----------------------------------------------------------"
echo " STAGE 3: CHẤM ĐIỂM & PUBLIC TRỌNG SỐ (EVALUATION - EXPORT)"
echo "----------------------------------------------------------"
# Lệnh này chấm điểm Checkpoint sinh ra bằng công cụ tính Toán LPIPS (evaluate.py bên trong)
python "${PROJECT_DIR}/backend/training/export_model.py"

echo ""
echo "=========================================================="
echo " ĐÃ HOÀN THÀNH TOÀN BỘ PIPELINE HUẤN LUYỆN!"
echo " Checkpoint tốt nhất hiện có mặt tại: backend/training/models/"
echo "=========================================================="
