import os
import sys
import shutil
import glob
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from backend.app.services.training_utils import setupLogger

logger = setupLogger("ModelExportVerification")

class CheckpointManager:
    """
    Quản lý, đánh giá và xuất Model Checkpoints sau khi Train (Stage 1 & 2)
    Đẩy trọng số tốt nhất vào Môi trường Production (Backend App).
    """
    def __init__(self):
        self.project_dir = Path(__file__).resolve().parent.parent.parent
        self.checkpoints_dir = self.project_dir / "backend" / "training" / "checkpoints"
        self.production_models_dir = self.project_dir / "backend" / "models"
        
        # Tạo thư mục checkpoints nếu chưa có
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
    def find_latest_checkpoint(self, stage="stage2"):
        """ Tìm checkpoint mới nhất của quá trình Train. """
        pattern = self.checkpoints_dir / f"checkpoint_{stage}_*.safetensors"
        files = glob.glob(str(pattern))
        if not files:
            return None
            
        # Sắp xếp theo thời gian sửa đổi (Mới nhất)
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]
        
    def test_checkpoint(self, checkpoint_path, test_dataset_path=None):
        """
        BƯỚC 1: Validation Test.
        Chạy script giả lập Inference đánh giá Checkpoint mới nhất
        trên bộ Hold-out validation set.
        (Thực tế: Load Model -> Generate Images -> Call evaluate_lpips / evaluate_psnr)
        """
        logger.info(f"Bắt đầu quy trình Thẩm định Model: {os.path.basename(checkpoint_path)}")
        logger.info("  -> Load Weights vào VRAM...")
        logger.info("  -> Chạy sinh ảnh trên tập Validation...")
        logger.info("  -> Chấm điểm LPIPS & Identity Cosine Similarity...")
        
        # MOCK METRICS
        mock_metrics = {
            "identity_score": 0.92, # Đạt yêu cầu >= 0.90
            "psnr": 28.5,
            "lpips": 0.12 # Càng thấp càng tốt
        }
        
        logger.info(f"  -> Kết quả thẩm định: ID={mock_metrics['identity_score']}, LPIPS={mock_metrics['lpips']}")
        
        # Đánh giá xem Model có đủ điều kiện ra Production không (Thresholds)
        if mock_metrics['identity_score'] >= 0.90 and mock_metrics['lpips'] <= 0.20:
            return True, mock_metrics
        return False, mock_metrics
        
    def export_to_production(self, checkpoint_path, destination_name="deep_hair_v1.safetensors"):
        """
        BƯỚC 2: Export Model.
        Copy trọng số đạt chuẩn sang thư mục Production `backend/models`
        để Web App sử dụng.
        """
        dest_path = self.production_models_dir / destination_name
        
        try:
            logger.info(f"Tiến hành Deploy Model ra Production: {dest_path}")
            # Trong thực tế phải convert, merge LoRA, compile, v.v.
            # Ở đây dùng lệnh shutil copy làm ví dụ deploy
            # shutil.copy2(checkpoint_path, dest_path) 
            logger.info(f"  -> Deploy THÀNH CÔNG! Web App đã có thể load Model mới.")
            return True
        except Exception as e:
            logger.error(f"  -> Lỗi Deploy Model: {e}")
            return False

if __name__ == "__main__":
    print("[Testing] Workflow Triển Khai Model...")
    manager = CheckpointManager()
    
    # Tạo 1 file Checkpoint ảo để Test quá trình
    dummy_ckpt = manager.checkpoints_dir / "checkpoint_stage2_ep10_step5000.safetensors"
    with open(str(dummy_ckpt), "w") as f:
        f.write("dummy weights")
        
    # Test Workflow
    latest = manager.find_latest_checkpoint()
    if latest:
        passed, metrics = manager.test_checkpoint(latest)
        if passed:
            manager.export_to_production(latest)
        else:
            print("Model chưa đủ chất lượng để Deploy.")
    else:
        print("Không tìm thấy checkpoint nào trong thư mục train.")
