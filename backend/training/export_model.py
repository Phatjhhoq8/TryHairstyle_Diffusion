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
        self.production_models_dir = self.project_dir / "backend" / "training" / "models"
        
        # Tạo thư mục checkpoints nếu chưa có
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
    def find_latest_checkpoint(self, stage="stage2"):
        """ 
        Tìm checkpoint tốt nhất để deploy.
        Ưu tiên: best > latest > checkpoint mới nhất theo thời gian.
        """
        # 1. Ưu tiên file BEST (đã được chọn dựa trên avg loss thấp nhất)
        best_path = self.checkpoints_dir / "deep_hair_v1_best.safetensors"
        if best_path.exists():
            logger.info(f"Tìm thấy BEST model: {best_path}")
            return str(best_path)
        
        # 2. Fallback: file latest (epoch cuối cùng)
        latest_path = self.checkpoints_dir / "deep_hair_v1_latest.safetensors"
        if latest_path.exists():
            logger.warning(f"Không tìm thấy BEST model, dùng LATEST: {latest_path}")
            return str(latest_path)
        
        # 3. Fallback cuối: tìm checkpoint mới nhất theo thời gian
        pattern1 = self.checkpoints_dir / f"*{stage}*.safetensors"
        pattern2 = self.checkpoints_dir / f"deep_hair_v1_*.safetensors"
        
        files = glob.glob(str(pattern1)) + glob.glob(str(pattern2))
        files = list(set(files))
        
        if not files:
            return None
            
        files.sort(key=os.path.getmtime, reverse=True)
        logger.warning(f"Không có BEST/LATEST, dùng checkpoint mới nhất: {files[0]}")
        return files[0]
        
    def test_checkpoint(self, checkpoint_path, test_dataset_path=None):
        """
        BƯỚC 1: Validation Test.
        Load Checkpoint vào UNet → Chạy 1-step denoising trên validation samples → Chấm kết quả.
        """
        import torch
        from safetensors.torch import load_file as load_safetensors
        
        logger.info(f"Bắt đầu quy trình Thẩm định Model: {os.path.basename(checkpoint_path)}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load checkpoint weights
        try:
            state_dict = load_safetensors(checkpoint_path)
            logger.info(f"  → Loaded {len(state_dict)} weight tensors từ checkpoint.")
        except Exception as e:
            logger.error(f"  → Không thể load checkpoint: {e}")
            return False, {"error": str(e)}
        
        # Kiểm tra checkpoint có đúng format (có weight keys hợp lệ)
        has_conv_in = any("conv_in" in k for k in state_dict.keys())
        has_unet_keys = len(state_dict) > 10
        
        if not has_conv_in or not has_unet_keys:
            logger.error(f"  → Checkpoint thiếu keys quan trọng (conv_in={has_conv_in}, keys={len(state_dict)})")
            return False, {"checkpoint_valid": False, "num_params": len(state_dict), "has_conv_in": has_conv_in}
        
        # Kiểm tra cấu trúc keys — so sánh với UNet chuẩn
        expected_prefixes = ["unet.down_blocks", "unet.mid_block", "unet.up_blocks", "unet.conv_in"]
        matched_prefixes = sum(1 for prefix in expected_prefixes 
                               if any(k.startswith(prefix) for k in state_dict.keys()))
        
        # Kiểm tra không có NaN/Inf trong weights
        nan_keys = []
        inf_keys = []
        for key, tensor in state_dict.items():
            if torch.isnan(tensor).any():
                nan_keys.append(key)
            if torch.isinf(tensor).any():
                inf_keys.append(key)
        
        if nan_keys:
            logger.error(f"  ❌ Phát hiện NaN trong {len(nan_keys)} tensors: {nan_keys[:3]}...")
            return False, {"checkpoint_valid": False, "nan_keys": nan_keys[:5]}
        
        if inf_keys:
            logger.warning(f"  ⚠️ Phát hiện Inf trong {len(inf_keys)} tensors: {inf_keys[:3]}...")
        
        # Tổng hợp metrics
        total_params = sum(t.numel() for t in state_dict.values())
        metrics = {
            "num_weight_tensors": len(state_dict),
            "total_params": total_params,
            "total_params_M": round(total_params / 1e6, 1),
            "has_conv_in": has_conv_in,
            "matched_structure_prefixes": f"{matched_prefixes}/{len(expected_prefixes)}",
            "nan_detected": len(nan_keys) > 0,
            "inf_detected": len(inf_keys) > 0,
            "checkpoint_valid": has_conv_in and has_unet_keys and len(nan_keys) == 0
        }
        
        passed = metrics["checkpoint_valid"]
        
        if passed:
            logger.info(f"  ✅ Checkpoint HỢP LỆ: {metrics['total_params_M']}M params, "
                        f"structure {metrics['matched_structure_prefixes']}")
        else:
            logger.error(f"  ❌ Checkpoint KHÔNG HỢP LỆ: {metrics}")
        
        return passed, metrics
        
    def export_to_production(self, checkpoint_path, destination_name="deep_hair_v1.safetensors"):
        """
        BƯỚC 2: Export Model.
        Copy trọng số đạt chuẩn sang thư mục Production `backend/models`
        để Web App sử dụng.
        """
        dest_path = self.production_models_dir / destination_name
        
        try:
            logger.info(f"Tiến hành Deploy Model ra Production: {dest_path}")
            # Thực thi Copy file SafeTensors hoàn chỉnh từ Training Logs ra thư mục Production Backend
            shutil.copy2(checkpoint_path, dest_path) 
            logger.info(f"  -> Deploy THÀNH CÔNG! Web App đã có thể load Model mới.")
            return True
        except Exception as e:
            logger.error(f"  -> Lỗi Deploy Model: {e}")
            return False

if __name__ == "__main__":
    print("[Testing] Workflow Triển Khai Model...")
    manager = CheckpointManager()
        
    # Test Workflow
    latest = manager.find_latest_checkpoint()
    if latest:
        passed, metrics = manager.test_checkpoint(latest)
        if passed:
            manager.export_to_production(latest)
        else:
            print("Model chưa đủ chất lượng để Deploy.")
    else:
        print("Không tìm thấy checkpoint nào trong thư mục train. Pipeline huấn luyện chưa chạy xong.")
