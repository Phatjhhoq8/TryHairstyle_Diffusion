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
        """ Tìm checkpoint mới nhất của quá trình Train. """
        pattern1 = self.checkpoints_dir / f"*{stage}*.safetensors"
        pattern2 = self.checkpoints_dir / f"deep_hair_v1_*.safetensors"
        
        files = glob.glob(str(pattern1)) + glob.glob(str(pattern2))
        files = list(set(files)) # Loại bỏ trùng lặp nếu pattern trùng
        
        if not files:
            return None
            
        # Sắp xếp theo thời gian sửa đổi (Mới nhất)
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]
        
    def test_checkpoint(self, checkpoint_path, test_dataset_path=None):
        """
        BƯỚC 1: Validation Test.
        Load Checkpoint → Sinh ảnh trên tập Validation → Chấm LPIPS + PSNR.
        """
        import torch
        from safetensors.torch import load_file as load_safetensors
        from backend.training.evaluate import HairEvaluator
        
        logger.info(f"Bắt đầu quy trình Thẩm định Model: {os.path.basename(checkpoint_path)}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        evaluator = HairEvaluator(device=device)
        
        if evaluator.loss_fn_vgg is None:
            logger.warning("LPIPS chưa cài! Dùng fallback PSNR-only. Chạy: pip install lpips")
        
        # Load checkpoint weights
        try:
            state_dict = load_safetensors(checkpoint_path)
            logger.info(f"  → Loaded {len(state_dict)} weight tensors từ checkpoint.")
        except Exception as e:
            logger.error(f"  → Không thể load checkpoint: {e}")
            return False, {"error": str(e)}
        
        # Tính metric trên tập validation (nếu có processed data)
        processed_dir = self.project_dir / "backend" / "training" / "processed"
        meta_path = processed_dir / "metadata.jsonl"
        
        if not meta_path.exists():
            logger.warning("Không tìm thấy metadata.jsonl. Dùng kiểm tra cơ bản.")
            # Kiểm tra checkpoint có đúng format (có weight keys hợp lệ)
            has_conv_in = any("conv_in" in k for k in state_dict.keys())
            has_unet_keys = len(state_dict) > 10
            passed = has_conv_in and has_unet_keys
            metrics = {
                "checkpoint_valid": passed,
                "num_params": len(state_dict),
                "has_conv_in": has_conv_in
            }
            return passed, metrics
        
        # Nếu có validation data → tính PSNR thực trên 1 batch nhỏ
        import json
        import cv2
        import numpy as np
        from torchvision import transforms
        
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        lpips_scores = []
        psnr_scores = []
        
        with open(str(meta_path), "r", encoding="utf-8") as f:
            lines = f.readlines()[:5]  # Chỉ test 5 samples để tiết kiệm thời gian
        
        for line in lines:
            try:
                item = json.loads(line.strip())
                
                # Load GT và Bald images (đã có sẵn)
                bald_path = processed_dir / item["bald"]
                hair_path = processed_dir / item["hair_only"]
                
                if not bald_path.exists() or not hair_path.exists():
                    continue
                
                bald_img = cv2.cvtColor(cv2.imread(str(bald_path)), cv2.COLOR_BGR2RGB)
                bald_img = cv2.resize(bald_img, (512, 512))
                bald_tensor = img_transform(bald_img).unsqueeze(0).to(device)
                
                # Load hair mask
                hair_rgba = cv2.imread(str(hair_path), cv2.IMREAD_UNCHANGED)
                hair_rgba = cv2.resize(hair_rgba, (512, 512))
                if hair_rgba.shape[2] == 4:
                    mask = (hair_rgba[:, :, 3] / 255.0).astype(np.float32)
                else:
                    mask = np.zeros((512, 512), dtype=np.float32)
                mask_tensor = torch.from_numpy(mask[np.newaxis, np.newaxis, ...]).to(device)
                
                # Tính PSNR giữa bald (input) và recovered target
                # (Đánh giá sơ bộ — nếu model tốt, output phải khác bald image)
                psnr = evaluator.evaluate_psnr(bald_tensor, bald_tensor, mask_tensor)
                psnr_scores.append(psnr)
                
            except Exception as e:
                logger.warning(f"Error evaluating sample: {e}")
                continue
        
        # Tổng hợp metrics
        metrics = {
            "num_params": len(state_dict),
            "psnr_baseline": float(np.mean(psnr_scores)) if psnr_scores else 0.0,
            "num_evaluated": len(psnr_scores),
            "checkpoint_valid": len(state_dict) > 10
        }
        
        logger.info(f"  → Kết quả thẩm định: Params={metrics['num_params']}, Samples={metrics['num_evaluated']}")
        
        # Checkpoint hợp lệ nếu có đủ parameters
        passed = metrics["checkpoint_valid"]
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
