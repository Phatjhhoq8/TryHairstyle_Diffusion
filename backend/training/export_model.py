import os
import sys
import shutil
import glob
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from backend.app.services.training_utils import setupLogger

logger = setupLogger("ModelExportVerification")

class CheckpointManager:
    """
    Quản lý, đánh giá và xuất LoRA Checkpoints sau khi Train Stage 2.
    Hỗ trợ 2 workflow:
      1. Export LoRA weights riêng (nhẹ ~50MB) cho inference với peft
      2. Merge LoRA vào UNet gốc → export full model (cho production không cần peft)
    """
    def __init__(self):
        self.project_dir = Path(__file__).resolve().parent.parent.parent
        self.checkpoints_dir = self.project_dir / "backend" / "training" / "checkpoints"
        self.production_models_dir = self.project_dir / "backend" / "models"
        
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
    def find_latest_checkpoint(self, stage="stage2"):
        """ 
        Tìm LoRA checkpoint tốt nhất để deploy.
        Ưu tiên: lora_best > lora_latest > deep_hair_v1_best (legacy) > mới nhất.
        """
        # 1. LoRA best
        lora_best = self.checkpoints_dir / "lora_best.safetensors"
        if lora_best.exists():
            logger.info(f"Tìm thấy BEST LoRA model: {lora_best}")
            return str(lora_best)
        
        # 2. LoRA latest
        lora_latest = self.checkpoints_dir / "lora_latest.safetensors"
        if lora_latest.exists():
            logger.warning(f"Không tìm thấy BEST LoRA, dùng LATEST: {lora_latest}")
            return str(lora_latest)
        
        # 3. Legacy: full UNet checkpoint (backward compat)
        best_path = self.checkpoints_dir / "deep_hair_v1_best.safetensors"
        if best_path.exists():
            logger.warning(f"Tìm thấy legacy full UNet model: {best_path}")
            return str(best_path)
        
        latest_path = self.checkpoints_dir / "deep_hair_v1_latest.safetensors"
        if latest_path.exists():
            logger.warning(f"Dùng legacy LATEST: {latest_path}")
            return str(latest_path)
        
        # 4. Fallback: checkpoint mới nhất theo thời gian
        pattern1 = self.checkpoints_dir / f"lora_*.safetensors"
        pattern2 = self.checkpoints_dir / f"deep_hair_v1_*.safetensors"
        
        files = glob.glob(str(pattern1)) + glob.glob(str(pattern2))
        files = list(set(files))
        
        if not files:
            return None
            
        files.sort(key=os.path.getmtime, reverse=True)
        logger.warning(f"Không có BEST/LATEST, dùng checkpoint mới nhất: {files[0]}")
        return files[0]
        
    def _is_lora_checkpoint(self, checkpoint_path):
        """Kiểm tra xem checkpoint có phải LoRA format không (có lora keys + conv_in keys)."""
        from safetensors.torch import load_file as load_safetensors
        state_dict = load_safetensors(checkpoint_path)
        has_lora = any("lora" in k.lower() for k in state_dict.keys())
        has_conv_in = any("conv_in" in k for k in state_dict.keys())
        return has_lora and has_conv_in, state_dict
    
    def test_checkpoint(self, checkpoint_path, test_dataset_path=None):
        """
        BƯỚC 1: Validation Test.
        Kiểm tra checkpoint (LoRA hoặc full UNet) có hợp lệ không.
        """
        from safetensors.torch import load_file as load_safetensors
        
        logger.info(f"Bắt đầu Thẩm định Model: {os.path.basename(checkpoint_path)}")
        
        try:
            state_dict = load_safetensors(checkpoint_path)
            logger.info(f"  → Loaded {len(state_dict)} weight tensors từ checkpoint.")
        except Exception as e:
            logger.error(f"  → Không thể load checkpoint: {e}")
            return False, {"error": str(e)}
        
        # Kiểm tra loại checkpoint
        is_lora = any("lora" in k.lower() for k in state_dict.keys())
        has_conv_in = any("conv_in" in k for k in state_dict.keys())
        
        if is_lora:
            logger.info(f"  → Phát hiện LoRA checkpoint")
            # LoRA checkpoint: kiểm tra có đủ adapter keys
            lora_keys = [k for k in state_dict.keys() if "lora" in k.lower()]
            conv_in_keys = [k for k in state_dict.keys() if "conv_in" in k]
            
            if len(lora_keys) < 4:
                logger.error(f"  → LoRA checkpoint thiếu keys ({len(lora_keys)} lora keys)")
                return False, {"checkpoint_valid": False, "is_lora": True, "num_lora_keys": len(lora_keys)}
        else:
            # Full UNet checkpoint (legacy)
            logger.info(f"  → Phát hiện full UNet checkpoint (legacy)")
            if not has_conv_in or len(state_dict) < 10:
                logger.error(f"  → Checkpoint thiếu keys (conv_in={has_conv_in}, keys={len(state_dict)})")
                return False, {"checkpoint_valid": False, "is_lora": False}
        
        # Kiểm tra NaN/Inf
        nan_keys = [k for k, t in state_dict.items() if torch.isnan(t).any()]
        inf_keys = [k for k, t in state_dict.items() if torch.isinf(t).any()]
        
        if nan_keys:
            logger.error(f"  ❌ NaN trong {len(nan_keys)} tensors: {nan_keys[:3]}...")
            return False, {"checkpoint_valid": False, "nan_keys": nan_keys[:5]}
        
        if inf_keys:
            logger.warning(f"  ⚠️ Inf trong {len(inf_keys)} tensors: {inf_keys[:3]}...")
        
        total_params = sum(t.numel() for t in state_dict.values())
        metrics = {
            "num_weight_tensors": len(state_dict),
            "total_params": total_params,
            "total_params_M": round(total_params / 1e6, 1),
            "is_lora": is_lora,
            "has_conv_in": has_conv_in,
            "nan_detected": len(nan_keys) > 0,
            "inf_detected": len(inf_keys) > 0,
            "checkpoint_valid": len(nan_keys) == 0,
        }
        
        passed = metrics["checkpoint_valid"]
        ckpt_type = "LoRA" if is_lora else "Full UNet"
        
        if passed:
            logger.info(f"  ✅ {ckpt_type} Checkpoint HỢP LỆ: {metrics['total_params_M']}M params")
        else:
            logger.error(f"  ❌ Checkpoint KHÔNG HỢP LỆ: {metrics}")
        
        return passed, metrics
    
    def _find_matching_injector(self, checkpoint_path):
        """
        Tìm file Injector checkpoint tương ứng.
        Ưu tiên: best > backup > latest.
        """
        name = os.path.basename(checkpoint_path)
        
        candidates = []
        if "best" in name:
            candidates.append(self.checkpoints_dir / "injector_best.safetensors")
        if "latest" in name:
            candidates.append(self.checkpoints_dir / "injector_latest.safetensors")
        if "backup" in name:
            candidates.append(self.checkpoints_dir / "injector_backup.safetensors")
        
        # Fallback
        candidates.extend([
            self.checkpoints_dir / "injector_best.safetensors",
            self.checkpoints_dir / "injector_latest.safetensors",
            self.checkpoints_dir / "injector_backup.safetensors",
        ])
        
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def merge_lora_and_export(self, lora_path, destination_name="deep_hair_v1.safetensors"):
        """
        Merge LoRA weights vào UNet gốc → export full model cho production.
        Production sẽ KHÔNG cần peft library.
        """
        from safetensors.torch import load_file as load_safetensors, save_file
        from peft import set_peft_model_state_dict
        
        logger.info(f"Bắt đầu Merge LoRA → Full UNet cho Production...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. Load base UNet (SDXL Inpainting)
        from backend.training.models.stage2_unet import HairInpaintingUNet
        from peft import LoraConfig, get_peft_model
        
        logger.info("  → Loading base UNet...")
        unet = HairInpaintingUNet().to(device)
        
        # 2. Apply LoRA config (phải match với training config)
        lora_config = LoraConfig(
            r=16, lora_alpha=16,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,  # No dropout cho inference
            bias="none",
        )
        unet.unet = get_peft_model(unet.unet, lora_config)
        
        # 3. Load LoRA + conv_in weights
        state_dict = load_safetensors(lora_path)
        conv_in_state = {k[len("conv_in."):]: v for k, v in state_dict.items() if k.startswith("conv_in.")}
        lora_state = {k: v for k, v in state_dict.items() if not k.startswith("conv_in.")}
        
        set_peft_model_state_dict(unet.unet, lora_state)
        if conv_in_state:
            unet.unet.base_model.model.conv_in.load_state_dict(conv_in_state)
        logger.info(f"  → LoRA weights loaded từ {os.path.basename(lora_path)}")
        
        # 4. Merge LoRA vào base model (permanent)
        unet.unet = unet.unet.merge_and_unload()
        logger.info("  → LoRA merged vào UNet base model")
        
        # 5. Export full merged model
        dest_path = self.production_models_dir / destination_name
        os.makedirs(str(self.production_models_dir), exist_ok=True)
        
        merged_state = unet.state_dict()
        save_file(merged_state, str(dest_path))
        size_mb = os.path.getsize(str(dest_path)) / (1024 * 1024)
        logger.info(f"  → Merged model saved: {dest_path} ({size_mb:.1f} MB)")
        
        # 6. Export Injector
        injector_path = self._find_matching_injector(lora_path)
        if injector_path:
            inj_dest = self.production_models_dir / "injector.safetensors"
            shutil.copy2(injector_path, str(inj_dest))
            logger.info(f"  → Injector exported: {inj_dest.name}")
        else:
            logger.warning("  ⚠️ Không tìm thấy Injector checkpoint!")
        
        logger.info(f"  ✅ Export hoàn tất! Web App có thể load model mới.")
        return True

    def export_to_production(self, checkpoint_path, destination_name="deep_hair_v1.safetensors"):
        """
        BƯỚC 2: Export Model.
        Tự động detect LoRA vs Full UNet checkpoint → export phù hợp.
        """
        try:
            is_lora, _ = self._is_lora_checkpoint(checkpoint_path)
        except Exception:
            is_lora = False
        
        if is_lora:
            # LoRA checkpoint → merge vào UNet rồi export
            logger.info("Phát hiện LoRA checkpoint → Merge và Export...")
            return self.merge_lora_and_export(checkpoint_path, destination_name)
        else:
            # Full UNet checkpoint (legacy) → copy trực tiếp
            logger.info("Phát hiện full UNet checkpoint (legacy) → Copy trực tiếp...")
            dest_path = self.production_models_dir / destination_name
            
            try:
                shutil.copy2(checkpoint_path, dest_path)
                logger.info(f"  → Deploy UNet THÀNH CÔNG: {dest_path}")
                
                injector_path = self._find_matching_injector(checkpoint_path)
                if injector_path:
                    inj_dest = self.production_models_dir / "injector.safetensors"
                    shutil.copy2(injector_path, str(inj_dest))
                    logger.info(f"  → Deploy Injector THÀNH CÔNG: {inj_dest.name}")
                else:
                    logger.warning("  ⚠️ Không tìm thấy Injector checkpoint!")
                
                logger.info(f"  → Deploy hoàn tất!")
                return True
            except Exception as e:
                logger.error(f"  → Lỗi Deploy Model: {e}")
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
