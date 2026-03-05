import os
import sys
import json
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from safetensors.torch import save_file

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_DIR))
from backend.app.services.training_utils import setupLogger, getDevice, ensureDir

logger = setupLogger("TrainStage1_Texture")
DEVICE = getDevice()

# Auto-detect Google Colab environment
IS_COLAB = os.path.exists("/content") and "COLAB_GPU" in os.environ


class HairTextureEncoder(nn.Module):
    """
    Mạng mã hóa Texture Tóc (Hair Texture Encoder).
    Nhiệm vụ: Trích xuất đặc trưng sâu từ các bản vá tóc (hair patches).
    Sử dụng ResNet50 (hoặc ViT) làm backbone.
    
    Output gồm 3 nhánh:
    1. embedding: Đặc trưng gốc (e.g. 2048-dim) để cấp cho UNet (Style Injection).
    2. proj: Vector đã qua không gian Projection (e.g. 128-dim) dùng cho Supervised Contrastive Loss.
    3. cls_logits: Từ điển chứa các logits dự đoán nhãn vật lý (VD: Độ xoăn, Volume).
    """

    def __init__(self, 
                 proj_dim=128, 
                 curl_classes=4,     # thẳng, vểnh, xoăn, xoăn tít (straight, wavy, curly, tightly curly)
                 volume_classes=3,   # ít, bình thường, nhiều (low, normal, high)
                 pretrained=True):
        super(HairTextureEncoder, self).__init__()
        
        # 1. Khởi tạo Backbone (ResNet50)
        # Sử dụng weights của ImageNet để hội tụ nhanh hơn
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Bỏ đi layer fully connected (FC) cuối cùng của ResNet
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embed_dim = resnet.fc.in_features # 2048
        
        # 2. Đầu ra Projection cho Contrastive Learning (SupConLoss)
        # Khuyến nghị dùng MLP 2 layer theo paper SimCLR/SupCon
        self.projection_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, proj_dim)
        )
        
        # 3. Đầu ra Phân loại Phụ (Auxiliary Classifier Heads)
        # Ép mô hình học được ranh giới vật lý rõ ràng thay vì chỉ ghép pixel
        self.curl_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim, curl_classes)
        )
        
        self.volume_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim, volume_classes)
        )

    def forward(self, x):
        """
        x shape: (B, 3, H, W) - Tensor ảnh patch tóc chuẩn hóa
        """
        # Trích xuất đặc trưng (B, 2048, 1, 1) -> (B, 2048)
        feats = self.backbone(x)
        embed = torch.flatten(feats, 1)
        
        # Mũi tiêm Contrastive
        proj = self.projection_head(embed)
        # L2 Normalize projection vector (rất quan trọng cho InfoNCE / SupCon loss)
        proj = F.normalize(proj, dim=1)
        
        # Mũi tiêm Auxiliary
        curl_logits = self.curl_classifier(embed)
        volume_logits = self.volume_classifier(embed)
        
        cls_logits = {
            "curl": curl_logits,
            "volume": volume_logits
        }
        
        return embed, proj, cls_logits

# Bảng ánh xạ text prompt keywords → nhãn số
CURL_MAP = {"straight": 0, "wavy": 1, "curly": 2, "tightly curly": 3}
VOLUME_MAP = {"low": 0, "normal": 1, "high": 2}

def _parse_curl_label(text_prompt: str) -> int:
    """Parse nhãn curl từ text prompt. Mặc định = 0 (straight)."""
    text = text_prompt.lower()
    # Kiểm tra theo thứ tự dài → ngắn để tránh match sai
    if "tightly curly" in text: return 3
    if "curly" in text: return 2
    if "wavy" in text: return 1
    return 0

def _parse_volume_label(text_prompt: str) -> int:
    """Parse nhãn volume từ text prompt. Mặc định = 1 (normal)."""
    text = text_prompt.lower()
    if "high volume" in text or "많은 volume" in text: return 2
    if "low volume" in text or "적은 volume" in text: return 0
    return 1

class HairTextureDataset(Dataset):
    def __init__(self, data_dir: Path, target_size=(128, 128)):
        self.data_dir = data_dir
        self.patches_dir = data_dir / "hair_patches"
        self.target_size = target_size
        self.patch_paths = []
        self.patch_labels = []  # (curl_label, volume_label) cho mỗi patch
        
        # Parse metadata để đếm số patch VÀ lấy nhãn thực
        meta_path = data_dir / "metadata.jsonl"
        if meta_path.exists():
            with open(str(meta_path), "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    img_id = item["id"]
                    num_patches = item.get("num_patches", 0)
                    text_prompt = item.get("text_prompt", "")
                    
                    # Parse nhãn thực từ text prompt
                    curl_label = _parse_curl_label(text_prompt)
                    volume_label = _parse_volume_label(text_prompt)
                    
                    for i in range(num_patches):
                        patch_file = self.patches_dir / f"{img_id}_patch_{i:03d}.png"
                        if patch_file.exists():
                            self.patch_paths.append(str(patch_file))
                            self.patch_labels.append((curl_label, volume_label))
                            
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet std
        ])

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        patch_path = self.patch_paths[idx]
        curl_label, volume_label = self.patch_labels[idx]
        
        # Thử đọc ảnh tối đa 3 lần (Drive FUSE có thể timeout tạm thời)
        for attempt in range(3):
            try:
                raw = cv2.imread(patch_path)
                if raw is None:
                    raise ValueError(f"cv2.imread returned None: {patch_path}")
                img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.target_size)
                return {
                    "patch": self.img_transform(img),
                    "curl_label": torch.tensor(curl_label, dtype=torch.long),
                    "volume_label": torch.tensor(volume_label, dtype=torch.long)
                }
            except Exception:
                if attempt < 2:
                    import time
                    time.sleep(0.5)  # Chờ 0.5s rồi thử lại
        
        # 3 lần đều thất bại → bỏ qua, lấy sample khác ngẫu nhiên
        import random
        new_idx = random.randint(0, len(self.patch_paths) - 1)
        while new_idx == idx:
            new_idx = random.randint(0, len(self.patch_paths) - 1)
        return self.__getitem__(new_idx)

class TextureEncoderTrainer:
    # Lưu training state mỗi N steps để giảm mất mát khi disconnect
    SAVE_EVERY_N_STEPS = 50
    
    def __init__(self):
        logger.info("Khởi tạo Stage 1 Trainer: Hair Texture Encoder")
        self.model = HairTextureEncoder().to(DEVICE)
        
        # Losses
        self.criterion_cls = nn.CrossEntropyLoss()
        # Supervised Contrastive Loss thực — kéo patch cùng nhãn curl lại gần, đẩy khác nhãn ra xa
        from backend.training.models.losses import SupConLoss
        self.criterion_contrastive = SupConLoss(temperature=0.07)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        
        # Thư mục checkpoints
        self.checkpoints_dir = PROJECT_DIR / "backend" / "training" / "checkpoints"
        ensureDir(str(self.checkpoints_dir))

    
    def _save_safetensors_safe(self, state_dict, path: str):
        """Lưu safetensors an toàn — ghi vào temp file rồi move để tránh corrupt."""
        import tempfile, shutil
        target_dir = os.path.dirname(path)
        fd, temp_path = tempfile.mkstemp(suffix=".safetensors", dir=target_dir)
        os.close(fd)
        try:
            save_file(state_dict, temp_path)
            shutil.move(temp_path, path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        patches = batch["patch"].to(DEVICE)
        curl_labels = batch["curl_label"].to(DEVICE)
        volume_labels = batch["volume_label"].to(DEVICE)
        
        # Forward pass
        embed, proj, cls_logits = self.model(patches)
        
        # 1. Auxiliary Classification Loss
        loss_curl = self.criterion_cls(cls_logits["curl"], curl_labels)
        loss_vol = self.criterion_cls(cls_logits["volume"], volume_labels)
        loss_class = loss_curl + loss_vol
        
        # 2. Supervised Contrastive Loss thực sự (SupCon)
        # SupConLoss yêu cầu features shape (B, n_views, embed_dim)
        # Với 1 view duy nhất → unsqueeze(1)
        # Guard: SupConLoss cần ít nhất 2 classes khác nhau trong batch
        unique_labels = curl_labels.unique()
        if len(unique_labels) >= 2:
            proj_views = proj.unsqueeze(1)  # (B, 1, 128)
            loss_contrastive = self.criterion_contrastive(proj_views, curl_labels)
        else:
            loss_contrastive = torch.tensor(0.0, device=patches.device)
        
        total_loss = loss_class + (0.5 * loss_contrastive)
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
        
    def validate_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            patches = batch["patch"].to(DEVICE)
            curl_labels = batch["curl_label"].to(DEVICE)
            volume_labels = batch["volume_label"].to(DEVICE)
            
            embed, proj, cls_logits = self.model(patches)
            
            loss_curl = self.criterion_cls(cls_logits["curl"], curl_labels)
            loss_vol = self.criterion_cls(cls_logits["volume"], volume_labels)
            loss_class = loss_curl + loss_vol
            
            unique_labels = curl_labels.unique()
            if len(unique_labels) >= 2:
                proj_views = proj.unsqueeze(1)
                loss_contrastive = self.criterion_contrastive(proj_views, curl_labels)
            else:
                loss_contrastive = torch.tensor(0.0, device=patches.device)
            
            total_loss = loss_class + (0.5 * loss_contrastive)
            
        return total_loss.item()
        
    def _discover_data_dirs(self):
        """Tìm tất cả processed_NNN directories. Fallback về processed/ nếu không có chunks."""
        import re as _re
        training_dir = PROJECT_DIR / "backend" / "training"
        chunks = sorted([
            d for d in training_dir.iterdir()
            if d.is_dir() and _re.match(r'^processed_\d+$', d.name)
            and (d / "metadata.jsonl").exists()
        ])
        if not chunks:
            single = training_dir / "processed"
            if single.exists() and (single / "metadata.jsonl").exists():
                chunks = [single]
                logger.info("  📁 Single processed/ directory (no chunks found)")
        return chunks

    def _save_training_state(self, epoch, global_step, best_val_loss, best_epoch, loss_history, batch_index=-1):
        """Save full training state cho Colab resume.
        batch_index: index batch vừa train xong (-1 = epoch hoàn tất).
        """
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "loss_history": loss_history,
            "optimizer": self.optimizer.state_dict(),
            "batch_index": batch_index,
        }
        state_path = self.checkpoints_dir / "stage1_training_state.pth"
        torch.save(state, str(state_path))
        batch_info = f", batch={batch_index}" if batch_index >= 0 else ""
        logger.info(f"💾 Training state saved: epoch={epoch}, step={global_step}{batch_info}")
    
    def _load_training_state(self):
        """Load full training state để resume. Returns dict hoặc None."""
        state_path = self.checkpoints_dir / "stage1_training_state.pth"
        
        best_ckpt = self.checkpoints_dir / "texture_encoder_best.safetensors"
        latest_ckpt = self.checkpoints_dir / "texture_encoder_latest.safetensors"
        
        if not state_path.exists():
            # Fallback: chỉ load model weights (backward compat — không có training state)
            # Ưu tiên best vì không biết latest ở trạng thái nào
            load_ckpt = best_ckpt if best_ckpt.exists() else (latest_ckpt if latest_ckpt.exists() else None)
            if load_ckpt:
                from safetensors.torch import load_file as load_safetensors
                try:
                    self.model.load_state_dict(load_safetensors(str(load_ckpt)))
                    logger.info(f"🔄 [RESUME] Model weights loaded: {load_ckpt.name}")
                    logger.warning("  ⚠️ Không có stage1_training_state.pth — optimizer bắt đầu lại")
                except Exception as e:
                    logger.error(f"❌ Lỗi load model weights: {e}")
            return None
        
        try:
            state = torch.load(str(state_path), map_location=DEVICE, weights_only=False)
            batch_index = state.get("batch_index", -1)
            
            # Khi có training_state.pth:
            # - Mid-epoch (batch_index >= 0): ưu tiên LATEST (weights mới nhất, saved mỗi 50 steps)
            # - End-epoch (batch_index == -1): ưu tiên BEST (model tốt nhất đã validate)
            if batch_index >= 0:
                load_ckpt = latest_ckpt if latest_ckpt.exists() else (best_ckpt if best_ckpt.exists() else None)
            else:
                load_ckpt = best_ckpt if best_ckpt.exists() else (latest_ckpt if latest_ckpt.exists() else None)
            
            if load_ckpt:
                from safetensors.torch import load_file as load_safetensors
                self.model.load_state_dict(load_safetensors(str(load_ckpt)))
                logger.info(f"  🔄 Model weights: {load_ckpt.name}")
            
            # Restore optimizer
            if "optimizer" in state:
                self.optimizer.load_state_dict(state["optimizer"])
                logger.info(f"  🔄 Optimizer state restored")
            
            return state
        except Exception as e:
            logger.error(f"❌ Lỗi load training state: {e}")
            return None
    
    def _cleanup_mid_epoch_state(self):
        """Xóa file training state trung gian khi epoch hoàn tất.
        File state chỉ cần thiết để resume giữa epoch.
        Sau khi epoch xong, model weights đã lưu vào latest/best."""
        state_path = self.checkpoints_dir / "stage1_training_state.pth"
        if state_path.exists():
            try:
                state_path.unlink()
                logger.info("  🗑️ Đã xóa checkpoint trung gian: stage1_training_state.pth")
            except Exception:
                pass

    def train_loop(self, num_epochs=1, batch_size=4, max_samples=0, resume=False):
        logger.info(f"Khởi động vòng lặp Training Stage 1 - {num_epochs} Epochs")
        logger.info(f"  💾 Checkpoint: mỗi {self.SAVE_EVERY_N_STEPS} steps + cuối epoch")
        logger.info(f"  💾 Drive: {'LƯU TẤT CẢ epoch' if IS_COLAB else 'N/A'}")
        
        # ==================================================
        # 1. RESUME FROM CHECKPOINT
        # ==================================================
        start_epoch = 0
        global_step = 0
        best_val_loss = float('inf')
        best_epoch = -1
        loss_history = {
            'train_loss': [], 'val_loss': [], 'epoch_avg_train': [], 'epoch_avg_val': [],
        }
        resume_batch_index = -1  # -1 = bắt đầu epoch mới
        
        if resume:
            loaded = self._load_training_state()
            if loaded:
                start_epoch = loaded['epoch']
                global_step = loaded['global_step']
                best_val_loss = loaded['best_val_loss']
                best_epoch = loaded.get('best_epoch', -1)
                loss_history = loaded.get('loss_history', loss_history)
                resume_batch_index = loaded.get('batch_index', -1)
                
                if resume_batch_index >= 0:
                    logger.info(f"🔄 [RESUME] Epoch {start_epoch+1}, Step {global_step}, Batch {resume_batch_index}, Best Val: {best_val_loss:.6f}")
                    # Epoch chưa hoàn tất → giữ nguyên epoch để tiếp tục
                else:
                    logger.info(f"🔄 [RESUME] Epoch {start_epoch}, Step {global_step}, Best Val: {best_val_loss:.6f}")
            elif not loaded:
                # Fallback đã xử lý trong _load_training_state
                pass
            
            # ── SKIP nếu đã train đủ epochs ──
            if loaded:
                completed_epoch = loaded.get('epoch', 0)
                batch_index = loaded.get('batch_index', -1)
                best_ckpt = self.checkpoints_dir / "texture_encoder_best.safetensors"
                # Đã hoàn tất (batch_index == -1) VÀ đủ epochs VÀ có best checkpoint
                if batch_index == -1 and completed_epoch >= num_epochs and best_ckpt.exists():
                    logger.info(f"⏭️ SKIP Stage 1: đã hoàn tất {completed_epoch} epochs (yêu cầu {num_epochs})")
                    logger.info(f"  🏆 Best model sẵn sàng: {best_ckpt.name}")
                    return
        
        # ==================================================
        # 2. LOAD DATASET
        # ==================================================
        data_dirs = self._discover_data_dirs()
        if not data_dirs:
            logger.error("❌ Dataset Trống! Không tìm thấy thư mục processed_NNN/ hoặc processed/.")
            return
        
        logger.info(f"📂 Tìm thấy {len(data_dirs)} thư mục dữ liệu: {[d.name for d in data_dirs]}")
        
        # Gộp tất cả patches từ mọi chunk vào 1 dataset
        from torch.utils.data import ConcatDataset
        all_datasets = []
        for data_dir in data_dirs:
            ds = HairTextureDataset(data_dir)
            if len(ds) > 0:
                all_datasets.append(ds)
                logger.info(f"  📁 {data_dir.name}: {len(ds)} patches")
        
        if not all_datasets:
            logger.error("❌ Dataset Trống! Không tìm thấy patches nào.")
            return
        
        full_dataset = ConcatDataset(all_datasets)
        logger.info(f"✅ Tổng cộng: {len(full_dataset)} patches từ {len(all_datasets)} chunks")
        
        if len(full_dataset) == 0:
            logger.error("Dataset Trống! Lỗi trích xuất Patches.")
            return
            
        # Giới hạn dataset nếu đang chạy test — random sample để tránh bias
        if max_samples > 0 and len(full_dataset) > max_samples:
            import random as _random
            import torch.utils.data as data
            _random.seed(42)  # Reproducible subset
            original_count = len(full_dataset)
            indices = _random.sample(range(len(full_dataset)), max_samples)
            full_dataset = data.Subset(full_dataset, indices)
            logger.info(f"📉 Dataset giảm xuống {max_samples} mẫu (random sample từ {original_count} gốc)")
        
        # Split dataset into training and validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible split (khớp với Stage 2)
        )
            
        num_workers = 0 if os.name == 'nt' else 2
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
        
        steps_per_epoch = len(train_loader)
        logger.info(f"  📊 Training: {num_epochs} epochs × {steps_per_epoch} steps/epoch")
        
        best_ckpt_path = None  # Theo dõi file best để xóa cái cũ
        
        # ==================================================
        # 3. TRAINING LOOP
        # ==================================================
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            
            # Xác định batch cần skip khi resume giữa epoch
            skip_until = 0
            if resume_batch_index >= 0 and epoch == start_epoch:
                skip_until = resume_batch_index + 1
                logger.info(f"\n🔄 RESUME Epoch {epoch+1} — skip {skip_until} batches đã train")
            
            # Training Phase
            self.model.train()
            train_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for step, batch in enumerate(pbar):
                # Skip batches đã train khi resume
                if step < skip_until:
                    continue
                
                loss = self.train_step(batch)
                train_losses.append(loss)
                loss_history['train_loss'].append(loss)
                pbar.set_postfix({"Loss": f"{loss:.4f}", "Step": global_step})
                global_step += 1
                
                # Lưu checkpoint trung gian mỗi N steps
                if global_step % self.SAVE_EVERY_N_STEPS == 0:
                    self._save_safetensors_safe(
                        self.model.state_dict(),
                        str(self.checkpoints_dir / "texture_encoder_latest.safetensors")
                    )
                    self._save_training_state(
                        epoch=epoch, global_step=global_step,
                        best_val_loss=best_val_loss, best_epoch=best_epoch,
                        loss_history=loss_history, batch_index=step
                    )
            
            # Reset resume state sau epoch đầu tiên
            resume_batch_index = -1
            
            if not train_losses:
                logger.warning(f"  ⚠️ Epoch {epoch+1} không có training data, bỏ qua.")
                continue
            
            avg_train_loss = sum(train_losses) / len(train_losses)
            loss_history['epoch_avg_train'].append(avg_train_loss)
            
            # Validation Phase
            val_losses = []
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for step, batch in enumerate(pbar_val):
                loss = self.validate_step(batch)
                val_losses.append(loss)
                pbar_val.set_postfix({"Loss": f"{loss:.4f}"})
                
            avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
            loss_history['val_loss'].append(avg_val_loss)
            loss_history['epoch_avg_val'].append(avg_val_loss)
            
            epoch_time = time.time() - epoch_start
            
            is_new_best = avg_val_loss < best_val_loss
            if is_new_best:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                logger.info(f"🏆 NEW BEST! Epoch {epoch+1} — Train: {avg_train_loss:.4f} — Val: {avg_val_loss:.4f} — Time: {epoch_time:.0f}s")
                
                # Tạo file best mới theo epoch
                new_best_path = self.checkpoints_dir / f"texture_encoder_best_ep{epoch+1}.safetensors"
                self._save_safetensors_safe(self.model.state_dict(), str(new_best_path))
                
                # Xóa file best cũ (chỉ giữ lại cái tốt nhất)
                if best_ckpt_path and best_ckpt_path.exists():
                    try:
                        os.remove(str(best_ckpt_path))
                        logger.info(f"🗑️ Đã xóa model cũ: {best_ckpt_path.name}")
                    except Exception as e:
                        logger.warning(f"Không thể xóa model cũ {best_ckpt_path}: {e}")
                
                best_ckpt_path = new_best_path
            else:
                logger.info(f"Epoch {epoch+1} — Train: {avg_train_loss:.4f} — Val: {avg_val_loss:.4f} — Time: {epoch_time:.0f}s (Best: {best_val_loss:.4f} at Ep {best_epoch})")
            
            # Lưu file latest mỗi epoch
            latest_path = self.checkpoints_dir / "texture_encoder_latest.safetensors"
            self._save_safetensors_safe(self.model.state_dict(), str(latest_path))
            
            # Lưu training state cuối epoch (batch_index=-1 = epoch hoàn tất)
            self._save_training_state(
                epoch=epoch + 1, global_step=global_step,
                best_val_loss=best_val_loss, best_epoch=best_epoch,
                loss_history=loss_history, batch_index=-1
            )
            
            # ✅ Epoch trung gian → xóa checkpoint mid-epoch (giữ lại ở epoch cuối để skip logic hoạt động)
            if epoch + 1 < num_epochs:
                self._cleanup_mid_epoch_state()
            
            logger.info(f"💾 Checkpoint Epoch {epoch+1}{' ⭐ (BEST)' if is_new_best else ''}")
        
        # Kết thúc: copy file best hiện tại sang tên chuẩn để export/deploy
        if best_ckpt_path and best_ckpt_path.exists():
            import shutil
            final_best_path = self.checkpoints_dir / "texture_encoder_best.safetensors"
            shutil.copy2(str(best_ckpt_path), str(final_best_path))
        
        logger.info(f"={'='*60}")
        logger.info(f"✅ Hoàn thành Training Stage 1!")
        logger.info(f"  🏆 Model tốt nhất (dùng để deploy): texture_encoder_best.safetensors (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")
        logger.info(f"  📁 Model cuối cùng: texture_encoder_latest.safetensors")
        logger.info(f"={'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Stage 1 - Texture Encoder")
    parser.add_argument("--epochs", type=int, default=1, help="Số epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=0, help="Giới hạn số mẫu dataset (0=tất cả) để test")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục huấn luyện từ checkpoint tốt nhất nếu có")
    args = parser.parse_args()

    trainer = TextureEncoderTrainer()
    trainer.train_loop(num_epochs=args.epochs, batch_size=args.batch_size, max_samples=args.max_samples, resume=args.resume)
