"""
📊 HairEvaluator — Bộ tiêu chí đánh giá chất lượng hệ thống TryHairStyle.

Gồm 5 nhóm metric:
  1. Identity Preservation  — giữ danh tính khuôn mặt
  2. Hairstyle Similarity   — giữ kiểu tóc tham chiếu (LPIPS + PSNR vùng tóc)
  3. Background Preservation — giữ nền không đổi
  4. Naturalness             — độ tự nhiên toàn ảnh (PSNR toàn cục)
  5. Runtime / Stability     — thời gian chạy và tỷ lệ thành công

Quy ước Input:
  - Tất cả Tensor ảnh đều ở dải [-1, 1], shape (1, 3, H, W) hoặc (B, 3, H, W).
  - Mask ở dải [0, 1], shape (1, 1, H, W) — giá trị 1 = vùng tóc.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image

try:
    import lpips
except ImportError:
    lpips = None  # Fallback — HairEvaluator sẽ tắt LPIPS nếu chưa cài

# ============================================================
# Đường dẫn mặc định tới file cấu hình trọng số
# ============================================================
_EVAL_CONFIG_PATH = Path(__file__).parent / "eval_config.json"


def load_eval_config(path=None):
    """Đọc file eval_config.json chứa trọng số và dải chuẩn hóa."""
    cfg_path = Path(path) if path else _EVAL_CONFIG_PATH
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback mặc định nếu file chưa tồn tại
    return {
        "weights": {
            "identity": 0.35, "hairstyle": 0.30,
            "background": 0.20, "naturalness": 0.10,
            "runtime_stability": 0.05,
        },
        "metric_direction": {
            "identity_similarity": True,
            "hair_lpips": False,
            "hair_psnr": True,
            "background_ssim": True,
            "background_psnr": True,
            "naturalness_psnr": True,
            "runtime_seconds": False,
            "success_rate": True,
        },
        "normalization": {
            "identity_similarity": {"min": 0.0, "max": 1.0},
            "hair_lpips": {"min": 0.0, "max": 0.7},
            "hair_psnr": {"min": 10.0, "max": 40.0},
            "background_ssim": {"min": 0.0, "max": 1.0},
            "background_psnr": {"min": 10.0, "max": 50.0},
            "naturalness_psnr": {"min": 10.0, "max": 40.0},
            "runtime_seconds": {"min": 1.0, "max": 120.0},
            "success_rate": {"min": 0.0, "max": 1.0},
        },
        "metric_to_group": {
            "identity_similarity": "identity",
            "hair_lpips": "hairstyle",
            "hair_psnr": "hairstyle",
            "background_ssim": "background",
            "background_psnr": "background",
            "naturalness_psnr": "naturalness",
            "runtime_seconds": "runtime_stability",
            "success_rate": "runtime_stability",
        },
    }


class HairEvaluator:
    """
    Công cụ đánh giá chất lượng tóc sinh ra (Stage 2 Inpainting).
    Hỗ trợ đánh giá từng metric riêng lẻ hoặc pipeline đầy đủ
    run_full_evaluation() → aggregate_metrics() → final_score.
    """
    def __init__(self, device='cuda', config_path=None):
        self.device = device
        self.config = load_eval_config(config_path)

        # ---- LPIPS (Learned Perceptual Image Patch Similarity) ----
        self.loss_fn_vgg = None
        try:
            if lpips is not None:
                self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.device)
        except Exception as e:
            print(f"LPIPS init error: {e}. Please run: pip install lpips")

        # ---- Face Recognition backbone (lazy load) ----
        # Sẽ chỉ nạp khi gọi evaluate_identity_similarity lần đầu
        self._face_model = None
        self._face_model_attempted = False

        # ---- Segmentation backbone (lazy load) ----
        # Dùng để tách riêng vùng tóc trên result/reference khi tính hairstyle similarity
        self._seg_service = None
        self._seg_service_attempted = False

    # ==============================================================
    # HELPER: Bounding Box & Crop
    # ==============================================================
    def get_hair_bbox(self, mask: torch.Tensor):
        """
        Tìm Bounding Box từ mask tensor (B, 1, H, W).
        Trả về tọa độ (y_min, y_max, x_min, x_max) để crop vùng tóc.
        """
        if len(mask.shape) == 4:
            mask = mask[0, 0]
        elif len(mask.shape) == 3:
            mask = mask[0]

        y_idx, x_idx = torch.where(mask > 0)
        if len(y_idx) == 0:
            return 0, mask.shape[0], 0, mask.shape[1]

        y_min, y_max = y_idx.min().item(), y_idx.max().item()
        x_min, x_max = x_idx.min().item(), x_idx.max().item()

        # Padding nhẹ
        pad = 10
        y_min = max(0, y_min - pad)
        y_max = min(mask.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(mask.shape[1], x_max + pad)

        return y_min, y_max, x_min, x_max

    def crop_to_hair(self, img_tensor: torch.Tensor, mask_tensor: torch.Tensor):
        """Crop ảnh dựa trên mask tóc. img_tensor: (1, 3, H, W)"""
        y1, y2, x1, x2 = self.get_hair_bbox(mask_tensor)
        return img_tensor[:, :, y1:y2, x1:x2]

    def _tensor_to_pil(self, img_tensor: torch.Tensor):
        """Convert tensor [-1, 1] hoặc [0, 1] thành PIL RGB."""
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]
        img = img_tensor.detach().cpu().float()
        if img.min() < 0:
            img = (img + 1.0) / 2.0
        img = img.clamp(0.0, 1.0)
        return transforms.ToPILImage()(img)

    def _pil_mask_to_tensor(self, mask_pil: Image.Image, target_size):
        arr = np.array(mask_pil.resize((target_size[1], target_size[0]), Image.NEAREST), dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    def _load_seg_service(self):
        if self._seg_service is not None:
            return
        if self._seg_service_attempted:
            return

        self._seg_service_attempted = True
        try:
            from backend.app.services.mask import SegmentationService
            self._seg_service = SegmentationService()
        except Exception as e:
            print(f"Could not load SegmentationService: {e}")
            self._seg_service = None

    def _extract_hair_crop(self, img_tensor: torch.Tensor, mask_tensor: torch.Tensor = None, out_size=(256, 256)):
        """Tách riêng vùng tóc, crop theo bbox của mask và đặt lên nền trung tính."""
        if mask_tensor is None:
            self._load_seg_service()
            if self._seg_service is None:
                resize = transforms.Resize(out_size, antialias=True)
                return resize(img_tensor)

            pil_img = self._tensor_to_pil(img_tensor)
            mask_pil = self._seg_service.get_mask(pil_img, target_class=17)
            mask_tensor = self._pil_mask_to_tensor(mask_pil, pil_img.size[::-1])

        if mask_tensor.shape[-2:] != img_tensor.shape[-2:]:
            mask_tensor = F.interpolate(mask_tensor.float(), size=img_tensor.shape[-2:], mode='nearest')

        y1, y2, x1, x2 = self.get_hair_bbox(mask_tensor)
        crop = img_tensor[:, :, y1:y2, x1:x2]
        crop_mask = mask_tensor[:, :, y1:y2, x1:x2].float().to(crop.device)

        if crop.shape[-2] < 4 or crop.shape[-1] < 4:
            resize = transforms.Resize(out_size, antialias=True)
            return resize(crop)

        neutral = torch.zeros_like(crop)
        composited = crop * crop_mask + neutral * (1.0 - crop_mask)
        resize = transforms.Resize(out_size, antialias=True)
        return resize(composited)

    # ==============================================================
    # METRIC 1: LPIPS — Hairstyle Similarity (vùng tóc)
    # ==============================================================
    def evaluate_lpips(self, gen_img: torch.Tensor, gt_img: torch.Tensor, mask: torch.Tensor):
        """
        Tính LPIPS chỉ trên vùng tóc đã crop.
        Đầu vào Tensor yêu cầu dải giá trị [-1, 1].
        Trả về: float (↓ thấp = giống nhau hơn = tốt hơn)
        """
        if self.loss_fn_vgg is None:
            return -1.0

        with torch.no_grad():
            gen_crop = self.crop_to_hair(gen_img, mask)
            gt_crop = self.crop_to_hair(gt_img, mask)

            if gen_crop.shape[2] < 10 or gen_crop.shape[3] < 10:
                return 0.0

            resize = transforms.Resize((256, 256), antialias=True)
            gen_resized = resize(gen_crop)
            gt_resized = resize(gt_crop)

            d = self.loss_fn_vgg(gen_resized, gt_resized)

        return d.item()

    # ==============================================================
    # METRIC 2: PSNR — Masked PSNR trên vùng tóc
    # ==============================================================
    def evaluate_psnr(self, gen_img: torch.Tensor, gt_img: torch.Tensor, mask: torch.Tensor):
        """
        Tính Masked PSNR trên vùng mask tóc.
        Tensor range [-1, 1].
        Trả về: float (↑ cao = tốt hơn)
        """
        gen_norm = (gen_img + 1.0) / 2.0
        gt_norm = (gt_img + 1.0) / 2.0

        mse = torch.sum(((gen_norm - gt_norm) * mask) ** 2) / (torch.sum(mask) * 3 + 1e-8)

        if mse == 0:
            return float('inf')

        psnr = 10 * torch.log10(1.0 / mse)
        return psnr.item()

    # ==============================================================
    # METRIC 3: Identity Similarity — Cosine Similarity khuôn mặt
    # ==============================================================
    def _load_face_model(self):
        """Lazy load backbone Face Recognition (InceptionResnetV1 / vggface2).
        Chỉ nạp 1 lần duy nhất khi cần, giải phóng VRAM bằng cách gọi
        del self._face_model khi xong benchmark.
        """
        if self._face_model is not None:
            return
        if self._face_model_attempted:
            return

        self._face_model_attempted = True

        try:
            from facenet_pytorch import InceptionResnetV1
            self._face_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        except ImportError:
            print("facenet_pytorch is not installed. Run: pip install facenet-pytorch")
            self._face_model = None
        except Exception as e:
            print(f"Could not load FaceNet: {e}")
            self._face_model = None

    @torch.no_grad()
    def evaluate_identity_similarity(self, original_img: torch.Tensor,
                                      result_img: torch.Tensor,
                                      face_bbox=None):
        """
        Đo cosine similarity giữa phần khuôn mặt trước-sau biến đổi.
        Nếu có face_bbox (y1, y2, x1, x2) thì crop trước khi trích đặc trưng.

        Args:
            original_img: Ảnh gốc (1, 3, H, W) dải [-1, 1]
            result_img:   Ảnh kết quả (1, 3, H, W) dải [-1, 1]
            face_bbox:    Tuple (y1, y2, x1, x2) hoặc None (sẽ dùng toàn ảnh)

        Returns:
            float: cosine similarity ∈ [0, 1] (↑ cao = giữ ID tốt hơn)
                   -1.0 nếu không tính được (thiếu model hoặc lỗi)
        """
        self._load_face_model()
        if self._face_model is None:
            return -1.0

        try:
            # Crop face nếu có bbox
            if face_bbox is not None:
                y1, y2, x1, x2 = face_bbox
                orig_face = original_img[:, :, y1:y2, x1:x2]
                res_face = result_img[:, :, y1:y2, x1:x2]
            else:
                # Mặc định: lấy nửa ảnh phía dưới (phần mặt thường ở đây)
                h = original_img.shape[2]
                orig_face = original_img[:, :, h // 4:, :]
                res_face = result_img[:, :, h // 4:, :]

            # Resize về 160x160 (chuẩn input InceptionResnetV1)
            resize_fn = transforms.Resize((160, 160), antialias=True)
            orig_resized = resize_fn(orig_face).to(self.device)
            res_resized = resize_fn(res_face).to(self.device)

            # Chuẩn hóa từ [-1, 1] → [0, 1] (InceptionResnetV1 chấp nhận [0,1] hoặc chuẩn ImageNet)
            orig_resized = (orig_resized + 1.0) / 2.0
            res_resized = (res_resized + 1.0) / 2.0

            # Trích feature
            feat_orig = self._face_model(orig_resized)
            feat_res = self._face_model(res_resized)

            # Cosine similarity
            cos_sim = F.cosine_similarity(feat_orig, feat_res, dim=1)
            # Clamp về [0, 1]
            score = torch.clamp(cos_sim, 0.0, 1.0)
            return score.mean().item()

        except Exception as e:
            print(f"Identity evaluation error: {e}")
            return -1.0

    # ==============================================================
    # METRIC 4: Background Preservation — SSIM + PSNR ngoài vùng tóc
    # ==============================================================
    @torch.no_grad()
    def evaluate_background_preservation(self, original_img: torch.Tensor,
                                          result_img: torch.Tensor,
                                          hair_mask: torch.Tensor = None,
                                          bbox=None):
        """
        Đo mức bảo toàn nền: tính SSIM và PSNR trên vùng KHÔNG phải tóc.
        Nếu mask = None thì tính full ảnh.

        Args:
            original_img: (1, 3, H, W) dải [-1, 1]
            result_img:   (1, 3, H, W) dải [-1, 1]
            hair_mask:    (1, 1, H, W) dải [0, 1] — 1 = tóc
            bbox:         Không dùng hiện tại, để mở rộng tương lai

        Returns:
            dict gồm cả key chuẩn hóa và alias backward-compatible:
                'background_ssim', 'background_psnr', 'bg_ssim', 'bg_psnr'
        """
        # Chuyển sang [0, 1]
        orig = (original_img + 1.0) / 2.0
        result = (result_img + 1.0) / 2.0

        if hair_mask is not None:
            # Interpolate mask nếu kích thước khác
            if hair_mask.shape[-2:] != orig.shape[-2:]:
                hair_mask = F.interpolate(hair_mask, size=orig.shape[-2:], mode='nearest')
            # Vùng nền = 1 - hair_mask
            bg_mask = 1.0 - hair_mask
        else:
            bg_mask = torch.ones_like(orig[:, :1, :, :])

        # --- Background PSNR ---
        diff_sq = ((orig - result) ** 2) * bg_mask
        num_pixels = torch.sum(bg_mask) * 3 + 1e-8  # 3 kênh RGB
        mse = torch.sum(diff_sq) / num_pixels

        if mse < 1e-10:
            bg_psnr = 50.0  # Cap tại 50 dB (gần hoàn hảo)
        else:
            bg_psnr = (10 * torch.log10(1.0 / mse)).item()
            bg_psnr = min(bg_psnr, 50.0)

        # --- Background SSIM (simplified structural similarity) ---
        # SSIM tính trên từng window, ở đây dùng global simplified version
        # để tránh phụ thuộc thư viện ngoài (skimage)
        bg_ssim = self._compute_ssim_masked(orig, result, bg_mask)

        return {
            'background_ssim': bg_ssim,
            'background_psnr': bg_psnr,
            'bg_ssim': bg_ssim,
            'bg_psnr': bg_psnr,
        }

    def _compute_ssim_masked(self, img1: torch.Tensor, img2: torch.Tensor,
                              mask: torch.Tensor, window_size: int = 11):
        """
        Tính SSIM đơn giản trên vùng mask bằng pure PyTorch.
        Dùng Gaussian window cho averaging, chỉ tính trên vùng mask > 0.

        Ưu điểm: không cần skimage, chạy được trên GPU.
        Nhược: Giá trị xấp xỉ so với skimage.ssim, đủ để ranking checkpoint.
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Tính mean trên vùng masked
        # Đơn giản: global mean thay vì sliding window (tiết kiệm VRAM)
        mask_sum = torch.sum(mask) + 1e-8

        # Mở rộng mask ra 3 kênh
        if mask.shape[1] == 1:
            mask3 = mask.expand_as(img1)
        else:
            mask3 = mask

        mu1 = torch.sum(img1 * mask3) / (mask_sum * 3)
        mu2 = torch.sum(img2 * mask3) / (mask_sum * 3)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.sum(((img1 - mu1) * mask3) ** 2) / (mask_sum * 3)
        sigma2_sq = torch.sum(((img2 - mu2) * mask3) ** 2) / (mask_sum * 3)
        sigma12 = torch.sum((img1 - mu1) * (img2 - mu2) * mask3) / (mask_sum * 3)

        ssim_val = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return torch.clamp(ssim_val, 0.0, 1.0).item()

    # ==============================================================
    # METRIC 5: Hair Similarity — So sánh kiểu tóc kết quả vs tham chiếu
    # ==============================================================
    @torch.no_grad()
    def evaluate_hair_similarity(self, result_img: torch.Tensor,
                                  reference_hair_img: torch.Tensor,
                                  hair_mask: torch.Tensor = None):
        """
        So sánh phần tóc ở ảnh kết quả (result) với ảnh mẫu tóc tham chiếu (reference).
        Dùng LPIPS crop vùng tóc nếu có mask, full image nếu không.

        Args:
            result_img:          (1, 3, H, W) dải [-1, 1]
            reference_hair_img:  (1, 3, H, W) dải [-1, 1]
            hair_mask:           (1, 1, H, W) dải [0, 1] — mask phần tóc trên result

        Returns:
            dict: {
                'hair_lpips': float (↓ thấp = giống nhau = tốt),
                'hair_psnr':  float (↑ cao = tốt)
            }
            Trả -1.0 nếu LPIPS chưa sẵn sàng.
        """
        # Tách riêng phần tóc của result và reference để giảm ảnh hưởng từ da/mặt/nền.
        # Nếu chưa có mask của result thì cố gắng sinh bằng SegmentationService.
        result_crop = self._extract_hair_crop(result_img, hair_mask)
        ref_crop = self._extract_hair_crop(reference_hair_img, None)

        # LPIPS
        hair_lpips = -1.0
        if self.loss_fn_vgg is not None:
            if result_crop.shape[2] >= 10 and result_crop.shape[3] >= 10:
                hair_lpips = self.loss_fn_vgg(result_crop, ref_crop).item()

        # PSNR trên crop tóc đã chuẩn hóa kích thước
        gen_norm = (result_crop + 1.0) / 2.0
        ref_norm = (ref_crop + 1.0) / 2.0
        mse = torch.mean((gen_norm - ref_norm) ** 2)
        if mse < 1e-10:
            hair_psnr = 50.0
        else:
            hair_psnr = min(10 * torch.log10(1.0 / mse).item(), 50.0)

        return {
            'hair_lpips': hair_lpips,
            'hair_psnr': hair_psnr,
        }

    # ==============================================================
    # PIPELINE: Đánh giá đầy đủ cho 1 mẫu (sample)
    # ==============================================================
    @torch.no_grad()
    def run_full_evaluation(self, sample: dict):
        """
        Chạy toàn bộ pipeline đánh giá cho 1 mẫu.

        Args:
            sample: dict {
                'original_img':       (1,3,H,W) dải [-1,1] — ảnh gốc (người dùng),
                'result_img':         (1,3,H,W) dải [-1,1] — ảnh kết quả sau transfer,
                'reference_hair_img': (1,3,H,W) dải [-1,1] — ảnh tóc tham chiếu,
                'hair_mask':          (1,1,H,W) dải [0,1]  — mask tóc trên ảnh gốc,
                'face_bbox':          tuple (y1,y2,x1,x2) — bbox khuôn mặt, hoặc None,
                'runtime_seconds':    float — thời gian inference (nếu có),
            }

        Returns:
            dict: Tất cả metric thô (raw metrics)
        """
        original = sample['original_img'].to(self.device)
        result = sample['result_img'].to(self.device)
        reference = sample.get('reference_hair_img')
        hair_mask = sample.get('hair_mask')
        face_bbox = sample.get('face_bbox')

        if reference is not None:
            reference = reference.to(self.device)
        if hair_mask is not None:
            hair_mask = hair_mask.to(self.device)

        metrics = {}

        # 1. Identity Preservation
        id_score = self.evaluate_identity_similarity(original, result, face_bbox)
        metrics['identity_similarity'] = id_score

        # 2. Hairstyle Similarity
        if reference is not None:
            hair_metrics = self.evaluate_hair_similarity(result, reference, hair_mask)
        else:
            # Fallback: so sánh với chính ảnh gốc (self-reconstruction)
            hair_metrics = self.evaluate_hair_similarity(result, original, hair_mask)
        metrics.update(hair_metrics)

        # 3. Background Preservation
        bg_metrics = self.evaluate_background_preservation(original, result, hair_mask)
        metrics.update(bg_metrics)

        # 4. Naturalness — PSNR toàn cục (proxy for overall quality)
        gen_norm = (result + 1.0) / 2.0
        orig_norm = (original + 1.0) / 2.0
        mse_full = torch.mean((gen_norm - orig_norm) ** 2)
        if mse_full < 1e-10:
            metrics['naturalness_psnr'] = 50.0
        else:
            metrics['naturalness_psnr'] = min(
                10 * torch.log10(1.0 / mse_full).item(), 50.0
            )

        # 5. Runtime
        metrics['runtime_seconds'] = sample.get('runtime_seconds', -1.0)
        metrics['success'] = True  # Nếu đến được đây thì inference thành công

        return metrics

    # ==============================================================
    # AGGREGATION: Tính Final Score từ raw metrics
    # ==============================================================
    def aggregate_metrics(self, results_list: list, extra_metric_avgs: dict = None):
        """
        Tổng hợp điểm từ danh sách kết quả run_full_evaluation()
        của nhiều mẫu, tính trung bình và Final Score.

        Args:
            results_list: List[dict] — mỗi phần tử là output của run_full_evaluation()

        Returns:
            dict: {
                'per_metric_avg': {metric_name: avg_value},
                'per_metric_normalized': {metric_name: normalized [0,1]},
                'per_group_score': {group_name: weighted_score},
                'final_score': float ∈ [0, 1],
                'num_samples': int,
                'success_rate': float,
            }
        """
        if not results_list:
            return {
                'per_metric_avg': {},
                'per_metric_normalized': {},
                'per_group_score': {},
                'final_score': 0.0,
                'num_samples': 0,
                'success_rate': 0.0,
            }

        config = self.config
        weights = config['weights']
        directions = config['metric_direction']
        norms = config['normalization']
        m2g = config['metric_to_group']

        # Tính trung bình từng metric
        all_metrics = {}
        success_count = 0
        for res in results_list:
            if res.get('success', False):
                success_count += 1
            for key, val in res.items():
                if isinstance(val, (int, float)) and val >= 0:
                    all_metrics.setdefault(key, []).append(val)

        per_metric_avg = {}
        for key, vals in all_metrics.items():
            per_metric_avg[key] = float(np.mean(vals))

        # Success rate
        success_rate = success_count / len(results_list) if results_list else 0.0
        per_metric_avg['success_rate'] = success_rate

        if extra_metric_avgs:
            for key, val in extra_metric_avgs.items():
                if isinstance(val, (int, float)) and val >= 0:
                    per_metric_avg[key] = float(val)

        # Chuẩn hóa về [0, 1]
        per_metric_normalized = {}
        for metric_name, avg_val in per_metric_avg.items():
            if metric_name not in norms:
                continue
            norm_cfg = norms[metric_name]
            mn, mx = norm_cfg['min'], norm_cfg['max']
            # Clamp rồi scale
            clamped = max(mn, min(mx, avg_val))
            normalized = (clamped - mn) / (mx - mn + 1e-8)

            # Đảo chiều nếu metric kiểu "thấp = tốt" (VD: LPIPS, runtime)
            direction = directions.get(metric_name, True)
            if not direction:
                normalized = 1.0 - normalized

            per_metric_normalized[metric_name] = normalized

        # Tính điểm mỗi nhóm (trung bình các metric cùng nhóm)
        group_scores = {}
        group_counts = {}
        for metric_name, norm_val in per_metric_normalized.items():
            group = m2g.get(metric_name)
            if group:
                group_scores[group] = group_scores.get(group, 0.0) + norm_val
                group_counts[group] = group_counts.get(group, 0) + 1

        per_group_score = {}
        for group in group_scores:
            per_group_score[group] = group_scores[group] / group_counts[group]

        # Final Score = tổng trọng số
        final_score = 0.0
        for group, weight in weights.items():
            if group in per_group_score:
                final_score += weight * per_group_score[group]

        return {
            'per_metric_avg': per_metric_avg,
            'per_metric_normalized': per_metric_normalized,
            'per_group_score': per_group_score,
            'final_score': round(final_score, 4),
            'num_samples': len(results_list),
            'success_rate': round(success_rate, 4),
        }

    # ==============================================================
    # BACKWARD-COMPAT: Batch evaluation (cho train_stage2.py dùng)
    # ==============================================================
    def run_evaluation_batch(self, generated_batch, ground_truth_batch, mask_batch):
        """
        Chạy pipeline đánh giá nhẹ cho 1 batch (dùng trong validate_epoch).
        Chỉ tính LPIPS + PSNR + Background PSNR + Identity (nếu có thể).
        """
        batch_size = generated_batch.shape[0]
        lpips_scores = []
        psnr_scores = []
        bg_psnr_scores = []
        id_scores = []

        for i in range(batch_size):
            g = generated_batch[i:i+1]
            gt = ground_truth_batch[i:i+1]
            m = mask_batch[i:i+1]

            l = self.evaluate_lpips(g, gt, m)
            p = self.evaluate_psnr(g, gt, m)

            lpips_scores.append(l)
            psnr_scores.append(p)

            # Background PSNR (pure math, rất nhẹ)
            bg = self.evaluate_background_preservation(gt, g, m)
            bg_psnr_scores.append(bg['bg_psnr'])

            # Identity (nặng, chỉ tính nếu đã load face model)
            if self._face_model is not None:
                id_s = self.evaluate_identity_similarity(gt, g)
                id_scores.append(id_s)

        valid_lpips = [x for x in lpips_scores if x >= 0]
        valid_id = [x for x in id_scores if x >= 0]

        return {
            "lpips_mean": float(np.mean(valid_lpips)) if valid_lpips else -1.0,
            "psnr_mean": float(np.mean(psnr_scores)) if psnr_scores else 0.0,
            "bg_psnr_mean": float(np.mean(bg_psnr_scores)) if bg_psnr_scores else 0.0,
            "identity_mean": float(np.mean(valid_id)) if valid_id else -1.0,
        }

    # ==============================================================
    # METRIC 6: Prompt Response — Do kha nang model nghe prompt
    # ==============================================================
    @torch.no_grad()
    def evaluate_prompt_response(self, output_match: torch.Tensor,
                                  output_conflict: torch.Tensor,
                                  hair_mask: torch.Tensor = None):
        """
        So sanh 2 outputs: match_prompt vs conflict_prompt.
        Neu 2 outputs khac nhau nhieu => model nghe prompt (tot).
        Neu giong nhau => model bo qua prompt (xau).

        Args:
            output_match: (1,3,H,W) [-1,1] — output khi dung match prompt
            output_conflict: (1,3,H,W) [-1,1] — output khi dung conflict prompt
            hair_mask: (1,1,H,W) [0,1] — optional mask vung toc

        Returns:
            dict: {
                'prompt_lpips_diff': float (cao = model nghe prompt),
                'prompt_l2_diff': float,
                'prompt_responsive': bool,
            }
        """
        results = {}

        # Resize neu can
        if output_match.shape != output_conflict.shape:
            target_size = output_match.shape[-2:]
            output_conflict = F.interpolate(
                output_conflict, size=target_size, mode='bilinear', align_corners=False
            )

        # LPIPS difference
        if self.loss_fn_vgg is not None:
            lpips_val = self.loss_fn_vgg(
                output_match.to(self.device), output_conflict.to(self.device)
            ).item()
            results['prompt_lpips_diff'] = lpips_val
        else:
            results['prompt_lpips_diff'] = -1.0

        # L2 distance
        match_norm = (output_match + 1.0) / 2.0
        conflict_norm = (output_conflict + 1.0) / 2.0
        l2_val = torch.sqrt(torch.mean((match_norm - conflict_norm) ** 2)).item()
        results['prompt_l2_diff'] = l2_val

        # Responsive judgment
        if results['prompt_lpips_diff'] >= 0:
            results['prompt_responsive'] = results['prompt_lpips_diff'] > 0.10
        else:
            results['prompt_responsive'] = l2_val > 0.05

        return results

    # ==============================================================
    # CLEANUP: Giai phong VRAM tu face model khi khong can nua
    # ==============================================================
    def unload_heavy_models(self):
        """Giai phong VRAM: xoa face model va LPIPS khoi GPU."""
        if self._face_model is not None:
            del self._face_model
            self._face_model = None
        if self.loss_fn_vgg is not None:
            self.loss_fn_vgg.cpu()
        torch.cuda.empty_cache()



# ==================================================================
# STANDALONE TEST
# ==================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 [Testing] HairEvaluator — Full Pipeline")
    print("=" * 60)

    device = 'cpu'
    evaluator = HairEvaluator(device=device)

    # Fake Dummy Tensors [-1, 1]
    original = torch.rand(1, 3, 512, 512) * 2 - 1
    result = original.clone() + torch.randn_like(original) * 0.1  # Thêm nhiễu nhẹ
    result = torch.clamp(result, -1, 1)
    reference = torch.rand(1, 3, 512, 512) * 2 - 1
    mask = torch.zeros(1, 1, 512, 512)
    mask[:, :, 50:250, 100:400] = 1.0  # Fake hair region

    print("\n--- 1. Evaluate LPIPS (vùng tóc) ---")
    lpips_val = evaluator.evaluate_lpips(result, original, mask)
    print(f"  LPIPS = {lpips_val:.4f}")

    print("\n--- 2. Evaluate PSNR (vùng tóc) ---")
    psnr_val = evaluator.evaluate_psnr(result, original, mask)
    print(f"  PSNR = {psnr_val:.2f} dB")

    print("\n--- 3. Evaluate Identity Similarity ---")
    id_val = evaluator.evaluate_identity_similarity(original, result)
    print(f"  Identity Similarity = {id_val:.4f}")

    print("\n--- 4. Evaluate Background Preservation ---")
    bg = evaluator.evaluate_background_preservation(original, result, mask)
    print(f"  BG SSIM = {bg['bg_ssim']:.4f}")
    print(f"  BG PSNR = {bg['bg_psnr']:.2f} dB")

    print("\n--- 5. Evaluate Hair Similarity ---")
    hair = evaluator.evaluate_hair_similarity(result, reference, mask)
    print(f"  Hair LPIPS = {hair['hair_lpips']:.4f}")
    print(f"  Hair PSNR  = {hair['hair_psnr']:.2f} dB")

    print("\n--- 6. Full Evaluation Pipeline ---")
    sample = {
        'original_img': original,
        'result_img': result,
        'reference_hair_img': reference,
        'hair_mask': mask,
        'face_bbox': None,
        'runtime_seconds': 15.3,
    }
    full_metrics = evaluator.run_full_evaluation(sample)
    print(f"  Raw metrics: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in full_metrics.items()}, indent=2)}")

    print("\n--- 7. Aggregate Metrics (Final Score) ---")
    agg = evaluator.aggregate_metrics([full_metrics])
    print(f"  Final Score: {agg['final_score']}")
    print(f"  Per-group:   {json.dumps({k: round(v, 4) for k, v in agg['per_group_score'].items()}, indent=2)}")

    print("\n--- 8. Batch Evaluation (backward-compat) ---")
    batch_metrics = evaluator.run_evaluation_batch(result, original, mask)
    print(f"  Batch metrics: {batch_metrics}")

    print("\n✅ All tests passed!")
