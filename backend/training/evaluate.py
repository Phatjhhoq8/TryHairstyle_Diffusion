import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

try:
    import lpips
except ImportError:
    pass # Sẽ yêu cầu user cài đặt sau nếu dùng

class HairEvaluator:
    """
    Công cụ đánh giá chất lượng tóc sinh ra (Stage 2 Inpainting).
    Các metric chỉ tính toán bên trong vùng bounding box của hair_mask
    để tránh bị nhiễu bởi phần khuôn mặt hay background.
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.logger = None
        
        # 1. LPIPS (Learned Perceptual Image Patch Similarity)
        # Đánh giá độ tương đồng về mặt thị giác của con người
        try:
            self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.device)
        except Exception as e:
            self.loss_fn_vgg = None
            print(f"Lỗi khởi tạo LPIPS: {e}. Vui lòng chạy: pip install lpips")

        # 2. FID (Frechet Inception Distance) 
        # Cần extract feature từ InceptionV3. Thường dùng torcheval hoặc pytorch-fid
        # Ở đây cung cấp cấu trúc xương sống để tính Masked MSE/PSNR/LPIPS trước.
        
    def get_hair_bbox(self, mask: torch.Tensor):
        """ 
        Tìm Bounding Box từ mask tensor (B, 1, H, W). 
        Đơn giản hóa: Trả về tọa độ crop để thu thập riêng mảng tóc.
        """
        # mask shape: (H, W) for single image
        if len(mask.shape) == 4:
            mask = mask[0, 0]
        elif len(mask.shape) == 3:
            mask = mask[0]
            
        y_idx, x_idx = torch.where(mask > 0)
        if len(y_idx) == 0:
            return 0, mask.shape[0], 0, mask.shape[1]
            
        y_min, y_max = y_idx.min().item(), y_idx.max().item()
        x_min, x_max = x_idx.min().item(), x_idx.max().item()
        
        # Thêm padding
        pad = 10
        y_min = max(0, y_min - pad)
        y_max = min(mask.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(mask.shape[1], x_max + pad)
        
        return y_min, y_max, x_min, x_max

    def crop_to_hair(self, img_tensor: torch.Tensor, mask_tensor: torch.Tensor):
        """ Crop ảnh dựa trên mask tóc. img_tensor: (1, 3, H, W) """
        y1, y2, x1, x2 = self.get_hair_bbox(mask_tensor)
        return img_tensor[:, :, y1:y2, x1:x2]

    def evaluate_lpips(self, gen_img: torch.Tensor, gt_img: torch.Tensor, mask: torch.Tensor):
        """
        Tính LPIPS chỉ trên vùng tóc đã crop.
        Đầu vào Tensor yêu cầu dải giá trị [-1, 1].
        """
        if self.loss_fn_vgg is None:
            return -1.0
            
        with torch.no_grad():
            # Cắt ảnh lấy đoạn thân tóc chính
            gen_crop = self.crop_to_hair(gen_img, mask)
            gt_crop = self.crop_to_hair(gt_img, mask)
            
            # Nếu crop quá nhỏ (không có tóc)
            if gen_crop.shape[2] < 10 or gen_crop.shape[3] < 10:
                return 0.0
                
            # Resize về 256x256 (Chuẩn của VGG LPIPS)
            resize = transforms.Resize((256, 256), antialias=True)
            gen_resized = resize(gen_crop)
            gt_resized = resize(gt_crop)
            
            # Tính điểm LPIPS
            d = self.loss_fn_vgg(gen_resized, gt_resized)
            
        return d.item()
        
    def evaluate_psnr(self, gen_img: torch.Tensor, gt_img: torch.Tensor, mask: torch.Tensor):
        """ Calculate Masked PSNR. Tensor range [-1, 1] """
        # Chuyển về dải [0, 1]
        gen_norm = (gen_img + 1.0) / 2.0
        gt_norm = (gt_img + 1.0) / 2.0
        
        # Chỉ lấy lỗi ở vùng mask
        mse = torch.sum(((gen_norm - gt_norm) * mask) ** 2) / (torch.sum(mask) * 3 + 1e-8)
        
        if mse == 0:
            return float('inf')
            
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr.item()

    def run_evaluation_batch(self, generated_batch, ground_truth_batch, mask_batch):
        """
        Chạy full pipeline đánh giá cho 1 batch
        """
        batch_size = generated_batch.shape[0]
        lpips_scores = []
        psnr_scores = []
        
        for i in range(batch_size):
            g = generated_batch[i:i+1]
            gt = ground_truth_batch[i:i+1]
            m = mask_batch[i:i+1]
            
            l = self.evaluate_lpips(g, gt, m)
            p = self.evaluate_psnr(g, gt, m)
            
            lpips_scores.append(l)
            psnr_scores.append(p)
            
        return {
            "lpips_mean": np.mean([x for x in lpips_scores if x >= 0]),
            "psnr_mean": np.mean(psnr_scores)
        }

if __name__ == "__main__":
    print("[Testing] LPIPS/PSNR Hair Evaluator Initialization...")
    evaluator = HairEvaluator(device='cpu')
    
    # Fake Dummy Tensors [-1, 1]
    gen = torch.rand(1, 3, 512, 512) * 2 - 1
    gt = torch.rand(1, 3, 512, 512) * 2 - 1
    mask = torch.zeros(1, 1, 512, 512)
    mask[:, :, 100:400, 150:350] = 1.0 # Fake mask
    
    metrics = evaluator.run_evaluation_batch(gen, gt, mask)
    print(f"Metrics: {metrics}")
