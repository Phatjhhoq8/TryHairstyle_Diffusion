import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, feature_layers=[3, 8, 15, 22]): # relu1_2, relu2_2, relu3_3, relu4_3
        super(VGG16FeatureExtractor, self).__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg16 = models.vgg16(weights=weights).eval()
        self.features = vgg16.features
        self.layer_indices = feature_layers
        
        # Freezing vgg16 weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = []
        for i, module in enumerate(self.features):
            x = module(x)
            if i in self.layer_indices:
                outputs.append(x)
        return outputs

def gram_matrix(x):
    (b, c, h, w) = x.size()
    features = x.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram

class TextureConsistencyLoss(nn.Module):
    """
    Đo lường sự giống nhau về mặt Texture (hạt màu, lọn tóc...) giữa ảnh Generated và Ground Truth
    Dựa trên VGG16 Gram Matrix (Style Loss).
    
    LƯU Ý: Input có thể ở range [-1, 1] (từ Normalize([0.5],[0.5])) hoặc [0, 1].
    Hàm forward tự động rescale về ImageNet normalize trước khi truyền vào VGG.
    """
    def __init__(self):
        super(TextureConsistencyLoss, self).__init__()
        self.vgg_extractor = VGG16FeatureExtractor()
        
        # ImageNet normalization constants
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _to_imagenet_norm(self, img):
        """
        Rescale input từ [-1, 1] → ImageNet normalize.
        VGG16 cần input chuẩn hóa theo ImageNet (mean/std) để features có ý nghĩa.
        """
        # [-1, 1] → [0, 1]
        img_01 = (img + 1.0) / 2.0
        img_01 = img_01.clamp(0, 1)
        # [0, 1] → ImageNet normalize
        return (img_01 - self.imagenet_mean.to(img.device)) / self.imagenet_std.to(img.device)

    def forward(self, generated_img, target_img, mask=None):
        # Rescale từ [-1,1] → ImageNet normalize trước khi truyền vào VGG
        gen_norm = self._to_imagenet_norm(generated_img)
        tar_norm = self._to_imagenet_norm(target_img)
        
        # Áp mask để chỉ tính texture trong vùng tóc (tránh nhiễu từ background)
        if mask is not None:
            gen_norm = gen_norm * mask
            tar_norm = tar_norm * mask
        
        gen_features = self.vgg_extractor(gen_norm)
        target_features = self.vgg_extractor(tar_norm)

        loss = 0
        for gen_f, tar_f in zip(gen_features, target_features):
            gm_gen = gram_matrix(gen_f)
            gm_tar = gram_matrix(tar_f)
            loss += F.mse_loss(gm_gen, gm_tar)
            
        return loss

class MaskAwareLoss(nn.Module):
    """
    Mask-Aware Diffusion Loss (Gradient Locking).
    Chỉ trừng phạt các sai số bên trong vùng Mask (Tóc). Trả Gradient của Môi trường = 0.
    """
    def __init__(self, loss_type='l2'):
        super(MaskAwareLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, mask):
        """
        pred, target: (B, C, H, W)
        mask: (B, 1, H, W) - chuẩn hóa vùng tóc = 1, nền = 0.
        """
        if self.loss_type == 'l1':
            diff = torch.abs(pred - target)
        else: # l2
            diff = (pred - target) ** 2
        
        # Vùng tóc (mask=1): trọng số cao → focus inpaint
        # Vùng nền (mask=0): trọng số nhỏ → giữ background/khuôn mặt ổn định
        bg_weight = 0.1
        weights = mask + bg_weight * (1.0 - mask)
        weighted_diff = diff * weights
        loss = weighted_diff.sum() / (weights.sum() + 1e-6)
        return loss

class IdentityCosineLoss(nn.Module):
    """
    Stage 2: Nhận diện khuôn mặt (Identity Custom Loss)
    Tính Cosine Similarity giữa Embedding của ảnh nguyên bản (Ground Truth)
    và Embedding trích xuất từ ảnh được tạo ra (Generated Image).
    Mục tiêu để đẩy Similarity lên mức >= 0.90, khóa không cho mô hình bóp méo khung xương mặt.
    """
    def __init__(self):
        super(IdentityCosineLoss, self).__init__()
        # Hàm CosineEmbeddingLoss của PyTorch nhận vào x1, x2 và một tensor mục tiêu (y = 1 có nghĩa là giống nhau)
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, generated_embeds, target_embeds):
        """
        generated_embeds: (B, Embed_Dim) - Mảng Vector đặc trưng từ mô hình AdaFace (hoặc ArcFace)
        target_embeds: (B, Embed_Dim) - Mảng Vector Identity gốc
        """
        # Sinh 1 mảng các số 1 để nói với hàm Loss là "Tớ muốn 2 cái này 100% giống nhau"
        target_similarity = torch.ones(generated_embeds.size(0)).to(generated_embeds.device)
        return self.cosine_loss(generated_embeds, target_embeds, target_similarity)

class FaceFeatureExtractor(nn.Module):
    """
    Frozen Face Feature Extractor dùng cho Identity Loss trong training loop.
    Sử dụng InceptionResnetV1 pretrained VGGFace2 để trích xuất face embedding (512-dim).
    
    Workflow:
    1. Từ ảnh RGB + hair mask → lấy inverse mask (vùng mặt)
    2. Crop bounding box vùng mặt → resize 112×112
    3. Feed qua frozen InceptionResnetV1 → embedding 512-dim (L2-normalized)
    
    LƯU Ý: Module này hoàn toàn frozen, KHÔNG tham gia backprop.
    """
    def __init__(self, device='cuda'):
        super(FaceFeatureExtractor, self).__init__()
        self.device = device
        
        # Dùng InceptionResnetV1 pretrained trên VGGFace2 (3.3M ảnh khuôn mặt)
        # để trích xuất face embedding 512-dim.
        # Khác ResNet50 ImageNet (nhận diện vật thể), model này thực sự hiểu
        # sự khác biệt giữa các khuôn mặt → Identity Loss chính xác hơn.
        from facenet_pytorch import InceptionResnetV1
        self.backbone = InceptionResnetV1(pretrained='vggface2').eval()
        # Output: (B, 512) — L2-normalized face embedding
        
        # Freeze toàn bộ — KHÔNG tham gia training
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _crop_face_region(self, images: torch.Tensor, hair_masks: torch.Tensor) -> torch.Tensor:
        """
        Crop vùng mặt từ ảnh, loại bỏ tóc và giới hạn background.
        
        Chiến lược: Tìm vùng KHÔNG phải tóc trong nửa dưới + center ảnh,
        vì mặt thường nằm ở center-bottom. Nếu không đủ dữ liệu, 
        fallback lấy center crop 60%.
        
        Args:
            images: (B, 3, H, W) — ảnh RGB, range [-1, 1]
            hair_masks: (B, 1, H, W) — hair mask (1 = tóc, 0 = không tóc)
        
        Returns:
            face_crops: (B, 3, 160, 160) — face crops đã resize, range [-1, 1]
        """
        B, _, H, W = images.shape
        face_crops = []
        
        for i in range(B):
            hair_2d = hair_masks[i, 0]  # (H, W), 1=tóc, 0=không tóc
            
            # Tạo face region mask: không phải tóc + nằm trong center 80% ảnh
            # (loại bỏ viền ngoài chứa background)
            center_mask = torch.zeros_like(hair_2d)
            margin_y = int(H * 0.1)
            margin_x = int(W * 0.1)
            center_mask[margin_y:H-margin_y, margin_x:W-margin_x] = 1.0
            
            # Face = không phải tóc AND nằm trong center
            face_mask = (1.0 - hair_2d) * center_mask
            
            # Tìm bounding box vùng mặt
            y_indices = torch.where(face_mask > 0.5)[0]
            x_indices = torch.where(face_mask > 0.5)[1]
            
            if len(y_indices) < 100:
                # Không đủ pixel mặt → fallback center crop 60%
                cy, cx = H // 2, W // 2
                half_h, half_w = int(H * 0.3), int(W * 0.3)
                y_min = max(0, cy - half_h)
                y_max = min(H, cy + half_h)
                x_min = max(0, cx - half_w)
                x_max = min(W, cx + half_w)
            else:
                y_min, y_max = y_indices.min().item(), y_indices.max().item()
                x_min, x_max = x_indices.min().item(), x_indices.max().item()
                
                # Thêm padding 10%
                h_pad = max(1, int((y_max - y_min) * 0.1))
                w_pad = max(1, int((x_max - x_min) * 0.1))
                y_min = max(0, y_min - h_pad)
                y_max = min(H, y_max + h_pad)
                x_min = max(0, x_min - w_pad)
                x_max = min(W, x_max + w_pad)
            
            # Crop + resize
            crop = images[i:i+1, :, y_min:y_max, x_min:x_max]
            crop = F.interpolate(crop, size=(160, 160), mode='bilinear', align_corners=False)
            face_crops.append(crop)
        
        return torch.cat(face_crops, dim=0)  # (B, 3, 160, 160)
    
    def forward(self, images: torch.Tensor, hair_masks: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất face embedding từ ảnh + hair mask.
        
        Args:
            images: (B, 3, H, W) — range [-1, 1]
            hair_masks: (B, 1, H, W) — hair mask (1 = tóc)
        
        Returns:
            embeddings: (B, 512) — L2-normalized face embeddings
        """
        # Crop vùng mặt → (B, 3, 160, 160)
        face_crops = self._crop_face_region(images, hair_masks)
        
        # Forward qua backbone
        embeddings = self.backbone(face_crops)
        
        # L2 normalize
        embeddings = F.normalize(embeddings, dim=1)
        
        return embeddings

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon) — Khosla et al. 2020.
    Kéo các embedding cùng nhãn lại gần nhau, đẩy các embedding khác nhãn ra xa
    trong không gian projection. Dùng cho Stage 1 Texture Encoder training.
    
    Input:
        features: (B, n_views, embed_dim) — projection vectors đã L2-normalize
        labels: (B,) — nhãn curl cho mỗi sample
    
    LƯU Ý: Cần ít nhất 2 classes khác nhau trong batch để loss có ý nghĩa.
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (B, n_views, D) — hidden vectors đã L2-normalize
            labels: (B,) — ground truth labels
        Returns:
            loss: scalar
        """
        device = features.device
        batch_size = features.shape[0]
        n_views = features.shape[1]

        # Flatten views: (B * n_views, D)
        features = features.reshape(batch_size * n_views, -1)

        # Repeat labels cho mỗi view: (B * n_views,)
        labels = labels.contiguous().view(-1, 1)
        labels = labels.repeat(n_views, 1).squeeze(1)

        # Cosine similarity matrix: (N, N) where N = B * n_views
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Mask loại bỏ diagonal (tự so với chính mình)
        N = batch_size * n_views
        mask_self = torch.eye(N, dtype=torch.bool, device=device)
        similarity_matrix = similarity_matrix.masked_fill(mask_self, -1e9)

        # Mask positive pairs (cùng nhãn, khác index)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        mask_pos = labels_eq & ~mask_self  # Cùng nhãn nhưng không phải chính mình

        # Số lượng positive pairs cho mỗi anchor
        num_positives = mask_pos.sum(dim=1)  # (N,)

        # Nếu không có positive pair → skip (tránh NaN)
        valid = num_positives > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Log-sum-exp trick cho numerical stability
        logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Log(sum(exp(negatives + positives))) — mẫu số
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits.masked_fill(mask_self, 0.0)
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log(exp(positive) / sum(exp(all))) cho mỗi anchor
        log_prob = logits - log_sum_exp  # (N, N)

        # Chỉ lấy mean trên positive pairs
        mean_log_prob_pos = (mask_pos.float() * log_prob).sum(dim=1) / (num_positives.float() + 1e-8)

        # Loss = -mean over valid anchors
        loss = -mean_log_prob_pos[valid].mean()

        return loss


if __name__ == "__main__":
    B, C, H, W = 2, 3, 256, 256
    gen = torch.rand(B, C, H, W)
    tar = torch.rand(B, C, H, W)
    mask = torch.ones(B, 1, H, W)
    
    # Test Mask Aware
    ms_loss = MaskAwareLoss('l2')
    print("Mask-Aware Loss:", ms_loss(gen, tar, mask).item())
    
    # Test Texture Consistency
    tex_loss_fn = TextureConsistencyLoss()
    print("Texture Gram-Matrix Loss:", tex_loss_fn(gen, tar).item())
    


    # Test Identity Cosine Loss (512-dim, matching FaceFeatureExtractor output)
    id_loss_fn = IdentityCosineLoss()
    gen_id = torch.rand(B, 512)
    tar_id = torch.rand(B, 512)
    print("Identity Cosine Loss:", id_loss_fn(gen_id, tar_id).item())

    print("Losses setups are functional!")
