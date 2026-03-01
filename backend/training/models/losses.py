import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (SupConLoss).
    Dành cho Stage 1: Kéo các điểm nhúng (embeddings) của các ảnh patch có cùng nhãn vật lý
    (VD: cùng độ xoăn 'curly') lại gần nhau trên hypersphere, và đẩy các patch khác nhãn ra xa.
    Dự trên paper: "Supervised Contrastive Learning" (Khósla et al.)
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        features: shape (B, ... , Embed_dim) - cần được L2-normalize trước khi truyền vào.
        labels: Nhãn vật lý của tệp (VD: Nhãn curliness). Shape: (B,)
        mask: Bảng ma trận mask tùy chỉnh n*n nếu labels=None.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` shape nên là (B, Multi-views, Embed_dim). '
                             'Nếu chỉ có 1 view, thêm 1 chiều dư ở giữa: features.unsqueeze(1).')
                             
        batch_size = features.shape[0]
        
        if labels is not None and mask is not None:
            raise ValueError('Chỉ được nạp 1 trong 2: labels hoặc mask.')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Số lượng nhãn phải bằng Batch size')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Tính toán phép chiếu vô hướng (Cosine Similarity do feature đã đc L2-norm)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # Dùng trick trừ phần tử số lớn nhất để tránh tràn số softmax (numerical stability)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tạo mask các góc âm đường chéo
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Exp sum
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Trừ tính toán mean của postive logits
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

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

    def forward(self, generated_img, target_img):
        # Rescale từ [-1,1] → ImageNet normalize trước khi truyền vào VGG
        gen_norm = self._to_imagenet_norm(generated_img)
        tar_norm = self._to_imagenet_norm(target_img)
        
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
            
        masked_diff = diff * mask
        # Tính normalize loss sum theo diện tích pixel của mask để không bị lệch biên độ gradient
        loss = masked_diff.sum() / (mask.sum() + 1e-6)
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
    Sử dụng ResNet50 pretrained ImageNet để trích xuất face embedding (512-dim).
    
    Workflow:
    1. Từ ảnh RGB + hair mask → lấy inverse mask (vùng mặt)
    2. Crop bounding box vùng mặt → resize 112×112
    3. Feed qua frozen ResNet50 → embedding 2048-dim (L2-normalized)
    
    LƯU Ý: Module này hoàn toàn frozen, KHÔNG tham gia backprop.
    """
    def __init__(self, device='cuda'):
        super(FaceFeatureExtractor, self).__init__()
        self.device = device
        
        # Dùng ResNet50 pretrained ImageNet làm face feature extractor
        # Lý do: ArcFace iresnet50 chỉ có ONNX weights (không load vào PyTorch),
        # còn ResNet50 ImageNet đã có sẵn weights tốt cho face structure features.
        # Output: 2048-dim vector (KHÔNG dùng Linear projection vì random weights
        # sẽ phá hủy semantic features từ ImageNet pretrained backbone)
        from torchvision import models
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],  # Output: (B, 2048, 1, 1)
            nn.Flatten(),                    # (B, 2048)
        )
        
        # Freeze toàn bộ — KHÔNG tham gia training
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _crop_face_region(self, images: torch.Tensor, hair_masks: torch.Tensor) -> torch.Tensor:
        """
        Crop vùng mặt từ ảnh dựa trên inverse hair mask.
        
        Args:
            images: (B, 3, H, W) — ảnh RGB, range [-1, 1]
            hair_masks: (B, 1, H, W) — hair mask (1 = tóc, 0 = không tóc)
        
        Returns:
            face_crops: (B, 3, 112, 112) — face crops đã resize, range [-1, 1]
        """
        B = images.shape[0]
        face_crops = []
        
        face_mask = (1.0 - hair_masks)  # Inverse: 1 = mặt, 0 = tóc
        
        for i in range(B):
            mask_2d = face_mask[i, 0]  # (H, W)
            
            # Tìm bounding box của vùng mặt
            y_indices = torch.where(mask_2d > 0.5)[0]
            x_indices = torch.where(mask_2d > 0.5)[1]
            
            if len(y_indices) == 0 or len(x_indices) == 0:
                # Không tìm thấy vùng mặt → dùng toàn bộ ảnh
                crop = F.interpolate(images[i:i+1], size=(112, 112), mode='bilinear', align_corners=False)
                face_crops.append(crop)
                continue
            
            y_min, y_max = y_indices.min().item(), y_indices.max().item()
            x_min, x_max = x_indices.min().item(), x_indices.max().item()
            
            # Thêm padding 10%
            h_pad = max(1, int((y_max - y_min) * 0.1))
            w_pad = max(1, int((x_max - x_min) * 0.1))
            y_min = max(0, y_min - h_pad)
            y_max = min(images.shape[2], y_max + h_pad)
            x_min = max(0, x_min - w_pad)
            x_max = min(images.shape[3], x_max + w_pad)
            
            # Crop + resize
            crop = images[i:i+1, :, y_min:y_max, x_min:x_max]
            crop = F.interpolate(crop, size=(112, 112), mode='bilinear', align_corners=False)
            face_crops.append(crop)
        
        return torch.cat(face_crops, dim=0)  # (B, 3, 112, 112)
    
    # KHÔNG dùng @torch.no_grad() — gradient cần flow qua input (decoded_img)
    # để Identity Loss thực sự update UNet weights.
    # Backbone đã frozen (requires_grad=False) → weights không bị update,
    # nhưng gradient cho INPUT tensor vẫn được tính → flow ngược về UNet.
    def forward(self, images: torch.Tensor, hair_masks: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất face embedding từ ảnh + hair mask.
        
        Args:
            images: (B, 3, H, W) — range [-1, 1]
            hair_masks: (B, 1, H, W) — hair mask (1 = tóc)
        
        Returns:
            embeddings: (B, 2048) — L2-normalized face embeddings
        """
        # Crop vùng mặt → (B, 3, 112, 112)
        face_crops = self._crop_face_region(images, hair_masks)
        
        # Forward qua backbone
        embeddings = self.backbone(face_crops)
        
        # L2 normalize
        embeddings = F.normalize(embeddings, dim=1)
        
        return embeddings

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
    
    # Test SupCon Loss
    # features shape: (B, n_views, embed_dim)
    feats = torch.nn.functional.normalize(torch.rand(B, 2, 128), dim=2)
    labels = torch.tensor([0, 0])
    supcon = SupConLoss()
    print("SupCon Loss:", supcon(feats, labels).item())

    # Test Identity Cosine Loss (2048-dim, matching FaceFeatureExtractor output)
    id_loss_fn = IdentityCosineLoss()
    gen_id = torch.rand(B, 2048)
    tar_id = torch.rand(B, 2048)
    print("Identity Cosine Loss:", id_loss_fn(gen_id, tar_id).item())

    print("Losses setups are functional!")
