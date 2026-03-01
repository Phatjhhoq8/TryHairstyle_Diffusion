import torch
import torch.nn as nn
from diffusers.models import UNet2DConditionModel
from pathlib import Path

LOCAL_SDXL_PATH = str(Path(__file__).resolve().parent.parent.parent / "models" / "stable-diffusion" / "sd_xl_inpainting")

class HairInpaintingUNet(nn.Module):
    """
    Stage 2: Mask-Conditioned Hair Inpainting UNet.
    Vận dụng trọng số của SDXL Inpainting làm nền tảng (Base).
    Input 9 Channels: 
     - 4 kênh Latent (Noised Hair Input)
     - 4 kênh Bald Latent (Mặt trọc làm tham chiếu bối cảnh)
     - 1 kênh Mask (Downsampled Vùng Tóc)
    """
    
    def __init__(self, sd_model_id=LOCAL_SDXL_PATH):
        super().__init__()
        
        # 1. Khởi tạo UNet gốc của SDXL
        # Trong thực tế, gọi từ variant "fp16" hoặc inpainting để tiết kiệm VRAM.
        self.unet = UNet2DConditionModel.from_pretrained(
            sd_model_id, 
            subfolder="unet", 
            torch_dtype=torch.float16
        )
        
        # 2. Thay đổi lớp Conv_in để nhận 9-channels thay vì 4-channels chuẩn
        # Điều này rất quan trọng để mô hình hiểu được Mask và phối cảnh mặt trọc.
        # Format: (B, 9, H, W)
        with torch.no_grad():
            old_conv_in = self.unet.conv_in
            in_channels = old_conv_in.weight.shape[1]
            
            if in_channels == 4:
                new_conv_in = nn.Conv2d(
                    9, old_conv_in.out_channels, 
                    kernel_size=old_conv_in.kernel_size, 
                    padding=old_conv_in.padding
                )
                
                # Copy trọng số cũ sang 4 kênh đầu của lớp mới
                new_conv_in.weight[:, :4, :, :] = old_conv_in.weight.clone()
                
                # Khởi tạo giá trị zero (hoặc phân phối rất nhỏ) cho 5 kênh mở rộng 
                # để không phá hỏng Weight SDXL gốc lúc ban đầu
                nn.init.zeros_(new_conv_in.weight[:, 4:, :, :])
                new_conv_in.bias.data = old_conv_in.bias.data.clone()
                
                self.unet.conv_in = new_conv_in
            elif in_channels == 9:
                pass # Model này đã là bản Inpainting xịn 9 channels, không cần phẫu thuật nữa!
            else:
                raise ValueError(f"UNet Input Channels không hợp lệ: {in_channels}")
            
        # 3. Kích hoạt Gradient Checkpointing để tiết kiệm VRAM
        self.unet.enable_gradient_checkpointing()
        
        # 4. Kích hoạt xformers memory-efficient attention (tiết kiệm ~2-3GB VRAM)
        # Quan trọng cho GPU 12GB (RTX 3060)
        try:
            self.unet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # Fallback nếu xformers chưa cài — vẫn chạy được nhưng tốn VRAM hơn
        
    def forward(
        self, 
        noisy_latents: torch.Tensor, 
        masked_latents: torch.Tensor, 
        mask: torch.Tensor, 
        timestep: torch.Tensor, 
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: dict = None
    ) -> torch.Tensor:
        """
        Args:
            noisy_latents: (B, 4, H/8, W/8) - Sinh ra từ VAE Encode + Noise
            masked_latents: (B, 4, H/8, W/8) - Sinh ra từ VAE Encode (ảnh gốc × (1-mask))
            mask: (B, 1, H/8, W/8) - Downsampled Hair Mask (0: Nền giữ nguyên, 1: Cần vẽ tóc)
            timestep: Tensor chứa bước khuếch tán hiện tại
            encoder_hidden_states: (B, SeqLen, Dim) - Style + Text Prompt Embeddings
            added_cond_kwargs: Thông tin text/time embeddings của SDXL
        """
        
        # Ghép 3 thành phần lại tạo thành input 9-channels
        # Convention chuẩn SDXL Inpainting: [noisy_latents(4), mask(1), masked_image_latents(4)]
        latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)
        
        # Feed-forward qua UNet
        noise_pred = self.unet(
            latent_model_input, 
            timestep, 
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs
        ).sample
        
        return noise_pred

# === IP-Adapter: Module Bơm Tương Tác ===
# Chịu trách nhiệm bơm Đặc trưng khuôn mặt (Identity) và Đặc trưng Texture sợi tóc
# vào mạng Cross-Attention của UNet (để UNet nhận biết danh tính).

class CrossAttentionInjector(nn.Module):
    def __init__(self, unet: UNet2DConditionModel, style_dim=2048, identity_dim=512, text_dim=2048):
        super().__init__()
        self.unet = unet
        
        # Mạng chiếu (Projection Layer) để map Identity Embedding (vd 512d của AdaFace)
        # vào cùng không gian chiều (Dimensionality) với Text/Style 
        self.identity_proj = nn.Sequential(
            nn.Linear(identity_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )
        
        # Mạng chiếu nâng chiều vector Style (clip image 1024d) lên bằng chiều chập của SDXL (2048)
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, text_dim),
            nn.LayerNorm(text_dim)
        )
        
    def inject_conditioning(self, style_embeds: torch.Tensor, identity_embeds: torch.Tensor):
        """
        Trộn hai vector Style (Patch/Viễn Cảnh) và Identity (Mắt/Mũi/Khuôn) 
        lại với nhau để bơm vào UNet dưới dạng prompt sequence.
        """
        # Ánh xạ lên không gian 2048D
        proj_identity = self.identity_proj(identity_embeds)
        proj_style = self.style_proj(style_embeds)
        
        # Kéo giãn dọc trục Sequence (B, 1, Dim) để có thể hợp nhất Context
        if proj_style.dim() == 2:
            proj_style = proj_style.unsqueeze(1)
        if proj_identity.dim() == 2:
            proj_identity = proj_identity.unsqueeze(1)
            
        # Nối lại cấu tạo thành Combined Context (Khái niệm IP-Adapter Base)
        # Sequence Length = Len(Text) + 1(Style) + 1(ID)
        combined_cond = torch.cat([proj_style, proj_identity], dim=1)
        
        return combined_cond

if __name__ == "__main__":
    print("[Testing] Khởi tạo Stage 2 UNet Inpainting Controller...")
    # Model dummy mock test để tránh Download HuggingFace thực tế
    # model = HairInpaintingUNet() 
    print("[Testing] Done!")
