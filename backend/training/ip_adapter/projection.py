"""
IP-Adapter Projection Model
Converts CLIP image embeddings to UNet cross-attention format

Reference: https://github.com/tencent-ailab/IP-Adapter
"""

import torch
import torch.nn as nn


class ImageProjModel(nn.Module):
    """
    Projection Model để convert CLIP embeddings → UNet cross-attention tokens
    
    Architecture:
    - Linear projection: clip_dim → (num_tokens * cross_attention_dim)
    - Reshape thành (batch, num_tokens, cross_attention_dim)
    - LayerNorm
    """
    
    def __init__(
        self, 
        cross_attention_dim: int = 2048,  # SDXL uses 2048
        clip_embeddings_dim: int = 1280,  # CLIP ViT-H/14
        clip_extra_context_tokens: int = 4
    ):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        
        # Linear projection: clip_dim → num_tokens * cross_attention_dim
        self.proj = nn.Linear(
            clip_embeddings_dim, 
            clip_extra_context_tokens * cross_attention_dim
        )
        self.norm = nn.LayerNorm(cross_attention_dim)
    
    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeds: CLIP image embeddings (batch, clip_embeddings_dim)
            
        Returns:
            ip_tokens: IP-Adapter tokens (batch, num_tokens, cross_attention_dim)
        """
        # Project and reshape
        clip_extra_context_tokens = self.proj(image_embeds)
        clip_extra_context_tokens = clip_extra_context_tokens.reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        
        return clip_extra_context_tokens


class MLPProjModel(nn.Module):
    """
    Simpler MLP projection for IP-Adapter-Plus
    """
    
    def __init__(
        self, 
        cross_attention_dim: int = 2048,
        clip_embeddings_dim: int = 1280
    ):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            nn.GELU(),
            nn.Linear(clip_embeddings_dim, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim)
        )
    
    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.proj(image_embeds).unsqueeze(1)
