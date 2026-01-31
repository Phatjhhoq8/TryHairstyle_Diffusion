"""
IP-Adapter Attention Processor
Custom attention processor để inject image embeddings vào UNet cross-attention

Reference: https://github.com/tencent-ailab/IP-Adapter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IPAttnProcessor(nn.Module):
    """
    Attention Processor với IP-Adapter
    
    Thêm decoupled cross-attention cho image prompt:
    - Attention ban đầu với text embeddings 
    - Attention thứ 2 với image embeddings (IP-Adapter)
    - Kết hợp cả 2 với scale factor
    """
    
    def __init__(
        self, 
        hidden_size: int,
        cross_attention_dim: int,
        scale: float = 1.0,
        num_tokens: int = 4
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        
        # Learnable projection layers for IP-Adapter
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        """
        Process attention với IP-Adapter embeddings
        
        encoder_hidden_states chứa [text_embeds, ip_embeds] concatenated
        """
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        # Split encoder_hidden_states into text and image parts
        # IP tokens are the last num_tokens
        end_pos = encoder_hidden_states.shape[1] - self.num_tokens
        encoder_hidden_states_text = encoder_hidden_states[:, :end_pos, :]
        encoder_hidden_states_ip = encoder_hidden_states[:, end_pos:, :]
        
        # Text cross-attention
        key = attn.to_k(encoder_hidden_states_text)
        value = attn.to_v(encoder_hidden_states_text)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # IP-Adapter cross-attention
        key_ip = self.to_k_ip(encoder_hidden_states_ip)
        value_ip = self.to_v_ip(encoder_hidden_states_ip)
        
        key_ip = attn.head_to_batch_dim(key_ip)
        value_ip = attn.head_to_batch_dim(value_ip)
        
        attention_probs_ip = attn.get_attention_scores(query, key_ip, None)
        hidden_states_ip = torch.bmm(attention_probs_ip, value_ip)
        hidden_states_ip = attn.batch_to_head_dim(hidden_states_ip)
        
        # Combine text and IP attention với scale
        hidden_states = hidden_states + self.scale * hidden_states_ip
        
        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # Dropout
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


class AttnProcessor(nn.Module):
    """
    Default attention processor (no IP-Adapter)
    Used for self-attention layers
    """
    
    def __init__(self):
        super().__init__()
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states
