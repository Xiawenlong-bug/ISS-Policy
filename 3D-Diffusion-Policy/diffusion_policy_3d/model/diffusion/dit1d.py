from typing import Union, Optional
import logging
from numpy import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.layers import Mlp 
from diffusion_policy_3d.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization with conditioning (AdaLN)
    
    Uses external embedding to generate channel-wise scale/shift for adaptive LayerNorm modulation.
    """
    def __init__(self, embed_dim, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, elementwise_affine=False)
        self.scale = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, num_channels, bias=True)
        )
        self.shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, num_channels, bias=True)
        )
    
    def forward(self, x, embed):
        """
        x: (B, T, C)
        embed: (B, embed_dim) or (B, T, embed_dim)
        """
        # Normalize
        x = self.ln(x)
        # Adaptive scaling and shifting
        if len(embed.shape) == 2:
            embed = embed.unsqueeze(1).expand(-1, x.shape[1], -1)  # (B, T, embed_dim)
        scale = self.scale(embed)  # (B, T, C)
        shift = self.shift(embed)  # (B, T, C)
        return scale * x + shift


class DiTBlock(nn.Module):
    """DiT block with AdaLN-Zero (two groups) and optional cross-attention.

    - Group 1 (α1, γ1, β1) modulates Self-Attention sublayer
    - Cross-Attention has no extra γ/β (only residual connection, no scaling)
    - Group 2 (α2, γ2, β2) modulates FFN sublayer
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Self-Attention
        self.ln_sa = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        # self.mlp = timm.models.mlp_block
        # FFN
        self.ln_ffn = nn.LayerNorm(hidden_size, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
    
    def forward(self, x, gamma1, beta1, alpha1, gamma2, beta2, alpha2):
        """
        x: (B, T, C)
        gamma1/beta1/alpha1: (B, 1, C) for SA
        gamma2/beta2/alpha2: (B, 1, C) for FFN
        """
        # Self-Attention with AdaLN modulate and gate
        y = self.ln_sa(x)
        y = y * (1.0 + gamma1) + beta1
        y, _ = self.attn(y, y, y)
        x = x + alpha1 * y
        # FFN with AdaLN modulate and gate
        y = self.ln_ffn(x)
        y = y * (1.0 + gamma2) + beta2
        y = self.mlp(y)
        x = x + alpha2 * y
        
        return x


class DiT1D(nn.Module):
    """Diffusion Transformer for 1D sequences"""
    def __init__(
        self,
        input_dim,
        state_embed_dim=128,
        pc_embed_dim=128,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        diffusion_step_embed_dim=256,
        horizon=16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.depth = depth
        self.fusion_prj = nn.Linear(hidden_size*2, hidden_size)
        self.input_proj = Mlp(
            in_features=input_dim,
            hidden_features=hidden_size,   
            out_features=hidden_size,
            act_layer=nn.GELU,             
        )

        self.state_proj = Mlp(
            in_features=state_embed_dim,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=nn.GELU,
        )

        self.pc_proj = Mlp(
            in_features=pc_embed_dim,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=nn.GELU,
        )       
        self.ada_proj = nn.Linear(hidden_size, 6*hidden_size)
        nn.init.zeros_(self.ada_proj.weight)
        nn.init.zeros_(self.ada_proj.bias)
        self.temporal_emb = nn.Embedding(num_embeddings=horizon, embedding_dim=hidden_size)
        nn.init.normal_(self.temporal_emb.weight, mean=0.0, std=0.02)

        # Timestep embedding
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(diffusion_step_embed_dim * 4, hidden_size),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            )
            for _ in range(depth)
        ])
        
        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, input_dim)

    
        
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
    
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond=None,
        global_cond=None,
        pc_cond=None,
        state_cond=None,
        **kwargs
    ):
        """
        sample: (B, T, input_dim) or (B, input_dim, T) - input sequence
        timestep: (B,) or int - diffusion timestep
        local_cond: (B, T, local_cond_dim) - local conditioning (not used in DiT)
        output: (B, T, input_dim) or (B, input_dim, T) - same shape as input
        """
        
        B, T, _ = sample.shape
        
        # Process timestep
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(B)
        
        timestep_embed = self.diffusion_step_encoder(timestep)
        state_embed = self.state_proj(state_cond)
        pc_embed = self.pc_proj(pc_cond)
        pc_embed_fusion = einops.repeat(pc_embed, 'b d -> b t d', t=T)
        ada_feat = self.ada_proj(timestep_embed + state_embed + pc_embed)
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = torch.chunk(ada_feat, 6, dim=-1)
        alpha1 = alpha1.unsqueeze(1)
        beta1 = beta1.unsqueeze(1)
        gamma1 = gamma1.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)
        beta2 = beta2.unsqueeze(1)
        gamma2 = gamma2.unsqueeze(1)
        x = self.input_proj(sample)  # (B, T, H)
        x = self.fusion_prj(torch.cat([x, pc_embed_fusion], dim=-1))
        t_emb = self.temporal_emb.weight[None, :, :] # (1, 16, H)
        x = x + t_emb
        for blk in self.blocks:
            x = blk(
                x,
                gamma1, beta1, alpha1,
                gamma2, beta2, alpha2,
            )
        x = self.final_norm(x)
        x = self.output_proj(x)  # (B, T, input_dim)
        
        
        return x

