from typing import Union, Optional
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusion_policy_3d.model.diffusion.positional_embedding import SinusoidalPosEmb
from timm.layers import Mlp

logger = logging.getLogger(__name__)


class ISS(nn.Module):
    """Implicit State Space model for point cloud prediction"""
    
    def __init__(
        self,
        input_dim: int,
        pc_embed_dim: int = 128,
        hidden_size: int = 512,
        output_dim: int = 64,
        mlp_ratio: float = 4.0,
        n_skip_steps: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.n_skip_steps = n_skip_steps

        # Action projection
        self.action_proj = Mlp(
            in_features=input_dim,
            hidden_features=int(hidden_size * mlp_ratio),
            out_features=hidden_size,
            act_layer=nn.GELU,
        )

        # Adaptive projection for action sequence
        self.action_ada_proj = Mlp(
            in_features=hidden_size * n_skip_steps,
            hidden_features=int(hidden_size * mlp_ratio),
            out_features=hidden_size * 3,
            act_layer=nn.GELU,
        )

        # Point cloud projection
        self.pc_proj = Mlp(
            in_features=pc_embed_dim,
            hidden_features=int(hidden_size * mlp_ratio),
            out_features=hidden_size,
            act_layer=nn.GELU,
        )

        # Gates projection
        self.gates_proj = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.output_proj = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            out_features=output_dim,
            act_layer=nn.GELU,
        )

        # Normalization layer
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, cur_action: torch.Tensor, pc_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cur_action: Current action sequence (B, T, input_dim)
            pc_cond: Point cloud condition (B, pc_embed_dim)
        
        Returns:
            pc_pred: Predicted point cloud embedding (B, output_dim)
        """
        # Project action
        action_embed = self.action_proj(cur_action)  # (B, T, H)
        action_embed = einops.rearrange(
            action_embed, 'b t d -> b (t d)', t=self.n_skip_steps
        )  # (B, T*H)
        
        # Project point cloud condition
        pc_embed = self.pc_proj(pc_cond)  # (B, H)
        
        # Generate adaptive parameters
        ada_feat = self.action_ada_proj(action_embed)  # (B, 3*H)
        beta, gamma, gates = torch.chunk(ada_feat, 3, dim=-1)  # each (B, H)

        # Apply adaptive modulation
        pc_embed = pc_embed * (1.0 + gamma) + beta

        # Apply gated transformation
        pc_embed = pc_embed + gates * self.gates_proj(pc_embed)
        
        # Normalize and project to output
        pc_emb = self.norm(pc_embed)
        pc_pred = self.output_proj(pc_emb)

        return pc_pred




        