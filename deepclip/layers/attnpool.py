"""
Attention pooling of OpenAI CLIP ModifiedResNet (https://github.com/openai/CLIP/blob/main/clip/model.py, MIT License).

The parameter names are the same as AttentionPool2d of CLIP (positional_embedding, q_proj, k_proj, v_proj, c_proj).
The attention is written batch first with F.scaled_dot_product_attention instead of F.multi_head_attention_forward,
which gives the same result.
"""

import torch
from torch import nn
import torch.nn.functional as F


class AttentionPool2d(nn.Module):
    """
    Pools a feature map into one vector: the mean token (+ positional embedding) is the query of a single
    multi-head attention over [mean token, HW tokens].

        inp: (B, H, W, C) channels last
        out: (B, output_dim)
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads: int,
        output_dim: int | None = None,
    ):
        """
        Args:
            spacial_dim: H (= W) of the input feature map.
            embed_dim: C of the input feature map.
            num_heads: Heads of the attention (embed_dim must be divisible by num_heads).
            output_dim: Output dimension (default: embed_dim).
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} is not divisible by num_heads {num_heads}"
            )

        self.spacial_dim = spacial_dim
        self.num_heads = num_heads

        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        # Same as CLIP.initialize_parameters for the attnpool
        std = self.c_proj.in_features**-0.5
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.c_proj):
            nn.init.normal_(proj.weight, std=std)
            nn.init.zeros_(proj.bias)
        nn.init.normal_(
            self.positional_embedding, std=self.positional_embedding.shape[1] ** -0.5
        )

    def forward(self, x):
        B, H, W, C = x.shape
        if H * W + 1 != self.positional_embedding.shape[0]:
            raise ValueError(
                f"AttentionPool2d was built for {self.spacial_dim}x{self.spacial_dim} feature maps, got {H}x{W}"
            )

        x = x.reshape(B, H * W, C)  # (B,HW,C)
        x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)  # (B,HW+1,C)
        x = x + self.positional_embedding.to(x.dtype)

        hd = C // self.num_heads
        q = (
            self.q_proj(x[:, :1]).reshape(B, 1, self.num_heads, hd).transpose(1, 2)
        )  # (B,h,1,hd)
        k = (
            self.k_proj(x).reshape(B, H * W + 1, self.num_heads, hd).transpose(1, 2)
        )  # (B,h,HW+1,hd)
        v = (
            self.v_proj(x).reshape(B, H * W + 1, self.num_heads, hd).transpose(1, 2)
        )  # (B,h,HW+1,hd)

        out = F.scaled_dot_product_attention(q, k, v)  # (B,h,1,hd)
        out = out.transpose(1, 2).reshape(B, C)  # (B,C)

        return self.c_proj(out)  # (B,output_dim)

    def extra_repr(self):
        return f"spacial_dim={self.spacial_dim}, num_heads={self.num_heads}"
