import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """
    AttentionPool2d of the OpenAI CLIP ModifiedResNet, for tokens instead of a (B,C,H,W) feature map.

        x   : (B, N, dim)
        out : (B, out_dim)

    The mean of the tokens is put in front of the tokens, a learnable positional embedding is added, and
    the mean token is the only query of a multi-head attention over [mean; tokens]. c_proj projects the
    output to out_dim (it replaces the norm + projection of a GAP head).
    """

    def __init__(self, num_tokens: int, dim: int, num_heads: int, out_dim: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.num_heads = num_heads
        self.positional_embedding = nn.Parameter(
            torch.randn(num_tokens + 1, dim) / dim**0.5
        )
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.c_proj = nn.Linear(dim, out_dim)

    def init_weights(self):
        """Same as the attnpool init of the OpenAI CLIP ModifiedResNet"""
        std = self.c_proj.in_features**-0.5
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.c_proj):
            nn.init.normal_(proj.weight, std=std)
            nn.init.zeros_(proj.bias)

    def forward(self, x):
        B, N, C = x.shape
        x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)  # (B,N+1,C)
        x = x + self.positional_embedding.to(x.dtype)

        def heads(t):  # (B,L,C) -> (B,num_heads,L,head_dim)
            return t.reshape(
                B, t.shape[1], self.num_heads, C // self.num_heads
            ).transpose(1, 2)

        q = heads(self.q_proj(x[:, :1]))  # query: the mean token only
        k = heads(self.k_proj(x))
        v = heads(self.v_proj(x))

        x = F.scaled_dot_product_attention(q, k, v)  # (B,num_heads,1,head_dim)
        x = x.transpose(1, 2).reshape(B, C)
        return self.c_proj(x)  # (B,out_dim)
