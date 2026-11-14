import torch.nn as nn
import torch.nn.functional as F
from typing import Type
from .mlp import Mlp
from .drop import DropPath


class CrossAttention(nn.Module):
    """
        q  : (B, N_q, q_dim)   query tokens (high-level branch)
        kv : (B, N_kv, kv_dim) key / value tokens (low-level branch)
        out: (B, N_q, q_dim)

    kv_dim may differ from q_dim: the key and value are projected kv_dim -> q_dim, so only the head_dim of
    the query and key has to match. N_q and N_kv may differ too (the attention map is N_q x N_kv).
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if q_dim % num_heads != 0:
            raise ValueError(
                f"q_dim {q_dim} must be divisible by num_heads {num_heads}"
            )

        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.attn_drop = attn_drop

        self.q = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.kv = nn.Linear(kv_dim, q_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(q_dim, q_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv):
        B, N_q, C = q.shape
        N_kv = kv.shape[1]

        # (B,N_q,C) -> (B,num_heads,N_q,head_dim)
        q = self.q(q).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        # (B,N_kv,kv_dim) -> (2,B,num_heads,N_kv,head_dim)
        kv = (
            self.kv(kv)
            .reshape(B, N_kv, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, N_q, C)  # (B,N_q,C)
        return self.proj_drop(self.proj(x))


class CrossAttnBlock(nn.Module):
    """
    Decoder block without self attention (pre-norm):
        q = q + DropPath(CrossAttn(norm_q(q), norm_kv(kv)))
        q = q + DropPath(Mlp(norm2(q)))

        q  : (B, N_q, q_dim)
        kv : (B, N_kv, kv_dim)
        out: (B, N_q, q_dim)
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.norm_q = norm_layer(q_dim)
        self.norm_kv = norm_layer(kv_dim)
        self.attn = CrossAttention(
            q_dim,
            kv_dim,
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(q_dim)
        self.mlp = Mlp(
            in_dims=q_dim,
            hidden_dims=int(q_dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, q, kv):
        q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_kv(kv)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q
