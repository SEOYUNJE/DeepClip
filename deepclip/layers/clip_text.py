"""
Text transformer blocks of OpenAI CLIP (https://github.com/openai/CLIP/blob/main/clip/model.py, MIT License).

The module names are the same as the original, so the text weights of an OpenAI checkpoint load with strict=True.
"""

from collections import OrderedDict
import torch
import torch.nn as nn

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    """
        inp: (B, L, D)
        out: (B, L, D)
    """
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(width * 4, width)),
        ]))
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, x, attn_mask=None):
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, need_weights=False, attn_mask=attn_mask)[0]
        x = x + self.mlp(self.ln_2(x))

        return x

class CLIPTransformer(nn.Module):
    """
        Causal text transformer: a token only attends to itself and the tokens before it.
        inp: (B, L, D)
        out: (B, L, D)
    """
    def __init__(self, width: int, layers: int, heads: int, context_length: int = 77):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])

        mask = torch.full((context_length, context_length), float("-inf")).triu_(1)
        # Not saved in the state_dict, as in the original
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x):
        L = x.shape[1]
        attn_mask = self.attn_mask[:L, :L].to(x.dtype)

        for block in self.resblocks:
            x = block(x, attn_mask)

        return x
