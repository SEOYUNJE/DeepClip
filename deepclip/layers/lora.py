"""
LoRA (https://arxiv.org/abs/2106.09685) for the attention of the CLIP text transformer.

LoRAMultiheadAttention keeps the parameter names of nn.MultiheadAttention (in_proj_weight, in_proj_bias,
out_proj.weight, out_proj.bias), so the text weights of an OpenAI checkpoint load into it unchanged.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LORA_TARGETS = ("q", "k", "v", "out")


class LoRAMultiheadAttention(nn.Module):
    """
    Self attention (batch_first) with frozen base weights and a low-rank update on q, k, v and out:

        W' = W + (alpha / r) * B @ A,   A: (r, D), B: (D, r)

    B starts at zero, so the output is the same as the base attention before training.

        inp: (B, L, D)
        out: (B, L, D)
    """

    def __init__(
        self,
        width: int,
        heads: int,
        r: int = 8,
        alpha: float = 16.0,
        drop: float = 0.0,
        targets: tuple = LORA_TARGETS,
    ):
        super().__init__()

        if width % heads != 0:
            raise ValueError(f"width {width} is not divisible by heads {heads}")
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        unknown = set(targets) - set(LORA_TARGETS)
        if unknown:
            raise ValueError(
                f"Unknown LoRA targets {sorted(unknown)}. Available: {list(LORA_TARGETS)}"
            )

        self.width = width
        self.heads = heads
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.targets = tuple(t for t in LORA_TARGETS if t in targets)

        # Same names as nn.MultiheadAttention
        self.in_proj_weight = nn.Parameter(torch.empty(3 * width, width))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * width))
        self.out_proj = nn.Linear(width, width)

        self.lora_drop = nn.Dropout(drop)
        for t in self.targets:
            setattr(self, f"lora_A_{t}", nn.Parameter(torch.empty(r, width)))
            setattr(self, f"lora_B_{t}", nn.Parameter(torch.zeros(width, r)))

        nn.init.xavier_uniform_(self.in_proj_weight)
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        # Same as peft: A ~ kaiming uniform, B = 0
        for t in self.targets:
            nn.init.kaiming_uniform_(getattr(self, f"lora_A_{t}"), a=math.sqrt(5))
            nn.init.zeros_(getattr(self, f"lora_B_{t}"))

    @classmethod
    def from_mha(
        cls, mha: nn.MultiheadAttention, **lora_kwargs
    ) -> "LoRAMultiheadAttention":
        """Copies the weights of a batch_first nn.MultiheadAttention"""

        if (
            not mha.batch_first
            or not mha._qkv_same_embed_dim
            or mha.in_proj_bias is None
        ):
            raise ValueError(
                "Only batch_first nn.MultiheadAttention with a packed in_proj and bias is supported"
            )

        attn = cls(mha.embed_dim, mha.num_heads, **lora_kwargs)
        attn.in_proj_weight.data.copy_(mha.in_proj_weight.data)
        attn.in_proj_bias.data.copy_(mha.in_proj_bias.data)
        attn.out_proj.load_state_dict(mha.out_proj.state_dict())

        return attn.to(mha.in_proj_weight.device, mha.in_proj_weight.dtype)

    def _lora(self, x, t):
        """(B,L,D) -> (B,L,D) low-rank update of target t"""
        A, B = getattr(self, f"lora_A_{t}"), getattr(self, f"lora_B_{t}")
        return F.linear(F.linear(x, A), B) * self.scaling

    def forward(self, query, key=None, value=None, need_weights=False, attn_mask=None):
        """Same call as nn.MultiheadAttention for self attention: returns (out, None)"""

        if need_weights:
            raise ValueError("need_weights is not supported")
        if (key is not None and key is not query) or (
            value is not None and value is not query
        ):
            raise ValueError("Only self attention (query is key is value) is supported")

        x = query
        Bsz, L, D = x.shape

        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        if self.targets:
            x_drop = self.lora_drop(x)
            if "q" in self.targets:
                q = q + self._lora(x_drop, "q")
            if "k" in self.targets:
                k = k + self._lora(x_drop, "k")
            if "v" in self.targets:
                v = v + self._lora(x_drop, "v")

        # (B,L,D) -> (B,H,L,hd)
        q, k, v = (
            t.reshape(Bsz, L, self.heads, D // self.heads).transpose(1, 2)
            for t in (q, k, v)
        )

        if attn_mask is not None and attn_mask.is_floating_point():
            attn_mask = attn_mask.to(q.dtype)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(Bsz, L, D)

        y = self.out_proj(out)
        if "out" in self.targets:
            y = y + self._lora(self.lora_drop(out), "out")

        return y, None

    def extra_repr(self):
        return f"width={self.width}, heads={self.heads}, r={self.r}, alpha={self.alpha}, targets={self.targets}"


def inject_lora(
    text_encoder: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    drop: float = 0.0,
    targets: tuple = LORA_TARGETS,
) -> int:
    """
    Replaces the attention of every block of text_encoder.transformer with LoRAMultiheadAttention (in place).
    Afterwards only the LoRA parameters of text_encoder are trainable.

    Returns the number of trainable (LoRA) parameters.
    """

    for block in text_encoder.transformer.resblocks:
        if not isinstance(block.attn, LoRAMultiheadAttention):
            block.attn = LoRAMultiheadAttention.from_mha(
                block.attn, r=r, alpha=alpha, drop=drop, targets=tuple(targets)
            )

    text_encoder.requires_grad_(False)
    n_trainable = 0
    for name, param in text_encoder.named_parameters():
        if ".lora_" in name:
            param.requires_grad_(True)
            n_trainable += param.numel()

    return n_trainable
