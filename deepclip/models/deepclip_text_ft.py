import torch
import torch.nn.functional as F
from timm.models import register_model
from deepclip.data.text import tokenize
from deepclip.layers.lora import LORA_TARGETS, inject_lora
from deepclip.models.deepclip import DeepClipBase, create_model, model_kwargs
from deepclip.models.text_encoder import DeepTextEncoder


class DeepClipTextFT(DeepClipBase):
    """
    DeepClip with the CLIP text encoder trained by LoRA instead of a frozen text_table.

    The text encoder is a part of the model: every step embeds the 21 captions (all_captions) with it, so the
    caption embeddings move together with the image embeddings. The CLIP weights stay frozen, only a low-rank
    update (r, alpha, drop) of the attention (q, k, v, out) of every text block is learned.

    Learnable parameters: image_encoder, LoRA of text_encoder and 2 scalars
        t_itc, b_itc : log temperature and bias of the loss (SigLIP)

    In eval mode under torch.no_grad() the caption embeddings are computed once and cached
    (cleared by train(), load_state_dict() and moving the model).
    """

    def __init__(
        self,
        clip_name: str = "ViT-B/16",
        embed_dim: int | None = None,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_drop: float = 0.1,
        lora_targets: tuple = LORA_TARGETS,
        load_openai: bool = True,
        **kwargs,
    ):
        """
        Args:
            clip_name: OpenAI CLIP text encoder ("ViT-B/16", "ViT-L/14").
            embed_dim: Only checked against clip_name when given.
            lora_r, lora_alpha, lora_drop: Rank, scale (alpha / r) and input dropout of LoRA.
            lora_targets: Projections of the text attention with LoRA, a subset of ("q", "k", "v", "out").
            load_openai: Load the OpenAI CLIP text weights (training). False when the whole text encoder is
                loaded from a DeepClipTextFT checkpoint.
            **kwargs: Arguments of DeepImageEncoder.
        """

        super().__init__(clip_name, embed_dim, **kwargs)

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_drop = lora_drop
        self.lora_targets = tuple(lora_targets)

        if load_openai:
            self.text_encoder = DeepTextEncoder.from_openai(clip_name)
        else:
            self.text_encoder = DeepTextEncoder.from_config(clip_name)

        if self.text_encoder.embed_dim != self.embed_dim:
            raise ValueError(
                f"Text encoder of {clip_name} gives {self.text_encoder.embed_dim}, not {self.embed_dim}"
            )

        self.num_lora_params = inject_lora(
            self.text_encoder, lora_r, lora_alpha, lora_drop, self.lora_targets
        )

        # The text transformer is causal and <|endoftext|> is the last token of a caption, so the padding after
        # the longest caption changes nothing: the tokens are cut there
        tokens = tokenize(self.captions)
        max_len = int(tokens.argmax(dim=-1).max()) + 1
        self.register_buffer("caption_tokens", tokens[:, :max_len], persistent=False)

        self._text_cache = None
        self.register_load_state_dict_post_hook(
            lambda module, _: module._clear_text_cache()
        )

    def _clear_text_cache(self):
        self._text_cache = None

    def train(self, mode: bool = True):
        self._clear_text_cache()
        return super().train(mode)

    def _apply(self, *args, **kwargs):
        # .to(), .cuda(), .half(), ...
        self._clear_text_cache()
        return super()._apply(*args, **kwargs)

    def text_features(self) -> torch.Tensor:
        """(num_captions, embed_dim) L2 normalized caption embeddings of the (LoRA) text encoder, float32"""

        if self._text_cache is not None:
            return self._text_cache

        text_feats = F.normalize(self.text_encoder(self.caption_tokens).float(), dim=-1)

        if not self.training and not torch.is_grad_enabled():
            self._text_cache = text_feats

        return text_feats


def _create_deepclip_text_ft(variant, pretrained=False, **kwargs):
    # the whole text encoder (CLIP weights + LoRA) comes from the checkpoint when pretrained
    kwargs.setdefault("load_openai", not pretrained)
    return create_model(DeepClipTextFT, variant, pretrained, **kwargs)


@register_model
def deepclip_b0_text_ft(pretrained=False, **kwargs) -> DeepClipTextFT:
    return _create_deepclip_text_ft(
        "deepclip_b0_text_ft", pretrained, **model_kwargs("b0", **kwargs)
    )


@register_model
def deepclip_b5_text_ft(pretrained=False, **kwargs) -> DeepClipTextFT:
    return _create_deepclip_text_ft(
        "deepclip_b5_text_ft", pretrained, **model_kwargs("b5", **kwargs)
    )
