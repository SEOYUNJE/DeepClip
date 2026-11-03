import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from timm.models import register_model, build_model_with_cfg
from deepclip.data.text import PAIR_ORDER, REAL_TEXT, FAKE_TEXT, all_captions, single_pair_caption, tokenize
from deepclip.models.image_encoder import DeepImageEncoder, IMAGE_ENCODER_CFGS
from deepclip.models.text_encoder import DeepTextEncoder

# Same clamp as OpenAI CLIP / open_clip: the logit scale (temperature) is at most 100
MAX_LOG_SCALE = math.log(100)

# Text embedding dimension of each CLIP text encoder
CLIP_EMBED_DIMS = {
    "ViT-B/16": 512,
    "ViT-L/14": 768,
}

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1, # Real or Fake
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.875,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        **kwargs
    }

default_cfgs = {
    'deepclip_b0_r1_t1': _cfg(input_size = (3,224,224)),
    'deepclip_b0_r1_t2': _cfg(input_size = (3,224,224)),
    'deepclip_b0_r2_t2': _cfg(input_size = (3,224,224)),
    'deepclip_b5_r1_t1': _cfg(input_size = (3,384,384)),
    'deepclip_b5_r1_t2': _cfg(input_size = (3,384,384)),
    'deepclip_b5_r2_t2': _cfg(input_size = (3,384,384)),
}

def available_models() -> List[str]:
    """Returns the names of DeepClip models"""
    return list(default_cfgs.keys())

class DeepClip(nn.Module):
    """
    DeepImageEncoder aligned with the frozen CLIP text embeddings of the face forgery captions.

    The text encoder is only run once in __init__: every caption which the collate can make (all_captions) is
    embedded and kept in the text_table buffer (L2 normalized). Training only looks up rows of text_table.

        text_table[REAL_IDX] : REAL_TEXT  (REAL caption of itc_loss, real caption of cls_loss)
        text_table[FAKE_IDX] : FAKE_TEXT  (fake caption of cls_loss)
        text_table[pair_idx] : the 20 captions with one (region, type) pair in PAIR_ORDER (pair_map)

    Learnable parameters: image_encoder and 5 scalars
        t_itc, b_itc : log temperature and bias of itc_loss (SigLIP)
        t_cls        : log temperature of cls_loss (CLIP)
        s_itc, s_cls : uncertainty weighting of the two losses (weight = exp(-s))
    """

    REAL_IDX = 0
    FAKE_IDX = 1

    def __init__(
            self,
            max_regions: int,
            max_types: int,
            clip_name: str = "ViT-B/16",
            embed_dim: int | None = None,
            build_text_table: bool = True,
            text_device: str | None = None,
            **kwargs
            ):
        """
        Args:
            max_regions: Maximum number of regions in a caption (1 or 2).
            max_types: Maximum number of types per region in a caption (1 or 2).
            clip_name: OpenAI CLIP text encoder ("ViT-B/16", "ViT-L/14").
            embed_dim: Only checked against clip_name when given.
            build_text_table: Run the CLIP text encoder to fill text_table. False when text_table is loaded
                from a DeepClip checkpoint.
            text_device: Device to build text_table on (default: cuda if available).
            **kwargs: Arguments of DeepImageEncoder.
        """

        super().__init__()

        if clip_name not in CLIP_EMBED_DIMS:
            raise ValueError(f"Unknown CLIP text encoder {clip_name!r}. Available: {list(CLIP_EMBED_DIMS)}")
        if embed_dim is not None and embed_dim != CLIP_EMBED_DIMS[clip_name]:
            raise ValueError(f"embed_dim of {clip_name} is {CLIP_EMBED_DIMS[clip_name]}, not {embed_dim}")

        self.max_regions = max_regions
        self.max_types = max_types
        self.clip_name = clip_name
        self.embed_dim = CLIP_EMBED_DIMS[clip_name]

        self.image_encoder = DeepImageEncoder(embed_dim=self.embed_dim, **kwargs)

        self.t_itc = nn.Parameter(torch.tensor(math.log(10.)))
        self.b_itc = nn.Parameter(torch.tensor(-10.))
        self.t_cls = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.s_itc = nn.Parameter(torch.zeros(()))
        self.s_cls = nn.Parameter(torch.zeros(()))

        # caption -> row of text_table. Rebuilt from all_captions, so it is not saved in the state_dict
        self.captions = all_captions(max_regions, max_types)
        self.caption_to_idx = {caption: idx for idx, caption in enumerate(self.captions)}
        assert self.captions[self.REAL_IDX] == REAL_TEXT and self.captions[self.FAKE_IDX] == FAKE_TEXT

        pair_idx = [self.caption_to_idx[single_pair_caption(region, type_)] for region, type_ in PAIR_ORDER]
        self.register_buffer("pair_idx", torch.tensor(pair_idx, dtype=torch.long), persistent=False)

        self.register_buffer("text_table", torch.zeros(len(self.captions), self.embed_dim))
        self.text_table_time = None

        if build_text_table:
            start = time.time()
            self.text_table.copy_(self._build_text_table(text_device))
            self.text_table_time = time.time() - start

    @torch.no_grad()
    def _build_text_table(self, device: str | None = None, batch_size: int = 256) -> torch.Tensor:
        """Embeds every caption with a temporary CLIP text encoder -> (num_captions, embed_dim), L2 normalized"""

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        text_encoder = DeepTextEncoder.from_openai(self.clip_name).to(device)

        tokens = tokenize(self.captions)
        embs = [text_encoder(tokens[i:i + batch_size].to(device)).float().cpu()
                for i in range(0, len(tokens), batch_size)]

        del text_encoder
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return F.normalize(torch.cat(embs), dim=-1)

    @property
    def num_captions(self) -> int:
        return len(self.captions)

    @torch.no_grad()
    def clamp_logit_scales_(self):
        """Called after every optimizer step"""
        self.t_itc.clamp_(max=MAX_LOG_SCALE)
        self.t_cls.clamp_(max=MAX_LOG_SCALE)

    def encode_image(self, x):
        """(B,3,H,W) -> (B,embed_dim), L2 normalized"""
        return F.normalize(self.image_encoder(x), dim=-1)

    def itc_logits(self, img_emb, text_idx):
        """(B,embed_dim), (U,) rows of text_table -> (B,U) logits of itc_loss"""
        txt_emb = self.text_table[text_idx] # (U,embed_dim)
        return self.t_itc.exp() * img_emb @ txt_emb.t() + self.b_itc

    def cls_logits(self, img_emb):
        """(B,embed_dim) -> (B,) fake logits = t_cls * (cos_fake - cos_real)"""
        direction = self.text_table[self.FAKE_IDX] - self.text_table[self.REAL_IDX] # (embed_dim,)
        return self.t_cls.exp() * img_emb @ direction

    def pair_logits(self, img_emb):
        """(B,embed_dim) -> (B,20) itc logits of the 20 captions with one pair (pair_map)"""
        return self.itc_logits(img_emb, self.pair_idx)

    def forward(self, x):
        """(B,3,H,W) -> (B,1) fake logits, same as the output of DeepGuard"""
        return self.cls_logits(self.encode_image(x)).unsqueeze(-1)

def _create_deepclip(variant, pretrained=False, **kwargs):

    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)

    pretrained_cfg = default_cfgs[variant]

    if pretrained and not pretrained_cfg['url']:
        raise ValueError(f"No pretrained weights are released for {variant} yet")

    # text_table comes from the checkpoint when pretrained
    kwargs.setdefault('build_text_table', not pretrained)

    return build_model_with_cfg(
        DeepClip,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        **kwargs
    )

def _model_kwargs(backbone, max_regions, max_types, **kwargs):
    clip_name = {"b0": "ViT-B/16", "b5": "ViT-L/14"}[backbone]
    # Values in kwargs (e.g. cfg.model of the yaml) override the defaults
    return {**IMAGE_ENCODER_CFGS[backbone], "clip_name": clip_name, **kwargs,
            "max_regions": max_regions, "max_types": max_types}

@register_model
def deepclip_b0_r1_t1(pretrained=False, **kwargs) -> DeepClip:
    return _create_deepclip("deepclip_b0_r1_t1", pretrained, **_model_kwargs("b0", 1, 1, **kwargs))

@register_model
def deepclip_b0_r1_t2(pretrained=False, **kwargs) -> DeepClip:
    return _create_deepclip("deepclip_b0_r1_t2", pretrained, **_model_kwargs("b0", 1, 2, **kwargs))

@register_model
def deepclip_b0_r2_t2(pretrained=False, **kwargs) -> DeepClip:
    return _create_deepclip("deepclip_b0_r2_t2", pretrained, **_model_kwargs("b0", 2, 2, **kwargs))

@register_model
def deepclip_b5_r1_t1(pretrained=False, **kwargs) -> DeepClip:
    return _create_deepclip("deepclip_b5_r1_t1", pretrained, **_model_kwargs("b5", 1, 1, **kwargs))

@register_model
def deepclip_b5_r1_t2(pretrained=False, **kwargs) -> DeepClip:
    return _create_deepclip("deepclip_b5_r1_t2", pretrained, **_model_kwargs("b5", 1, 2, **kwargs))

@register_model
def deepclip_b5_r2_t2(pretrained=False, **kwargs) -> DeepClip:
    return _create_deepclip("deepclip_b5_r2_t2", pretrained, **_model_kwargs("b5", 2, 2, **kwargs))
