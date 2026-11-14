import torch
import torch.nn as nn
from typing import List
from deepclip.layers.attnpool import AttentionPool
from deepclip.layers.cross_attention import CrossAttnBlock
from deepclip.layers.featextractor import FeatExtractor
from deepclip.layers.transformer import GCViT
from deepclip.layers.weight_init import trunc_normal_

NORM_LAYERS = {
    "rmsnorm": nn.RMSNorm,
    "layernorm": nn.LayerNorm,
}

# Same values as ms_eff_gcvit_b0 / ms_eff_gcvit_b5 of DeepGuard
IMAGE_ENCODER_CFGS = {
    "b0": dict(
        model_name="tf_efficientnet_b0.ns_jft_in1k",
        img_size=[224, 224],
        l_dim=24,
        h_dim=256,
        l_depths=[2, 2, 4, 2],
        h_depths=[4],
        l_windows=[7, 7, 14, 7],
        h_windows=[7],
        l_heads=[1, 2, 4, 8],
        h_heads=[4],
        l_ratio=[4, 4, 4, 4],
        h_ratio=[4],
        h_drop=0.05,
        l_attn_drop=0.05,
        l_drop_path=0.1,
        h_drop_path=0.05,
    ),
    "b5": dict(
        model_name="tf_efficientnet_b5.ns_jft_in1k",
        img_size=[384, 384],
        l_dim=48,
        h_dim=512,
        l_depths=[2, 2, 6, 2],
        h_depths=[6],
        l_windows=[12, 12, 24, 12],
        h_windows=[12],
        l_heads=[2, 4, 8, 16],
        h_heads=[16],
        l_ratio=[3, 3, 3, 3],
        h_ratio=[3],
        h_drop=0.1,
        l_attn_drop=0.1,
        l_drop_path=0.15,
        h_drop_path=0.1,
    ),
}


def get_norm_layer(image_norm: str):
    if image_norm not in NORM_LAYERS:
        raise ValueError(
            f"image_norm must be one of {list(NORM_LAYERS)}: {image_norm!r}"
        )

    return NORM_LAYERS[image_norm]


def build_metadata(
    feat_extractor: FeatExtractor, img_size: List[int], block_idx: int
) -> dict:
    """Backbone feature (channels, resolution) of the block block_idx"""

    idx_map = {6: 4, 4: 3}
    feature_idx = idx_map.get(block_idx, block_idx)

    info = feat_extractor.backbone.feature_info[feature_idx]

    meta = {
        "block_idx": block_idx,
        "feature_idx": feature_idx,
        "in_chs": info["num_chs"],
        "input_resolution": img_size[0] // info["reduction"],
    }
    return meta


class DeepImageEncoder(nn.Module):
    """
    Image encoder of DeepClip, based on MultiScaleEffGCViT of DeepGuard (https://github.com/HanMoonSub/DeepGuard).

    EfficientNet backbone (ImageNet pretrained) -> low-level GCViT + high-level GCViT -> concat (num_features)
    -> norm (image_norm) -> @ image_projection -> (B, embed_dim). The output is not L2 normalized.
    norm + image_projection mirror ln_final + text_projection of DeepTextEncoder.

    Differences from MultiScaleEffGCViT: both branches are always used (no use_low / use_high), the
    classification head is replaced by norm + image_projection, and every norm of the GCViT branches and norm is
    image_norm (rmsnorm or layernorm). The BatchNorm of the backbone is not changed.
    """

    def __init__(
        self,
        model_name: str,
        img_size: List[int],
        embed_dim: int,
        l_block_idx: int = 1,
        h_block_idx: int = 6,
        l_dim: int = 24,
        h_dim: int = 256,
        l_depths: List[int] = [2, 2, 4, 2],
        h_depths: List[int] = [4],
        l_windows: List[int] = [7, 7, 14, 7],
        h_windows: List[int] = [7],
        l_heads: List[int] = [1, 2, 4, 8],
        h_heads: List[int] = [4],
        l_ratio: List[float] = [4, 4, 4, 4],
        h_ratio: List[float] = [4],
        l_drop: float = 0.0,
        h_drop: float = 0.05,
        l_attn_drop: float = 0.05,
        h_attn_drop: float = 0,
        l_drop_path: float = 0.1,
        h_drop_path: float = 0.05,
        image_norm: str = "rmsnorm",
        **kwargs,
    ):
        """
        Args:
            model_name: Name of the timm backbone (always ImageNet pretrained).
            img_size: Input image resolution as [Height, Width].
            embed_dim: Output dimension (the text embedding dimension of the CLIP text encoder).
            l_block_idx: Index of the backbone block to extract low-level (high-resolution) features.
            h_block_idx: Index of the backbone block to extract high-level (low-resolution) features.
            l_*, h_*: GCViT settings of the low-level and high-level branches (see DeepGuard).
            image_norm: "rmsnorm" or "layernorm", used in the GCViT branches and norm.
            **kwargs: Ignored (e.g. arguments added by timm).
        """

        super().__init__()
        norm_layer = get_norm_layer(image_norm)

        self.image_norm = image_norm
        self.embed_dim = embed_dim

        self.feat_extractor = FeatExtractor(
            model_name=model_name,
            img_size=img_size,
            l_block_idx=l_block_idx,
            h_block_idx=h_block_idx,
        )

        self.l_meta = self._build_metadata(img_size, l_block_idx)
        self.h_meta = self._build_metadata(img_size, h_block_idx)

        self.l_gcvit = GCViT(
            in_chs=self.l_meta["in_chs"],
            input_resolution=self.l_meta["input_resolution"],
            dim=l_dim,
            depths=l_depths,
            window_size=l_windows,
            mlp_ratio=l_ratio,
            num_heads=l_heads,
            drop=l_drop,
            attn_drop=l_attn_drop,
            drop_path=l_drop_path,
            norm_layer=norm_layer,
        )
        self.h_gcvit = GCViT(
            in_chs=self.h_meta["in_chs"],
            input_resolution=self.h_meta["input_resolution"],
            dim=h_dim,
            depths=h_depths,
            window_size=h_windows,
            mlp_ratio=h_ratio,
            num_heads=h_heads,
            drop=h_drop,
            attn_drop=h_attn_drop,
            drop_path=h_drop_path,
            norm_layer=norm_layer,
        )

        self.num_features = l_dim * (2 ** (len(l_depths) - 1)) + h_dim * (
            2 ** (len(h_depths) - 1)
        )

        self.norm = norm_layer(self.num_features)
        self.image_projection = nn.Parameter(torch.empty(self.num_features, embed_dim))

        self._init_proj_weights()

    def _init_proj_weights(self):
        if getattr(self.norm, "bias", None) is not None:
            nn.init.constant_(self.norm.bias, 0.0)
        nn.init.constant_(self.norm.weight, 1.0)
        trunc_normal_(self.image_projection, std=0.02)

    def _build_metadata(self, img_size, block_idx):
        return build_metadata(self.feat_extractor, img_size, block_idx)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table", "bn", "norm"}

    def forward_features(self, x):
        """(B,3,H,W) -> (B,num_features), before norm and image_projection"""

        feat = self.feat_extractor(x)  # (B,C,H,W)

        l_out = self.l_gcvit(feat[self.l_meta["feature_idx"]])  # (B,D_l)
        h_out = self.h_gcvit(feat[self.h_meta["feature_idx"]])  # (B,D_h)

        return torch.cat([l_out, h_out], dim=-1)  # (B,num_features)

    def forward(self, x):
        x = self.norm(self.forward_features(x))  # (B,num_features)

        return x @ self.image_projection  # (B,embed_dim)


# v2 = v1 GCViT branches + cross attention (x_*) + CLIP attention pool (pool_*)
IMAGE_ENCODER_V2_CFGS = {
    "b0": dict(
        **IMAGE_ENCODER_CFGS["b0"],
        x_depth=1,
        x_heads=4,
        x_ratio=4,
        x_drop=0.05,
        x_attn_drop=0.0,
        x_drop_path=0.05,
        pool_heads=4,
    ),
    "b5": dict(
        **IMAGE_ENCODER_CFGS["b5"],
        x_depth=1,
        x_heads=16,
        x_ratio=3,
        x_drop=0.1,
        x_attn_drop=0.0,
        x_drop_path=0.1,
        pool_heads=8,
    ),
}


class DeepImageEncoderV2(nn.Module):
    """
    Image encoder of DeepClip v2. The GCViT branches are the same as DeepImageEncoder, but they are fused
    by cross attention instead of GAP + concat:

        low-level GCViT  (encoder) -> tokens L (B,N_l,D_l)   key / value
        high-level GCViT (decoder) -> tokens H (B,N_h,D_h)   query
        H = CrossAttnBlock(H + pos_q, L + pos_kv) x x_depth  -> (B,N_h,D_h)
        AttentionPool (CLIP): query = mean of H              -> (B,embed_dim)

    The output is not L2 normalized. D_l may differ from D_h: the key and value are projected D_l -> D_h.
    """

    def __init__(
        self,
        model_name: str,
        img_size: List[int],
        embed_dim: int,
        l_block_idx: int = 1,
        h_block_idx: int = 6,
        l_dim: int = 24,
        h_dim: int = 256,
        l_depths: List[int] = [2, 2, 4, 2],
        h_depths: List[int] = [4],
        l_windows: List[int] = [7, 7, 14, 7],
        h_windows: List[int] = [7],
        l_heads: List[int] = [1, 2, 4, 8],
        h_heads: List[int] = [4],
        l_ratio: List[float] = [4, 4, 4, 4],
        h_ratio: List[float] = [4],
        l_drop: float = 0.0,
        h_drop: float = 0.05,
        l_attn_drop: float = 0.05,
        h_attn_drop: float = 0,
        l_drop_path: float = 0.1,
        h_drop_path: float = 0.05,
        x_depth: int = 1,
        x_heads: int | None = None,
        x_ratio: float = 4.0,
        x_drop: float | None = None,
        x_attn_drop: float = 0.0,
        x_drop_path: float = 0.0,
        pool_heads: int | None = None,
        image_norm: str = "rmsnorm",
        **kwargs,
    ):
        """
        Args:
            model_name, img_size, embed_dim, l_*, h_*, image_norm: same as DeepImageEncoder.
            x_depth: Number of cross attention blocks (query: high-level tokens, key / value: low-level tokens).
            x_heads: Heads of the cross attention (default: h_heads[-1]).
            x_ratio: MLP ratio of the cross attention blocks.
            x_drop: Proj / MLP dropout of the cross attention blocks (default: h_drop).
            x_attn_drop: Attention dropout of the cross attention blocks.
            x_drop_path: Stochastic depth of the last cross attention block (linear from 0).
            pool_heads: Heads of the attention pool (default: D_h // 64, as CLIP).
            **kwargs: Ignored (e.g. arguments added by timm).
        """

        super().__init__()
        norm_layer = get_norm_layer(image_norm)

        self.image_norm = image_norm
        self.embed_dim = embed_dim

        self.feat_extractor = FeatExtractor(
            model_name=model_name,
            img_size=img_size,
            l_block_idx=l_block_idx,
            h_block_idx=h_block_idx,
        )

        self.l_meta = build_metadata(self.feat_extractor, img_size, l_block_idx)
        self.h_meta = build_metadata(self.feat_extractor, img_size, h_block_idx)

        self.l_gcvit = GCViT(
            in_chs=self.l_meta["in_chs"],
            input_resolution=self.l_meta["input_resolution"],
            dim=l_dim,
            depths=l_depths,
            window_size=l_windows,
            mlp_ratio=l_ratio,
            num_heads=l_heads,
            drop=l_drop,
            attn_drop=l_attn_drop,
            drop_path=l_drop_path,
            norm_layer=norm_layer,
            global_pool=False,
        )
        self.h_gcvit = GCViT(
            in_chs=self.h_meta["in_chs"],
            input_resolution=self.h_meta["input_resolution"],
            dim=h_dim,
            depths=h_depths,
            window_size=h_windows,
            mlp_ratio=h_ratio,
            num_heads=h_heads,
            drop=h_drop,
            attn_drop=h_attn_drop,
            drop_path=h_drop_path,
            norm_layer=norm_layer,
            global_pool=False,
        )

        # Dimension and number of the tokens of the last stage of each branch (no downsample in the last stage)
        l_out_dim = l_dim * 2 ** (len(l_depths) - 1)
        h_out_dim = h_dim * 2 ** (len(h_depths) - 1)
        l_tokens = (self.l_meta["input_resolution"] // 2 ** (len(l_depths) - 1)) ** 2
        h_tokens = (self.h_meta["input_resolution"] // 2 ** (len(h_depths) - 1)) ** 2

        self.num_features = h_out_dim
        self.num_tokens = h_tokens

        # Cross attention is permutation invariant, so the tokens get a learnable absolute position
        self.pos_q = nn.Parameter(torch.zeros(1, h_tokens, h_out_dim))
        self.pos_kv = nn.Parameter(torch.zeros(1, l_tokens, l_out_dim))

        dpr = [x.item() for x in torch.linspace(0, x_drop_path, x_depth)]
        self.x_blocks = nn.ModuleList(
            [
                CrossAttnBlock(
                    q_dim=h_out_dim,
                    kv_dim=l_out_dim,
                    num_heads=x_heads or h_heads[-1],
                    mlp_ratio=x_ratio,
                    drop=h_drop if x_drop is None else x_drop,
                    attn_drop=x_attn_drop,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(x_depth)
            ]
        )

        self.attnpool = AttentionPool(
            num_tokens=h_tokens,
            dim=h_out_dim,
            num_heads=pool_heads or max(h_out_dim // 64, 1),
            out_dim=embed_dim,
        )

        self._init_head_weights()

    def _init_head_weights(self):
        trunc_normal_(self.pos_q, std=0.02)
        trunc_normal_(self.pos_kv, std=0.02)
        for m in self.x_blocks.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
        self.attnpool.init_weights()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {
            "relative_position_bias_table",
            "bn",
            "norm",
            "pos_q",
            "pos_kv",
            "positional_embedding",
        }

    def forward_features(self, x):
        """(B,3,H,W) -> (B,N_h,D_h) high-level tokens after the cross attention, before the attention pool"""

        feat = self.feat_extractor(x)

        l_tok = (
            self.l_gcvit(feat[self.l_meta["feature_idx"]]) + self.pos_kv
        )  # (B,N_l,D_l)
        h_tok = (
            self.h_gcvit(feat[self.h_meta["feature_idx"]]) + self.pos_q
        )  # (B,N_h,D_h)

        for block in self.x_blocks:
            h_tok = block(h_tok, l_tok)

        return h_tok

    def forward(self, x):
        return self.attnpool(self.forward_features(x))  # (B,embed_dim)
