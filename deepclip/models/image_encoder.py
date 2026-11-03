import torch
import torch.nn as nn
from typing import List
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
        model_name = "tf_efficientnet_b0.ns_jft_in1k",
        img_size = [224,224],
        l_dim = 24,
        h_dim = 256,
        l_depths = [2,2,4,2],
        h_depths = [4],
        l_windows = [7,7,14,7],
        h_windows = [7],
        l_heads = [1,2,4,8],
        h_heads = [4],
        l_ratio = [4,4,4,4],
        h_ratio = [4],
        h_drop = 0.05,
        l_attn_drop = 0.05,
        l_drop_path = 0.1,
        h_drop_path = 0.05,
    ),
    "b5": dict(
        model_name = "tf_efficientnet_b5.ns_jft_in1k",
        img_size = [384,384],
        l_dim = 48,
        h_dim = 512,
        l_depths = [2,2,6,2],
        h_depths = [6],
        l_windows = [12,12,24,12],
        h_windows = [12],
        l_heads = [2,4,8,16],
        h_heads = [16],
        l_ratio = [3,3,3,3],
        h_ratio = [3],
        h_drop = 0.1,
        l_attn_drop = 0.1,
        l_drop_path = 0.15,
        h_drop_path = 0.1,
    ),
}

def get_norm_layer(image_norm: str):
    if image_norm not in NORM_LAYERS:
        raise ValueError(f"image_norm must be one of {list(NORM_LAYERS)}: {image_norm!r}")

    return NORM_LAYERS[image_norm]

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
            l_depths: List[int] = [2,2,4,2],
            h_depths: List[int] = [4],
            l_windows: List[int] = [7,7,14,7],
            h_windows: List[int] = [7],
            l_heads: List[int] = [1,2,4,8],
            h_heads: List[int] = [4],
            l_ratio: List[float] = [4,4,4,4],
            h_ratio: List[float] = [4],
            l_drop: float = 0.,
            h_drop: float = 0.05,
            l_attn_drop: float = 0.05,
            h_attn_drop: float = 0,
            l_drop_path: float = 0.1,
            h_drop_path: float = 0.05,
            image_norm: str = "rmsnorm",
            **kwargs
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
                            model_name = model_name,
                            img_size = img_size,
                            l_block_idx = l_block_idx,
                            h_block_idx = h_block_idx
                            )

        self.l_meta = self._build_metadata(img_size, l_block_idx)
        self.h_meta = self._build_metadata(img_size, h_block_idx)

        self.l_gcvit = GCViT(
                            in_chs = self.l_meta['in_chs'],
                            input_resolution = self.l_meta['input_resolution'],
                            dim = l_dim,
                            depths = l_depths,
                            window_size = l_windows,
                            mlp_ratio = l_ratio,
                            num_heads = l_heads,
                            drop = l_drop,
                            attn_drop = l_attn_drop,
                            drop_path = l_drop_path,
                            norm_layer = norm_layer,
                        )
        self.h_gcvit = GCViT(
                            in_chs = self.h_meta['in_chs'],
                            input_resolution = self.h_meta['input_resolution'],
                            dim = h_dim,
                            depths = h_depths,
                            window_size = h_windows,
                            mlp_ratio = h_ratio,
                            num_heads = h_heads,
                            drop = h_drop,
                            attn_drop = h_attn_drop,
                            drop_path = h_drop_path,
                            norm_layer = norm_layer,
                        )

        self.num_features = l_dim * (2 ** (len(l_depths) - 1)) + h_dim * (2 ** (len(h_depths) - 1))

        self.norm = norm_layer(self.num_features)
        self.image_projection = nn.Parameter(torch.empty(self.num_features, embed_dim))

        self._init_proj_weights()

    def _init_proj_weights(self):
        if getattr(self.norm, "bias", None) is not None:
            nn.init.constant_(self.norm.bias, 0.)
        nn.init.constant_(self.norm.weight, 1.0)
        trunc_normal_(self.image_projection, std=.02)

    def _build_metadata(self, img_size, block_idx):

        idx_map = {6: 4, 4: 3}
        feature_idx = idx_map.get(block_idx, block_idx)

        info = self.feat_extractor.backbone.feature_info[feature_idx]

        meta = {
            "block_idx": block_idx,
            "feature_idx": feature_idx,
            "in_chs": info['num_chs'],
            "input_resolution": img_size[0] // info['reduction']
        }
        return meta

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table','bn','norm'}

    def forward_features(self, x):
        """(B,3,H,W) -> (B,num_features), before norm and image_projection"""

        feat = self.feat_extractor(x) # (B,C,H,W)

        l_out = self.l_gcvit(feat[self.l_meta["feature_idx"]]) # (B,D_l)
        h_out = self.h_gcvit(feat[self.h_meta["feature_idx"]]) # (B,D_h)

        return torch.cat([l_out, h_out], dim=-1) # (B,num_features)

    def forward(self, x):
        x = self.norm(self.forward_features(x)) # (B,num_features)

        return x @ self.image_projection # (B,embed_dim)
