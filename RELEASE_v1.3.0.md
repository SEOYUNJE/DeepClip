DeepClip v1.3.0 adds **attention pooling** to the image encoder of DeepClipV2, released as `deepclip_b0_v2_ap` and
`deepclip_b5_v2_ap`. Available on PyPI as [`deepclip==1.3.0`](https://pypi.org/project/deepclip/1.3.0/).

In `*_v2` the last feature map of the image encoder (GCViT) is reduced by global average pooling. In `*_v2_ap` the new
**GCViTV2** replaces it with the **attention pooling of CLIP** (`AttentionPool2d`: a query from the mean token attends to
every spatial token), so the image embedding can weight the face regions that carry forgery artifacts. The image encoder
also uses **LayerNorm** instead of RMSNorm, and the LoRA of the CLIP text encoder is raised to `r=8`, `alpha=16`. At
inference the caption embeddings are computed once and cached, so each image costs only the image encoder
(0.84G FLOPs for b0, 13.38G for b5).

**Full Changelog**: https://github.com/SEOYUNJE/DeepClip/compare/v1.2.0...v1.3.0

## Pretrained weights

The files below are downloaded automatically with `pretrained=True`.

| File | Image encoder | Input | CLIP text encoder | Embed dim | Params |
| ---- | ------------- | ----- | ----------------- | --------- | ------ |
| `deepclip_b0_v2_ap.pt` | ms_eff_gcvit_b0 (attention pooling) | 224 × 224 | ViT-B/16 (LoRA, r=8) | 512 | 73.07M |
| `deepclip_b5_v2_ap.pt` | ms_eff_gcvit_b5 (attention pooling) | 384 × 384 | ViT-L/14 (LoRA, r=8) | 768 | 176.64M |

- Each checkpoint holds the whole text encoder (CLIP weights + LoRA), so no OpenAI CLIP download is needed when
  loading pretrained weights.
- Trained on FaceForensics++ with the configs `deepclip/config/deepclip_b0_v2_ap.yaml` and
  `deepclip/config/deepclip_b5_v2_ap.yaml`.

## What's changed

- ✨ CLIP `AttentionPool2d` layer and `GCViTV2`, a GCViT with attention pooling instead of global average pooling
- ✨ `image_pool` option (`gap` / `attn`) of `DeepImageEncoder`
- ✨ `deepclip_b0_v2_ap` and `deepclip_b5_v2_ap` registered in timm (LayerNorm, attention pooling, LoRA `r=8`, `alpha=16`)
- 🔧 Training configs `deepclip_b0_v2_ap.yaml` and `deepclip_b5_v2_ap.yaml`
- 📝 README: `*_v2_ap` in *Architecture* and *Model Zoo*, evaluation results of `deepclip_b0_v2` / `deepclip_b5_v2`, and
  the preprocessed dataset [`KoreaPeter/faceforensics-c23-forgery-annotation`](https://huggingface.co/datasets/KoreaPeter/faceforensics-c23-forgery-annotation)
  on the Hugging Face Hub

## Quick start

```bash
pip install deepclip==1.3.0
```

```python
import torch
import timm
import deepclip  # registers the DeepClip models in timm

model = timm.create_model("deepclip_b0_v2_ap", pretrained=True).eval()  # or "deepclip_b5_v2_ap"

x = torch.randn(1, 3, 224, 224)  # 384 x 384 for deepclip_b5_v2_ap
with torch.no_grad():
    p_fake = model(x).sigmoid()  # (1, 1) fake probability
```

## Sources and licenses

- DeepClip code and weights: MIT
- Image backbone: `tf_efficientnet_b0.ns_jft_in1k` / `tf_efficientnet_b5.ns_jft_in1k` from [timm](https://github.com/huggingface/pytorch-image-models) (Apache-2.0)
- Text encoder: [OpenAI CLIP](https://github.com/openai/CLIP) ViT-B/16 / ViT-L/14 (MIT)
- Training data: [FaceForensics++](https://github.com/ondyari/FaceForensics) (check its terms of use; research purposes only)
