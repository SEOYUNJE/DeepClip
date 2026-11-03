from pathlib import Path
import torch
import torch.nn as nn
from deepclip.layers.clip_text import CLIPTransformer

CACHE_DIR = Path.home() / ".cache" / "deepclip"

# OpenAI checkpoints (https://github.com/openai/CLIP/blob/main/clip/clip.py). The folder name is the SHA256 of the file
_MODELS = {
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}
TEXT_KEYS = ("token_embedding", "positional_embedding", "transformer", "ln_final", "text_projection")

def available_text_encoders() -> list:
    return list(_MODELS)

def _read_openai_checkpoint(path: Path) -> dict:
    # The OpenAI checkpoints are TorchScript archives
    try:
        return torch.jit.load(str(path), map_location="cpu").state_dict()
    except RuntimeError:
        return torch.load(path, map_location="cpu", weights_only=True)

def load_openai_text_state_dict(name: str, cache_dir: Path = CACHE_DIR) -> dict:
    """
    Text weights (float32) of an OpenAI CLIP checkpoint.
    The checkpoint is downloaded once (its SHA256 is checked), and only its text weights are kept in
    cache_dir/clip_text_{name}.pt. The full checkpoint is deleted afterwards.
    """

    if name not in _MODELS:
        raise ValueError(f"Unknown CLIP text encoder {name!r}. Available: {available_text_encoders()}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_path = cache_dir / f"clip_text_{name.replace('/', '-')}.pt"

    if text_path.exists():
        return torch.load(text_path, map_location="cpu", weights_only=True)

    url = _MODELS[name]
    full_path = cache_dir / url.split("/")[-1]

    if not full_path.exists():
        print(f"Downloading CLIP {name} from {url}")
        torch.hub.download_url_to_file(url, str(full_path), hash_prefix=url.split("/")[-2])

    state_dict = _read_openai_checkpoint(full_path)
    text_state_dict = {k: v.float() for k, v in state_dict.items() if k.split(".")[0] in TEXT_KEYS}

    tmp_path = text_path.with_name(text_path.name + ".tmp")
    torch.save(text_state_dict, tmp_path)
    tmp_path.replace(text_path)
    full_path.unlink()

    return text_state_dict

class DeepTextEncoder(nn.Module):
    """
    Text encoder of OpenAI CLIP (https://github.com/openai/CLIP, MIT License): token + positional embedding ->
    causal transformer -> ln_final -> output at the <|endoftext|> token @ text_projection.
    The output is not L2 normalized.

        inp: (B, context_length) tokens
        out: (B, embed_dim)
    """

    def __init__(
            self,
            embed_dim: int,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            ):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.transformer = CLIPTransformer(width=width, layers=layers, heads=heads, context_length=context_length)
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))

        self._init_weights()

    def _init_weights(self):
        # Same as CLIP.initialize_parameters (only used when the weights are not loaded)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @classmethod
    def from_openai(cls, name: str, cache_dir: Path = CACHE_DIR) -> "DeepTextEncoder":
        """Builds the text encoder of an OpenAI CLIP model ("ViT-B/16", "ViT-L/14"), frozen and in eval mode"""

        state_dict = load_openai_text_state_dict(name, cache_dir)

        width = state_dict["ln_final.weight"].shape[0]
        layers = len({k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")})

        model = cls(
            embed_dim = state_dict["text_projection"].shape[1],
            context_length = state_dict["positional_embedding"].shape[0],
            vocab_size = state_dict["token_embedding.weight"].shape[0],
            width = width,
            heads = width // 64,
            layers = layers,
        )
        model.load_state_dict(state_dict, strict=True)

        return model.requires_grad_(False).eval()

    def forward(self, tokens):
        x = self.token_embedding(tokens) + self.positional_embedding[:tokens.shape[1]] # (B,L,D)
        x = self.transformer(x)
        x = self.ln_final(x)

        # <|endoftext|> has the largest token id
        x = x[torch.arange(x.shape[0], device=x.device), tokens.argmax(dim=-1)] # (B,D)

        return x @ self.text_projection # (B,embed_dim)
