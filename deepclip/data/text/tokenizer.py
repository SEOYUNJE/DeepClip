"""
BPE tokenizer of OpenAI CLIP (https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py, MIT License).

The captions of the template are plain ASCII, so the ftfy text cleaning of the original is not used.
Unlike clip.tokenize, a caption longer than the context length raises an error instead of being truncated.
"""

import gzip
import html
from functools import lru_cache
from pathlib import Path
import regex as re
import torch

BPE_PATH = Path(__file__).with_name("bpe_simple_vocab_16e6.txt.gz")
CONTEXT_LENGTH = 77

@lru_cache()
def bytes_to_unicode() -> dict:
    """Maps every byte to a printable unicode character, which BPE works on"""

    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0

    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1

    return dict(zip(bs, [chr(c) for c in cs]))

def get_pairs(word: tuple) -> set:
    return set(zip(word[:-1], word[1:]))

def clean_text(text: str) -> str:
    text = html.unescape(html.unescape(text))

    return re.sub(r"\s+", " ", text).strip()

class SimpleTokenizer:
    def __init__(self, bpe_path: Path = BPE_PATH):
        self.byte_encoder = bytes_to_unicode()

        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = [tuple(merge.split()) for merge in merges[1:49152 - 256 - 2 + 1]]

        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        vocab += ["".join(merge) for merge in merges]
        vocab += ["<|startoftext|>", "<|endoftext|>"]

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

        self.sot_token = self.encoder["<|startoftext|>"]
        self.eot_token = self.encoder["<|endoftext|>"]

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))

            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0

            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break

                new_word.extend(word[i:j])
                i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)

            if len(word) == 1:
                break

            pairs = get_pairs(word)

        word = " ".join(word)
        self.cache[token] = word

        return word

    def encode(self, text: str) -> list:
        bpe_tokens = []

        for token in re.findall(self.pat, clean_text(text).lower()):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))

        return bpe_tokens

@lru_cache()
def get_tokenizer() -> SimpleTokenizer:
    return SimpleTokenizer()

def tokenize(texts, context_length: int = CONTEXT_LENGTH) -> torch.Tensor:
    """
    texts: a caption or a list of captions
    return: (N, context_length) long tensor, <|startoftext|> + BPE tokens + <|endoftext|>, padded with 0
    """

    if isinstance(texts, str):
        texts = [texts]

    tokenizer = get_tokenizer()
    all_tokens = [[tokenizer.sot_token] + tokenizer.encode(text) + [tokenizer.eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Caption has {len(tokens)} tokens, more than the context length {context_length}: {texts[i]!r}")

        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
