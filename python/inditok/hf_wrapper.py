from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Union

from .tokenizer import IndicTokenizer


class BatchEncoding(dict):
    """Small HuggingFace-compatible mapping returned by tokenizer calls."""

    def to(self, device: object) -> "BatchEncoding":
        for key, value in list(self.items()):
            if hasattr(value, "to"):
                self[key] = value.to(device)
        return self


class IndicHFTokenizer:
    model_input_names = ["input_ids", "attention_mask"]
    tokenizer_class = "IndicHFTokenizer"
    model_max_length = 512

    bos_token = "<s>"
    eos_token = "</s>"
    unk_token = "<unk>"
    pad_token = "<pad>"
    mask_token = "<mask>"

    def __init__(
        self,
        tokenizer: Optional[IndicTokenizer] = None,
        model_max_length: int = 512,
    ):
        self.tokenizer = tokenizer or IndicTokenizer()
        self.model_max_length = model_max_length

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "IndicHFTokenizer":
        tokenizer = IndicTokenizer.from_pretrained(path)
        config_path = Path(path) / "tokenizer_config.json"
        max_length = cls.model_max_length
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            max_length = int(config.get("model_max_length", max_length))
        return cls(tokenizer, model_max_length=max_length)

    def save_pretrained(self, path: Union[str, Path]) -> tuple[str, str, str]:
        target = Path(path)
        self.tokenizer.save_pretrained(target)
        config_path = target / "tokenizer_config.json"
        config = {
            "tokenizer_class": self.tokenizer_class,
            "model_max_length": self.model_max_length,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "mask_token": self.mask_token,
        }
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        return (str(target / "vocab.json"), str(target / "merges.txt"), str(config_path))

    def encode(self, text: str, lang: str | None = None, **kwargs: object) -> list[int]:
        return self.tokenizer.encode(text, lang=lang, code_mix=bool(kwargs.get("code_mix", False)))

    def decode(self, token_ids: Iterable[int], **_: object) -> str:
        return self.tokenizer.decode(token_ids)

    def batch_encode_plus(self, texts: Iterable[str], **kwargs: object) -> BatchEncoding:
        return self(list(texts), **kwargs)

    def __call__(
        self,
        text: Union[str, list[str]],
        lang: str | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs: object,
    ) -> BatchEncoding:
        texts = [text] if isinstance(text, str) else list(text)
        limit = max_length or self.model_max_length
        input_ids = [
            self.tokenizer.encode(item, lang=lang, code_mix=bool(kwargs.get("code_mix", False)))
            for item in texts
        ]
        if truncation:
            input_ids = [ids[:limit] for ids in input_ids]

        pad_to: int | None = None
        if padding is True or padding == "longest":
            pad_to = max((len(ids) for ids in input_ids), default=0)
        elif padding == "max_length":
            pad_to = limit

        pad_id = self.token_to_id(self.pad_token)
        attention_mask: list[list[int]] = []
        if pad_to is not None:
            padded = []
            for ids in input_ids:
                clipped = ids[:pad_to]
                pad_count = max(pad_to - len(clipped), 0)
                padded.append(clipped + [pad_id] * pad_count)
                attention_mask.append([1] * len(clipped) + [0] * pad_count)
            input_ids = padded
        else:
            attention_mask = [[1] * len(ids) for ids in input_ids]

        result = BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})
        if return_tensors:
            result = _convert_tensors(result, return_tensors)
        return result

    def token_to_id(self, token: str) -> int:
        return int(self.get_vocab().get(token, self.get_vocab().get(self.unk_token, 0)))

    def id_to_token(self, token_id: int) -> str:
        reverse = {idx: token for token, idx in self.get_vocab().items()}
        return reverse.get(int(token_id), self.unk_token)

    def vocab(self) -> dict[str, int]:
        return self.get_vocab()

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()

    def convert_tokens_to_ids(self, tokens: str | Iterable[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self.token_to_id(tokens)
        return [self.token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: int | Iterable[int]) -> str | list[str]:
        if isinstance(ids, int):
            return self.id_to_token(ids)
        return [self.id_to_token(token_id) for token_id in ids]


def _convert_tensors(result: BatchEncoding, return_tensors: str) -> BatchEncoding:
    if return_tensors == "np":
        import numpy as np

        return BatchEncoding({key: np.asarray(value) for key, value in result.items()})
    if return_tensors == "pt":
        import torch

        return BatchEncoding({key: torch.tensor(value) for key, value in result.items()})
    if return_tensors == "tf":
        import tensorflow as tf

        return BatchEncoding({key: tf.constant(value) for key, value in result.items()})
    raise ValueError("return_tensors must be one of None, 'np', 'pt', or 'tf'")
