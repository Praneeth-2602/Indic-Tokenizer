from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

from .tokenizer import IndicTokenizer


class IndicHFTokenizer:
    """Small Transformers-style wrapper.

    This intentionally implements the common encode/decode/save/load surface without
    pretending to be a full `PreTrainedTokenizerFast` replacement.
    """

    model_input_names = ["input_ids"]

    def __init__(self, tokenizer: Optional[IndicTokenizer] = None):
        self.tokenizer = tokenizer or IndicTokenizer()

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "IndicHFTokenizer":
        return cls(IndicTokenizer.from_pretrained(path))

    def save_pretrained(self, path: Union[str, Path]) -> tuple[str, str]:
        target = Path(path)
        self.tokenizer.save_pretrained(target)
        return (str(target / "vocab.json"), str(target / "merges.txt"))

    def encode(self, text: str, **_: object) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: Iterable[int], **_: object) -> str:
        return self.tokenizer.decode(token_ids)

    def batch_encode_plus(self, texts: Iterable[str], **_: object) -> dict[str, list[list[int]]]:
        return {"input_ids": self.tokenizer.encode_batch(texts)}

    def __call__(self, text: Union[str, list[str]], **kwargs: object) -> dict[str, object]:
        if isinstance(text, str):
            return {"input_ids": self.encode(text, **kwargs)}
        return self.batch_encode_plus(text, **kwargs)
