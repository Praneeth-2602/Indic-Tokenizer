from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

try:
    from ._inditok import IndicTokenizer as _NativeIndicTokenizer
except ModuleNotFoundError:  # pragma: no cover - exercised only before maturin builds the extension
    from ._fallback import IndicTokenizer as _NativeIndicTokenizer
else:
    if not hasattr(_NativeIndicTokenizer, "fertility"):  # pragma: no cover - stale local extension
        from ._fallback import IndicTokenizer as _NativeIndicTokenizer


class IndicTokenizer:
    """Ergonomic Python wrapper around the Rust tokenizer core."""

    def __init__(
        self,
        vocab_path: Optional[Union[str, Path]] = None,
        merges_path: Optional[Union[str, Path]] = None,
    ):
        vocab = str(vocab_path) if vocab_path is not None else None
        merges = str(merges_path) if merges_path is not None else None
        self._inner = _NativeIndicTokenizer(vocab, merges)

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "IndicTokenizer":
        obj = cls.__new__(cls)
        obj._inner = _NativeIndicTokenizer.from_pretrained(str(path))
        return obj

    def save_pretrained(self, path: Union[str, Path]) -> None:
        self._inner.save_pretrained(str(path))

    def encode(self, text: str, lang: str | None = None) -> list[int]:
        return list(self._inner.encode(text, lang))

    def encode_with_tokens(self, text: str, lang: str | None = None):
        return self._inner.encode_with_tokens(text, lang)

    def decode(self, ids: Iterable[int]) -> str:
        return self._inner.decode([int(token_id) for token_id in ids])

    def encode_batch(self, texts: Iterable[str], lang: str | None = None) -> list[list[int]]:
        return [list(ids) for ids in self._inner.encode_batch(list(texts), lang)]

    def normalize(self, text: str, lang: str | None = None) -> str:
        return self._inner.normalize(text, lang)

    def pre_tokenize(self, text: str, lang: str | None = None) -> list[str]:
        return list(self._inner.pre_tokenize(text, lang))

    def fertility(self, texts: Iterable[str], lang: str | None = None) -> dict[str, object]:
        report = self._inner.fertility(list(texts), lang)
        return {
            "fertility": float(report.fertility),
            "total_tokens": int(report.total_tokens),
            "total_words": int(report.total_words),
            "total_sentences": int(report.total_sentences),
        }

    @property
    def vocab_size(self) -> int:
        return int(self._inner.vocab_size())

    def __call__(self, text: Union[str, list[str]], **_: object) -> dict[str, object]:
        if isinstance(text, str):
            return {"input_ids": self.encode(text)}
        return {"input_ids": self.encode_batch(text)}
