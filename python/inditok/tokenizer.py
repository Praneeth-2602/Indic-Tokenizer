from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional, Union

try:
    from ._inditok import IndicTokenizer as _NativeIndicTokenizer
except ModuleNotFoundError:  # pragma: no cover - exercised only before maturin builds the extension
    from ._fallback import IndicTokenizer as _NativeIndicTokenizer
else:
    if not hasattr(_NativeIndicTokenizer, "get_vocab"):  # pragma: no cover - stale local extension
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
        path_obj = Path(path)
        if not path_obj.exists():
            path_obj = cls._load_from_hub(str(path))
        obj = cls.__new__(cls)
        obj._inner = _NativeIndicTokenizer.from_pretrained(str(path_obj))
        return obj

    @classmethod
    def _load_from_hub(cls, model_id: str) -> Path:
        cache_root = Path.home() / ".cache" / "inditok" / model_id.replace("/", "--")
        vocab_path = cache_root / "vocab.json"
        merges_path = cache_root / "merges.txt"
        if vocab_path.exists() and merges_path.exists():
            return cache_root

        cache_root.mkdir(parents=True, exist_ok=True)
        base_url = f"https://huggingface.co/{model_id}/resolve/main"
        try:
            for filename in ("vocab.json", "merges.txt"):
                cls._download_file(f"{base_url}/{filename}", cache_root / filename)
        except (OSError, urllib.error.URLError) as exc:
            if vocab_path.exists() and merges_path.exists():
                return cache_root
            raise ConnectionError(
                f"Could not download inditok model '{model_id}' from HuggingFace Hub"
            ) from exc
        return cache_root

    @staticmethod
    def _download_file(url: str, target: Path) -> None:
        with urllib.request.urlopen(url) as response, target.open("wb") as fh:
            total = int(response.headers.get("Content-Length") or 0)
            seen = 0
            while True:
                chunk = response.read(1024 * 64)
                if not chunk:
                    break
                fh.write(chunk)
                seen += len(chunk)
                if total:
                    pct = seen * 100 / total
                    print(f"\rDownloading {target.name}: {pct:5.1f}%", end="", file=sys.stderr)
            if total:
                print(file=sys.stderr)

    def save_pretrained(self, path: Union[str, Path]) -> None:
        target = Path(path)
        self._inner.save_pretrained(str(target))
        config = {
            "tokenizer_class": "IndicHFTokenizer",
            "model_max_length": 512,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
        }
        (target / "tokenizer_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def encode(self, text: str, lang: str | None = None, code_mix: bool = False) -> list[int]:
        return list(self._inner.encode(text, lang, code_mix))

    def encode_with_tokens(self, text: str, lang: str | None = None, code_mix: bool = False):
        return self._inner.encode_with_tokens(text, lang, code_mix)

    def decode(self, ids: Iterable[int]) -> str:
        return self._inner.decode([int(token_id) for token_id in ids])

    def encode_batch(
        self,
        texts: Iterable[str],
        lang: str | None = None,
        code_mix: bool = False,
        num_threads: int | None = None,
        show_progress: bool = False,
    ) -> list[list[int]]:
        text_list = list(texts)
        if show_progress:
            results: list[list[int]] = []
            total = len(text_list)
            for start in range(0, total, 1000):
                chunk = text_list[start : start + 1000]
                results.extend(
                    list(ids) for ids in self._inner.encode_batch(chunk, lang, code_mix, num_threads)
                )
                print(f"\r{min(start + 1000, total)}/{total}", end="", file=sys.stderr)
            print(file=sys.stderr)
            return results
        return [
            list(ids)
            for ids in self._inner.encode_batch(text_list, lang, code_mix, num_threads)
        ]

    def normalize(self, text: str, lang: str | None = None) -> str:
        return self._inner.normalize(text, lang)

    def pre_tokenize(
        self, text: str, lang: str | None = None, code_mix: bool = False
    ) -> list[str]:
        return list(self._inner.pre_tokenize(text, lang, code_mix))

    def fertility(self, texts: Iterable[str], lang: str | None = None) -> dict[str, object]:
        text_list = list(texts)
        total_words = sum(len(text.split()) for text in text_list)
        total_tokens = sum(
            sum(token != " " for token in self.encode_with_tokens(text, lang=lang).tokens)
            for text in text_list
        )
        return {
            "fertility": total_tokens / max(total_words, 1),
            "total_tokens": total_tokens,
            "total_words": total_words,
            "total_sentences": len(text_list),
        }

    @property
    def vocab_size(self) -> int:
        return int(self._inner.vocab_size())

    def get_vocab(self) -> dict[str, int]:
        if hasattr(self._inner, "get_vocab"):
            return {str(token): int(idx) for token, idx in self._inner.get_vocab().items()}
        return {}

    def __call__(self, text: Union[str, list[str]], **_: object) -> dict[str, object]:
        if isinstance(text, str):
            return {"input_ids": self.encode(text)}
        return {"input_ids": self.encode_batch(text)}
