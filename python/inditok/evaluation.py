from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from .tokenizer import IndicTokenizer


LANG_NAMES = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "kok": "Konkani",
    "ks": "Kashmiri",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}

BENCHMARK_FILE_ALIASES = {
    "bn": "ben",
    "hi": "hin",
    "ta": "tam",
    "te": "tel",
}
BENCHMARK_LANG_ALIASES = {alias: lang for lang, alias in BENCHMARK_FILE_ALIASES.items()}


@dataclass
class FertilityResult:
    lang: str
    lang_name: str
    tokenizer_name: str
    fertility: float
    vocab_coverage: float
    unk_rate: float
    chars_per_token: float
    total_words: int
    total_tokens: int


def evaluate_fertility(
    tokenizer: object,
    tokenizer_name: str,
    benchmark_dir: str | Path,
    langs: list[str] | None = None,
) -> list[FertilityResult]:
    encode = tokenizer if callable(tokenizer) else getattr(tokenizer, "encode")
    root = Path(benchmark_dir)
    selected = _selected_langs(root, langs)
    results = []
    for lang in selected:
        path = _resolve_benchmark_file(root, lang)
        if path is None:
            continue
        texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        total_words = sum(len(text.split()) for text in texts)
        total_chars = sum(len(text) for text in texts)
        total_tokens = 0
        unk_tokens = 0
        for text in texts:
            if hasattr(tokenizer, "encode_with_tokens"):
                out = tokenizer.encode_with_tokens(text, lang=lang)
                counted_tokens = [token for token in out.tokens if token != " "]
                total_tokens += len(counted_tokens)
                unk_tokens += sum(token == "<unk>" for token in out.tokens)
            else:
                ids = _call_encode(encode, text, lang)
                total_tokens += len(ids)
        fertility = total_tokens / max(total_words, 1)
        unk_rate = unk_tokens / max(total_tokens, 1)
        results.append(
            FertilityResult(
                lang=lang,
                lang_name=LANG_NAMES.get(lang, lang),
                tokenizer_name=tokenizer_name,
                fertility=fertility,
                vocab_coverage=1.0 - unk_rate,
                unk_rate=unk_rate,
                chars_per_token=total_chars / max(total_tokens, 1),
                total_words=total_words,
                total_tokens=total_tokens,
            )
        )
    return results


def _selected_langs(root: Path, langs: list[str] | None) -> list[str]:
    if langs:
        return [_canonical_lang(lang) for lang in langs]

    selected: list[str] = []
    seen: set[str] = set()
    for path in sorted(root.glob("*.txt")):
        lang = _canonical_lang(path.stem)
        if lang in seen:
            continue
        seen.add(lang)
        selected.append(lang)
    return selected


def _canonical_lang(lang: str) -> str:
    return BENCHMARK_LANG_ALIASES.get(lang, lang)


def _resolve_benchmark_file(root: Path, lang: str) -> Path | None:
    lang = _canonical_lang(lang)
    candidates = [root / f"{lang}.txt"]
    alias = BENCHMARK_FILE_ALIASES.get(lang)
    if alias:
        candidates.append(root / f"{alias}.txt")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def compare_tokenizers(
    tokenizers: dict[str, Callable[[str], Iterable[int]] | object],
    benchmark_dir: str | Path,
    output_format: str = "table",
) -> str:
    results: list[FertilityResult] = []
    for name, tokenizer in tokenizers.items():
        results.extend(evaluate_fertility(tokenizer, name, benchmark_dir))

    if output_format == "json":
        return json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2)
    if output_format == "markdown":
        rows = [
            "| tokenizer | lang | fertility | unk_rate | chars/token |",
            "|---|---:|---:|---:|---:|",
        ]
        rows.extend(_format_row(result, markdown=True) for result in results)
        return "\n".join(rows)
    if output_format != "table":
        raise ValueError("output_format must be one of table, json, or markdown")
    rows = ["tokenizer          lang  fertility  unk_rate  chars/token"]
    rows.extend(_format_row(result, markdown=False) for result in results)
    return "\n".join(rows)


def _call_encode(encode: Callable[..., Iterable[int]], text: str, lang: str) -> list[int]:
    try:
        return list(encode(text, lang=lang))
    except TypeError:
        return list(encode(text))


def _format_row(result: FertilityResult, markdown: bool) -> str:
    if markdown:
        return (
            f"| {result.tokenizer_name} | {result.lang} | {result.fertility:.2f} | "
            f"{result.unk_rate:.2f} | {result.chars_per_token:.2f} |"
        )
    return (
        f"{result.tokenizer_name:<18} {result.lang:<5} {result.fertility:<10.2f} "
        f"{result.unk_rate:<8.2f} {result.chars_per_token:.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m inditok.evaluation")
    parser.add_argument("--model", default=None)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks/data"))
    parser.add_argument("--langs", default=None)
    parser.add_argument("--output", choices=["table", "json", "markdown"], default="table")
    args = parser.parse_args(argv)

    tokenizer = IndicTokenizer.from_pretrained(args.model) if args.model else IndicTokenizer()
    langs = None if not args.langs or args.langs == "all" else args.langs.split(",")
    print(compare_tokenizers({"inditok": tokenizer}, args.benchmark_dir, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
