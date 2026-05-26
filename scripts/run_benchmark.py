from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from inditok import IndicTokenizer  # noqa: E402
from inditok.evaluation import evaluate_fertility  # noqa: E402


def run_benchmark(
    model: str | Path | None,
    benchmark_dir: str | Path,
    langs: list[str] | None,
    include_optional: bool = False,
) -> list[dict[str, object]]:
    tokenizers: dict[str, object] = {
        "inditok": IndicTokenizer.from_pretrained(model) if model else IndicTokenizer()
    }
    if include_optional:
        tokenizers.update(_optional_tokenizers())

    rows: list[dict[str, object]] = []
    for name, tokenizer in tokenizers.items():
        rows.extend(asdict(row) for row in evaluate_fertility(tokenizer, name, benchmark_dir, langs))
    return rows


def _optional_tokenizers() -> dict[str, object]:
    tokenizers: dict[str, object] = {}
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        tokenizers["tiktoken/cl100k"] = enc.encode
    except Exception as exc:  # pragma: no cover - optional dependency path
        print(f"Skipping tiktoken: {exc}", file=sys.stderr)

    try:
        from transformers import AutoTokenizer

        tokenizers["mbert"] = AutoTokenizer.from_pretrained("bert-base-multilingual-cased").encode
    except Exception as exc:  # pragma: no cover - optional dependency/network path
        print(f"Skipping mBERT: {exc}", file=sys.stderr)
    return tokenizers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run inditok fertility benchmark")
    parser.add_argument("--model", default=None)
    parser.add_argument("--benchmark-dir", default="benchmarks/data")
    parser.add_argument("--langs", nargs="*", default=None)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--output", default="benchmarks/results.json")
    args = parser.parse_args(argv)

    rows = run_benchmark(args.model, args.benchmark_dir, args.langs, args.include_optional)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
