from __future__ import annotations

import argparse
import statistics
import time

from inditok import IndicTokenizer


TEXTS = [
    "नमस्ते भारत! हिंदी भाषा के लिए एक छोटा परीक्षण।",
    "నమస్కారం తెలుగు! ఇది ఒక చిన్న పరీక్ష.",
    "hello हिंदी and తెలుగు mixed with English.",
] * 1000


def measure(name: str, encode_fn) -> dict[str, float | str]:
    latencies = []
    token_count = 0
    char_count = 0
    start = time.perf_counter()
    for text in TEXTS:
        item_start = time.perf_counter()
        ids = encode_fn(text)
        latencies.append(time.perf_counter() - item_start)
        token_count += len(ids)
        char_count += len(text)
    elapsed = time.perf_counter() - start
    return {
        "name": name,
        "texts_per_second": len(TEXTS) / elapsed,
        "chars_per_second": char_count / elapsed,
        "p50_latency_ms": statistics.median(latencies) * 1000,
        "token_fertility": token_count / char_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()

    results = [measure("inditok", IndicTokenizer().encode)]

    if args.include_optional:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            results.append(measure("tiktoken/cl100k_base", enc.encode))
        except Exception as exc:  # pragma: no cover - optional dependency path
            print(f"Skipping tiktoken: {exc}")

        try:
            from transformers import AutoTokenizer

            hf = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            results.append(measure("hf/bert-base-multilingual-cased", hf.encode))
        except Exception as exc:  # pragma: no cover - optional dependency/network path
            print(f"Skipping transformers: {exc}")

    for row in results:
        print(
            "{name}: {texts_per_second:.0f} texts/s, {chars_per_second:.0f} chars/s, "
            "p50={p50_latency_ms:.3f} ms, fertility={token_fertility:.3f}".format(**row)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

