from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .tokenizer import IndicTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inditok", description="Indic tokenizer CLI")
    parser.add_argument("--model", type=Path, help="Directory containing vocab.json and merges.txt")

    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="Encode text into token ids")
    enc.add_argument("text")
    enc.add_argument("--tokens", action="store_true", help="Include token strings")

    dec = sub.add_parser("decode", help="Decode token ids into text")
    dec.add_argument("ids", nargs="+", type=int)

    norm = sub.add_parser("normalize", help="Normalize text")
    norm.add_argument("text")

    return parser


def load_tokenizer(model: Path | None) -> IndicTokenizer:
    if model is None:
        return IndicTokenizer()
    return IndicTokenizer.from_pretrained(model)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)
    tokenizer = load_tokenizer(args.model)

    if args.command == "encode":
        if args.tokens:
            encoded = tokenizer.encode_with_tokens(args.text)
            print(json.dumps({"ids": encoded.ids, "tokens": encoded.tokens}, ensure_ascii=False))
        else:
            print(json.dumps(tokenizer.encode(args.text), ensure_ascii=False))
        return 0

    if args.command == "decode":
        print(tokenizer.decode(args.ids))
        return 0

    if args.command == "normalize":
        print(tokenizer.normalize(args.text))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
