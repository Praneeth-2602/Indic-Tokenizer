from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ._fallback import detect_script_spans
from .evaluation import compare_tokenizers
from .tokenizer import IndicTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inditok", description="Indic tokenizer CLI")
    parser.add_argument("--model", help="Local model directory or HuggingFace Hub model id")

    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="Encode text into token ids")
    enc.add_argument("text")
    enc.add_argument("--tokens", action="store_true", help="Include token strings")
    enc.add_argument("--lang")
    enc.add_argument("--code-mix", action="store_true")

    dec = sub.add_parser("decode", help="Decode token ids into text")
    dec.add_argument("ids", nargs="+", type=int)

    norm = sub.add_parser("normalize", help="Normalize text")
    norm.add_argument("text")
    norm.add_argument("--lang")

    train = sub.add_parser("train", help="Train an inditok model")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--vocab-size", type=int, default=64000)
    train.add_argument("--langs", default=None)
    train.add_argument("--model-type", default="bpe")
    train.add_argument("--allocation", choices=["equal", "proportional"], default="proportional")
    train.add_argument("--morpheme-hints", choices=["auto", "none"], default="auto")
    train.add_argument("--clean-corpus", dest="clean_corpus", action="store_true", default=True)
    train.add_argument("--no-clean-corpus", dest="clean_corpus", action="store_false")
    train.add_argument("--min-line-chars", type=int, default=20)

    fert = sub.add_parser("fertility", help="Measure fertility on a text file")
    fert.add_argument("--input", required=True, type=Path)
    fert.add_argument("--lang")
    fert.add_argument("--compare", default=None)

    pre = sub.add_parser("pre-tokenize", help="Show pretokenization output")
    pre.add_argument("text", nargs="?")
    pre.add_argument("--lang")
    pre.add_argument("--code-mix", action="store_true")

    detect = sub.add_parser("detect-script", help="Show script spans")
    detect.add_argument("text", nargs="?")

    bench = sub.add_parser("benchmark", help="Run fertility benchmark")
    bench.add_argument("--langs", default="all")
    bench.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks/data"))
    bench.add_argument("--output-format", choices=["table", "json", "markdown"], default="table")

    return parser


def load_tokenizer(model: str | None) -> IndicTokenizer:
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
            encoded = tokenizer.encode_with_tokens(args.text, lang=args.lang, code_mix=args.code_mix)
            print(
                json.dumps(
                    {"ids": encoded.ids, "tokens": encoded.tokens, "offsets": encoded.offsets},
                    ensure_ascii=False,
                )
            )
        else:
            print(json.dumps(tokenizer.encode(args.text, lang=args.lang, code_mix=args.code_mix), ensure_ascii=False))
        return 0

    if args.command == "decode":
        print(tokenizer.decode(args.ids))
        return 0

    if args.command == "normalize":
        print(tokenizer.normalize(args.text, lang=args.lang))
        return 0

    if args.command == "train":
        script = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
        cmd = [
            sys.executable,
            str(script),
            "--data-dir",
            args.data_dir,
            "--output-dir",
            args.output_dir,
            "--vocab-size",
            str(args.vocab_size),
            "--model-type",
            args.model_type,
            "--allocation",
            args.allocation,
            "--morpheme-hints",
            args.morpheme_hints,
            "--min-line-chars",
            str(args.min_line_chars),
        ]
        if not args.clean_corpus:
            cmd.append("--no-clean-corpus")
        if args.langs:
            cmd.extend(["--langs", args.langs])
        return subprocess.call(cmd)

    if args.command == "fertility":
        lines = [
            line.strip()
            for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report = tokenizer.fertility(lines, lang=args.lang)
        total_chars = sum(len(line) for line in lines)
        chars_per_token = total_chars / max(int(report["total_tokens"]), 1)
        print("tokenizer          lang  fertility  unk_rate  chars/token")
        print(
            f"inditok            {args.lang or '-':<5} {report['fertility']:<10.2f} "
            f"{0.0:<8.2f} {chars_per_token:.2f}"
        )
        return 0

    if args.command == "pre-tokenize":
        text = args.text if args.text is not None else sys.stdin.read()
        print(json.dumps(tokenizer.pre_tokenize(text, lang=args.lang, code_mix=args.code_mix), ensure_ascii=False))
        return 0

    if args.command == "detect-script":
        text = args.text if args.text is not None else sys.stdin.read()
        for span in detect_script_spans(text):
            print(f"[{span['start']}:{span['end']}]\t{span['script']}\t{span['text']!r}")
        return 0

    if args.command == "benchmark":
        langs = None if args.langs == "all" else args.langs.split(",")
        output = compare_tokenizers({"inditok": tokenizer}, args.benchmark_dir, args.output_format)
        if langs is not None and args.output_format != "json":
            selected = set(langs)
            output = "\n".join(line for line in output.splitlines() if not line or line.split()[1] in selected or line.startswith(("tokenizer", "|")))
        print(output)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
