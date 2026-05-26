from __future__ import annotations

import argparse
import json
from pathlib import Path

SPECIAL_ORDER = ["<pad>", "<s>", "</s>", "<mask>", "<unk>"]
BYTE_TOKENS = [f"<0x{idx:02X}>" for idx in range(256)]


def merge_models(model_dirs: list[str | Path], output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tokens: list[str] = []
    for token in SPECIAL_ORDER + BYTE_TOKENS:
        _append_unique(tokens, token)

    merges: list[str] = []
    seen_merges: set[str] = set()
    for model_dir in model_dirs:
        root = Path(model_dir)
        vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
        for token, _idx in sorted(vocab.items(), key=lambda item: int(item[1])):
            _append_unique(tokens, token)
        merges_path = root / "merges.txt"
        if merges_path.exists():
            for line in merges_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and line not in seen_merges:
                    seen_merges.add(line)
                    merges.append(line)

    vocab = {token: idx for idx, token in enumerate(tokens)}
    vocab_path = output / "vocab.json"
    merges_path = output / "merges.txt"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    merges_path.write_text("\n".join(merges), encoding="utf-8")
    return vocab_path, merges_path


def _append_unique(tokens: list[str], token: str) -> None:
    if token not in tokens:
        tokens.append(token)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge inditok vocab/merges directories")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    merge_models(args.models, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
