from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from inditok import IndicTokenizer  # noqa: E402
from train import audit_corpus  # noqa: E402


def audit_data_dir(data_dir: str | Path, langs: list[str] | None = None) -> dict[str, object]:
    root = Path(data_dir)
    tokenizer = IndicTokenizer()
    selected = langs or sorted(path.stem for path in root.glob("*.txt"))
    lines: list[tuple[str, str]] = []
    missing = []
    for lang in selected:
        path = root / f"{lang}.txt"
        if not path.exists():
            missing.append(lang)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            normalized = tokenizer.normalize(line, lang=lang)
            if normalized:
                lines.append((lang, normalized))
    return {"missing_langs": missing, "corpus_audit": audit_corpus(lines)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit inditok training corpora")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--langs", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    report = audit_data_dir(args.data_dir, args.langs.split(",") if args.langs else None)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
