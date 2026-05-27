from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FLORES_LANGS = {
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "brx": "brx_Deva",
    "doi": "doi_Deva",
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "kok": "kok_Deva",
    "ks": "kas_Arab",
    "mai": "mai_Deva",
    "ml": "mal_Mlym",
    "mni": "mni_Beng",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "sa": "san_Deva",
    "sat": "sat_Olck",
    "sd": "snd_Arab",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
}

FLORES101_FILE_ALIASES = {
    "bn": "ben",
    "hi": "hin",
    "ta": "tam",
    "te": "tel",
}


def prepare_flores_benchmark(
    flores_root: str | Path,
    output_dir: str | Path = "benchmarks/data",
    langs: list[str] | None = None,
) -> list[Path]:
    source_root = Path(flores_root)
    target_root = Path(output_dir)
    selected = langs or sorted(FLORES_LANGS)
    written: list[Path] = []

    for lang in selected:
        flores_code = FLORES_LANGS[lang]
        source = _find_devtest_file(source_root, flores_code)
        target = target_root / f"{lang}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        written.append(target)

    return written


def _find_devtest_file(root: Path, flores_code: str) -> Path:
    candidates = [
        root / "devtest" / f"{flores_code}.devtest",
        root / "flores200_dataset" / "devtest" / f"{flores_code}.devtest",
        root / f"{flores_code}.devtest",
        root / f"{flores_code}_devtest.txt",
    ]
    flores101_code = next(
        (alias for lang, alias in FLORES101_FILE_ALIASES.items() if FLORES_LANGS[lang] == flores_code),
        None,
    )
    if flores101_code:
        candidates.extend(
            [
                root / f"{flores101_code}.txt",
                root / f"{flores101_code}.dev",
                root / f"{flores101_code}.devtest",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(root.rglob(f"{flores_code}.devtest"))
    matches.extend(sorted(root.rglob(f"{flores_code}_devtest.txt")))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find FLORES devtest file for {flores_code} under {root}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy downloaded FLORES-200 devtest files into benchmarks/data")
    parser.add_argument("flores_root", help="Directory containing FLORES-200 devtest files")
    parser.add_argument("--output-dir", default="benchmarks/data")
    parser.add_argument("--langs", nargs="*", choices=sorted(FLORES_LANGS), default=None)
    args = parser.parse_args(argv)

    written = prepare_flores_benchmark(args.flores_root, args.output_dir, args.langs)
    for path in written:
        lines = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"Wrote {path} ({lines} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
