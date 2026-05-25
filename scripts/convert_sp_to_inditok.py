from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path


SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<mask>", "<unk>"]
BYTE_TOKENS = [f"<0x{idx:02X}>" for idx in range(256)]
MORPHEME_BOUNDARY = "\u2063"


def convert_sentencepiece_model(model_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install inditok[train] to convert SentencePiece models") from exc

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab: dict[str, int] = {}
    learned_pieces: list[str] = []
    for token in SPECIAL_TOKENS + BYTE_TOKENS:
        vocab.setdefault(token, len(vocab))
    for idx in range(processor.get_piece_size()):
        for piece in _inditok_piece_variants(processor.id_to_piece(idx)):
            vocab.setdefault(piece, len(vocab))
            learned_pieces.append(piece)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vocab_path = output / "vocab.json"
    merges_path = output / "merges.txt"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    merges_path.write_text(_synthesize_merges(learned_pieces), encoding="utf-8")
    return vocab_path, merges_path


def _inditok_piece_variants(piece: str) -> list[str]:
    if piece.startswith("<0x") and piece.endswith(">"):
        return [piece]
    if piece in SPECIAL_TOKENS:
        return [piece]

    normalized = piece.replace("▁", " ").replace(MORPHEME_BOUNDARY, "")
    variants = [normalized]
    stripped = normalized.strip()
    if stripped and stripped != normalized:
        variants.append(stripped)
    if piece.startswith("▁") and piece[1:]:
        variants.append(piece[1:])

    seen = set()
    return [item for item in variants if item and not (item in seen or seen.add(item))]


def _synthesize_merges(pieces: list[str]) -> str:
    merges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for piece in sorted(set(pieces), key=lambda item: (len(_clusters(item)), item)):
        if _skip_merge_piece(piece):
            continue
        symbols = _clusters(piece)
        if len(symbols) < 2:
            continue
        current = symbols[0]
        for symbol in symbols[1:]:
            pair = (current, symbol)
            if pair not in seen:
                seen.add(pair)
                merges.append(pair)
            current = f"{current}{symbol}"
    return "\n".join(f"{left} {right}" for left, right in merges)


def _skip_merge_piece(piece: str) -> bool:
    return (
        piece in SPECIAL_TOKENS
        or piece in BYTE_TOKENS
        or piece == " "
        or " " in piece
        or piece == MORPHEME_BOUNDARY
    )


def _clusters(text: str) -> list[str]:
    clusters: list[str] = []
    for ch in text:
        if clusters and (unicodedata.combining(ch) or unicodedata.category(ch).startswith("M")):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert SentencePiece to inditok files")
    parser.add_argument("model")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    convert_sentencepiece_model(args.model, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
