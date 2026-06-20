from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path


SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<mask>", "<unk>"]
BYTE_TOKENS = [f"<0x{idx:02X}>" for idx in range(256)]
MORPHEME_BOUNDARY = "\u2063"


def convert_sentencepiece_model(
    model_path: str | Path,
    output_dir: str | Path,
    expected_vocab_size: int | None = None,
) -> tuple[Path, Path]:
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install inditok[train] to convert SentencePiece models") from exc

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab: dict[str, int] = {}
    learned_pieces: list[str] = []
    for token in SPECIAL_TOKENS + BYTE_TOKENS:
        vocab.setdefault(token, len(vocab))
    target_vocab_size = expected_vocab_size or processor.get_piece_size()
    duplicate_pieces: set[str] = set()
    empty_pieces = 0
    seen_pieces: set[str] = set()
    for idx in range(processor.get_piece_size()):
        variants = _inditok_piece_variants(processor.id_to_piece(idx))
        for piece in variants:
            if piece in seen_pieces:
                duplicate_pieces.add(piece)
            seen_pieces.add(piece)
            vocab.setdefault(piece, len(vocab))
            if piece not in vocab:
                vocab[piece] = len(vocab)
                learned_pieces.append(piece)
        if not variants:
            empty_pieces += 1

    _assert_vocab_size(
        actual=len(vocab),
        expected=target_vocab_size,
        duplicate_count=len(duplicate_pieces),
        empty_count=empty_pieces,
    )

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

    clean = piece.replace("▁", "Ġ")
    clean = clean.replace(MORPHEME_BOUNDARY, "")
    clean = unicodedata.normalize("NFC", clean)

    return [clean] if clean else []


def _assert_vocab_size(
    actual: int,
    expected: int,
    duplicate_count: int = 0,
    empty_count: int = 0,
) -> None:
    if actual >= expected * 0.99:
        return
    raise RuntimeError(
        f"Converted vocab has {actual} entries, expected ~{expected}. "
        "SentencePiece may have emitted duplicate or empty pieces. "
        f"Observed {duplicate_count} duplicate converted pieces and {empty_count} empty pieces. "
        "Inspect the .model file with `spm_export_vocab`."
    )


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
    try:
        import regex as _regex  # type: ignore
    except ImportError:
        # Fallback: approximate grapheme clustering using combining marks
        clusters: list[str] = []
        for ch in text:
            if clusters and (unicodedata.combining(ch) or unicodedata.category(ch).startswith("M")):
                clusters[-1] += ch
            else:
                clusters.append(ch)
        return clusters

    # Use the `regex` package's \X to get extended grapheme clusters.
    # This is important for correct segmentation in Telugu/Tamil and other
    # Indic scripts that rely on complex grapheme rules.
    return _regex.findall(r"\X", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert SentencePiece to inditok files")
    parser.add_argument("model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-vocab-size", type=int, default=None)
    args = parser.parse_args(argv)
    convert_sentencepiece_model(args.model, args.output_dir, args.expected_vocab_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
