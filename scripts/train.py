from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from inditok.tokenizer import IndicTokenizer  # noqa: E402

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<mask>", "<unk>"]
BYTE_TOKENS = [f"<0x{idx:02X}>" for idx in range(256)]
MORPHEME_BOUNDARY = "\u2063"
TAMIL_SUFFIXES = [
    "களிலிருந்து",
    "கிறீர்கள்",
    "கிறார்கள்",
    "களுக்கு",
    "க்கிறேன்",
    "க்கிறாய்",
    "க்கிறார்",
    "கிறேன்",
    "கிறாய்",
    "கிறார்",
    "கிறோம்",
    "ந்தார்கள்",
    "ந்தேன்",
    "ந்தாய்",
    "ந்தார்",
    "ந்தோம்",
    "வார்கள்",
    "வேன்",
    "வாய்",
    "வார்",
    "வோம்",
    "களை",
    "களோடு",
    "களில்",
    "உக்கு",
    "இலிருந்து",
    "இல்",
    "ஓடு",
    "ஆல்",
    "மட்டும்",
    "தான்",
    "கூட",
    "என்று",
    "ஆக",
]
TELUGU_SUFFIXES = [
    "తున్నాను",
    "తున్నావు",
    "తున్నారు",
    "తున్నాం",
    "నున్నాను",
    "నున్నావు",
    "న్నారు",
    "యాను",
    "యావు",
    "యారు",
    "యాం",
    "తాను",
    "తావు",
    "తారు",
    "తాం",
    "ాను",
    "ావు",
    "ారు",
    "నుండి",
    "గురించి",
    "కోసం",
    "వరకు",
    "తర్వాత",
    "వల్ల",
    "లను",
    "లకు",
    "లతో",
    "లలో",
    "కు",
    "లో",
    "తో",
]
HINDI_SUFFIXES = [
    "के बारे में",
    "के लिए",
    "के साथ",
    "में से",
    "ता है",
    "ती है",
    "ते हैं",
    "ता था",
    "ती थी",
    "एगा",
    "एगी",
    "एंगे",
    "ओगे",
    "ओगी",
    "ों को",
    "ों में",
    "ों से",
    "ों का",
    "ों की",
]
BENGALI_SUFFIXES = [
    "গুলো",
    "গুলোর",
    "গুলোকে",
    "দের",
    "ভাবে",
    "টার",
    "টি",
    "ের",
    "কে",
]
SUFFIXES = {
    "hi": HINDI_SUFFIXES,
    "ta": TAMIL_SUFFIXES,
    "te": TELUGU_SUFFIXES,
    "bn": BENGALI_SUFFIXES,
}


def train(
    data_dir: str | Path,
    output_dir: str | Path,
    vocab_size: int = 64000,
    langs: list[str] | None = None,
    min_frequency: int = 2,
    character_coverage: float = 0.9999,
    allocation: str = "proportional",
    model_type: str = "bpe",
    morpheme_hints: str = "auto",
    clean_corpus: bool = True,
    min_line_chars: int = 20,
) -> dict[str, object]:
    data_root = Path(data_dir)
    output_root = Path(output_dir)
    langs = langs or sorted(path.stem for path in data_root.glob("*.txt"))
    files = [(lang, data_root / f"{lang}.txt") for lang in langs if (data_root / f"{lang}.txt").exists()]
    missing = sorted(set(langs) - {lang for lang, _ in files})
    for lang in missing:
        print(f"warning: skipping missing language file {lang}.txt", file=sys.stderr)
    if not files:
        raise FileNotFoundError("no language files found for training")

    normalizer = IndicTokenizer()
    normalized_lines: list[tuple[str, str]] = []
    training_lines: list[tuple[str, str]] = []
    byte_sizes: dict[str, int] = {}
    dropped_lines: dict[str, int] = {}
    for lang, path in files:
        size = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            cleaned = clean_line(line, lang, min_chars=min_line_chars) if clean_corpus else line
            if cleaned is None:
                dropped_lines[lang] = dropped_lines.get(lang, 0) + 1
                continue
            normalized = normalizer.normalize(cleaned, lang=lang)
            if normalized:
                normalized_lines.append((lang, normalized))
                training_text = (
                    inject_morpheme_hints(normalized, lang)
                    if morpheme_hints == "auto"
                    else normalized
                )
                training_lines.append((lang, training_text))
                size += len(normalized.encode("utf-8"))
        byte_sizes[lang] = size
    if not normalized_lines:
        raise ValueError(
            "corpus cleaning removed every line; lower --min-line-chars or pass --no-clean-corpus"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    sp_trained = _try_sentencepiece_train(
        training_lines,
        output_root,
        vocab_size,
        character_coverage,
        model_type,
    )
    if not sp_trained:
        _write_simple_vocab(training_lines, output_root, vocab_size, min_frequency)

    tokenizer = IndicTokenizer.from_pretrained(output_root)
    stats = _fertility_stats(tokenizer, normalized_lines)
    metadata = {
        "langs": [lang for lang, _ in files],
        "missing_langs": missing,
        "allocation": _allocate_vocab(byte_sizes, vocab_size, allocation),
        "corpus_audit": audit_corpus(normalized_lines),
        "cleaning": {
            "enabled": clean_corpus,
            "min_line_chars": min_line_chars,
            "dropped_lines": dropped_lines,
        },
        "morpheme_hints": morpheme_hints,
        "sentencepiece": sp_trained,
        "fertility": stats,
    }
    (output_root / "training_stats.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _validate_round_trip(tokenizer, normalized_lines)
    return metadata


def _try_sentencepiece_train(
    normalized_lines: list[tuple[str, str]],
    output_root: Path,
    vocab_size: int,
    character_coverage: float,
    model_type: str,
) -> bool:
    try:
        import sentencepiece as spm
    except ImportError:
        return False

    from convert_sp_to_inditok import convert_sentencepiece_model

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = Path(tmpdir) / "corpus.txt"
        corpus.write_text("\n".join(text for _, text in normalized_lines), encoding="utf-8")
        prefix = str(Path(tmpdir) / "inditok")
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=prefix,
            model_type=model_type,
            vocab_size=vocab_size,
            max_sentencepiece_length=32,
            character_coverage=character_coverage,
            byte_fallback=True,
            split_digits=True,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=4,
            pad_piece="<pad>",
            bos_piece="<s>",
            eos_piece="</s>",
            unk_piece="<unk>",
            user_defined_symbols=["<mask>", MORPHEME_BOUNDARY],
            split_by_whitespace=True,
            treat_whitespace_as_suffix=False,
            allow_whitespace_only_pieces=False,
            remove_extra_whitespaces=True,
            normalization_rule_name="identity",
            hard_vocab_limit=False,
        )
        convert_sentencepiece_model(f"{prefix}.model", output_root)
    return True


def _write_simple_vocab(
    normalized_lines: list[tuple[str, str]],
    output_root: Path,
    vocab_size: int,
    min_frequency: int,
) -> None:
    counts: Counter[str] = Counter()
    tokenizer = IndicTokenizer()
    for lang, text in normalized_lines:
        text = text.replace(MORPHEME_BOUNDARY, "")
        counts.update(tokenizer.pre_tokenize(text, lang=lang))
        counts.update(ch for ch in text)
    tokens = SPECIAL_TOKENS + BYTE_TOKENS
    for token, count in counts.most_common(max(vocab_size - len(tokens), 0)):
        if count >= min_frequency and token not in tokens:
            tokens.append(token)
    vocab = {token: idx for idx, token in enumerate(tokens[:vocab_size])}
    (output_root / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "merges.txt").write_text("", encoding="utf-8")


def _allocate_vocab(byte_sizes: dict[str, int], vocab_size: int, allocation: str) -> dict[str, int]:
    usable_vocab_size = max(vocab_size - len(SPECIAL_TOKENS) - len(BYTE_TOKENS), 0)
    if allocation == "equal":
        langs = list(byte_sizes)
        base = usable_vocab_size // max(len(langs), 1)
        remainder = usable_vocab_size - (base * len(langs))
        return {
            lang: base + (1 if idx < remainder else 0)
            for idx, lang in enumerate(langs)
        }
    total = sum(byte_sizes.values()) or 1
    raw = {lang: usable_vocab_size * size / total for lang, size in byte_sizes.items()}
    allocation_map = {lang: int(value) for lang, value in raw.items()}
    remainder = usable_vocab_size - sum(allocation_map.values())
    order = sorted(raw, key=lambda lang: raw[lang] - allocation_map[lang], reverse=True)
    for lang in order[:remainder]:
        allocation_map[lang] += 1
    return allocation_map


def inject_morpheme_hints(text: str, lang: str) -> str:
    suffixes = SUFFIXES.get(lang, [])
    if not suffixes:
        return text
    for suffix in sorted((item for item in suffixes if " " in item), key=len, reverse=True):
        text = text.replace(suffix, f"{MORPHEME_BOUNDARY}{suffix}")
    words = []
    word_suffixes = [item for item in suffixes if " " not in item]
    for word in text.split(" "):
        words.append(_hint_word(word, word_suffixes))
    return " ".join(words)


def clean_line(line: str, lang: str, min_chars: int = 20) -> str | None:
    del lang
    line = line.strip()
    if not line:
        return None
    if len(line) < min_chars:
        return None
    ascii_ratio = sum(ch.isascii() for ch in line) / max(len(line), 1)
    if ascii_ratio > 0.30:
        return None
    urls = re.findall(r"https?://\S+", line)
    if urls and len(urls) > 1:
        return None
    if re.search(r"\d{6,}", line):
        return None
    if re.search(r"<[^>]+>", line):
        return None
    return line


def _hint_word(word: str, suffixes: list[str]) -> str:
    if MORPHEME_BOUNDARY in word:
        return word
    for suffix in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suffix) and len(word) > len(suffix):
            return f"{word[:-len(suffix)]}{MORPHEME_BOUNDARY}{suffix}"
    return word


def audit_corpus(normalized_lines: list[tuple[str, str]]) -> dict[str, dict[str, float | int]]:
    audit: dict[str, dict[str, float | int]] = {}
    by_lang: dict[str, list[str]] = {}
    for lang, text in normalized_lines:
        by_lang.setdefault(lang, []).append(text)

    for lang, lines in by_lang.items():
        total = len(lines)
        duplicates = total - len(set(lines))
        non_native = 0
        htmlish = 0
        total_chars = 0
        for line in lines:
            if "<" in line and ">" in line:
                htmlish += 1
            chars = [ch for ch in line if not ch.isspace()]
            total_chars += len(chars)
            native = sum(1 for ch in chars if _is_expected_script(ch, lang) or ch.isdigit() or ch.isascii())
            if chars and native / len(chars) < 0.75:
                non_native += 1
        audit[lang] = {
            "lines": total,
            "duplicate_lines": duplicates,
            "duplicate_rate": duplicates / max(total, 1),
            "non_native_script_lines": non_native,
            "non_native_script_rate": non_native / max(total, 1),
            "htmlish_lines": htmlish,
            "avg_chars_per_line": total_chars / max(total, 1),
        }
    return audit


def _is_expected_script(ch: str, lang: str) -> bool:
    cp = ord(ch)
    ranges = {
        "hi": (0x0900, 0x097F),
        "mr": (0x0900, 0x097F),
        "bn": (0x0980, 0x09FF),
        "as": (0x0980, 0x09FF),
        "ta": (0x0B80, 0x0BFF),
        "te": (0x0C00, 0x0C7F),
        "kn": (0x0C80, 0x0CFF),
        "ml": (0x0D00, 0x0D7F),
        "gu": (0x0A80, 0x0AFF),
        "pa": (0x0A00, 0x0A7F),
        "or": (0x0B00, 0x0B7F),
        "ur": (0x0600, 0x06FF),
    }
    start_end = ranges.get(lang)
    if not start_end:
        return True
    return start_end[0] <= cp <= start_end[1]


def _validate_round_trip(tokenizer: IndicTokenizer, normalized_lines: list[tuple[str, str]]) -> None:
    seen: dict[str, int] = {}
    for lang, text in normalized_lines:
        if seen.get(lang, 0) >= 10:
            continue
        decoded = tokenizer.decode(tokenizer.encode(text, lang=lang))
        if decoded != text:
            raise AssertionError(f"round-trip failed for {lang}: {text!r} -> {decoded!r}")
        seen[lang] = seen.get(lang, 0) + 1


def _fertility_stats(
    tokenizer: IndicTokenizer, normalized_lines: list[tuple[str, str]]
) -> dict[str, float]:
    by_lang: dict[str, list[str]] = {}
    for lang, text in normalized_lines:
        by_lang.setdefault(lang, []).append(text)
    return {lang: tokenizer.fertility(lines, lang=lang)["fertility"] for lang, lines in by_lang.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an inditok vocabulary")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=64000)
    parser.add_argument("--langs", default=None)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--character-coverage", type=float, default=0.9999)
    parser.add_argument("--allocation", choices=["equal", "proportional"], default="proportional")
    parser.add_argument("--model-type", choices=["bpe", "unigram"], default="bpe")
    parser.add_argument("--morpheme-hints", choices=["auto", "none"], default="auto")
    parser.add_argument("--clean-corpus", dest="clean_corpus", action="store_true", default=True)
    parser.add_argument("--no-clean-corpus", dest="clean_corpus", action="store_false")
    parser.add_argument("--min-line-chars", type=int, default=20)
    args = parser.parse_args(argv)
    stats = train(
        args.data_dir,
        args.output_dir,
        args.vocab_size,
        args.langs.split(",") if args.langs else None,
        args.min_frequency,
        args.character_coverage,
        args.allocation,
        args.model_type,
        args.morpheme_hints,
        args.clean_corpus,
        args.min_line_chars,
    )
    print(json.dumps(stats["fertility"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
