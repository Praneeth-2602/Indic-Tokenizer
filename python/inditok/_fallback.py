from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SPACE_TOKEN = " "


@dataclass
class EncodeOutput:
    ids: list[int]
    tokens: list[str]
    lang: str | None = None
    offsets: list[tuple[int, int]] | None = None


class IndicTokenizer:
    def __init__(self, vocab_path: Optional[str] = None, merges_path: Optional[str] = None):
        if vocab_path is None and merges_path is not None:
            raise ValueError("merges_path requires vocab_path")
        if vocab_path is None:
            self.vocab = _default_vocab()
            self.merges = []
        else:
            self.vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
            self.merges = _load_merges(Path(merges_path)) if merges_path else []
        if UNK_TOKEN not in self.vocab:
            raise ValueError("vocab.json must contain <unk>")
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.unk_id = int(self.vocab[UNK_TOKEN])

    @classmethod
    def from_pretrained(cls, path: str) -> "IndicTokenizer":
        root = Path(path)
        return cls(str(root / "vocab.json"), str(root / "merges.txt"))

    def save_pretrained(self, path: str) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "vocab.json").write_text(
            json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (root / "merges.txt").write_text(
            "\n".join(f"{left} {right}" for left, right in self.merges), encoding="utf-8"
        )

    def normalize(self, text: str, lang: str | None = None) -> str:
        family = _lang_to_family(lang) or _detect_dominant_family(text)
        if family == "devanagari":
            return _normalize_devanagari(text)
        if family in {"tamil", "telugu", "kannada", "malayalam"}:
            return _normalize_dravidian(text, family)
        if family in {"bengali", "odia"}:
            return _collapse_repeated(_collapse_repeated(_base_normalize(text), "\u09cd"), "\u0982")
        if family == "gujarati":
            return _collapse_repeated(_collapse_repeated(_base_normalize(text), "\u0acd"), "\u0a82")
        if family == "persian_arabic":
            return _normalize_perso_arabic(text)
        return _base_normalize(text)

    def pre_tokenize(
        self, text: str, lang: str | None = None, code_mix: bool = False
    ) -> list[str]:
        if code_mix:
            pieces: list[str] = []
            for span in detect_script_spans(text):
                pieces.extend(self.pre_tokenize(span["text"], _script_to_lang(span["script"], lang), False))
            return pieces
        text = self.normalize(text, lang)
        pieces = []
        current = []
        for cluster in _clusters(text):
            if cluster == SPACE_TOKEN:
                _flush(current, pieces)
                pieces.append(SPACE_TOKEN)
            elif cluster[0] in _SPLIT_BOUNDARIES:
                _flush(current, pieces)
                pieces.append(cluster)
            else:
                current.append(cluster)
        _flush(current, pieces)
        return pieces

    def encode(self, text: str, lang: str | None = None, code_mix: bool = False) -> list[int]:
        return self.encode_with_tokens(text, lang, code_mix).ids

    def encode_with_tokens(
        self, text: str, lang: str | None = None, code_mix: bool = False
    ) -> EncodeOutput:
        if code_mix:
            ids: list[int] = []
            tokens: list[str] = []
            offsets: list[tuple[int, int]] = []
            for span in detect_script_spans(text):
                span_lang = _script_to_lang(span["script"], lang)
                encoded = self.encode_with_tokens(span["text"], span_lang, False)
                ids.extend(encoded.ids)
                tokens.extend(encoded.tokens)
                offsets.extend((span["start"] + start, span["start"] + end) for start, end in (encoded.offsets or []))
            return EncodeOutput(ids, tokens, lang, offsets)

        normalized = self.normalize(text, lang)
        ids = []
        tokens = []
        offsets = []
        search_start = 0
        for piece in self.pre_tokenize(normalized, lang):
            piece_start = normalized.find(piece, search_start)
            if piece_start < 0:
                piece_start = search_start
            encoded = self._encode_piece(piece)
            ids.extend(encoded.ids)
            tokens.extend(encoded.tokens)
            offsets.extend((piece_start + start, piece_start + end) for start, end in (encoded.offsets or []))
            search_start = piece_start + len(piece)
        return EncodeOutput(ids, tokens, lang, offsets)

    def encode_batch(
        self,
        texts: Iterable[str],
        lang: str | None = None,
        code_mix: bool = False,
        num_threads: int | None = None,
    ) -> list[list[int]]:
        del num_threads
        return [self.encode(text, lang, code_mix) for text in texts]

    def fertility(self, texts: Iterable[str], lang: str | None = None) -> object:
        texts = list(texts)
        total_words = sum(len(text.split()) for text in texts)
        total_tokens = sum(len(self.encode(text, lang)) for text in texts)
        return type(
            "FertilityReport",
            (),
            {
                "fertility": total_tokens / max(total_words, 1),
                "total_tokens": total_tokens,
                "total_words": total_words,
                "total_sentences": len(texts),
            },
        )()

    def _legacy_normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        out = []
        previous_space = False
        for ch in text:
            ch = _normalize_punctuation(ch)
            if ch in {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}:
                continue
            if ch.isspace():
                if not previous_space:
                    out.append(" ")
                    previous_space = True
                continue
            previous_space = False
            out.append(ch)
        return unicodedata.normalize("NFC", "".join(out).strip())

    def decode(self, ids: Iterable[int]) -> str:
        byte_buf = bytearray()
        result: list[str] = []

        def flush_bytes() -> None:
            if byte_buf:
                result.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf.clear()

        for idx in ids:
            token = self.id_to_token.get(int(idx))
            if token is None:
                continue
            if token in {UNK_TOKEN, PAD_TOKEN}:
                flush_bytes()
                continue
            byte_value = _byte_token_value(token)
            if byte_value is not None:
                byte_buf.append(byte_value)
                continue
            flush_bytes()
            result.append(" " if token == SPACE_TOKEN else token)
        flush_bytes()
        return "".join(result)

    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def _encode_piece(self, piece: str) -> EncodeOutput:
        if piece in self.vocab:
            return EncodeOutput([int(self.vocab[piece])], [piece], offsets=[(0, len(piece))])

        symbols = [(cluster, start, start + len(cluster)) for start, cluster in _cluster_indices(piece)]
        while True:
            best = None
            for idx, pair in enumerate(zip(symbols, symbols[1:])):
                rank = self.ranks.get((pair[0][0], pair[1][0]))
                if rank is not None and (best is None or rank < best[1]):
                    best = (idx, rank)
            if best is None:
                break
            idx = best[0]
            symbols[idx : idx + 2] = [
                (symbols[idx][0] + symbols[idx + 1][0], symbols[idx][1], symbols[idx + 1][2])
            ]

        ids: list[int] = []
        tokens: list[str] = []
        offsets: list[tuple[int, int]] = []
        for symbol, start, end in symbols:
            if symbol in self.vocab:
                ids.append(int(self.vocab[symbol]))
                tokens.append(symbol)
                offsets.append((start, end))
                continue
            for byte in symbol.encode("utf-8"):
                byte_token = f"<0x{byte:02X}>"
                ids.append(int(self.vocab.get(byte_token, self.unk_id)))
                tokens.append(byte_token if byte_token in self.vocab else UNK_TOKEN)
                offsets.append((start, end))
        return EncodeOutput(ids, tokens, offsets=offsets)


def _default_vocab() -> dict[str, int]:
    tokens = [
        PAD_TOKEN,
        "<s>",
        "</s>",
        "<mask>",
        UNK_TOKEN,
        *[f"<0x{idx:02X}>" for idx in range(256)],
        SPACE_TOKEN,
        "न",
        "म",
        "स्",
        "ते",
        "नमस्ते",
        "भा",
        "र",
        "त",
        "भारत",
        "हिं",
        "दी",
        "हिंदी",
        "లు",
        "గు",
        "తెలుగు",
        "నం",
        "స్కా",
        "రం",
        "నమస్కారం",
        "ఇం",
        "డి",
        "యా",
        "ఇండియా",
        "hello",
        "world",
        "India",
        "!",
        "?",
        ",",
        ".",
        "।",
        "-",
        ":",
        ";",
        "(",
        ")",
    ]
    if len(tokens) != len(set(tokens)):
        raise ValueError("built-in vocabulary contains duplicate token strings")
    return {token: idx for idx, token in enumerate(tokens)}


def _byte_token_value(token: str) -> int | None:
    if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
        try:
            return int(token[3:5], 16)
        except ValueError:
            return None
    return None


_SPLIT_BOUNDARIES = set("।॥،؛؟.,!?;:()[]{}\\/|@#$%^&*+=<>`~")


def _base_normalize(text: str) -> str:
    chars = list(unicodedata.normalize("NFC", text))
    out = []
    previous_space = False
    for idx, ch in enumerate(chars):
        ch = _normalize_punctuation(ch)
        if ch in {"\u200b", "\u2060", "\ufeff"}:
            continue
        if ch in {"\u200c", "\u200d"}:
            prev_virama = idx > 0 and _is_virama(chars[idx - 1])
            next_virama = idx + 1 < len(chars) and _is_virama(chars[idx + 1])
            if prev_virama or next_virama:
                out.append(ch)
                previous_space = False
            continue
        if ch.isspace():
            if not previous_space and out:
                out.append(" ")
                previous_space = True
            continue
        previous_space = False
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out).strip())


def _normalize_devanagari(text: str) -> str:
    text = _base_normalize(text)
    pairs = {
        "क": "\u0958",
        "ख": "\u0959",
        "ग": "\u095a",
        "ज": "\u095b",
        "ड": "\u095c",
        "ढ": "\u095d",
        "फ": "\u095e",
        "य": "\u095f",
    }
    chars = list(text)
    out = []
    idx = 0
    while idx < len(chars):
        if idx + 1 < len(chars) and chars[idx + 1] == "\u093c" and chars[idx] in pairs:
            out.append(pairs[chars[idx]])
            idx += 2
        else:
            out.append(chars[idx])
            idx += 1
    return _collapse_repeated("".join(out).replace("।।", "॥"), "\u0902")


def _normalize_dravidian(text: str, family: str) -> str:
    text = _base_normalize(text)
    if family == "tamil":
        return _collapse_repeated(text, "\u0bcd")
    if family == "telugu":
        return _collapse_repeated(text, "\u0c4d")
    if family == "kannada":
        return _collapse_repeated(text, "\u0ccd")
    if family != "malayalam":
        return text
    replacements = {
        "\u0d28\u0d4d\u200d": "\u0d7b",
        "\u0d30\u0d4d\u200d": "\u0d7c",
        "\u0d32\u0d4d\u200d": "\u0d7d",
        "\u0d33\u0d4d\u200d": "\u0d7e",
        "\u0d15\u0d4d\u200d": "\u0d7f",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _normalize_perso_arabic(text: str) -> str:
    mapping = {
        "\u0674": "\u0621",
        "\u0675": "\u0623",
        "\u0647": "\u06c1",
        "\u064a": "\u06cc",
        "\u0643": "\u06a9",
    }
    out = []
    for ch in _base_normalize(text):
        if ch == "\u0640":
            continue
        if "\u0660" <= ch <= "\u0669":
            out.append(str(ord(ch) - 0x0660))
        elif "\u06f0" <= ch <= "\u06f9":
            out.append(str(ord(ch) - 0x06F0))
        else:
            out.append(mapping.get(ch, ch))
    return "".join(out)


def _collapse_repeated(text: str, target: str) -> str:
    out = []
    previous = False
    for ch in text:
        if ch == target:
            if not previous:
                out.append(ch)
            previous = True
        else:
            previous = False
            out.append(ch)
    return "".join(out)


def _is_virama(ch: str) -> bool:
    return ord(ch) in {0x094D, 0x09CD, 0x0A4D, 0x0ACD, 0x0B4D, 0x0BCD, 0x0C4D, 0x0CCD, 0x0D4D}


def _lang_to_family(lang: str | None) -> str | None:
    if not lang:
        return None
    return {
        "hi": "devanagari",
        "mr": "devanagari",
        "ne": "devanagari",
        "sa": "devanagari",
        "bn": "bengali",
        "as": "bengali",
        "pa": "gurmukhi",
        "gu": "gujarati",
        "or": "odia",
        "ta": "tamil",
        "te": "telugu",
        "kn": "kannada",
        "ml": "malayalam",
        "ur": "persian_arabic",
        "ks": "persian_arabic",
        "sd": "persian_arabic",
        "sat": "ol_chiki",
        "mni": "meetei_mayek",
        "mai": "tirhuta",
    }.get(lang.lower())


def _detect_dominant_family(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in text[:200]:
        family = _script_family(ch)
        if family in {"other", "punctuation", "numeric"}:
            continue
        counts[family] = counts.get(family, 0) + 1
    return max(counts, key=counts.get) if counts else "other"


def _script_family(ch: str) -> str:
    cp = ord(ch)
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"
    if 0x0980 <= cp <= 0x09FF:
        return "bengali"
    if 0x0A00 <= cp <= 0x0A7F:
        return "gurmukhi"
    if 0x0A80 <= cp <= 0x0AFF:
        return "gujarati"
    if 0x0B00 <= cp <= 0x0B7F:
        return "odia"
    if 0x0B80 <= cp <= 0x0BFF:
        return "tamil"
    if 0x0C00 <= cp <= 0x0C7F:
        return "telugu"
    if 0x0C80 <= cp <= 0x0CFF:
        return "kannada"
    if 0x0D00 <= cp <= 0x0D7F:
        return "malayalam"
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
        return "persian_arabic"
    if 0x1C50 <= cp <= 0x1C7F:
        return "ol_chiki"
    if 0xABC0 <= cp <= 0xABFF:
        return "meetei_mayek"
    if 0x11480 <= cp <= 0x114DF:
        return "tirhuta"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isdigit():
        return "numeric"
    if ch in _SPLIT_BOUNDARIES:
        return "punctuation"
    return "other"


def detect_script_spans(text: str) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    current_script: str | None = None
    current_text: list[str] = []
    current_start = 0
    byte_pos = 0

    for ch in text:
        script = "other" if ch.isspace() else _script_family(ch)
        char_len = len(ch.encode("utf-8"))
        if current_script is not None and script != current_script:
            spans.append(
                {
                    "script": current_script,
                    "text": "".join(current_text),
                    "start": current_start,
                    "end": byte_pos,
                }
            )
            current_text = []
            current_start = byte_pos
        current_script = script
        current_text.append(ch)
        byte_pos += char_len

    if current_script is not None:
        spans.append(
            {
                "script": current_script,
                "text": "".join(current_text),
                "start": current_start,
                "end": byte_pos,
            }
        )
    return spans


def _script_to_lang(script: object, fallback: str | None = None) -> str | None:
    return {
        "devanagari": "hi",
        "bengali": "bn",
        "gurmukhi": "pa",
        "gujarati": "gu",
        "odia": "or",
        "tamil": "ta",
        "telugu": "te",
        "kannada": "kn",
        "malayalam": "ml",
        "persian_arabic": "ur",
        "ol_chiki": "sat",
        "meetei_mayek": "mni",
        "tirhuta": "mai",
        "latin": None,
    }.get(str(script), fallback)


def _load_merges(path: Path) -> list[tuple[str, str]]:
    merges = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"invalid merge at line {line_no}: expected two tokens")
        merges.append((parts[0], parts[1]))
    return merges


def _normalize_punctuation(ch: str) -> str:
    return {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\uff01": "!",
        "\uff0c": ",",
        "\uff0e": ".",
        "\uff1a": ":",
        "\uff1b": ";",
        "\uff1f": "?",
    }.get(ch, ch)


def _clusters(text: str) -> list[str]:
    clusters = []
    for ch in text:
        if clusters and (unicodedata.combining(ch) or unicodedata.category(ch).startswith("M")):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def _cluster_indices(text: str) -> list[tuple[int, str]]:
    clusters: list[tuple[int, str]] = []
    for idx, ch in enumerate(text):
        if clusters and (unicodedata.combining(ch) or unicodedata.category(ch).startswith("M")):
            start, previous = clusters[-1]
            clusters[-1] = (start, previous + ch)
        else:
            clusters.append((idx, ch))
    return clusters


def _flush(current: list[str], pieces: list[str]) -> None:
    if current:
        pieces.append("".join(current))
        current.clear()
