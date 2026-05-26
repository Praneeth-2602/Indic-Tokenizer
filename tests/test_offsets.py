from inditok import IndicTokenizer


def test_offsets_length_matches_tokens():
    tok = IndicTokenizer()
    out = tok.encode_with_tokens("नमस्ते भारत", lang="hi")
    assert len(out.offsets) == len(out.tokens)
    assert len(out.offsets) == len(out.ids)


def test_offsets_point_into_normalized_text_for_representative_inputs():
    tok = IndicTokenizer()
    samples = [
        ("नमस्ते भारत", "hi"),
        ("বাংলা ভাষা", "bn"),
        ("வணக்கம் தமிழ்", "ta"),
        ("నమస్కారం తెలుగు", "te"),
        ("hello नमस्ते world", "hi"),
    ]
    for text, lang in samples:
        normalized = tok.normalize(text, lang=lang)
        encoded = tok.encode_with_tokens(text, lang=lang)
        normalized_bytes = normalized.encode("utf-8")
        for token, (start, end) in zip(encoded.tokens, encoded.offsets):
            assert 0 <= start <= end <= len(normalized_bytes)
            if token.startswith("<0x"):
                continue
            span = normalized_bytes[start:end].decode("utf-8", errors="replace")
            assert token.replace(" ", "") == span.replace(" ", "")
