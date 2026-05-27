import pytest

from inditok import IndicTokenizer

try:
    from hypothesis import given, settings, strategies
except ImportError:  # pragma: no cover - exercised only without dev extras installed
    given = None
    settings = None
    strategies = None

SAMPLE_TEXTS = [
    ("नमस्ते भारत", "hi"),
    ("বাংলা ভাষা", "bn"),
    ("வணக்கம் தமிழ்", "ta"),
    ("నమస్కారం తెలుగు", "te"),
    ("hello नमस्ते world", "hi"),
]


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


if given is not None and settings is not None and strategies is not None:

    @settings(max_examples=40)
    @given(strategies.lists(strategies.sampled_from(SAMPLE_TEXTS), min_size=1, max_size=4))
    def test_offsets_round_trip_for_generated_repetitions(samples):
        _assert_offset_round_trip(samples)

else:

    def test_offsets_round_trip_for_generated_repetitions():
        pytest.skip("hypothesis is not installed")


def _assert_offset_round_trip(samples):
    lang = samples[0][1]
    text = " ".join(text for text, _ in samples)
    tok = IndicTokenizer()
    normalized = tok.normalize(text, lang=lang)
    encoded = tok.encode_with_tokens(text, lang=lang)
    normalized_bytes = normalized.encode("utf-8")

    previous_end = 0
    for token, (start, end) in zip(encoded.tokens, encoded.offsets):
        assert previous_end <= start <= end <= len(normalized_bytes)
        normalized_bytes[start:end].decode("utf-8")
        if not token.startswith("<0x"):
            assert token.replace(" ", "") == normalized_bytes[start:end].decode(
                "utf-8", errors="replace"
            ).replace(" ", "")
        previous_end = end
