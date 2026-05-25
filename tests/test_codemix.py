from inditok import IndicTokenizer
from inditok._fallback import detect_script_spans


def test_script_span_detection():
    spans = detect_script_spans("hello नमस्ते world")
    assert [span["script"] for span in spans] == ["latin", "other", "devanagari", "other", "latin"]


def test_code_mix_encode_no_crash():
    tok = IndicTokenizer()
    assert tok.encode("hello नमस्ते world", lang="hi", code_mix=True)

