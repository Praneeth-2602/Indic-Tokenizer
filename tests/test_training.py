import sys
from pathlib import Path

from inditok import IndicTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train import MORPHEME_BOUNDARY, audit_corpus, clean_line, inject_morpheme_hints, train  # noqa: E402
from convert_sp_to_inditok import _inditok_piece_variants  # noqa: E402


def test_train_creates_valid_vocab(tmp_path):
    (tmp_path / "hi.txt").write_text("नमस्ते\nभारत\n" * 3, encoding="utf-8")
    train(
        tmp_path,
        tmp_path / "model",
        vocab_size=320,
        langs=["hi"],
        min_frequency=1,
        clean_corpus=False,
    )
    tok = IndicTokenizer.from_pretrained(tmp_path / "model")
    assert tok.decode(tok.encode("नमस्ते", lang="hi")) == "नमस्ते"


def test_morpheme_hints_are_training_only():
    hinted = inject_morpheme_hints("படிக்கிறேன்", "ta")
    assert MORPHEME_BOUNDARY in hinted


def test_clean_line_filters_web_noise():
    assert clean_line("https://a.test x https://b.test", "hi", min_chars=1) is None
    assert clean_line("१२३४५६७", "hi", min_chars=1) is None
    assert clean_line("यह एक साफ लंबी पंक्ति है जिसमें शोर नहीं है", "hi") is not None


def test_audit_corpus_reports_duplicates_and_htmlish():
    report = audit_corpus([("te", "తెలుగు"), ("te", "తెలుగు"), ("te", "<p>hello</p>")])
    assert report["te"]["duplicate_lines"] == 1
    assert report["te"]["htmlish_lines"] == 1


def test_sentencepiece_piece_conversion_is_canonical():
    assert _inditok_piece_variants("▁தமிழ்") == ["தமிழ்"]
    assert _inditok_piece_variants(f"▁படி{MORPHEME_BOUNDARY}க்கிறேன்") == ["படிக்கிறேன்"]
