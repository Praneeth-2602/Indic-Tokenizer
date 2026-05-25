from inditok import IndicTokenizer
from inditok.evaluation import compare_tokenizers, evaluate_fertility


def test_fertility_report_shape():
    tok = IndicTokenizer()
    report = tok.fertility(["hello world", "नमस्ते भारत"], lang="hi")
    assert report["fertility"] > 0
    assert report["total_words"] == 4
    assert report["total_sentences"] == 2


def test_evaluate_fertility_returns_structure(tmp_path):
    (tmp_path / "hi.txt").write_text("नमस्ते भारत\n", encoding="utf-8")
    results = evaluate_fertility(IndicTokenizer(), "inditok", tmp_path)
    assert results[0].lang == "hi"
    assert results[0].fertility > 0


def test_compare_tokenizers_markdown(tmp_path):
    (tmp_path / "hi.txt").write_text("hello world\n", encoding="utf-8")
    output = compare_tokenizers({"inditok": IndicTokenizer()}, tmp_path, "markdown")
    assert "| tokenizer |" in output
