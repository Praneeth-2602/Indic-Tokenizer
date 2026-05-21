from inditok import IndicTokenizer


def test_fertility_report_shape():
    tok = IndicTokenizer()
    report = tok.fertility(["hello world", "नमस्ते भारत"], lang="hi")
    assert report["fertility"] > 0
    assert report["total_words"] == 4
    assert report["total_sentences"] == 2
