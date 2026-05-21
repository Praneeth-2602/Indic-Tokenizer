from inditok import IndicTokenizer


def test_code_mixed_script_boundaries_are_tokenized():
    tok = IndicTokenizer()
    assert tok.pre_tokenize("hello नमस्ते world", lang="hi") == [
        "hello",
        " ",
        "नमस्ते",
        " ",
        "world",
    ]


def test_urdu_rtl_order_and_punctuation():
    tok = IndicTokenizer()
    assert tok.pre_tokenize("اردو، زبان؟", lang="ur") == ["اردو", "،", " ", "زبان", "؟"]
