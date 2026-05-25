from inditok import IndicTokenizer


def test_offsets_length_matches_tokens():
    tok = IndicTokenizer()
    out = tok.encode_with_tokens("नमस्ते भारत", lang="hi")
    assert len(out.offsets) == len(out.tokens)
    assert len(out.offsets) == len(out.ids)

