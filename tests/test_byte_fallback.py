from inditok import IndicTokenizer


def test_byte_fallback_round_trip_rare_script():
    tok = IndicTokenizer()
    text = "ᱥᱟᱱᱛᱟᱲᱤ"
    out = tok.encode_with_tokens(text, lang="sat")
    assert "<unk>" not in out.tokens
    assert tok.decode(out.ids) == text

