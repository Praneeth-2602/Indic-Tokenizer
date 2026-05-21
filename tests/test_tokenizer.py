from pathlib import Path

from inditok import IndicTokenizer
from inditok.hf_wrapper import IndicHFTokenizer


def test_normalization_removes_zero_width_and_cleans_spaces():
    tok = IndicTokenizer()
    assert tok.normalize("  नम\u200dस्ते\tभारत  ") == "नमस्ते भारत"


def test_hindi_round_trip_known_tokens():
    tok = IndicTokenizer()
    ids = tok.encode("नमस्ते भारत!")
    assert ids
    assert tok.decode(ids) == "नमस्ते भारत!"


def test_telugu_round_trip_known_tokens():
    tok = IndicTokenizer()
    ids = tok.encode("నమస్కారం తెలుగు!")
    assert ids
    assert tok.decode(ids) == "నమస్కారం తెలుగు!"


def test_batch_encoding():
    tok = IndicTokenizer()
    batch = tok.encode_batch(["नमस्ते भारत", "నమస్కారం తెలుగు"])
    assert len(batch) == 2
    assert all(isinstance(item, list) for item in batch)


def test_punctuation_and_mixed_english():
    tok = IndicTokenizer()
    assert tok.pre_tokenize("hello, हिंदी!") == ["hello", ",", " ", "हिंदी", "!"]
    assert tok.decode(tok.encode("hello India!")) == "hello India!"


def test_save_load(tmp_path: Path):
    tok = IndicTokenizer()
    tok.save_pretrained(tmp_path)
    loaded = IndicTokenizer.from_pretrained(tmp_path)
    assert loaded.decode(loaded.encode("नमस्ते भारत!")) == "नमस्ते भारत!"


def test_hf_wrapper_minimal_api(tmp_path: Path):
    hf = IndicHFTokenizer()
    assert hf("नमस्ते")["input_ids"]
    hf.save_pretrained(tmp_path)
    loaded = IndicHFTokenizer.from_pretrained(tmp_path)
    assert loaded.decode(loaded.encode("తెలుగు")) == "తెలుగు"


def test_zwj_adjacent_to_virama_preserved_and_malayalam_chillu_normalizes():
    tok = IndicTokenizer()
    old_chillu = "\u0d28\u0d4d\u200d"
    assert tok.normalize(old_chillu, lang="ml") == "\u0d7b"
    assert tok.normalize("a\u200db") == "ab"


def test_apostrophe_hyphen_quote_and_underscore_not_split():
    tok = IndicTokenizer()
    assert tok.pre_tokenize("India's Hindi-Urdu _id") == ["India's", " ", "Hindi-Urdu", " ", "_id"]


def test_lang_specific_normalization_rules():
    tok = IndicTokenizer()
    assert "\u095b" in tok.normalize("ज\u093cरूर", lang="hi")
    assert "\u0640" not in tok.normalize("اللہ\u0640", lang="ur")
    assert tok.normalize("\u0660\u0661\u0662", lang="ur") == "012"


def test_empty_and_long_inputs_do_not_crash():
    tok = IndicTokenizer()
    assert tok.normalize("", lang="ta") == ""
    assert tok.encode("", lang="ur") == []
    assert tok.encode("नमस्ते भारत। " * 5000, lang="hi")


def test_encode_output_lang_offsets_and_fertility():
    tok = IndicTokenizer()
    output = tok.encode_with_tokens("hello भारत", lang="hi")
    assert output.lang == "hi"
    assert len(output.offsets) == len(output.ids)
    report = tok.fertility(["नमस्ते भारत"], lang="hi")
    assert {"fertility", "total_tokens", "total_words", "total_sentences"} <= set(report)
