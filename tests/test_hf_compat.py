from inditok.hf_wrapper import BatchEncoding, IndicHFTokenizer


def test_call_returns_batch_encoding_with_attention_mask():
    hf = IndicHFTokenizer()
    out = hf("नमस्ते भारत")
    assert isinstance(out, BatchEncoding)
    assert "input_ids" in out
    assert "attention_mask" in out
    assert len(out["input_ids"]) == 1


def test_padding_truncation_and_vocab_methods():
    hf = IndicHFTokenizer()
    out = hf(["hello", "hello भारत"], padding=True, truncation=True, max_length=4)
    assert len(out["input_ids"][0]) == len(out["input_ids"][1])
    assert hf.convert_tokens_to_ids("<pad>") == hf.token_to_id("<pad>")
    assert isinstance(hf.get_vocab(), dict)

