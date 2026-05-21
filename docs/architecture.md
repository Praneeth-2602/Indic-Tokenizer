# inditok Architecture

`inditok` is split into a small Rust core and a thin Python API.

- `normalizer`: safe Unicode normalization, whitespace cleanup, zero-width removal, and punctuation canonicalization.
- `pretokenizer`: Unicode grapheme-aware splitting around whitespace and punctuation.
- `bpe`: vocabulary and merge application with deterministic encode/decode behavior.
- `tokenizer`: composition layer used by both Rust tests and PyO3 bindings.
- `python/inditok`: ergonomic Python wrapper, minimal Transformers-style wrapper, and CLI.

The MVP supports Hindi and Telugu through Unicode-safe processing and a small built-in vocabulary for local smoke tests. Real deployments should pass a `vocab.json` and `merges.txt` directory via `from_pretrained`.

