# inditok

`inditok` is a Rust-backed tokenizer library for Indian languages with Python bindings via PyO3. The MVP focuses on Hindi and Telugu, Unicode-safe normalization, Indic-aware pre-tokenization, simple BPE vocabulary loading, a Python API, CLI utilities, tests, and benchmarks.

## Install

```bash
pip install -e ".[dev]"
maturin develop
```

## Python Usage

```python
from inditok import IndicTokenizer

tok = IndicTokenizer()
ids = tok.encode("नमस्ते भारत!")
text = tok.decode(ids)
batch = tok.encode_batch(["नमस्ते भारत", "నమస్కారం తెలుగు"])
```

Load a file-backed tokenizer:

```python
tok = IndicTokenizer.from_pretrained("path/to/model-dir")
tok.save_pretrained("out-dir")
```

The model directory contains:

- `vocab.json`: JSON object mapping token strings to integer ids. It must include `<unk>`.
- `merges.txt`: one merge pair per line, matching the common BPE text format.

## CLI

```bash
inditok encode "नमस्ते भारत!" --tokens
inditok decode 6 1 10 29
inditok normalize "  नमस्ते\tभारत  "
```

## Tests

```bash
cargo test --manifest-path rust/inditok-core/Cargo.toml
pytest
```

## Benchmarks

```bash
python benchmarks/bench_tokenizers.py
python benchmarks/bench_tokenizers.py --include-optional
```

The optional benchmark path compares against `tiktoken` and a HuggingFace tokenizer when those packages and model files are available.

## Architecture

See [docs/architecture.md](docs/architecture.md).

