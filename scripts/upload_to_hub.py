from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
library_name: inditok
tags:
  - tokenizer
  - indic
  - sentencepiece
---

# {repo_id}

Rust-backed Indic tokenizer model for `inditok`.

## Files

- `vocab.json`
- `merges.txt`
- `tokenizer_config.json`

## Training Stats

```json
{training_stats}
```

## Intended Use

Use with `IndicTokenizer.from_pretrained("{repo_id}")`.

## Limitations

Fertility depends on corpus quality and language coverage. Re-run the benchmark
suite for your target domain before production use.
"""


def upload_to_hub(model_dir: str | Path, repo_id: str, private: bool = False) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install inditok[hub] to upload models") from exc

    model_path = Path(model_dir)
    required = ["vocab.json", "merges.txt", "tokenizer_config.json"]
    missing = [name for name in required if not (model_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing required model files: {', '.join(missing)}")

    card_path = model_path / "README.md"
    stats_path = model_path / "training_stats.json"
    stats = "{}"
    if stats_path.exists():
        stats = json.dumps(json.loads(stats_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
    card_path.write_text(
        MODEL_CARD_TEMPLATE.format(repo_id=repo_id, training_stats=stats),
        encoding="utf-8",
    )

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(model_path), repo_id=repo_id, repo_type="model")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload an inditok model to HuggingFace Hub")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args(argv)
    upload_to_hub(args.model_dir, args.repo_id, args.private)
    print(f"Uploaded {args.model_dir} to {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
