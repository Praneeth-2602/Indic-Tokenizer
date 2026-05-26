from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from merge_vocabs import merge_models


DEFAULT_CLUSTERS = {
    "devanagari": {"langs": ["hi", "mr", "ne", "sa", "mai", "kok", "doi", "brx"], "vocab_size": 64000},
    "dravidian": {"langs": ["ta", "te", "kn", "ml"], "vocab_size": 64000},
    "bengali": {"langs": ["bn", "as", "or"], "vocab_size": 32000},
    "perso_arabic": {"langs": ["ur", "ks", "sd"], "vocab_size": 32000},
    "other": {"langs": ["pa", "gu", "mni", "sat"], "vocab_size": 32000},
}


def train_clusters(
    data_dir: str | Path,
    output_dir: str | Path,
    clusters: dict[str, dict[str, object]] | None = None,
    model_type: str = "bpe",
) -> None:
    clusters = clusters or DEFAULT_CLUSTERS
    output = Path(output_dir)
    cluster_root = output / "clusters"
    cluster_root.mkdir(parents=True, exist_ok=True)

    trained_dirs: list[Path] = []
    for name, spec in clusters.items():
        langs = [lang for lang in spec["langs"] if (Path(data_dir) / f"{lang}.txt").exists()]
        if not langs:
            print(f"Skipping {name}: no corpus files found", file=sys.stderr)
            continue
        target = cluster_root / name
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("train.py")),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(target),
            "--vocab-size",
            str(spec["vocab_size"]),
            "--langs",
            ",".join(langs),
            "--model-type",
            model_type,
            "--allocation",
            "proportional",
            "--morpheme-hints",
            "auto",
        ]
        subprocess.check_call(cmd)
        trained_dirs.append(target)

    if not trained_dirs:
        raise FileNotFoundError("no cluster models were trained")
    merge_models(trained_dirs, output)
    (output / "cluster_training_config.json").write_text(
        json.dumps(clusters, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and merge script-cluster inditok models")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clusters-config", type=Path)
    parser.add_argument("--model-type", choices=["bpe", "unigram"], default="bpe")
    args = parser.parse_args(argv)
    clusters = None
    if args.clusters_config:
        clusters = json.loads(args.clusters_config.read_text(encoding="utf-8"))
    train_clusters(args.data_dir, args.output_dir, clusters, args.model_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
