from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_benchmark import run_benchmark


def generate_leaderboard(input_json: str | Path, output_html: str | Path) -> Path:
    rows = json.loads(Path(input_json).read_text(encoding="utf-8"))
    html_rows = "\n".join(
        "<tr>"
        f"<td>{row['tokenizer_name']}</td>"
        f"<td>{row['lang']}</td>"
        f"<td>{row['lang_name']}</td>"
        f"<td>{row['fertility']:.2f}</td>"
        f"<td>{row['unk_rate']:.2f}</td>"
        f"<td>{row['chars_per_token']:.2f}</td>"
        "</tr>"
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>inditok Fertility Leaderboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d5dde5; padding: .55rem; text-align: left; }}
    th {{ background: #eef3f7; cursor: pointer; }}
  </style>
</head>
<body>
  <h1>inditok Fertility Leaderboard</h1>
  <table>
    <thead><tr><th>Tokenizer</th><th>Lang</th><th>Name</th><th>Fertility</th><th>UNK rate</th><th>Chars/token</th></tr></thead>
    <tbody>
      {html_rows}
    </tbody>
  </table>
</body>
</html>
"""
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate static leaderboard HTML")
    parser.add_argument("input_json", nargs="?")
    parser.add_argument("--output", default="docs/leaderboard.html")
    parser.add_argument("--model", default=None)
    parser.add_argument("--benchmark-dir", default="benchmarks/data")
    parser.add_argument("--langs", nargs="*", default=None)
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args(argv)
    input_json = args.input_json
    if input_json is None:
        rows = run_benchmark(args.model, args.benchmark_dir, args.langs, args.include_optional)
        tmp = Path("benchmarks/results.json")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        input_json = str(tmp)
    generate_leaderboard(input_json, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
