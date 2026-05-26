from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


DEFAULT_URLS = [
    # Users can override with --url as dataset mirrors move over time.
    "https://raw.githubusercontent.com/lingo-iitgn/Hinglish-TOP-Dataset/master/data/hinglish.txt",
]


def download_hinglish(output: str | Path, urls: list[str] | None = None) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls or DEFAULT_URLS:
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                text = response.read().decode("utf-8", errors="replace")
            if text.strip():
                target.write_text(text, encoding="utf-8")
                return target
        except Exception as exc:  # pragma: no cover - network path
            errors.append(f"{url}: {exc}")
    raise ConnectionError("could not download Hinglish data:\n" + "\n".join(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a Hinglish corpus file")
    parser.add_argument("--output", default="data/hinglish.txt")
    parser.add_argument("--url", action="append", default=None)
    args = parser.parse_args(argv)
    path = download_hinglish(args.output, args.url)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
