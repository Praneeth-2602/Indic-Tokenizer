#!/usr/bin/env python3
"""
inditok corpus downloader — scripts/download_corpus.py

Downloads training corpora for inditok from open, freely available sources.
All sources are CC-0, CC-BY, or equivalent open licenses.

Usage:
    python scripts/download_corpus.py --langs hi bn ta te --output-dir ./data --max-lines 500000
    python scripts/download_corpus.py --langs all --output-dir ./data --source indiccorp
    python scripts/download_corpus.py --langs hi --output-dir ./data --source cc100 --max-lines 100000

Sources (in order of quality for tokenizer training):
    1. indiccorp  — AI4Bharat IndicCorp v2 (best, 20.9B tokens, CC-0)
    2. cc100      — Common Crawl CC-100 (good, all languages, CC-BY)
    3. oscar      — OSCAR 23.01 (good, filtered, research license)
    4. wikipedia  — Wikipedia dumps (clean, smaller, CC-BY-SA)
    5. samanantar — AI4Bharat parallel corpus (Indic-English pairs, CC-0)
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
import urllib.request
from pathlib import Path

# ─── Language config ──────────────────────────────────────────────────────────

# ISO 639-1/3 → display name + script family
LANGUAGES: dict[str, dict] = {
    "hi":  {"name": "Hindi",     "script": "devanagari",   "cc100": "hi",  "oscar": "hi", "wiki": "hi"},
    "bn":  {"name": "Bengali",   "script": "bengali",      "cc100": "bn",  "oscar": "bn", "wiki": "bn"},
    "ta":  {"name": "Tamil",     "script": "tamil",        "cc100": "ta",  "oscar": "ta", "wiki": "ta"},
    "te":  {"name": "Telugu",    "script": "telugu",       "cc100": "te",  "oscar": "te", "wiki": "te"},
    "mr":  {"name": "Marathi",   "script": "devanagari",   "cc100": "mr",  "oscar": "mr", "wiki": "mr"},
    "gu":  {"name": "Gujarati",  "script": "gujarati",     "cc100": "gu",  "oscar": "gu", "wiki": "gu"},
    "kn":  {"name": "Kannada",   "script": "kannada",      "cc100": "kn",  "oscar": "kn", "wiki": "kn"},
    "ml":  {"name": "Malayalam", "script": "malayalam",    "cc100": "ml",  "oscar": "ml", "wiki": "ml"},
    "pa":  {"name": "Punjabi",   "script": "gurmukhi",     "cc100": "pa",  "oscar": "pa", "wiki": "pa"},
    "or":  {"name": "Odia",      "script": "odia",         "cc100": "or",  "oscar": "or", "wiki": "or"},
    "as":  {"name": "Assamese",  "script": "bengali",      "cc100": "as",  "oscar": "as", "wiki": "as"},
    "ur":  {"name": "Urdu",      "script": "perso_arabic", "cc100": "ur",  "oscar": "ur", "wiki": "ur"},
    "ne":  {"name": "Nepali",    "script": "devanagari",   "cc100": "ne",  "oscar": "ne", "wiki": "ne"},
    "sa":  {"name": "Sanskrit",  "script": "devanagari",   "cc100": "sa",  "oscar": "sa", "wiki": "sa"},
    "mai": {"name": "Maithili",  "script": "devanagari",   "cc100": None,  "oscar": None, "wiki": "mai"},
    "kok": {"name": "Konkani",   "script": "devanagari",   "cc100": None,  "oscar": None, "wiki": "gom"},
    "doi": {"name": "Dogri",     "script": "devanagari",   "cc100": None,  "oscar": None, "wiki": "dgo"},
    "mni": {"name": "Manipuri",  "script": "meitei_mayek", "cc100": None,  "oscar": None, "wiki": "mni"},
    "brx": {"name": "Bodo",      "script": "devanagari",   "cc100": None,  "oscar": None, "wiki": "brx"},
    "sat": {"name": "Santali",   "script": "ol_chiki",     "cc100": None,  "oscar": None, "wiki": "sat"},
    "ks":  {"name": "Kashmiri",  "script": "perso_arabic", "cc100": "ks",  "oscar": None, "wiki": "ks"},
    "sd":  {"name": "Sindhi",    "script": "perso_arabic", "cc100": "sd",  "oscar": None, "wiki": "sd"},
}

# ─── Source URLs ───────────────────────────────────────────────────────────────

def indiccorp_v2_url(lang: str) -> str:
    """
    IndicCorp v2 — HuggingFace Datasets (ai4bharat/IndicCorpV2)
    License: CC-0
    Size: up to 5.5B tokens for Hindi, ~200M for low-resource languages
    Best for: tokenizer training (clean, monolingual, deduped)
    
    Download via HuggingFace Datasets library (recommended):
        from datasets import load_dataset
        ds = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2",
                          data_dir=f"data/{lang2flores[lang]}", streaming=True)
    """
    # FLORES-200 language codes used by IndicCorp v2
    flores_codes = {
        "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
        "mr": "mar_Deva", "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym",
        "pa": "pan_Guru", "or": "ory_Orya", "as": "asm_Beng", "ur": "urd_Arab",
        "ne": "npi_Deva", "sa": "san_Deva", "mai": "mai_Deva", "kok": "gom_Deva",
        "mni": "mni_Mtei", "brx": "brx_Deva", "sat": "sat_Olck",
    }
    code = flores_codes.get(lang, lang)
    return f"https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/{code}/train-00000-of-00001.parquet"


def cc100_url(lang: str) -> str:
    """
    CC-100 — University of Edinburgh / Facebook Research
    License: Common Crawl terms (open)
    Size: Hindi ~20GB, Bengali ~6GB, Tamil ~5GB, Telugu ~3GB (compressed)
    Direct download: data.statmt.org/cc-100/{lang}.txt.xz
    """
    return f"https://data.statmt.org/cc-100/{lang}.txt.xz"


def wikipedia_url(lang: str, date: str = "20240101") -> str:
    """
    Wikipedia dumps — Wikimedia Foundation
    License: CC-BY-SA 4.0
    Size: Hindi ~700MB, Bengali ~500MB, Tamil ~300MB, Telugu ~250MB (compressed)
    Best for: clean, high-quality text; use with wikiextractor after download
    """
    return (
        f"https://dumps.wikimedia.org/{lang}wiki/{date}/"
        f"{lang}wiki-{date}-pages-articles.xml.bz2"
    )


def oscar_hf_snippet(lang: str) -> str:
    """
    OSCAR 23.01 — HuggingFace Datasets
    License: CC-BY 4.0 (requires HF account + agreement)
    Load with:
        from datasets import load_dataset
        ds = load_dataset("oscar-corpus/OSCAR-2301", language="{lang}",
                          use_auth_token=True, streaming=True)
    """
    return f'load_dataset("oscar-corpus/OSCAR-2301", language="{lang}", streaming=True)'


def samanantar_snippet(lang: str) -> str:
    """
    Samanantar — AI4Bharat parallel corpus (English ↔ Indic)
    License: CC-0
    Note: parallel corpus — extract only Indic side for monolingual tokenizer training
    Load with:
        from datasets import load_dataset
        ds = load_dataset("ai4bharat/samanantar", f"en-{lang}")
        indic_texts = [row["{lang}"] for row in ds["train"]]
    """
    return f'load_dataset("ai4bharat/samanantar", "en-{lang}")'


# ─── Downloader ───────────────────────────────────────────────────────────────

class ProgressBar:
    def __init__(self, total: int, desc: str):
        self.total = total
        self.desc = desc
        self.current = 0

    def update(self, n: int):
        self.current += n
        pct = min(100, int(self.current / max(self.total, 1) * 100))
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  {self.desc}: [{bar}] {pct}%", end="", flush=True)

    def close(self):
        print()


def download_file(url: str, dest: Path, desc: str) -> bool:
    """Download a file with progress reporting. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "inditok-downloader/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            bar = ProgressBar(total, desc)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                while chunk := resp.read(65536):
                    f.write(chunk)
                    bar.update(len(chunk))
            bar.close()
        return True
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}", file=sys.stderr)
        return False


def extract_wikipedia(xml_bz2: Path, output_txt: Path, max_lines: int) -> int:
    """
    Extract plain text from Wikipedia XML dump using wikiextractor.
    Requires: pip install wikiextractor
    Falls back to basic extraction if wikiextractor not available.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "wikiextractor", str(xml_bz2),
             "--output", str(output_txt.parent / "wiki_extracted"),
             "--bytes", "100M", "--quiet"],
            capture_output=True, text=True
        )
        # Concatenate extracted files
        extracted_dir = output_txt.parent / "wiki_extracted"
        count = 0
        with open(output_txt, "w", encoding="utf-8") as out:
            for subdir in sorted(extracted_dir.rglob("wiki_*")):
                for line in open(subdir, encoding="utf-8"):
                    line = line.strip()
                    if line and not line.startswith("<"):
                        out.write(line + "\n")
                        count += 1
                        if count >= max_lines:
                            return count
        return count
    except ImportError:
        print("  wikiextractor not found. Install: pip install wikiextractor", file=sys.stderr)
        return 0


def stream_indiccorp(lang: str, output_path: Path, max_lines: int) -> int:
    """Stream IndicCorp v2 via HuggingFace Datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets not found. Install: pip install datasets", file=sys.stderr)
        return 0

    flores_codes = {
        "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
        "mr": "mar_Deva", "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym",
        "pa": "pan_Guru", "or": "ory_Orya", "as": "asm_Beng", "ur": "urd_Arab",
        "ne": "npi_Deva", "sa": "san_Deva",
    }
    flores_code = flores_codes.get(lang)
    if not flores_code:
        print(f"  IndicCorp v2 not available for {lang}", file=sys.stderr)
        return 0

    print(f"  Streaming IndicCorp v2 for {lang} ({flores_code})...")
    ds = load_dataset(
        "ai4bharat/IndicCorpV2", "indiccorp_v2",
        data_dir=f"data/{flores_code}",
        split="train",
        streaming=True,
    )
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            text = item.get("text", "").strip()
            if text:
                f.write(text + "\n")
                count += 1
                if count % 10000 == 0:
                    print(f"\r  {count:,} lines written...", end="", flush=True)
                if count >= max_lines:
                    break
    print(f"\r  ✓ {count:,} lines written to {output_path.name}")
    return count


def stream_cc100(lang: str, output_path: Path, max_lines: int) -> int:
    """Stream CC-100 via HuggingFace Datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        return 0

    print(f"  Streaming CC-100 for {lang}...")
    try:
        ds = load_dataset("statmt/cc100", lang=lang, split="train", streaming=True)
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for item in ds:
                text = item.get("text", "").strip()
                if text:
                    f.write(text + "\n")
                    count += 1
                    if count % 10000 == 0:
                        print(f"\r  {count:,} lines...", end="", flush=True)
                    if count >= max_lines:
                        break
        print(f"\r  ✓ {count:,} lines written")
        return count
    except Exception as e:
        print(f"  CC-100 stream failed: {e}", file=sys.stderr)
        return 0


# ─── Main pipeline ────────────────────────────────────────────────────────────

def download_for_lang(lang: str, output_dir: Path, source: str, max_lines: int) -> None:
    info = LANGUAGES[lang]
    output_path = output_dir / f"{lang}.txt"

    if output_path.exists() and output_path.stat().st_size > 1000:
        print(f"  ✓ {lang}.txt already exists ({output_path.stat().st_size // 1024:,} KB), skipping.")
        return

    print(f"\n{'─' * 60}")
    print(f"  Language: {info['name']} ({lang}) | Script: {info['script']}")
    print(f"  Source:   {source} | Max lines: {max_lines:,}")
    print(f"{'─' * 60}")

    count = 0
    if source == "indiccorp":
        count = stream_indiccorp(lang, output_path, max_lines)
    elif source == "cc100":
        count = stream_cc100(lang, output_path, max_lines)
    elif source == "wikipedia":
        wiki_code = info.get("wiki", lang)
        url = wikipedia_url(wiki_code)
        bz2_path = output_dir / f"{lang}_wiki.xml.bz2"
        print(f"  Downloading Wikipedia dump from:\n  {url}")
        if download_file(url, bz2_path, f"wiki-{lang}"):
            count = extract_wikipedia(bz2_path, output_path, max_lines)
            bz2_path.unlink(missing_ok=True)
    elif source == "auto":
        # Try sources in order of quality
        for src in ["indiccorp", "cc100", "wikipedia"]:
            if info.get(src if src != "wikipedia" else "wiki"):
                count = download_for_lang.__wrapped__(lang, output_dir, src, max_lines)
                if count > 0:
                    break

    if count == 0:
        print(f"  ✗ No data downloaded for {lang}. See manual download instructions below.")
        print_manual_instructions(lang, info)
    else:
        size_kb = output_path.stat().st_size // 1024
        print(f"  ✓ Done: {output_path.name} ({count:,} lines, {size_kb:,} KB)")


def print_manual_instructions(lang: str, info: dict) -> None:
    """Print copy-paste instructions for manual download."""
    print(f"""
  Manual download options for {info['name']} ({lang}):

  Option 1 — IndicCorp v2 (BEST, CC-0 license):
  ─────────────────────────────────────────────
  pip install datasets
  python3 -c "
  from datasets import load_dataset
  ds = load_dataset('ai4bharat/IndicCorpV2', 'indiccorp_v2',
                    data_dir='data/{info.get('flores', lang + '_Deva')}',
                    streaming=True, split='train')
  with open('data/{lang}.txt', 'w') as f:
      for i, row in enumerate(ds):
          f.write(row['text'].strip() + '\\\\n')
          if i >= 500000: break
  "

  Option 2 — CC-100 (Common Crawl, open license):
  ────────────────────────────────────────────────
  # Direct file download (~compressed):
  wget https://data.statmt.org/cc-100/{lang}.txt.xz
  xz -d {lang}.txt.xz
  head -500000 {lang}.txt > data/{lang}.txt

  # OR via HuggingFace Datasets:
  python3 -c "
  from datasets import load_dataset
  ds = load_dataset('statmt/cc100', lang='{lang}', streaming=True, split='train')
  with open('data/{lang}.txt', 'w') as f:
      for i, row in enumerate(ds):
          f.write(row['text'].strip() + '\\\\n')
          if i >= 500000: break
  "

  Option 3 — Wikipedia (clean, CC-BY-SA):
  ────────────────────────────────────────
  pip install wikiextractor
  wget https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2
  python3 -m wikiextractor {lang}wiki-latest-pages-articles.xml.bz2 -o wiki_out/
  find wiki_out -name 'wiki_*' | xargs cat | grep -v '<' > data/{lang}.txt

  Option 4 — OSCAR 23.01 (requires HF login):
  ─────────────────────────────────────────────
  huggingface-cli login
  python3 -c "
  from datasets import load_dataset
  ds = load_dataset('oscar-corpus/OSCAR-2301', language='{lang}',
                    use_auth_token=True, streaming=True, split='train')
  with open('data/{lang}.txt', 'w') as f:
      for i, row in enumerate(ds):
          f.write(row['text'].strip() + '\\\\n')
          if i >= 500000: break
  "
""")


def print_corpus_sources_table() -> None:
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPEN CORPUS SOURCES FOR INDITOK                          │
├──────────────────┬──────────────────┬──────────────────┬────────────────────┤
│ Source           │ License          │ Size (4 langs)   │ Quality            │
├──────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ IndicCorp v2     │ CC-0 (best)      │ ~30GB total      │ ★★★★★ Best         │
│ (ai4bharat)      │                  │ hi: 8.5B tokens  │ Cleaned, deduped   │
│                  │                  │ bn: 1.3B tokens  │ 22 Indic languages │
│                  │                  │ ta: 1.1B tokens  │                    │
│                  │                  │ te: 0.8B tokens  │                    │
├──────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ CC-100           │ CC-BY / open     │ ~45GB compressed │ ★★★★ Good          │
│ (statmt.org)     │                  │ hi: 20GB         │ Raw web crawl      │
│                  │                  │ bn: 6GB          │ Needs filtering    │
│                  │                  │ ta: 5GB          │ 100+ languages     │
│                  │                  │ te: 3GB          │                    │
├──────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ OSCAR 23.01      │ CC-BY 4.0        │ ~20GB filtered   │ ★★★★ Good          │
│ (oscar-corpus)   │ (HF account req) │ hi: 4.5GB        │ Quality annotated  │
│                  │                  │ bn: 2.1GB        │ Near-dedup LSH     │
│                  │                  │ ta: 1.8GB        │                    │
├──────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ Wikipedia        │ CC-BY-SA 4.0     │ ~2.5GB total     │ ★★★ Good, small    │
│ (wikimedia.org)  │                  │ hi: 700MB        │ High quality text  │
│                  │                  │ bn: 500MB        │ But small corpus   │
│                  │                  │ ta: 300MB        │                    │
│                  │                  │ te: 250MB        │                    │
├──────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ Samanantar       │ CC-0             │ 49.6M pairs      │ ★★★ Good           │
│ (ai4bharat)      │                  │ Use Indic side   │ Parallel corpus    │
│                  │                  │ only for mono    │ Extract target     │
│                  │                  │ training         │ side only          │
└──────────────────┴──────────────────┴──────────────────┴────────────────────┘

  RECOMMENDATION:
  For tokenizer training use IndicCorp v2 as primary source.
  Supplement with CC-100 for languages where IndicCorp is small.
  Use Wikipedia for quality filtering reference.

  MINIMUM recommended lines per language for a 64k vocab:
    High-resource  (hi, bn, ta, te, mr): 1,000,000+ lines
    Mid-resource   (gu, kn, ml, pa, or): 500,000+ lines
    Low-resource   (mai, kok, doi, mni): 50,000+ lines (use all available)
""")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Indic language training corpora for inditok",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Hindi and Bengali from IndicCorp v2 (best source)
  python scripts/download_corpus.py --langs hi bn --source indiccorp

  # Download all 4 training languages with 500k lines each from CC-100
  python scripts/download_corpus.py --langs hi bn ta te --source cc100 --max-lines 500000

  # Show all available sources and corpus sizes
  python scripts/download_corpus.py --list-sources

  # Print manual download instructions for Tamil
  python scripts/download_corpus.py --langs ta --show-instructions
        """
    )
    parser.add_argument(
        "--langs", nargs="+", default=["hi", "bn", "ta", "te"],
        help="Language codes to download (default: hi bn ta te). Use 'all' for all 22."
    )
    parser.add_argument(
        "--source", choices=["indiccorp", "cc100", "oscar", "wikipedia", "auto"],
        default="indiccorp",
        help="Corpus source to use (default: indiccorp)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data"),
        help="Directory to save corpus files (default: ./data)"
    )
    parser.add_argument(
        "--max-lines", type=int, default=500_000,
        help="Maximum lines per language (default: 500000)"
    )
    parser.add_argument(
        "--list-sources", action="store_true",
        help="Print corpus source comparison table and exit"
    )
    parser.add_argument(
        "--show-instructions", action="store_true",
        help="Print manual download instructions for selected languages and exit"
    )
    args = parser.parse_args()

    if args.list_sources:
        print_corpus_sources_table()
        return 0

    langs = list(LANGUAGES.keys()) if args.langs == ["all"] else args.langs
    invalid = [l for l in langs if l not in LANGUAGES]
    if invalid:
        print(f"Unknown language codes: {invalid}", file=sys.stderr)
        print(f"Valid codes: {sorted(LANGUAGES.keys())}", file=sys.stderr)
        return 1

    if args.show_instructions:
        for lang in langs:
            print_manual_instructions(lang, LANGUAGES[lang])
        return 0

    print(f"\n{'═' * 60}")
    print(f"  inditok corpus downloader")
    print(f"  Languages: {', '.join(langs)}")
    print(f"  Source:    {args.source}")
    print(f"  Output:    {args.output_dir}/")
    print(f"  Max lines: {args.max_lines:,} per language")
    print(f"{'═' * 60}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    for lang in langs:
        download_for_lang(lang, args.output_dir, args.source, args.max_lines)

    elapsed = time.perf_counter() - start
    print(f"\n{'═' * 60}")
    print(f"  Done in {elapsed:.1f}s.")
    print(f"  Files saved to: {args.output_dir.resolve()}/")

    # Print summary
    total_lines = 0
    total_size = 0
    for lang in langs:
        p = args.output_dir / f"{lang}.txt"
        if p.exists():
            lines = sum(1 for _ in open(p, encoding="utf-8"))
            size = p.stat().st_size
            total_lines += lines
            total_size += size
            print(f"  {lang}: {lines:>10,} lines  {size // 1024:>8,} KB")
    print(f"{'─' * 60}")
    print(f"  Total: {total_lines:>10,} lines  {total_size // 1024:>8,} KB")
    print(f"{'═' * 60}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())