from datasets import load_dataset
from tqdm import tqdm
import os

LANGUAGES = {
    "ta": "tam_Taml",
    "bn": "ben_Beng",
}

LINES_PER_LANG = 500_000
OUTPUT_DIR = "datasets"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def sample_language(lang_code, split_name):
    print(f"\nLoading {lang_code} ({split_name})...")

    dataset = load_dataset(
        "ai4bharat/IndicCorpV2",
        "indiccorp_v2",
        split=split_name,
        streaming=True,
    )

    output_path = os.path.join(OUTPUT_DIR, f"{lang_code}.txt")

    count = 0
    seen = set()

    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, total=LINES_PER_LANG):

            text = item.get("text", "").strip()

            if not text:
                continue

            if len(text) < 10:
                continue

            if text in seen:
                continue

            seen.add(text)

            f.write(text.replace("\n", " ") + "\n")

            count += 1

            if count >= LINES_PER_LANG:
                break

    print(f"Saved {count} lines to {output_path}")


for lang, split_name in LANGUAGES.items():
    sample_language(lang, split_name)