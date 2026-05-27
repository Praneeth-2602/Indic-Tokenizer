import os
import pandas as pd
from huggingface_hub import hf_hub_download

langs = [
    "hin_Deva",
    "ben_Beng",
    "tam_Taml",
    "tel_Telu"
]

save_dir = "benchmarks/data"
os.makedirs(save_dir, exist_ok=True)

for lang in langs:
    print(f"Downloading {lang}...")

    # DEV
    dev_file = hf_hub_download(
        repo_id="facebook/flores",
        repo_type="dataset",
        filename=f"{lang}/dev-00000-of-00001.parquet"
    )

    # DEVTEST
    devtest_file = hf_hub_download(
        repo_id="facebook/flores",
        repo_type="dataset",
        filename=f"{lang}/devtest-00000-of-00001.parquet"
    )

    # Read parquet
    dev_df = pd.read_parquet(dev_file)
    devtest_df = pd.read_parquet(devtest_file)

    # Save txt
    dev_txt = os.path.join(save_dir, f"{lang}_dev.txt")
    devtest_txt = os.path.join(save_dir, f"{lang}_devtest.txt")

    dev_df["sentence"].to_csv(
        dev_txt,
        index=False,
        header=False
    )

    devtest_df["sentence"].to_csv(
        devtest_txt,
        index=False,
        header=False
    )

    print(f"Saved {lang}")

print("Done.")