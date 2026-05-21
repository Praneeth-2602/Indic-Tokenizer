import sys

from inditok import IndicTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

tokenizer = IndicTokenizer()

for text in ["नमस्ते भारत!", "నమస్కారం తెలుగు!", "hello हिंदी India!"]:
    ids = tokenizer.encode(text)
    print(text)
    print(ids)
    print(tokenizer.decode(ids))
