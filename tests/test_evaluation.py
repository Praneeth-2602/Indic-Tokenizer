from __future__ import annotations

from inditok import IndicTokenizer
from inditok.evaluation import evaluate_fertility


def test_evaluate_fertility_resolves_flores101_file_aliases(tmp_path):
    (tmp_path / "hin.txt").write_text("नमस्ते भारत\n", encoding="utf-8")

    rows = evaluate_fertility(IndicTokenizer(), "inditok", tmp_path, ["hi"])

    assert len(rows) == 1
    assert rows[0].lang == "hi"
    assert rows[0].lang_name == "Hindi"
    assert rows[0].total_words == 2


def test_evaluate_fertility_canonicalizes_discovered_flores101_names(tmp_path):
    (tmp_path / "hin.txt").write_text("नमस्ते भारत\n", encoding="utf-8")
    (tmp_path / "ben.txt").write_text("বাংলা ভাষা\n", encoding="utf-8")

    rows = evaluate_fertility(IndicTokenizer(), "inditok", tmp_path)

    assert [row.lang for row in rows] == ["bn", "hi"]
