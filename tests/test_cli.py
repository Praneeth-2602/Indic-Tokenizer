from pathlib import Path

from inditok.cli import main


def test_cli_pre_tokenize(capsys):
    assert main(["pre-tokenize", "नमस्ते, भारत", "--lang", "hi"]) == 0
    assert "नमस्ते" in capsys.readouterr().out


def test_cli_detect_script(capsys):
    assert main(["detect-script", "hello नमस्ते"]) == 0
    assert "devanagari" in capsys.readouterr().out


def test_cli_fertility(tmp_path: Path, capsys):
    data = tmp_path / "hi.txt"
    data.write_text("नमस्ते भारत\n", encoding="utf-8")
    assert main(["fertility", "--input", str(data), "--lang", "hi"]) == 0
    assert "inditok" in capsys.readouterr().out
