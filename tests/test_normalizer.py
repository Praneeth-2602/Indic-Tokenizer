import pytest

from inditok import IndicTokenizer


@pytest.mark.parametrize(
    ("lang", "sample"),
    [
        ("hi", "नमस्ते"),
        ("bn", "বাংলা"),
        ("pa", "ਪੰਜਾਬੀ"),
        ("gu", "ગુજરાતી"),
        ("or", "ଓଡ଼ିଆ"),
        ("ta", "தமிழ்"),
        ("te", "తెలుగు"),
        ("kn", "ಕನ್ನಡ"),
        ("ml", "മലയാളം"),
        ("ur", "اردو"),
        ("sat", "ᱥᱟᱱᱛᱟᱲᱤ"),
        ("mni", "ꯃꯤꯇꯩ"),
    ],
)
def test_normalize_idempotent_for_script_families(lang, sample):
    tok = IndicTokenizer()
    normalized = tok.normalize(sample, lang=lang)
    assert tok.normalize(normalized, lang=lang) == normalized


def test_dravidian_duplicate_viramas_collapse():
    tok = IndicTokenizer()
    assert tok.normalize("க்\u0bcd", lang="ta") == "க்"
    assert tok.normalize("క్\u0c4d", lang="te") == "క్"
    assert tok.normalize("ಕ್\u0ccd", lang="kn") == "ಕ್"
