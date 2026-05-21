mod base;
mod bengali;
mod devanagari;
mod dravidian;
mod gujarati;
mod perso_arabic;

use crate::utils::{detect_dominant_family, lang_to_script_family, ScriptFamily};

#[derive(Clone, Debug, Default)]
pub struct IndicNormalizer;

impl IndicNormalizer {
    pub fn normalize(&self, text: &str) -> String {
        self.normalize_with_lang(text, None)
    }

    pub fn normalize_with_lang(&self, text: &str, lang: Option<&str>) -> String {
        let family = lang
            .and_then(lang_to_script_family)
            .unwrap_or_else(|| detect_dominant_family(text));

        match family {
            ScriptFamily::Devanagari => devanagari::normalize(text),
            ScriptFamily::Tamil
            | ScriptFamily::Telugu
            | ScriptFamily::Kannada
            | ScriptFamily::Malayalam => dravidian::normalize(text, family),
            ScriptFamily::Bengali | ScriptFamily::Odia => bengali::normalize(text),
            ScriptFamily::Gujarati => gujarati::normalize(text),
            ScriptFamily::PersianArabic => perso_arabic::normalize(text),
            _ => base::normalize(text),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::IndicNormalizer;

    #[test]
    fn cleans_spacing_and_zero_width() {
        let normalizer = IndicNormalizer;
        assert_eq!(normalizer.normalize("  नम\u{200D}स्ते\tभारत  "), "नमस्ते भारत");
    }

    #[test]
    fn preserves_virama_adjacent_zwj() {
        let normalizer = IndicNormalizer;
        let input = "\u{0D28}\u{0D4D}\u{200D}";
        assert_eq!(
            normalizer.normalize_with_lang(input, Some("ml")),
            "\u{0D7B}"
        );
    }
}
