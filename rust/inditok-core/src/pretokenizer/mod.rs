mod arabic;
mod indic;

use crate::utils::{detect_dominant_family, lang_to_script_family, ScriptFamily};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Clone, Debug, Default)]
pub struct IndicPretokenizer;

impl IndicPretokenizer {
    pub fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut current = String::new();

        for grapheme in text.graphemes(true) {
            if grapheme == " " {
                flush(&mut current, &mut out);
                out.push(" ".to_string());
            } else if is_split_boundary(grapheme) {
                flush(&mut current, &mut out);
                out.push(grapheme.to_string());
            } else {
                current.push_str(grapheme);
            }
        }

        flush(&mut current, &mut out);
        out
    }

    pub fn pre_tokenize_with_lang(&self, text: &str, lang: Option<&str>) -> Vec<String> {
        let family = lang
            .and_then(lang_to_script_family)
            .unwrap_or_else(|| detect_dominant_family(text));

        match family {
            ScriptFamily::PersianArabic => arabic::pre_tokenize(text),
            ScriptFamily::Devanagari
            | ScriptFamily::Bengali
            | ScriptFamily::Gurmukhi
            | ScriptFamily::Gujarati
            | ScriptFamily::Odia
            | ScriptFamily::Tamil
            | ScriptFamily::Telugu
            | ScriptFamily::Kannada
            | ScriptFamily::Malayalam
            | ScriptFamily::OlChiki
            | ScriptFamily::MeeteiMayek
            | ScriptFamily::Tirhuta => indic::pre_tokenize(text),
            _ => self.pre_tokenize(text),
        }
    }
}

fn flush(current: &mut String, out: &mut Vec<String>) {
    if !current.is_empty() {
        out.push(std::mem::take(current));
    }
}

pub(crate) fn is_split_boundary(grapheme: &str) -> bool {
    let Some(ch) = grapheme.chars().next() else {
        return false;
    };
    matches!(
        ch,
        '।' | '॥'
            | '.'
            | ','
            | '!'
            | '?'
            | ';'
            | ':'
            | '('
            | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | '/'
            | '\\'
            | '|'
            | '@'
            | '#'
            | '$'
            | '%'
            | '^'
            | '&'
            | '*'
            | '+'
            | '='
            | '<'
            | '>'
            | '`'
            | '~'
    )
}

#[cfg(test)]
mod tests {
    use super::IndicPretokenizer;

    #[test]
    fn preserves_indic_graphemes() {
        let pre = IndicPretokenizer;
        assert_eq!(
            pre.pre_tokenize("नमस्ते, भारत!"),
            vec!["नमस्ते", ",", " ", "भारत", "!"]
        );
    }
}
