use crate::utils::is_virama;
use unicode_normalization::UnicodeNormalization;

pub fn normalize(text: &str) -> String {
    let chars: Vec<char> = text.nfc().collect();
    let mut normalized = String::with_capacity(text.len());
    let mut previous_was_space = false;

    for (idx, ch) in chars.iter().copied().enumerate() {
        let mapped = normalize_punctuation(ch);

        if matches!(mapped, '\u{200B}' | '\u{2060}' | '\u{FEFF}') {
            continue;
        }

        if matches!(mapped, '\u{200C}' | '\u{200D}') {
            let prev_is_virama = idx > 0 && is_virama(chars[idx - 1]);
            let next_is_virama = idx + 1 < chars.len() && is_virama(chars[idx + 1]);
            if prev_is_virama || next_is_virama {
                normalized.push(mapped);
                previous_was_space = false;
            }
            continue;
        }

        if mapped.is_whitespace() {
            if !previous_was_space && !normalized.is_empty() {
                normalized.push(' ');
                previous_was_space = true;
            }
            continue;
        }

        previous_was_space = false;
        normalized.push(mapped);
    }

    normalized.trim().nfc().collect()
}

fn normalize_punctuation(ch: char) -> char {
    match ch {
        '\u{2018}' | '\u{2019}' => '\'',
        '\u{201C}' | '\u{201D}' => '"',
        '\u{2013}' | '\u{2014}' | '\u{2212}' => '-',
        '\u{FF01}' => '!',
        '\u{FF0C}' => ',',
        '\u{FF0E}' => '.',
        '\u{FF1A}' => ':',
        '\u{FF1B}' => ';',
        '\u{FF1F}' => '?',
        _ => ch,
    }
}
