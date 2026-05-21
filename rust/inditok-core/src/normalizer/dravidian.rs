use super::base;
use crate::utils::ScriptFamily;

pub fn normalize(text: &str, family: ScriptFamily) -> String {
    let normalized = base::normalize(text);
    match family {
        ScriptFamily::Tamil => collapse_repeated(&normalized, '\u{0BCD}'),
        ScriptFamily::Telugu => collapse_repeated(&normalized, '\u{0C4D}'),
        ScriptFamily::Kannada => collapse_repeated(&normalized, '\u{0CCD}'),
        ScriptFamily::Malayalam => normalize_malayalam(&normalized),
        _ => normalized,
    }
}

fn normalize_malayalam(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut idx = 0;

    while idx < chars.len() {
        if idx + 2 < chars.len() && chars[idx + 1] == '\u{0D4D}' && chars[idx + 2] == '\u{200D}' {
            if let Some(chillu) = match chars[idx] {
                '\u{0D28}' => Some('\u{0D7B}'),
                '\u{0D30}' => Some('\u{0D7C}'),
                '\u{0D32}' => Some('\u{0D7D}'),
                '\u{0D33}' => Some('\u{0D7E}'),
                '\u{0D15}' => Some('\u{0D7F}'),
                _ => None,
            } {
                out.push(chillu);
                idx += 3;
                continue;
            }
        }
        out.push(chars[idx]);
        idx += 1;
    }

    out
}

fn collapse_repeated(text: &str, target: char) -> String {
    let mut out = String::with_capacity(text.len());
    let mut previous_target = false;
    for ch in text.chars() {
        if ch == target {
            if !previous_target {
                out.push(ch);
            }
            previous_target = true;
        } else {
            previous_target = false;
            out.push(ch);
        }
    }
    out
}
