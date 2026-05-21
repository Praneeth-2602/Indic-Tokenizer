use super::base;

pub fn normalize(text: &str) -> String {
    let normalized = base::normalize(text);
    let normalized = compose_nukta(&normalized);
    let normalized = collapse_repeated(&normalized, '\u{0902}');
    normalize_danda(&normalized)
}

fn compose_nukta(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut idx = 0;

    while idx < chars.len() {
        if idx + 1 < chars.len() && chars[idx + 1] == '\u{093C}' {
            if let Some(composed) = match chars[idx] {
                'क' => Some('\u{0958}'),
                'ख' => Some('\u{0959}'),
                'ग' => Some('\u{095A}'),
                'ज' => Some('\u{095B}'),
                'ड' => Some('\u{095C}'),
                'ढ' => Some('\u{095D}'),
                'फ' => Some('\u{095E}'),
                'य' => Some('\u{095F}'),
                _ => None,
            } {
                out.push(composed);
                idx += 2;
                continue;
            }
        }
        out.push(chars[idx]);
        idx += 1;
    }

    out
}

fn normalize_danda(text: &str) -> String {
    text.replace("।।", "॥")
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
