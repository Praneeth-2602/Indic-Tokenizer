use super::base;

pub fn normalize(text: &str) -> String {
    let normalized = base::normalize(text);
    let normalized = collapse_repeated(&normalized, '\u{09CD}');
    collapse_repeated(&normalized, '\u{0982}')
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
