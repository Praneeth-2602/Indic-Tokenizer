use super::base;

pub fn normalize(text: &str) -> String {
    base::normalize(text)
        .chars()
        .filter_map(|ch| match ch {
            '\u{0640}' => None,
            '\u{0674}' => Some('\u{0621}'),
            '\u{0675}' => Some('\u{0623}'),
            '\u{0647}' => Some('\u{06C1}'),
            '\u{064A}' => Some('\u{06CC}'),
            '\u{0643}' => Some('\u{06A9}'),
            '\u{0660}'..='\u{0669}' => char::from_u32((ch as u32) - 0x0660 + 0x0030),
            '\u{06F0}'..='\u{06F9}' => char::from_u32((ch as u32) - 0x06F0 + 0x0030),
            _ => Some(ch),
        })
        .collect()
}
