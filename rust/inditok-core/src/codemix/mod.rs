use crate::utils::{script_family_of, ScriptFamily};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptSpan {
    pub script: ScriptFamily,
    pub text: String,
    pub start: usize,
    pub end: usize,
}

pub fn contains_ascii_word(text: &str) -> bool {
    text.chars().any(|ch| ch.is_ascii_alphabetic())
}

pub fn detect_script_spans(text: &str) -> Vec<ScriptSpan> {
    let mut spans = Vec::new();
    let mut current_family = None;
    let mut current_text = String::new();
    let mut current_start = 0usize;
    let mut byte_pos = 0usize;

    for ch in text.chars() {
        let family = if ch.is_whitespace() {
            ScriptFamily::Other
        } else {
            script_family_of(ch)
        };
        let char_len = ch.len_utf8();

        if Some(family) != current_family {
            if let Some(script) = current_family {
                spans.push(ScriptSpan {
                    script,
                    text: std::mem::take(&mut current_text),
                    start: current_start,
                    end: byte_pos,
                });
            }
            current_family = Some(family);
            current_start = byte_pos;
        }

        current_text.push(ch);
        byte_pos += char_len;
    }

    if let Some(script) = current_family {
        spans.push(ScriptSpan {
            script,
            text: current_text,
            start: current_start,
            end: byte_pos,
        });
    }

    spans
}

pub fn is_code_mixed(text: &str) -> bool {
    let families = text
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .map(script_family_of)
        .filter(|family| {
            !matches!(
                family,
                ScriptFamily::Other | ScriptFamily::Punctuation | ScriptFamily::Numeric
            )
        })
        .collect::<HashSet<_>>();
    families.len() > 1
}
