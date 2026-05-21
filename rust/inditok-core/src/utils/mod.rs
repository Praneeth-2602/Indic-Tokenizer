pub fn is_supported_indic_char(ch: char) -> bool {
    let cp = ch as u32;
    matches!(cp,
        0x0900..=0x097F
        | 0x0980..=0x09FF
        | 0x0A00..=0x0A7F
        | 0x0A80..=0x0AFF
        | 0x0B00..=0x0B7F
        | 0x0B80..=0x0BFF
        | 0x0C00..=0x0C7F
        | 0x0C80..=0x0CFF
        | 0x0D00..=0x0D7F
        | 0x0600..=0x06FF
        | 0x0750..=0x077F
        | 0x1C50..=0x1C7F
        | 0xABC0..=0xABFF
        | 0x11480..=0x114DF
    )
}

pub fn is_virama(ch: char) -> bool {
    matches!(
        ch as u32,
        0x094D | 0x09CD | 0x0A4D | 0x0ACD | 0x0B4D | 0x0BCD | 0x0C4D | 0x0CCD | 0x0D4D
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScriptFamily {
    Devanagari,
    Bengali,
    Gurmukhi,
    Gujarati,
    Odia,
    Tamil,
    Telugu,
    Kannada,
    Malayalam,
    PersianArabic,
    OlChiki,
    MeeteiMayek,
    Tirhuta,
    Latin,
    Numeric,
    Punctuation,
    Other,
}

pub fn script_family_of(ch: char) -> ScriptFamily {
    let cp = ch as u32;
    match cp {
        0x0900..=0x097F => ScriptFamily::Devanagari,
        0x0980..=0x09FF => ScriptFamily::Bengali,
        0x0A00..=0x0A7F => ScriptFamily::Gurmukhi,
        0x0A80..=0x0AFF => ScriptFamily::Gujarati,
        0x0B00..=0x0B7F => ScriptFamily::Odia,
        0x0B80..=0x0BFF => ScriptFamily::Tamil,
        0x0C00..=0x0C7F => ScriptFamily::Telugu,
        0x0C80..=0x0CFF => ScriptFamily::Kannada,
        0x0D00..=0x0D7F => ScriptFamily::Malayalam,
        0x0600..=0x06FF | 0x0750..=0x077F => ScriptFamily::PersianArabic,
        0x1C50..=0x1C7F => ScriptFamily::OlChiki,
        0xABC0..=0xABFF => ScriptFamily::MeeteiMayek,
        0x11480..=0x114DF => ScriptFamily::Tirhuta,
        _ if ch.is_ascii_alphabetic() => ScriptFamily::Latin,
        _ if ch.is_ascii_digit() => ScriptFamily::Numeric,
        _ if ch.is_ascii_punctuation() || matches!(ch, '।' | '॥' | '،' | '؛' | '؟') => {
            ScriptFamily::Punctuation
        }
        _ => ScriptFamily::Other,
    }
}

pub fn lang_to_script_family(lang: &str) -> Option<ScriptFamily> {
    match lang.to_ascii_lowercase().as_str() {
        "hi" | "mr" | "ne" | "sa" | "kok" | "doi" | "brx" => Some(ScriptFamily::Devanagari),
        "bn" | "as" => Some(ScriptFamily::Bengali),
        "pa" | "pan" => Some(ScriptFamily::Gurmukhi),
        "gu" => Some(ScriptFamily::Gujarati),
        "or" | "od" => Some(ScriptFamily::Odia),
        "ta" => Some(ScriptFamily::Tamil),
        "te" => Some(ScriptFamily::Telugu),
        "kn" => Some(ScriptFamily::Kannada),
        "ml" => Some(ScriptFamily::Malayalam),
        "ur" | "ks" | "sd" => Some(ScriptFamily::PersianArabic),
        "sat" => Some(ScriptFamily::OlChiki),
        "mni" => Some(ScriptFamily::MeeteiMayek),
        "mai" => Some(ScriptFamily::Tirhuta),
        _ => None,
    }
}

pub fn detect_dominant_family(text: &str) -> ScriptFamily {
    let mut counts = std::collections::HashMap::new();
    for ch in text.chars().take(200) {
        let family = script_family_of(ch);
        if matches!(
            family,
            ScriptFamily::Other | ScriptFamily::Punctuation | ScriptFamily::Numeric
        ) {
            continue;
        }
        *counts.entry(family).or_insert(0usize) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(family, _)| family)
        .unwrap_or(ScriptFamily::Other)
}
