pub fn pre_tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();

    for grapheme in unicode_segmentation::UnicodeSegmentation::graphemes(text, true) {
        if grapheme == " " {
            flush(&mut current, &mut out);
            out.push(" ".to_string());
        } else if is_arabic_split_boundary(grapheme) {
            flush(&mut current, &mut out);
            out.push(grapheme.to_string());
        } else {
            current.push_str(grapheme);
        }
    }

    flush(&mut current, &mut out);
    out
}

fn is_arabic_split_boundary(grapheme: &str) -> bool {
    let Some(ch) = grapheme.chars().next() else {
        return false;
    };
    matches!(
        ch,
        '،' | '؛' | '؟' | '!' | '?' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}'
    )
}

fn flush(current: &mut String, out: &mut Vec<String>) {
    if !current.is_empty() {
        out.push(std::mem::take(current));
    }
}
