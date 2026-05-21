use crate::codemix::detect_script_spans;

pub fn pre_tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for span in detect_script_spans(text) {
        split_span(&span.text, &mut out);
    }
    out
}

fn split_span(text: &str, out: &mut Vec<String>) {
    let mut current = String::new();
    for grapheme in unicode_segmentation::UnicodeSegmentation::graphemes(text, true) {
        if grapheme == " " {
            flush(&mut current, out);
            out.push(" ".to_string());
        } else if super::is_split_boundary(grapheme) {
            flush(&mut current, out);
            out.push(grapheme.to_string());
        } else {
            current.push_str(grapheme);
        }
    }
    flush(&mut current, out);
}

fn flush(current: &mut String, out: &mut Vec<String>) {
    if !current.is_empty() {
        out.push(std::mem::take(current));
    }
}
