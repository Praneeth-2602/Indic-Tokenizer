use crate::bpe::BpeModel;
use crate::codemix::detect_script_spans;
use crate::normalizer::IndicNormalizer;
use crate::pretokenizer::IndicPretokenizer;
use crate::utils::ScriptFamily;
use hashbrown::HashMap;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const UNK_TOKEN: &str = "<unk>";
pub const PAD_TOKEN: &str = "<pad>";
pub const SPACE_TOKEN: &str = " ";

#[derive(Debug)]
pub enum TokenizerError {
    Io(String),
    Json(String),
    InvalidVocabulary(String),
    InvalidMerges(String),
}

impl From<std::io::Error> for TokenizerError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value.to_string())
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodeOutput {
    #[pyo3(get)]
    pub ids: Vec<u32>,
    #[pyo3(get)]
    pub tokens: Vec<String>,
    #[pyo3(get)]
    pub lang: Option<String>,
    #[pyo3(get)]
    pub offsets: Vec<(usize, usize)>,
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FertilityReport {
    #[pyo3(get)]
    pub fertility: f64,
    #[pyo3(get)]
    pub total_tokens: usize,
    #[pyo3(get)]
    pub total_words: usize,
    #[pyo3(get)]
    pub total_sentences: usize,
}

#[derive(Clone, Debug)]
pub struct IndicTokenizer {
    normalizer: IndicNormalizer,
    pretokenizer: IndicPretokenizer,
    bpe: BpeModel,
}

impl IndicTokenizer {
    pub fn new(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
    ) -> Result<Self, TokenizerError> {
        Ok(Self {
            normalizer: IndicNormalizer::default(),
            pretokenizer: IndicPretokenizer::default(),
            bpe: BpeModel::new(vocab, merges)?,
        })
    }

    pub fn default_indic() -> Self {
        Self::new(default_vocab(), Vec::new()).expect("built-in vocabulary is valid")
    }

    pub fn from_files<P: AsRef<Path>>(
        vocab_path: P,
        merges_path: Option<P>,
    ) -> Result<Self, TokenizerError> {
        let vocab_text = fs::read_to_string(vocab_path.as_ref())?;
        let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_text)?;
        let merges = match merges_path {
            Some(path) => BpeModel::load_merges(path)?,
            None => Vec::new(),
        };
        Self::new(vocab, merges)
    }

    pub fn from_pretrained<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let dir = path.as_ref();
        Self::from_files(dir.join("vocab.json"), Some(dir.join("merges.txt")))
    }

    pub fn save_pretrained<P: AsRef<Path>>(&self, path: P) -> Result<(), TokenizerError> {
        let dir: PathBuf = path.as_ref().into();
        fs::create_dir_all(&dir)?;
        let vocab_json = serde_json::to_string_pretty(self.bpe.vocab())?;
        fs::write(dir.join("vocab.json"), vocab_json)?;
        fs::write(dir.join("merges.txt"), self.bpe.merges_text())?;
        Ok(())
    }

    pub fn normalize(&self, text: &str) -> String {
        self.normalize_with_lang(text, None)
    }

    pub fn normalize_with_lang(&self, text: &str, lang: Option<&str>) -> String {
        self.normalizer.normalize_with_lang(text, lang)
    }

    pub fn pre_tokenize(&self, text: &str) -> Vec<String> {
        self.pre_tokenize_with_lang(text, None)
    }

    pub fn pre_tokenize_with_lang(&self, text: &str, lang: Option<&str>) -> Vec<String> {
        let normalized = self.normalize_with_lang(text, lang);
        self.pretokenizer.pre_tokenize_with_lang(&normalized, lang)
    }

    pub fn encode(&self, text: &str) -> EncodeOutput {
        self.encode_with_lang(text, None)
    }

    pub fn encode_with_lang(&self, text: &str, lang: Option<&str>) -> EncodeOutput {
        self.encode_with_options(text, lang, false)
    }

    pub fn encode_with_options(
        &self,
        text: &str,
        lang: Option<&str>,
        code_mix: bool,
    ) -> EncodeOutput {
        if code_mix {
            return self.encode_code_mixed(text, lang);
        }
        let normalized = self.normalize_with_lang(text, lang);
        let pieces = self.pretokenizer.pre_tokenize_with_lang(&normalized, lang);
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();
        let mut search_start = 0usize;

        for piece in pieces {
            let piece_start = normalized[search_start..]
                .find(&piece)
                .map(|offset| search_start + offset)
                .unwrap_or(search_start);
            let encoded = self.bpe.encode_piece(&piece);
            ids.extend(encoded.ids);
            tokens.extend(encoded.tokens);
            offsets.extend(
                encoded
                    .offsets
                    .into_iter()
                    .map(|(start, end)| (piece_start + start, piece_start + end)),
            );
            search_start = piece_start + piece.len();
        }

        EncodeOutput {
            ids,
            tokens,
            lang: lang.map(ToOwned::to_owned),
            offsets,
        }
    }

    fn encode_code_mixed(&self, text: &str, lang: Option<&str>) -> EncodeOutput {
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();

        for span in detect_script_spans(text) {
            let span_lang = script_family_to_lang(span.script).or(lang);
            let encoded = self.encode_with_options(&span.text, span_lang, false);
            ids.extend(encoded.ids);
            tokens.extend(encoded.tokens);
            offsets.extend(
                encoded
                    .offsets
                    .into_iter()
                    .map(|(start, end)| (span.start + start, span.start + end)),
            );
        }

        EncodeOutput {
            ids,
            tokens,
            lang: lang.map(ToOwned::to_owned),
            offsets,
        }
    }

    pub fn encode_batch(&self, texts: &[String]) -> Vec<EncodeOutput> {
        texts.par_iter().map(|text| self.encode(text)).collect()
    }

    pub fn encode_batch_with_lang(
        &self,
        texts: &[String],
        lang: Option<&str>,
    ) -> Vec<EncodeOutput> {
        self.encode_batch_with_options(texts, lang, false, None)
    }

    pub fn encode_batch_with_options(
        &self,
        texts: &[String],
        lang: Option<&str>,
        code_mix: bool,
        num_threads: Option<usize>,
    ) -> Vec<EncodeOutput> {
        if let Some(num_threads) = num_threads {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("failed to build rayon thread pool");
            return pool.install(|| {
                texts
                    .par_iter()
                    .map(|text| self.encode_with_options(text, lang, code_mix))
                    .collect()
            });
        }
        texts
            .par_iter()
            .map(|text| self.encode_with_options(text, lang, code_mix))
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.bpe.decode(ids)
    }

    pub fn vocab_size(&self) -> usize {
        self.bpe.vocab().len()
    }

    pub fn vocab(&self) -> &HashMap<String, u32> {
        self.bpe.vocab()
    }

    pub fn fertility(&self, texts: &[String], lang: Option<&str>) -> FertilityReport {
        let mut total_tokens = 0usize;
        let mut total_words = 0usize;

        for text in texts {
            total_words += text.split_whitespace().count();
            total_tokens += self.encode_with_lang(text, lang).ids.len();
        }

        FertilityReport {
            fertility: total_tokens as f64 / total_words.max(1) as f64,
            total_tokens,
            total_words,
            total_sentences: texts.len(),
        }
    }
}

fn default_vocab() -> HashMap<String, u32> {
    let mut tokens = vec![PAD_TOKEN, "<s>", "</s>", "<mask>", UNK_TOKEN];
    for idx in 0..=255u16 {
        tokens.push(Box::leak(format!("<0x{idx:02X}>").into_boxed_str()));
    }
    tokens.extend([
        SPACE_TOKEN,
        "न",
        "म",
        "स्",
        "ते",
        "नमस्ते",
        "भा",
        "र",
        "त",
        "भारत",
        "हिं",
        "दी",
        "हिंदी",
        "లు",
        "గు",
        "తెలుగు",
        "నం",
        "స్కా",
        "రం",
        "నమస్కారం",
        "ఇం",
        "డి",
        "యా",
        "ఇండియా",
        "hello",
        "world",
        "India",
        "!",
        "?",
        ",",
        ".",
        "।",
        "-",
        ":",
        ";",
        "(",
        ")",
    ]);
    debug_assert_eq!(
        tokens.len(),
        tokens
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        "built-in vocabulary contains duplicate tokens"
    );
    tokens
        .iter()
        .enumerate()
        .map(|(idx, token)| ((*token).to_string(), idx as u32))
        .collect()
}

fn script_family_to_lang(script: ScriptFamily) -> Option<&'static str> {
    match script {
        ScriptFamily::Devanagari => Some("hi"),
        ScriptFamily::Bengali => Some("bn"),
        ScriptFamily::Gurmukhi => Some("pa"),
        ScriptFamily::Gujarati => Some("gu"),
        ScriptFamily::Odia => Some("or"),
        ScriptFamily::Tamil => Some("ta"),
        ScriptFamily::Telugu => Some("te"),
        ScriptFamily::Kannada => Some("kn"),
        ScriptFamily::Malayalam => Some("ml"),
        ScriptFamily::PersianArabic => Some("ur"),
        ScriptFamily::OlChiki => Some("sat"),
        ScriptFamily::MeeteiMayek => Some("mni"),
        ScriptFamily::Tirhuta => Some("mai"),
        _ => None,
    }
}
