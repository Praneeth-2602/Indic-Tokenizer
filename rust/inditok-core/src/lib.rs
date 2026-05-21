pub mod bpe;
pub mod codemix;
pub mod normalizer;
pub mod pretokenizer;
pub mod tokenizer;
pub mod utils;

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use tokenizer::{EncodeOutput, FertilityReport, IndicTokenizer};

fn map_error(err: tokenizer::TokenizerError) -> PyErr {
    match err {
        tokenizer::TokenizerError::Io(message) => PyIOError::new_err(message),
        tokenizer::TokenizerError::Json(message)
        | tokenizer::TokenizerError::InvalidVocabulary(message)
        | tokenizer::TokenizerError::InvalidMerges(message) => PyValueError::new_err(message),
    }
}

#[pyclass(name = "IndicTokenizer")]
struct PyIndicTokenizer {
    inner: IndicTokenizer,
}

#[pymethods]
impl PyIndicTokenizer {
    #[new]
    #[pyo3(signature = (vocab_path=None, merges_path=None))]
    fn new(vocab_path: Option<String>, merges_path: Option<String>) -> PyResult<Self> {
        let inner = match (vocab_path, merges_path) {
            (Some(vocab), Some(merges)) => {
                IndicTokenizer::from_files(vocab, Some(merges)).map_err(map_error)?
            }
            (Some(vocab), None) => IndicTokenizer::from_files(vocab, None).map_err(map_error)?,
            (None, None) => IndicTokenizer::default_indic(),
            (None, Some(_)) => {
                return Err(PyValueError::new_err(
                    "merges_path requires vocab_path; pass both files or neither",
                ))
            }
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (path))]
    fn from_pretrained(path: String) -> PyResult<Self> {
        Ok(Self {
            inner: IndicTokenizer::from_pretrained(path).map_err(map_error)?,
        })
    }

    #[pyo3(signature = (text, lang=None))]
    fn encode(&self, text: &str, lang: Option<&str>) -> Vec<u32> {
        self.inner.encode_with_lang(text, lang).ids
    }

    #[pyo3(signature = (text, lang=None))]
    fn encode_with_tokens(&self, text: &str, lang: Option<&str>) -> EncodeOutput {
        self.inner.encode_with_lang(text, lang)
    }

    fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }

    #[pyo3(signature = (texts, lang=None))]
    fn encode_batch(&self, texts: Vec<String>, lang: Option<&str>) -> Vec<Vec<u32>> {
        self.inner
            .encode_batch_with_lang(&texts, lang)
            .into_iter()
            .map(|item| item.ids)
            .collect()
    }

    #[pyo3(signature = (text, lang=None))]
    fn normalize(&self, text: &str, lang: Option<&str>) -> String {
        self.inner.normalize_with_lang(text, lang)
    }

    #[pyo3(signature = (text, lang=None))]
    fn pre_tokenize(&self, text: &str, lang: Option<&str>) -> Vec<String> {
        self.inner.pre_tokenize_with_lang(text, lang)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    #[pyo3(signature = (path))]
    fn save_pretrained(&self, path: String) -> PyResult<()> {
        self.inner.save_pretrained(path).map_err(map_error)
    }

    #[pyo3(signature = (texts, lang=None))]
    fn fertility(&self, texts: Vec<String>, lang: Option<&str>) -> FertilityReport {
        self.inner.fertility(&texts, lang)
    }
}

#[pymodule]
fn _inditok(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIndicTokenizer>()?;
    m.add_class::<EncodeOutput>()?;
    m.add_class::<FertilityReport>()?;
    Ok(())
}
