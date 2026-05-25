use crate::tokenizer::{TokenizerError, PAD_TOKEN, SPACE_TOKEN, UNK_TOKEN};
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PieceEncoding {
    pub ids: Vec<u32>,
    pub tokens: Vec<String>,
    pub offsets: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct BpeModel {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    ranks: HashMap<(String, String), usize>,
    merges: Vec<(String, String)>,
    unk_id: u32,
}

impl BpeModel {
    pub fn new(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
    ) -> Result<Self, TokenizerError> {
        let unk_id = *vocab.get(UNK_TOKEN).ok_or_else(|| {
            TokenizerError::InvalidVocabulary("vocab.json must contain <unk>".to_string())
        })?;

        let id_to_token = vocab
            .iter()
            .map(|(token, id)| (*id, token.clone()))
            .collect::<HashMap<_, _>>();

        if id_to_token.len() != vocab.len() {
            return Err(TokenizerError::InvalidVocabulary(
                "vocab.json contains duplicate token ids".to_string(),
            ));
        }

        let ranks = merges
            .iter()
            .enumerate()
            .map(|(rank, pair)| (pair.clone(), rank))
            .collect();

        Ok(Self {
            vocab,
            id_to_token,
            ranks,
            merges,
            unk_id,
        })
    }

    pub fn load_merges<P: AsRef<Path>>(path: P) -> Result<Vec<(String, String)>, TokenizerError> {
        let text = fs::read_to_string(path)?;
        let mut merges = Vec::new();

        for (line_no, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parts = trimmed.split_whitespace().collect::<Vec<_>>();
            if parts.len() != 2 {
                return Err(TokenizerError::InvalidMerges(format!(
                    "invalid merge at line {}: expected two tokens",
                    line_no + 1
                )));
            }
            merges.push((parts[0].to_string(), parts[1].to_string()));
        }

        Ok(merges)
    }

    pub fn encode_piece(&self, piece: &str) -> PieceEncoding {
        if let Some(id) = self.vocab.get(piece) {
            return PieceEncoding {
                ids: vec![*id],
                tokens: vec![piece.to_string()],
                offsets: vec![(0, piece.len())],
            };
        }

        let mut symbols = initial_symbols(piece);
        self.apply_merges(&mut symbols);

        let mut ids = Vec::with_capacity(symbols.len());
        let mut tokens = Vec::with_capacity(symbols.len());
        let mut offsets = Vec::with_capacity(symbols.len());
        for symbol in symbols {
            if let Some(id) = self.vocab.get(&symbol.text) {
                ids.push(*id);
                offsets.push(symbol.offset);
                tokens.push(symbol.text);
                continue;
            }
            for byte in symbol.text.as_bytes() {
                let byte_token = format!("<0x{byte:02X}>");
                let id = self.vocab.get(&byte_token).copied().unwrap_or(self.unk_id);
                ids.push(id);
                offsets.push(symbol.offset);
                tokens.push(if id == self.unk_id {
                    UNK_TOKEN.to_string()
                } else {
                    byte_token
                });
            }
        }

        PieceEncoding {
            ids,
            tokens,
            offsets,
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        let mut result = String::new();

        for id in ids {
            let Some(token) = self.id_to_token.get(id) else {
                continue;
            };

            if token == UNK_TOKEN || token == PAD_TOKEN {
                flush_bytes(&mut bytes, &mut result);
                continue;
            }

            if let Some(byte) = parse_byte_token(token) {
                bytes.push(byte);
                continue;
            }

            flush_bytes(&mut bytes, &mut result);
            if token == SPACE_TOKEN {
                result.push(' ');
            } else {
                result.push_str(token);
            }
        }

        flush_bytes(&mut bytes, &mut result);
        result
    }

    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    pub fn merges_text(&self) -> String {
        self.merges
            .iter()
            .map(|(left, right)| format!("{left} {right}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn apply_merges(&self, symbols: &mut Vec<Symbol>) {
        loop {
            let Some((idx, _rank)) = best_pair(symbols, &self.ranks) else {
                break;
            };
            let merged = Symbol {
                text: format!("{}{}", symbols[idx].text, symbols[idx + 1].text),
                offset: (symbols[idx].offset.0, symbols[idx + 1].offset.1),
            };
            symbols.splice(idx..=idx + 1, [merged]);
        }
    }
}

fn parse_byte_token(token: &str) -> Option<u8> {
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        u8::from_str_radix(&token[3..5], 16).ok()
    } else {
        None
    }
}

fn flush_bytes(bytes: &mut Vec<u8>, result: &mut String) {
    if !bytes.is_empty() {
        result.push_str(&String::from_utf8_lossy(bytes));
        bytes.clear();
    }
}

#[derive(Clone, Debug)]
struct Symbol {
    text: String,
    offset: (usize, usize),
}

fn initial_symbols(piece: &str) -> Vec<Symbol> {
    if piece == SPACE_TOKEN {
        vec![Symbol {
            text: SPACE_TOKEN.to_string(),
            offset: (0, SPACE_TOKEN.len()),
        }]
    } else {
        piece
            .grapheme_indices(true)
            .map(|(start, grapheme)| Symbol {
                text: grapheme.to_string(),
                offset: (start, start + grapheme.len()),
            })
            .collect()
    }
}

fn best_pair(
    symbols: &[Symbol],
    ranks: &HashMap<(String, String), usize>,
) -> Option<(usize, usize)> {
    symbols
        .windows(2)
        .enumerate()
        .filter_map(|(idx, pair)| {
            ranks
                .get(&(pair[0].text.clone(), pair[1].text.clone()))
                .map(|rank| (idx, *rank))
        })
        .min_by_key(|(_, rank)| *rank)
}
