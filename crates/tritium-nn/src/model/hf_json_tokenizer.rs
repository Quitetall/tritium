//! Strict Hugging Face `tokenizer.json` adapter for schema-v3 bundles.

use std::path::Path;

use serde_json::Value;
use tokenizers::Tokenizer as InnerTokenizer;

use crate::error::NnError;
use crate::model::Tokenizer;

/// Tokenizer loaded from exact Hugging Face tokenizer assets.
pub struct HfJsonTokenizer {
    inner: InnerTokenizer,
    bos: u32,
    eos: u32,
}

impl std::fmt::Debug for HfJsonTokenizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HfJsonTokenizer")
            .field("vocab", &self.inner.get_vocab_size(true))
            .field("bos", &self.bos)
            .field("eos", &self.eos)
            .finish()
    }
}

fn token(value: Option<&Value>, label: &str) -> Result<Option<String>, NnError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if !text.is_empty() => Ok(Some(text.clone())),
        Some(Value::Object(object)) => object
            .get("content")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
            .map(|text| Some(text.to_owned()))
            .ok_or_else(|| NnError::Tokenizer(format!("{label}.content must be non-empty"))),
        _ => Err(NnError::Tokenizer(format!(
            "{label} must be a string, token object, or null"
        ))),
    }
}

impl HfJsonTokenizer {
    /// Load tokenizer graph and special-token contract from bundle assets.
    ///
    /// Qwen does not add BOS during chat encoding. When `bos_token` is absent,
    /// [`Tokenizer::bos`] returns EOS; encoding remains unaffected.
    pub fn from_files(tokenizer_json: &Path, tokenizer_config: &Path) -> Result<Self, NnError> {
        let tokenizer_bytes = std::fs::read(tokenizer_json)
            .map_err(|error| NnError::Tokenizer(format!("read tokenizer.json: {error}")))?;
        let config_bytes = std::fs::read(tokenizer_config)
            .map_err(|error| NnError::Tokenizer(format!("read tokenizer_config.json: {error}")))?;
        Self::from_bytes(&tokenizer_bytes, &config_bytes)
    }

    /// Parse already-authenticated tokenizer assets without reopening paths.
    pub fn from_bytes(tokenizer_json: &[u8], tokenizer_config: &[u8]) -> Result<Self, NnError> {
        let inner = InnerTokenizer::from_bytes(tokenizer_json)
            .map_err(|error| NnError::Tokenizer(format!("load tokenizer.json: {error}")))?;
        let config: Value = serde_json::from_slice(tokenizer_config)
            .map_err(|error| NnError::Tokenizer(format!("parse tokenizer_config.json: {error}")))?;
        let object = config
            .as_object()
            .ok_or_else(|| NnError::Tokenizer("tokenizer_config.json must be an object".into()))?;
        let eos_name = token(object.get("eos_token"), "eos_token")?
            .ok_or_else(|| NnError::Tokenizer("eos_token is required".into()))?;
        let eos = inner.token_to_id(&eos_name).ok_or_else(|| {
            NnError::Tokenizer("eos_token is absent from tokenizer vocabulary".into())
        })?;
        let bos = match token(object.get("bos_token"), "bos_token")? {
            Some(name) => inner.token_to_id(&name).ok_or_else(|| {
                NnError::Tokenizer("bos_token is absent from tokenizer vocabulary".into())
            })?,
            None => eos,
        };
        Ok(Self { inner, bos, eos })
    }
}

impl Tokenizer for HfJsonTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, NnError> {
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| NnError::Tokenizer(format!("encode: {error}")))
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, NnError> {
        self.inner
            .decode(tokens, true)
            .map_err(|error| NnError::Tokenizer(format!("decode: {error}")))
    }

    fn bos(&self) -> u32 {
        self.bos
    }

    fn eos(&self) -> u32 {
        self.eos
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    use super::*;

    #[test]
    fn loads_special_tokens_and_does_not_inject_bos() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("tritium-hf-tokenizer-{unique}"));
        fs::create_dir(&root).unwrap();
        let vocab = [
            ("[UNK]".to_owned(), 0),
            ("hello".to_owned(), 1),
            ("<|im_start|>".to_owned(), 2),
            ("<|im_end|>".to_owned(), 3),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let mut inner = InnerTokenizer::new(model);
        inner.with_pre_tokenizer(Some(Whitespace));
        let tokenizer_path = root.join("tokenizer.json");
        inner.save(&tokenizer_path, false).unwrap();
        let config_path = root.join("tokenizer_config.json");
        fs::write(
            &config_path,
            r#"{"bos_token":"<|im_start|>","eos_token":{"content":"<|im_end|>"}}"#,
        )
        .unwrap();
        let tokenizer = HfJsonTokenizer::from_files(&tokenizer_path, &config_path).unwrap();
        assert_eq!(tokenizer.bos(), 2);
        assert_eq!(tokenizer.eos(), 3);
        assert_eq!(tokenizer.encode("hello").unwrap(), vec![1]);
        fs::remove_dir_all(root).unwrap();
    }
}
