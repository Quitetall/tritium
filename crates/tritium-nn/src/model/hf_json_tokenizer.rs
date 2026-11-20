//! Strict Hugging Face `tokenizer.json` adapter for schema-v3 bundles.

use std::path::Path;

use serde_json::Value;
use tokenizers::Tokenizer as InnerTokenizer;

use super::chat_renderer::SourceChatTemplate;
use crate::error::NnError;
use crate::model::{ChatRenderOptions, Tokenizer};

// serde_json::Value normally keeps the last duplicate map key. Tokenizer and
// named-template selection must not depend on that ambiguous interpretation.
struct UniqueJson(Value);

impl<'de> serde::Deserialize<'de> for UniqueJson {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = UniqueJson;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("JSON with unique object keys")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<UniqueJson, M::Error> {
                let mut object = serde_json::Map::new();
                while let Some((key, value)) = map.next_entry::<String, UniqueJson>()? {
                    if object.insert(key, value.0).is_some() {
                        return Err(serde::de::Error::custom(
                            "duplicate tokenizer configuration key",
                        ));
                    }
                }
                Ok(UniqueJson(Value::Object(object)))
            }
            fn visit_seq<S: serde::de::SeqAccess<'de>>(
                self,
                mut sequence: S,
            ) -> Result<UniqueJson, S::Error> {
                let mut items = Vec::new();
                while let Some(item) = sequence.next_element::<UniqueJson>()? {
                    items.push(item.0);
                }
                Ok(UniqueJson(Value::Array(items)))
            }
            fn visit_str<E: serde::de::Error>(self, text: &str) -> Result<UniqueJson, E> {
                Ok(UniqueJson(Value::String(text.into())))
            }
            fn visit_string<E: serde::de::Error>(self, text: String) -> Result<UniqueJson, E> {
                Ok(UniqueJson(Value::String(text)))
            }
            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<UniqueJson, E> {
                Ok(UniqueJson(value.into()))
            }
            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<UniqueJson, E> {
                Ok(UniqueJson(value.into()))
            }
            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<UniqueJson, E> {
                Ok(UniqueJson(value.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<UniqueJson, E> {
                serde_json::Number::from_f64(value)
                    .map(|number| UniqueJson(Value::Number(number)))
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<UniqueJson, E> {
                Ok(UniqueJson(Value::Null))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

/// Tokenizer loaded from exact Hugging Face tokenizer assets.
pub struct HfJsonTokenizer {
    inner: InnerTokenizer,
    bos: u32,
    eos: u32,
    chat_template: Option<SourceChatTemplate>,
}

impl std::fmt::Debug for HfJsonTokenizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HfJsonTokenizer")
            .field("vocab", &self.inner.get_vocab_size(true))
            .field("bos", &self.bos)
            .field("eos", &self.eos)
            .field("source_chat_template", &self.chat_template.is_some())
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
    /// Whether an actual source template was parsed; not proof every request renders.
    /// Explicit candidate chat serving can require this instead of a legacy formatter.
    #[must_use]
    pub fn has_source_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

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
        if tokenizer_config.len() > 1024 * 1024 {
            return Err(NnError::Tokenizer(
                "tokenizer_config.json exceeds 1 MiB".into(),
            ));
        }
        let inner = InnerTokenizer::from_bytes(tokenizer_json)
            .map_err(|error| NnError::Tokenizer(format!("load tokenizer.json: {error}")))?;
        let config = serde_json::from_slice::<UniqueJson>(tokenizer_config)
            .map_err(|error| NnError::Tokenizer(format!("parse tokenizer_config.json: {error}")))?
            .0;
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
        let chat_template = SourceChatTemplate::from_config(object)?;
        Ok(Self {
            inner,
            bos,
            eos,
            chat_template,
        })
    }

    pub(crate) fn validate_model_vocab(&self, rows: usize) -> Result<(), NnError> {
        if rows == 0
            || self
                .inner
                .get_vocab(true)
                .values()
                .any(|&id| usize::try_from(id).map_or(true, |id| id >= rows))
        {
            return Err(NnError::Tokenizer(
                "tokenizer id exceeds the model embedding vocabulary".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_capture_windows(&self) -> Result<(), NnError> {
        if self.inner.get_truncation().is_some() || self.inner.get_padding().is_some() {
            return Err(NnError::Tokenizer(
                "staged capture forbids configured tokenizer truncation/padding".into(),
            ));
        }
        Ok(())
    }
}

impl Tokenizer for HfJsonTokenizer {
    fn render_chat_prompt_with_options(
        &self,
        messages: &[(&str, &str)],
        options: ChatRenderOptions,
        max_output_bytes: usize,
    ) -> Result<Option<String>, NnError> {
        if options == ChatRenderOptions::default() {
            return self.render_chat_prompt(messages, max_output_bytes);
        }
        let template = self.chat_template.as_ref().ok_or_else(|| {
            NnError::Tokenizer("explicit source chat controls require a source template".into())
        })?;
        template
            .render_with_options(messages, options, max_output_bytes)
            .map(Some)
    }

    fn render_chat_prompt(
        &self,
        messages: &[(&str, &str)],
        max_output_bytes: usize,
    ) -> Result<Option<String>, NnError> {
        self.chat_template
            .as_ref()
            .map(|template| template.render(messages, max_output_bytes))
            .transpose()
    }

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
    fn public_options_bind_source_booleans_and_refuse_missing_template() {
        let model = WordLevel::builder()
            .vocab(
                [("[UNK]".into(), 0), ("eos".into(), 1)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let bytes = InnerTokenizer::new(model).to_string(false).unwrap();
        let without =
            HfJsonTokenizer::from_bytes(bytes.as_bytes(), br#"{"eos_token":"eos"}"#).unwrap();
        let with = HfJsonTokenizer::from_bytes(bytes.as_bytes(), br#"{"eos_token":"eos","chat_template":"{% if enable_thinking is not defined %}default{% elif enable_thinking is false %}false{% else %}true{% endif %}"}"#).unwrap();
        let messages = [("user", "Hi")];
        let options = ChatRenderOptions::default();
        let absent: &dyn Tokenizer = &without;
        let present: &dyn Tokenizer = &with;
        assert_eq!(
            absent
                .render_chat_prompt_with_options(&messages, options, 64)
                .unwrap(),
            None
        );
        for (enabled, expected) in [(false, "false"), (true, "true")] {
            assert!(
                absent
                    .render_chat_prompt_with_options(&messages, options.with_thinking(enabled), 64)
                    .is_err()
            );
            assert_eq!(
                present
                    .render_chat_prompt_with_options(&messages, options.with_thinking(enabled), 64)
                    .unwrap()
                    .as_deref(),
                Some(expected)
            );
        }
        assert_eq!(
            present
                .render_chat_prompt_with_options(&messages, options, 64)
                .unwrap()
                .as_deref(),
            Some("default")
        );
    }

    #[test]
    fn duplicate_configuration_and_named_template_keys_refuse() {
        for config in [
            r#"{"eos_token":"x","chat_template":"x","chat_template":"y"}"#,
            r#"{"chat_template":{"default":"x","default":"y"}}"#,
            r#"{"chat_template":[{"name":"default","name":"other","template":"x"}]}"#,
        ] {
            assert!(serde_json::from_str::<UniqueJson>(config).is_err());
        }
    }

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
