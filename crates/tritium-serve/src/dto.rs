//! OpenAI-compatible wire DTOs (serde only — always compiled, no async deps).
//!
//! Faithful to the `/v1/chat/completions` schema so a LAMU `local-llm`
//! OpenAI-compatible backend (or any OpenAI client) can target the server
//! unchanged. Unknown request fields are tolerated (no `deny_unknown_fields`),
//! matching OpenAI's permissive behavior.

use serde::{Deserialize, Serialize};

/// A chat message (`role` in {system, user, assistant}; `content` is plain text).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    /// Message role.
    pub role: String,
    /// Message text.
    pub content: String,
}

/// `stop`: either a single string or up to four strings (OpenAI allows both).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum StopField {
    /// A single stop sequence.
    One(String),
    /// Up to four stop sequences.
    Many(Vec<String>),
}

impl StopField {
    /// Flatten to a vec of stop sequences.
    #[must_use]
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopField::One(s) => vec![s],
            StopField::Many(v) => v,
        }
    }
}

fn default_temperature() -> f32 {
    1.0
}

/// Tritium's typed source-template inputs, not arbitrary template keywords.
/// A source template can ignore these inputs; they do not enforce output policy.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ChatTemplateKwargs {
    /// Exact source `enable_thinking` input; omission leaves it undefined.
    /// A present null or nonboolean value is rejected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
}

impl<'de> Deserialize<'de> for ChatTemplateKwargs {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ObjectOnly;
        impl<'de> serde::de::Visitor<'de> for ObjectOnly {
            type Value = ChatTemplateKwargs;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an object with an optional boolean enable_thinking")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<Self::Value, M::Error> {
                let mut enable_thinking = None;
                while let Some(key) = map.next_key::<String>()? {
                    if key != "enable_thinking" {
                        return Err(serde::de::Error::unknown_field(&key, &["enable_thinking"]));
                    }
                    if enable_thinking.is_some() {
                        return Err(serde::de::Error::duplicate_field("enable_thinking"));
                    }
                    enable_thinking = Some(map.next_value::<bool>()?);
                }
                Ok(ChatTemplateKwargs { enable_thinking })
            }
        }
        // Derived structs also accept sequences. The extension is object-only.
        deserializer.deserialize_map(ObjectOnly)
    }
}

/// `POST /v1/chat/completions` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    /// Requested model id (validated against the served model).
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Optional Tritium source-template extension. Omission, null and an empty
    /// object preserve source defaults. Explicit controls require source support.
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// Stream the response as SSE deltas.
    #[serde(default)]
    pub stream: bool,
    /// OpenAI `stream_options` (only `include_usage` is meaningful here).
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Return per-token logprobs.
    #[serde(default)]
    pub logprobs: bool,
    /// Number of top alternatives per token (0..=20; needs `logprobs: true`).
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    /// Max new tokens to generate.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Sampling temperature (`0.0` = greedy).
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Nucleus sampling cutoff.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// PRNG seed for reproducible sampling.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Stop sequence(s).
    #[serde(default)]
    pub stop: Option<StopField>,
}

/// Token accounting returned with a completion.
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    /// Prompt token count.
    pub prompt_tokens: usize,
    /// Generated token count.
    pub completion_tokens: usize,
    /// Sum of the two.
    pub total_tokens: usize,
}

/// One non-streaming choice.
#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    /// Choice index (always 0 — we return a single completion).
    pub index: u32,
    /// The assistant message.
    pub message: ChatMessage,
    /// Why generation stopped (`"stop"` / `"length"`).
    pub finish_reason: String,
    /// Per-token logprobs, present only when the request asked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// OpenAI `choices[].logprobs` object.
#[derive(Debug, Clone, Serialize)]
pub struct ChoiceLogprobs {
    /// One entry per emitted completion token.
    pub content: Vec<TokenLogprob>,
}

/// One token's logprob record.
#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprob {
    /// The token's text.
    pub token: String,
    /// Its log-probability.
    pub logprob: f32,
    /// UTF-8 bytes of `token` (OpenAI convention).
    pub bytes: Vec<u8>,
    /// The top-k alternatives (may include the sampled token itself).
    pub top_logprobs: Vec<TopLogprob>,
}

/// One alternative in `top_logprobs`.
#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    /// The alternative token's text.
    pub token: String,
    /// Its log-probability.
    pub logprob: f32,
    /// UTF-8 bytes of `token`.
    pub bytes: Vec<u8>,
}

/// `object: "chat.completion"` — the non-streaming response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletion {
    /// Completion id (`chatcmpl-...`).
    pub id: String,
    /// Always `"chat.completion"`.
    pub object: &'static str,
    /// Unix creation time (seconds).
    pub created: u64,
    /// Served model id.
    pub model: String,
    /// The (single) choice.
    pub choices: Vec<Choice>,
    /// Token usage.
    pub usage: Usage,
}

/// A streaming delta (role on the first chunk, then content).
#[derive(Debug, Clone, Default, Serialize)]
pub struct Delta {
    /// Set only on the first chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Incremental content (absent on the role-first and terminal chunks).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// One streaming choice (a delta + an optional terminal finish_reason).
#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    /// Choice index (always 0).
    pub index: u32,
    /// The incremental delta.
    pub delta: Delta,
    /// `null` until the terminal chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// This chunk's token logprobs, when the request asked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// `object: "chat.completion.chunk"` — one SSE `data:` payload.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChunk {
    /// Completion id (stable across all chunks of one response).
    pub id: String,
    /// Always `"chat.completion.chunk"`.
    pub object: &'static str,
    /// Unix creation time (seconds), stable across chunks.
    pub created: u64,
    /// Served model id.
    pub model: String,
    /// The (single) choice delta. Empty on the final usage chunk when the
    /// client asked for `stream_options.include_usage` (OpenAI convention).
    pub choices: Vec<ChunkChoice>,
    /// Token accounting, present only on the final chunk when
    /// `stream_options.include_usage` was requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// OpenAI `stream_options` request object.
#[derive(Debug, Clone, Copy, Default, serde::Deserialize)]
pub struct StreamOptions {
    /// Emit a final pre-`[DONE]` chunk with empty `choices` and `usage`.
    #[serde(default)]
    pub include_usage: bool,
}

/// `GET /v1/models` entry.
#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    /// Model id.
    pub id: String,
    /// Always `"model"`.
    pub object: &'static str,
    /// Unix creation time (seconds).
    pub created: u64,
    /// Owner (`"tritium"`).
    pub owned_by: &'static str,
}

/// `GET /v1/models` response.
#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    /// Always `"list"`.
    pub object: &'static str,
    /// The served models (one).
    pub data: Vec<ModelEntry>,
}

/// The body of an OpenAI error response.
#[derive(Debug, Clone, Serialize)]
pub struct ApiErrorBody {
    /// Human-readable message.
    pub message: String,
    /// Error class (`invalid_request_error`, `model_not_found`, ...).
    #[serde(rename = "type")]
    pub kind: String,
    /// Offending request field, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    /// Machine-readable code, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// `{ "error": { ... } }` envelope.
#[derive(Debug, Clone, Serialize)]
pub struct ApiError {
    /// The error body.
    pub error: ApiErrorBody,
}

impl ApiError {
    /// Build an error envelope.
    #[must_use]
    pub fn new(kind: &str, message: impl Into<String>, param: Option<&str>) -> Self {
        Self {
            error: ApiErrorBody {
                message: message.into(),
                kind: kind.to_owned(),
                param: param.map(str::to_owned),
                code: None,
            },
        }
    }
}

#[cfg(test)]
mod chat_controls_tests {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn omitted_null_empty_and_boolean_controls_preserve_typed_meaning() {
        for value in [
            None,
            Some(Value::Null),
            Some(json!({})),
            Some(json!({"enable_thinking":true})),
            Some(json!({"enable_thinking":false})),
        ] {
            let mut body = json!({"model":"m","messages":[],"opaque_vendor_field":42});
            if let Some(value) = value {
                body["chat_template_kwargs"] = value;
            }
            let request: ChatRequest = serde_json::from_value(body.clone()).unwrap();
            assert_eq!(
                request.chat_template_kwargs.and_then(|v| v.enable_thinking),
                body["chat_template_kwargs"]["enable_thinking"].as_bool()
            );
        }
        assert_eq!(
            serde_json::to_value(ChatTemplateKwargs::default()).unwrap(),
            json!({})
        );
    }

    #[test]
    fn nested_controls_reject_unknown_duplicate_null_and_nonboolean_values() {
        for value in [
            json!(false),
            json!([]),
            json!({"enable_thinking":null}),
            json!({"enable_thinking":"false"}),
            json!({"enable_thinking":0}),
            json!({"enable_thinking":[]}),
            json!({"messages":[]}),
            json!({"enable_thinking":false,"tools":[]}),
        ] {
            assert!(
                serde_json::from_value::<ChatRequest>(
                    json!({"model":"m","messages":[],"chat_template_kwargs":value.clone()})
                )
                .is_err(),
                "accepted malformed controls: {value}"
            );
        }
        for body in [
            r#"{"model":"m","messages":[],"chat_template_kwargs":{"enable_thinking":false,"enable_thinking":true}}"#,
            r#"{"model":"m","messages":[],"chat_template_kwargs":null,"chat_template_kwargs":{}}"#,
        ] {
            assert!(serde_json::from_str::<ChatRequest>(body).is_err());
        }
    }
}
