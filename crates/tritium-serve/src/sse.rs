//! SSE helpers: incremental detokenization, stop-string matching, chunk builders.
//! Pure logic (serde_json only); the axum `Event` wrapping lives in `router.rs`.

use std::sync::Arc;

use tritium_nn::Tokenizer;

use crate::dto::{ChatChunk, ChunkChoice, Delta};
use crate::generator::FinishReason;

/// Largest byte index `<= i` that is a UTF-8 char boundary in `s`.
fn floor_char_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Turns a stream of token IDs into incremental text by decoding the whole token
/// buffer each step and emitting the new byte-suffix (snapped to a char boundary,
/// so a multi-byte codepoint split across tokens is never emitted partial). This
/// guarantees `concat(all suffixes) == decode(all tokens)` — the invariant the
/// stream-equals-buffered contract test pins. Special tokens (EOS) are dropped.
pub(crate) struct IncrementalDetok {
    tok: Arc<dyn Tokenizer + Send + Sync>,
    eos: u32,
    tokens: Vec<u32>,
    emitted_bytes: usize,
}

impl IncrementalDetok {
    /// New detokenizer over `tok`, dropping `eos` from the surfaced content.
    #[must_use]
    pub(crate) fn new(tok: Arc<dyn Tokenizer + Send + Sync>) -> Self {
        let eos = tok.eos();
        Self {
            tok,
            eos,
            tokens: Vec::new(),
            emitted_bytes: 0,
        }
    }

    /// Append `token`, returning the newly-decoded text (empty for special tokens).
    pub(crate) fn push(&mut self, token: u32) -> String {
        if token == self.eos {
            return String::new();
        }
        self.tokens.push(token);
        let full = self.tok.decode(&self.tokens).unwrap_or_default();
        let start = floor_char_boundary(&full, self.emitted_bytes);
        let new = full[start..].to_string();
        self.emitted_bytes = full.len();
        new
    }
}

/// Detects OpenAI `stop` sequences in the decoded stream. Buffers a tail of up to
/// `max_len - 1` chars so a stop sequence split across token boundaries is still
/// caught, and never emits text that could be the prefix of a stop sequence.
pub(crate) struct StopMatcher {
    stops: Vec<String>,
    max_len: usize,
    buffer: String,
}

impl StopMatcher {
    /// New matcher for the given stop sequences (empty = never stops, emit eagerly).
    #[must_use]
    pub(crate) fn new(stops: Vec<String>) -> Self {
        let max_len = stops.iter().map(String::len).max().unwrap_or(0);
        Self {
            stops,
            max_len,
            buffer: String::new(),
        }
    }

    /// Feed newly-decoded `text`. Returns `(safe_to_emit, hit)`: text safe to send
    /// now, and whether a stop sequence completed (content truncated at the stop).
    pub(crate) fn feed(&mut self, text: &str) -> (String, bool) {
        self.buffer.push_str(text);
        // A completed stop sequence: emit everything before it, then stop.
        if let Some(pos) = self
            .stops
            .iter()
            .filter_map(|s| self.buffer.find(s.as_str()))
            .min()
        {
            let emit = self.buffer[..pos].to_string();
            self.buffer.clear();
            return (emit, true);
        }
        // No stop yet: emit all but the last (max_len - 1) bytes, which might be the
        // start of a stop sequence still being formed.
        let hold = self.max_len.saturating_sub(1);
        if self.buffer.len() > hold {
            let cut = floor_char_boundary(&self.buffer, self.buffer.len() - hold);
            let emit = self.buffer[..cut].to_string();
            self.buffer = self.buffer[cut..].to_string();
            (emit, false)
        } else {
            (String::new(), false)
        }
    }

    /// Flush any held tail (no stop sequence completed).
    pub(crate) fn flush(&mut self) -> String {
        std::mem::take(&mut self.buffer)
    }
}

/// The role-first delta chunk (`delta: {role: "assistant"}`, `finish_reason: null`).
#[must_use]
pub(crate) fn role_chunk(id: &str, created: u64, model: &str) -> ChatChunk {
    ChatChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk",
        created,
        model: model.to_owned(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: Some("assistant".to_owned()),
                content: None,
            },
            finish_reason: None,
        }],
    }
}

/// A content delta chunk.
#[must_use]
pub(crate) fn content_chunk(id: &str, created: u64, model: &str, content: &str) -> ChatChunk {
    ChatChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk",
        created,
        model: model.to_owned(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: Some(content.to_owned()),
            },
            finish_reason: None,
        }],
    }
}

/// The terminal chunk (empty delta + non-null finish_reason).
#[must_use]
pub(crate) fn terminal_chunk(
    id: &str,
    created: u64,
    model: &str,
    finish: FinishReason,
) -> ChatChunk {
    ChatChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk",
        created,
        model: model.to_owned(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta::default(),
            finish_reason: Some(finish.as_str().to_owned()),
        }],
    }
}

/// A terminal chunk signaling a backend error mid-stream: `finish_reason "error"`,
/// distinct from a clean `"stop"`, so a streaming client can detect the failure
/// (the non-stream path returns HTTP 500 for the same condition).
#[must_use]
pub(crate) fn error_chunk(id: &str, created: u64, model: &str) -> ChatChunk {
    ChatChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk",
        created,
        model: model.to_owned(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta::default(),
            finish_reason: Some("error".to_owned()),
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer_passthrough::IdPassthroughTokenizer;

    #[test]
    fn incremental_detok_concat_equals_full() {
        let tok = Arc::new(IdPassthroughTokenizer::default());
        let mut d = IncrementalDetok::new(tok.clone());
        let ids = [10u32, 11, 12, 13];
        let mut concat = String::new();
        for &t in &ids {
            concat.push_str(&d.push(t));
        }
        assert_eq!(concat, tok.decode(&ids).unwrap());
    }

    #[test]
    fn detok_drops_eos() {
        let tok = Arc::new(IdPassthroughTokenizer::default());
        let eos = tok.eos();
        let mut d = IncrementalDetok::new(tok);
        assert_eq!(d.push(eos), "");
    }

    #[test]
    fn stop_matcher_no_stops_emits_eagerly() {
        let mut m = StopMatcher::new(vec![]);
        let (e, hit) = m.feed("hello");
        assert_eq!(e, "hello");
        assert!(!hit);
    }

    #[test]
    fn stop_matcher_catches_split_sequence_and_truncates() {
        let mut m = StopMatcher::new(vec!["STOP".to_owned()]);
        // max_len=4 → up to 3 trailing chars are held as a possible stop prefix.
        // "abST": emit "a", hold "bST". Then "OPxy" completes "STOP" → emit "b", hit.
        // concat("a","b") == "ab" == the content before "STOP" in "abSTOPxy".
        let (e1, hit1) = m.feed("abST");
        assert_eq!(e1, "a");
        assert!(!hit1);
        let (e2, hit2) = m.feed("OPxy");
        assert_eq!(e2, "b");
        assert!(hit2);
    }
}
