//! GGUF-embedded byte-level BPE tokenizer (`tokenizer` feature).
//!
//! A GGUF converted from a HuggingFace checkpoint embeds the whole tokenizer:
//! `tokenizer.ggml.tokens` (the byte-level BPE vocab, index = id),
//! `tokenizer.ggml.merges`, `tokenizer.ggml.token_type` (control tokens), and
//! the special-token ids. [`GgufBpeTokenizer::from_gguf`] rebuilds a
//! HuggingFace `tokenizers` pipeline from that metadata — single-file UX, no
//! sidecar `tokenizer.json` — and implements the [`Tokenizer`] seam the runner
//! and server depend on.
//!
//! Only the `gpt2` tokenizer model (byte-level BPE — the LLaMA-3 / BitNet
//! family) is supported; anything else is rejected loudly rather than
//! mis-tokenized silently.
//!
//! Conformance: encoding the committed reference prompt must reproduce the
//! `transformers` token ids exactly (`tests/tokenizer_conformance.rs`, gated
//! on the real model file).

use tokenizers::models::bpe::BPE;
use tokenizers::{AddedToken, Tokenizer as HfTokenizer};
use tritium_format::{GgufFile, GgufValue};

use crate::error::NnError;
use crate::model::tokenizer::Tokenizer;

/// ggml token type for control/special tokens (`LLAMA_TOKEN_TYPE_CONTROL`).
const GGML_TOKEN_TYPE_CONTROL: i64 = 3;

/// The LLaMA-3 pre-split regex (verbatim from the official tokenizer.json).
/// Notable vs gpt2: digits group in runs of at most THREE, contractions match
/// case-insensitively, and a word may absorb one leading non-letter.
const LLAMA3_SPLIT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// A [`Tokenizer`] rebuilt from GGUF-embedded vocab/merges (byte-level BPE).
pub struct GgufBpeTokenizer {
    inner: HfTokenizer,
    bos: u32,
    eos: u32,
}

impl std::fmt::Debug for GgufBpeTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufBpeTokenizer")
            .field("vocab", &self.inner.get_vocab_size(true))
            .field("bos", &self.bos)
            .field("eos", &self.eos)
            .finish()
    }
}

fn meta_err(what: &str) -> NnError {
    NnError::Tokenizer(format!("GGUF tokenizer metadata: {what}"))
}

fn string_array<'a>(file: &'a GgufFile, key: &str) -> Result<Vec<&'a str>, NnError> {
    file.metadata
        .get(key)
        .and_then(GgufValue::as_array)
        .ok_or_else(|| meta_err(&format!("missing array `{key}`")))?
        .iter()
        .map(|v| v.as_str().ok_or_else(|| meta_err(&format!("`{key}` holds a non-string"))))
        .collect()
}

fn u32_value(file: &GgufFile, key: &str) -> Result<u32, NnError> {
    file.metadata
        .get(key)
        .and_then(GgufValue::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| meta_err(&format!("missing/invalid `{key}`")))
}

impl GgufBpeTokenizer {
    /// Build from a parsed GGUF's embedded tokenizer metadata.
    ///
    /// The end-of-sequence id prefers the vocab's `<|eot_id|>` (the turn
    /// terminator the official BitNet/LLaMA-3 chat template stops on) over
    /// `tokenizer.ggml.eos_token_id` (converters record `<|end_of_text|>`
    /// there, which chat generations never emit).
    ///
    /// # Errors
    /// [`NnError::Tokenizer`] if the GGUF embeds no tokenizer, the tokenizer
    /// model is not `gpt2` (byte-level BPE), or the metadata is malformed.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, NnError> {
        let model = file
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(GgufValue::as_str)
            .ok_or_else(|| meta_err("missing `tokenizer.ggml.model` (no embedded tokenizer)"))?;
        if model != "gpt2" {
            return Err(meta_err(&format!(
                "unsupported tokenizer model {model:?} (only `gpt2` byte-level BPE)"
            )));
        }

        let tokens = string_array(file, "tokenizer.ggml.tokens")?;
        let merges_raw = string_array(file, "tokenizer.ggml.merges")?;

        let vocab: tokenizers::models::bpe::Vocab = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| ((*t).to_owned(), i as u32))
            .collect();
        let merges: Vec<(String, String)> = merges_raw
            .iter()
            .map(|m| {
                m.split_once(' ')
                    .map(|(a, b)| (a.to_owned(), b.to_owned()))
                    .ok_or_else(|| meta_err(&format!("malformed merge entry {m:?}")))
            })
            .collect::<Result<_, _>>()?;

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            // The official LLaMA-3 tokenizer.json sets ignore_merges: vocab
            // hits bypass merge derivation.
            .ignore_merges(true)
            .build()
            .map_err(|e| meta_err(&format!("BPE build failed: {e}")))?;

        let mut inner = HfTokenizer::new(bpe);
        // The LLaMA-3 pipeline is Sequence[Split(llama3 regex, Isolated),
        // ByteLevel(use_regex=FALSE)] — NOT ByteLevel's built-in gpt2 regex,
        // which diverges on >=4-digit runs (\p{N}{1,3} vs \p{N}+), words with
        // leading punctuation, case-insensitive contractions and \r\n
        // (review-measured: 9/21 probe inputs mis-tokenized under gpt2).
        let split = tokenizers::pre_tokenizers::split::Split::new(
            // SplitPattern::Regex must be NAMED: From<&str> yields the
            // String (literal-match) variant.
            tokenizers::pre_tokenizers::split::SplitPattern::Regex(
                LLAMA3_SPLIT_PATTERN.to_owned(),
            ),
            tokenizers::SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| meta_err(&format!("split regex: {e}")))?;
        inner.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::PreTokenizerWrapper::Sequence(
            tokenizers::pre_tokenizers::sequence::Sequence::new(vec![
                tokenizers::pre_tokenizers::PreTokenizerWrapper::Split(split),
                tokenizers::pre_tokenizers::PreTokenizerWrapper::ByteLevel(
                    tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, false),
                ),
            ]),
        )));
        inner.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::new(
            false, true, true,
        )));

        // Control tokens (<|begin_of_text|>, <|eot_id|>, …) become special
        // added tokens so `encode` recognises them verbatim in text and
        // `decode(skip_special)` drops them from chat output.
        if let Some(types) = file
            .metadata
            .get("tokenizer.ggml.token_type")
            .and_then(GgufValue::as_array)
        {
            let specials: Vec<AddedToken> = types
                .iter()
                .enumerate()
                .filter(|(_, t)| t.as_i64() == Some(GGML_TOKEN_TYPE_CONTROL))
                .filter_map(|(i, _)| tokens.get(i))
                .map(|t| AddedToken::from((*t).to_owned(), true))
                .collect();
            inner
                .add_special_tokens(specials)
                .map_err(|e| meta_err(&format!("adding special tokens failed: {e}")))?;
        }

        let bos = u32_value(file, "tokenizer.ggml.bos_token_id")?;
        let eos_meta = u32_value(file, "tokenizer.ggml.eos_token_id")?;
        let eos = inner.token_to_id("<|eot_id|>").unwrap_or(eos_meta);

        Ok(Self { inner, bos, eos })
    }
}

impl Tokenizer for GgufBpeTokenizer {
    /// Encode `text`, prepending BOS — the `tokenizer.ggml.add_bos_token`
    /// convention the GGUF declares and the committed reference uses
    /// (raw prefill = `[bos] + encode(text)`). NOTE this diverges from
    /// transformers' `apply_chat_template` (which adds no BOS for this
    /// family); a client that embeds a literal `<|begin_of_text|>` in its
    /// text will get a double BOS.
    fn encode(&self, text: &str) -> Result<Vec<u32>, NnError> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| NnError::Tokenizer(format!("encode: {e}")))?;
        let mut ids = Vec::with_capacity(enc.get_ids().len() + 1);
        ids.push(self.bos);
        ids.extend_from_slice(enc.get_ids());
        Ok(ids)
    }

    /// Decode, dropping special tokens (chat output should not contain
    /// `<|eot_id|>` markers).
    fn decode(&self, tokens: &[u32]) -> Result<String, NnError> {
        self.inner
            .decode(tokens, true)
            .map_err(|e| NnError::Tokenizer(format!("decode: {e}")))
    }

    fn bos(&self) -> u32 {
        self.bos
    }

    fn eos(&self) -> u32 {
        self.eos
    }
}
