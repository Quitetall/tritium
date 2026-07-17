//! Print the content-derived identity of one Qwen3.5-family HF source.

use std::error::Error;
use std::path::PathBuf;

use serde_json::json;
use tritium_nn::Qwen35HfSource;

fn main() -> Result<(), Box<dyn Error>> {
    let source_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: qwen35_hf_identity <source-directory>")?;
    let source = Qwen35HfSource::open(&source_dir)?.verify_semantic_identity()?;
    println!(
        "{}",
        json!({
            "source_model_id": hex(source.model_id().as_bytes()),
            "source_config_digest": hex(source.identity().manifest().config_digest()),
        })
    );
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}
