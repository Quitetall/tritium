//! Tokenizer-only differential probe; never loads weights or initializes a GPU.
#[cfg(feature = "tokenizer")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;
    use tritium_nn::{ChatRenderOptions, HfJsonTokenizer, Tokenizer};
    let args: Vec<_> = std::env::args().collect();
    if !(4..=5).contains(&args.len()) {
        return Err(
            "usage: source_chat_prompt TOKENIZER_JSON TOKENIZER_CONFIG MESSAGES_JSON [true|false]"
                .into(),
        );
    }
    let options = match args.get(4).map(String::as_str) {
        None => ChatRenderOptions::default(),
        Some("true") => ChatRenderOptions::default().with_thinking(true),
        Some("false") => ChatRenderOptions::default().with_thinking(false),
        _ => return Err("thinking input must be true or false".into()),
    };
    let tokenizer = HfJsonTokenizer::from_files(Path::new(&args[1]), Path::new(&args[2]))?;
    let messages: Vec<serde_json::Value> = serde_json::from_str(&args[3])?;
    let messages: Vec<_> = messages
        .iter()
        .map(|message| {
            Ok((
                message["role"].as_str().ok_or("role must be a string")?,
                message["content"]
                    .as_str()
                    .ok_or("content must be a string")?,
            ))
        })
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;
    let prompt = tokenizer
        .render_chat_prompt_with_options(&messages, options, 1024 * 1024)?
        .ok_or("no source template")?;
    let ids = tokenizer.encode(&prompt)?;
    println!("{}", serde_json::json!({"prompt":prompt,"input_ids":ids}));
    Ok(())
}

#[cfg(not(feature = "tokenizer"))]
fn main() {
    eprintln!("source_chat_prompt requires --features tokenizer");
    std::process::exit(2);
}
