use std::fs;

use tritium_spec::{TrainingVectorSetV2, TrainingVectorSetV3};
use tritium_testkit::{
    TrainingReceiptSourcePolicyV1, TrainingVectorCorpus, admit_training_receipts,
    render_development_training_capability_table, render_training_capability_table,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments: Vec<_> = std::env::args().skip(1).collect();
    let source_policy = if arguments
        .first()
        .is_some_and(|value| value == "--allow-dirty")
    {
        arguments.remove(0);
        TrainingReceiptSourcePolicyV1::Development
    } else {
        TrainingReceiptSourcePolicyV1::ReleaseCandidate
    };
    let schema = if arguments.first().is_some_and(|value| value == "--schema") {
        arguments.remove(0);
        if arguments.is_empty() {
            return Err("--schema requires v2 or v3".into());
        }
        arguments.remove(0)
    } else {
        "v3".to_owned()
    };
    if arguments.is_empty() {
        return Err("usage: training_capability_table [--allow-dirty] [--schema v2|v3] DIGEST=RECEIPT.json [...]".into());
    }
    match schema.as_str() {
        "v2" => {
            let vectors = TrainingVectorSetV2::parse_json(include_bytes!(
                "../../../spec/training/v2/vectors/v2.json"
            ))?;
            render_table(&vectors, arguments, source_policy)
        }
        "v3" => {
            let vectors = TrainingVectorSetV3::parse_json(include_bytes!(
                "../../../spec/training/v3/vectors/v3.json"
            ))?;
            render_table(&vectors, arguments, source_policy)
        }
        _ => Err("schema must be v2 or v3".into()),
    }
}

fn render_table(
    vectors: &impl TrainingVectorCorpus,
    arguments: Vec<String>,
    source_policy: TrainingReceiptSourcePolicyV1,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut admitted = Vec::with_capacity(arguments.len());
    for argument in arguments {
        let (digest, path) = argument
            .split_once('=')
            .ok_or("receipt argument must be DIGEST=PATH")?;
        admitted.push(admit_training_receipts(
            &fs::read(path)?,
            vectors,
            parse_digest(digest)?,
            source_policy,
        )?);
    }
    let table = match source_policy {
        TrainingReceiptSourcePolicyV1::ReleaseCandidate => {
            render_training_capability_table(&admitted)?
        }
        TrainingReceiptSourcePolicyV1::Development => {
            render_development_training_capability_table(&admitted)?
        }
    };
    print!("{table}");
    Ok(())
}

fn parse_digest(value: &str) -> Result<[u8; 32], Box<dyn std::error::Error>> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("receipt digest must be 64 lowercase hexadecimal characters".into());
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in value.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        let text = std::str::from_utf8(pair)?;
        digest[index] = u8::from_str_radix(text, 16)?;
    }
    Ok(digest)
}
