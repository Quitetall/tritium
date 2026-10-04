//! Seal portable-training v2 receipts from inside a real WASI guest.
//!
//! `TRITIUM_WASM_PHYSICAL_DEVICE` is populated by the host runner from its
//! observed Wasmtime version and host architecture. The release evidence must
//! retain that host invocation alongside the emitted content-addressed bundle.

use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::Path;

use tritium_spec::TrainingVectorSetV2;
use tritium_testkit::seal_training_receipts;
use tritium_wasm::WasmTrainBackendV1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let output_dir = arguments
        .next()
        .ok_or("usage: seal_wasi_training_receipts OUTPUT_DIR")?;
    if arguments.next().is_some() {
        return Err("unexpected arguments".into());
    }

    let physical_device = option_env!("TRITIUM_WASM_PHYSICAL_DEVICE")
        .filter(|value| value.starts_with("wasmtime:") && value.len() > "wasmtime:".len())
        .ok_or("host runner must compile with an observed wasmtime identity")?;
    let vectors = TrainingVectorSetV2::parse_json(include_bytes!(
        "../../../spec/training/v2/vectors/v2.json"
    ))?;
    let backend = WasmTrainBackendV1::new(physical_device)?;
    let sealed = seal_training_receipts(&backend, &vectors)?;

    let output_dir = Path::new(&output_dir);
    fs::create_dir_all(output_dir)?;
    let destination = output_dir.join(format!("{}.json", sealed.digest_hex()));
    if destination.exists() {
        if fs::read(&destination)? != sealed.bytes() {
            return Err("content-addressed receipt path contains different bytes".into());
        }
    } else {
        let temporary = output_dir.join(format!(
            ".{}.{}.{}.tmp",
            sealed.digest_hex(),
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(sealed.bytes())?;
        file.sync_all()?;
        match fs::hard_link(&temporary, &destination) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if fs::read(&destination)? != sealed.bytes() {
                    let _ = fs::remove_file(&temporary);
                    return Err("content-addressed receipt path contains different bytes".into());
                }
            }
            Err(error) => {
                let _ = fs::remove_file(&temporary);
                return Err(error.into());
            }
        }
        fs::remove_file(&temporary)?;
    }
    println!("{}={}", sealed.digest_hex(), destination.display());
    Ok(())
}
