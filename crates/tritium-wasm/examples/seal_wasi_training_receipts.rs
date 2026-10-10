//! Seal portable-training v2 receipts from inside a real WASI guest.
//!
//! `TRITIUM_WASM_PHYSICAL_DEVICE` is populated by the host runner from its
//! observed Wasmtime version and host architecture. The release evidence must
//! retain that host invocation alongside the emitted content-addressed bundle.

use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use tritium_spec::TrainingVectorSetV2;
use tritium_testkit::seal_training_receipts;
use tritium_wasm::WasmTrainBackendV1;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn observed_runtime_identity(value: Option<&str>) -> Option<&str> {
    value.filter(|value| {
        let Some((version, architecture)) = value
            .strip_prefix("wasmtime:")
            .and_then(|identity| identity.split_once(':'))
        else {
            return false;
        };
        !version.is_empty()
            && !architecture.is_empty()
            && !version
                .chars()
                .chain(architecture.chars())
                .any(|character| {
                    character == ':' || character.is_whitespace() || character.is_control()
                })
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let output_dir = arguments
        .next()
        .ok_or("usage: seal_wasi_training_receipts OUTPUT_DIR")?;
    if arguments.next().is_some() {
        return Err("unexpected arguments".into());
    }

    let physical_device = observed_runtime_identity(option_env!("TRITIUM_WASM_PHYSICAL_DEVICE"))
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
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
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

#[cfg(test)]
mod tests {
    use super::observed_runtime_identity;

    #[test]
    fn retains_complete_observed_runtime_identity() {
        for identity in [
            "wasmtime:48.0.1:x86_64",
            "wasmtime:46.0.0:aarch64",
            "wasmtime:49.0.0-dev+custom:riscv64",
        ] {
            assert_eq!(observed_runtime_identity(Some(identity)), Some(identity));
        }
    }

    #[test]
    fn rejects_missing_or_malformed_runtime_identity() {
        assert_eq!(observed_runtime_identity(None), None);
        for identity in [
            "",
            "wasmtime:",
            "wasmtime::",
            "wasmtime:48.0.1",
            "wasmtime:48.0.1:",
            "wasmtime::x86_64",
            "wasmtime:48.0.1:x86_64:extra",
            "wasmtime: 48.0.1:x86_64",
            "wasmtime:48.0.1:x86_64 ",
            "wasmtime:48.0.1:\tx86_64",
            "wasmtime:48.0.1:x86_64\n",
            "wasmtime:48.0.1:x86_64\0",
            "wasmtime:48.0.1:\u{a0}x86_64",
            "other:48.0.1:x86_64",
        ] {
            assert_eq!(
                observed_runtime_identity(Some(identity)),
                None,
                "{identity:?}"
            );
        }
    }
}
