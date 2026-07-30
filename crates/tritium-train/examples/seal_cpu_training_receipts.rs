use std::fs::{self, OpenOptions};
use std::io::Write as _;

use tritium_spec::TrainingVectorSetV2;
use tritium_testkit::seal_training_receipts;
use tritium_train::CpuTrainBackendV1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vectors = TrainingVectorSetV2::parse_json(include_bytes!(
        "../../../spec/training/v2/vectors/v2.json"
    ))?;
    let output_dir = std::env::args_os()
        .nth(1)
        .ok_or("usage: seal_cpu_training_receipts OUTPUT_DIR")?;
    let backend = CpuTrainBackendV1::new();
    let sealed = seal_training_receipts(&backend, &vectors)?;
    fs::create_dir_all(&output_dir)?;
    let destination =
        std::path::Path::new(&output_dir).join(format!("{}.json", sealed.digest_hex()));
    if destination.exists() {
        if fs::read(&destination)? != sealed.bytes() {
            return Err("content-addressed receipt path contains different bytes".into());
        }
    } else {
        let temporary = std::path::Path::new(&output_dir).join(format!(
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
