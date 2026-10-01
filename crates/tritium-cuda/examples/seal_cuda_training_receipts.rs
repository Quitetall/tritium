use std::fs::{self, OpenOptions};
use std::io::Write as _;

use tritium_cuda::train::CudaTrainBackendV1;
use tritium_spec::TrainingVectorSetV3;
use tritium_testkit::seal_training_receipts;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let output_dir = arguments
        .next()
        .ok_or("usage: seal_cuda_training_receipts OUTPUT_DIR [DEVICE_INDEX]")?;
    let device_index = arguments
        .next()
        .map(|raw| {
            raw.into_string()
                .map_err(|_| "DEVICE_INDEX must be UTF-8")?
                .parse::<usize>()
                .map_err(|_| "DEVICE_INDEX must be a nonnegative integer")
        })
        .transpose()?
        .unwrap_or(0);
    if arguments.next().is_some() {
        return Err("unexpected arguments".into());
    }

    let vectors = TrainingVectorSetV3::parse_json(include_bytes!(
        "../../../spec/training/v3/vectors/v3.json"
    ))?;
    let backend = CudaTrainBackendV1::new(device_index)?;
    let sealed = seal_training_receipts(&backend, &vectors)?;
    fs::create_dir_all(&output_dir)?;
    let output_dir = std::path::Path::new(&output_dir);
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
