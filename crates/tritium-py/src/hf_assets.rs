//! Bounded, content-bound Hugging Face language assets for Qwen3.6 bundles.

use std::{
    fs::{self, File, Metadata},
    io::{Read, Write},
    path::Path,
};

use tritium_format::{PackageHasher, PackageId};
use tritium_nn::{QWEN36_27B_REVISION, Qwen35CheckpointConfig};

pub(crate) const ASSET_SPECS: [HfAssetSpec; 8] = [
    HfAssetSpec::new("chat_template.jinja", 1_048_576),
    HfAssetSpec::new("config.json", 1_048_576),
    HfAssetSpec::new("configuration.json", 65_536),
    HfAssetSpec::new("generation_config.json", 1_048_576),
    HfAssetSpec::new("merges.txt", 8_388_608),
    HfAssetSpec::new("tokenizer.json", 33_554_432),
    HfAssetSpec::new("tokenizer_config.json", 1_048_576),
    HfAssetSpec::new("vocab.json", 16_777_216),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HfAssetSpec {
    file: &'static str,
    max_bytes: u64,
}

impl HfAssetSpec {
    const fn new(file: &'static str, max_bytes: u64) -> Self {
        Self { file, max_bytes }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HfAssetReceipt {
    file: &'static str,
    package_id: PackageId,
    bytes: u64,
}

impl HfAssetReceipt {
    pub(crate) const fn file(self) -> &'static str {
        self.file
    }

    pub(crate) const fn package_id(self) -> PackageId {
        self.package_id
    }

    pub(crate) const fn bytes(self) -> u64 {
        self.bytes
    }
}

pub(crate) fn stage_language_assets(
    source: &Path,
    staging: &Path,
) -> Result<Vec<HfAssetReceipt>, String> {
    require_directory(source, "source model")?;
    require_directory(staging, "staging")?;
    let mut receipts = Vec::new();
    receipts
        .try_reserve_exact(ASSET_SPECS.len())
        .map_err(|_| "allocate HF asset receipts failed".to_owned())?;
    for spec in ASSET_SPECS {
        receipts.push(stage_asset(source, staging, spec)?);
    }
    Ok(receipts)
}

pub(crate) fn verify_language_asset(
    path: &Path,
    expected_package_id: &str,
    expected_bytes: u64,
) -> Result<(String, u64), String> {
    let file = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "HF asset path has no UTF-8 filename".to_owned())?;
    let spec = ASSET_SPECS
        .iter()
        .copied()
        .find(|spec| spec.file == file)
        .ok_or_else(|| "HF asset filename is not allowlisted".to_owned())?;
    if expected_package_id.is_empty() || expected_bytes == 0 || expected_bytes > spec.max_bytes {
        return Err(format!("HF asset {file} manifest fields are invalid"));
    }
    let before = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect HF asset {file} failed: {:?}", error.kind()))?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() != expected_bytes {
        return Err(format!(
            "HF asset {file} length or file type differs from manifest"
        ));
    }
    let mut input = File::open(path)
        .map_err(|error| format!("open HF asset {file} failed: {:?}", error.kind()))?;
    let opened = input
        .metadata()
        .map_err(|error| format!("inspect opened HF asset {file} failed: {:?}", error.kind()))?;
    if !opened.is_file() || !same_file(&before, &opened) {
        return Err(format!("HF asset {file} changed before open"));
    }
    let capacity = usize::try_from(expected_bytes)
        .map_err(|_| format!("HF asset {file} exceeds platform bounds"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| format!("allocate HF asset {file} failed"))?;
    Read::by_ref(&mut input)
        .take(spec.max_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read HF asset {file} failed: {:?}", error.kind()))?;
    let after = fs::symlink_metadata(path)
        .map_err(|error| format!("reinspect HF asset {file} failed: {:?}", error.kind()))?;
    let final_opened = input.metadata().map_err(|error| {
        format!(
            "reinspect opened HF asset {file} failed: {:?}",
            error.kind()
        )
    })?;
    if bytes.len() as u64 != expected_bytes
        || !same_file(&before, &after)
        || !same_file(&opened, &final_opened)
    {
        return Err(format!("HF asset {file} changed while reading"));
    }
    validate_asset(file, &bytes)?;
    let mut hasher = PackageHasher::new();
    hasher.update(&bytes);
    let actual_id = hasher.finalize().to_string();
    if actual_id != expected_package_id {
        return Err(format!("HF asset {file} identity differs from manifest"));
    }
    Ok((actual_id, expected_bytes))
}

fn stage_asset(source: &Path, staging: &Path, spec: HfAssetSpec) -> Result<HfAssetReceipt, String> {
    let path = source.join(spec.file);
    let before = fs::symlink_metadata(&path)
        .map_err(|error| format!("inspect HF asset {} failed: {:?}", spec.file, error.kind()))?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() > spec.max_bytes {
        return Err(format!(
            "HF asset {} must be an ordinary file no larger than {} bytes",
            spec.file, spec.max_bytes
        ));
    }
    let mut input = File::open(&path)
        .map_err(|error| format!("open HF asset {} failed: {:?}", spec.file, error.kind()))?;
    let opened = input.metadata().map_err(|error| {
        format!(
            "inspect opened HF asset {} failed: {:?}",
            spec.file,
            error.kind()
        )
    })?;
    if !opened.is_file() || !same_file(&before, &opened) {
        return Err(format!("HF asset {} changed before open", spec.file));
    }
    let capacity = usize::try_from(opened.len())
        .map_err(|_| format!("HF asset {} exceeds platform bounds", spec.file))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| format!("allocate HF asset {} failed", spec.file))?;
    Read::by_ref(&mut input)
        .take(spec.max_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read HF asset {} failed: {:?}", spec.file, error.kind()))?;
    let after = fs::symlink_metadata(&path).map_err(|error| {
        format!(
            "reinspect HF asset {} failed: {:?}",
            spec.file,
            error.kind()
        )
    })?;
    let final_opened = input.metadata().map_err(|error| {
        format!(
            "reinspect opened HF asset {} failed: {:?}",
            spec.file,
            error.kind()
        )
    })?;
    if bytes.is_empty()
        || bytes.len() as u64 != opened.len()
        || !same_file(&before, &after)
        || !same_file(&opened, &final_opened)
    {
        return Err(format!("HF asset {} changed while reading", spec.file));
    }
    validate_asset(spec.file, &bytes)?;
    let mut hasher = PackageHasher::new();
    hasher.update(&bytes);
    let package_id = hasher.finalize();
    let destination = staging.join(spec.file);
    let mut output = File::create_new(&destination).map_err(|error| {
        format!(
            "create staged HF asset {} failed: {:?}",
            spec.file,
            error.kind()
        )
    })?;
    output.write_all(&bytes).map_err(|error| {
        format!(
            "write staged HF asset {} failed: {:?}",
            spec.file,
            error.kind()
        )
    })?;
    output.sync_all().map_err(|error| {
        format!(
            "sync staged HF asset {} failed: {:?}",
            spec.file,
            error.kind()
        )
    })?;
    Ok(HfAssetReceipt {
        file: spec.file,
        package_id,
        bytes: bytes.len() as u64,
    })
}

fn validate_asset(file: &str, bytes: &[u8]) -> Result<(), String> {
    if file.ends_with(".json") {
        serde_json::from_slice::<serde_json::Value>(bytes)
            .map_err(|error| format!("HF asset {file} is invalid JSON: {error}"))?;
    } else {
        std::str::from_utf8(bytes).map_err(|_| format!("HF asset {file} must be valid UTF-8"))?;
    }
    if file == "config.json" {
        let config = std::str::from_utf8(bytes)
            .map_err(|_| "HF asset config.json must be valid UTF-8".to_owned())?;
        Qwen35CheckpointConfig::from_hf_config(config)
            .and_then(|config| config.validate_pinned_qwen36_27b(QWEN36_27B_REVISION))
            .map_err(|error| format!("HF asset config.json differs from pinned model: {error}"))?;
    }
    Ok(())
}

fn require_directory(path: &Path, label: &str) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {label} directory failed: {:?}", error.kind()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!("{label} path must be an ordinary directory"));
    }
    Ok(())
}

#[cfg(unix)]
fn same_file(left: &Metadata, right: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev()
        && left.ino() == right.ino()
        && left.len() == right.len()
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
}

#[cfg(not(unix))]
fn same_file(left: &Metadata, right: &Metadata) -> bool {
    left.len() == right.len()
        && left.modified().ok() == right.modified().ok()
        && left.is_file() == right.is_file()
}

#[cfg(test)]
mod tests {
    use super::{ASSET_SPECS, stage_language_assets, verify_language_asset};
    use std::{fs, path::Path};

    fn fixture_root(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "tritium-hf-assets-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn write_source(root: &Path) {
        fs::create_dir(root).unwrap();
        for spec in ASSET_SPECS {
            let bytes: &[u8] = match spec.file {
                "config.json" => {
                    include_bytes!("../../tritium-nn/tests/fixtures/qwen36-27b-config.json")
                }
                file if file.ends_with(".json") => b"{}",
                _ => b"fixture\n",
            };
            fs::write(root.join(spec.file), bytes).unwrap();
        }
    }

    #[test]
    fn stages_exact_bounded_language_assets() {
        let root = fixture_root("exact");
        let source = root.join("source");
        let staging = root.join("staging");
        fs::create_dir(&root).unwrap();
        write_source(&source);
        fs::create_dir(&staging).unwrap();
        let receipts = stage_language_assets(&source, &staging).unwrap();
        assert_eq!(receipts.len(), ASSET_SPECS.len());
        for receipt in receipts {
            assert_eq!(
                fs::read(staging.join(receipt.file())).unwrap(),
                fs::read(source.join(receipt.file())).unwrap()
            );
            assert_eq!(
                receipt.bytes(),
                fs::metadata(staging.join(receipt.file())).unwrap().len()
            );
            assert!(receipt.package_id().to_string().starts_with("trp1_"));
            assert_eq!(
                verify_language_asset(
                    &staging.join(receipt.file()),
                    &receipt.package_id().to_string(),
                    receipt.bytes(),
                )
                .unwrap(),
                (receipt.package_id().to_string(), receipt.bytes())
            );
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_and_oversized_assets() {
        use std::os::unix::fs::symlink;

        let root = fixture_root("reject");
        let source = root.join("source");
        let staging = root.join("staging");
        fs::create_dir(&root).unwrap();
        write_source(&source);
        fs::create_dir(&staging).unwrap();
        fs::rename(
            source.join("tokenizer_config.json"),
            source.join("actual.json"),
        )
        .unwrap();
        symlink("actual.json", source.join("tokenizer_config.json")).unwrap();
        assert!(
            stage_language_assets(&source, &staging)
                .unwrap_err()
                .contains("ordinary file")
        );

        fs::remove_dir_all(&staging).unwrap();
        fs::create_dir(&staging).unwrap();
        fs::remove_file(source.join("tokenizer_config.json")).unwrap();
        fs::rename(
            source.join("actual.json"),
            source.join("tokenizer_config.json"),
        )
        .unwrap();
        fs::write(source.join("configuration.json"), vec![b' '; 65_537]).unwrap();
        assert!(
            stage_language_assets(&source, &staging)
                .unwrap_err()
                .contains("ordinary file")
        );
        fs::remove_dir_all(root).unwrap();
    }
}
