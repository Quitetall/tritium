//! Original bundle filesystem adapter over the shared model importer and replay.
use anyhow::{Context, Result, ensure};
use clap::{Args, Subcommand, ValueEnum};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    fs::{self, File, Metadata, OpenOptions},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};
use tritium_evidence::{EvidenceRecorder, LEGACY_MODEL_IMPORT_PREPARED_REGISTRATION, verify_jsonl};
use tritium_format::{
    PackageHasher, SafeTensorsReader, SaltTensorSpoolLimits, TritPackageReadLimits,
    salt_v2_package::SaltV2PackageReader, trit_model::TritModelReader,
};
use tritium_quantize::engine::{
    DirectoryTensorSink, QuantEngineError, SafeTensorsWeightSource, SaltV2ModelImportRequest,
    WeightSource, import_salt_v2_model, inspect_salt_v2_model_import,
    verify_imported_model_with_log,
};
use tritium_schema::{
    LEGACY_SALT_V2_HF_ASSETS, LegacySaltV2BundleDescriptor, PlaneCodec, QuantAction, QuantLimits,
    QuantPlan, QuantSourceTensor, QuantTensorPlan,
};

#[derive(Subcommand, Debug)]
pub(crate) enum MigrateCommand {
    /// Import one original provisional SALT V2/HF bundle profile without fitting.
    SaltV2Bundle(BundleArgs),
    /// Reopen a .trit and replay its actual prepared-import log (not qualification).
    Verify(VerifyArgs),
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Profile {
    CompactV1,
    NearLosslessV1,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Codec {
    D2,
    B3,
}

#[derive(Args, Debug)]
pub(crate) struct BundleArgs {
    /// Strict read-only source/inventory/logical sizing; no scale proof or output.
    #[arg(long)]
    inspect_only: bool,
    /// Original schema-v3 bundle directory; not a downloaded/official source claim.
    #[arg(long)]
    bundle: PathBuf,
    /// Explicit original profile; the other payload is neither read nor admitted.
    #[arg(long, value_enum)]
    profile: Profile,
    /// New destination directory containing model.trit and evidence.jsonl.
    #[arg(
        long,
        required_unless_present = "inspect_only",
        conflicts_with = "inspect_only"
    )]
    output: Option<PathBuf>,
    /// Explicit new temporary spool file; removed on success or refusal.
    #[arg(
        long,
        required_unless_present = "inspect_only",
        conflicts_with = "inspect_only"
    )]
    spool: Option<PathBuf>,
    /// Stable run identity for mandatory prepared evidence, separate from CLI summary.
    #[arg(
        long,
        required_unless_present = "inspect_only",
        conflicts_with = "inspect_only"
    )]
    import_run_id: Option<String>,
    /// Canonical target plane coding, not an execution-profile admission.
    #[arg(long, value_enum)]
    codec: Codec,
    /// Explicit target scale group; original FP16 scale bits must be representable.
    #[arg(long)]
    group: u16,
    #[command(flatten)]
    limits: Limits,
}

#[derive(Args, Debug)]
struct Limits {
    /// Physical original profile cap, checked before strict indexing; not RSS.
    #[arg(long, default_value_t = 32 * 1024 * 1024 * 1024u64)]
    max_source_bytes: u64,
    /// Raw dense tensor cap; original preserved container also has a 64MiB ceiling.
    #[arg(long, default_value_t = 64 * 1024 * 1024u64)]
    max_dense_bytes: u64,
    /// Per-asset logical cap, additionally bounded by the original catalog.
    #[arg(long, default_value_t = 32 * 1024 * 1024u64)]
    max_asset_bytes: u64,
    /// Per-tensor logical trit count ceiling, not a heap ceiling.
    #[arg(long, default_value_t = 2 * 1024 * 1024 * 1024u64)]
    max_trits: u64,
    /// Per-tensor logical scale count ceiling.
    #[arg(long, default_value_t = 128 * 1024 * 1024u64)]
    max_scales: u64,
    /// Tile importer staging cap, excluding retained metadata and total process RSS.
    #[arg(long, default_value_t = 32 * 1024 * 1024u64)]
    max_working_bytes: u64,
    /// Per-tensor encoded additive payload ceiling.
    #[arg(long, default_value_t = 8 * 1024 * 1024 * 1024u64)]
    max_tensor_bytes: u64,
    /// Complete raw spool ceiling, not physical filesystem free-space admission.
    #[arg(long, default_value_t = 32 * 1024 * 1024 * 1024u64)]
    max_spool_bytes: u64,
    #[command(flatten)]
    receiving: Receiving,
}
#[derive(Args, Debug)]
struct Receiving {
    /// Complete output/receiving package cap, not process RSS.
    #[arg(long, default_value_t = 32 * 1024 * 1024 * 1024u64)]
    max_package_bytes: u64,
    /// Encoded model manifest cap.
    #[arg(long, default_value_t = 16 * 1024 * 1024u64)]
    max_manifest_bytes: u64,
    /// Encoded blob directory cap.
    #[arg(long, default_value_t = 16 * 1024 * 1024u64)]
    max_directory_bytes: u64,
}
impl Receiving {
    fn read_limits(&self) -> Result<TritPackageReadLimits> {
        ensure!(
            self.max_package_bytes > 0
                && self.max_manifest_bytes > 0
                && self.max_directory_bytes > 0,
            "receiving limits must be positive"
        );
        Ok(TritPackageReadLimits {
            max_source_bytes: self.max_package_bytes,
            max_manifest_bytes: usize::try_from(self.max_manifest_bytes)?,
            max_directory_bytes: usize::try_from(self.max_directory_bytes)?,
        })
    }
}
#[derive(Args, Debug)]
pub(crate) struct VerifyArgs {
    #[arg(long)]
    model: PathBuf,
    /// Actual prepared import JSONL, not the CLI summary log.
    #[arg(long)]
    import_evidence: PathBuf,
    #[arg(long, default_value_t = 16 * 1024 * 1024u64)]
    max_evidence_bytes: u64,
    #[command(flatten)]
    receiving: Receiving,
}

pub(crate) fn run(command: MigrateCommand) -> Result<()> {
    match command {
        MigrateCommand::SaltV2Bundle(args) => import(&args),
        MigrateCommand::Verify(args) => {
            let limits = args.receiving.read_limits()?;
            let (model, _) = FileGuard::open(&args.model, limits.max_source_bytes, false)?;
            let (mut log_file, _) =
                FileGuard::open(&args.import_evidence, args.max_evidence_bytes, false)?;
            let bytes = read_small(&mut log_file, args.max_evidence_bytes)?;
            let log = verify_jsonl(std::str::from_utf8(&bytes).context("import evidence UTF-8")?)?;
            let mut model = TritModelReader::open(model, limits)?;
            let view = verify_imported_model_with_log(&mut model, &log)?;
            println!(
                "{}",
                serde_json::json!({
                    "status": "import-consistent-not-qualified",
                    "package_id": model.package_id().to_string(),
                    "legacy_id": view.observation().legacy_id,
                })
            );
            Ok(())
        }
    }
}

fn import(args: &BundleArgs) -> Result<()> {
    let limits = &args.limits;
    let receiving = limits.receiving.read_limits()?;
    ensure!(
        [
            limits.max_source_bytes,
            limits.max_dense_bytes,
            limits.max_asset_bytes,
            limits.max_trits,
            limits.max_scales,
            limits.max_working_bytes,
            limits.max_tensor_bytes,
            limits.max_spool_bytes
        ]
        .into_iter()
        .all(|n| n > 0),
        "source/spool/working limits must be positive"
    );
    let recorder = if args.inspect_only {
        None
    } else {
        ensure!(
            args.output.is_some() && args.spool.is_some(),
            "import requires output and spool"
        );
        Some(EvidenceRecorder::new(
            args.import_run_id
                .as_deref()
                .context("import requires --import-run-id")?,
            [LEGACY_MODEL_IMPORT_PREPARED_REGISTRATION],
        )?)
    };
    ensure!(
        !fs::symlink_metadata(&args.bundle)?.file_type().is_symlink()
            && fs::metadata(&args.bundle)?.is_dir(),
        "bundle must be an original regular directory"
    );
    let (mut descriptor_file, descriptor_guard) =
        FileGuard::open(&args.bundle.join("tritium.json"), 1024 * 1024, true)?;
    let descriptor_bytes = read_small(&mut descriptor_file, 1024 * 1024)?;
    let descriptor: LegacySaltV2BundleDescriptor = serde_json::from_slice(&descriptor_bytes)
        .context("parse original strict schema-v3 bundle descriptor")?;
    descriptor
        .validate(tritium_nn::QWEN36_27B_REVISION)
        .map_err(anyhow::Error::msg)?;
    let mut guards = vec![descriptor_guard];
    let mut assets = BTreeMap::from([("tritium.json".into(), descriptor_bytes)]);
    for (asset, (_, original_cap)) in descriptor.hf_assets.iter().zip(LEGACY_SALT_V2_HF_ASSETS) {
        let (mut file, guard) = FileGuard::open(
            &args.bundle.join(&asset.file),
            original_cap.min(limits.max_asset_bytes),
            true,
        )?;
        ensure!(
            guard.size == asset.bytes && guard.hash.as_deref() == Some(&asset.package_id),
            "HF asset {} differs from original descriptor",
            asset.file
        );
        assets.insert(
            asset.file.clone(),
            read_small(&mut file, original_cap.min(limits.max_asset_bytes))?,
        );
        guards.push(guard);
    }
    let config: serde_json::Value = serde_json::from_slice(&assets["config.json"])?;
    let architecture = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty())
        .context("original config has no model_type")?
        .to_owned();
    let preserved = &descriptor.preserved;
    let (mut file, guard) =
        FileGuard::open(&args.bundle.join(&preserved.file), 64 * 1024 * 1024, true)?;
    ensure!(
        guard.size == preserved.serialized_bytes
            && guard.hash.as_deref() == Some(&preserved.package_id),
        "preserved container differs from original descriptor"
    );
    file.seek(SeekFrom::Start(0))?;
    let mut header = [0; 8];
    file.read_exact(&mut header)?;
    let data_start = u64::from_le_bytes(header)
        .checked_add(8)
        .context("preserved header overflow")?;
    ensure!(
        guard.size.checked_sub(data_start) == Some(preserved.payload_bytes),
        "preserved payload byte ledger differs from original descriptor"
    );
    file.seek(SeekFrom::Start(0))?;
    let count = SafeTensorsReader::new(file.try_clone()?)?.len();
    ensure!(
        u64::try_from(count)? == preserved.tensors,
        "preserved tensor count differs from descriptor"
    );
    file.seek(SeekFrom::Start(0))?;
    guards.push(guard);
    let inner = SafeTensorsWeightSource::new([file], assets, BTreeMap::new())?;
    let profile = match args.profile {
        Profile::CompactV1 => &descriptor.profiles.compact,
        Profile::NearLosslessV1 => &descriptor.profiles.near_lossless,
    };
    let (file, guard) = FileGuard::open(
        &args.bundle.join(&profile.file),
        limits.max_source_bytes,
        false,
    )?;
    let mut legacy = SaltV2PackageReader::new_strict(file)?;
    let codec_name = match legacy.codec() {
        tritium_format::salt_v2::SaltV2Codec::D2 => "d2",
        tritium_format::salt_v2::SaltV2Codec::B3 => "b3",
        tritium_format::salt_v2::SaltV2Codec::S34 => "s34",
        _ => anyhow::bail!("unsupported original codec"),
    };
    ensure!(
        legacy.package_id().to_string() == profile.package_id
            && legacy.ledger().total_bytes == profile.serialized_bytes
            && legacy.indexed_runtime_ledger()?.steady_resident_bytes() == profile.resident_bytes
            && codec_name == descriptor.packing,
        "selected profile identity/physical ledger/codec differs from descriptor"
    );
    guards.push(guard);
    let mut source = GuardedCompanions { inner, guards };
    let inventory = source.inventory()?;
    let qwen_inventory_matches = if architecture == "qwen3_5" {
        let config = tritium_nn::Qwen35CheckpointConfig::from_hf_config(std::str::from_utf8(
            &source
                .inner
                .asset("config.json", platform_asset_limit(limits.max_asset_bytes)?)?,
        )?)?;
        let schema = tritium_nn::qwen35_language_mtp_tensor_schema(&config.text)?;
        ensure!(
            legacy.len().checked_add(inventory.len()) == Some(schema.len()),
            "original Qwen language/MTP inventory count differs from config"
        );
        for expected in schema {
            let shape = expected.shape.iter().map(|&n| n as u64).collect::<Vec<_>>();
            match expected.role {
                tritium_nn::Qwen35TensorSchemaRole::Matrix => ensure!(
                    legacy
                        .tensor_info(&expected.name)
                        .is_some_and(|t| t.dims() == shape),
                    "original Qwen matrix {} missing or has wrong shape",
                    expected.name
                ),
                tritium_nn::Qwen35TensorSchemaRole::Preserved => ensure!(
                    inventory
                        .iter()
                        .any(|t| t.name == expected.name && t.shape == shape),
                    "original Qwen preserved tensor {} missing or has wrong shape",
                    expected.name
                ),
            }
        }
        true
    } else {
        false
    };
    let request = SaltV2ModelImportRequest {
        companions: QuantPlan {
            architecture,
            tensors: inventory
                .into_iter()
                .map(|source| QuantTensorPlan {
                    source,
                    action: QuantAction::KeepDense,
                })
                .collect(),
            assets: source.asset_names(),
            levels: vec!["imported".into()],
            limits: QuantLimits {
                max_source_bytes: limits.max_dense_bytes,
                max_asset_bytes: limits.max_asset_bytes,
                max_retained_blob_bytes: limits.max_spool_bytes,
                max_package_bytes: receiving.max_source_bytes,
            },
        },
        codec: match args.codec {
            Codec::D2 => PlaneCodec::D2,
            Codec::B3 => PlaneCodec::B3,
        },
        group: args.group,
        tensor_limits: SaltTensorSpoolLimits {
            max_source_bytes: limits.max_source_bytes,
            max_trits: limits.max_trits,
            max_scales: limits.max_scales,
            max_working_bytes: limits.max_working_bytes,
            max_payload_bytes: limits.max_tensor_bytes,
        },
        max_spool_bytes: limits.max_spool_bytes,
        receiving,
    };
    let inspection = inspect_salt_v2_model_import(&request, &legacy, &mut source)?;
    if args.inspect_only {
        legacy.verify_unchanged()?;
        source.verify()?;
        println!(
            "{}",
            serde_json::json!({
                "status":"strict-indexed-plan-not-imported-not-qualified",
                "legacy_id":legacy.package_id().to_string(),
                "selected_source_bytes":legacy.ledger().total_bytes,
                "matrix_tensors":inspection.matrices.len(),"preserved_tensors":request.companions.tensors.len(),
                "qwen_language_mtp_inventory":if qwen_inventory_matches {"matches-config-metadata-only"} else {"unknown-generic-architecture"},
                "additive_payload_bytes":inspection.additive_payload_bytes,"dense_payload_bytes":inspection.dense_payload_bytes,
                "asset_payload_bytes":inspection.asset_payload_bytes,"spool_payload_bytes":inspection.spool_payload_bytes,
                "maximum_tensor_trits":inspection.matrices.iter().map(|m|m.spool.trits).max(),
                "maximum_tensor_scales":inspection.matrices.iter().map(|m|m.spool.scales).max(),
                "maximum_tensor_payload_bytes":inspection.matrices.iter().map(|m|m.spool.payload_bytes).max(),
                "maximum_tensor_working_bytes":inspection.matrices.iter().map(|m|m.spool.working_bytes).max(),
                "scale_representability":"not-checked","dense_payload_validation":"not-checked-by-plan",
                "container_overhead":"not-planned","physical_memory":"not-covered",
                "complete_model":descriptor.complete_model,"official_payload_authenticated":descriptor.official_payload_authenticated,
                "source_identity_status":descriptor.source_identity_status,"source_authentication":"not-verified-by-import",
                "output_created":false,"spool_created":false,
            })
        );
        return Ok(());
    }
    let output = args.output.as_ref().context("import requires --output")?;
    let spool_path = args.spool.as_ref().context("import requires --spool")?;
    let recorder = recorder.context("import requires prepared recorder")?;
    let mut spool = OwnedSpool::new(spool_path)?;
    let imported = import_salt_v2_model(
        &request,
        &mut legacy,
        &mut source,
        &mut spool.file,
        &mut DirectoryTensorSink::new(output),
        recorder,
    )?;
    let mut warnings = imported
        .warnings
        .iter()
        .map(|w| format!("{w:?}"))
        .collect::<Vec<_>>();
    if let Err(error) = spool.remove() {
        warnings.push(format!("owned spool cleanup: {error}"));
    }
    println!(
        "{}",
        serde_json::json!({
            "status": "imported-not-qualified",
            "package_id": imported.model.package_id().to_string(),
            "complete_model": descriptor.complete_model,
            "official_payload_authenticated": descriptor.official_payload_authenticated,
            "source_identity_status": descriptor.source_identity_status,
            "source_authentication": "not-verified-by-import",
            "warnings": warnings,
        })
    );
    Ok(())
}

fn platform_asset_limit(cap: u64) -> Result<usize> {
    usize::try_from(cap).context("asset cap exceeds platform")
}

fn same_file(a: &Metadata, b: &Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        a.dev() == b.dev() && a.ino() == b.ino()
    }
    #[cfg(not(unix))]
    {
        // No destructive cleanup based on size/mtime pretending to be file identity.
        let _ = (a, b);
        false
    }
}
fn hash_file(file: &mut File, expected: u64) -> Result<String> {
    ensure!(
        file.metadata()?.len() == expected,
        "original file length changed before hash"
    );
    file.seek(SeekFrom::Start(0))?;
    let mut hasher = PackageHasher::new();
    let mut buffer = [0; 64 * 1024];
    let mut consumed = 0u64;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        consumed = consumed
            .checked_add(n as u64)
            .context("file hash length overflow")?;
        ensure!(consumed <= expected, "original file grew while hashing");
        hasher.update(&buffer[..n]);
    }
    file.seek(SeekFrom::Start(0))?;
    ensure!(consumed == expected, "original file shrank while hashing");
    Ok(hasher.finalize().to_string())
}
fn read_small(file: &mut File, cap: u64) -> Result<Vec<u8>> {
    let len = file.metadata()?.len();
    ensure!(len <= cap, "input exceeds encoded cap");
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(usize::try_from(len)?)?;
    file.seek(SeekFrom::Start(0))?;
    file.take(cap.checked_add(1).context("input cap overflow")?)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 == len,
        "input length changed during bounded read"
    );
    Ok(bytes)
}
struct FileGuard {
    path: PathBuf,
    file: File,
    size: u64,
    hash: Option<String>,
}
impl FileGuard {
    fn open(path: &Path, cap: u64, hash: bool) -> Result<(File, Self)> {
        ensure!(cap > 0, "input cap must be positive");
        let path_meta = fs::symlink_metadata(path)?;
        ensure!(
            path_meta.is_file(),
            "input must be a regular non-symlink file: {}",
            path.display()
        );
        ensure!(
            path_meta.len() <= cap,
            "input {} exceeds cap {cap}",
            path.display()
        );
        let mut file = File::open(path)?;
        ensure!(
            file.metadata()?.is_file() && same_file(&path_meta, &file.metadata()?),
            "input changed while opening: {}",
            path.display()
        );
        let size = file.metadata()?.len();
        ensure!(size <= cap, "opened input exceeds encoded cap");
        let guard = Self {
            path: path.into(),
            size,
            hash: if hash {
                Some(hash_file(&mut file, size)?)
            } else {
                None
            },
            file: file.try_clone()?,
        };
        Ok((file, guard))
    }
    fn verify(&self) -> Result<()> {
        let current = fs::symlink_metadata(&self.path)?;
        let retained = self.file.metadata()?;
        ensure!(
            current.is_file()
                && same_file(&current, &retained)
                && current.len() == self.size
                && retained.len() == self.size,
            "original file/path changed: {}",
            self.path.display()
        );
        if let Some(expected) = &self.hash {
            ensure!(
                &hash_file(&mut self.file.try_clone()?, self.size)? == expected,
                "original container bytes changed: {}",
                self.path.display()
            );
        }
        Ok(())
    }
}
struct GuardedCompanions {
    inner: SafeTensorsWeightSource<File>,
    guards: Vec<FileGuard>,
}
impl GuardedCompanions {
    fn verify(&self) -> std::result::Result<(), QuantEngineError> {
        for guard in &self.guards {
            guard
                .verify()
                .map_err(|e| QuantEngineError::Source(e.to_string()))?;
        }
        Ok(())
    }
}
impl WeightSource for GuardedCompanions {
    fn inventory(&self) -> std::result::Result<Vec<QuantSourceTensor>, QuantEngineError> {
        self.verify()?;
        self.inner.inventory()
    }
    fn asset_names(&self) -> Vec<String> {
        self.inner.asset_names()
    }
    fn tensor(
        &mut self,
        name: &str,
        max: usize,
    ) -> std::result::Result<Cow<'_, [u8]>, QuantEngineError> {
        let bytes = self.inner.tensor(name, max)?.into_owned();
        if self
            .inner
            .inventory()?
            .last()
            .is_some_and(|t| t.name == name)
        {
            self.verify()?;
        }
        Ok(Cow::Owned(bytes))
    }
    fn asset(
        &mut self,
        name: &str,
        max: usize,
    ) -> std::result::Result<Cow<'_, [u8]>, QuantEngineError> {
        self.inner.asset(name, max)
    }
    fn input_gram(
        &mut self,
        name: &str,
        max: usize,
    ) -> std::result::Result<Cow<'_, [f64]>, QuantEngineError> {
        self.inner.input_gram(name, max)
    }
}
struct OwnedSpool {
    path: PathBuf,
    file: File,
    removed: bool,
}
impl OwnedSpool {
    fn new(path: &Path) -> Result<Self> {
        Ok(Self {
            path: path.into(),
            file: OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(path)
                .context("create new import spool without clobber")?,
            removed: false,
        })
    }
    fn remove(&mut self) -> Result<()> {
        let metadata = fs::symlink_metadata(&self.path)?;
        ensure!(
            metadata.is_file() && same_file(&metadata, &self.file.metadata()?),
            "spool path replaced; refusing cleanup"
        );
        fs::remove_file(&self.path).context("remove owned import spool")?;
        self.removed = true;
        Ok(())
    }
}
impl Drop for OwnedSpool {
    fn drop(&mut self) {
        if !self.removed
            && let Err(error) = self.remove()
        {
            eprintln!("owned spool cleanup warning: {error}");
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "tritium-bundle-guard-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    #[test]
    fn retained_guards_reject_container_byte_and_path_replacement() {
        let scratch = Scratch::new();
        let path = scratch.0.join("original");
        fs::write(&path, b"original").unwrap();
        let (_, guard) = FileGuard::open(&path, 8, true).unwrap();
        guard.verify().unwrap();
        fs::write(&path, b"changed!").unwrap();
        assert!(guard.verify().is_err());
        fs::write(&path, b"original").unwrap();
        guard.verify().unwrap();
        fs::rename(&path, scratch.0.join("retained")).unwrap();
        fs::write(&path, b"original").unwrap();
        #[cfg(unix)]
        assert!(guard.verify().is_err());
    }
    #[test]
    fn actual_companion_adapter_detects_header_only_and_cached_asset_mutations() {
        for mutate_asset in [false, true] {
            let scratch = Scratch::new();
            let path = scratch.0.join("preserved.safetensors");
            let header = br#"{"norm":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            bytes.extend([0x80, 0x3f, 0x80, 0xbf]);
            fs::write(&path, &bytes).unwrap();
            let (file, guard) = FileGuard::open(&path, 1024, true).unwrap();
            let asset = scratch.0.join("config.json");
            fs::write(&asset, b"original").unwrap();
            let (_, asset_guard) = FileGuard::open(&asset, 1024, true).unwrap();
            let inner = SafeTensorsWeightSource::new(
                [file],
                BTreeMap::from([("config.json".into(), b"original".to_vec())]),
                BTreeMap::new(),
            )
            .unwrap();
            let mut source = GuardedCompanions {
                inner,
                guards: vec![guard, asset_guard],
            };
            source.inventory().unwrap();
            assert_eq!(
                &*source.tensor("norm", 1024).unwrap(),
                [0x80, 0x3f, 0x80, 0xbf]
            );
            if mutate_asset {
                fs::write(&asset, b"changed!").unwrap();
            } else {
                // Same raw data and physical length, but original dtype header changed.
                let i = bytes.windows(4).position(|w| w == b"BF16").unwrap();
                bytes[i..i + 4].copy_from_slice(b"F16 ");
                fs::write(&path, bytes).unwrap();
            }
            assert!(source.tensor("norm", 1024).is_err());
            assert!(source.inventory().is_err());
        }
    }
    #[test]
    fn owned_spool_never_removes_replacement_or_preexisting_file() {
        let scratch = Scratch::new();
        let path = scratch.0.join("spool");
        fs::write(&path, b"foreign").unwrap();
        assert!(OwnedSpool::new(&path).is_err());
        fs::remove_file(&path).unwrap();
        let mut spool = OwnedSpool::new(&path).unwrap();
        fs::rename(&path, scratch.0.join("owned")).unwrap();
        fs::write(&path, b"foreign").unwrap();
        #[cfg(unix)]
        assert!(spool.remove().is_err());
        drop(spool);
        #[cfg(unix)]
        assert_eq!(fs::read(path).unwrap(), b"foreign");
    }
}
