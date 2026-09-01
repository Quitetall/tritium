//! Bounded primitives for release-candidate evidence.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use clap::Subcommand;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tritium_format::PackageHasher;

const BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Debug, Subcommand)]
pub(crate) enum ReleaseCommand {
    /// Stream one ordinary file and emit its exact physical identity as JSON.
    Digest {
        /// File to hash. Symlinks and non-regular files are rejected.
        path: PathBuf,
    },
    /// Stream standard input and emit raw plus transport-package identities.
    DigestStream,
}

#[derive(Debug, Serialize)]
struct FileIdentity {
    schema: &'static str,
    bytes: u64,
    sha256: String,
    blake3: String,
}

#[derive(Debug, Serialize)]
struct StreamIdentity {
    schema: &'static str,
    bytes: u64,
    sha256: String,
    blake3: String,
    package_id: String,
}

pub(crate) fn run(command: ReleaseCommand) -> anyhow::Result<()> {
    match command {
        ReleaseCommand::Digest { path } => {
            let identity = digest(&path)?;
            println!("{}", serde_json::to_string(&identity)?);
        }
        ReleaseCommand::DigestStream => {
            let identity = digest_stream(io::stdin().lock())?;
            println!("{}", serde_json::to_string(&identity)?);
        }
    }
    Ok(())
}

fn digest_stream(mut reader: impl Read) -> anyhow::Result<StreamIdentity> {
    let mut sha256 = Sha256::new();
    let mut blake3 = blake3::Hasher::new();
    let mut package = PackageHasher::new();
    let mut bytes = 0_u64;
    let mut buffer = vec![0_u8; BUFFER_BYTES];
    loop {
        let read = reader
            .read(&mut buffer)
            .context("read release input stream")?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(u64::try_from(read).expect("buffer length fits u64"))
            .context("release input byte count overflow")?;
        sha256.update(&buffer[..read]);
        blake3.update(&buffer[..read]);
        package.update(&buffer[..read]);
    }
    Ok(StreamIdentity {
        schema: "tritium.stream-identity.v1",
        bytes,
        sha256: crate::hex::hex_digest(&sha256.finalize()),
        blake3: blake3.finalize().to_hex().to_string(),
        package_id: package.finalize().to_string(),
    })
}

fn digest(path: &Path) -> anyhow::Result<FileIdentity> {
    let metadata = path
        .symlink_metadata()
        .with_context(|| format!("inspect release input {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!("release input must be an ordinary file");
    }
    let file =
        File::open(path).with_context(|| format!("open release input {}", path.display()))?;
    let mut reader = BufReader::with_capacity(BUFFER_BYTES, file);
    let mut sha256 = Sha256::new();
    let mut blake3 = blake3::Hasher::new();
    let mut bytes = 0_u64;
    let mut buffer = vec![0_u8; BUFFER_BYTES];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("read release input {}", path.display()))?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(u64::try_from(read).expect("buffer length fits u64"))
            .context("release input byte count overflow")?;
        sha256.update(&buffer[..read]);
        blake3.update(&buffer[..read]);
    }
    if bytes != metadata.len() {
        bail!("release input changed while it was being hashed");
    }
    Ok(FileIdentity {
        schema: "tritium.file-identity.v1",
        bytes,
        sha256: crate::hex::hex_digest(&sha256.finalize()),
        blake3: blake3.finalize().to_hex().to_string(),
    })
}
