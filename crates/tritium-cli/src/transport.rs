//! `tritium transport`: pack, inspect, and restore seekable outer transport.
//!
//! Transport bytes are delivery/storage bytes only. The command reports the logical
//! fixed-codec size separately and never presents the transport size as resident VRAM.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, bail};
use clap::Subcommand;
use tritium_format::{
    EntropyTransportError, read_entropy_transport, write_entropy_transport_with_chunk_size,
};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Operations over the seekable `TRNS` outer transport.
#[derive(Subcommand, Debug)]
pub(crate) enum TransportCommand {
    /// Encode fixed-codec bytes for deterministic storage or transfer.
    Pack {
        /// Source artifact bytes (for example a SALT or GGUF package).
        input: PathBuf,
        /// Destination `.trns` path.
        output: PathBuf,
        /// Independently seekable chunk size; must be a power of two from 64 KiB to 1 MiB.
        #[arg(long, default_value_t = tritium_format::ENTROPY_TRANSPORT_DEFAULT_CHUNK_BYTES)]
        chunk_size: usize,
    },
    /// Decode a `TRNS` artifact back to its logical fixed-codec bytes.
    Unpack {
        /// Source `.trns` path.
        input: PathBuf,
        /// Destination fixed-codec artifact path.
        output: PathBuf,
    },
    /// Inspect transport index and report storage bytes separately from logical bytes.
    Inspect {
        /// Source `.trns` path.
        input: PathBuf,
    },
}

/// Run one transport operation.
pub(crate) fn run(command: TransportCommand) -> anyhow::Result<()> {
    match command {
        TransportCommand::Pack {
            input,
            output,
            chunk_size,
        } => pack(&input, &output, chunk_size),
        TransportCommand::Unpack { input, output } => unpack(&input, &output),
        TransportCommand::Inspect { input } => inspect(&input),
    }
}

fn pack(input: &Path, output: &Path, chunk_size: usize) -> anyhow::Result<()> {
    let source = fs::read(input).with_context(|| format!("read {}", input.display()))?;
    let encoded = write_entropy_transport_with_chunk_size(&source, chunk_size)
        .map_err(format_error)
        .context("encode TRNS transport")?;
    let parsed = read_entropy_transport(&encoded)
        .map_err(format_error)
        .context("validate encoded TRNS transport")?;
    if parsed.logical_len() != source.len() {
        bail!("encoded TRNS logical length differs from source");
    }
    atomic_write(output, &encoded)?;
    print_summary(
        source.len(),
        encoded.len(),
        parsed.chunk_count(),
        parsed.chunk_size(),
    );
    Ok(())
}

fn unpack(input: &Path, output: &Path) -> anyhow::Result<()> {
    let encoded = fs::read(input).with_context(|| format!("read {}", input.display()))?;
    let parsed = read_entropy_transport(&encoded)
        .map_err(format_error)
        .context("parse TRNS transport")?;
    let logical = parsed
        .read_all()
        .map_err(format_error)
        .context("decode TRNS transport")?;
    atomic_write(output, &logical)?;
    println!("decoded {} logical bytes", logical.len());
    Ok(())
}

fn inspect(input: &Path) -> anyhow::Result<()> {
    let encoded = fs::read(input).with_context(|| format!("read {}", input.display()))?;
    let parsed = read_entropy_transport(&encoded)
        .map_err(format_error)
        .context("parse TRNS transport")?;
    let huffman_chunks = (0..parsed.chunk_count())
        .filter(|&index| parsed.chunk_info(index).is_ok_and(|info| info.huffman))
        .count();
    println!(
        "transport: TRNS v{}",
        tritium_format::ENTROPY_TRANSPORT_VERSION
    );
    println!("logical_bytes: {}", parsed.logical_len());
    println!("transport_bytes: {}", encoded.len());
    println!("chunk_size: {}", parsed.chunk_size());
    println!("chunks: {}", parsed.chunk_count());
    println!("huffman_chunks: {huffman_chunks}");
    println!("resident_denominator: logical_bytes");
    Ok(())
}

fn print_summary(logical_bytes: usize, transport_bytes: usize, chunks: usize, chunk_size: usize) {
    let ratio = if logical_bytes == 0 {
        1.0
    } else {
        transport_bytes as f64 / logical_bytes as f64
    };
    println!("packed TRNS: logical={logical_bytes} transport={transport_bytes} ratio={ratio:.6}");
    println!("chunks={chunks} chunk_size={chunk_size} resident_denominator=logical_bytes");
}

fn atomic_write(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("transport");
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(".{name}.tmp-{}-{sequence}", std::process::id()));
    let result = (|| -> anyhow::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .with_context(|| format!("create temporary {}", temporary.display()))?;
        file.write_all(bytes)
            .with_context(|| format!("write {}", temporary.display()))?;
        file.sync_all()
            .with_context(|| format!("sync {}", temporary.display()))?;
        drop(file);
        fs::rename(&temporary, path).with_context(|| format!("publish {}", path.display()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn format_error(error: EntropyTransportError) -> anyhow::Error {
    anyhow::anyhow!(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_handles_empty_source() {
        print_summary(0, 36, 0, 65_536);
    }
}
