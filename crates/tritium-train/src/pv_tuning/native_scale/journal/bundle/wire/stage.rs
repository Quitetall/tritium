//! One preparation and streamed staging path for memory and filesystem callers.
use super::*;

pub(super) struct Prepared {
    pub(super) manifest: Manifest,
    metadata: Vec<u8>,
    checkpoint: Option<Vec<u8>>,
    total: u64,
}
struct Count {
    bytes: u64,
    max: u64,
}
impl Write for Count {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(b.len() as u64)
            .filter(|n| *n <= self.max)
            .ok_or_else(|| io::Error::other("native resume manifest limit"))?;
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(super) fn prepare(
    journal: &mut impl Live,
    limits: NativeScaleResumeBundleLimits,
) -> Result<Prepared> {
    let c = journal.checkpoint_bytes();
    // Admit checkpoint allocation and scratch before snapshot flush/serialization.
    if limits.max_checkpoint_bytes < c
        || limits.max_manifest_bytes == 0
        || limits.max_bundle_bytes == 0
        || add(if journal.streamed() { 0 } else { c }, SCRATCH as u64)? > limits.max_working_bytes
    {
        return Err(NativeScaleResumeBundleError::Limit(
            "native resume bundle byte limit exceeded",
        ));
    }
    journal.check()?;
    let (checkpoint, manifest) = journal.snapshot()?;
    let mut count = Count {
        bytes: 0,
        max: limits.max_manifest_bytes as u64,
    };
    manifest.write(&mut count).map_err(|_| {
        NativeScaleResumeBundleError::Limit("native resume manifest byte limit exceeded")
    })?;
    let total = admit(
        limits,
        count.bytes,
        c,
        manifest.point().journal_prefix_bytes,
        journal.streamed(),
    )?;
    journal.flush()?;
    let mut metadata = bytes(count.bytes)?;
    let mut encoded = io::Cursor::new(metadata.as_mut_slice());
    manifest.write(&mut encoded)?;
    if encoded.position() != count.bytes {
        return Err(invalid("native resume manifest size changed"));
    }
    Ok(Prepared {
        manifest,
        metadata,
        checkpoint,
        total,
    })
}

struct Output<W> {
    source: W,
    hash: blake3::Hasher,
    bytes: u64,
}
impl<W: Write> Output<W> {
    fn write(&mut self, b: &[u8]) -> Result<()> {
        for chunk in b.chunks(SCRATCH) {
            self.source.write_all(chunk)?;
            self.hash.update(chunk);
            self.bytes = add(self.bytes, chunk.len() as u64)?;
        }
        Ok(())
    }
}

fn prefix<O: Write>(journal: &mut impl Live, point: Point, output: &mut Output<O>) -> Result<()> {
    let append = point.journal_prefix_bytes;
    let (source, poisoned) = journal.stream();
    *poisoned = true;
    let position = source.stream_position()?;
    if position != append {
        return Err(invalid("native resume journal append position differs"));
    }
    let mut source_ok = true;
    let result = (|| {
        if let Err(error) = seek(source, SeekFrom::Start(0), 0) {
            source_ok = false;
            return Err(error);
        }
        let mut scratch = [0; SCRATCH];
        let mut hash = blake3::Hasher::new();
        let mut left = append;
        while left > 0 {
            let n = left.min(SCRATCH as u64) as usize;
            if let Err(error) = source.read_exact(&mut scratch[..n]) {
                source_ok = false;
                return Err(error.into());
            }
            hash.update(&scratch[..n]);
            output.write(&scratch[..n])?;
            left -= n as u64;
        }
        if hash.finalize().as_bytes() != point.journal_prefix.as_bytes() {
            source_ok = false;
            return Err(invalid("native resume journal prefix changed"));
        }
        // Journal is live and must still end at the selected complete cursor.
        if let Err(error) = seek(source, SeekFrom::End(0), append) {
            source_ok = false;
            return Err(error);
        }
        Ok(())
    })();
    let restored = seek(source, SeekFrom::Start(append), append);
    *poisoned = !(source_ok && restored.is_ok());
    result?;
    restored
}

// Reuse the exact bounded prefix traversal/restoration/poisoning after output
// effects. Hashing transferred bytes alone misses a source changed just after
// its final read. No second format or full-prefix buffer is introduced.
pub(super) fn verify_prefix(journal: &mut impl Live, point: Point) -> Result<()> {
    journal.check()?;
    prefix(
        journal,
        point,
        &mut Output {
            source: io::sink(),
            hash: blake3::Hasher::new(),
            bytes: 0,
        },
    )?;
    journal.check()
}

fn write<O: Read + Write + Seek>(
    prepared: &Prepared,
    journal: &mut impl Live,
    mut output: O,
) -> Result<(O, BlobId)> {
    seek(&mut output, SeekFrom::End(0), 0)?;
    seek(&mut output, SeekFrom::Start(0), 0)?;
    let mut out = Output {
        source: output,
        hash: blake3::Hasher::new(),
        bytes: 0,
    };
    out.write(&prepared.manifest.kind().magic())?;
    out.write(&(prepared.metadata.len() as u64).to_le_bytes())?;
    out.write(&prepared.manifest.point().checkpoint_bytes.to_le_bytes())?;
    out.write(&prepared.manifest.point().journal_prefix_bytes.to_le_bytes())?;
    out.write(&prepared.metadata)?;
    journal.write_checkpoint(
        prepared.checkpoint.as_deref(),
        &prepared.manifest.point(),
        |b| out.write(b),
    )?;
    prefix(journal, prepared.manifest.point(), &mut out)?;
    let body = out.hash.finalize();
    out.write(body.as_bytes())?;
    if out.bytes != prepared.total {
        return Err(invalid("native resume bundle emitted length differs"));
    }
    out.source.flush()?;
    seek(&mut out.source, SeekFrom::End(0), prepared.total)?;
    verify_prefix(journal, prepared.manifest.point())?;
    Ok((
        out.source,
        BlobId::from_bytes(*out.hash.finalize().as_bytes()),
    ))
}

pub(super) fn stage<O: Read + Write + Seek>(
    prepared: Prepared,
    journal: &mut impl Live,
    output: O,
    limits: NativeScaleResumeBundleLimits,
) -> Result<ReaderCore<O>> {
    let kind = prepared.manifest.kind();
    let point = prepared.manifest.point();
    let (output, id) = write(&prepared, journal, output)?;
    // Do not retain two encoded metadata/checkpoint copies while reopening.
    drop(prepared);
    let opened = if journal.streamed() {
        ReaderCore::open_streamed(output, id, limits, kind)
    } else {
        ReaderCore::open(output, id, limits, kind)
    };
    verify_prefix(journal, point)?;
    opened
}

/// Stage, flush and reopen a self-contained recovery file in an empty owned stream.
/// No publication/fsync is implied; failures can leave a private destination tail.
/// A destination-only failure permits retry if the journal cursor was restored;
/// source read/hash/position uncertainty poisons the live writer.
/// An unwinding stream during prefix copying also leaves it poisoned.
pub(in crate::pv_tuning::native_scale) fn stage_bundle<O: Read + Write + Seek>(
    journal: &mut impl Live,
    output: O,
    limits: NativeScaleResumeBundleLimits,
) -> Result<ReaderCore<O>> {
    let prepared = prepare(journal, limits)?;
    stage(prepared, journal, output, limits)
}
