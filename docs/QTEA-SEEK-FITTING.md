# Explicit seek-backed QTEA feedback fitting

`tritium_quantize::qtea_reference::TensorFit::fit_seek_backed` runs the same
`qtea.cpu-tensor.f32.v1` arithmetic as `fit`, but stores the mutable feedback
matrix in a caller-owned empty `Read + Write + Seek` stream. It does not select
this mode in a campaign, change a recipe, register a solver or qualify a model.

```rust,ignore
let caps = SeekBackedFitLimits {
    max_work_bytes: TensorFit::required_seek_work_bytes(rows, columns, config.group_size)?,
    max_scratch_bytes: TensorFit::required_seek_scratch_bytes(rows, columns)?,
};
// Open a new private file on the intended scratch disk, never an existing file.
let mut scratch = std::fs::OpenOptions::new()
    .read(true).write(true).create_new(true).open(&scratch_path)?;
let result = TensorFit::fit_seek_backed(
    &weights, rows, columns, &gram, config, &mut scratch, caps,
);
drop(scratch);
// Discard owned scratch on success AND failure; it is not a resumable checkpoint.
std::fs::remove_file(&scratch_path)?;
let fitted = result?;
```

The caller retains exclusive custody of the stream and owns its disposal. The
module creates/deletes no files. Existing scratch is refused without overwriting
its bytes. Too-small RAM/scratch caps and invalid numerical inputs refuse before
scratch access. Later numerical/I/O errors return no fit, may leave private bytes,
and never retry with another recipe or fall back to resident feedback storage.
Normal short reads/writes are completed exactly; truncated data, inconsistent
seek positions, nonfinite readback, flush errors and a wrong terminal extent refuse.
Flush is not an fsync or power-loss durability guarantee.

Scratch payload is exactly `4*rows*columns` bytes, row-major little-endian f32.
The logical new-live array ceiling is
`32*C*C + 32*R*C + 16*R*G + 64*R + 512*C + 8192` bytes. It conservatively covers
owned output/nested metadata, original working copy, full factors and bounded
group/block/column/I/O row staging. Ordinary `fit` retains its existing ceiling.
These are logical array-element bounds, not allocator/RSS or physical memory
receipts. Caller stream buffers and OS page cache are not counted. A `Cursor<Vec>`
can exercise the interface but does not move feedback memory out of the process.

This is a partial out-of-core fitting path: borrowed full weights/f64 Gram,
original working copy, curvature factors, group parameters, whole-column reductions
and fitted output remain in memory. It does not yet stream source tensors or full
curvature, accelerate factorization on a GPU, or establish large-model feasibility,
speed, source custody, model quality or release readiness. It preserves global
saliency and column reduction order instead of independently fitting row tiles.

Verification includes exact resident/seek-backed fitted and encoded equality over
all 72 pinned tensor fixtures, short/failing I/O, cap boundaries, dead channels,
signed zero and actual bounded file-backed heap requests. The file-backed probe
removes only its newly created scratch. Run with `TMPDIR` on the intended SSD;
do not use a shared model/output file as fitting scratch.
