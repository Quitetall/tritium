//! `tritium repack` — weight-value-preserving ternary GGUF conversion.
//!
//! Ternary weights are trits × scales regardless of container: I2_S (2 bpw,
//! per-tensor scale), standard Q2_0 (2.25 bpw, group-64), TQ2_0 (2.06 bpw,
//! group-256, GPU compute) and TQ1_0 (1.69 bpw base-243, group-256, storage)
//! encode the same dequantized weights when scale geometry is compatible. This command
//! rewrites every 2-D ternary tensor into the target format and copies
//! everything else (metadata, norms, embeddings) verbatim. Zero-scale groups are
//! semantic zeros; conversion to G256 canonicalizes their stored codes to zero.
//!
//! I2_S/TQ1_0/TQ2_0 normalize into backend-native layouts at load. Standard Q2_0
//! remains packed with its exact G64 scales in the portable Q2 projection.

use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, bail};
use clap::ValueEnum;
use half::f16;
use tritium_core::Trit;
use tritium_format::{
    GGML_TYPE_I2_S, GGML_TYPE_Q2_0, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GgufFile, I2S_SCALE_BYTES,
    Q2_0_BLOCK_BYTES, Q2_0_GROUP_SIZE, QK_K, TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, TensorOut,
    pack_q2_0_row, pack_tq1_0_row, pack_tq2_0_row, read_gguf, unpack_i2s_tensor, unpack_q2_0_row,
    unpack_tq1_0_row, unpack_tq2_0_row, write_gguf,
};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Target ternary format for `tritium repack`.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RepackTarget {
    /// TQ1_0 — base-243, 1.69 bpw (~18% smaller ternary payload; storage).
    Tq1,
    /// TQ2_0 — base-4, 2.06 bpw (the GPU compute format).
    Tq2,
    /// Standard llama.cpp Q2_0 — base-4, 2.25 bpw, one f16 scale per 64 values.
    Q2,
}

impl RepackTarget {
    fn ggml_type(self) -> u32 {
        match self {
            RepackTarget::Tq1 => GGML_TYPE_TQ1_0,
            RepackTarget::Tq2 => GGML_TYPE_TQ2_0,
            RepackTarget::Q2 => GGML_TYPE_Q2_0,
        }
    }
    fn block_bytes(self) -> usize {
        match self {
            RepackTarget::Tq1 => TQ1_0_BLOCK_BYTES,
            RepackTarget::Tq2 => TQ2_0_BLOCK_BYTES,
            RepackTarget::Q2 => Q2_0_BLOCK_BYTES,
        }
    }

    fn group_size(self) -> usize {
        match self {
            RepackTarget::Tq1 | RepackTarget::Tq2 => QK_K,
            RepackTarget::Q2 => Q2_0_GROUP_SIZE,
        }
    }
}

/// A tensor's payload span, honoring the I2_S sizing quirk (`n_bytes == 0`
/// from the reader; the payload is `n_elements/4 + I2S_SCALE_BYTES`).
fn payload<'a>(
    file: &GgufFile,
    bytes: &'a [u8],
    idx: usize,
    n_elements: usize,
) -> anyhow::Result<&'a [u8]> {
    let info = &file.tensors[idx];
    let start = file
        .tensor_data_offset
        .checked_add(info.offset)
        .context("tensor payload offset overflow")?;
    let start = usize::try_from(start).context("tensor payload offset exceeds usize")?;
    let len = if info.ggml_type == GGML_TYPE_I2_S {
        n_elements
            .checked_div(4)
            .and_then(|packed| packed.checked_add(I2S_SCALE_BYTES))
            .context("I2_S payload length overflow")?
    } else {
        usize::try_from(info.n_bytes).context("tensor payload length exceeds usize")?
    };
    let end = start
        .checked_add(len)
        .context("tensor payload end overflow")?;
    bytes
        .get(start..end)
        .with_context(|| format!("{}: payload out of bounds", info.name))
}

pub(crate) fn run(input: &Path, output: &Path, to: RepackTarget) -> anyhow::Result<()> {
    let bytes = std::fs::read(input).with_context(|| format!("read {}", input.display()))?;
    let file = read_gguf(&bytes).context("parse GGUF")?;

    // Converted payloads live here; TensorOut borrows from either this arena
    // or the source mmap.
    let mut arena: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut passthrough = 0usize;
    let mut converted = 0usize;
    // TQ block scales are f16, but I2_S per-tensor scales are f32 and (for
    // real BitNet checkpoints) NOT f16-representable. The authoritative f32
    // scale rides in metadata so Tritium's loader reproduces the source model
    // BIT-EXACTLY; foreign loaders fall back to the f16 block scales (~1e-4
    // relative, the formats' native precision).
    // Every converted tensor owns this exporter namespace. Record `None` when
    // the target's f16 blocks are exact so stale input metadata is removed.
    let mut scale_meta: Vec<(String, Option<f32>)> = Vec::new();

    for (idx, info) in file.tensors.iter().enumerate() {
        let ternary = matches!(
            info.ggml_type,
            GGML_TYPE_I2_S | GGML_TYPE_Q2_0 | GGML_TYPE_TQ1_0 | GGML_TYPE_TQ2_0
        ) && info.dims.len() == 2;
        if !ternary {
            passthrough += 1;
            continue;
        }
        let k_in = usize::try_from(info.dims[0])
            .with_context(|| format!("{}: K exceeds usize", info.name))?;
        let n_out = usize::try_from(info.dims[1])
            .with_context(|| format!("{}: N exceeds usize", info.name))?;
        if !k_in.is_multiple_of(to.group_size()) {
            // repack writes per-row-padded blocks but the GGUF reader sizes TQ
            // tensors flat (ceil(k*n/256)) — a ragged k would produce a file
            // whose n_bytes under-counts its payload. ggml itself requires
            // ne0 % 256 == 0 for TQ types; refuse rather than emit it.
            bail!(
                "{}: k={k_in} is not a multiple of target {:?} group {}",
                info.name,
                to,
                to.group_size()
            );
        }
        let n_elements = k_in
            .checked_mul(n_out)
            .with_context(|| format!("{}: element count overflow", info.name))?;
        let p = payload(&file, &bytes, idx, n_elements)?;

        // Normalize scale geometry to G64. I2_S replicates its tensor scale;
        // TQ1/TQ2 replicate each G256 scale four times; Q2 is already G64.
        // This makes Q2 output exact and lets coarser targets fail closed when
        // four source groups cannot share one scale.
        let groups64 = k_in.div_ceil(Q2_0_GROUP_SIZE);
        let mut trits = zeroed_vec(n_elements, Trit::ZERO, &format!("{} trits", info.name))?;
        let mut scales64 = Vec::new();
        scales64
            .try_reserve_exact(n_out)
            .with_context(|| format!("{}: allocate {n_out} scale rows", info.name))?;
        for _ in 0..n_out {
            scales64.push(zeroed_vec(
                groups64,
                f16::ZERO,
                &format!("{} G64 scales", info.name),
            )?);
        }
        match info.ggml_type {
            GGML_TYPE_I2_S => {
                let s = unpack_i2s_tensor(p, n_elements, &mut trits)
                    .map_err(|e| anyhow::anyhow!("{}: {e}", info.name))?;
                let s16 = f16::from_f32(s);
                if !s.is_finite() {
                    bail!("{}: I2_S scale is non-finite ({s})", info.name);
                }
                if !s16.is_finite() {
                    bail!("{}: I2_S scale {s} overflows finite f16", info.name);
                }
                if s != 0.0 && s16 == f16::ZERO {
                    bail!(
                        "{}: nonzero I2_S scale {s} underflows to f16 zero",
                        info.name
                    );
                }
                scale_meta.push((
                    format!("tritium.i2s_scale.{}", info.name),
                    (f32::from(s16) != s).then_some(s),
                ));
                for row in &mut scales64 {
                    row.fill(s16);
                }
            }
            GGML_TYPE_Q2_0 => {
                let row_bytes = groups64
                    .checked_mul(Q2_0_BLOCK_BYTES)
                    .with_context(|| format!("{}: Q2_0 row length overflow", info.name))?;
                for r in 0..n_out {
                    unpack_q2_0_row(
                        &p[r * row_bytes..(r + 1) * row_bytes],
                        &mut trits[r * k_in..(r + 1) * k_in],
                        &mut scales64[r],
                    )
                    .map_err(|e| anyhow::anyhow!("{}: {e}", info.name))?;
                }
            }
            t @ (GGML_TYPE_TQ1_0 | GGML_TYPE_TQ2_0) => {
                let bb = if t == GGML_TYPE_TQ1_0 {
                    TQ1_0_BLOCK_BYTES
                } else {
                    TQ2_0_BLOCK_BYTES
                };
                let groups256 = k_in.div_ceil(QK_K);
                let row_bytes = groups256
                    .checked_mul(bb)
                    .with_context(|| format!("{}: TQ row length overflow", info.name))?;
                let mut scales256 =
                    zeroed_vec(groups256, f16::ZERO, &format!("{} G256 scales", info.name))?;
                for r in 0..n_out {
                    let row = &p[r * row_bytes..(r + 1) * row_bytes];
                    let out = &mut trits[r * k_in..(r + 1) * k_in];
                    if t == GGML_TYPE_TQ1_0 {
                        unpack_tq1_0_row(row, out, &mut scales256)
                    } else {
                        unpack_tq2_0_row(row, out, &mut scales256)
                    }
                    .map_err(|e| anyhow::anyhow!("{}: {e}", info.name))?;
                    for (group, &scale) in scales256.iter().enumerate() {
                        scales64[r][group * 4..group * 4 + 4].fill(scale);
                    }
                }
            }
            _ => unreachable!("ternary match guarded above"),
        }
        for (row, scales) in scales64.iter().enumerate() {
            for (group, scale) in scales.iter().enumerate() {
                if !scale.is_finite() {
                    bail!(
                        "{}: row {row} G64 group {group} has non-finite f16 scale bits 0x{:04x}",
                        info.name,
                        scale.to_bits()
                    );
                }
            }
        }
        if info.ggml_type != GGML_TYPE_I2_S {
            let key = format!("tritium.i2s_scale.{}", info.name);
            let rebound = match file.metadata.get(&key) {
                Some(value) => {
                    let exact = value.as_f32().with_context(|| {
                        format!("{}: stale scale metadata {key} is not f32", info.name)
                    })?;
                    if !exact.is_finite() {
                        bail!(
                            "{}: stale scale metadata {key} is non-finite ({exact})",
                            info.name
                        );
                    }
                    let rounded = f16::from_f32(exact);
                    if !rounded.is_finite() || (exact != 0.0 && rounded == f16::ZERO) {
                        bail!(
                            "{}: stale scale metadata {key} value {exact} is not representable as finite nonzero f16",
                            info.name
                        );
                    }
                    let mut saw_nonzero = false;
                    for (row, scales) in scales64.iter().enumerate() {
                        for (group, scale) in scales.iter().enumerate() {
                            if *scale == f16::ZERO {
                                continue;
                            }
                            saw_nonzero = true;
                            if scale.to_bits() != rounded.to_bits() {
                                bail!(
                                    "{}: stale scale metadata {key} rounds to 0x{:04x}, but row {row} G64 group {group} stores 0x{:04x}",
                                    info.name,
                                    rounded.to_bits(),
                                    scale.to_bits()
                                );
                            }
                        }
                    }
                    saw_nonzero.then_some(exact)
                }
                None => None,
            };
            scale_meta.push((key, rebound));
        }

        // Pack every row into target geometry. G256 targets require each four
        // G64 groups to carry one common nonzero scale; zero-scale groups are
        // semantically zero and are canonicalized to zero trits before collapse.
        let target_groups = k_in.div_ceil(to.group_size());
        let row_bytes = target_groups
            .checked_mul(to.block_bytes())
            .with_context(|| format!("{}: target row length overflow", info.name))?;
        let out_len = n_out
            .checked_mul(row_bytes)
            .with_context(|| format!("{}: target payload length overflow", info.name))?;
        let mut out = zeroed_vec(out_len, 0u8, &format!("{} target payload", info.name))?;
        for r in 0..n_out {
            let row_trits = &mut trits[r * k_in..(r + 1) * k_in];
            let dst = &mut out[r * row_bytes..(r + 1) * row_bytes];
            match to {
                RepackTarget::Q2 => pack_q2_0_row(row_trits, &scales64[r], dst),
                RepackTarget::Tq1 | RepackTarget::Tq2 => {
                    let scales256 = collapse_g64_scales(&info.name, row_trits, &scales64[r], k_in)?;
                    if to == RepackTarget::Tq1 {
                        pack_tq1_0_row(row_trits, &scales256, dst)
                    } else {
                        pack_tq2_0_row(row_trits, &scales256, dst)
                    }
                }
            }
            .map_err(|e| anyhow::anyhow!("{}: {e}", info.name))?;
        }
        arena.push((idx, out));
        converted += 1;
    }

    if converted == 0 {
        bail!("no 2-D ternary tensors (I2_S / Q2_0 / TQ1_0 / TQ2_0) found — nothing to repack");
    }

    // Assemble the output tensor table in file order.
    let mut tensors: Vec<TensorOut<'_>> = Vec::with_capacity(file.tensors.len());
    let mut arena_it = arena.iter().peekable();
    for (idx, info) in file.tensors.iter().enumerate() {
        if let Some((ai, data)) = arena_it.peek()
            && *ai == idx
        {
            tensors.push(TensorOut {
                name: info.name.clone(),
                dims: info.dims.clone(),
                ggml_type: to.ggml_type(),
                data,
            });
            arena_it.next();
        } else {
            let n_elements = info
                .dims
                .iter()
                .try_fold(1u64, |total, &dimension| total.checked_mul(dimension))
                .context("passthrough tensor element count overflow")?;
            let n_elements =
                usize::try_from(n_elements).context("passthrough element count exceeds usize")?;
            let p = payload(&file, &bytes, idx, n_elements)?;
            tensors.push(TensorOut {
                name: info.name.clone(),
                dims: info.dims.clone(),
                ggml_type: info.ggml_type,
                data: p,
            });
        }
    }

    let mut metadata = file.metadata.clone();
    for (key, value) in scale_meta {
        metadata.remove(&key);
        if let Some(value) = value {
            metadata.insert(key, tritium_format::GgufValue::F32(value));
        }
    }
    let out_bytes = write_gguf(file.version, &metadata, &tensors).context("serialize GGUF")?;
    publish_atomic_verified(output, &out_bytes)?;
    println!(
        "repacked {} of {} tensors to {:?}: {} -> {} bytes ({:+.1}%)",
        converted,
        converted + passthrough,
        to,
        bytes.len(),
        out_bytes.len(),
        (out_bytes.len() as f64 / bytes.len() as f64 - 1.0) * 100.0,
    );
    Ok(())
}

struct TemporaryOutput {
    path: PathBuf,
    committed: bool,
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn publish_atomic_verified(output: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let parent = match output.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create output directory {}", parent.display()))?;
    let file_name = output
        .file_name()
        .context("repack output path has no file name")?;
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let mut temporary_name = OsString::from(".");
    temporary_name.push(file_name);
    temporary_name.push(format!(".repack-{}-{sequence}.tmp", std::process::id()));
    let mut temporary = TemporaryOutput {
        path: parent.join(temporary_name),
        committed: false,
    };

    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary.path)
        .with_context(|| format!("create temporary output {}", temporary.path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write temporary output {}", temporary.path.display()))?;
    file.sync_all()
        .with_context(|| format!("fsync temporary output {}", temporary.path.display()))?;
    drop(file);

    let expected = blake3::hash(bytes);
    let mut actual = blake3::Hasher::new();
    let mut reopened = File::open(&temporary.path)
        .with_context(|| format!("reopen temporary output {}", temporary.path.display()))?;
    let mut buffer = [0u8; 64 * 1024];
    let mut verified_bytes = 0usize;
    loop {
        let count = reopened
            .read(&mut buffer)
            .with_context(|| format!("verify temporary output {}", temporary.path.display()))?;
        if count == 0 {
            break;
        }
        verified_bytes = verified_bytes
            .checked_add(count)
            .context("verified repack output length overflow")?;
        actual.update(&buffer[..count]);
    }
    if verified_bytes != bytes.len() || actual.finalize() != expected {
        bail!(
            "temporary output {} failed byte-for-byte verification",
            temporary.path.display()
        );
    }

    std::fs::rename(&temporary.path, output).with_context(|| {
        format!(
            "atomically publish {} as {}",
            temporary.path.display(),
            output.display()
        )
    })?;
    temporary.committed = true;
    #[cfg(unix)]
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("fsync output directory {}", parent.display()))?;
    Ok(())
}

fn zeroed_vec<T: Clone>(len: usize, value: T, label: &str) -> anyhow::Result<Vec<T>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(len)
        .with_context(|| format!("allocate {len} values for {label}"))?;
    output.resize(len, value);
    Ok(output)
}

fn collapse_g64_scales(
    name: &str,
    row_trits: &mut [Trit],
    scales64: &[f16],
    k_in: usize,
) -> anyhow::Result<Vec<f16>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(k_in / QK_K)
        .with_context(|| format!("{name}: allocate collapsed G256 scales"))?;
    for (group256, chunk) in scales64.chunks_exact(4).enumerate() {
        let mut selected = None;
        for (within, &scale) in chunk.iter().enumerate() {
            if scale == f16::ZERO {
                let start = group256 * QK_K + within * Q2_0_GROUP_SIZE;
                row_trits[start..start + Q2_0_GROUP_SIZE].fill(Trit::ZERO);
                continue;
            }
            match selected {
                None => selected = Some(scale),
                Some(expected) if expected.to_bits() == scale.to_bits() => {}
                Some(expected) => {
                    bail!(
                        "{name}: Q2_0 G64 scales 0x{:04x} and 0x{:04x} differ inside G256 group {group256}; refusing lossy TQ repack",
                        expected.to_bits(),
                        scale.to_bits()
                    );
                }
            }
        }
        output.push(selected.unwrap_or(f16::ZERO));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Model cache root: override via `TRITIUM_MODEL_DIR`; default `~/.cache/tritium-models`; tests skip cleanly when absent.
    static GGUF: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        let dir = std::env::var("TRITIUM_MODEL_DIR").unwrap_or_else(|_| {
            format!(
                "{}/.cache/tritium-models",
                std::env::var("HOME").unwrap_or_default()
            )
        });
        format!("{dir}/bitnet-2b4t-gguf/ggml-model-i2_s.gguf")
    });

    /// Deterministic trits.
    fn trits(n: usize, seed: u64) -> Vec<Trit> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                Trit::from_i8(((s >> 33) % 3) as i8 - 1).expect("trit")
            })
            .collect()
    }

    /// Pack an I2_S payload: per 32-byte block, byte `gp` holds elements
    /// `[gp, 32+gp, 64+gp, 96+gp]` at bit-pairs `[7:6]..[1:0]` (code =
    /// trit + 1), followed by the little-endian f32 scale trailer.
    fn pack_i2s_tensor(t: &[Trit], scale: f32) -> Vec<u8> {
        assert_eq!(t.len() % 128, 0);
        let mut out = vec![0u8; t.len() / 4 + I2S_SCALE_BYTES];
        for (b, blk) in t.chunks_exact(128).enumerate() {
            for gp in 0..32 {
                let mut byte = 0u8;
                for group in 0..4 {
                    let code = (blk[group * 32 + gp].get() + 1) as u8;
                    byte |= code << (6 - 2 * group);
                }
                out[b * 32 + gp] = byte;
            }
        }
        let off = t.len() / 4;
        out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
        out
    }

    fn synthetic_gguf() -> (Vec<u8>, Vec<Trit>, f32, Vec<f32>) {
        use std::collections::BTreeMap;
        let (n_out, k_in) = (4usize, 512usize);
        let t = trits(n_out * k_in, 7);
        let scale = 0.125f32;
        let packed = pack_i2s_tensor(&t, scale);
        let dense: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let dense_bytes: Vec<u8> = dense.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.name".to_owned(),
            tritium_format::GgufValue::String("repack-test".into()),
        );
        let tensors = vec![
            TensorOut {
                name: "blk.0.w".into(),
                dims: vec![k_in as u64, n_out as u64],
                ggml_type: GGML_TYPE_I2_S,
                data: &packed,
            },
            TensorOut {
                name: "norm".into(),
                dims: vec![8],
                ggml_type: 0, // GGML_TYPE_F32
                data: &dense_bytes,
            },
        ];
        let bytes = write_gguf(3, &meta, &tensors).expect("write");
        (bytes, t, scale, dense)
    }

    /// Diagnostic: are the real model's I2_S f32 scales f16-representable?
    #[test]
    fn probe_i2s_scales_f16_exact() {
        if !Path::new(&*GGUF).exists() {
            return;
        }
        let bytes = std::fs::read(&*GGUF).expect("read");
        let file = read_gguf(&bytes).expect("parse");
        for name in ["blk.0.attn_q.weight", "blk.0.ffn_down.weight"] {
            let info = file.tensor(name).expect("tensor");
            let n: usize = info.dims.iter().product::<u64>() as usize;
            let start = (file.tensor_data_offset + info.offset) as usize;
            let p = &bytes[start..start + n / 4 + 32];
            let mut trits = vec![Trit::ZERO; n];
            let s = unpack_i2s_tensor(p, n, &mut trits).expect("unpack");
            let h = f16::from_f32(s);
            eprintln!(
                "{name}: f32 {s} (bits {:08x}) -> f16 {}",
                s.to_bits(),
                f32::from(h),
            );
            // Load-bearing premise of the tritium.i2s_scale metadata: real
            // BitNet scales are NOT f16-representable. If a future checkpoint
            // makes this exact, the metadata mechanism is dead code — worth
            // knowing, so gate it.
            assert_ne!(
                f32::from(h),
                s,
                "{name}: scale became f16-exact — revisit the metadata path"
            );
        }
    }

    /// Real-model gate: repack the BitNet gguf to TQ1_0 and prove the loaded
    /// model is IDENTICAL — same prefill logits, bit for bit, on the CPU
    /// backend (identical trits + scale make every downstream op identical by
    /// construction; this catches any packing/eq drift).
    #[test]
    #[ignore = "requires the external BitNet 2B4T GGUF; run explicitly for qualification"]
    fn repacked_tq1_model_loads_bit_identical() {
        assert!(
            Path::new(&*GGUF).exists(),
            "real-model gate requires TRITIUM_MODEL_DIR/bitnet-b1.58-2B-4T.gguf at {}",
            *GGUF
        );
        let dir = std::env::temp_dir().join("tritium-repack-model-test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let tq1 = dir.join("bitnet-tq1.gguf");
        run(Path::new(&*GGUF), &tq1, RepackTarget::Tq1).expect("repack real model");

        let load = |path: &Path| {
            let bytes = std::fs::read(path).expect("read gguf");
            let file = read_gguf(&bytes).expect("parse gguf");
            let init = tritium_runtime::BACKENDS
                .iter()
                .find(|e| e.name == "cpu")
                .expect("cpu backend")
                .init;
            let backend = init().expect("init cpu");
            tritium_nn::ModelRunner::load(&file, &bytes, backend).expect("load model")
        };
        let mut a = load(Path::new(&*GGUF));
        let mut b = load(&tq1);
        let tokens = [128000u32, 791, 6864, 315, 9822, 374];
        let positions: Vec<usize> = (0..tokens.len()).collect();
        let la = a.forward(&tokens, &positions).expect("forward i2s");
        let lb = b.forward(&tokens, &positions).expect("forward tq1");
        assert_eq!(la.len(), lb.len());
        let diff = la
            .iter()
            .zip(&lb)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            diff, 0,
            "{diff} logit(s) differ between I2_S and TQ1_0 loads"
        );
    }

    #[test]
    fn repack_i2s_to_tq1_and_back_to_tq2_preserves_trits_and_scales() {
        let (src, want_trits, scale, dense) = synthetic_gguf();
        let dir = std::env::temp_dir().join("tritium-repack-test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let a = dir.join("a.gguf");
        let b = dir.join("b.gguf");
        let c = dir.join("c.gguf");
        std::fs::write(&a, &src).expect("write src");

        run(&a, &b, RepackTarget::Tq1).expect("repack to tq1");
        run(&b, &c, RepackTarget::Tq2).expect("repack tq1 to tq2");

        // TQ1 file is smaller than the TQ2 file.
        let (b_len, c_len) = (
            std::fs::metadata(&b).expect("b").len(),
            std::fs::metadata(&c).expect("c").len(),
        );
        assert!(b_len < c_len, "tq1 {b_len} !< tq2 {c_len}");

        for path in [&b, &c] {
            let bytes = std::fs::read(path).expect("read out");
            let file = read_gguf(&bytes).expect("parse out");
            assert_eq!(
                file.metadata.get("general.name"),
                Some(&tritium_format::GgufValue::String("repack-test".into()))
            );
            let info = file.tensor("blk.0.w").expect("tensor");
            let (k_in, n_out) = (info.dims[0] as usize, info.dims[1] as usize);
            let nb = k_in.div_ceil(QK_K);
            let p = &bytes[(file.tensor_data_offset + info.offset) as usize..];
            let mut got = vec![Trit::ZERO; k_in * n_out];
            let mut scales = vec![f16::ZERO; nb];
            let bb = if info.ggml_type == GGML_TYPE_TQ1_0 {
                TQ1_0_BLOCK_BYTES
            } else {
                TQ2_0_BLOCK_BYTES
            };
            for r in 0..n_out {
                let row = &p[r * nb * bb..(r + 1) * nb * bb];
                let out = &mut got[r * k_in..(r + 1) * k_in];
                if info.ggml_type == GGML_TYPE_TQ1_0 {
                    unpack_tq1_0_row(row, out, &mut scales)
                } else {
                    unpack_tq2_0_row(row, out, &mut scales)
                }
                .expect("unpack");
                for &d in &scales {
                    assert_eq!(f32::from(d), scale);
                }
            }
            assert_eq!(got, want_trits, "trits differ in {}", path.display());

            // The dense tensor survives verbatim.
            let ninfo = file.tensor("norm").expect("norm");
            let np = &bytes[(file.tensor_data_offset + ninfo.offset) as usize
                ..(file.tensor_data_offset + ninfo.offset + ninfo.n_bytes) as usize];
            let got_dense: Vec<f32> = np
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(got_dense, dense);
        }
    }

    #[test]
    fn repack_i2s_to_standard_q2_preserves_model_payloads() {
        let (src, want_trits, scale, dense) = synthetic_gguf();
        let dir = std::env::temp_dir().join("tritium-repack-q2-test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let input = dir.join("input.gguf");
        let output = dir.join("q2.gguf");
        std::fs::write(&input, src).expect("write source");

        run(&input, &output, RepackTarget::Q2).expect("repack to standard Q2_0");
        let bytes = std::fs::read(output).expect("read Q2_0 output");
        let file = read_gguf(&bytes).expect("parse Q2_0 output");
        let info = file.tensor("blk.0.w").expect("Q2_0 tensor");
        assert_eq!(info.ggml_type, tritium_format::GGML_TYPE_Q2_0);
        let (k_in, n_out) = (info.dims[0] as usize, info.dims[1] as usize);
        let blocks = tritium_format::q2_0_num_blocks(k_in);
        let row_bytes = blocks * tritium_format::Q2_0_BLOCK_BYTES;
        let payload_start = (file.tensor_data_offset + info.offset) as usize;
        let payload = &bytes[payload_start..payload_start + n_out * row_bytes];
        let mut got = vec![Trit::ZERO; n_out * k_in];
        let mut scales = vec![f16::ZERO; blocks];
        for row in 0..n_out {
            tritium_format::unpack_q2_0_row(
                &payload[row * row_bytes..(row + 1) * row_bytes],
                &mut got[row * k_in..(row + 1) * k_in],
                &mut scales,
            )
            .expect("unpack Q2_0 row");
            assert!(scales.iter().all(|value| f32::from(*value) == scale));
        }
        assert_eq!(got, want_trits);

        let dense_info = file.tensor("norm").expect("dense passthrough");
        let start = (file.tensor_data_offset + dense_info.offset) as usize;
        let got_dense: Vec<f32> = bytes[start..start + dense_info.n_bytes as usize]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 bytes")))
            .collect();
        assert_eq!(got_dense, dense);
    }

    #[test]
    fn repack_q2_to_tq_refuses_incompatible_g64_scales() {
        let (src, _, _, _) = synthetic_gguf();
        let dir = std::env::temp_dir().join("tritium-repack-q2-collapse-test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let input = dir.join("input.gguf");
        let q2_path = dir.join("q2.gguf");
        let incompatible = dir.join("q2-incompatible.gguf");
        let output = dir.join("tq2.gguf");
        if output.exists() {
            std::fs::remove_file(&output).expect("remove stale output");
        }
        std::fs::write(&input, src).expect("write source");
        run(&input, &q2_path, RepackTarget::Q2).expect("make Q2_0 source");

        let mut bytes = std::fs::read(q2_path).expect("read Q2_0 source");
        let file = read_gguf(&bytes).expect("parse Q2_0 source");
        let info = file.tensor("blk.0.w").expect("Q2_0 tensor");
        let start = (file.tensor_data_offset + info.offset) as usize;
        bytes[start + Q2_0_BLOCK_BYTES..start + Q2_0_BLOCK_BYTES + 2]
            .copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
        std::fs::write(&incompatible, bytes).expect("write incompatible Q2_0");

        let error = run(&incompatible, &output, RepackTarget::Tq2)
            .expect_err("incompatible G64 scales must refuse");
        assert!(
            error.to_string().contains("refusing lossy TQ repack"),
            "unexpected error: {error}"
        );
        assert!(!output.exists(), "failed repack must not publish output");
    }
}
