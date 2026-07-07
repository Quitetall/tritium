//! `tritium repack` — lossless ternary format conversion between GGUF files.
//!
//! Ternary weights are trits × scales regardless of container: I2_S (2 bpw,
//! per-tensor scale), TQ2_0 (2.06 bpw, per-block scale, the GPU compute
//! format) and TQ1_0 (1.69 bpw base-243, per-block scale, ~18% smaller — the
//! interchange/storage format) all encode the same information. This command
//! rewrites every 2-D ternary tensor into the target format and copies
//! everything else (metadata, norms, embeddings) verbatim, so
//! `repack(repack(x)) == x` at the trits level.
//!
//! Ship TQ1_0, run TQ2_0: the model loader unpacks any of the three to trits
//! at load and each backend packs its own native layout, so a repacked file
//! generates bit-identically to its source.

use std::path::Path;

use anyhow::{Context, bail};
use clap::ValueEnum;
use half::f16;
use tritium_core::Trit;
use tritium_format::{
    GGML_TYPE_I2_S, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GgufFile, QK_K, TQ1_0_BLOCK_BYTES,
    TQ2_0_BLOCK_BYTES, TensorOut, pack_tq1_0_row, pack_tq2_0_row, read_gguf, unpack_i2s_tensor,
    unpack_tq1_0_row, unpack_tq2_0_row, write_gguf,
};

/// Target ternary format for `tritium repack`.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RepackTarget {
    /// TQ1_0 — base-243, 1.69 bpw (~18% smaller ternary payload; storage).
    Tq1,
    /// TQ2_0 — base-4, 2.06 bpw (the GPU compute format).
    Tq2,
}

impl RepackTarget {
    fn ggml_type(self) -> u32 {
        match self {
            RepackTarget::Tq1 => GGML_TYPE_TQ1_0,
            RepackTarget::Tq2 => GGML_TYPE_TQ2_0,
        }
    }
    fn block_bytes(self) -> usize {
        match self {
            RepackTarget::Tq1 => TQ1_0_BLOCK_BYTES,
            RepackTarget::Tq2 => TQ2_0_BLOCK_BYTES,
        }
    }
}

/// A tensor's payload span, honoring the I2_S sizing quirk (`n_bytes == 0`
/// from the reader; the payload is `n_elements/4 + 32`).
fn payload<'a>(
    file: &GgufFile,
    bytes: &'a [u8],
    idx: usize,
    n_elements: usize,
) -> anyhow::Result<&'a [u8]> {
    let info = &file.tensors[idx];
    let start = (file.tensor_data_offset + info.offset) as usize;
    let len = if info.ggml_type == GGML_TYPE_I2_S {
        n_elements / 4 + 32
    } else {
        info.n_bytes as usize
    };
    bytes
        .get(start..start + len)
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
    let mut scale_meta: Vec<(String, f32)> = Vec::new();

    for (idx, info) in file.tensors.iter().enumerate() {
        let ternary = matches!(
            info.ggml_type,
            GGML_TYPE_I2_S | GGML_TYPE_TQ1_0 | GGML_TYPE_TQ2_0
        ) && info.dims.len() == 2;
        if !ternary {
            passthrough += 1;
            continue;
        }
        let k_in = info.dims[0] as usize;
        let n_out = info.dims[1] as usize;
        let n_elements = k_in * n_out;
        let p = payload(&file, &bytes, idx, n_elements)?;

        // Unpack to trits + per-block scales (I2_S: replicate the per-tensor
        // scale into every block — all-zero blocks decode to 0 at any scale,
        // so this is exact).
        let nb = k_in.div_ceil(QK_K);
        let mut trits = vec![Trit::ZERO; n_elements];
        let mut scales = vec![vec![f16::ZERO; nb]; n_out];
        match info.ggml_type {
            GGML_TYPE_I2_S => {
                let s = unpack_i2s_tensor(p, n_elements, &mut trits)
                    .map_err(|e| anyhow::anyhow!("{}: {e}", info.name))?;
                let s16 = f16::from_f32(s);
                if f32::from(s16) != s {
                    scale_meta.push((format!("tritium.i2s_scale.{}", info.name), s));
                }
                for row in &mut scales {
                    row.fill(s16);
                }
            }
            t @ (GGML_TYPE_TQ1_0 | GGML_TYPE_TQ2_0) => {
                let bb = if t == GGML_TYPE_TQ1_0 {
                    TQ1_0_BLOCK_BYTES
                } else {
                    TQ2_0_BLOCK_BYTES
                };
                let row_bytes = nb * bb;
                for r in 0..n_out {
                    let row = &p[r * row_bytes..(r + 1) * row_bytes];
                    let out = &mut trits[r * k_in..(r + 1) * k_in];
                    if t == GGML_TYPE_TQ1_0 {
                        unpack_tq1_0_row(row, out, &mut scales[r])
                    } else {
                        unpack_tq2_0_row(row, out, &mut scales[r])
                    }
                    .map_err(|e| anyhow::anyhow!("{}: {e}", info.name))?;
                }
            }
            _ => unreachable!("ternary match guarded above"),
        }

        // Pack every row into the target format, preserving block scales.
        let row_bytes = nb * to.block_bytes();
        let mut out = vec![0u8; n_out * row_bytes];
        for r in 0..n_out {
            let row_trits = &trits[r * k_in..(r + 1) * k_in];
            let dst = &mut out[r * row_bytes..(r + 1) * row_bytes];
            match to {
                RepackTarget::Tq1 => pack_tq1_0_row(row_trits, &scales[r], dst),
                RepackTarget::Tq2 => pack_tq2_0_row(row_trits, &scales[r], dst),
            }
            .map_err(|e| anyhow::anyhow!("{}: {e}", info.name))?;
        }
        arena.push((idx, out));
        converted += 1;
    }

    if converted == 0 {
        bail!("no 2-D ternary tensors (I2_S / TQ1_0 / TQ2_0) found — nothing to repack");
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
            let n_elements = info.dims.iter().product::<u64>() as usize;
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
    for (k, v) in scale_meta {
        metadata.insert(k, tritium_format::GgufValue::F32(v));
    }
    let out_bytes = write_gguf(file.version, &metadata, &tensors).context("serialize GGUF")?;
    std::fs::write(output, &out_bytes).with_context(|| format!("write {}", output.display()))?;
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

#[cfg(test)]
mod tests {
    use super::*;

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
    /// trit + 1), followed by the f32 scale padded to 32 trailer bytes.
    fn pack_i2s_tensor(t: &[Trit], scale: f32) -> Vec<u8> {
        assert_eq!(t.len() % 128, 0);
        let mut out = vec![0u8; t.len() / 4 + 32];
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
        const GGUF: &str =
            "/home/brianklam/.cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf";
        if !Path::new(GGUF).exists() {
            return;
        }
        let bytes = std::fs::read(GGUF).expect("read");
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
                "{name}: f32 {s} (bits {:08x}) -> f16 {} exact={}",
                s.to_bits(),
                f32::from(h),
                f32::from(h) == s
            );
        }
    }

    /// Real-model gate: repack the BitNet gguf to TQ1_0 and prove the loaded
    /// model is IDENTICAL — same prefill logits, bit for bit, on the CPU
    /// backend (identical trits + scale make every downstream op identical by
    /// construction; this catches any packing/eq drift).
    #[test]
    fn repacked_tq1_model_loads_bit_identical() {
        const GGUF: &str =
            "/home/brianklam/.cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf";
        if !Path::new(GGUF).exists() {
            eprintln!("skipping: {GGUF} absent (gated real-model test)");
            return;
        }
        let dir = std::env::temp_dir().join("tritium-repack-model-test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let tq1 = dir.join("bitnet-tq1.gguf");
        run(Path::new(GGUF), &tq1, RepackTarget::Tq1).expect("repack real model");

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
        let mut a = load(Path::new(GGUF));
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
}
