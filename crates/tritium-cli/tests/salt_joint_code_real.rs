//! **Lever 3 — stop overcharging for the symbols. Measured on a real artifact.**
//!
//! Three encodings of the **same** converted bundle, all lossless:
//!
//! - **dense** — the TQ2_0 bundle as shipped. 2 bits per trit (a ternary value needs log2(3) = 1.585,
//!   so the unused fourth code wastes 20% before any statistics), plus whole-block padding: a
//!   576-wide row occupies three 256-trit blocks, 33% of which hold nothing.
//! - **byte transport** — `tritium transport pack`, canonical Huffman over the packed *bytes*. It
//!   collapses the padding and the dead code, but a byte holds four trits of *one plane* across four
//!   *different* weights, so it cannot see how one weight's digits relate across planes.
//! - **joint symbol** — [`tritium_format::salt_joint_code`], one code word per weight over its `3^T`
//!   joint state. Encodes no padding and sees the cross-plane structure directly.
//!
//! The byte transport is the real bar, not the dense file. It already exists and it is strong.
//!
//! Every rate here **includes** the per-plane f16 scales exactly as the container stores them, and
//! the random-access index, so no row wins by quietly dropping side information. The one
//! informational row that does change side information says so in its label.
//!
//! ```text
//! TRITIUM_JOINT_BUNDLE=/path/to/model.tslb \
//!   cargo test -p tritium-cli --release --test salt_joint_code_real -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use half::f16;
use tritium_core::Trit;
use tritium_format::salt_joint_code::{self, JointCode};
use tritium_format::{QK_K, TQ2_0_BLOCK_BYTES, read_salt_bundle, unpack_tq2_0_block};

#[test]
#[ignore = "needs a converted bundle; set TRITIUM_JOINT_BUNDLE"]
fn joint_symbols_against_the_byte_transport_on_a_real_bundle() {
    let Ok(path) = std::env::var("TRITIUM_JOINT_BUNDLE") else {
        eprintln!("skipping: TRITIUM_JOINT_BUNDLE is not set");
        return;
    };
    let path = PathBuf::from(path);
    let bytes = std::fs::read(&path).expect("read bundle");
    let tensors = read_salt_bundle(&bytes).expect("parse bundle");

    // ── Decode every real weight's digits. Padding slots past `k` are skipped: they are not weights.
    let mut t_planes = 0usize;
    let mut params = 0u64;
    let mut scale_bits = 0u64;
    let mut anchor_bits = 0u64;
    let mut symbols: Vec<u16> = Vec::new();
    for tensor in &tensors {
        let k = tensor.k;
        let blocks = k.div_ceil(QK_K);
        for row in &tensor.salt_rows {
            let t = row.planes.len();
            if t_planes == 0 {
                t_planes = t;
            }
            assert_eq!(
                t, t_planes,
                "{}: mixed plane counts; a single joint alphabet assumes uniform T",
                tensor.name
            );
            let mut digits = vec![vec![0i8; k]; t];
            for (p, plane) in row.planes.iter().enumerate() {
                for b in 0..blocks {
                    let blk = &plane[b * TQ2_0_BLOCK_BYTES..(b + 1) * TQ2_0_BLOCK_BYTES];
                    let mut trits = [Trit::ZERO; QK_K];
                    let mut scale = f16::ZERO;
                    unpack_tq2_0_block(blk, &mut trits, &mut scale).expect("unpack block");
                    let start = b * QK_K;
                    for (i, &trit) in trits.iter().enumerate().take(QK_K.min(k - start)) {
                        digits[p][start + i] = i8::from(trit);
                    }
                }
            }
            let refs: Vec<&[i8]> = digits.iter().map(Vec::as_slice).collect();
            symbols.extend(salt_joint_code::joint_symbols(&refs).expect("fold planes"));
            params += k as u64;
            // Exactly what the container stores: one f16 per plane per 256-trit block.
            scale_bits += (blocks * t * 16) as u64;
            // What the ladder actually needs: s_p = s0·3^-p, so one anchor per block determines all.
            anchor_bits += (blocks * 16) as u64;
        }
    }

    let alphabet = 3usize.pow(t_planes as u32);
    let mut freq = vec![0u64; alphabet];
    for &s in &symbols {
        freq[usize::from(s)] += 1;
    }
    let live = freq.iter().filter(|&&f| f > 0).count();
    let entropy: f64 = {
        let n = symbols.len() as f64;
        freq.iter()
            .filter(|&&f| f > 0)
            .map(|&f| {
                let p = f as f64 / n;
                -p * p.log2()
            })
            .sum()
    };
    let code = JointCode::build(&freq, t_planes).expect("build code");
    let mean_len = code.cost_bits(&freq) as f64 / symbols.len() as f64;

    let bpw = |bits: f64| bits / params as f64;
    let dense_bpw = bpw(bytes.len() as f64 * 8.0);
    let transport = tritium_format::write_entropy_transport(&bytes).expect("byte transport");
    let transport_bpw = bpw(transport.len() as f64 * 8.0);

    println!(
        "{} | T={t_planes} | {} tensors, {params} weights, {live}/{alphabet} joint states live\n\
         joint symbol entropy {entropy:.4} b/sym | Huffman mean {mean_len:.4} b/sym \
         ({:+.2}% over entropy)\n",
        path.display(),
        tensors.len(),
        100.0 * (mean_len - entropy) / entropy
    );
    println!(
        "{:<52} {:>9} {:>12} {:>13}",
        "encoding", "bpw", "vs dense", "vs transport"
    );
    println!("{}", "-".repeat(90));
    let row = |label: &str, v: f64| {
        println!(
            "{label:<52} {v:>9.4} {:>11.2}% {:>12.2}%",
            100.0 * (v - dense_bpw) / dense_bpw,
            100.0 * (v - transport_bpw) / transport_bpw
        );
    };
    row("dense TQ2_0 bundle (as shipped)", dense_bpw);
    row("byte transport (tritium transport pack)", transport_bpw);

    let mut best = f64::INFINITY;
    for block in [256usize, 4096, 65_536] {
        let stream = salt_joint_code::encode(&code, &symbols, block).expect("encode");
        // Lossless is the contract; a coder that does not round-trip has not earned a rate.
        let decoded = salt_joint_code::decode(&code, &stream).expect("decode");
        assert_eq!(
            decoded,
            symbols,
            "joint code did not round-trip {} real symbols at block {block}",
            symbols.len()
        );
        let v = bpw(stream.chargeable_bits() as f64 + scale_bits as f64);
        best = best.min(v);
        row(&format!("joint symbol, index every {block} weights"), v);
    }
    let stream = salt_joint_code::encode(&code, &symbols, 4096).expect("encode");
    row(
        "  + one anchor/block (ladder-exact; CONTAINER CHANGE)",
        bpw(stream.chargeable_bits() as f64 + anchor_bits as f64),
    );

    println!(
        "\nSymbol payload alone: {:.4} bpw at entropy, {mean_len:.4} as Huffman. Scales as stored add \
         {:.4} bpw; the ladder needs only {:.4}.",
        entropy,
        bpw(scale_bits as f64),
        bpw(anchor_bits as f64)
    );

    assert!(
        best < transport_bpw,
        "the joint-symbol code ({best:.4} bpw) does not beat the existing byte transport \
         ({transport_bpw:.4}). Then it is not worth a second coder, whatever its entropy says."
    );
}
