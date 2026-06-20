//! `TOPT` checkpoint: byte-exact round-trip, bit-exact resume == uninterrupted run,
//! and never-panic parsing of adversarial bytes (ADR 0007, plan 0008).

mod common;

use tritium_train::checkpoint::{
    CHECKPOINT_MAGIC, CHECKPOINT_VERSION, Checkpoint, CheckpointError, LeafCheckpoint,
    read_checkpoint, write_checkpoint,
};
use tritium_train::optim::{AdamState, AdamW};

fn sample_checkpoint() -> Checkpoint<AdamState> {
    Checkpoint {
        step: 7,
        leaves: vec![
            LeafCheckpoint {
                param: vec![1.0, -2.0, 3.0, 4.0, -5.0, 6.0],
                state: AdamState {
                    m: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    v: vec![0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                },
            },
            LeafCheckpoint {
                param: vec![7.0, 8.0],
                state: AdamState {
                    m: vec![-0.1, -0.2],
                    v: vec![0.001, 0.002],
                },
            },
        ],
    }
}

#[test]
fn checkpoint_write_read_is_byte_and_value_exact() {
    let opt = AdamW::new(1e-3);
    let ckpt = sample_checkpoint();
    let bytes1 = write_checkpoint(&opt, &ckpt);
    let parsed = read_checkpoint(&opt, &bytes1).expect("parse");
    assert_eq!(parsed, ckpt, "parsed checkpoint != original");
    let bytes2 = write_checkpoint(&opt, &parsed);
    assert_eq!(bytes1, bytes2, "write(read(write(x))) not byte-exact");
}

#[test]
fn resume_equals_uninterrupted_run() {
    let opt = AdamW {
        lr: 5e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };
    let d = common::ToyData {
        s_q: common::linear_region_s_q(),
        act: common::seeded(1, common::M * common::K, -1.0, 1.0),
        s: vec![0.7f32, 0.5, 0.6],
        target: common::seeded(2, common::M * common::N, -0.5, 0.5),
    };
    let wf0 = common::seeded(3, common::N * common::K, -1.0, 1.0);
    let b0 = common::seeded(4, common::N, -0.2, 0.2);

    // Run A: 40 steps uninterrupted.
    let mut pa = common::ToyParams {
        wf: wf0.clone(),
        b: b0.clone(),
    };
    let mut sa = common::ToyState::init(&opt);
    for t in 1..=40 {
        common::train_step(t, &opt, &mut pa, &mut sa, &d);
    }

    // Run B: 20 steps, checkpoint, restore into fresh buffers, then 20 more.
    let mut pb = common::ToyParams {
        wf: wf0.clone(),
        b: b0.clone(),
    };
    let mut sb = common::ToyState::init(&opt);
    for t in 1..=20 {
        common::train_step(t, &opt, &mut pb, &mut sb, &d);
    }
    let ckpt = Checkpoint {
        step: 20,
        leaves: vec![
            LeafCheckpoint {
                param: pb.wf.clone(),
                state: sb.wf.clone(),
            },
            LeafCheckpoint {
                param: pb.b.clone(),
                state: sb.b.clone(),
            },
        ],
    };
    let bytes = write_checkpoint(&opt, &ckpt);
    let restored = read_checkpoint(&opt, &bytes).expect("roundtrip");
    assert_eq!(restored.step, 20);

    let mut pb2 = common::ToyParams {
        wf: restored.leaves[0].param.clone(),
        b: restored.leaves[1].param.clone(),
    };
    let mut sb2 = common::ToyState {
        wf: restored.leaves[0].state.clone(),
        b: restored.leaves[1].state.clone(),
    };
    // The next step after a `step`-count of 20 is t = 21.
    for t in (restored.step + 1)..=40 {
        common::train_step(t, &opt, &mut pb2, &mut sb2, &d);
    }

    assert_eq!(pa.wf, pb2.wf, "wf diverged after resume");
    assert_eq!(pa.b, pb2.b, "b diverged after resume");
    assert_eq!(sa.wf, sb2.wf, "wf optimizer state diverged after resume");
    assert_eq!(sa.b, sb2.b, "b optimizer state diverged after resume");
}

// ── adversarial parsing: every malformed buffer errors, none panics ────────────────

fn valid_bytes() -> Vec<u8> {
    write_checkpoint(&AdamW::new(1e-3), &sample_checkpoint())
}

#[test]
fn bad_magic_is_error() {
    let opt = AdamW::new(1e-3);
    let mut bytes = valid_bytes();
    bytes[0] = b'X';
    assert_eq!(
        read_checkpoint(&opt, &bytes),
        Err(CheckpointError::BadMagic)
    );
}

#[test]
fn unsupported_version_is_error() {
    let opt = AdamW::new(1e-3);
    let mut bytes = valid_bytes();
    bytes[4] = 99; // version byte follows the 4 magic bytes
    assert_eq!(
        read_checkpoint(&opt, &bytes),
        Err(CheckpointError::UnsupportedVersion(99))
    );
}

#[test]
fn truncation_at_every_length_is_error_not_panic() {
    let opt = AdamW::new(1e-3);
    let valid = valid_bytes();
    for cut in 0..valid.len() {
        let r = read_checkpoint(&opt, &valid[..cut]);
        assert!(
            matches!(
                r,
                Err(CheckpointError::Truncated { .. }) | Err(CheckpointError::BadMagic)
            ),
            "cut {cut}: expected Truncated/BadMagic, got {r:?}"
        );
    }
    // Sanity: the full buffer parses.
    assert!(read_checkpoint(&opt, &valid).is_ok());
    assert_eq!(&valid[0..4], &CHECKPOINT_MAGIC);
}

#[test]
fn trailing_bytes_is_error() {
    let opt = AdamW::new(1e-3);
    let mut bytes = valid_bytes();
    bytes.push(0);
    assert_eq!(
        read_checkpoint(&opt, &bytes),
        Err(CheckpointError::TrailingBytes(1))
    );
}

/// Assemble a valid header (magic, version, step=0) claiming `leaf_count` leaves, then
/// append the raw per-leaf `len` fields in `lens` with NO payload following.
fn crafted_header(leaf_count: u32, lens: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&CHECKPOINT_MAGIC);
    bytes.push(CHECKPOINT_VERSION);
    bytes.extend_from_slice(&0u64.to_le_bytes()); // step
    bytes.extend_from_slice(&leaf_count.to_le_bytes());
    for &len in lens {
        bytes.extend_from_slice(&len.to_le_bytes());
    }
    bytes
}

#[test]
fn crafted_oversized_len_is_error_not_panic() {
    let opt = AdamW::new(1e-3);
    // (a) u64::MAX/4: `len*4` succeeds but `pos + len*4` would overflow — the never-panic
    // hazard the review caught. (b) 1_000_000: merely exceeds the buffer, and must not
    // pre-allocate 4 MB before erroring. Both must yield Truncated, never panic.
    for len in [u64::MAX / 4, 1_000_000] {
        let bytes = crafted_header(1, &[len]);
        let r = read_checkpoint(&opt, &bytes);
        assert!(
            matches!(r, Err(CheckpointError::Truncated { .. })),
            "len {len}: expected Truncated, got {r:?}"
        );
    }
}

#[test]
fn crafted_huge_leaf_count_is_error_not_panic() {
    let opt = AdamW::new(1e-3);
    // 4 billion leaves claimed with no data: must not pre-allocate 4 G entries, and the
    // first per-leaf read past the buffer must error.
    let bytes = crafted_header(u32::MAX, &[]);
    let r = read_checkpoint(&opt, &bytes);
    assert!(
        matches!(r, Err(CheckpointError::Truncated { .. })),
        "expected Truncated, got {r:?}"
    );
}
