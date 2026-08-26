//! Golden vectors pinning the DeltaNet activation kernels across backends.
//!
//! `softplus`, `sigmoid` and `silu` exist **twice** in this workspace, character for character:
//! once in `tritium-nn` (`layers/qwen35_deltanet.rs`, the native DeltaNet recurrence) and once in
//! `tritium-onnx` (`onnx_op.rs`, as `deltanet_*`). The duplication has exactly one purpose —
//! cross-backend agreement — and nothing asserted it. Two copies of a numeric kernel whose only
//! job is to agree is a latent parity bug: either can be "cleaned up" and the other will not
//! notice.
//!
//! # Why this is a table and not a shared function (yet)
//!
//! De-duplicating properly means one definition in `tritium-core`, and that is blocked on ADR 0039
//! WS-A for a load-bearing reason: `tritium-core` is `no_std`-able and contains **zero
//! transcendentals**, which is precisely the property `tritium-mcu` cites for CPU↔MCU byte
//! identity. Moving `value.exp().ln_1p()` there today would trade this latent parity bug for a
//! broken byte-identity claim. The other candidate, sharing via `tritium-nn`, needs a new
//! dependency edge because `tritium-onnx` only pulls it behind `qwen-package` while the ONNX ops
//! sit behind `onnx`.
//!
//! So until WS-A supplies an exact/LUT form, this table pins the agreement instead. Delete it when
//! the two copies collapse into one.
//!
//! # What it does and does not guarantee
//!
//! Bits are recorded from the `tritium-nn` implementation on x86-64 Linux. Asserting them from
//! both crates catches **drift between the copies**, which is the actual failure mode. It is *not*
//! a cross-platform bit-identity claim: both sides call the same libm, so both would move together
//! on a platform whose `exp`/`ln_1p` differ. That caveat is the second half of ADR 0039 WS-G —
//! stated here rather than left implied.

/// `(input, softplus_bits, sigmoid_bits, silu_bits)`, as raw `f32` bit patterns so the comparison
/// is exact rather than an epsilon that could hide a real divergence.
///
/// Inputs cover both `softplus` branches and the boundary between them (`> 20.0`), the sign
/// change, and the saturating tails.
pub const DELTANET_ACTIVATION_VECTORS: &[(f32, u32, u32, u32)] = &[
    (-3e1, 0x29d2_b706, 0x29d2_b707, 0xac45_8b97),
    (-8e0, 0x39af_d97b, 0x39af_d1ef, 0xbb2f_d1ef),
    (-1.5e0, 0x3e4e_3f49, 0x3e3a_cdc2, 0xbe8c_1a52),
    (-5e-1, 0x3ef2_ba38, 0x3ec1_4d03, 0xbe41_4d03),
    (0e0, 0x3f31_7218, 0x3f00_0000, 0x0000_0000),
    (5e-1, 0x3f79_5d1c, 0x3f1f_597f, 0x3e9f_597f),
    (1.5e0, 0x3fd9_c7e9, 0x3f51_4c8f, 0x3f9c_f96b),
    (8e0, 0x4100_0160, 0x3f7f_ea06, 0x40ff_ea06),
    // 19.9 and 20.0 both take the `else` branch; in f32 `ln_1p(exp(x))` has already saturated to
    // x by here, so the branch at `> 20.0` is continuous. Pinned so a change to the threshold
    // shows up as a diff rather than as silence.
    (1.99e1, 0x419f_3333, 0x3f80_0000, 0x419f_3333),
    (2e1, 0x41a0_0000, 0x3f80_0000, 0x41a0_0000),
    (2.01e1, 0x41a0_cccd, 0x3f80_0000, 0x41a0_cccd),
    (3e1, 0x41f0_0000, 0x3f80_0000, 0x41f0_0000),
];

/// Assert one implementation triple against [`DELTANET_ACTIVATION_VECTORS`].
///
/// Takes the three functions so each crate can pass its own private copies without exporting them.
///
/// # Panics
/// Panics naming the function, the input, and both bit patterns when any value disagrees.
pub fn assert_deltanet_activations(
    softplus: impl Fn(f32) -> f32,
    sigmoid: impl Fn(f32) -> f32,
    silu: impl Fn(f32) -> f32,
) {
    for &(x, want_softplus, want_sigmoid, want_silu) in DELTANET_ACTIVATION_VECTORS {
        for (name, got, want) in [
            ("softplus", softplus(x).to_bits(), want_softplus),
            ("sigmoid", sigmoid(x).to_bits(), want_sigmoid),
            ("silu", silu(x).to_bits(), want_silu),
        ] {
            assert_eq!(
                got,
                want,
                "{name}({x}) = 0x{got:08x} ({}), expected 0x{want:08x} ({}). The native and ONNX \
                 DeltaNet activations are duplicated on purpose and must agree bit-for-bit; a diff \
                 here means one copy moved. See ADR 0039 WS-G.",
                f32::from_bits(got),
                f32::from_bits(want),
            );
        }
    }
}
