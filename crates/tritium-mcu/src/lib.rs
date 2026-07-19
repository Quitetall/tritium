//! # tritium-mcu
//!
//! A `no_std`, allocation-free executor for the ternary biosignal codec on a $2 MCU (STM32N6
//! Cortex-M55, RP2350 Cortex-M33) — ADR 0030 Tier 3. It runs the codec's ternary **mpGEMM**,
//! **Conv1d**, and **FSQ** through the bit-exact [`tritium_core`] reference oracles, out of a **fixed
//! SRAM [`Arena`]** whose whole budget is declared up front (no global allocator, no heap). The
//! float build LamQuant ships `.bss`-OOM'd at ~94 KB; this model is arena/overlay-aware — reset the
//! arena between streamed windows and the peak is a hard, inspectable number.
//!
//! **Byte-identity.** The reference ops use only `+ − ×` and integer rounding — no `libm`
//! transcendentals, no fused multiply-add — so single-precision IEEE-754 on the Cortex-M FPU is
//! bit-identical to the host, and the FSQ grid uses the explicit round-half-away path. The host
//! conformance test drives the `tritium-testkit` codec vectors through this executor and grades them
//! bit-exact, so CPU↔MCU byte-identity holds by construction. (The `thumbv8m.main-none-eabi*`
//! cross-compile and on-board validation need the embedded toolchain + hardware and are a follow-on;
//! the executor itself is host-compilable and gated now.)
//!
//! A fixed-point (integer-only) path built on `tritium-cpu`'s order-free int8 accumulate is the next
//! step within this crate for the parts of the pipeline that can drop the FPU.
#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

use tritium_core::{
    ConvShape, GemmShape, Trit, TritError, reference_conv1d, reference_fsq, reference_mpgemm,
};

/// A failure inside the MCU executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McuError {
    /// The fixed arena could not satisfy an allocation of `needed` f32 with `remaining` left.
    ArenaOverflow {
        /// f32 elements requested.
        needed: usize,
        /// f32 elements still free in the arena.
        remaining: usize,
    },
    /// A reference op rejected the operand shapes.
    Shape(TritError),
}

/// A fixed-size SRAM scratch arena that bump-allocates zeroed `f32` buffers — the static-budget,
/// overlay-aware memory model a $2 MCU needs. Construct it over a `&'static mut [f32]` carved from a
/// linker section; [`reset`](Self::reset) reclaims it between streamed windows.
pub struct Arena<'buf> {
    buf: &'buf mut [f32],
    used: usize,
    peak: usize,
}

impl core::fmt::Debug for Arena<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Arena")
            .field("used", &self.used)
            .field("peak", &self.peak)
            .field("capacity", &self.buf.len())
            .finish()
    }
}

impl<'buf> Arena<'buf> {
    /// Wrap a fixed backing buffer (e.g. a `static mut [f32; N]` in a dedicated SRAM section).
    #[must_use]
    pub fn new(buf: &'buf mut [f32]) -> Self {
        Self {
            buf,
            used: 0,
            peak: 0,
        }
    }

    /// Total capacity in f32 elements.
    #[must_use]
    pub fn capacity_f32(&self) -> usize {
        self.buf.len()
    }

    /// Currently-allocated f32 elements.
    #[must_use]
    pub fn used_f32(&self) -> usize {
        self.used
    }

    /// High-water mark in f32 elements — the number that must fit the declared SRAM budget.
    #[must_use]
    pub fn peak_f32(&self) -> usize {
        self.peak
    }

    /// Reclaim the whole arena (between windows). Does not change the peak.
    pub fn reset(&mut self) {
        self.used = 0;
    }

    /// Bump-allocate `n` zeroed f32, or [`McuError::ArenaOverflow`] if the budget is exceeded.
    fn alloc(&mut self, n: usize) -> Result<&mut [f32], McuError> {
        let end = self.used.checked_add(n).filter(|&e| e <= self.buf.len());
        let Some(end) = end else {
            return Err(McuError::ArenaOverflow {
                needed: n,
                remaining: self.buf.len() - self.used,
            });
        };
        let slice = &mut self.buf[self.used..end];
        slice.fill(0.0);
        self.used = end;
        if self.used > self.peak {
            self.peak = self.used;
        }
        Ok(slice)
    }
}

/// Ternary mpGEMM `Y[m,n] = scale[n]·Σ_k act[m,k]·W[n,k]` into arena scratch. Byte-identical to the host.
///
/// # Errors
/// [`McuError`] if the arena is exhausted or the shapes disagree.
pub fn mpgemm<'a>(
    arena: &'a mut Arena<'_>,
    act: &[f32],
    weights: &[Trit],
    scale: &[f32],
    shape: GemmShape,
) -> Result<&'a [f32], McuError> {
    let out = arena.alloc(shape.m * shape.n)?;
    reference_mpgemm(act, weights, scale, shape, &mut *out).map_err(McuError::Shape)?;
    Ok(&*out)
}

/// Ternary 1-D convolution into arena scratch. Byte-identical to the host reference.
///
/// # Errors
/// [`McuError`] if the arena is exhausted or the geometry/operands disagree.
pub fn conv1d<'a>(
    arena: &'a mut Arena<'_>,
    x: &[f32],
    weights: &[Trit],
    scale: &[f32],
    shape: ConvShape,
) -> Result<&'a [f32], McuError> {
    let out_len = shape.batch * shape.c_out * shape.l_out();
    let out = arena.alloc(out_len)?;
    reference_conv1d(x, weights, scale, shape, &mut *out).map_err(McuError::Shape)?;
    Ok(&*out)
}

/// FSQ (clamp deploy grid) into arena scratch. Byte-identical to the host reference.
///
/// # Errors
/// [`McuError`] if the arena is exhausted or the shapes disagree.
pub fn fsq<'a>(
    arena: &'a mut Arena<'_>,
    x: &[f32],
    levels: &[u32],
    channels: usize,
    len: usize,
) -> Result<&'a [f32], McuError> {
    let out = arena.alloc(channels * len)?;
    reference_fsq(x, levels, channels, len, &mut *out).map_err(McuError::Shape)?;
    Ok(&*out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_core::Trit;
    use tritium_testkit::{generate_conv_vectors, generate_fsq_vectors, grade_conv, grade_fsq};

    #[test]
    fn mcu_conv1d_passes_conformance() {
        let mut scratch = vec![0.0f32; 4096];
        for v in generate_conv_vectors(0xC0DEC, 24) {
            let trits: Vec<Trit> = v.weights.iter().map(|&w| Trit::from_sign(w)).collect();
            let mut arena = Arena::new(&mut scratch);
            let out = conv1d(&mut arena, &v.activation, &trits, &v.scales, v.shape).unwrap();
            assert!(grade_conv(&v, out), "MCU conv {} failed conformance", v.id);
        }
    }

    #[test]
    fn mcu_fsq_passes_conformance() {
        let mut scratch = vec![0.0f32; 1024];
        for v in generate_fsq_vectors(0xF5, 24) {
            let mut arena = Arena::new(&mut scratch);
            let out = fsq(&mut arena, &v.input, &v.levels, v.channels, v.len).unwrap();
            assert!(grade_fsq(&v, out), "MCU fsq {} failed conformance", v.id);
        }
    }

    #[test]
    fn arena_overflow_is_a_clean_error() {
        let mut small = vec![0.0f32; 4];
        let mut arena = Arena::new(&mut small);
        assert_eq!(
            arena.alloc(5),
            Err(McuError::ArenaOverflow {
                needed: 5,
                remaining: 4
            })
        );
        // A fitting allocation still succeeds and tracks the peak.
        assert!(arena.alloc(3).is_ok());
        assert_eq!(arena.used_f32(), 3);
        assert_eq!(arena.peak_f32(), 3);
    }

    #[test]
    fn reset_reclaims_but_keeps_peak() {
        let mut scratch = vec![0.0f32; 16];
        let mut arena = Arena::new(&mut scratch);
        arena.alloc(10).unwrap();
        arena.reset();
        assert_eq!(arena.used_f32(), 0);
        assert_eq!(
            arena.peak_f32(),
            10,
            "peak is the budget-defining high-water mark"
        );
        arena.alloc(4).unwrap();
        assert_eq!(arena.peak_f32(), 10);
    }
}
