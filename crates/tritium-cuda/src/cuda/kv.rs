//! KV-cache element dtype (ADR 0020 precision ladder) + env selection
//! (P2a split: move-only from `cuda/mod.rs`).

use tritium_spec::BackendError;

/// KV-cache element type for this process' decode models (ADR 0020 ladder).
/// Selected by `TRITIUM_KV=f32|f16|i8` (legacy `TRITIUM_KV_F16=1` = f16).
/// Every rung below f32 rounds each written K/V once, so outputs are
/// perplexity-gated rather than bit-exact vs the f32 reference.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum KvDtype {
    F32,
    F16,
    /// i8 with per-(token, kv-head, `KV_QGROUP`-dim group) dynamic scales
    /// (absmax/127 at append; rung 2).
    I8,
    /// Ternary KV experiment (rung 3, "KVTQ"): values quantize to
    /// {-s, 0, +s} per group but ride the i8 lattice + scale arena, so only
    /// the APPEND kernels differ from I8.
    T2,
}

impl KvDtype {
    pub(super) fn elem(self) -> usize {
        match self {
            KvDtype::F32 => 4,
            KvDtype::F16 => 2,
            KvDtype::I8 => 1,
            KvDtype::T2 => 1,
        }
    }

    /// Rungs that carry a per-group scale arena (and pass it as the trailing
    /// kernel arg).
    pub(super) fn has_scales(self) -> bool {
        matches!(self, KvDtype::I8 | KvDtype::T2)
    }
}

/// Keep in sync with `KV_QGROUP` in decode.cu (i8 rung group size).
pub(super) const KV_QGROUP: usize = 64;

pub(super) fn kv_dtype_from_env() -> Result<KvDtype, BackendError> {
    // Legacy alias first (still honored; the new var wins if both are set).
    let legacy = match std::env::var("TRITIUM_KV_F16") {
        Err(std::env::VarError::NotPresent) => None,
        Ok(v) if v == "1" => Some(KvDtype::F16),
        Ok(v) if v == "0" || v.is_empty() => None,
        Ok(v) => {
            return Err(BackendError::InvalidInput(format!(
                "TRITIUM_KV_F16={v:?} — use 1 (f16 KV) or 0/unset (f32)"
            )));
        }
        Err(e) => return Err(BackendError::InvalidInput(format!("TRITIUM_KV_F16: {e}"))),
    };
    match std::env::var("TRITIUM_KV") {
        Err(std::env::VarError::NotPresent) => Ok(legacy.unwrap_or(KvDtype::F32)),
        Ok(v) => match v.as_str() {
            "f32" | "" => Ok(KvDtype::F32),
            "f16" => Ok(KvDtype::F16),
            "i8" => Ok(KvDtype::I8),
            "t2" => Ok(KvDtype::T2),
            // Reject loudly: a typo silently running f32 would invalidate
            // whatever comparison the user thought they were making.
            other => Err(BackendError::InvalidInput(format!(
                "TRITIUM_KV={other:?} — use f32, f16, i8 or t2"
            ))),
        },
        Err(e) => Err(BackendError::InvalidInput(format!("TRITIUM_KV: {e}"))),
    }
}
