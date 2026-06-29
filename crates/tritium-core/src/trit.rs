//! The ternary scalar.

use crate::error::TritError;

/// A ternary value constrained to `{-1, 0, +1}`.
///
/// `#[repr(transparent)]` over `i8`, so `&[Trit]` is bit-compatible with `&[i8]`
/// whose elements are all in range — backends can transmute packed buffers freely
/// once the invariant is established. The invariant (`-1 | 0 | 1`) is upheld by
/// every constructor; do not build one by transmuting arbitrary bytes.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Trit(i8);

impl Trit {
    /// The `-1` state.
    pub const NEG: Trit = Trit(-1);
    /// The `0` state (pruned weight; contributes nothing to a dot product).
    pub const ZERO: Trit = Trit(0);
    /// The `+1` state.
    pub const POS: Trit = Trit(1);

    /// Construct from a raw `i8`, validating the `{-1, 0, 1}` invariant.
    #[inline]
    pub const fn from_i8(v: i8) -> Result<Self, TritError> {
        match v {
            -1..=1 => Ok(Trit(v)),
            other => Err(TritError::OutOfRange(other as i32)),
        }
    }

    /// Collapse any integer to its sign: `> 0 → +1`, `< 0 → -1`, `0 → 0`.
    ///
    /// This is the quantization primitive — an already-thresholded weight maps
    /// onto the nearest ternary state by sign.
    #[inline]
    pub const fn from_sign(v: i8) -> Self {
        if v > 0 {
            Trit(1)
        } else if v < 0 {
            Trit(-1)
        } else {
            Trit(0)
        }
    }

    /// The underlying `i8` in `{-1, 0, 1}`.
    #[inline]
    pub const fn get(self) -> i8 {
        self.0
    }

    /// As `f32` for reference arithmetic.
    #[inline]
    pub const fn to_f32(self) -> f32 {
        self.0 as f32
    }

    /// True for the pruned state — backends skip these in the inner loop.
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }
}

impl Default for Trit {
    #[inline]
    fn default() -> Self {
        Trit::ZERO
    }
}

impl core::fmt::Debug for Trit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Trit({})", self.0)
    }
}

impl core::fmt::Display for Trit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<i8> for Trit {
    type Error = TritError;
    #[inline]
    fn try_from(v: i8) -> Result<Self, Self::Error> {
        Trit::from_i8(v)
    }
}

impl From<Trit> for i8 {
    #[inline]
    fn from(t: Trit) -> i8 {
        t.0
    }
}

impl From<Trit> for f32 {
    #[inline]
    fn from(t: Trit) -> f32 {
        t.to_f32()
    }
}
