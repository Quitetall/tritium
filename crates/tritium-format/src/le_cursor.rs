//! A little-endian read cursor that errors (never panics, never reads out of bounds) on a short
//! read — the bounds-checked framing discipline shared by the `.tqbin`/`.tqidx` parsers (and the
//! same idiom `salt_bundle.rs` keeps private). Every `take` is checked, so a truncated or crafted
//! buffer yields a typed [`FormatError`] rather than an index panic.

use crate::FormatError;

/// A bounds-checked little-endian cursor over a byte slice.
pub(crate) struct LeCursor<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> LeCursor<'a> {
    pub(crate) fn new(b: &'a [u8]) -> Self {
        Self { b, o: 0 }
    }

    /// Bytes not yet consumed. `o` never exceeds `b.len()` (only `take` advances it, and only
    /// after a bounds check), so this never underflows.
    pub(crate) fn remaining(&self) -> usize {
        self.b.len() - self.o
    }

    /// Take the next `n` bytes, or error if fewer remain. Uses checked addition so a crafted `n`
    /// near `usize::MAX` errors instead of wrapping.
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], FormatError> {
        let end = self.o.checked_add(n).ok_or(FormatError::WrongBlockLen {
            expected: n,
            got: self.remaining(),
        })?;
        if end > self.b.len() {
            return Err(FormatError::WrongBlockLen {
                expected: n,
                got: self.remaining(),
            });
        }
        let s = &self.b[self.o..end];
        self.o = end;
        Ok(s)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, FormatError> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16, FormatError> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, FormatError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, FormatError> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes(
            s.try_into().expect("take(8) yields 8 bytes"),
        ))
    }
}
