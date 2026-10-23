//! Private bounded pending adjoints, not a new recovery or evidence format.
use std::io::{self, Read, Seek, SeekFrom, Write};
const HEADER: [u8; 8] = *b"TPVG\0\0\0\x01";
const IO: usize = 8192;
pub(super) const HEADROOM: u64 = 16384;
fn invalid(s: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s)
}

#[derive(Clone)]
pub(super) struct Layout {
    owners: Vec<(u64, usize, usize)>,
    pub bytes: u64,
}
impl Layout {
    pub fn new(counts: &[(usize, usize)], max_spool: u64, max_owner: u64) -> io::Result<Self> {
        let mut bytes = 8u64;
        let mut owners = Vec::new();
        owners
            .try_reserve_exact(counts.len())
            .map_err(io::Error::other)?;
        for &(s, w) in counts {
            let count = s
                .checked_add(w)
                .ok_or_else(|| invalid("capture gradient count overflow"))?;
            let owner = (count as u64)
                .checked_mul(8)
                .ok_or_else(|| invalid("capture owner extent overflow"))?;
            if owner.checked_add(HEADROOM).is_none_or(|n| n > max_owner) {
                return Err(invalid("capture owner working cap exceeded"));
            }
            owners.push((bytes, s, w));
            bytes = bytes
                .checked_add(owner)
                .ok_or_else(|| invalid("capture spool extent overflow"))?;
        }
        if bytes > max_spool {
            return Err(invalid("capture spool byte cap exceeded"));
        }
        Ok(Self { owners, bytes })
    }
    pub fn count(&self, i: usize) -> io::Result<usize> {
        let &(_, s, w) = self
            .owners
            .get(i)
            .ok_or_else(|| invalid("capture owner absent"))?;
        Ok(s + w)
    }
}
pub(super) struct Window<R> {
    file: R,
    layout: Layout,
    seen: Vec<bool>,
    pub contributions: u64,
    pub poisoned: bool,
    max_contributions: u64,
}
impl<R: Read + Write + Seek> Window<R> {
    pub fn new(mut file: R, layout: Layout, max_contributions: u64) -> io::Result<Self> {
        if max_contributions == 0 || file.seek(SeekFrom::End(0))? != 0 {
            return Err(invalid("capture output/counter differs"));
        }
        file.write_all(&HEADER)?;
        if layout.bytes > 8 {
            file.seek(SeekFrom::Start(layout.bytes - 1))?;
            file.write_all(&[0])?;
        }
        file.flush()?;
        let mut seen = Vec::new();
        seen.try_reserve_exact(layout.owners.len())
            .map_err(io::Error::other)?;
        seen.resize(layout.owners.len(), false);
        let mut result = Self {
            file,
            layout,
            seen,
            contributions: 0,
            poisoned: false,
            max_contributions,
        };
        result.check()?;
        Ok(result)
    }
    pub fn bytes(&self) -> u64 {
        self.layout.bytes
    }
    fn check(&mut self) -> io::Result<()> {
        if self.file.seek(SeekFrom::End(0))? != self.layout.bytes {
            return Err(invalid("capture spool extent changed"));
        }
        self.file.rewind()?;
        let mut h = [0; 8];
        self.file.read_exact(&mut h)?;
        if h != HEADER {
            return Err(invalid("capture spool header changed"));
        }
        Ok(())
    }
    pub fn admit(&self, i: usize) -> io::Result<()> {
        if self.poisoned {
            return Err(invalid("capture window poisoned; zero_grad required"));
        }
        if self.contributions >= self.max_contributions {
            return Err(invalid("capture contribution cap exceeded"));
        }
        self.layout.count(i)?;
        Ok(())
    }
    pub fn capture(&mut self, i: usize, values: &[f64]) -> io::Result<()> {
        self.admit(i)?;
        self.poisoned = true;
        if values.len() != self.layout.count(i)? || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("capture gradient shape/nonfinite"));
        }
        self.check()?;
        let (offset, _, _) = self.layout.owners[i];
        let mut buffer = [0u8; IO];
        for (ordinal, chunk) in values.chunks(IO / 8).enumerate() {
            let at = offset + (ordinal * IO) as u64;
            let bytes = &mut buffer[..chunk.len() * 8];
            if self.seen[i] {
                self.file.seek(SeekFrom::Start(at))?;
                self.file.read_exact(bytes)?;
            }
            let mut changed = false;
            for (dst, &value) in bytes.as_chunks_mut::<8>().0.iter_mut().zip(chunk) {
                let before = if self.seen[i] {
                    u64::from_le_bytes(*dst)
                } else {
                    0
                };
                let value = if self.seen[i] {
                    f64::from_bits(before) + value
                } else {
                    value
                };
                if !value.is_finite() {
                    return Err(invalid("capture f64 accumulation overflow"));
                }
                changed |= value.to_bits() != before;
                dst.copy_from_slice(&value.to_le_bytes());
            }
            // Preserve initial sparse holes and avoid bit-identical rewrites.
            // Negative zero is not a hole: its bit pattern must still be stored.
            if changed {
                self.file.seek(SeekFrom::Start(at))?;
                self.file.write_all(bytes)?;
            }
        }
        self.file.flush()?;
        self.check()?;
        self.seen[i] = true;
        self.contributions += 1;
        self.poisoned = false;
        Ok(())
    }
    pub fn gradients(
        &mut self,
        i: usize,
        scales: usize,
        weights: usize,
    ) -> io::Result<(Vec<f32>, Vec<f32>)> {
        if self.poisoned {
            return Err(invalid("capture window poisoned; zero_grad required"));
        }
        self.poisoned = true;
        self.check()?;
        let &(offset, s, w) = self
            .layout
            .owners
            .get(i)
            .ok_or_else(|| invalid("capture owner absent"))?;
        if (s, w) != (scales, weights) {
            return Err(invalid("capture current owner geometry differs"));
        }
        let mut a = Vec::new();
        let mut b = Vec::new();
        a.try_reserve_exact(s).map_err(io::Error::other)?;
        b.try_reserve_exact(w).map_err(io::Error::other)?;
        if !self.seen[i] {
            a.resize(s, 0.);
            b.resize(w, 0.);
        } else {
            self.file.seek(SeekFrom::Start(offset))?;
            let mut buffer = [0; IO];
            let mut index = 0;
            while index < s + w {
                let count = (s + w - index).min(IO / 8);
                self.file.read_exact(&mut buffer[..count * 8])?;
                for bytes in buffer[..count * 8].as_chunks::<8>().0 {
                    let value = f64::from_le_bytes(*bytes) as f32;
                    if !value.is_finite() {
                        return Err(invalid("capture accumulated gradient overflowed f32"));
                    }
                    if index < s {
                        a.push(value)
                    } else {
                        b.push(value)
                    }
                    index += 1;
                }
            }
        }
        self.check()?;
        self.poisoned = false;
        Ok((a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    #[test]
    fn ordered_capture_narrows_once_and_unseen_owners_are_zero() {
        let layout = Layout::new(&[(1, 2), (0, 3), (0, 0)], 1000, HEADROOM + 100).unwrap();
        let mut w = Window::new(Cursor::new(Vec::new()), layout, 4).unwrap();
        w.capture(0, &[1., 1e8, 0.125]).unwrap();
        w.capture(0, &[2., 1., -0.25]).unwrap();
        w.capture(0, &[-1., -1e8, 0.5]).unwrap();
        assert_eq!(w.gradients(0, 1, 2).unwrap(), (vec![2.], vec![1., 0.375]));
        assert_eq!(w.gradients(1, 0, 3).unwrap(), (vec![], vec![0.; 3]));
        w.capture(2, &[]).unwrap();
        assert_eq!(w.contributions, 4);
        assert!(w.admit(0).is_err());
        assert_eq!(w.gradients(2, 0, 0).unwrap(), (vec![], vec![]));
    }
    #[test]
    fn cap_shape_nonfinite_and_source_change_refuse_or_poison() {
        assert!(Layout::new(&[(4, 5)], 79, HEADROOM + 100).is_err());
        assert!(Layout::new(&[(4, 5)], 100, HEADROOM + 71).is_err());
        let layout = Layout::new(&[(1, 1)], 100, HEADROOM + 100).unwrap();
        for v in [vec![1.], vec![f64::NAN, 1.]] {
            let mut w = Window::new(Cursor::new(Vec::new()), layout.clone(), 3).unwrap();
            assert!(w.capture(0, &v).is_err());
            assert!(w.poisoned);
            assert!(w.gradients(0, 1, 1).is_err());
        }
        let mut w = Window::new(Cursor::new(Vec::new()), layout, 3).unwrap();
        w.capture(0, &[f64::MAX, 1.]).unwrap();
        assert!(w.gradients(0, 1, 1).is_err());
        assert!(w.poisoned);
    }
    #[test]
    fn sparse_file_extent_and_many_owners_do_not_need_zero_window_buffer() {
        let layout = Layout::new(&vec![(4, 1024); 128], 2_000_000, HEADROOM + 9000).unwrap();
        let file = tempfile::tempfile().unwrap();
        let mut w = Window::new(file, layout, 4).unwrap();
        assert_eq!(w.bytes(), 8 + 128 * 1028 * 8);
        w.capture(127, &vec![0.25; 1028]).unwrap();
        assert_eq!(
            w.gradients(127, 4, 1024).unwrap(),
            (vec![0.25; 4], vec![0.25; 1024])
        );
        assert_eq!(
            w.gradients(0, 4, 1024).unwrap(),
            (vec![0.; 4], vec![0.; 1024])
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let before = w.file.metadata().unwrap().blocks();
            w.capture(0, &vec![0.; 1028]).unwrap();
            assert_eq!(w.file.metadata().unwrap().blocks(), before);
        }
    }
    #[test]
    fn sparse_zero_skipping_preserves_signed_zero_and_real_deltas() {
        let layout = Layout::new(&[(1, 1)], 100, HEADROOM + 100).unwrap();
        let mut w = Window::new(Cursor::new(Vec::new()), layout, 8).unwrap();
        w.capture(0, &[-0., 0.]).unwrap();
        let (s, a) = w.gradients(0, 1, 1).unwrap();
        assert_eq!(s[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(a[0].to_bits(), 0);
        w.capture(0, &[0., 1.]).unwrap();
        let (s, a) = w.gradients(0, 1, 1).unwrap();
        assert_eq!(s[0].to_bits(), 0);
        assert_eq!(a, [1.]);
    }
    struct Fault {
        file: Cursor<Vec<u8>>,
        remaining: usize,
        unwind: bool,
    }
    impl Read for Fault {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            assert!(b.len() <= IO);
            self.file.read(b)
        }
    }
    impl Seek for Fault {
        fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
            self.file.seek(s)
        }
    }
    impl Write for Fault {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            assert!(b.len() <= IO);
            if self.remaining == 0 {
                if self.unwind {
                    panic!("injected gradient write unwind");
                }
                return Err(io::Error::other("injected gradient write refusal"));
            }
            let n = b.len().min(self.remaining);
            self.remaining -= n;
            self.file.write(&b[..n])
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn late_partial_write_error_and_unwind_poison_without_consuming_contribution() {
        for unwind in [false, true] {
            let layout = Layout::new(&[(4, 2048)], 100_000, HEADROOM + 20_000).unwrap();
            let file = Fault {
                file: Cursor::new(Vec::new()),
                remaining: usize::MAX,
                unwind,
            };
            let mut w = Window::new(file, layout, 8).unwrap();
            w.file.remaining = 13;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                w.capture(0, &vec![0.5; 2052])
            }));
            if unwind {
                assert!(result.is_err());
            } else {
                assert!(result.unwrap().is_err());
            }
            assert!(w.poisoned);
            assert_eq!(w.contributions, 0);
            assert!(w.gradients(0, 4, 2048).is_err());
        }
    }
    #[test]
    fn changed_extent_or_header_never_supplies_step_gradients() {
        for header in [false, true] {
            let layout = Layout::new(&[(1, 1)], 100, HEADROOM + 100).unwrap();
            let mut w = Window::new(Cursor::new(Vec::new()), layout, 4).unwrap();
            w.capture(0, &[1., 2.]).unwrap();
            if header {
                w.file.get_mut()[0] ^= 1;
            } else {
                w.file.get_mut().push(0);
            }
            assert!(w.gradients(0, 1, 1).is_err());
            assert!(w.poisoned);
        }
    }
}
