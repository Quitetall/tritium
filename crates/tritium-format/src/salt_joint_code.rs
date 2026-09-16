//! Canonical Huffman over **joint** SALT symbols — one code word per weight, not per byte.
//!
//! # Why a second entropy coder, when [`crate::write_entropy_transport`] exists
//!
//! The transport Huffman-codes **bytes** of an already-packed artifact. A TQ2_0 byte holds four
//! trits of *one plane* for four *different* weights, so a byte coder sees the wrong symbol
//! boundaries: it can learn that a plane is mostly zero, and it collapses block padding, but it
//! cannot see how one weight's digits relate across planes.
//!
//! That cross-plane structure is where SALT's redundancy lives. The geometric ladder stores a
//! weight as the balanced-ternary expansion of one integer `k ∈ ±(3^T−1)/2`, so the `T` digits of a
//! weight are not independent symbols — they are one symbol with `3^T` values, and its distribution
//! is sharply peaked near zero. Coding that integer directly is the natural unit.
//!
//! Measured on a converted SmolLM2-135M at T=3/g256 (see `tests/salt_joint_code_real.rs`), against
//! the same artifact:
//!
//! | encoding | bpw |
//! |---|---|
//! | dense TQ2_0 bundle as shipped | 7.965 |
//! | byte-level transport (`tritium transport pack`) | 5.715 |
//! | joint-symbol code (this module) | see the test |
//!
//! # Format contract
//!
//! - Symbol for a weight: `Σ_p (trit_p + 1)·3^(T−1−p)`, plane 0 most significant, in `[0, 3^T)`.
//! - Code: canonical Huffman, length-limited to [`MAX_CODE_LEN`], MSB-first bitstream.
//! - Random access: a bit offset is recorded every `block` symbols, so any block decodes alone.
//! - Lossless by contract: decode returns exactly the symbols encoded. Anything else is a defect,
//!   not a rate trade.

use core::fmt;

/// Longest code word emitted. Bounds the decode table at `2^MAX_CODE_LEN` entries.
pub const MAX_CODE_LEN: u8 = 15;
/// Largest plane count a `u16` symbol can hold: `3^10 = 59049 ≤ 65536`.
pub const MAX_JOINT_PLANES: usize = 10;

/// Errors from building or decoding a joint code.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum JointCodeError {
    /// `t` outside `1..=MAX_JOINT_PLANES`.
    UnsupportedPlanes(usize),
    /// A frequency table did not have exactly `3^t` entries.
    AlphabetSize {
        /// `3^t`.
        expected: usize,
        /// Entries actually supplied.
        got: usize,
    },
    /// A symbol at or above `3^t`.
    SymbolOutOfRange {
        /// The offending symbol.
        symbol: u16,
        /// `3^t`.
        alphabet: usize,
    },
    /// A trit outside `{-1, 0, 1}`.
    BadTrit(i8),
    /// Planes of unequal length, or an empty plane set.
    PlaneShape,
    /// A symbol was encoded that the code gives no code word (frequency zero at build time).
    UncodedSymbol(u16),
    /// The bitstream ended mid-symbol, or held a bit pattern that is no code word.
    CorruptStream,
    /// `block` is zero, or a block index was past the end.
    BadBlock,
}

impl fmt::Display for JointCodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlanes(t) => {
                write!(
                    f,
                    "joint code supports 1..={MAX_JOINT_PLANES} planes, got {t}"
                )
            }
            Self::AlphabetSize { expected, got } => {
                write!(
                    f,
                    "frequency table has {got} entries, expected 3^t = {expected}"
                )
            }
            Self::SymbolOutOfRange { symbol, alphabet } => {
                write!(f, "symbol {symbol} is outside the alphabet of {alphabet}")
            }
            Self::BadTrit(v) => write!(f, "trit {v} is not in {{-1, 0, 1}}"),
            Self::PlaneShape => write!(f, "planes must be non-empty and of equal length"),
            Self::UncodedSymbol(s) => write!(f, "symbol {s} has no code word"),
            Self::CorruptStream => write!(f, "bitstream is truncated or holds no valid code word"),
            Self::BadBlock => write!(f, "block size is zero or block index is out of range"),
        }
    }
}

impl std::error::Error for JointCodeError {}

fn alphabet(t: usize) -> Result<usize, JointCodeError> {
    if t == 0 || t > MAX_JOINT_PLANES {
        return Err(JointCodeError::UnsupportedPlanes(t));
    }
    Ok(3usize.pow(t as u32))
}

/// Fold `T` planes of trits into one joint symbol per weight. `planes[0]` is most significant.
///
/// # Errors
/// [`JointCodeError::PlaneShape`] for an empty or ragged plane set, [`JointCodeError::BadTrit`]
/// for a value outside `{-1, 0, 1}`, or [`JointCodeError::UnsupportedPlanes`].
pub fn joint_symbols(planes: &[&[i8]]) -> Result<Vec<u16>, JointCodeError> {
    let t = planes.len();
    alphabet(t)?;
    let n = planes.first().ok_or(JointCodeError::PlaneShape)?.len();
    if planes.iter().any(|p| p.len() != n) {
        return Err(JointCodeError::PlaneShape);
    }
    let mut out = vec![0u16; n];
    for (i, sym) in out.iter_mut().enumerate() {
        let mut s: u32 = 0;
        for plane in planes {
            let trit = plane[i];
            if !(-1..=1).contains(&trit) {
                return Err(JointCodeError::BadTrit(trit));
            }
            s = s * 3 + (trit + 1) as u32;
        }
        *sym = s as u16;
    }
    Ok(out)
}

/// Unfold joint symbols back into `T` planes of trits. Inverse of [`joint_symbols`].
///
/// # Errors
/// [`JointCodeError::UnsupportedPlanes`] or [`JointCodeError::SymbolOutOfRange`].
pub fn planes_from_symbols(symbols: &[u16], t: usize) -> Result<Vec<Vec<i8>>, JointCodeError> {
    let a = alphabet(t)?;
    let mut planes = vec![vec![0i8; symbols.len()]; t];
    for (i, &sym) in symbols.iter().enumerate() {
        if usize::from(sym) >= a {
            return Err(JointCodeError::SymbolOutOfRange {
                symbol: sym,
                alphabet: a,
            });
        }
        let mut s = u32::from(sym);
        for p in (0..t).rev() {
            planes[p][i] = (s % 3) as i8 - 1;
            s /= 3;
        }
    }
    Ok(planes)
}

/// A canonical, length-limited Huffman code over `3^T` joint symbols, with its decode table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JointCode {
    t: usize,
    /// Code length per symbol; `0` means the symbol never occurred and has no code word.
    lengths: Vec<u8>,
    /// Canonical code word per symbol, right-aligned in `lengths[s]` bits.
    codes: Vec<u32>,
    /// Width of the decode table index, `= max(lengths)`.
    table_bits: u8,
    /// `2^table_bits` entries of `(symbol, length)`; length `0` marks an invalid prefix.
    table: Vec<(u16, u8)>,
}

impl JointCode {
    /// Plane count this code is for.
    #[must_use]
    pub const fn planes(&self) -> usize {
        self.t
    }

    /// Code length per symbol, `0` for symbols that never occurred. This is the whole code: a
    /// canonical Huffman code is fully determined by its lengths, so this is what a container
    /// stores.
    #[must_use]
    pub fn lengths(&self) -> &[u8] {
        &self.lengths
    }

    /// Total bits this code spends on `freq`. Divide by the symbol count for the per-weight rate.
    #[must_use]
    pub fn cost_bits(&self, freq: &[u64]) -> u64 {
        freq.iter()
            .zip(&self.lengths)
            .map(|(&f, &l)| f * u64::from(l))
            .sum()
    }

    /// Build the code from symbol frequencies (`3^t` entries).
    ///
    /// # Errors
    /// [`JointCodeError::UnsupportedPlanes`] or [`JointCodeError::AlphabetSize`].
    pub fn build(freq: &[u64], t: usize) -> Result<Self, JointCodeError> {
        let a = alphabet(t)?;
        if freq.len() != a {
            return Err(JointCodeError::AlphabetSize {
                expected: a,
                got: freq.len(),
            });
        }
        let mut lengths = huffman_lengths(freq);
        limit_lengths(&mut lengths, freq, MAX_CODE_LEN);
        Self::from_lengths(lengths, t)
    }

    /// Rebuild a code from stored lengths — what a reader does.
    ///
    /// # Errors
    /// [`JointCodeError::AlphabetSize`] if `lengths` is not `3^t` long, or
    /// [`JointCodeError::CorruptStream`] if the lengths violate the Kraft inequality (they cannot
    /// form a prefix code) or exceed [`MAX_CODE_LEN`].
    pub fn from_lengths(lengths: Vec<u8>, t: usize) -> Result<Self, JointCodeError> {
        let a = alphabet(t)?;
        if lengths.len() != a {
            return Err(JointCodeError::AlphabetSize {
                expected: a,
                got: lengths.len(),
            });
        }
        let max = lengths.iter().copied().max().unwrap_or(0);
        if max > MAX_CODE_LEN {
            return Err(JointCodeError::CorruptStream);
        }
        // Kraft: Σ 2^(max − l) ≤ 2^max over coded symbols. A violation cannot be a prefix code, and
        // a reader must refuse it rather than build a table with overlapping entries.
        if max > 0 {
            let kraft: u64 = lengths
                .iter()
                .filter(|&&l| l > 0)
                .map(|&l| 1u64 << (max - l))
                .sum();
            if kraft > (1u64 << max) {
                return Err(JointCodeError::CorruptStream);
            }
        }

        // Canonical assignment: ascending (length, symbol).
        let mut order: Vec<usize> = (0..a).filter(|&s| lengths[s] > 0).collect();
        order.sort_by_key(|&s| (lengths[s], s));
        let mut codes = vec![0u32; a];
        let mut code: u32 = 0;
        let mut prev_len: u8 = 0;
        for &s in &order {
            let l = lengths[s];
            if prev_len != 0 {
                code = (code + 1) << (l - prev_len);
            }
            codes[s] = code;
            prev_len = l;
        }

        let table_bits = max;
        let mut table = vec![(0u16, 0u8); 1usize << table_bits];
        for &s in &order {
            let l = lengths[s];
            let shift = table_bits - l;
            let first = (codes[s] as usize) << shift;
            for e in table.iter_mut().skip(first).take(1usize << shift) {
                *e = (s as u16, l);
            }
        }
        Ok(Self {
            t,
            lengths,
            codes,
            table_bits,
            table,
        })
    }
}

/// Unrestricted Huffman code lengths. A lone symbol gets length 1 — a length-0 code word cannot be
/// written into a bitstream, and every decoder needs at least one bit per symbol to count them.
fn huffman_lengths(freq: &[u64]) -> Vec<u8> {
    let a = freq.len();
    let live: Vec<usize> = (0..a).filter(|&s| freq[s] > 0).collect();
    let mut lengths = vec![0u8; a];
    match live.len() {
        0 => return lengths,
        1 => {
            lengths[live[0]] = 1;
            return lengths;
        }
        _ => {}
    }
    // Nodes: leaves 0..a, internal nodes appended. Parent index per node.
    let mut weight: Vec<u64> = freq.to_vec();
    let mut parent: Vec<usize> = vec![usize::MAX; a];
    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, usize)>> = live
        .iter()
        .map(|&s| std::cmp::Reverse((freq[s], s)))
        .collect();
    while heap.len() > 1 {
        let std::cmp::Reverse((w1, n1)) = heap.pop().expect("len > 1");
        let std::cmp::Reverse((w2, n2)) = heap.pop().expect("len > 1");
        let id = weight.len();
        weight.push(w1 + w2);
        parent.push(usize::MAX);
        parent[n1] = id;
        parent[n2] = id;
        heap.push(std::cmp::Reverse((w1 + w2, id)));
    }
    for &s in &live {
        let mut depth = 0u32;
        let mut n = s;
        while parent[n] != usize::MAX {
            n = parent[n];
            depth += 1;
        }
        lengths[s] = depth.min(255) as u8;
    }
    lengths
}

/// Clamp lengths to `max_len` and restore the Kraft inequality by lengthening the least-frequent
/// codes that still have room. Not optimal (package-merge is), but always a valid prefix code, and
/// at SALT's alphabet sizes the natural Huffman depth rarely reaches the limit at all.
fn limit_lengths(lengths: &mut [u8], freq: &[u64], max_len: u8) {
    if lengths.iter().all(|&l| l <= max_len) {
        return;
    }
    for l in lengths.iter_mut() {
        if *l > max_len {
            *l = max_len;
        }
    }
    let cap = 1u64 << max_len;
    let mut kraft: u64 = lengths
        .iter()
        .filter(|&&l| l > 0)
        .map(|&l| 1u64 << (max_len - l))
        .sum();
    // Least-frequent first: lengthening them costs the fewest bits.
    let mut order: Vec<usize> = (0..lengths.len()).filter(|&s| lengths[s] > 0).collect();
    order.sort_by_key(|&s| freq[s]);
    while kraft > cap {
        let Some(&s) = order.iter().find(|&&s| lengths[s] < max_len) else {
            break; // unreachable while the alphabet fits in 2^max_len
        };
        kraft -= 1u64 << (max_len - lengths[s] - 1);
        lengths[s] += 1;
    }
}

/// A joint-symbol bitstream with a random-access index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JointStream {
    /// MSB-first packed code words.
    pub bits: Vec<u8>,
    /// Bit offset of the first symbol of each block.
    pub block_offsets: Vec<u64>,
    /// Symbols per block (the last block may be shorter).
    pub block: usize,
    /// Total symbols encoded.
    pub symbols: usize,
}

impl JointStream {
    /// Chargeable size in bits: the payload plus the index a reader needs for random access.
    /// Excludes the code-length table, which is one per model and amortizes to nothing.
    #[must_use]
    pub fn chargeable_bits(&self) -> u64 {
        self.bits.len() as u64 * 8 + self.block_offsets.len() as u64 * 64
    }
}

/// Encode `symbols` with `code`, recording a bit offset every `block` symbols.
///
/// # Errors
/// [`JointCodeError::BadBlock`], [`JointCodeError::SymbolOutOfRange`], or
/// [`JointCodeError::UncodedSymbol`] for a symbol the code was not built to carry.
pub fn encode(
    code: &JointCode,
    symbols: &[u16],
    block: usize,
) -> Result<JointStream, JointCodeError> {
    if block == 0 {
        return Err(JointCodeError::BadBlock);
    }
    let a = code.lengths.len();
    let mut bits: Vec<u8> = Vec::with_capacity(symbols.len() / 2 + 8);
    let mut acc: u64 = 0;
    let mut n_acc: u32 = 0;
    let mut total: u64 = 0;
    let mut block_offsets = Vec::with_capacity(symbols.len().div_ceil(block));
    for (i, &sym) in symbols.iter().enumerate() {
        if i % block == 0 {
            block_offsets.push(total);
        }
        let s = usize::from(sym);
        if s >= a {
            return Err(JointCodeError::SymbolOutOfRange {
                symbol: sym,
                alphabet: a,
            });
        }
        let l = code.lengths[s];
        if l == 0 {
            return Err(JointCodeError::UncodedSymbol(sym));
        }
        acc = (acc << l) | u64::from(code.codes[s]);
        n_acc += u32::from(l);
        total += u64::from(l);
        while n_acc >= 8 {
            n_acc -= 8;
            bits.push((acc >> n_acc) as u8);
        }
    }
    if n_acc > 0 {
        bits.push((acc << (8 - n_acc)) as u8);
    }
    Ok(JointStream {
        bits,
        block_offsets,
        block,
        symbols: symbols.len(),
    })
}

/// Peek `width` bits starting at `bit`, zero-padded past the end of the stream.
fn peek(bits: &[u8], bit: u64, width: u8) -> usize {
    let mut v: usize = 0;
    for k in 0..u64::from(width) {
        let pos = bit + k;
        let byte = (pos / 8) as usize;
        let b = if byte < bits.len() {
            (bits[byte] >> (7 - (pos % 8))) & 1
        } else {
            0
        };
        v = (v << 1) | usize::from(b);
    }
    v
}

fn decode_run(
    code: &JointCode,
    bits: &[u8],
    mut bit: u64,
    count: usize,
) -> Result<Vec<u16>, JointCodeError> {
    let total_bits = bits.len() as u64 * 8;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let (sym, len) = code.table[peek(bits, bit, code.table_bits)];
        if len == 0 || bit + u64::from(len) > total_bits {
            return Err(JointCodeError::CorruptStream);
        }
        out.push(sym);
        bit += u64::from(len);
    }
    Ok(out)
}

/// Decode every symbol in `stream`.
///
/// # Errors
/// [`JointCodeError::CorruptStream`] if the stream is truncated or holds an invalid prefix.
pub fn decode(code: &JointCode, stream: &JointStream) -> Result<Vec<u16>, JointCodeError> {
    decode_run(code, &stream.bits, 0, stream.symbols)
}

/// Decode one block on its own, starting from its recorded bit offset — the random-access path.
///
/// # Errors
/// [`JointCodeError::BadBlock`] for an index past the end, or [`JointCodeError::CorruptStream`].
pub fn decode_block(
    code: &JointCode,
    stream: &JointStream,
    index: usize,
) -> Result<Vec<u16>, JointCodeError> {
    let &start = stream
        .block_offsets
        .get(index)
        .ok_or(JointCodeError::BadBlock)?;
    let first = index * stream.block;
    let count = stream.block.min(stream.symbols - first);
    decode_run(code, &stream.bits, start, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    /// Peaked near the middle of the alphabet, the way SALT's integer `k` is.
    fn peaked(n: usize, t: usize, seed: u64) -> Vec<u16> {
        let a = 3usize.pow(t as u32);
        let mid = (a / 2) as i64;
        let mut r = rng(seed);
        (0..n)
            .map(|_| {
                let spread = (r() % 4 + 1) as i64;
                let off = (r() % (2 * spread as u64 + 1)) as i64 - spread;
                (mid + off).clamp(0, a as i64 - 1) as u16
            })
            .collect()
    }

    fn freq_of(symbols: &[u16], t: usize) -> Vec<u64> {
        let mut f = vec![0u64; 3usize.pow(t as u32)];
        for &s in symbols {
            f[usize::from(s)] += 1;
        }
        f
    }

    fn entropy_bits(freq: &[u64]) -> f64 {
        let n: u64 = freq.iter().sum();
        freq.iter()
            .filter(|&&f| f > 0)
            .map(|&f| {
                let p = f as f64 / n as f64;
                -p * p.log2()
            })
            .sum()
    }

    #[test]
    fn roundtrip_is_exact_across_plane_counts_and_distributions() {
        for t in 1..=6 {
            for (label, symbols) in [
                ("peaked", peaked(10_000, t, 0xA5 + t as u64)),
                ("uniform", {
                    let a = 3u64.pow(t as u32);
                    let mut r = rng(0x77 + t as u64);
                    (0..10_000).map(|_| (r() % a) as u16).collect()
                }),
            ] {
                let freq = freq_of(&symbols, t);
                let code = JointCode::build(&freq, t).unwrap();
                for block in [1usize, 7, 256, 100_000] {
                    let stream = encode(&code, &symbols, block).unwrap();
                    assert_eq!(
                        decode(&code, &stream).unwrap(),
                        symbols,
                        "t={t} {label} block={block}: decode must be exact"
                    );
                }
            }
        }
    }

    /// Every block must decode on its own from its recorded offset — the property that makes the
    /// stream usable without decoding everything before it.
    #[test]
    fn every_block_decodes_independently() {
        let symbols = peaked(5_003, 3, 0x1234);
        let code = JointCode::build(&freq_of(&symbols, 3), 3).unwrap();
        let block = 128;
        let stream = encode(&code, &symbols, block).unwrap();
        for (b, chunk) in symbols.chunks(block).enumerate() {
            assert_eq!(decode_block(&code, &stream, b).unwrap(), chunk, "block {b}");
        }
        assert_eq!(
            decode_block(&code, &stream, symbols.len().div_ceil(block)),
            Err(JointCodeError::BadBlock)
        );
    }

    /// Huffman's optimality bound: mean length in `[H, H+1)`. Below `H` is impossible for a prefix
    /// code; at or above `H+1` means the code is not Huffman.
    #[test]
    fn rate_sits_inside_the_huffman_bound() {
        for t in 2..=5 {
            let symbols = peaked(50_000, t, 0xBEEF + t as u64);
            let freq = freq_of(&symbols, t);
            let code = JointCode::build(&freq, t).unwrap();
            let mean = code.cost_bits(&freq) as f64 / symbols.len() as f64;
            let h = entropy_bits(&freq);
            assert!(
                mean >= h - 1e-9 && mean < h + 1.0,
                "t={t}: mean length {mean:.4} outside [H, H+1) with H={h:.4}"
            );
        }
    }

    #[test]
    fn plane_folding_roundtrips() {
        let mut r = rng(0x5151);
        for t in 1..=MAX_JOINT_PLANES {
            let planes: Vec<Vec<i8>> = (0..t)
                .map(|_| (0..300).map(|_| (r() % 3) as i8 - 1).collect())
                .collect();
            let refs: Vec<&[i8]> = planes.iter().map(Vec::as_slice).collect();
            let symbols = joint_symbols(&refs).unwrap();
            assert!(
                symbols
                    .iter()
                    .all(|&s| usize::from(s) < 3usize.pow(t as u32))
            );
            assert_eq!(planes_from_symbols(&symbols, t).unwrap(), planes, "t={t}");
        }
    }

    /// Plane 0 must be the most significant digit, matching the ladder's `Σ s₀·3^−p` order. A
    /// silent reversal would still round-trip, so it is pinned explicitly.
    #[test]
    fn plane_zero_is_most_significant() {
        let p0: &[i8] = &[1];
        let p1: &[i8] = &[-1];
        // digits (2, 0) in base 3 = 6.
        assert_eq!(joint_symbols(&[p0, p1]).unwrap(), vec![6]);
    }

    #[test]
    fn a_lone_symbol_still_costs_one_bit_and_roundtrips() {
        let symbols = vec![13u16; 1000];
        let code = JointCode::build(&freq_of(&symbols, 3), 3).unwrap();
        assert_eq!(code.lengths()[13], 1);
        let stream = encode(&code, &symbols, 64).unwrap();
        assert_eq!(decode(&code, &stream).unwrap(), symbols);
    }

    /// Fibonacci frequencies force the deepest possible Huffman tree. The limiter must bring it
    /// within `MAX_CODE_LEN`, keep Kraft, and still decode exactly.
    #[test]
    fn length_limiting_holds_on_a_pathological_distribution() {
        let t = 4; // 81 symbols
        let mut freq = vec![0u64; 81];
        let (mut a, mut b) = (1u64, 1u64);
        for f in freq.iter_mut().take(40) {
            *f = a;
            (a, b) = (b, a.saturating_add(b));
        }
        let code = JointCode::build(&freq, t).unwrap();
        let max = code.lengths().iter().copied().max().unwrap();
        assert!(max <= MAX_CODE_LEN, "limiter left a {max}-bit code");
        let kraft: f64 = code
            .lengths()
            .iter()
            .filter(|&&l| l > 0)
            .map(|&l| 2f64.powi(-i32::from(l)))
            .sum();
        assert!(
            kraft <= 1.0 + 1e-12,
            "Kraft sum {kraft} > 1 is not a prefix code"
        );

        let mut symbols = Vec::new();
        for (s, &f) in freq.iter().enumerate() {
            for _ in 0..f.min(50) {
                symbols.push(s as u16);
            }
        }
        let stream = encode(&code, &symbols, 32).unwrap();
        assert_eq!(decode(&code, &stream).unwrap(), symbols);
    }

    /// A reader must refuse lengths that cannot form a prefix code, not build a table from them.
    #[test]
    fn lengths_violating_kraft_are_refused() {
        let mut lengths = vec![0u8; 9];
        lengths[0] = 1;
        lengths[1] = 1;
        lengths[2] = 1; // three 1-bit codes: Kraft sum 1.5
        assert_eq!(
            JointCode::from_lengths(lengths, 2),
            Err(JointCodeError::CorruptStream)
        );
    }

    #[test]
    fn a_stored_code_rebuilds_identically_from_its_lengths() {
        let symbols = peaked(8_000, 3, 0x9);
        let code = JointCode::build(&freq_of(&symbols, 3), 3).unwrap();
        let rebuilt = JointCode::from_lengths(code.lengths().to_vec(), 3).unwrap();
        assert_eq!(rebuilt, code);
    }

    #[test]
    fn misuse_is_refused() {
        assert_eq!(
            JointCode::build(&[1, 2, 3], 3),
            Err(JointCodeError::AlphabetSize {
                expected: 27,
                got: 3
            })
        );
        assert!(matches!(
            JointCode::build(&[0; 1], 0),
            Err(JointCodeError::UnsupportedPlanes(0))
        ));
        let code = JointCode::build(&freq_of(&[0, 0, 1], 1), 1).unwrap();
        assert_eq!(
            encode(&code, &[2], 4),
            Err(JointCodeError::UncodedSymbol(2))
        );
        assert_eq!(encode(&code, &[0], 0), Err(JointCodeError::BadBlock));
        assert_eq!(
            joint_symbols(&[&[2i8][..]]),
            Err(JointCodeError::BadTrit(2))
        );
        assert_eq!(
            joint_symbols(&[&[0i8, 0][..], &[0i8][..]]),
            Err(JointCodeError::PlaneShape)
        );
    }

    #[test]
    fn a_truncated_stream_is_reported_not_misread() {
        let symbols = peaked(2_000, 3, 0x42);
        let code = JointCode::build(&freq_of(&symbols, 3), 3).unwrap();
        let mut stream = encode(&code, &symbols, 2_000).unwrap();
        stream.bits.truncate(stream.bits.len() / 2);
        assert_eq!(decode(&code, &stream), Err(JointCodeError::CorruptStream));
    }
}
