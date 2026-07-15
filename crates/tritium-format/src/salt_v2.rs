//! Zero-point-free physical codec primitives for SALT V2 ternary planes.

use core::fmt;

use tritium_core::Trit;

/// Number of logical D2 trits carried by one physical byte.
pub const D2_TRITS_PER_BYTE: usize = 4;

/// Number of logical B3 radix digits carried by one physical byte.
pub const B3_TRITS_PER_BYTE: usize = 5;

/// Number of canonical five-trit B3 byte codes (`3^5`).
pub const B3_CODE_COUNT: u16 = 243;

/// Number of logical trits in one S34 structured group.
pub const S34_TRITS_PER_GROUP: usize = 4;

/// Number of physical bits in one S34 structured group.
pub const S34_BITS_PER_GROUP: usize = 5;

/// Physical SALT V2 ternary codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaltV2Codec {
    /// Dense, linearly aligned two-bit trits (`code = trit + 1`).
    D2,
    /// Dense radix-3 packing, five trits per byte.
    B3,
    /// Structured 3:4 ternary: one zero and three signs in every four trits.
    S34,
}

/// Exact accounting for one codec payload, excluding scales and containers.
///
/// Canonical padding trits are encoded as semantic zero. `encoded_bits` includes
/// complete D2/B3 codec units, while `canonical_padding_bits` records terminal
/// byte bits that are outside the codec bitstream (used by S34).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalByteLedger {
    /// Codec whose payload is described.
    pub codec: SaltV2Codec,
    /// Number of caller-visible trits.
    pub logical_trits: usize,
    /// Bits occupied by complete codec units before terminal byte padding.
    pub encoded_bits: usize,
    /// Bytes occupied by the physical payload.
    pub physical_bytes: usize,
    /// Semantic-zero trit slots added to complete a codec unit.
    pub canonical_padding_trits: usize,
    /// Zero bits added after the final codec unit to complete a byte.
    pub canonical_padding_bits: u8,
}

/// Failure to encode or canonically decode a SALT V2 physical payload.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaltV2CodecError {
    /// The requested logical length cannot be represented by the byte ledger.
    LengthOverflow {
        /// Codec being accounted.
        codec: SaltV2Codec,
        /// Requested logical trit count.
        logical_trits: usize,
    },
    /// The physical payload did not have the one canonical length.
    WrongPackedLength {
        /// Codec being decoded.
        codec: SaltV2Codec,
        /// Canonical payload length.
        expected: usize,
        /// Supplied payload length.
        got: usize,
    },
    /// A D2 slot used the reserved two-bit code `0b11`.
    InvalidD2Code {
        /// Physical trit-slot index.
        trit_index: usize,
        /// Invalid two-bit code.
        code: u8,
    },
    /// A B3 byte was outside the canonical `0..3^5` code domain.
    InvalidB3Code {
        /// Byte index in the physical payload.
        byte_index: usize,
        /// Invalid byte code.
        code: u8,
    },
    /// An unused trit slot was not canonically encoded as semantic zero.
    NonCanonicalTritPadding {
        /// Codec carrying the padding.
        codec: SaltV2Codec,
        /// Physical trit-slot index.
        trit_index: usize,
        /// Noncanonical code found in the slot.
        code: u8,
    },
    /// S34 requires complete groups of four logical trits.
    S34TritCountNotMultipleOfFour {
        /// Supplied logical trit count.
        logical_trits: usize,
    },
    /// An S34 source group did not contain exactly one zero.
    S34ZeroCount {
        /// Four-trit group index.
        group_index: usize,
        /// Number of zeros found in the group.
        zero_count: usize,
    },
    /// Bits outside the final S34 code were not canonically zero.
    NonCanonicalBitPadding {
        /// Codec carrying the padding.
        codec: SaltV2Codec,
        /// Byte containing the terminal padding.
        byte_index: usize,
        /// Nonzero padding bits, already masked from the byte.
        bits: u8,
    },
}

impl fmt::Display for SaltV2CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthOverflow {
                codec,
                logical_trits,
            } => write!(
                f,
                "{codec:?} payload length overflows for {logical_trits} logical trits"
            ),
            Self::WrongPackedLength {
                codec,
                expected,
                got,
            } => write!(
                f,
                "wrong {codec:?} payload length: expected {expected} bytes, got {got}"
            ),
            Self::InvalidD2Code { trit_index, code } => write!(
                f,
                "invalid D2 code {code:#04b} at physical trit slot {trit_index}"
            ),
            Self::InvalidB3Code { byte_index, code } => {
                write!(f, "invalid B3 code {code} at byte {byte_index}")
            }
            Self::NonCanonicalTritPadding {
                codec,
                trit_index,
                code,
            } => write!(
                f,
                "noncanonical {codec:?} padding code {code} at physical trit slot {trit_index}"
            ),
            Self::S34TritCountNotMultipleOfFour { logical_trits } => write!(
                f,
                "S34 requires a multiple of four trits, got {logical_trits}"
            ),
            Self::S34ZeroCount {
                group_index,
                zero_count,
            } => write!(
                f,
                "S34 group {group_index} requires exactly one zero, got {zero_count}"
            ),
            Self::NonCanonicalBitPadding {
                codec,
                byte_index,
                bits,
            } => write!(
                f,
                "noncanonical {codec:?} terminal padding bits {bits:#010b} at byte {byte_index}"
            ),
        }
    }
}

impl core::error::Error for SaltV2CodecError {}

impl SaltV2Codec {
    /// Return the exact codec-only physical byte ledger for `logical_trits`.
    ///
    /// # Errors
    /// Returns [`SaltV2CodecError::LengthOverflow`] if the encoded bit count
    /// cannot fit in `usize`.
    pub fn ledger(self, logical_trits: usize) -> Result<PhysicalByteLedger, SaltV2CodecError> {
        match self {
            Self::D2 => {
                let physical_bytes = logical_trits.div_ceil(D2_TRITS_PER_BYTE);
                let encoded_bits =
                    physical_bytes
                        .checked_mul(8)
                        .ok_or(SaltV2CodecError::LengthOverflow {
                            codec: self,
                            logical_trits,
                        })?;
                let remainder = logical_trits % D2_TRITS_PER_BYTE;
                Ok(PhysicalByteLedger {
                    codec: self,
                    logical_trits,
                    encoded_bits,
                    physical_bytes,
                    canonical_padding_trits: if remainder == 0 {
                        0
                    } else {
                        D2_TRITS_PER_BYTE - remainder
                    },
                    canonical_padding_bits: 0,
                })
            }
            Self::B3 => {
                let physical_bytes = logical_trits.div_ceil(B3_TRITS_PER_BYTE);
                let encoded_bits =
                    physical_bytes
                        .checked_mul(8)
                        .ok_or(SaltV2CodecError::LengthOverflow {
                            codec: self,
                            logical_trits,
                        })?;
                let remainder = logical_trits % B3_TRITS_PER_BYTE;
                Ok(PhysicalByteLedger {
                    codec: self,
                    logical_trits,
                    encoded_bits,
                    physical_bytes,
                    canonical_padding_trits: if remainder == 0 {
                        0
                    } else {
                        B3_TRITS_PER_BYTE - remainder
                    },
                    canonical_padding_bits: 0,
                })
            }
            Self::S34 => {
                if !logical_trits.is_multiple_of(S34_TRITS_PER_GROUP) {
                    return Err(SaltV2CodecError::S34TritCountNotMultipleOfFour { logical_trits });
                }
                let groups = logical_trits / S34_TRITS_PER_GROUP;
                let encoded_bits = groups.checked_mul(S34_BITS_PER_GROUP).ok_or(
                    SaltV2CodecError::LengthOverflow {
                        codec: self,
                        logical_trits,
                    },
                )?;
                let physical_bytes = encoded_bits.div_ceil(8);
                let canonical_padding_bits = ((8 - encoded_bits % 8) % 8) as u8;
                Ok(PhysicalByteLedger {
                    codec: self,
                    logical_trits,
                    encoded_bits,
                    physical_bytes,
                    canonical_padding_trits: 0,
                    canonical_padding_bits,
                })
            }
        }
    }
}

/// Pack trits contiguously as D2, four per byte and least-significant slot first.
///
/// D2 stores `code = trit + 1`, hence `00 = -1`, `01 = 0`, `10 = +1`, and
/// reserves `11`. A partial final byte is padded with zero-trit code `01`.
/// No scale or affine zero point is stored.
///
/// # Errors
/// Returns [`SaltV2CodecError::LengthOverflow`] if the payload size cannot be
/// represented by `usize`.
pub fn pack_d2(trits: &[Trit]) -> Result<Vec<u8>, SaltV2CodecError> {
    let ledger = SaltV2Codec::D2.ledger(trits.len())?;
    let mut packed = vec![0x55; ledger.physical_bytes];
    for (index, trit) in trits.iter().enumerate() {
        let shift = 2 * (index % D2_TRITS_PER_BYTE);
        let code = (trit.get() + 1) as u8;
        packed[index / D2_TRITS_PER_BYTE] &= !(0b11 << shift);
        packed[index / D2_TRITS_PER_BYTE] |= code << shift;
    }
    Ok(packed)
}

/// Canonically decode a contiguous D2 payload.
///
/// # Errors
/// Returns a typed error for an overflowing count, a non-exact byte length, a
/// reserved `11` code, or a partial byte not padded with zero-trit code `01`.
pub fn unpack_d2(packed: &[u8], logical_trits: usize) -> Result<Vec<Trit>, SaltV2CodecError> {
    let ledger = SaltV2Codec::D2.ledger(logical_trits)?;
    require_exact_length(SaltV2Codec::D2, packed, ledger.physical_bytes)?;

    let physical_trits = ledger.physical_bytes * D2_TRITS_PER_BYTE;
    let mut trits = Vec::with_capacity(logical_trits);
    for index in 0..physical_trits {
        let code = (packed[index / D2_TRITS_PER_BYTE] >> (2 * (index % D2_TRITS_PER_BYTE))) & 0b11;
        if code == 0b11 {
            return Err(SaltV2CodecError::InvalidD2Code {
                trit_index: index,
                code,
            });
        }
        if index >= logical_trits {
            if code != 1 {
                return Err(SaltV2CodecError::NonCanonicalTritPadding {
                    codec: SaltV2Codec::D2,
                    trit_index: index,
                    code,
                });
            }
            continue;
        }
        trits.push(Trit::from_i8(code as i8 - 1).expect("validated D2 code"));
    }
    Ok(trits)
}

/// Pack trits as B3, five little-endian radix-3 digits per byte.
///
/// Each digit is `trit + 1`, and the first trit occupies the least-significant
/// radix position. A partial final byte is completed with semantic-zero digits
/// (`1`). No scale or affine zero point is stored.
///
/// # Errors
/// Returns [`SaltV2CodecError::LengthOverflow`] if the payload size cannot be
/// represented by `usize`.
pub fn pack_b3(trits: &[Trit]) -> Result<Vec<u8>, SaltV2CodecError> {
    let ledger = SaltV2Codec::B3.ledger(trits.len())?;
    let mut packed = Vec::with_capacity(ledger.physical_bytes);
    for byte_index in 0..ledger.physical_bytes {
        let mut code = 0u16;
        let mut place = 1u16;
        for slot in 0..B3_TRITS_PER_BYTE {
            let trit_index = byte_index * B3_TRITS_PER_BYTE + slot;
            let digit = trits
                .get(trit_index)
                .map_or(1u16, |trit| (trit.get() + 1) as u16);
            code += digit * place;
            place *= 3;
        }
        debug_assert!(code < B3_CODE_COUNT);
        packed.push(code as u8);
    }
    Ok(packed)
}

/// Canonically decode a five-trit-per-byte B3 payload.
///
/// # Errors
/// Returns a typed error for an overflowing count, a non-exact byte length, a
/// byte code at or above `3^5`, or a tail radix digit that is not semantic zero.
pub fn unpack_b3(packed: &[u8], logical_trits: usize) -> Result<Vec<Trit>, SaltV2CodecError> {
    let ledger = SaltV2Codec::B3.ledger(logical_trits)?;
    require_exact_length(SaltV2Codec::B3, packed, ledger.physical_bytes)?;

    let mut trits = Vec::with_capacity(logical_trits);
    for (byte_index, &byte) in packed.iter().enumerate() {
        if u16::from(byte) >= B3_CODE_COUNT {
            return Err(SaltV2CodecError::InvalidB3Code {
                byte_index,
                code: byte,
            });
        }
        let mut code = byte;
        for slot in 0..B3_TRITS_PER_BYTE {
            let digit = code % 3;
            code /= 3;
            let trit_index = byte_index * B3_TRITS_PER_BYTE + slot;
            if trit_index >= logical_trits {
                if digit != 1 {
                    return Err(SaltV2CodecError::NonCanonicalTritPadding {
                        codec: SaltV2Codec::B3,
                        trit_index,
                        code: digit,
                    });
                }
                continue;
            }
            trits.push(Trit::from_i8(digit as i8 - 1).expect("validated B3 digit"));
        }
    }
    Ok(trits)
}

/// Pack structured 3:4 trits as one five-bit code per four-trit group.
///
/// Code bits `0..2` hold the zero position. Bits `2..5` hold the signs of the
/// three nonzero trits in increasing trit-position order (`0 = -1`, `1 = +1`).
/// Codes form a contiguous little-endian bitstream, and terminal byte padding is
/// zero. No scale or affine zero point is stored.
///
/// # Errors
/// Returns a typed error if the length is not divisible by four, size accounting
/// overflows, or any group does not contain exactly one zero.
pub fn pack_s34(trits: &[Trit]) -> Result<Vec<u8>, SaltV2CodecError> {
    let ledger = SaltV2Codec::S34.ledger(trits.len())?;
    let mut packed = vec![0u8; ledger.physical_bytes];
    for (group_index, group) in trits.chunks_exact(S34_TRITS_PER_GROUP).enumerate() {
        let zero_count = group.iter().filter(|trit| trit.is_zero()).count();
        if zero_count != 1 {
            return Err(SaltV2CodecError::S34ZeroCount {
                group_index,
                zero_count,
            });
        }
        let zero_index = group
            .iter()
            .position(|trit| trit.is_zero())
            .expect("validated one-zero S34 group");
        let mut code = zero_index as u8;
        let mut sign_index = 0usize;
        for trit in group {
            if trit.is_zero() {
                continue;
            }
            if *trit == Trit::POS {
                code |= 1 << (2 + sign_index);
            }
            sign_index += 1;
        }
        write_s34_code(&mut packed, group_index, code);
    }
    Ok(packed)
}

/// Canonically decode a structured 3:4 five-bit payload.
///
/// Every five-bit value is a valid S34 group because the two low bits select one
/// of four zero positions and the remaining bits select three signs.
///
/// # Errors
/// Returns a typed error for an invalid trit count, overflowing size accounting,
/// a non-exact byte length, or nonzero terminal byte padding.
pub fn unpack_s34(packed: &[u8], logical_trits: usize) -> Result<Vec<Trit>, SaltV2CodecError> {
    let ledger = SaltV2Codec::S34.ledger(logical_trits)?;
    require_exact_length(SaltV2Codec::S34, packed, ledger.physical_bytes)?;
    validate_s34_bit_padding(packed, &ledger)?;

    let groups = logical_trits / S34_TRITS_PER_GROUP;
    let mut trits = Vec::with_capacity(logical_trits);
    for group_index in 0..groups {
        let code = read_s34_code(packed, group_index);
        let zero_index = usize::from(code & 0b11);
        let mut sign_index = 0usize;
        for trit_index in 0..S34_TRITS_PER_GROUP {
            if trit_index == zero_index {
                trits.push(Trit::ZERO);
                continue;
            }
            let positive = code & (1 << (2 + sign_index)) != 0;
            trits.push(if positive { Trit::POS } else { Trit::NEG });
            sign_index += 1;
        }
    }
    Ok(trits)
}

fn write_s34_code(packed: &mut [u8], group_index: usize, code: u8) {
    debug_assert!(code < (1 << S34_BITS_PER_GROUP));
    let bit_index = group_index * S34_BITS_PER_GROUP;
    let byte_index = bit_index / 8;
    let shift = bit_index % 8;
    let shifted = u16::from(code) << shift;
    packed[byte_index] |= shifted as u8;
    if shift + S34_BITS_PER_GROUP > 8 {
        packed[byte_index + 1] |= (shifted >> 8) as u8;
    }
}

fn read_s34_code(packed: &[u8], group_index: usize) -> u8 {
    let bit_index = group_index * S34_BITS_PER_GROUP;
    let byte_index = bit_index / 8;
    let shift = bit_index % 8;
    let word = u16::from(packed[byte_index])
        | (packed.get(byte_index + 1).copied().map_or(0, u16::from) << 8);
    ((word >> shift) & 0b1_1111) as u8
}

fn validate_s34_bit_padding(
    packed: &[u8],
    ledger: &PhysicalByteLedger,
) -> Result<(), SaltV2CodecError> {
    if ledger.canonical_padding_bits == 0 {
        return Ok(());
    }
    let byte_index = packed.len() - 1;
    let used_bits = 8 - ledger.canonical_padding_bits;
    let padding_mask = u8::MAX << used_bits;
    let bits = packed[byte_index] & padding_mask;
    if bits != 0 {
        return Err(SaltV2CodecError::NonCanonicalBitPadding {
            codec: SaltV2Codec::S34,
            byte_index,
            bits,
        });
    }
    Ok(())
}

fn require_exact_length(
    codec: SaltV2Codec,
    packed: &[u8],
    expected: usize,
) -> Result<(), SaltV2CodecError> {
    if packed.len() != expected {
        return Err(SaltV2CodecError::WrongPackedLength {
            codec,
            expected,
            got: packed.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_core::Trit;

    fn trits(values: &[i8]) -> Vec<Trit> {
        values
            .iter()
            .copied()
            .map(|value| Trit::from_i8(value).expect("test trit is in range"))
            .collect()
    }

    #[test]
    fn d2_round_trips_a_partial_byte_with_zero_trit_padding() {
        let input = trits(&[-1, 0, 1, -1, 1]);
        let packed = pack_d2(&input).expect("pack D2");

        assert_eq!(packed, [0x24, 0x56]);
        assert_eq!(unpack_d2(&packed, input.len()).expect("unpack D2"), input);
    }

    #[test]
    fn d2_rejects_the_reserved_two_bit_code() {
        assert_eq!(
            unpack_d2(&[0b0101_0111], 1),
            Err(SaltV2CodecError::InvalidD2Code {
                trit_index: 0,
                code: 0b11,
            })
        );
    }

    #[test]
    fn d2_rejects_nonzero_tail_trits() {
        assert_eq!(
            unpack_d2(&[0x00], 1),
            Err(SaltV2CodecError::NonCanonicalTritPadding {
                codec: SaltV2Codec::D2,
                trit_index: 1,
                code: 0,
            })
        );
    }

    #[test]
    fn d2_rejects_truncated_payloads() {
        assert_eq!(
            unpack_d2(&[], 1),
            Err(SaltV2CodecError::WrongPackedLength {
                codec: SaltV2Codec::D2,
                expected: 1,
                got: 0,
            })
        );
    }

    #[test]
    fn d2_byte_ledger_is_exact() {
        assert_eq!(
            SaltV2Codec::D2.ledger(5).expect("D2 ledger"),
            PhysicalByteLedger {
                codec: SaltV2Codec::D2,
                logical_trits: 5,
                encoded_bits: 16,
                physical_bytes: 2,
                canonical_padding_trits: 3,
                canonical_padding_bits: 0,
            }
        );
    }

    #[test]
    fn b3_round_trips_a_partial_radix_byte_with_zero_trit_padding() {
        let input = trits(&[-1, 0, 1, -1, 1, 1]);
        let packed = pack_b3(&input).expect("pack B3");

        assert_eq!(packed, [0xb7, 0x7a]);
        assert_eq!(unpack_b3(&packed, input.len()).expect("unpack B3"), input);
    }

    #[test]
    fn b3_rejects_codes_above_the_radix_three_domain() {
        assert_eq!(
            unpack_b3(&[243], 5),
            Err(SaltV2CodecError::InvalidB3Code {
                byte_index: 0,
                code: 243,
            })
        );
    }

    #[test]
    fn b3_rejects_nonzero_tail_trits() {
        assert_eq!(
            unpack_b3(&[0], 1),
            Err(SaltV2CodecError::NonCanonicalTritPadding {
                codec: SaltV2Codec::B3,
                trit_index: 1,
                code: 0,
            })
        );
    }

    #[test]
    fn b3_rejects_trailing_payload_bytes() {
        assert_eq!(
            unpack_b3(&[121, 121], 1),
            Err(SaltV2CodecError::WrongPackedLength {
                codec: SaltV2Codec::B3,
                expected: 1,
                got: 2,
            })
        );
    }

    #[test]
    fn every_full_b3_code_is_canonical_and_round_trips() {
        for code in 0..B3_CODE_COUNT as u8 {
            let decoded = unpack_b3(&[code], B3_TRITS_PER_BYTE).expect("valid B3 code");
            assert_eq!(pack_b3(&decoded).expect("repack B3"), [code]);
        }
    }

    #[test]
    fn b3_byte_ledger_is_exact() {
        assert_eq!(
            SaltV2Codec::B3.ledger(6).expect("B3 ledger"),
            PhysicalByteLedger {
                codec: SaltV2Codec::B3,
                logical_trits: 6,
                encoded_bits: 16,
                physical_bytes: 2,
                canonical_padding_trits: 4,
                canonical_padding_bits: 0,
            }
        );
    }

    #[test]
    fn s34_round_trips_two_groups_at_exactly_five_bits_each() {
        let input = trits(&[-1, 0, 1, -1, 1, -1, 0, 1]);
        let packed = pack_s34(&input).expect("pack S34");

        assert_eq!(packed, [0xc9, 0x02]);
        assert_eq!(unpack_s34(&packed, input.len()).expect("unpack S34"), input);
    }

    #[test]
    fn s34_rejects_groups_without_exactly_one_zero() {
        assert_eq!(
            pack_s34(&trits(&[-1, 1, -1, 1])),
            Err(SaltV2CodecError::S34ZeroCount {
                group_index: 0,
                zero_count: 0,
            })
        );
    }

    #[test]
    fn s34_rejects_nonzero_terminal_bit_padding() {
        assert_eq!(
            unpack_s34(&[0xc9, 0x82], 8),
            Err(SaltV2CodecError::NonCanonicalBitPadding {
                codec: SaltV2Codec::S34,
                byte_index: 1,
                bits: 0x80,
            })
        );
    }

    #[test]
    fn s34_rejects_truncated_payloads() {
        assert_eq!(
            unpack_s34(&[], 4),
            Err(SaltV2CodecError::WrongPackedLength {
                codec: SaltV2Codec::S34,
                expected: 1,
                got: 0,
            })
        );
    }

    #[test]
    fn every_s34_code_is_canonical_and_round_trips() {
        for code in 0u8..32 {
            let decoded = unpack_s34(&[code], S34_TRITS_PER_GROUP).expect("valid S34 code");
            assert_eq!(pack_s34(&decoded).expect("repack S34"), [code]);
        }
    }

    #[test]
    fn s34_rejects_partial_four_trit_groups() {
        assert_eq!(
            SaltV2Codec::S34.ledger(3),
            Err(SaltV2CodecError::S34TritCountNotMultipleOfFour { logical_trits: 3 })
        );
    }

    #[test]
    fn s34_byte_ledger_is_exact() {
        assert_eq!(
            SaltV2Codec::S34.ledger(12).expect("S34 ledger"),
            PhysicalByteLedger {
                codec: SaltV2Codec::S34,
                logical_trits: 12,
                encoded_bits: 15,
                physical_bytes: 2,
                canonical_padding_trits: 0,
                canonical_padding_bits: 1,
            }
        );
    }
}
