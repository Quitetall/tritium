//! Exact physical-byte allocation for SALT V2 nested profiles.

use core::fmt;

const SALT_V2_PLANES: usize = 3;

/// Exact byte counts in the serialized package and in the resident runtime image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhysicalBytes {
    /// Bytes written to the serialized model artifact.
    pub serialized: u64,
    /// Bytes occupied by the runtime-resident representation.
    pub resident: u64,
}

impl PhysicalBytes {
    /// Zero bytes in both accounting domains.
    pub const ZERO: Self = Self {
        serialized: 0,
        resident: 0,
    };

    /// Add both byte domains, returning `None` rather than wrapping either counter.
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        let Some(serialized) = self.serialized.checked_add(rhs.serialized) else {
            return None;
        };
        let Some(resident) = self.resident.checked_add(rhs.resident) else {
            return None;
        };
        Some(Self {
            serialized,
            resident,
        })
    }

    /// Whether both exact counters are at or below `maximum`.
    pub const fn fits_within(self, maximum: Self) -> bool {
        self.serialized <= maximum.serialized && self.resident <= maximum.resident
    }
}

/// One incremental physical cost with both its pre-materialization declaration
/// and optional post-materialization measurement.
///
/// Exact optimization uses [`Self::effective`], so measured bytes take
/// precedence. When a measurement is absent, the declaration is the exact
/// contract used for the final budget gate. Declarations remain visible in the
/// output ledger for estimate-versus-measurement auditing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteDelta {
    /// Byte delta declared by the selected codec before materialization.
    pub declared: PhysicalBytes,
    /// Exact measured delta, when the artifact and resident layout were materialized.
    pub measured: Option<PhysicalBytes>,
}

impl ByteDelta {
    /// Construct a delta whose declaration is authoritative.
    pub const fn declared(declared: PhysicalBytes) -> Self {
        Self {
            declared,
            measured: None,
        }
    }

    /// Construct a delta that retains its declaration and overrides it with a measurement.
    pub const fn measured(declared: PhysicalBytes, measured: PhysicalBytes) -> Self {
        Self {
            declared,
            measured: Some(measured),
        }
    }

    /// Return measured bytes when present, otherwise the authoritative declaration.
    pub const fn effective(self) -> PhysicalBytes {
        match self.measured {
            Some(measured) => measured,
            None => self.declared,
        }
    }
}

/// One nested plane-count candidate for a group.
///
/// `byte_delta` is incremental: the `P=1` entry is the first-plane cost and
/// entries `P=2` and `P=3` are the costs of appending those refinements. The
/// distortion is cumulative after all planes through `planes` are decoded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneCandidate {
    /// Consecutive plane ordinal, exactly `1`, `2`, or `3`.
    pub planes: u8,
    /// Incremental physical cost of adding this plane.
    pub byte_delta: ByteDelta,
    /// Non-negative finite distortion after decoding this plane prefix.
    pub distortion: f64,
}

/// All three nested candidates for one independently allocated group.
#[derive(Clone, Copy, Debug)]
pub struct GroupCandidates<'a> {
    /// Candidates in consecutive `P=1`, `P=2`, `P=3` order.
    pub candidates: &'a [PlaneCandidate],
}

/// Exact limits and fixed metadata cost for one published profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProfileBudget {
    /// Inclusive serialized and resident byte ceilings.
    pub maximum: PhysicalBytes,
    /// Fixed package/runtime metadata charged before any group plane bytes.
    pub metadata: ByteDelta,
}

/// Exact independent limits for the nested CompactV1 and NearLosslessV1 artifacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NestedProfileBudgets {
    /// Limits and metadata for the compact plane prefix.
    pub compact: ProfileBudget,
    /// Limits and metadata for the near-lossless refinement containing that prefix.
    pub near_lossless: ProfileBudget,
}

/// The profile whose validation or budget gate failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaltV2Profile {
    /// The compact physical-storage profile.
    CompactV1,
    /// The near-lossless refinement profile.
    NearLosslessV1,
}

/// A realized exact-byte allocation for one profile.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileAllocation {
    /// Plane count selected for each input group, in input order.
    pub plane_counts: Vec<u8>,
    /// Metadata plus selected incremental deltas using declarations only.
    pub declared_bytes: PhysicalBytes,
    /// Metadata plus selected deltas using measurements when available.
    pub physical_bytes: PhysicalBytes,
    /// Sum of the selected groups' cumulative distortion values.
    pub total_distortion: f64,
    /// Whether metadata and every selected plane delta carried a measurement.
    pub all_selected_bytes_measured: bool,
}

impl ProfileAllocation {
    /// Whether every group in `self` is a plane prefix of the corresponding group in `refinement`.
    pub fn is_prefix_of(&self, refinement: &Self) -> bool {
        self.plane_counts.len() == refinement.plane_counts.len()
            && self
                .plane_counts
                .iter()
                .zip(&refinement.plane_counts)
                .all(|(prefix, refined)| prefix <= refined)
    }
}

/// The pair of successively refinable SALT V2 profile allocations.
#[derive(Clone, Debug, PartialEq)]
pub struct NestedProfileAllocation {
    /// CompactV1 allocation and exact physical ledger.
    pub compact: ProfileAllocation,
    /// NearLosslessV1 allocation, constrained to contain `compact` plane-for-plane.
    pub near_lossless: ProfileAllocation,
}

/// Why an exact physical-byte allocation could not be produced.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum PhysicalAllocError {
    /// No independently allocated groups were supplied.
    EmptyGroups,
    /// A group did not provide exactly the required `P=1..=3` frontier.
    CandidateCount {
        /// Index of the invalid group.
        group: usize,
        /// Candidate count that was supplied.
        count: usize,
    },
    /// A candidate's plane ordinal did not match its position in the nested frontier.
    PlaneOrdinal {
        /// Index of the invalid group.
        group: usize,
        /// Expected consecutive plane ordinal.
        expected: u8,
        /// Plane ordinal that was supplied.
        actual: u8,
    },
    /// A cumulative candidate distortion was negative or non-finite.
    InvalidDistortion {
        /// Index of the invalid group.
        group: usize,
        /// Candidate plane ordinal.
        planes: u8,
        /// Invalid distortion value.
        distortion: f64,
    },
    /// Adding a refinement increased rather than reduced cumulative distortion.
    NonMonotoneDistortion {
        /// Index of the invalid group.
        group: usize,
        /// Candidate plane ordinal whose distortion increased.
        planes: u8,
    },
    /// The mandatory profile prefix plus metadata already exceeded an exact byte ceiling.
    BudgetTooSmall {
        /// Profile whose mandatory prefix did not fit.
        profile: SaltV2Profile,
        /// Exact bytes required by the mandatory prefix.
        required: PhysicalBytes,
        /// Requested exact byte ceilings.
        maximum: PhysicalBytes,
    },
    /// A declared byte ledger or aggregate distortion could not be represented.
    AccountingOverflow {
        /// Profile whose ledger overflowed.
        profile: SaltV2Profile,
    },
    /// The exact Pareto dynamic program exceeded its fail-closed state bound.
    StateSpaceTooLarge {
        /// Profile whose exact frontier grew too large.
        profile: SaltV2Profile,
        /// Number of states that the next expansion would require.
        states: usize,
    },
    /// The exact allocator could not reserve its bounded working set.
    WorkingMemoryUnavailable {
        /// Profile whose allocation workspace could not be reserved.
        profile: SaltV2Profile,
    },
}

impl fmt::Display for PhysicalAllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGroups => write!(f, "SALT V2 allocation requires at least one group"),
            Self::CandidateCount { group, count } => write!(
                f,
                "group {group} supplied {count} candidates; SALT V2 requires P=1..=3"
            ),
            Self::PlaneOrdinal {
                group,
                expected,
                actual,
            } => write!(
                f,
                "group {group} candidate ordinal {actual}; expected consecutive P={expected}"
            ),
            Self::InvalidDistortion {
                group,
                planes,
                distortion,
            } => write!(
                f,
                "group {group} P={planes} has invalid distortion {distortion}"
            ),
            Self::NonMonotoneDistortion { group, planes } => write!(
                f,
                "group {group} P={planes} increases cumulative distortion"
            ),
            Self::BudgetTooSmall {
                profile,
                required,
                maximum,
            } => write!(
                f,
                "{profile:?} mandatory prefix needs {required:?}, above {maximum:?}"
            ),
            Self::AccountingOverflow { profile } => {
                write!(
                    f,
                    "{profile:?} physical-byte or distortion ledger overflowed"
                )
            }
            Self::StateSpaceTooLarge { profile, states } => write!(
                f,
                "{profile:?} exact allocation frontier would require {states} states"
            ),
            Self::WorkingMemoryUnavailable { profile } => {
                write!(f, "{profile:?} exact allocation workspace is unavailable")
            }
        }
    }
}

impl std::error::Error for PhysicalAllocError {}

/// Allocate exact-byte CompactV1 and NearLosslessV1 plane prefixes.
///
/// The compact profile is allocated first. NearLosslessV1 starts at those exact
/// plane counts and can only append planes, making CompactV1 a byte-semantic
/// prefix rather than a separately quantized artifact. Regular equal-increment
/// layouts use an exact scalable chain decomposition; irregular layouts use an
/// exact Pareto dynamic program over serialized bytes, resident bytes, and
/// cumulative distortion. The solver fails closed if an irregular frontier
/// exceeds a bounded reference-state budget; it never falls back to ratio-greedy
/// selection. Every admission and the final gate compare integer serialized and
/// resident counters independently; logical bpw is never consulted.
pub fn allocate_nested_profiles(
    groups: &[GroupCandidates<'_>],
    budgets: &NestedProfileBudgets,
) -> Result<NestedProfileAllocation, PhysicalAllocError> {
    validate_groups(groups)?;

    let compact_floor = try_filled_plane_counts(groups.len(), 1, SaltV2Profile::CompactV1)?;
    let compact = allocate_profile(
        groups,
        budgets.compact,
        compact_floor,
        SaltV2Profile::CompactV1,
    )?;
    let near_lossless_floor =
        try_copy_plane_counts(&compact.plane_counts, SaltV2Profile::NearLosslessV1)?;
    let near_lossless = allocate_profile(
        groups,
        budgets.near_lossless,
        near_lossless_floor,
        SaltV2Profile::NearLosslessV1,
    )?;
    debug_assert!(compact.is_prefix_of(&near_lossless));

    Ok(NestedProfileAllocation {
        compact,
        near_lossless,
    })
}

fn validate_groups(groups: &[GroupCandidates<'_>]) -> Result<(), PhysicalAllocError> {
    if groups.is_empty() {
        return Err(PhysicalAllocError::EmptyGroups);
    }
    for (group_index, group) in groups.iter().enumerate() {
        if group.candidates.len() != SALT_V2_PLANES {
            return Err(PhysicalAllocError::CandidateCount {
                group: group_index,
                count: group.candidates.len(),
            });
        }
        for (candidate_index, candidate) in group.candidates.iter().enumerate() {
            let expected = (candidate_index + 1) as u8;
            if candidate.planes != expected {
                return Err(PhysicalAllocError::PlaneOrdinal {
                    group: group_index,
                    expected,
                    actual: candidate.planes,
                });
            }
            if !(candidate.distortion.is_finite() && candidate.distortion >= 0.0) {
                return Err(PhysicalAllocError::InvalidDistortion {
                    group: group_index,
                    planes: candidate.planes,
                    distortion: candidate.distortion,
                });
            }
            if candidate_index > 0
                && candidate.distortion > group.candidates[candidate_index - 1].distortion
            {
                return Err(PhysicalAllocError::NonMonotoneDistortion {
                    group: group_index,
                    planes: candidate.planes,
                });
            }
        }
    }
    Ok(())
}

fn allocate_profile(
    groups: &[GroupCandidates<'_>],
    limit: ProfileBudget,
    floors: Vec<u8>,
    profile: SaltV2Profile,
) -> Result<ProfileAllocation, PhysicalAllocError> {
    let minimum = sum_bytes(groups, &floors, limit.metadata, ByteMode::Effective);
    let Some(minimum_narrow) = minimum.to_physical() else {
        return Err(PhysicalAllocError::AccountingOverflow { profile });
    };
    if !minimum_narrow.fits_within(limit.maximum) {
        return Err(PhysicalAllocError::BudgetTooSmall {
            profile,
            required: minimum_narrow,
            maximum: limit.maximum,
        });
    }

    let plane_counts = exact_plane_counts(groups, &floors, limit, profile)?;

    let physical_bytes = sum_bytes(groups, &plane_counts, limit.metadata, ByteMode::Effective)
        .to_physical()
        .filter(|bytes| bytes.fits_within(limit.maximum))
        .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
    let declared_bytes = sum_bytes(groups, &plane_counts, limit.metadata, ByteMode::Declared)
        .to_physical()
        .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
    let total_distortion = total_distortion(groups, &plane_counts)
        .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
    let all_selected_bytes_measured =
        all_selected_bytes_measured(groups, &plane_counts, limit.metadata);

    Ok(ProfileAllocation {
        plane_counts,
        declared_bytes,
        physical_bytes,
        total_distortion,
        all_selected_bytes_measured,
    })
}

#[derive(Clone, Copy)]
enum ByteMode {
    Declared,
    Effective,
}

impl ByteMode {
    fn bytes(self, delta: ByteDelta) -> PhysicalBytes {
        match self {
            Self::Declared => delta.declared,
            Self::Effective => delta.effective(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct WideBytes {
    serialized: u128,
    resident: u128,
}

impl WideBytes {
    fn add(self, delta: PhysicalBytes) -> Self {
        Self {
            serialized: self.serialized.saturating_add(u128::from(delta.serialized)),
            resident: self.resident.saturating_add(u128::from(delta.resident)),
        }
    }

    fn to_physical(self) -> Option<PhysicalBytes> {
        Some(PhysicalBytes {
            serialized: u64::try_from(self.serialized).ok()?,
            resident: u64::try_from(self.resident).ok()?,
        })
    }
}

fn sum_bytes(
    groups: &[GroupCandidates<'_>],
    plane_counts: &[u8],
    metadata: ByteDelta,
    mode: ByteMode,
) -> WideBytes {
    let mut total = WideBytes::default().add(mode.bytes(metadata));
    for (group, &planes) in groups.iter().zip(plane_counts) {
        for candidate in &group.candidates[..usize::from(planes)] {
            total = total.add(mode.bytes(candidate.byte_delta));
        }
    }
    total
}

fn bundle_bytes(
    group: GroupCandidates<'_>,
    from: u8,
    to: u8,
    mode: ByteMode,
) -> Option<PhysicalBytes> {
    let mut total = PhysicalBytes::ZERO;
    for candidate in &group.candidates[usize::from(from)..usize::from(to)] {
        total = total.checked_add(mode.bytes(candidate.byte_delta))?;
    }
    Some(total)
}

fn distortion_at(group: GroupCandidates<'_>, planes: u8) -> f64 {
    group.candidates[usize::from(planes - 1)].distortion
}

fn total_distortion(groups: &[GroupCandidates<'_>], plane_counts: &[u8]) -> Option<f64> {
    exact_total_distortion(groups, plane_counts)?.to_f64()
}

fn exact_total_distortion(
    groups: &[GroupCandidates<'_>],
    plane_counts: &[u8],
) -> Option<ExactDistortion> {
    let mut total = ExactDistortion::ZERO;
    for (group, &planes) in groups.iter().zip(plane_counts) {
        total = total.checked_add_f64(distortion_at(*group, planes))?;
    }
    Some(total)
}

fn all_selected_bytes_measured(
    groups: &[GroupCandidates<'_>],
    plane_counts: &[u8],
    metadata: ByteDelta,
) -> bool {
    metadata.measured.is_some()
        && groups.iter().zip(plane_counts).all(|(group, &planes)| {
            group.candidates[..usize::from(planes)]
                .iter()
                .all(|candidate| candidate.byte_delta.measured.is_some())
        })
}

const MAX_EXACT_FRONTIER_STATES: usize = 4_096;

// A finite non-negative binary64 is an integer multiple of 2^-1074. The largest
// finite value occupies 2_098 bits in those units. Adding fewer than 2^usize::BITS
// such values needs at most another usize::BITS carry bits, so this fixed-width
// accumulator is sufficient for every slice addressable by this process.
const EXACT_F64_VALUE_BITS: usize = 2_098;
const EXACT_DISTORTION_BITS: usize = EXACT_F64_VALUE_BITS + usize::BITS as usize;
const EXACT_DISTORTION_LIMBS: usize = EXACT_DISTORTION_BITS.div_ceil(u64::BITS as usize);

/// Exact sum of finite, non-negative binary64 values in units of 2^-1074.
///
/// The little-endian limb representation makes addition independent of input
/// order and preserves distinctions below the current binary64 accumulator ULP.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExactDistortion {
    limbs: [u64; EXACT_DISTORTION_LIMBS],
}

impl ExactDistortion {
    const ZERO: Self = Self {
        limbs: [0; EXACT_DISTORTION_LIMBS],
    };

    fn from_f64(value: f64) -> Option<Self> {
        if !value.is_finite() || value < 0.0 {
            return None;
        }
        if value == 0.0 {
            return Some(Self::ZERO);
        }

        let bits = value.to_bits();
        if bits >> 63 != 0 {
            return None;
        }
        let raw_exponent = ((bits >> 52) & 0x7ff) as usize;
        let fraction = bits & ((1_u64 << 52) - 1);
        let (significand, shift) = if raw_exponent == 0 {
            (fraction, 0)
        } else {
            ((1_u64 << 52) | fraction, raw_exponent - 1)
        };
        let mut exact = Self::ZERO;
        exact
            .checked_add_shifted(significand, shift)
            .then_some(exact)
    }

    fn checked_add_f64(mut self, value: f64) -> Option<Self> {
        let exact = Self::from_f64(value)?;
        self.checked_add_assign(exact).then_some(self)
    }

    fn checked_add(mut self, rhs: Self) -> Option<Self> {
        self.checked_add_assign(rhs).then_some(self)
    }

    fn checked_sub(mut self, rhs: Self) -> Option<Self> {
        if self < rhs {
            return None;
        }
        let mut borrow = false;
        for (left, right) in self.limbs.iter_mut().zip(rhs.limbs) {
            let (difference, first_borrow) = left.overflowing_sub(right);
            let (difference, second_borrow) = difference.overflowing_sub(u64::from(borrow));
            *left = difference;
            borrow = first_borrow || second_borrow;
        }
        debug_assert!(!borrow);
        Some(self)
    }

    fn checked_add_assign(&mut self, rhs: Self) -> bool {
        let mut carry = false;
        for (left, right) in self.limbs.iter_mut().zip(rhs.limbs) {
            let (sum, first_carry) = left.overflowing_add(right);
            let (sum, second_carry) = sum.overflowing_add(u64::from(carry));
            *left = sum;
            carry = first_carry || second_carry;
        }
        !carry
    }

    fn checked_add_shifted(&mut self, value: u64, shift: usize) -> bool {
        if value == 0 {
            return true;
        }
        let limb = shift / u64::BITS as usize;
        let offset = shift % u64::BITS as usize;
        if limb >= self.limbs.len() || !self.checked_add_word(limb, value << offset) {
            return false;
        }
        offset == 0 || self.checked_add_word(limb + 1, value >> (u64::BITS as usize - offset))
    }

    fn checked_add_word(&mut self, mut limb: usize, word: u64) -> bool {
        if word == 0 {
            return true;
        }
        if limb >= self.limbs.len() {
            return false;
        }
        let (sum, mut carry) = self.limbs[limb].overflowing_add(word);
        self.limbs[limb] = sum;
        while carry {
            limb += 1;
            if limb >= self.limbs.len() {
                return false;
            }
            let (sum, next_carry) = self.limbs[limb].overflowing_add(1);
            self.limbs[limb] = sum;
            carry = next_carry;
        }
        true
    }

    /// Convert the exact integer to the correctly rounded binary64 value.
    fn to_f64(self) -> Option<f64> {
        let Some(mut highest) = self.highest_set_bit() else {
            return Some(0.0);
        };
        if highest < 52 {
            return Some(f64::from_bits(self.limbs[0]));
        }

        let shift = highest - 52;
        let mut significand = self.shifted_low_u64(shift);
        if shift > 0 {
            let round_bit = self.bit_is_set(shift - 1);
            let sticky = self.any_bit_below(shift - 1);
            if round_bit && (sticky || significand & 1 != 0) {
                significand = significand.checked_add(1)?;
                if significand == 1_u64 << 53 {
                    significand >>= 1;
                    highest += 1;
                }
            }
        }

        let raw_exponent = highest.checked_sub(51)?;
        if raw_exponent >= 0x7ff {
            return None;
        }
        let fraction = significand & ((1_u64 << 52) - 1);
        Some(f64::from_bits((raw_exponent as u64) << 52 | fraction))
    }

    fn highest_set_bit(self) -> Option<usize> {
        self.limbs.iter().rposition(|&limb| limb != 0).map(|index| {
            index * u64::BITS as usize
                + (u64::BITS - 1 - self.limbs[index].leading_zeros()) as usize
        })
    }

    fn shifted_low_u64(self, shift: usize) -> u64 {
        let limb = shift / u64::BITS as usize;
        let offset = shift % u64::BITS as usize;
        let low = self.limbs.get(limb).copied().unwrap_or(0) >> offset;
        if offset == 0 {
            low
        } else {
            low | self.limbs.get(limb + 1).copied().unwrap_or(0) << (u64::BITS as usize - offset)
        }
    }

    fn bit_is_set(self, bit: usize) -> bool {
        self.limbs
            .get(bit / u64::BITS as usize)
            .is_some_and(|limb| limb & (1_u64 << (bit % u64::BITS as usize)) != 0)
    }

    fn any_bit_below(self, exclusive: usize) -> bool {
        let full_limbs = exclusive / u64::BITS as usize;
        if self.limbs[..full_limbs.min(self.limbs.len())]
            .iter()
            .any(|&limb| limb != 0)
        {
            return true;
        }
        let remaining = exclusive % u64::BITS as usize;
        remaining > 0
            && self
                .limbs
                .get(full_limbs)
                .is_some_and(|limb| limb & ((1_u64 << remaining) - 1) != 0)
    }
}

impl Ord for ExactDistortion {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.limbs.iter().rev().cmp(other.limbs.iter().rev())
    }
}

impl PartialOrd for ExactDistortion {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Compact exact difference between two finite, non-negative binary64 values.
///
/// Keeping the source operands avoids storing a 272-byte superaccumulator in
/// every model-scale candidate. Comparisons and prefix checkpoints expand into
/// [`ExactDistortion`] only while arithmetic is performed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExactReduction {
    from_bits: u64,
    to_bits: u64,
}

impl ExactReduction {
    fn new(from: f64, to: f64) -> Option<Self> {
        if !from.is_finite() || !to.is_finite() || from < 0.0 || to < 0.0 || from < to {
            return None;
        }
        Some(Self {
            from_bits: canonical_nonnegative_bits(from),
            to_bits: canonical_nonnegative_bits(to),
        })
    }

    fn is_zero(self) -> bool {
        self.from_bits == self.to_bits
    }

    fn exact(self) -> ExactDistortion {
        ExactDistortion::from_f64(f64::from_bits(self.from_bits))
            .expect("validated reduction minuend")
            .checked_sub(
                ExactDistortion::from_f64(f64::from_bits(self.to_bits))
                    .expect("validated reduction subtrahend"),
            )
            .expect("validated non-negative reduction")
    }

    fn cmp(self, other: Self) -> core::cmp::Ordering {
        self.exact().cmp(&other.exact())
    }
}

fn canonical_nonnegative_bits(value: f64) -> u64 {
    if value == 0.0 { 0 } else { value.to_bits() }
}

trait RankedUpgrade {
    fn group(&self) -> usize;
    fn reduction(&self) -> ExactReduction;
}

const EXACT_PREFIX_BLOCK: usize = 64;

/// Exact block checkpoints over compact reductions.
///
/// One 272-byte checkpoint covers 64 upgrades. Prefix lookup expands at most
/// 63 compact reductions, keeping model-scale storage linear with a small
/// constant while preserving exact comparison semantics.
struct ExactPrefixIndex {
    checkpoints: Vec<ExactDistortion>,
    len: usize,
}

impl ExactPrefixIndex {
    fn new<T: RankedUpgrade>(
        upgrades: &[T],
        profile: SaltV2Profile,
    ) -> Result<Self, PhysicalAllocError> {
        let checkpoint_capacity = upgrades
            .len()
            .div_ceil(EXACT_PREFIX_BLOCK)
            .checked_add(1)
            .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
        let mut checkpoints = Vec::new();
        checkpoints
            .try_reserve_exact(checkpoint_capacity)
            .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
        checkpoints.push(ExactDistortion::ZERO);
        let mut total = ExactDistortion::ZERO;
        for (index, upgrade) in upgrades.iter().enumerate() {
            total = total
                .checked_add(upgrade.reduction().exact())
                .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
            if (index + 1) % EXACT_PREFIX_BLOCK == 0 {
                checkpoints.push(total);
            }
        }
        Ok(Self {
            checkpoints,
            len: upgrades.len(),
        })
    }

    fn sum<T: RankedUpgrade>(
        &self,
        upgrades: &[T],
        count: usize,
        profile: SaltV2Profile,
    ) -> Result<ExactDistortion, PhysicalAllocError> {
        debug_assert_eq!(self.len, upgrades.len());
        debug_assert!(count <= self.len);
        let block = count / EXACT_PREFIX_BLOCK;
        let mut total = self.checkpoints[block];
        for upgrade in &upgrades[block * EXACT_PREFIX_BLOCK..count] {
            total = total
                .checked_add(upgrade.reduction().exact())
                .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
        }
        Ok(total)
    }
}

const LEX_INDEX_BLOCK: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RankedGroup {
    group: usize,
    rank: usize,
}

impl RankedGroup {
    const NONE: Self = Self {
        group: usize::MAX,
        rank: usize::MAX,
    };

    fn min(self, other: Self) -> Self {
        if (self.group, self.rank) <= (other.group, other.rank) {
            self
        } else {
            other
        }
    }
}

/// Blocked range-minimum index for the first group changed between two prefixes.
struct LexGroupIndex {
    tree: Vec<RankedGroup>,
    leaves: usize,
    len: usize,
}

impl LexGroupIndex {
    fn new<T: RankedUpgrade>(
        upgrades: &[T],
        profile: SaltV2Profile,
    ) -> Result<Self, PhysicalAllocError> {
        let blocks = upgrades.len().div_ceil(LEX_INDEX_BLOCK);
        let leaves = blocks
            .max(1)
            .checked_next_power_of_two()
            .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
        let slots = leaves
            .checked_mul(2)
            .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
        let mut tree = Vec::new();
        tree.try_reserve_exact(slots)
            .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
        tree.resize(slots, RankedGroup::NONE);
        for block in 0..blocks {
            let start = block * LEX_INDEX_BLOCK;
            let end = (start + LEX_INDEX_BLOCK).min(upgrades.len());
            tree[leaves + block] = min_ranked_group(upgrades, start, end);
        }
        for index in (1..leaves).rev() {
            tree[index] = tree[index * 2].min(tree[index * 2 + 1]);
        }
        Ok(Self {
            tree,
            leaves,
            len: upgrades.len(),
        })
    }

    fn range_min<T: RankedUpgrade>(
        &self,
        upgrades: &[T],
        mut start: usize,
        mut end: usize,
    ) -> Option<RankedGroup> {
        debug_assert_eq!(self.len, upgrades.len());
        debug_assert!(start <= end && end <= self.len);
        if start == end {
            return None;
        }

        let mut best = RankedGroup::NONE;
        while start < end && !start.is_multiple_of(LEX_INDEX_BLOCK) {
            best = best.min(ranked_group(upgrades, start));
            start += 1;
        }
        while start < end && !end.is_multiple_of(LEX_INDEX_BLOCK) {
            end -= 1;
            best = best.min(ranked_group(upgrades, end));
        }
        if start < end {
            let mut left = self.leaves + start / LEX_INDEX_BLOCK;
            let mut right = self.leaves + end / LEX_INDEX_BLOCK;
            while left < right {
                if left & 1 != 0 {
                    best = best.min(self.tree[left]);
                    left += 1;
                }
                if right & 1 != 0 {
                    right -= 1;
                    best = best.min(self.tree[right]);
                }
                left /= 2;
                right /= 2;
            }
        }
        (best != RankedGroup::NONE).then_some(best)
    }
}

fn ranked_group<T: RankedUpgrade>(upgrades: &[T], rank: usize) -> RankedGroup {
    RankedGroup {
        group: upgrades[rank].group(),
        rank,
    }
}

fn min_ranked_group<T: RankedUpgrade>(upgrades: &[T], start: usize, end: usize) -> RankedGroup {
    (start..end).fold(RankedGroup::NONE, |best, rank| {
        best.min(ranked_group(upgrades, rank))
    })
}

#[derive(Debug)]
struct ExactState {
    bytes: PhysicalBytes,
    distortion: ExactDistortion,
    plane_counts: Vec<u8>,
}

fn exact_plane_counts(
    groups: &[GroupCandidates<'_>],
    floors: &[u8],
    limit: ProfileBudget,
    profile: SaltV2Profile,
) -> Result<Vec<u8>, PhysicalAllocError> {
    if let Some(plane_counts) = regular_plane_counts(groups, floors, limit, profile)? {
        return Ok(plane_counts);
    }

    bounded_exact_plane_counts(groups, floors, limit, profile)
}

fn bounded_exact_plane_counts(
    groups: &[GroupCandidates<'_>],
    floors: &[u8],
    limit: ProfileBudget,
    profile: SaltV2Profile,
) -> Result<Vec<u8>, PhysicalAllocError> {
    let minimum = minimum_bytes_for_prefix(groups, floors, limit.metadata, profile)?;
    let distortion = exact_total_distortion(groups, floors)
        .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
    let mut frontier = Vec::new();
    frontier
        .try_reserve_exact(1)
        .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
    frontier.push(ExactState {
        bytes: minimum,
        distortion,
        plane_counts: try_copy_plane_counts(floors, profile)?,
    });

    for (group_index, group) in groups.iter().copied().enumerate() {
        if floors[group_index] == SALT_V2_PLANES as u8 {
            continue;
        }
        let choices = usize::from(SALT_V2_PLANES as u8 - floors[group_index] + 1);
        let expanded = frontier
            .len()
            .checked_mul(choices)
            .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
        if expanded > MAX_EXACT_FRONTIER_STATES {
            return Err(PhysicalAllocError::StateSpaceTooLarge {
                profile,
                states: expanded,
            });
        }
        let mut next = Vec::new();
        next.try_reserve_exact(expanded)
            .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
        for state in &frontier {
            for planes in floors[group_index]..=SALT_V2_PLANES as u8 {
                let delta = bundle_bytes(group, floors[group_index], planes, ByteMode::Effective)
                    .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
                let Some(bytes) = state.bytes.checked_add(delta) else {
                    continue;
                };
                if !bytes.fits_within(limit.maximum) {
                    continue;
                }
                let distortion = state
                    .distortion
                    .checked_sub(
                        ExactDistortion::from_f64(distortion_at(group, floors[group_index]))
                            .ok_or(PhysicalAllocError::AccountingOverflow { profile })?,
                    )
                    .and_then(|without_group| {
                        without_group.checked_add_f64(distortion_at(group, planes))
                    })
                    .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
                let mut plane_counts = try_copy_plane_counts(&state.plane_counts, profile)?;
                plane_counts[group_index] = planes;
                next.push(ExactState {
                    bytes,
                    distortion,
                    plane_counts,
                });
            }
        }
        if next.is_empty() {
            return Err(PhysicalAllocError::BudgetTooSmall {
                profile,
                required: minimum,
                maximum: limit.maximum,
            });
        }
        frontier = pareto_prune(next, profile)?;
    }

    frontier
        .into_iter()
        .min_by(|left, right| {
            left.distortion
                .cmp(&right.distortion)
                .then_with(|| left.bytes.serialized.cmp(&right.bytes.serialized))
                .then_with(|| left.bytes.resident.cmp(&right.bytes.resident))
                .then_with(|| right.plane_counts.cmp(&left.plane_counts))
        })
        .map(|state| state.plane_counts)
        .ok_or(PhysicalAllocError::BudgetTooSmall {
            profile,
            required: minimum,
            maximum: limit.maximum,
        })
}

#[derive(Clone, Copy, Debug)]
struct UnitUpgrade {
    group: usize,
    target_planes: u8,
    distortion_reduction: ExactReduction,
}

impl RankedUpgrade for UnitUpgrade {
    fn group(&self) -> usize {
        self.group
    }

    fn reduction(&self) -> ExactReduction {
        self.distortion_reduction
    }
}

#[derive(Clone, Copy, Debug)]
struct BundledUpgrade {
    group: usize,
    distortion_reduction: ExactReduction,
    first_plane_reduction: ExactReduction,
}

impl RankedUpgrade for BundledUpgrade {
    fn group(&self) -> usize {
        self.group
    }

    fn reduction(&self) -> ExactReduction {
        self.distortion_reduction
    }
}

#[derive(Clone, Copy, Debug)]
struct HullChoice {
    bundle_count: usize,
    unit_count: usize,
    replacement_after_exclusion: bool,
}

impl HullChoice {
    fn upgrades(self) -> usize {
        self.bundle_count * 2 + self.unit_count
    }

    fn bundle_prefix_len(self) -> usize {
        self.bundle_count + usize::from(self.replacement_after_exclusion)
    }
}

#[derive(Clone, Copy, Debug)]
struct RegularChoice {
    hull: HullChoice,
    exceptional_bundle_rank: Option<usize>,
}

struct RegularIndexes<'a> {
    unit_prefix: &'a ExactPrefixIndex,
    bundle_prefix: &'a ExactPrefixIndex,
    unit_lex: &'a LexGroupIndex,
    bundle_lex: &'a LexGroupIndex,
    units: &'a [UnitUpgrade],
    bundles: &'a [BundledUpgrade],
}

/// Solve the regular full-tile case without constructing a model-sized Pareto frontier.
///
/// Equal effective incremental costs reduce the two physical ceilings to one
/// cardinality ceiling. This remains exactly solvable when a group's second
/// marginal reduction exceeds its first: two such groups can never both stop at
/// `P=2` in an optimum, because completing the one with the larger first
/// reduction strictly improves distortion at the same two-upgrade cost. Thus at
/// most one non-concave group is exceptional. Every other non-concave group is a
/// weight-two `P=1 -> P=3` bundle, while concave marginals are independent
/// weight-one upgrades. Sorted prefix sums enumerate the exact bundle count and
/// prefix/suffix leave-one-out optima evaluate every possible exception in
/// `O(n log n)` time and `O(n)` memory.
///
/// Unequal effective increments retain the general multiple-choice,
/// two-dimensional knapsack problem (NP-hard even after dropping one byte
/// dimension). Such inputs return `None` and use the bounded reference DP,
/// which fails closed rather than silently approximating.
fn regular_plane_counts(
    groups: &[GroupCandidates<'_>],
    floors: &[u8],
    limit: ProfileBudget,
    profile: SaltV2Profile,
) -> Result<Option<Vec<u8>>, PhysicalAllocError> {
    let mut common_cost = None;
    let mut available_upgrades = 0usize;

    for (group_index, group) in groups.iter().copied().enumerate() {
        for target_planes in floors[group_index].saturating_add(1)..=SALT_V2_PLANES as u8 {
            let candidate = group.candidates[usize::from(target_planes - 1)];
            let cost = candidate.byte_delta.effective();
            if common_cost.is_some_and(|expected| expected != cost) {
                return Ok(None);
            }
            common_cost = Some(cost);
            available_upgrades = available_upgrades
                .checked_add(1)
                .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
        }
    }

    let Some(common_cost) = common_cost else {
        return Ok(Some(try_copy_plane_counts(floors, profile)?));
    };
    let base = minimum_bytes_for_prefix(groups, floors, limit.metadata, profile)?;
    let serialized_capacity = equal_cost_capacity(
        limit.maximum.serialized - base.serialized,
        common_cost.serialized,
        available_upgrades,
    );
    let resident_capacity = equal_cost_capacity(
        limit.maximum.resident - base.resident,
        common_cost.resident,
        available_upgrades,
    );
    let capacity = serialized_capacity.min(resident_capacity);
    if capacity == 0 {
        return Ok(Some(try_copy_plane_counts(floors, profile)?));
    }
    if common_cost == PhysicalBytes::ZERO {
        return Ok(Some(try_filled_plane_counts(
            groups.len(),
            SALT_V2_PLANES as u8,
            profile,
        )?));
    }

    let mut unit_upgrades = Vec::new();
    let mut bundled_upgrades = Vec::new();
    for (group_index, group) in groups.iter().copied().enumerate() {
        match floors[group_index] {
            1 => {
                let first_reduction =
                    exact_reduction(distortion_at(group, 1), distortion_at(group, 2), profile)?;
                let second_reduction =
                    exact_reduction(distortion_at(group, 2), distortion_at(group, 3), profile)?;
                if second_reduction.cmp(first_reduction) == core::cmp::Ordering::Greater {
                    bundled_upgrades
                        .try_reserve(1)
                        .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
                    bundled_upgrades.push(BundledUpgrade {
                        group: group_index,
                        distortion_reduction: exact_reduction(
                            distortion_at(group, 1),
                            distortion_at(group, 3),
                            profile,
                        )?,
                        first_plane_reduction: first_reduction,
                    });
                } else {
                    if !first_reduction.is_zero() {
                        unit_upgrades.try_reserve(1).map_err(|_| {
                            PhysicalAllocError::WorkingMemoryUnavailable { profile }
                        })?;
                        unit_upgrades.push(UnitUpgrade {
                            group: group_index,
                            target_planes: 2,
                            distortion_reduction: first_reduction,
                        });
                    }
                    if !second_reduction.is_zero() {
                        unit_upgrades.try_reserve(1).map_err(|_| {
                            PhysicalAllocError::WorkingMemoryUnavailable { profile }
                        })?;
                        unit_upgrades.push(UnitUpgrade {
                            group: group_index,
                            target_planes: 3,
                            distortion_reduction: second_reduction,
                        });
                    }
                }
            }
            2 => {
                let reduction =
                    exact_reduction(distortion_at(group, 2), distortion_at(group, 3), profile)?;
                if !reduction.is_zero() {
                    unit_upgrades
                        .try_reserve(1)
                        .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
                    unit_upgrades.push(UnitUpgrade {
                        group: group_index,
                        target_planes: 3,
                        distortion_reduction: reduction,
                    });
                }
            }
            3 => {}
            _ => unreachable!("validated SALT V2 floor"),
        }
    }

    unit_upgrades.sort_unstable_by(|left, right| {
        right
            .distortion_reduction
            .cmp(left.distortion_reduction)
            .then_with(|| left.group.cmp(&right.group))
            .then_with(|| left.target_planes.cmp(&right.target_planes))
    });
    bundled_upgrades.sort_unstable_by(|left, right| {
        right
            .distortion_reduction
            .cmp(left.distortion_reduction)
            .then_with(|| left.group.cmp(&right.group))
    });

    let unit_prefix = ExactPrefixIndex::new(&unit_upgrades, profile)?;
    let bundle_prefix = ExactPrefixIndex::new(&bundled_upgrades, profile)?;
    let unit_lex = LexGroupIndex::new(&unit_upgrades, profile)?;
    let bundle_lex = LexGroupIndex::new(&bundled_upgrades, profile)?;
    let indexes = RegularIndexes {
        unit_prefix: &unit_prefix,
        bundle_prefix: &bundle_prefix,
        unit_lex: &unit_lex,
        bundle_lex: &bundle_lex,
        units: &unit_upgrades,
        bundles: &bundled_upgrades,
    };
    let choice = best_regular_choice(&indexes, capacity, profile)?;
    let plane_counts = materialize_regular_choice(choice, &indexes, floors, profile)?;
    debug_assert_eq!(
        plane_counts
            .iter()
            .zip(floors)
            .map(|(&planes, &floor)| usize::from(planes - floor))
            .sum::<usize>(),
        choice.hull.upgrades() + usize::from(choice.exceptional_bundle_rank.is_some())
    );
    debug_assert!(
        choice.hull.upgrades() + usize::from(choice.exceptional_bundle_rank.is_some()) <= capacity
    );
    Ok(Some(plane_counts))
}

fn best_regular_choice(
    indexes: &RegularIndexes<'_>,
    capacity: usize,
    profile: SaltV2Profile,
) -> Result<RegularChoice, PhysicalAllocError> {
    let unit_count = indexes.units.len();
    let bundle_count = indexes.bundles.len();
    let maximum_bundles = bundle_count.min(capacity / 2);
    let mut best = None;
    for selected_bundles in 0..=maximum_bundles {
        let selected_units = unit_count.min(capacity - selected_bundles * 2);
        retain_better_regular_choice(
            &mut best,
            RegularChoice {
                hull: HullChoice {
                    bundle_count: selected_bundles,
                    unit_count: selected_units,
                    replacement_after_exclusion: false,
                },
                exceptional_bundle_rank: None,
            },
            indexes,
            profile,
        )?;
    }

    if capacity > 0 && !indexes.bundles.is_empty() {
        let residual_capacity = capacity - 1;
        let ordinary_maximum = bundle_count.min(residual_capacity / 2);
        let ordinary_len = ordinary_maximum
            .checked_add(1)
            .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
        let mut ordinary_prefix_best = Vec::new();
        ordinary_prefix_best
            .try_reserve_exact(ordinary_len)
            .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
        let mut prefix_best = None;
        for selected_bundles in 0..=ordinary_maximum {
            let selected_units = unit_count.min(residual_capacity - selected_bundles * 2);
            let choice = HullChoice {
                bundle_count: selected_bundles,
                unit_count: selected_units,
                replacement_after_exclusion: false,
            };
            retain_better_hull_choice(&mut prefix_best, choice, None, indexes, profile)?;
            ordinary_prefix_best.push(prefix_best.expect("current prefix choice"));
        }

        let replacement_maximum = bundle_count.saturating_sub(1).min(residual_capacity / 2);
        let replacement_len = replacement_maximum
            .checked_add(1)
            .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
        let mut replacement_suffix_best = Vec::new();
        replacement_suffix_best
            .try_reserve_exact(replacement_len)
            .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
        replacement_suffix_best.resize(
            replacement_len,
            HullChoice {
                bundle_count: 0,
                unit_count: 0,
                replacement_after_exclusion: false,
            },
        );
        let mut suffix_best = None;
        for selected_bundles in (0..=replacement_maximum).rev() {
            let selected_units = unit_count.min(residual_capacity - selected_bundles * 2);
            let choice = HullChoice {
                // This prefix contains one extra bundle. The excluded bundle's
                // reduction is subtracted per exception below.
                bundle_count: selected_bundles,
                unit_count: selected_units,
                replacement_after_exclusion: true,
            };
            retain_better_hull_choice(&mut suffix_best, choice, None, indexes, profile)?;
            replacement_suffix_best[selected_bundles] = suffix_best.expect("current suffix choice");
        }

        for (excluded_rank, bundle) in indexes.bundles.iter().enumerate() {
            if bundle.first_plane_reduction.is_zero() {
                continue;
            }
            let mut best_without_bundle =
                Some(ordinary_prefix_best[excluded_rank.min(ordinary_maximum)]);
            if excluded_rank < replacement_maximum {
                let replacement = replacement_suffix_best[excluded_rank + 1];
                retain_better_hull_choice(
                    &mut best_without_bundle,
                    replacement,
                    Some(excluded_rank),
                    indexes,
                    profile,
                )?;
            }
            let hull = best_without_bundle.expect("ordinary zero-bundle choice");
            retain_better_regular_choice(
                &mut best,
                RegularChoice {
                    hull,
                    exceptional_bundle_rank: Some(excluded_rank),
                },
                indexes,
                profile,
            )?;
        }
    }

    Ok(best.expect("zero-upgrade regular choice"))
}

fn checked_reduction_add(
    left: ExactDistortion,
    right: ExactDistortion,
    profile: SaltV2Profile,
) -> Result<ExactDistortion, PhysicalAllocError> {
    left.checked_add(right)
        .ok_or(PhysicalAllocError::AccountingOverflow { profile })
}

fn hull_reduction(
    choice: HullChoice,
    excluded_bundle_rank: Option<usize>,
    indexes: &RegularIndexes<'_>,
    profile: SaltV2Profile,
) -> Result<ExactDistortion, PhysicalAllocError> {
    let unit_reduction = indexes
        .unit_prefix
        .sum(indexes.units, choice.unit_count, profile)?;
    let mut bundle_reduction =
        indexes
            .bundle_prefix
            .sum(indexes.bundles, choice.bundle_prefix_len(), profile)?;
    if let Some(excluded_rank) = excluded_bundle_rank
        && excluded_rank < choice.bundle_prefix_len()
    {
        bundle_reduction = bundle_reduction
            .checked_sub(indexes.bundles[excluded_rank].distortion_reduction.exact())
            .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
    }
    checked_reduction_add(unit_reduction, bundle_reduction, profile)
}

fn regular_reduction(
    choice: RegularChoice,
    indexes: &RegularIndexes<'_>,
    profile: SaltV2Profile,
) -> Result<ExactDistortion, PhysicalAllocError> {
    let mut reduction = hull_reduction(
        choice.hull,
        choice.exceptional_bundle_rank,
        indexes,
        profile,
    )?;
    if let Some(exceptional_rank) = choice.exceptional_bundle_rank {
        reduction = checked_reduction_add(
            reduction,
            indexes.bundles[exceptional_rank]
                .first_plane_reduction
                .exact(),
            profile,
        )?;
    }
    Ok(reduction)
}

fn exact_reduction(
    from: f64,
    to: f64,
    profile: SaltV2Profile,
) -> Result<ExactReduction, PhysicalAllocError> {
    ExactReduction::new(from, to).ok_or(PhysicalAllocError::AccountingOverflow { profile })
}

fn retain_better_hull_choice(
    best: &mut Option<HullChoice>,
    candidate: HullChoice,
    excluded_bundle_rank: Option<usize>,
    indexes: &RegularIndexes<'_>,
    profile: SaltV2Profile,
) -> Result<(), PhysicalAllocError> {
    let replace = match *best {
        None => true,
        Some(current) => {
            hull_choice_is_better(candidate, current, excluded_bundle_rank, indexes, profile)?
        }
    };
    if replace {
        *best = Some(candidate);
    }
    Ok(())
}

fn hull_choice_is_better(
    candidate: HullChoice,
    current: HullChoice,
    excluded_bundle_rank: Option<usize>,
    indexes: &RegularIndexes<'_>,
    profile: SaltV2Profile,
) -> Result<bool, PhysicalAllocError> {
    match hull_reduction(candidate, excluded_bundle_rank, indexes, profile)?.cmp(&hull_reduction(
        current,
        excluded_bundle_rank,
        indexes,
        profile,
    )?) {
        core::cmp::Ordering::Greater => return Ok(true),
        core::cmp::Ordering::Less => return Ok(false),
        core::cmp::Ordering::Equal => {}
    }
    if candidate.upgrades() != current.upgrades() {
        return Ok(candidate.upgrades() < current.upgrades());
    }
    Ok(hull_choice_lex_is_greater(
        candidate,
        current,
        excluded_bundle_rank,
        indexes,
    ))
}

fn retain_better_regular_choice(
    best: &mut Option<RegularChoice>,
    candidate: RegularChoice,
    indexes: &RegularIndexes<'_>,
    profile: SaltV2Profile,
) -> Result<(), PhysicalAllocError> {
    let replace = match *best {
        None => true,
        Some(current) => regular_choice_is_better(candidate, current, indexes, profile)?,
    };
    if replace {
        *best = Some(candidate);
    }
    Ok(())
}

fn regular_choice_is_better(
    candidate: RegularChoice,
    current: RegularChoice,
    indexes: &RegularIndexes<'_>,
    profile: SaltV2Profile,
) -> Result<bool, PhysicalAllocError> {
    let candidate_upgrades =
        candidate.hull.upgrades() + usize::from(candidate.exceptional_bundle_rank.is_some());
    let current_upgrades =
        current.hull.upgrades() + usize::from(current.exceptional_bundle_rank.is_some());
    match regular_reduction(candidate, indexes, profile)?
        .cmp(&regular_reduction(current, indexes, profile)?)
    {
        core::cmp::Ordering::Greater => return Ok(true),
        core::cmp::Ordering::Less => return Ok(false),
        core::cmp::Ordering::Equal => {}
    }
    if candidate_upgrades != current_upgrades {
        return Ok(candidate_upgrades < current_upgrades);
    }
    Ok(regular_choice_lex_is_greater(candidate, current, indexes))
}

fn regular_choice_lex_is_greater(
    candidate: RegularChoice,
    current: RegularChoice,
    indexes: &RegularIndexes<'_>,
) -> bool {
    let unit_difference = prefix_difference(
        indexes.unit_lex,
        indexes.units,
        candidate.hull.unit_count,
        current.hull.unit_count,
    );
    let bundle_difference = first_regular_bundle_difference(candidate, current, indexes);
    first_difference_prefers_candidate(
        unit_difference,
        bundle_difference,
        |rank| bundle_planes_for_choice(candidate, rank),
        |rank| bundle_planes_for_choice(current, rank),
    )
}

fn hull_choice_lex_is_greater(
    candidate: HullChoice,
    current: HullChoice,
    excluded_bundle_rank: Option<usize>,
    indexes: &RegularIndexes<'_>,
) -> bool {
    let unit_difference = prefix_difference(
        indexes.unit_lex,
        indexes.units,
        candidate.unit_count,
        current.unit_count,
    );
    let bundle_difference =
        first_hull_bundle_difference(candidate, current, excluded_bundle_rank, indexes);
    first_difference_prefers_candidate(
        unit_difference,
        bundle_difference,
        |rank| hull_bundle_planes(candidate, rank, excluded_bundle_rank),
        |rank| hull_bundle_planes(current, rank, excluded_bundle_rank),
    )
}

fn prefix_difference<T: RankedUpgrade>(
    index: &LexGroupIndex,
    upgrades: &[T],
    candidate_count: usize,
    current_count: usize,
) -> Option<(RankedGroup, bool)> {
    (candidate_count != current_count).then(|| {
        (
            index
                .range_min(
                    upgrades,
                    candidate_count.min(current_count),
                    candidate_count.max(current_count),
                )
                .expect("non-empty prefix difference"),
            candidate_count > current_count,
        )
    })
}

fn first_difference_prefers_candidate(
    unit_difference: Option<(RankedGroup, bool)>,
    bundle_difference: Option<RankedGroup>,
    candidate_bundle_planes: impl FnOnce(usize) -> u8,
    current_bundle_planes: impl FnOnce(usize) -> u8,
) -> bool {
    match (unit_difference, bundle_difference) {
        (None, None) => false,
        (Some((_, candidate_has_more)), None) => candidate_has_more,
        (None, Some(bundle)) => {
            candidate_bundle_planes(bundle.rank) > current_bundle_planes(bundle.rank)
        }
        (Some((unit, candidate_has_more)), Some(bundle)) => {
            debug_assert_ne!(unit.group, bundle.group);
            if unit.group < bundle.group {
                candidate_has_more
            } else {
                candidate_bundle_planes(bundle.rank) > current_bundle_planes(bundle.rank)
            }
        }
    }
}

fn first_hull_bundle_difference(
    candidate: HullChoice,
    current: HullChoice,
    excluded_bundle_rank: Option<usize>,
    indexes: &RegularIndexes<'_>,
) -> Option<RankedGroup> {
    let candidate_take = candidate.bundle_prefix_len();
    let current_take = current.bundle_prefix_len();
    let mut best = range_min_excluding(
        indexes.bundle_lex,
        indexes.bundles,
        candidate_take.min(current_take),
        candidate_take.max(current_take),
        [excluded_bundle_rank, None],
    );
    if let Some(rank) = excluded_bundle_rank
        && hull_bundle_planes(candidate, rank, excluded_bundle_rank)
            != hull_bundle_planes(current, rank, excluded_bundle_rank)
    {
        best = min_optional_ranked(best, ranked_group(indexes.bundles, rank));
    }
    best
}

fn first_regular_bundle_difference(
    candidate: RegularChoice,
    current: RegularChoice,
    indexes: &RegularIndexes<'_>,
) -> Option<RankedGroup> {
    let candidate_take = candidate.hull.bundle_prefix_len();
    let current_take = current.hull.bundle_prefix_len();
    let exceptions = [
        candidate.exceptional_bundle_rank,
        current.exceptional_bundle_rank,
    ];
    let mut best = range_min_excluding(
        indexes.bundle_lex,
        indexes.bundles,
        candidate_take.min(current_take),
        candidate_take.max(current_take),
        exceptions,
    );
    let mut checked_rank = None;
    for rank in exceptions.into_iter().flatten() {
        if checked_rank == Some(rank) {
            continue;
        }
        checked_rank = Some(rank);
        if bundle_planes_for_choice(candidate, rank) != bundle_planes_for_choice(current, rank) {
            best = min_optional_ranked(best, ranked_group(indexes.bundles, rank));
        }
    }
    best
}

fn range_min_excluding<T: RankedUpgrade>(
    index: &LexGroupIndex,
    upgrades: &[T],
    start: usize,
    end: usize,
    excluded: [Option<usize>; 2],
) -> Option<RankedGroup> {
    let mut excluded = excluded.map(|rank| rank.unwrap_or(usize::MAX));
    excluded.sort_unstable();
    let mut best = None;
    let mut cursor = start;
    for rank in excluded {
        if rank < cursor || rank >= end {
            continue;
        }
        if let Some(candidate) = index.range_min(upgrades, cursor, rank) {
            best = min_optional_ranked(best, candidate);
        }
        cursor = rank + 1;
    }
    if let Some(candidate) = index.range_min(upgrades, cursor, end) {
        best = min_optional_ranked(best, candidate);
    }
    best
}

fn min_optional_ranked(best: Option<RankedGroup>, candidate: RankedGroup) -> Option<RankedGroup> {
    Some(best.map_or(candidate, |current| current.min(candidate)))
}

fn hull_bundle_planes(choice: HullChoice, rank: usize, excluded_bundle_rank: Option<usize>) -> u8 {
    if excluded_bundle_rank == Some(rank) {
        1
    } else if rank < choice.bundle_prefix_len() {
        3
    } else {
        1
    }
}

fn bundle_planes_for_choice(choice: RegularChoice, rank: usize) -> u8 {
    if choice.exceptional_bundle_rank == Some(rank) {
        return 2;
    }
    let take = choice.hull.bundle_count + usize::from(choice.hull.replacement_after_exclusion);
    if rank < take { 3 } else { 1 }
}

fn materialize_regular_choice(
    choice: RegularChoice,
    indexes: &RegularIndexes<'_>,
    floors: &[u8],
    profile: SaltV2Profile,
) -> Result<Vec<u8>, PhysicalAllocError> {
    let mut plane_counts = try_copy_plane_counts(floors, profile)?;
    apply_unit_prefix(&mut plane_counts, indexes.units, choice.hull.unit_count);
    let take = choice.hull.bundle_count + usize::from(choice.hull.replacement_after_exclusion);
    for (rank, upgrade) in indexes.bundles.iter().take(take).enumerate() {
        if choice.exceptional_bundle_rank != Some(rank) {
            plane_counts[upgrade.group] = 3;
        }
    }
    if let Some(excluded_rank) = choice.exceptional_bundle_rank {
        plane_counts[indexes.bundles[excluded_rank].group] = 2;
    }
    Ok(plane_counts)
}

fn try_copy_plane_counts(
    source: &[u8],
    profile: SaltV2Profile,
) -> Result<Vec<u8>, PhysicalAllocError> {
    let mut counts = Vec::new();
    counts
        .try_reserve_exact(source.len())
        .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
    counts.extend_from_slice(source);
    Ok(counts)
}

fn try_filled_plane_counts(
    len: usize,
    planes: u8,
    profile: SaltV2Profile,
) -> Result<Vec<u8>, PhysicalAllocError> {
    let mut counts = Vec::new();
    counts
        .try_reserve_exact(len)
        .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
    counts.resize(len, planes);
    Ok(counts)
}

fn apply_unit_prefix(plane_counts: &mut [u8], units: &[UnitUpgrade], unit_count: usize) {
    for upgrade in units.iter().take(unit_count) {
        debug_assert_eq!(plane_counts[upgrade.group] + 1, upgrade.target_planes);
        plane_counts[upgrade.group] = upgrade.target_planes;
    }
}

fn equal_cost_capacity(remaining: u64, cost: u64, available: usize) -> usize {
    remaining.checked_div(cost).map_or(available, |capacity| {
        usize::try_from(capacity)
            .unwrap_or(usize::MAX)
            .min(available)
    })
}

fn minimum_bytes_for_prefix(
    groups: &[GroupCandidates<'_>],
    floors: &[u8],
    metadata: ByteDelta,
    profile: SaltV2Profile,
) -> Result<PhysicalBytes, PhysicalAllocError> {
    sum_bytes(groups, floors, metadata, ByteMode::Effective)
        .to_physical()
        .ok_or(PhysicalAllocError::AccountingOverflow { profile })
}

fn pareto_prune(
    mut states: Vec<ExactState>,
    profile: SaltV2Profile,
) -> Result<Vec<ExactState>, PhysicalAllocError> {
    states.sort_unstable_by(|left, right| {
        left.bytes
            .serialized
            .cmp(&right.bytes.serialized)
            .then_with(|| left.bytes.resident.cmp(&right.bytes.resident))
            .then_with(|| left.distortion.cmp(&right.distortion))
            .then_with(|| right.plane_counts.cmp(&left.plane_counts))
    });
    let mut deduplicated: Vec<ExactState> = Vec::new();
    deduplicated
        .try_reserve_exact(states.len())
        .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
    for state in states {
        if let Some(previous) = deduplicated.last_mut()
            && previous.bytes == state.bytes
        {
            if state.distortion < previous.distortion
                || (state.distortion == previous.distortion
                    && state.plane_counts > previous.plane_counts)
            {
                *previous = state;
            }
        } else {
            deduplicated.push(state);
        }
    }

    let mut keep = Vec::new();
    keep.try_reserve_exact(deduplicated.len())
        .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
    keep.resize(deduplicated.len(), true);
    'candidate: for (index, candidate) in deduplicated.iter().enumerate() {
        for (other_index, other) in deduplicated.iter().enumerate() {
            if index == other_index {
                continue;
            }
            let no_more_bytes = other.bytes.serialized <= candidate.bytes.serialized
                && other.bytes.resident <= candidate.bytes.resident;
            let no_worse_objective = other.distortion <= candidate.distortion;
            if no_more_bytes && no_worse_objective {
                keep[index] = false;
                continue 'candidate;
            }
        }
    }
    let kept_count = keep.iter().filter(|&&retain| retain).count();
    let mut kept = Vec::new();
    kept.try_reserve_exact(kept_count)
        .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
    kept.extend(
        deduplicated
            .into_iter()
            .zip(keep)
            .filter_map(|(state, retain)| retain.then_some(state)),
    );
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(serialized: u64, resident: u64) -> PhysicalBytes {
        PhysicalBytes {
            serialized,
            resident,
        }
    }

    fn delta(serialized: u64, resident: u64) -> ByteDelta {
        ByteDelta::declared(bytes(serialized, resident))
    }

    fn measured_delta(
        declared_serialized: u64,
        declared_resident: u64,
        measured_serialized: u64,
        measured_resident: u64,
    ) -> ByteDelta {
        ByteDelta::measured(
            bytes(declared_serialized, declared_resident),
            bytes(measured_serialized, measured_resident),
        )
    }

    fn candidate(planes: u8, bytes: ByteDelta, distortion: f64) -> PlaneCandidate {
        PlaneCandidate {
            planes,
            byte_delta: bytes,
            distortion,
        }
    }

    fn group(candidates: &[PlaneCandidate; 3]) -> GroupCandidates<'_> {
        GroupCandidates { candidates }
    }

    fn budget(serialized: u64, resident: u64, metadata: ByteDelta) -> ProfileBudget {
        ProfileBudget {
            maximum: bytes(serialized, resident),
            metadata,
        }
    }

    #[test]
    fn physical_totals_include_exact_measured_metadata_and_group_deltas() {
        let candidates = [
            candidate(1, measured_delta(10, 20, 12, 23), 8.0),
            candidate(2, delta(50, 50), 4.0),
            candidate(3, delta(50, 50), 2.0),
        ];
        let groups = [group(&candidates)];
        let budgets = NestedProfileBudgets {
            compact: budget(18, 32, measured_delta(5, 7, 6, 9)),
            near_lossless: budget(20, 34, measured_delta(7, 8, 8, 11)),
        };

        let allocation = allocate_nested_profiles(&groups, &budgets).unwrap();

        assert_eq!(allocation.compact.plane_counts, vec![1]);
        assert_eq!(allocation.compact.declared_bytes, bytes(15, 27));
        assert_eq!(allocation.compact.physical_bytes, bytes(18, 32));
        assert_eq!(allocation.near_lossless.declared_bytes, bytes(17, 28));
        assert_eq!(allocation.near_lossless.physical_bytes, bytes(20, 34));
        assert!(allocation.compact.all_selected_bytes_measured);
        assert!(allocation.near_lossless.all_selected_bytes_measured);
    }

    #[test]
    fn empty_group_set_is_rejected() {
        let budgets = NestedProfileBudgets {
            compact: budget(0, 0, delta(0, 0)),
            near_lossless: budget(0, 0, delta(0, 0)),
        };

        assert_eq!(
            allocate_nested_profiles(&[], &budgets),
            Err(PhysicalAllocError::EmptyGroups)
        );
    }

    #[test]
    fn measured_costs_drive_the_exact_frontier_without_exceeding_either_budget() {
        let first = [
            candidate(1, measured_delta(2, 2, 2, 2), 10.0),
            candidate(2, measured_delta(2, 2, 20, 20), 1.0),
            candidate(3, delta(50, 50), 0.0),
        ];
        let second = [
            candidate(1, measured_delta(2, 2, 2, 2), 10.0),
            candidate(2, measured_delta(3, 3, 3, 3), 4.0),
            candidate(3, delta(50, 50), 3.0),
        ];
        let groups = [group(&first), group(&second)];
        let budgets = NestedProfileBudgets {
            compact: budget(7, 7, measured_delta(0, 0, 0, 0)),
            near_lossless: budget(7, 7, measured_delta(0, 0, 0, 0)),
        };

        let allocation = allocate_nested_profiles(&groups, &budgets).unwrap();

        assert_eq!(allocation.compact.plane_counts, vec![1, 2]);
        assert!(
            allocation
                .compact
                .physical_bytes
                .fits_within(budgets.compact.maximum)
        );
        assert!(
            allocation
                .near_lossless
                .physical_bytes
                .fits_within(budgets.near_lossless.maximum)
        );
    }

    #[test]
    fn every_exact_budget_pair_is_a_hard_ceiling() {
        let first = [
            candidate(1, measured_delta(3, 4, 3, 5), 15.0),
            candidate(2, measured_delta(2, 3, 5, 2), 6.0),
            candidate(3, delta(1, 6), 2.0),
        ];
        let second = [
            candidate(1, delta(4, 2), 12.0),
            candidate(2, measured_delta(2, 2, 2, 4), 7.0),
            candidate(3, delta(7, 1), 1.0),
        ];
        let groups = [group(&first), group(&second)];

        for serialized in 9..=30 {
            for resident in 10..=30 {
                let limit = budget(serialized, resident, measured_delta(2, 3, 2, 3));
                let allocation = allocate_nested_profiles(
                    &groups,
                    &NestedProfileBudgets {
                        compact: limit,
                        near_lossless: limit,
                    },
                )
                .unwrap();
                assert!(allocation.compact.physical_bytes.fits_within(limit.maximum));
                assert!(
                    allocation
                        .near_lossless
                        .physical_bytes
                        .fits_within(limit.maximum)
                );
            }
        }
    }

    #[test]
    fn equal_marginals_break_ties_toward_lower_group_index() {
        let a = [
            candidate(1, delta(2, 2), 10.0),
            candidate(2, delta(2, 2), 5.0),
            candidate(3, delta(20, 20), 4.0),
        ];
        let b = a;
        let groups = [group(&a), group(&b)];
        let budgets = NestedProfileBudgets {
            compact: budget(6, 6, delta(0, 0)),
            near_lossless: budget(6, 6, delta(0, 0)),
        };

        let first = allocate_nested_profiles(&groups, &budgets).unwrap();
        let second = allocate_nested_profiles(&groups, &budgets).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.compact.plane_counts, vec![2, 1]);
    }

    #[test]
    fn exact_distortion_is_order_independent_across_the_binary64_range() {
        let mut seed = 0xa076_1d64_78bd_642f_u64;
        let mut values = Vec::new();
        for _ in 0..256 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let raw_exponent = (seed % 0x7ff) << 52;
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let fraction = seed & ((1_u64 << 52) - 1);
            let value = f64::from_bits(raw_exponent | fraction);
            assert_eq!(
                ExactDistortion::from_f64(value).unwrap().to_f64(),
                Some(value)
            );
            values.push(value);
        }

        let forward = values
            .iter()
            .try_fold(ExactDistortion::ZERO, |sum, &value| {
                sum.checked_add_f64(value)
            });
        let reverse = values
            .iter()
            .rev()
            .try_fold(ExactDistortion::ZERO, |sum, &value| {
                sum.checked_add_f64(value)
            });

        assert_eq!(forward, reverse);
        let smallest = ExactDistortion::from_f64(f64::from_bits(1)).unwrap();
        let largest = ExactDistortion::from_f64(f64::MAX).unwrap();
        assert_eq!(
            largest.checked_add(smallest).unwrap().checked_sub(largest),
            Some(smallest)
        );

        let two_to_53 = ExactDistortion::from_f64(9_007_199_254_740_992.0).unwrap();
        assert_eq!(
            two_to_53.checked_add_f64(1.0).unwrap().to_f64(),
            Some(9_007_199_254_740_992.0)
        );
        assert_eq!(
            two_to_53
                .checked_add_f64(1.0)
                .unwrap()
                .checked_add_f64(1.0)
                .unwrap()
                .checked_add_f64(1.0)
                .unwrap()
                .to_f64(),
            Some(9_007_199_254_740_996.0)
        );

        let largest_subnormal =
            ExactDistortion::from_f64(f64::from_bits((1_u64 << 52) - 1)).unwrap();
        assert_eq!(
            largest_subnormal.checked_add(smallest).unwrap().to_f64(),
            Some(f64::MIN_POSITIVE)
        );
    }

    #[test]
    fn exact_distortion_handles_limb_carry_borrow_and_overflow_rounding() {
        // Raw exponent 12 represents exactly bit 63 in 2^-1074 units.
        let high_word = ExactDistortion::from_f64(f64::from_bits(12_u64 << 52)).unwrap();
        let doubled = high_word.checked_add(high_word).unwrap();
        assert_eq!(doubled.limbs[0], 0);
        assert_eq!(doubled.limbs[1], 1);

        let smallest = ExactDistortion::from_f64(f64::from_bits(1)).unwrap();
        let borrowed = doubled.checked_sub(smallest).unwrap();
        assert_eq!(borrowed.limbs[0], u64::MAX);
        assert_eq!(borrowed.limbs[1], 0);

        let largest = ExactDistortion::from_f64(f64::MAX).unwrap();
        let below_overflow_midpoint = f64::from_bits((969_u64 + 1023) << 52);
        let overflow_midpoint = f64::from_bits((970_u64 + 1023) << 52);
        assert_eq!(
            largest
                .checked_add_f64(below_overflow_midpoint)
                .unwrap()
                .to_f64(),
            Some(f64::MAX)
        );
        assert_eq!(
            largest.checked_add_f64(overflow_midpoint).unwrap().to_f64(),
            None
        );
    }

    #[test]
    fn exact_distortion_matches_an_independent_u128_oracle() {
        let mut seed = 0xe703_7ed1_a0b4_28db_u64;
        let mut exact = ExactDistortion::ZERO;
        let mut oracle = 0_u128;
        for _ in 0..512 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let raw_exponent = seed % 66;
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let fraction = seed & ((1_u64 << 52) - 1);
            let value = f64::from_bits(raw_exponent << 52 | fraction);
            let coefficient = if raw_exponent == 0 {
                u128::from(fraction)
            } else {
                u128::from((1_u64 << 52) | fraction) << (raw_exponent - 1)
            };
            oracle = oracle.checked_add(coefficient).unwrap();
            exact = exact.checked_add_f64(value).unwrap();
        }

        assert_eq!(exact.limbs[0], oracle as u64);
        assert_eq!(exact.limbs[1], (oracle >> 64) as u64);
        assert!(exact.limbs[2..].iter().all(|&limb| limb == 0));
    }

    #[test]
    fn bounded_frontier_does_not_round_away_a_strict_gain() {
        let small = [
            candidate(1, delta(1, 1), 1.0),
            candidate(2, delta(1, 1), 0.5),
            // The unequal third-plane cost keeps this case on the bounded DP.
            candidate(3, delta(2, 2), 0.5),
        ];
        let large = [
            candidate(1, delta(1, 1), 9_007_199_254_740_992.0),
            candidate(2, delta(1, 1), 9_007_199_254_740_991.0),
            candidate(3, delta(1, 1), 9_007_199_254_740_991.0),
        ];
        let groups = [group(&small), group(&large)];
        let limit = budget(3, 3, delta(0, 0));

        let allocation = allocate_nested_profiles(
            &groups,
            &NestedProfileBudgets {
                compact: limit,
                near_lossless: limit,
            },
        )
        .expect("exact bounded allocation");

        // Exact represented-binary64 totals are 2^53+1, 2^53+0.5, and 2^53.
        assert_eq!(allocation.compact.plane_counts, vec![1, 2]);
        assert_eq!(allocation.compact.total_distortion, 9_007_199_254_740_992.0);
    }

    #[test]
    fn regular_solver_does_not_drop_a_small_gain_beside_a_large_gain() {
        let small_bundle = [
            candidate(1, delta(1, 1), 1.0),
            candidate(2, delta(1, 1), 1.0),
            candidate(3, delta(1, 1), 0.0),
        ];
        let large_unit = [
            candidate(1, delta(1, 1), 9_007_199_254_740_992.0),
            candidate(2, delta(1, 1), 0.0),
            candidate(3, delta(1, 1), 0.0),
        ];
        let groups = [group(&small_bundle), group(&large_unit)];
        let limit = budget(5, 5, delta(0, 0));

        let allocation = allocate_nested_profiles(
            &groups,
            &NestedProfileBudgets {
                compact: limit,
                near_lossless: limit,
            },
        )
        .expect("exact regular allocation");

        assert_eq!(allocation.compact.plane_counts, vec![3, 2]);
        assert_eq!(allocation.compact.total_distortion, 0.0);
    }

    #[test]
    fn scalable_regular_solver_uses_the_full_plane_vector_for_ties() {
        let earlier_bundle = [
            candidate(1, delta(1, 1), 10.0),
            candidate(2, delta(1, 1), 9.0),
            candidate(3, delta(1, 1), 0.0),
        ];
        let later_units = [
            candidate(1, delta(1, 1), 10.0),
            candidate(2, delta(1, 1), 5.0),
            candidate(3, delta(1, 1), 0.0),
        ];
        let inert = [
            candidate(1, delta(1, 1), 0.0),
            candidate(2, delta(1, 1), 0.0),
            candidate(3, delta(1, 1), 0.0),
        ];
        let groups = [
            group(&earlier_bundle),
            group(&later_units),
            group(&inert),
            group(&inert),
            group(&inert),
            group(&inert),
            group(&inert),
            group(&inert),
        ];
        let limit = budget(10, 10, delta(0, 0));

        let allocation = allocate_nested_profiles(
            &groups,
            &NestedProfileBudgets {
                compact: limit,
                near_lossless: limit,
            },
        )
        .expect("scalable regular allocation");

        assert_eq!(
            allocation.compact.plane_counts,
            vec![3, 1, 1, 1, 1, 1, 1, 1]
        );
    }

    #[test]
    fn scalable_regular_solver_preserves_gains_below_a_large_prefix() {
        let small_bundle = [
            candidate(1, delta(1, 1), 1.0),
            candidate(2, delta(1, 1), 1.0),
            candidate(3, delta(1, 1), 0.0),
        ];
        let large_unit = [
            candidate(1, delta(1, 1), 9_007_199_254_740_992.0),
            candidate(2, delta(1, 1), 0.0),
            candidate(3, delta(1, 1), 0.0),
        ];
        let inert = [
            candidate(1, delta(1, 1), 0.0),
            candidate(2, delta(1, 1), 0.0),
            candidate(3, delta(1, 1), 0.0),
        ];
        let groups = [
            group(&small_bundle),
            group(&large_unit),
            group(&inert),
            group(&inert),
            group(&inert),
            group(&inert),
            group(&inert),
            group(&inert),
        ];
        let limit = budget(11, 11, delta(0, 0));

        let allocation = allocate_nested_profiles(
            &groups,
            &NestedProfileBudgets {
                compact: limit,
                near_lossless: limit,
            },
        )
        .expect("exact scalable allocation");

        assert_eq!(
            allocation.compact.plane_counts,
            vec![3, 2, 1, 1, 1, 1, 1, 1]
        );
        assert_eq!(allocation.compact.total_distortion, 0.0);
    }

    #[test]
    fn scalable_regular_solver_accepts_realistic_fractional_frontiers() {
        let candidates = [
            candidate(1, delta(1, 1), 10.1),
            candidate(2, delta(1, 1), 5.2),
            candidate(3, delta(1, 1), 2.3),
        ];
        let groups = vec![group(&candidates); 8];
        let limit = budget(12, 12, delta(0, 0));

        let allocation = allocate_nested_profiles(
            &groups,
            &NestedProfileBudgets {
                compact: limit,
                near_lossless: limit,
            },
        )
        .expect("arbitrary binary64 marginals must remain exactly rankable");

        assert_eq!(
            allocation.compact.plane_counts,
            vec![2, 2, 2, 2, 1, 1, 1, 1]
        );
        assert_eq!(
            allocation.compact.plane_counts,
            brute_force_best(&groups, limit)
        );
    }

    #[test]
    fn scalable_regular_solver_resolves_an_eight_group_aggregate_tie() {
        let distortions = [
            [4.0, 2.0, 0.0],
            [4.0, 3.0, 0.0],
            [2.0, 2.0, 0.0],
            [4.0, 3.0, 0.0],
            [5.0, 5.0, 1.0],
            [7.0, 4.0, 1.0],
            [6.0, 4.0, 2.0],
            [4.0, 2.0, 0.0],
        ];
        let candidates = distortions.map(|distortion| {
            [
                candidate(1, delta(1, 1), distortion[0]),
                candidate(2, delta(1, 1), distortion[1]),
                candidate(3, delta(1, 1), distortion[2]),
            ]
        });
        let groups = candidates.iter().map(group).collect::<Vec<_>>();
        let limit = budget(16, 16, delta(0, 0));

        let allocation = allocate_nested_profiles(
            &groups,
            &NestedProfileBudgets {
                compact: limit,
                near_lossless: limit,
            },
        )
        .expect("full-vector ties must not trigger an arbitrary work cap");

        assert_eq!(
            allocation.compact.plane_counts,
            brute_force_best(&groups, limit)
        );
    }

    #[test]
    fn scalable_regular_solver_resolves_full_vector_ties_without_a_cap() {
        let bundled = [
            candidate(1, delta(1, 1), 10.0),
            candidate(2, delta(1, 1), 9.0),
            candidate(3, delta(1, 1), 0.0),
        ];
        let units = [
            candidate(1, delta(1, 1), 10.0),
            candidate(2, delta(1, 1), 5.0),
            candidate(3, delta(1, 1), 0.0),
        ];
        let mut candidates = vec![bundled; 70];
        candidates.extend(vec![units; 70]);
        let groups = candidates.iter().map(group).collect::<Vec<_>>();
        let limit = budget(280, 280, delta(0, 0));

        let allocation = allocate_nested_profiles(
            &groups,
            &NestedProfileBudgets {
                compact: limit,
                near_lossless: limit,
            },
        )
        .expect("exact ties must have a deterministic winner");

        assert!(
            allocation.compact.plane_counts[..70]
                .iter()
                .all(|&planes| planes == 3)
        );
        assert!(
            allocation.compact.plane_counts[70..]
                .iter()
                .all(|&planes| planes == 1)
        );
    }

    #[test]
    fn equal_distortion_prefers_serialized_then_resident_bytes_before_plane_tie() {
        let candidates = [
            candidate(1, delta(1, 4), 5.0),
            candidate(2, delta(1, 0), 5.0),
            candidate(3, delta(0, 20), 5.0),
        ];
        let groups = [group(&candidates)];
        let budgets = NestedProfileBudgets {
            compact: budget(2, 24, delta(0, 0)),
            near_lossless: budget(2, 24, delta(0, 0)),
        };

        let allocation = allocate_nested_profiles(&groups, &budgets).expect("allocation");

        assert_eq!(allocation.compact.plane_counts, vec![1]);
        assert_eq!(allocation.compact.physical_bytes, bytes(1, 4));
    }

    #[test]
    fn regular_solver_does_not_spend_nonzero_bytes_on_zero_reduction_upgrades() {
        let candidates = [
            candidate(1, delta(1, 2), 5.0),
            candidate(2, delta(1, 2), 5.0),
            candidate(3, delta(1, 2), 5.0),
        ];
        let groups = [group(&candidates)];
        let limit = budget(3, 6, delta(0, 0));

        let allocation = allocate_nested_profiles(
            &groups,
            &NestedProfileBudgets {
                compact: limit,
                near_lossless: limit,
            },
        )
        .expect("regular allocation");

        assert_eq!(allocation.compact.plane_counts, vec![1]);
        assert_eq!(allocation.compact.physical_bytes, bytes(1, 2));
    }

    #[test]
    fn serialized_tie_priority_overrides_earlier_group_plane_vector() {
        let earlier = [
            candidate(1, delta(0, 0), 10.0),
            candidate(2, delta(2, 1), 5.0),
            candidate(3, delta(20, 20), 4.0),
        ];
        let later = [
            candidate(1, delta(0, 0), 10.0),
            candidate(2, delta(1, 10), 5.0),
            candidate(3, delta(20, 20), 4.0),
        ];
        let groups = [group(&earlier), group(&later)];
        let limit = budget(2, 10, delta(0, 0));

        let allocation = allocate_nested_profiles(
            &groups,
            &NestedProfileBudgets {
                compact: limit,
                near_lossless: limit,
            },
        )
        .expect("allocation");

        assert_eq!(allocation.compact.plane_counts, vec![1, 2]);
        assert_eq!(allocation.compact.physical_bytes, bytes(1, 10));
    }

    #[test]
    fn regular_concave_costs_scale_beyond_reference_frontier_bound_exactly() {
        let candidates = [
            candidate(1, delta(1, 1), 10.0),
            candidate(2, delta(1, 1), 6.0),
            candidate(3, delta(1, 1), 4.0),
        ];
        let groups = vec![group(&candidates); 700];
        let limit = budget(1_050, 1_050, delta(0, 0));

        let allocation = allocate_nested_profiles(
            &groups,
            &NestedProfileBudgets {
                compact: limit,
                near_lossless: limit,
            },
        )
        .expect("regular exact fast path");

        assert_eq!(allocation.compact.plane_counts.len(), 700);
        assert_eq!(
            allocation
                .compact
                .plane_counts
                .iter()
                .filter(|&&planes| planes == 2)
                .count(),
            350
        );
        assert!(
            allocation.compact.plane_counts[..350]
                .iter()
                .all(|&planes| planes == 2)
        );
        assert!(
            allocation.compact.plane_counts[350..]
                .iter()
                .all(|&planes| planes == 1)
        );
        assert_eq!(allocation.compact.physical_bytes, bytes(1_050, 1_050));
    }

    #[test]
    fn large_nonconcave_regular_case_is_solved_exactly_without_a_model_sized_dp() {
        let candidates = [
            candidate(1, delta(1, 1), 10.0),
            candidate(2, delta(1, 1), 9.0),
            candidate(3, delta(1, 1), 0.0),
        ];
        let group_count = 100_001;
        let groups = vec![group(&candidates); group_count];
        let limit = budget(
            (group_count * 2) as u64,
            (group_count * 2) as u64,
            delta(0, 0),
        );

        let allocation = allocate_nested_profiles(
            &groups,
            &NestedProfileBudgets {
                compact: limit,
                near_lossless: limit,
            },
        )
        .expect("uniform non-concave allocation is exactly scalable");

        assert_eq!(
            allocation
                .compact
                .plane_counts
                .iter()
                .filter(|&&planes| planes == 3)
                .count(),
            group_count / 2
        );
        assert_eq!(
            allocation
                .compact
                .plane_counts
                .iter()
                .filter(|&&planes| planes == 2)
                .count(),
            1
        );
        assert_eq!(
            allocation.compact.physical_bytes,
            bytes((group_count * 2) as u64, (group_count * 2) as u64)
        );
        assert!(allocation.compact.is_prefix_of(&allocation.near_lossless));
    }

    #[test]
    fn nonconcave_uniform_cost_solver_matches_global_brute_force_at_every_capacity() {
        let a = [
            candidate(1, delta(1, 2), 20.0),
            candidate(2, delta(1, 2), 18.9),
            candidate(3, delta(1, 2), 1.0),
        ];
        let b = [
            candidate(1, delta(1, 2), 17.0),
            candidate(2, delta(1, 2), 9.7),
            candidate(3, delta(1, 2), 8.8),
        ];
        let c = [
            candidate(1, delta(1, 2), 15.0),
            candidate(2, delta(1, 2), 13.7),
            candidate(3, delta(1, 2), 0.2),
        ];
        let d = [
            candidate(1, delta(1, 2), 12.0),
            candidate(2, delta(1, 2), 4.9),
            candidate(3, delta(1, 2), 1.1),
        ];
        let groups = [group(&a), group(&b), group(&c), group(&d)];

        for optional_planes in 0..=groups.len() * 2 {
            let limit = budget(
                (groups.len() + optional_planes) as u64,
                ((groups.len() + optional_planes) * 2) as u64,
                delta(0, 0),
            );
            let allocation = allocate_nested_profiles(
                &groups,
                &NestedProfileBudgets {
                    compact: limit,
                    near_lossless: limit,
                },
            )
            .expect("exact uniform-cost allocation");

            assert_eq!(
                allocation.compact.plane_counts,
                brute_force_best(&groups, limit),
                "optional-plane capacity {optional_planes}"
            );
        }
    }

    #[test]
    fn uniform_cost_decomposition_matches_brute_force_across_mixed_marginals() {
        let mut seed = 0x9e37_79b9_u64;
        for case in 0..128 {
            let mut candidates = Vec::new();
            for _ in 0..5 {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let third = (seed % 5) as f64;
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let second = third + (seed % 11) as f64;
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let first = second + (seed % 11) as f64;
                candidates.push([
                    candidate(1, delta(2, 3), first),
                    candidate(2, delta(2, 3), second),
                    candidate(3, delta(2, 3), third),
                ]);
            }
            let groups: Vec<_> = candidates.iter().map(group).collect();

            for optional_planes in 0..=groups.len() * 2 {
                let limit = budget(
                    ((groups.len() + optional_planes) * 2) as u64,
                    ((groups.len() + optional_planes) * 3) as u64,
                    delta(0, 0),
                );
                let allocation = allocate_nested_profiles(
                    &groups,
                    &NestedProfileBudgets {
                        compact: limit,
                        near_lossless: limit,
                    },
                )
                .expect("exact regular allocation");
                let oracle = brute_force_best(&groups, limit);
                let oracle_distortion = total_distortion(&groups, &oracle).unwrap();

                assert_eq!(
                    allocation.compact.total_distortion, oracle_distortion,
                    "case {case}, optional-plane capacity {optional_planes}"
                );
                assert!(allocation.compact.physical_bytes.fits_within(limit.maximum));
            }
        }
    }

    #[test]
    fn scalable_uniform_cost_solver_matches_brute_force_across_mixed_marginals() {
        let mut seed = 0xd1b5_4a32_d192_ed03_u64;
        for case in 0..64 {
            let mut candidates = Vec::new();
            for group_index in 0..8_u64 {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let first_reduction = ((seed % 127) * 256 + group_index * 2 + 1) as f64 / 256.0;
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let second_reduction = ((seed % 127) * 256 + group_index * 2 + 2) as f64 / 256.0;
                candidates.push([
                    candidate(1, delta(1, 1), first_reduction + second_reduction),
                    candidate(2, delta(1, 1), second_reduction),
                    candidate(3, delta(1, 1), 0.0),
                ]);
            }
            let groups: Vec<_> = candidates.iter().map(group).collect();

            for optional_planes in 0..=groups.len() * 2 {
                let maximum = (groups.len() + optional_planes) as u64;
                let limit = budget(maximum, maximum, delta(0, 0));
                let allocation = allocate_nested_profiles(
                    &groups,
                    &NestedProfileBudgets {
                        compact: limit,
                        near_lossless: limit,
                    },
                )
                .expect("certifiable scalable allocation");

                assert_eq!(
                    allocation.compact.plane_counts,
                    brute_force_best_scaled_u128(&groups, limit, 256),
                    "case {case}, optional-plane capacity {optional_planes}"
                );
            }
        }
    }

    #[test]
    fn uniform_cost_frontier_matches_brute_force_oracle() {
        let a = [
            candidate(1, delta(2, 3), 10.0),
            candidate(2, delta(2, 3), 6.0),
            candidate(3, delta(2, 3), 4.0),
        ];
        let b = [
            candidate(1, delta(2, 3), 8.0),
            candidate(2, delta(2, 3), 5.0),
            candidate(3, delta(2, 3), 4.0),
        ];
        let c = [
            candidate(1, delta(2, 3), 7.0),
            candidate(2, delta(2, 3), 5.0),
            candidate(3, delta(2, 3), 4.5),
        ];
        let groups = [group(&a), group(&b), group(&c)];
        let limit = budget(10, 15, delta(0, 0));
        let budgets = NestedProfileBudgets {
            compact: limit,
            near_lossless: limit,
        };

        let allocation = allocate_nested_profiles(&groups, &budgets).unwrap();
        let oracle = brute_force_best(&groups, limit);

        assert_eq!(allocation.compact.plane_counts, oracle);
    }

    #[test]
    fn variable_cost_frontier_matches_the_global_brute_force_oracle() {
        let a = [
            candidate(1, delta(0, 0), 20.0),
            candidate(2, delta(6, 6), 13.0),
            candidate(3, delta(100, 100), 12.0),
        ];
        let b = [
            candidate(1, delta(0, 0), 20.0),
            candidate(2, delta(10, 10), 9.0),
            candidate(3, delta(100, 100), 8.0),
        ];
        let groups = [group(&a), group(&b)];
        let limit = budget(10, 10, delta(0, 0));
        let budgets = NestedProfileBudgets {
            compact: limit,
            near_lossless: limit,
        };

        let allocation = allocate_nested_profiles(&groups, &budgets).unwrap();
        let oracle = brute_force_best(&groups, limit);

        assert_eq!(oracle, vec![1, 2]);
        assert_eq!(allocation.compact.plane_counts, oracle);
    }

    #[test]
    fn compact_is_always_a_plane_prefix_of_near_lossless() {
        let a = [
            candidate(1, delta(2, 3), 12.0),
            candidate(2, delta(2, 3), 5.0),
            candidate(3, delta(2, 3), 1.0),
        ];
        let b = [
            candidate(1, delta(2, 3), 10.0),
            candidate(2, delta(2, 3), 7.0),
            candidate(3, delta(2, 3), 2.0),
        ];
        let groups = [group(&a), group(&b)];
        let budgets = NestedProfileBudgets {
            compact: budget(6, 9, delta(0, 0)),
            near_lossless: budget(12, 18, delta(0, 0)),
        };

        let allocation = allocate_nested_profiles(&groups, &budgets).unwrap();

        assert!(allocation.compact.is_prefix_of(&allocation.near_lossless));
        assert!(
            allocation
                .compact
                .plane_counts
                .iter()
                .zip(&allocation.near_lossless.plane_counts)
                .all(|(compact, near)| compact <= near)
        );
    }

    fn brute_force_best(groups: &[GroupCandidates<'_>], limit: ProfileBudget) -> Vec<u8> {
        let mut best: Option<(ExactDistortion, PhysicalBytes, Vec<u8>)> = None;
        let combinations = 3usize.pow(groups.len() as u32);
        for encoded in 0..combinations {
            let mut cursor = encoded;
            let mut counts = Vec::with_capacity(groups.len());
            let mut physical = limit.metadata.effective();
            let mut distortion = ExactDistortion::ZERO;
            for group in groups {
                let planes = (cursor % 3 + 1) as u8;
                cursor /= 3;
                counts.push(planes);
                for candidate in &group.candidates[..usize::from(planes)] {
                    physical = physical
                        .checked_add(candidate.byte_delta.effective())
                        .unwrap();
                }
                distortion = distortion
                    .checked_add_f64(group.candidates[usize::from(planes - 1)].distortion)
                    .unwrap();
            }
            if !physical.fits_within(limit.maximum) {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|(best_distortion, best_bytes, best_counts)| {
                    distortion < *best_distortion
                        || (distortion == *best_distortion
                            && (physical.serialized < best_bytes.serialized
                                || (physical.serialized == best_bytes.serialized
                                    && (physical.resident < best_bytes.resident
                                        || (physical.resident == best_bytes.resident
                                            && counts > *best_counts)))))
                })
            {
                best = Some((distortion, physical, counts));
            }
        }
        best.unwrap().2
    }

    fn brute_force_best_scaled_u128(
        groups: &[GroupCandidates<'_>],
        limit: ProfileBudget,
        scale: u64,
    ) -> Vec<u8> {
        let mut best: Option<(u128, PhysicalBytes, Vec<u8>)> = None;
        let combinations = 3usize.pow(groups.len() as u32);
        for encoded in 0..combinations {
            let mut cursor = encoded;
            let mut counts = Vec::with_capacity(groups.len());
            let mut physical = limit.metadata.effective();
            let mut distortion = 0_u128;
            for group in groups {
                let planes = (cursor % 3 + 1) as u8;
                cursor /= 3;
                counts.push(planes);
                for candidate in &group.candidates[..usize::from(planes)] {
                    physical = physical
                        .checked_add(candidate.byte_delta.effective())
                        .unwrap();
                }
                let scaled = group.candidates[usize::from(planes - 1)].distortion * scale as f64;
                assert!(scaled.is_finite() && scaled >= 0.0 && scaled.fract() == 0.0);
                distortion = distortion.checked_add(scaled as u128).unwrap();
            }
            if !physical.fits_within(limit.maximum) {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|(best_distortion, best_bytes, best_counts)| {
                    distortion < *best_distortion
                        || (distortion == *best_distortion
                            && (physical.serialized < best_bytes.serialized
                                || (physical.serialized == best_bytes.serialized
                                    && (physical.resident < best_bytes.resident
                                        || (physical.resident == best_bytes.resident
                                            && counts > *best_counts)))))
                })
            {
                best = Some((distortion, physical, counts));
            }
        }
        best.unwrap().2
    }
}
