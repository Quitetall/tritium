//! Compact replay-oriented allocation for uniform full-tile physical costs.

use core::fmt;

use super::{
    ExactDistortion, ExactPrefixIndex, ExactReduction, LexGroupIndex, PhysicalAllocError,
    RankedBundledUpgrade, RankedUnitUpgrade, RankedUpgrade, RegularIndexes, SaltV2Profile,
    best_regular_choice, compare_reduction_descending, exact_reduction, regular_reduction,
};

#[derive(Clone, Copy, Debug)]
struct CompactUnitUpgrade {
    distortion_reduction: ExactReduction,
    group: u32,
    target_planes: u8,
}

impl RankedUpgrade for CompactUnitUpgrade {
    fn group(&self) -> usize {
        self.group as usize
    }

    fn reduction(&self) -> ExactReduction {
        self.distortion_reduction
    }
}

impl RankedUnitUpgrade for CompactUnitUpgrade {
    fn target_planes(&self) -> u8 {
        self.target_planes
    }
}

#[derive(Clone, Copy, Debug)]
struct CompactBundledUpgrade {
    distortion_reduction: ExactReduction,
    first_plane_reduction: ExactReduction,
    group: u32,
}

impl RankedUpgrade for CompactBundledUpgrade {
    fn group(&self) -> usize {
        self.group as usize
    }

    fn reduction(&self) -> ExactReduction {
        self.distortion_reduction
    }
}

impl RankedBundledUpgrade for CompactBundledUpgrade {
    fn first_plane_reduction(&self) -> ExactReduction {
        self.first_plane_reduction
    }
}

/// Canonical two-bit plane counts (`00 => 1`, `01 => 2`, `10 => 3`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedPlaneCounts {
    bytes: Vec<u8>,
    len: u64,
}

impl PackedPlaneCounts {
    /// Allocate one repeated valid count without a byte-per-tile vector.
    pub fn filled(len: u64, count: u8, profile: SaltV2Profile) -> Result<Self, PhysicalAllocError> {
        if len == 0 {
            return Err(PhysicalAllocError::EmptyGroups);
        }
        if !(1..=3).contains(&count) {
            return Err(PhysicalAllocError::PlaneOrdinal {
                group: 0,
                expected: 1,
                actual: count,
            });
        }
        let bytes_u64 = len
            .checked_add(3)
            .ok_or(PhysicalAllocError::AccountingOverflow { profile })?
            / 4;
        let bytes = usize::try_from(bytes_u64)
            .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
        let code = count - 1;
        let repeated = code | code << 2 | code << 4 | code << 6;
        let mut packed = Vec::new();
        packed
            .try_reserve_exact(bytes)
            .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
        packed.resize(bytes, repeated);
        if !len.is_multiple_of(4) {
            let valid_bits = (len % 4) * 2;
            let mask = (1u8 << valid_bits) - 1;
            if let Some(last) = packed.last_mut() {
                *last &= mask;
            }
        }
        Ok(Self { bytes: packed, len })
    }

    /// Number of represented allocation groups.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.len
    }

    /// Whether no groups are represented.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Packed canonical map payload.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Selected count at one global group ordinal.
    #[must_use]
    pub fn get(&self, index: u64) -> Option<u8> {
        if index >= self.len {
            return None;
        }
        let byte = usize::try_from(index / 4).ok()?;
        let shift = (index % 4) * 2;
        Some(((self.bytes[byte] >> shift) & 0b11) + 1)
    }

    fn set(&mut self, index: u64, count: u8) {
        debug_assert!(index < self.len && (1..=3).contains(&count));
        let byte = usize::try_from(index / 4).expect("packed map fits host memory");
        let shift = ((index % 4) * 2) as u32;
        self.bytes[byte] &= !(0b11 << shift);
        self.bytes[byte] |= (count - 1) << shift;
    }

    /// Sum all represented plane counts without unpacking the map.
    #[must_use]
    pub fn present_planes(&self) -> u64 {
        (0..self.len)
            .map(|index| u64::from(self.get(index).expect("in-range packed count")))
            .sum()
    }
}

/// One validated P1/P2/P3 cumulative distortion curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformPrefixCurve {
    distortions: [f64; 3],
}

impl UniformPrefixCurve {
    /// Construct a finite, non-negative, non-increasing prefix curve.
    pub fn new(distortions: [f64; 3]) -> Result<Self, PhysicalAllocError> {
        for (index, distortion) in distortions.into_iter().enumerate() {
            if !distortion.is_finite() || distortion < 0.0 {
                return Err(PhysicalAllocError::InvalidDistortion {
                    group: 0,
                    planes: (index + 1) as u8,
                    distortion,
                });
            }
            if index > 0 && distortion > distortions[index - 1] {
                return Err(PhysicalAllocError::NonMonotoneDistortion {
                    group: 0,
                    planes: (index + 1) as u8,
                });
            }
        }
        Ok(Self { distortions })
    }

    /// Cumulative distortion after one through three planes.
    #[must_use]
    pub const fn distortions(self) -> [f64; 3] {
        self.distortions
    }
}

/// Exact compact allocation result for one physical profile.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedUniformProfileAllocation {
    /// Canonical selected count map.
    pub plane_counts: PackedPlaneCounts,
    /// Number of equal-cost refinements selected above the supplied floor.
    pub selected_upgrades: u64,
    /// Total exact-sum distortion rounded once to binary64.
    pub total_distortion: f64,
    /// Sum of selected planes across every group.
    pub present_planes: u64,
}

/// Failure from compact uniform-cost profile allocation.
#[derive(Debug)]
pub enum UniformProfileAllocError<E> {
    /// Curve source failed at its own boundary.
    Source(E),
    /// Exact allocation validation, accounting, or memory failed.
    Allocation(PhysicalAllocError),
    /// Supplied floor map did not cover the declared group count.
    FloorLength {
        /// Declared curve count.
        expected: u64,
        /// Packed floor count.
        actual: u64,
    },
    /// Curve source ended before the declared count.
    CurveSourceShort {
        /// Declared curve count.
        expected: u64,
        /// Curves observed before EOF.
        actual: u64,
    },
    /// Curve source produced at least one record after the declared count.
    CurveSourceLong {
        /// Declared exact curve count.
        expected: u64,
    },
    /// Compact record indices cannot represent this many groups.
    TooManyGroups {
        /// Declared group count.
        groups: u64,
    },
}

impl<E: fmt::Display> fmt::Display for UniformProfileAllocError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(error) => write!(formatter, "uniform curve source failed: {error}"),
            Self::Allocation(error) => write!(formatter, "uniform allocation failed: {error}"),
            Self::FloorLength { expected, actual } => write!(
                formatter,
                "uniform floor has {actual} groups, expected {expected}"
            ),
            Self::CurveSourceShort { expected, actual } => write!(
                formatter,
                "uniform curve source ended at {actual}, expected {expected}"
            ),
            Self::CurveSourceLong { expected } => write!(
                formatter,
                "uniform curve source contains records after declared count {expected}"
            ),
            Self::TooManyGroups { groups } => {
                write!(
                    formatter,
                    "uniform compact allocator cannot index {groups} groups"
                )
            }
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for UniformProfileAllocError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            Self::Allocation(error) => Some(error),
            Self::FloorLength { .. }
            | Self::CurveSourceShort { .. }
            | Self::CurveSourceLong { .. }
            | Self::TooManyGroups { .. } => None,
        }
    }
}

impl<E> From<PhysicalAllocError> for UniformProfileAllocError<E> {
    fn from(error: PhysicalAllocError) -> Self {
        Self::Allocation(error)
    }
}

/// Incremental exact allocator for callback-driven or decoded curve sources.
///
/// The planner accepts one curve at a time, so campaign stores can verify and
/// decode master records directly into allocation without materializing a
/// second model-sized curve collection.
#[derive(Debug)]
pub struct PackedUniformProfilePlanner<'floor> {
    expected_groups: u64,
    floors: &'floor PackedPlaneCounts,
    additional_capacity: u64,
    profile: SaltV2Profile,
    observed_groups: u64,
    unit_upgrades: Vec<CompactUnitUpgrade>,
    bundled_upgrades: Vec<CompactBundledUpgrade>,
    floor_distortion: ExactDistortion,
    available_upgrades: usize,
}

impl<'floor> PackedUniformProfilePlanner<'floor> {
    /// Start a compact allocation against one exact packed floor.
    pub fn new(
        expected_groups: u64,
        floors: &'floor PackedPlaneCounts,
        additional_capacity: u64,
        profile: SaltV2Profile,
    ) -> Result<Self, UniformProfileAllocError<core::convert::Infallible>> {
        if floors.len != expected_groups {
            return Err(UniformProfileAllocError::FloorLength {
                expected: expected_groups,
                actual: floors.len,
            });
        }
        let _ = u32::try_from(expected_groups).map_err(|_| {
            UniformProfileAllocError::TooManyGroups {
                groups: expected_groups,
            }
        })?;
        Ok(Self {
            expected_groups,
            floors,
            additional_capacity,
            profile,
            observed_groups: 0,
            unit_upgrades: Vec::new(),
            bundled_upgrades: Vec::new(),
            floor_distortion: ExactDistortion::ZERO,
            available_upgrades: 0,
        })
    }

    /// Add the next global curve in canonical tensor/tile order.
    pub fn push(
        &mut self,
        curve: UniformPrefixCurve,
    ) -> Result<(), UniformProfileAllocError<core::convert::Infallible>> {
        if self.observed_groups == self.expected_groups {
            return Err(UniformProfileAllocError::CurveSourceLong {
                expected: self.expected_groups,
            });
        }
        let group = usize::try_from(self.observed_groups).map_err(|_| {
            UniformProfileAllocError::TooManyGroups {
                groups: self.expected_groups,
            }
        })?;
        validate_curve_at(curve, group)?;
        let floor =
            self.floors
                .get(self.observed_groups)
                .ok_or(UniformProfileAllocError::FloorLength {
                    expected: self.expected_groups,
                    actual: self.observed_groups,
                })?;
        self.floor_distortion = self
            .floor_distortion
            .checked_add_f64(curve.distortions[usize::from(floor - 1)])
            .ok_or(PhysicalAllocError::AccountingOverflow {
                profile: self.profile,
            })?;
        self.available_upgrades = self
            .available_upgrades
            .checked_add(usize::from(3 - floor))
            .ok_or(PhysicalAllocError::AccountingOverflow {
                profile: self.profile,
            })?;
        let compact_group = u32::try_from(group).expect("planner group bound checked at creation");
        match floor {
            1 => {
                let first =
                    exact_reduction(curve.distortions[0], curve.distortions[1], self.profile)?;
                let second =
                    exact_reduction(curve.distortions[1], curve.distortions[2], self.profile)?;
                if second.cmp(first) == core::cmp::Ordering::Greater {
                    self.bundled_upgrades.try_reserve(1).map_err(|_| {
                        PhysicalAllocError::WorkingMemoryUnavailable {
                            profile: self.profile,
                        }
                    })?;
                    self.bundled_upgrades.push(CompactBundledUpgrade {
                        distortion_reduction: exact_reduction(
                            curve.distortions[0],
                            curve.distortions[2],
                            self.profile,
                        )?,
                        first_plane_reduction: first,
                        group: compact_group,
                    });
                } else {
                    push_unit(
                        &mut self.unit_upgrades,
                        compact_group,
                        2,
                        first,
                        self.profile,
                    )?;
                    push_unit(
                        &mut self.unit_upgrades,
                        compact_group,
                        3,
                        second,
                        self.profile,
                    )?;
                }
            }
            2 => push_unit(
                &mut self.unit_upgrades,
                compact_group,
                3,
                exact_reduction(curve.distortions[1], curve.distortions[2], self.profile)?,
                self.profile,
            )?,
            3 => {}
            _ => unreachable!("packed counts admit only one through three"),
        }
        self.observed_groups += 1;
        Ok(())
    }

    /// Solve and materialize the canonical packed map after exact source coverage.
    pub fn finish(
        mut self,
    ) -> Result<PackedUniformProfileAllocation, UniformProfileAllocError<core::convert::Infallible>>
    {
        if self.observed_groups != self.expected_groups {
            return Err(UniformProfileAllocError::CurveSourceShort {
                expected: self.expected_groups,
                actual: self.observed_groups,
            });
        }
        self.unit_upgrades.sort_unstable_by(|left, right| {
            compare_reduction_descending(left.distortion_reduction, right.distortion_reduction)
                .then_with(|| left.group.cmp(&right.group))
                .then_with(|| left.target_planes.cmp(&right.target_planes))
        });
        self.bundled_upgrades.sort_unstable_by(|left, right| {
            compare_reduction_descending(left.distortion_reduction, right.distortion_reduction)
                .then_with(|| left.group.cmp(&right.group))
        });
        finish_uniform_profile(
            self.floors,
            self.additional_capacity,
            self.profile,
            self.floor_distortion,
            self.available_upgrades,
            self.unit_upgrades,
            self.bundled_upgrades,
        )
        .map_err(UniformProfileAllocError::Allocation)
    }
}

/// Allocate one exact equal-increment profile from a replayed curve stream.
///
/// This avoids `GroupCandidates` and a byte-per-group output vector. It retains
/// only compact ranked upgrades plus the two-bit floor/result maps, then reuses
/// the same exact non-concave-chain solver and tie policy as
/// [`super::allocate_nested_profiles`]. `additional_capacity` is an exact count
/// of physically interchangeable plane deltas derived by the package rate model.
pub fn allocate_uniform_profile_packed<E>(
    expected_groups: u64,
    floors: &PackedPlaneCounts,
    additional_capacity: u64,
    profile: SaltV2Profile,
    curves: impl IntoIterator<Item = Result<UniformPrefixCurve, E>>,
) -> Result<PackedUniformProfileAllocation, UniformProfileAllocError<E>> {
    let mut planner =
        PackedUniformProfilePlanner::new(expected_groups, floors, additional_capacity, profile)
            .map_err(convert_infallible_error)?;
    let mut curves = curves.into_iter();
    for ordinal in 0..expected_groups {
        let curve = match curves.next() {
            Some(Ok(curve)) => curve,
            Some(Err(error)) => return Err(UniformProfileAllocError::Source(error)),
            None => {
                return Err(UniformProfileAllocError::CurveSourceShort {
                    expected: expected_groups,
                    actual: ordinal,
                });
            }
        };
        planner.push(curve).map_err(convert_infallible_error)?;
    }
    match curves.next() {
        Some(Ok(_)) => {
            return Err(UniformProfileAllocError::CurveSourceLong {
                expected: expected_groups,
            });
        }
        Some(Err(error)) => return Err(UniformProfileAllocError::Source(error)),
        None => {}
    }
    planner.finish().map_err(convert_infallible_error)
}

#[allow(clippy::too_many_arguments)]
fn finish_uniform_profile(
    floors: &PackedPlaneCounts,
    additional_capacity: u64,
    profile: SaltV2Profile,
    floor_distortion: ExactDistortion,
    available_upgrades: usize,
    unit_upgrades: Vec<CompactUnitUpgrade>,
    bundled_upgrades: Vec<CompactBundledUpgrade>,
) -> Result<PackedUniformProfileAllocation, PhysicalAllocError> {
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
    let capacity = usize::try_from(additional_capacity)
        .unwrap_or(usize::MAX)
        .min(available_upgrades);
    let choice = best_regular_choice(&indexes, capacity, profile)?;
    let selected_upgrades = choice
        .hull
        .upgrades()
        .checked_add(usize::from(choice.exceptional_bundle_rank.is_some()))
        .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
    let reduction = regular_reduction(choice, &indexes, profile)?;
    let total_distortion = floor_distortion
        .checked_sub(reduction)
        .and_then(ExactDistortion::to_f64)
        .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
    let mut plane_counts = floors.clone();
    for upgrade in unit_upgrades.iter().take(choice.hull.unit_count) {
        let group = u64::from(upgrade.group);
        debug_assert_eq!(
            plane_counts.get(group).map(|count| count + 1),
            Some(upgrade.target_planes)
        );
        plane_counts.set(group, upgrade.target_planes);
    }
    let take = choice.hull.bundle_count + usize::from(choice.hull.replacement_after_exclusion);
    for (rank, upgrade) in bundled_upgrades.iter().take(take).enumerate() {
        if choice.exceptional_bundle_rank != Some(rank) {
            plane_counts.set(u64::from(upgrade.group), 3);
        }
    }
    if let Some(rank) = choice.exceptional_bundle_rank {
        plane_counts.set(u64::from(bundled_upgrades[rank].group), 2);
    }
    let present_planes = plane_counts.present_planes();
    Ok(PackedUniformProfileAllocation {
        plane_counts,
        selected_upgrades: selected_upgrades as u64,
        total_distortion,
        present_planes,
    })
}

fn convert_infallible_error<E>(
    error: UniformProfileAllocError<core::convert::Infallible>,
) -> UniformProfileAllocError<E> {
    match error {
        UniformProfileAllocError::Source(source) => match source {},
        UniformProfileAllocError::Allocation(error) => UniformProfileAllocError::Allocation(error),
        UniformProfileAllocError::FloorLength { expected, actual } => {
            UniformProfileAllocError::FloorLength { expected, actual }
        }
        UniformProfileAllocError::CurveSourceShort { expected, actual } => {
            UniformProfileAllocError::CurveSourceShort { expected, actual }
        }
        UniformProfileAllocError::CurveSourceLong { expected } => {
            UniformProfileAllocError::CurveSourceLong { expected }
        }
        UniformProfileAllocError::TooManyGroups { groups } => {
            UniformProfileAllocError::TooManyGroups { groups }
        }
    }
}

fn push_unit(
    upgrades: &mut Vec<CompactUnitUpgrade>,
    group: u32,
    target_planes: u8,
    reduction: ExactReduction,
    profile: SaltV2Profile,
) -> Result<(), PhysicalAllocError> {
    if reduction.is_zero() {
        return Ok(());
    }
    upgrades
        .try_reserve(1)
        .map_err(|_| PhysicalAllocError::WorkingMemoryUnavailable { profile })?;
    upgrades.push(CompactUnitUpgrade {
        distortion_reduction: reduction,
        group,
        target_planes,
    });
    Ok(())
}

fn validate_curve_at<E>(
    curve: UniformPrefixCurve,
    group: usize,
) -> Result<(), UniformProfileAllocError<E>> {
    for (index, distortion) in curve.distortions.into_iter().enumerate() {
        if !distortion.is_finite() || distortion < 0.0 {
            return Err(PhysicalAllocError::InvalidDistortion {
                group,
                planes: (index + 1) as u8,
                distortion,
            }
            .into());
        }
        if index > 0 && distortion > curve.distortions[index - 1] {
            return Err(PhysicalAllocError::NonMonotoneDistortion {
                group,
                planes: (index + 1) as u8,
            }
            .into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;

    use super::*;
    use crate::salt_v2_allocator::{
        ByteDelta, GroupCandidates, NestedProfileBudgets, PhysicalBytes, PlaneCandidate,
        ProfileBudget, allocate_nested_profiles,
    };

    fn curve(values: [f64; 3]) -> UniformPrefixCurve {
        UniformPrefixCurve::new(values).expect("curve")
    }

    #[test]
    fn packed_uniform_path_matches_reference_for_concave_and_nonconcave_curves() {
        let curves = [
            curve([9.0, 5.0, 4.0]),
            curve([8.0, 7.0, 1.0]),
            curve([6.0, 3.0, 0.5]),
            curve([4.0, 4.0, 4.0]),
        ];
        let candidates = curves
            .iter()
            .map(|curve| {
                [
                    PlaneCandidate {
                        planes: 1,
                        byte_delta: ByteDelta::declared(PhysicalBytes {
                            serialized: 1,
                            resident: 1,
                        }),
                        distortion: curve.distortions[0],
                    },
                    PlaneCandidate {
                        planes: 2,
                        byte_delta: ByteDelta::declared(PhysicalBytes {
                            serialized: 1,
                            resident: 1,
                        }),
                        distortion: curve.distortions[1],
                    },
                    PlaneCandidate {
                        planes: 3,
                        byte_delta: ByteDelta::declared(PhysicalBytes {
                            serialized: 1,
                            resident: 1,
                        }),
                        distortion: curve.distortions[2],
                    },
                ]
            })
            .collect::<Vec<_>>();
        let groups = candidates
            .iter()
            .map(|candidates| GroupCandidates { candidates })
            .collect::<Vec<_>>();

        for capacity in 0..=8u64 {
            let maximum = PhysicalBytes {
                serialized: curves.len() as u64 + capacity,
                resident: curves.len() as u64 + capacity,
            };
            let budgets = NestedProfileBudgets {
                compact: ProfileBudget {
                    maximum,
                    metadata: ByteDelta::declared(PhysicalBytes::ZERO),
                },
                near_lossless: ProfileBudget {
                    maximum,
                    metadata: ByteDelta::declared(PhysicalBytes::ZERO),
                },
            };
            let reference = allocate_nested_profiles(&groups, &budgets).expect("reference");
            let floors =
                PackedPlaneCounts::filled(curves.len() as u64, 1, SaltV2Profile::CompactV1)
                    .expect("floors");
            let compact = allocate_uniform_profile_packed(
                curves.len() as u64,
                &floors,
                capacity,
                SaltV2Profile::CompactV1,
                curves.iter().copied().map(Ok::<_, Infallible>),
            )
            .expect("compact");
            let actual = (0..curves.len())
                .map(|index| compact.plane_counts.get(index as u64).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                actual, reference.compact.plane_counts,
                "capacity {capacity}"
            );
            assert_eq!(compact.total_distortion, reference.compact.total_distortion);
        }
    }

    #[test]
    fn packed_counts_use_two_bits_and_reject_source_length_drift() {
        let floors = PackedPlaneCounts::filled(5, 1, SaltV2Profile::CompactV1).expect("floors");
        assert_eq!(floors.as_bytes().len(), 2);
        assert_eq!(
            (0..5).map(|index| floors.get(index)).collect::<Vec<_>>(),
            vec![Some(1); 5]
        );
        assert_eq!(floors.as_bytes()[1] & 0b1111_1100, 0);

        let short = allocate_uniform_profile_packed(
            5,
            &floors,
            1,
            SaltV2Profile::CompactV1,
            [Ok::<_, Infallible>(curve([3.0, 2.0, 1.0]))],
        );
        assert!(matches!(
            short,
            Err(UniformProfileAllocError::CurveSourceShort {
                expected: 5,
                actual: 1
            })
        ));
    }

    #[test]
    fn ranked_records_keep_model_scale_ordinals_compact() {
        assert_eq!(core::mem::size_of::<ExactReduction>(), 16);
        assert_eq!(core::mem::size_of::<CompactUnitUpgrade>(), 24);
        assert_eq!(core::mem::size_of::<CompactBundledUpgrade>(), 40);
    }
}
