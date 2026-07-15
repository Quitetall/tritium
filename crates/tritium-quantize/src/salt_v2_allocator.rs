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
    /// The scalable equal-cost solver encountered rounded arithmetic that could
    /// change the exact distortion ordering. Small inputs use the reference
    /// frontier instead; model-sized inputs fail closed.
    NumericallyAmbiguousFastPath {
        /// Profile whose scalable solver could not certify.
        profile: SaltV2Profile,
    },
    /// Exact lexicographic tie resolution would exceed its linear-time work budget.
    ScalableTieLimit {
        /// Profile whose scalable solver encountered the tie pattern.
        profile: SaltV2Profile,
        /// Number of full-vector comparisons requested.
        comparisons: usize,
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
            Self::NumericallyAmbiguousFastPath { profile } => write!(
                f,
                "{profile:?} equal-cost allocation has non-exact floating-point reductions"
            ),
            Self::ScalableTieLimit {
                profile,
                comparisons,
            } => write!(
                f,
                "{profile:?} equal-cost allocation needs {comparisons} full-vector tie comparisons"
            ),
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

    let compact_floor = vec![1; groups.len()];
    let compact = allocate_profile(
        groups,
        budgets.compact,
        compact_floor,
        SaltV2Profile::CompactV1,
    )?;
    let near_lossless = allocate_profile(
        groups,
        budgets.near_lossless,
        compact.plane_counts.clone(),
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
    let mut total = 0.0;
    for (group, &planes) in groups.iter().zip(plane_counts) {
        total += distortion_at(*group, planes);
        if !total.is_finite() {
            return None;
        }
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

#[derive(Clone, Debug)]
struct ExactState {
    bytes: PhysicalBytes,
    distortion: f64,
    plane_counts: Vec<u8>,
}

fn exact_plane_counts(
    groups: &[GroupCandidates<'_>],
    floors: &[u8],
    limit: ProfileBudget,
    profile: SaltV2Profile,
) -> Result<Vec<u8>, PhysicalAllocError> {
    if reference_frontier_fits(floors) {
        return bounded_exact_plane_counts(groups, floors, limit, profile);
    }
    if let Some(plane_counts) = regular_plane_counts(groups, floors, limit, profile)? {
        return Ok(plane_counts);
    }

    bounded_exact_plane_counts(groups, floors, limit, profile)
}

fn reference_frontier_fits(floors: &[u8]) -> bool {
    floors
        .iter()
        .try_fold(1usize, |states, &floor| {
            states.checked_mul(usize::from(SALT_V2_PLANES as u8 - floor + 1))
        })
        .is_some_and(|states| states <= MAX_EXACT_FRONTIER_STATES)
}

fn bounded_exact_plane_counts(
    groups: &[GroupCandidates<'_>],
    floors: &[u8],
    limit: ProfileBudget,
    profile: SaltV2Profile,
) -> Result<Vec<u8>, PhysicalAllocError> {
    let metadata = limit.metadata.effective();
    let mut frontier = vec![ExactState {
        bytes: metadata,
        distortion: 0.0,
        plane_counts: Vec::with_capacity(groups.len()),
    }];

    for (group_index, group) in groups.iter().copied().enumerate() {
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
            .map_err(|_| PhysicalAllocError::StateSpaceTooLarge {
                profile,
                states: expanded,
            })?;
        for state in &frontier {
            for planes in floors[group_index]..=SALT_V2_PLANES as u8 {
                let delta = bundle_bytes(group, 0, planes, ByteMode::Effective)
                    .ok_or(PhysicalAllocError::AccountingOverflow { profile })?;
                let Some(bytes) = state.bytes.checked_add(delta) else {
                    continue;
                };
                if !bytes.fits_within(limit.maximum) {
                    continue;
                }
                let distortion = state.distortion + distortion_at(group, planes);
                if !distortion.is_finite() {
                    return Err(PhysicalAllocError::AccountingOverflow { profile });
                }
                let mut plane_counts = state.plane_counts.clone();
                plane_counts.push(planes);
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
                required: minimum_bytes_for_prefix(groups, floors, limit.metadata, profile)?,
                maximum: limit.maximum,
            });
        }
        frontier = pareto_prune(next);
    }

    frontier
        .into_iter()
        .min_by(|left, right| {
            left.distortion
                .total_cmp(&right.distortion)
                .then_with(|| left.bytes.serialized.cmp(&right.bytes.serialized))
                .then_with(|| left.bytes.resident.cmp(&right.bytes.resident))
                .then_with(|| right.plane_counts.cmp(&left.plane_counts))
        })
        .map(|state| state.plane_counts)
        .ok_or(PhysicalAllocError::BudgetTooSmall {
            profile,
            required: metadata,
            maximum: limit.maximum,
        })
}

#[derive(Clone, Copy, Debug)]
struct UnitUpgrade {
    group: usize,
    target_planes: u8,
    distortion_reduction: f64,
}

#[derive(Clone, Copy, Debug)]
struct BundledUpgrade {
    group: usize,
    distortion_reduction: f64,
    first_plane_reduction: f64,
}

#[derive(Clone, Copy, Debug)]
struct HullChoice {
    distortion_reduction: f64,
    upgrades: usize,
    bundle_count: usize,
    unit_count: usize,
    replacement_after_exclusion: bool,
}

#[derive(Clone, Copy, Debug)]
struct RegularChoice {
    hull: HullChoice,
    exceptional_bundle_rank: Option<usize>,
}

const MAX_SCALABLE_FULL_VECTOR_TIES: usize = 8;

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
        return Ok(Some(floors.to_vec()));
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
        return Ok(Some(floors.to_vec()));
    }
    if common_cost == PhysicalBytes::ZERO {
        return Ok(Some(vec![SALT_V2_PLANES as u8; groups.len()]));
    }

    let mut unit_upgrades = Vec::with_capacity(available_upgrades);
    let mut bundled_upgrades = Vec::with_capacity(groups.len());
    for (group_index, group) in groups.iter().copied().enumerate() {
        match floors[group_index] {
            1 => {
                let first_reduction =
                    certified_reduction(distortion_at(group, 1), distortion_at(group, 2), profile)?;
                let second_reduction =
                    certified_reduction(distortion_at(group, 2), distortion_at(group, 3), profile)?;
                if second_reduction > first_reduction {
                    bundled_upgrades.push(BundledUpgrade {
                        group: group_index,
                        distortion_reduction: certified_reduction(
                            distortion_at(group, 1),
                            distortion_at(group, 3),
                            profile,
                        )?,
                        first_plane_reduction: first_reduction,
                    });
                } else {
                    if first_reduction > 0.0 {
                        unit_upgrades.push(UnitUpgrade {
                            group: group_index,
                            target_planes: 2,
                            distortion_reduction: first_reduction,
                        });
                    }
                    if second_reduction > 0.0 {
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
                    certified_reduction(distortion_at(group, 2), distortion_at(group, 3), profile)?;
                if reduction > 0.0 {
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

    unit_upgrades.sort_by(|left, right| {
        right
            .distortion_reduction
            .total_cmp(&left.distortion_reduction)
            .then_with(|| left.group.cmp(&right.group))
            .then_with(|| left.target_planes.cmp(&right.target_planes))
    });
    bundled_upgrades.sort_by(|left, right| {
        right
            .distortion_reduction
            .total_cmp(&left.distortion_reduction)
            .then_with(|| left.group.cmp(&right.group))
    });

    let unit_prefix = reduction_prefix(
        unit_upgrades
            .iter()
            .map(|upgrade| upgrade.distortion_reduction),
        profile,
    )?;
    let bundle_prefix = reduction_prefix(
        bundled_upgrades
            .iter()
            .map(|upgrade| upgrade.distortion_reduction),
        profile,
    )?;
    let choice = best_regular_choice(
        &unit_prefix,
        &bundle_prefix,
        &unit_upgrades,
        &bundled_upgrades,
        floors,
        capacity,
        profile,
    )?;
    let plane_counts =
        materialize_regular_choice(choice, &unit_upgrades, &bundled_upgrades, floors);
    debug_assert_eq!(
        plane_counts
            .iter()
            .zip(floors)
            .map(|(&planes, &floor)| usize::from(planes - floor))
            .sum::<usize>(),
        choice.hull.upgrades + usize::from(choice.exceptional_bundle_rank.is_some())
    );
    debug_assert!(
        choice.hull.upgrades + usize::from(choice.exceptional_bundle_rank.is_some()) <= capacity
    );
    Ok(Some(plane_counts))
}

fn reduction_prefix(
    reductions: impl IntoIterator<Item = f64>,
    profile: SaltV2Profile,
) -> Result<Vec<f64>, PhysicalAllocError> {
    let iterator = reductions.into_iter();
    let mut prefix = Vec::with_capacity(iterator.size_hint().0.saturating_add(1));
    prefix.push(0.0);
    for reduction in iterator {
        let sum = prefix.last().copied().unwrap_or(0.0) + reduction;
        if !sum.is_finite() {
            return Err(PhysicalAllocError::AccountingOverflow { profile });
        }
        if two_sum_error(prefix.last().copied().unwrap_or(0.0), reduction, sum) != 0.0 {
            return Err(PhysicalAllocError::NumericallyAmbiguousFastPath { profile });
        }
        prefix.push(sum);
    }
    Ok(prefix)
}

fn best_regular_choice(
    unit_prefix: &[f64],
    bundle_prefix: &[f64],
    units: &[UnitUpgrade],
    bundles: &[BundledUpgrade],
    floors: &[u8],
    capacity: usize,
    profile: SaltV2Profile,
) -> Result<RegularChoice, PhysicalAllocError> {
    let unit_count = unit_prefix.len() - 1;
    let bundle_count = bundle_prefix.len() - 1;
    let maximum_bundles = bundle_count.min(capacity / 2);
    let mut best = None;
    let mut full_vector_ties = 0usize;
    for (selected_bundles, &bundle_reduction) in
        bundle_prefix.iter().take(maximum_bundles + 1).enumerate()
    {
        let selected_units = unit_count.min(capacity - selected_bundles * 2);
        let distortion_reduction =
            checked_reduction_add(bundle_reduction, unit_prefix[selected_units], profile)?;
        retain_better_regular_choice(
            &mut best,
            RegularChoice {
                hull: HullChoice {
                    distortion_reduction,
                    upgrades: selected_bundles * 2 + selected_units,
                    bundle_count: selected_bundles,
                    unit_count: selected_units,
                    replacement_after_exclusion: false,
                },
                exceptional_bundle_rank: None,
            },
            units,
            bundles,
            floors,
            &mut full_vector_ties,
            profile,
        )?;
    }

    if capacity > 0 && !bundles.is_empty() {
        let residual_capacity = capacity - 1;
        let ordinary_maximum = bundle_count.min(residual_capacity / 2);
        let mut ordinary_prefix_best = Vec::with_capacity(ordinary_maximum + 1);
        let mut prefix_best = None;
        for (selected_bundles, &bundle_reduction) in
            bundle_prefix.iter().take(ordinary_maximum + 1).enumerate()
        {
            let selected_units = unit_count.min(residual_capacity - selected_bundles * 2);
            let choice = HullChoice {
                distortion_reduction: checked_reduction_add(
                    bundle_reduction,
                    unit_prefix[selected_units],
                    profile,
                )?,
                upgrades: selected_bundles * 2 + selected_units,
                bundle_count: selected_bundles,
                unit_count: selected_units,
                replacement_after_exclusion: false,
            };
            retain_better_hull_choice(
                &mut prefix_best,
                choice,
                units,
                bundles,
                floors,
                &mut full_vector_ties,
                profile,
            )?;
            ordinary_prefix_best.push(prefix_best.expect("current prefix choice"));
        }

        let replacement_maximum = bundle_count.saturating_sub(1).min(residual_capacity / 2);
        let mut replacement_suffix_best = vec![None; replacement_maximum + 2];
        let mut suffix_best = None;
        for selected_bundles in (0..=replacement_maximum).rev() {
            let selected_units = unit_count.min(residual_capacity - selected_bundles * 2);
            let choice = HullChoice {
                // This prefix contains one extra bundle. The excluded bundle's
                // reduction is subtracted per exception below.
                distortion_reduction: checked_reduction_add(
                    bundle_prefix[selected_bundles + 1],
                    unit_prefix[selected_units],
                    profile,
                )?,
                upgrades: selected_bundles * 2 + selected_units,
                bundle_count: selected_bundles,
                unit_count: selected_units,
                replacement_after_exclusion: true,
            };
            retain_better_hull_choice(
                &mut suffix_best,
                choice,
                units,
                bundles,
                floors,
                &mut full_vector_ties,
                profile,
            )?;
            replacement_suffix_best[selected_bundles] = suffix_best;
        }

        for (excluded_rank, bundle) in bundles.iter().enumerate() {
            if bundle.first_plane_reduction == 0.0 {
                continue;
            }
            let mut best_without_bundle =
                Some(ordinary_prefix_best[excluded_rank.min(ordinary_maximum)]);
            if excluded_rank < replacement_maximum {
                let replacement =
                    replacement_suffix_best[excluded_rank + 1].expect("valid replacement suffix");
                let adjusted_reduction =
                    replacement.distortion_reduction - bundle.distortion_reduction;
                if !adjusted_reduction.is_finite() {
                    return Err(PhysicalAllocError::AccountingOverflow { profile });
                }
                if two_diff_error(
                    replacement.distortion_reduction,
                    bundle.distortion_reduction,
                    adjusted_reduction,
                ) != 0.0
                {
                    return Err(PhysicalAllocError::NumericallyAmbiguousFastPath { profile });
                }
                retain_better_hull_choice(
                    &mut best_without_bundle,
                    HullChoice {
                        distortion_reduction: adjusted_reduction,
                        ..replacement
                    },
                    units,
                    bundles,
                    floors,
                    &mut full_vector_ties,
                    profile,
                )?;
            }
            let mut hull = best_without_bundle.expect("ordinary zero-bundle choice");
            hull.distortion_reduction = checked_reduction_add(
                hull.distortion_reduction,
                bundle.first_plane_reduction,
                profile,
            )?;
            retain_better_regular_choice(
                &mut best,
                RegularChoice {
                    hull,
                    exceptional_bundle_rank: Some(excluded_rank),
                },
                units,
                bundles,
                floors,
                &mut full_vector_ties,
                profile,
            )?;
        }
    }

    Ok(best.expect("zero-upgrade regular choice"))
}

fn checked_reduction_add(
    left: f64,
    right: f64,
    profile: SaltV2Profile,
) -> Result<f64, PhysicalAllocError> {
    let sum = left + right;
    if !sum.is_finite() {
        return Err(PhysicalAllocError::AccountingOverflow { profile });
    }
    if two_sum_error(left, right, sum) != 0.0 {
        return Err(PhysicalAllocError::NumericallyAmbiguousFastPath { profile });
    }
    Ok(sum)
}

fn certified_reduction(
    from: f64,
    to: f64,
    profile: SaltV2Profile,
) -> Result<f64, PhysicalAllocError> {
    let reduction = from - to;
    if two_diff_error(from, to, reduction) != 0.0 {
        return Err(PhysicalAllocError::NumericallyAmbiguousFastPath { profile });
    }
    Ok(reduction)
}

fn two_sum_error(left: f64, right: f64, sum: f64) -> f64 {
    let right_virtual = sum - left;
    let left_virtual = sum - right_virtual;
    let right_roundoff = right - right_virtual;
    let left_roundoff = left - left_virtual;
    left_roundoff + right_roundoff
}

fn two_diff_error(left: f64, right: f64, difference: f64) -> f64 {
    let right_virtual = left - difference;
    let left_virtual = difference + right_virtual;
    let right_roundoff = right_virtual - right;
    let left_roundoff = left - left_virtual;
    left_roundoff + right_roundoff
}

fn retain_better_hull_choice(
    best: &mut Option<HullChoice>,
    candidate: HullChoice,
    units: &[UnitUpgrade],
    bundles: &[BundledUpgrade],
    floors: &[u8],
    full_vector_ties: &mut usize,
    profile: SaltV2Profile,
) -> Result<(), PhysicalAllocError> {
    let replace = match *best {
        None => true,
        Some(current) => hull_choice_is_better(
            candidate,
            current,
            units,
            bundles,
            floors,
            full_vector_ties,
            profile,
        )?,
    };
    if replace {
        *best = Some(candidate);
    }
    Ok(())
}

fn hull_choice_is_better(
    candidate: HullChoice,
    current: HullChoice,
    units: &[UnitUpgrade],
    bundles: &[BundledUpgrade],
    floors: &[u8],
    full_vector_ties: &mut usize,
    profile: SaltV2Profile,
) -> Result<bool, PhysicalAllocError> {
    match candidate
        .distortion_reduction
        .total_cmp(&current.distortion_reduction)
    {
        core::cmp::Ordering::Greater => return Ok(true),
        core::cmp::Ordering::Less => return Ok(false),
        core::cmp::Ordering::Equal => {}
    }
    if candidate.upgrades != current.upgrades {
        return Ok(candidate.upgrades < current.upgrades);
    }
    charge_full_vector_tie(full_vector_ties, profile)?;
    Ok(materialize_hull_choice(candidate, units, bundles, floors)
        > materialize_hull_choice(current, units, bundles, floors))
}

fn retain_better_regular_choice(
    best: &mut Option<RegularChoice>,
    candidate: RegularChoice,
    units: &[UnitUpgrade],
    bundles: &[BundledUpgrade],
    floors: &[u8],
    full_vector_ties: &mut usize,
    profile: SaltV2Profile,
) -> Result<(), PhysicalAllocError> {
    let replace = match *best {
        None => true,
        Some(current) => regular_choice_is_better(
            candidate,
            current,
            units,
            bundles,
            floors,
            full_vector_ties,
            profile,
        )?,
    };
    if replace {
        *best = Some(candidate);
    }
    Ok(())
}

fn regular_choice_is_better(
    candidate: RegularChoice,
    current: RegularChoice,
    units: &[UnitUpgrade],
    bundles: &[BundledUpgrade],
    floors: &[u8],
    full_vector_ties: &mut usize,
    profile: SaltV2Profile,
) -> Result<bool, PhysicalAllocError> {
    let candidate_upgrades =
        candidate.hull.upgrades + usize::from(candidate.exceptional_bundle_rank.is_some());
    let current_upgrades =
        current.hull.upgrades + usize::from(current.exceptional_bundle_rank.is_some());
    match candidate
        .hull
        .distortion_reduction
        .total_cmp(&current.hull.distortion_reduction)
    {
        core::cmp::Ordering::Greater => return Ok(true),
        core::cmp::Ordering::Less => return Ok(false),
        core::cmp::Ordering::Equal => {}
    }
    if candidate_upgrades != current_upgrades {
        return Ok(candidate_upgrades < current_upgrades);
    }
    regular_choice_lex_is_greater(
        candidate,
        current,
        units,
        bundles,
        floors,
        full_vector_ties,
        profile,
    )
}

fn regular_choice_lex_is_greater(
    candidate: RegularChoice,
    current: RegularChoice,
    units: &[UnitUpgrade],
    bundles: &[BundledUpgrade],
    floors: &[u8],
    full_vector_ties: &mut usize,
    profile: SaltV2Profile,
) -> Result<bool, PhysicalAllocError> {
    let same_hull_shape = candidate.hull.bundle_count == current.hull.bundle_count
        && candidate.hull.unit_count == current.hull.unit_count
        && candidate.hull.replacement_after_exclusion == current.hull.replacement_after_exclusion;
    if same_hull_shape
        && let (Some(candidate_rank), Some(current_rank)) = (
            candidate.exceptional_bundle_rank,
            current.exceptional_bundle_rank,
        )
        && candidate_rank != current_rank
    {
        let candidate_group = bundles[candidate_rank].group;
        let current_group = bundles[current_rank].group;
        let first_rank = if candidate_group < current_group {
            candidate_rank
        } else {
            current_rank
        };
        return Ok(bundle_planes_for_choice(candidate, first_rank)
            > bundle_planes_for_choice(current, first_rank));
    }
    charge_full_vector_tie(full_vector_ties, profile)?;
    Ok(
        materialize_regular_choice(candidate, units, bundles, floors)
            > materialize_regular_choice(current, units, bundles, floors),
    )
}

fn charge_full_vector_tie(
    comparisons: &mut usize,
    profile: SaltV2Profile,
) -> Result<(), PhysicalAllocError> {
    *comparisons = comparisons.saturating_add(1);
    if *comparisons > MAX_SCALABLE_FULL_VECTOR_TIES {
        return Err(PhysicalAllocError::ScalableTieLimit {
            profile,
            comparisons: *comparisons,
        });
    }
    Ok(())
}

fn bundle_planes_for_choice(choice: RegularChoice, rank: usize) -> u8 {
    if choice.exceptional_bundle_rank == Some(rank) {
        return 2;
    }
    let take = choice.hull.bundle_count + usize::from(choice.hull.replacement_after_exclusion);
    if rank < take { 3 } else { 1 }
}

fn materialize_hull_choice(
    choice: HullChoice,
    units: &[UnitUpgrade],
    bundles: &[BundledUpgrade],
    floors: &[u8],
) -> Vec<u8> {
    let mut plane_counts = floors.to_vec();
    apply_unit_prefix(&mut plane_counts, units, choice.unit_count);
    let take = choice.bundle_count + usize::from(choice.replacement_after_exclusion);
    for upgrade in bundles.iter().take(take) {
        plane_counts[upgrade.group] = 3;
    }
    plane_counts
}

fn materialize_regular_choice(
    choice: RegularChoice,
    units: &[UnitUpgrade],
    bundles: &[BundledUpgrade],
    floors: &[u8],
) -> Vec<u8> {
    let mut plane_counts = floors.to_vec();
    apply_unit_prefix(&mut plane_counts, units, choice.hull.unit_count);
    let take = choice.hull.bundle_count + usize::from(choice.hull.replacement_after_exclusion);
    for (rank, upgrade) in bundles.iter().take(take).enumerate() {
        if choice.exceptional_bundle_rank != Some(rank) {
            plane_counts[upgrade.group] = 3;
        }
    }
    if let Some(excluded_rank) = choice.exceptional_bundle_rank {
        plane_counts[bundles[excluded_rank].group] = 2;
    }
    plane_counts
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

fn pareto_prune(mut states: Vec<ExactState>) -> Vec<ExactState> {
    states.sort_by(|left, right| {
        left.bytes
            .serialized
            .cmp(&right.bytes.serialized)
            .then_with(|| left.bytes.resident.cmp(&right.bytes.resident))
            .then_with(|| left.distortion.total_cmp(&right.distortion))
            .then_with(|| right.plane_counts.cmp(&left.plane_counts))
    });
    let mut deduplicated: Vec<ExactState> = Vec::with_capacity(states.len());
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

    let mut kept = Vec::with_capacity(deduplicated.len());
    'candidate: for (index, candidate) in deduplicated.iter().enumerate() {
        for (other_index, other) in deduplicated.iter().enumerate() {
            if index == other_index {
                continue;
            }
            let no_more_bytes = other.bytes.serialized <= candidate.bytes.serialized
                && other.bytes.resident <= candidate.bytes.resident;
            let no_worse_objective = other.distortion <= candidate.distortion;
            if no_more_bytes && no_worse_objective {
                continue 'candidate;
            }
        }
        kept.push(candidate.clone());
    }
    kept
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
    fn scalable_regular_solver_fails_closed_when_gain_addition_loses_bits() {
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

        let error = allocate_nested_profiles(
            &groups,
            &NestedProfileBudgets {
                compact: limit,
                near_lossless: limit,
            },
        )
        .expect_err("rounded model-scale gain must not be ranked approximately");

        assert_eq!(
            error,
            PhysicalAllocError::NumericallyAmbiguousFastPath {
                profile: SaltV2Profile::CompactV1,
            }
        );
    }

    #[test]
    fn scalable_regular_solver_bounds_full_vector_tie_work() {
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

        let error = allocate_nested_profiles(
            &groups,
            &NestedProfileBudgets {
                compact: limit,
                near_lossless: limit,
            },
        )
        .expect_err("pathological aggregate ties must not become quadratic");

        assert!(matches!(
            error,
            PhysicalAllocError::ScalableTieLimit {
                profile: SaltV2Profile::CompactV1,
                comparisons,
            } if comparisons == MAX_SCALABLE_FULL_VECTOR_TIES + 1
        ));
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
                    brute_force_best(&groups, limit),
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
        let mut best: Option<(f64, PhysicalBytes, Vec<u8>)> = None;
        let combinations = 3usize.pow(groups.len() as u32);
        for encoded in 0..combinations {
            let mut cursor = encoded;
            let mut counts = Vec::with_capacity(groups.len());
            let mut physical = limit.metadata.effective();
            let mut distortion = 0.0;
            for group in groups {
                let planes = (cursor % 3 + 1) as u8;
                cursor /= 3;
                counts.push(planes);
                for candidate in &group.candidates[..usize::from(planes)] {
                    physical = physical
                        .checked_add(candidate.byte_delta.effective())
                        .unwrap();
                }
                distortion += group.candidates[usize::from(planes - 1)].distortion;
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
