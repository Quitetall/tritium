//! Scoped original-package loading; no retained decoded owner cache.
use super::*;

/// One decoded owner plus bounded source/codec windows and caller overlap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeRefinementOwnerReadLimits {
    /// Largest owner's decode working bytes plus extra payload ceiling.
    /// Excludes metadata, allocator slack, caller stream buffers and OS caches.
    pub max_working_bytes: u64,
    /// Explicit overlapping caller payload; not inferred from external graphs.
    pub extra_payload_bytes: u64,
}
/// Original parent/source refusal versus the unchanged consumer error.
#[derive(Debug)]
#[non_exhaustive]
pub enum NativeRefinementOwnerSourceError<E> {
    /// Actual typed source/evidence/cap/traversal failure.
    Source(Error),
    /// Original scoped consumer error.
    Consumer(E),
}
impl<E: std::fmt::Display> std::fmt::Display for NativeRefinementOwnerSourceError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(e) => e.fmt(f),
            Self::Consumer(e) => write!(f, "native owner consumer: {e}"),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for NativeRefinementOwnerSourceError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Source(e) => Some(e),
            Self::Consumer(e) => Some(e),
        }
    }
}

/// Scoped owner provider bound to an actual retained immediate parent package.
///
/// Available only inside `with`: every selected owner is decoded individually
/// in canonical order. Complete traversal and fresh terminal parent verification
/// precede returning an owned consumer result. Never retains decoded owners or
/// clones the manifest/count catalog. Not official-source authentication,
/// complete lineage, numerical replay, model quality or publication authority.
pub struct NativeRefinementOwnerSource<'a, R> {
    reader: &'a mut TritModelReader<R>,
    parent: &'a NativeRefinementParentView,
    plan: &'a NativeRefinementPlan,
    limits: NativeRefinementOwnerReadLimits,
    working: u64,
    next: usize,
    traversals: u64,
    poisoned: bool,
}
impl<R> std::fmt::Debug for NativeRefinementOwnerSource<'_, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeRefinementOwnerSource")
            .field("owners", &self.plan.owners.len())
            .field("next", &self.next)
            .field("traversals", &self.traversals)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}
impl<'a, R: Read + Seek> NativeRefinementOwnerSource<'a, R> {
    /// Forecast the largest owner's decode windows/payload plus caller overlap
    /// from admitted metadata only. Performs no source reads or owner payload
    /// allocation; this does not revalidate current bytes or grant provenance.
    pub fn required_working_bytes(
        reader: &mut TritModelReader<R>,
        parent: &NativeRefinementParentView,
        plan: &NativeRefinementPlan,
        extra_payload_bytes: u64,
    ) -> Result<u64, Error> {
        plan.validate_parent(reader.manifest(), parent.binding())
            .map_err(|_| Error::Binding("native parent/request mismatch"))?;
        let profile = match plan.recipe {
            NativeRecoveryRecipe::ScaleOnly { .. } => super::super::publication::Profile::Scale,
            NativeRecoveryRecipe::AlternatingPv { .. } => super::super::publication::Profile::Pv,
            _ => {
                return Err(Error::Binding(
                    "native owner refinement profile unsupported",
                ));
            }
        };
        let (state, _) = parent_state_sizes(reader, plan, profile)?;
        if state > plan.limits.max_state_bytes {
            return Err(Error::Limit("native owner aggregate state cap exceeded"));
        }
        let mut largest = 0;
        for name in &plan.owners {
            let stream = reader.tensor_stream(&plan.parent.level, name)?;
            largest = largest.max(stream.additive_decode_working_bytes()?);
        }
        add(largest, extra_payload_bytes)
    }
    /// Preflight all selected metadata and the largest-owner cap, verify actual
    /// parent evidence, run a scoped consumer, then verify the complete parent
    /// again before returning its owned result. Consumer effects/outputs must
    /// remain private until success; errors do not roll back arbitrary effects.
    pub fn with<T, E>(
        reader: &'a mut TritModelReader<R>,
        parent: &'a NativeRefinementParentView,
        plan: &'a NativeRefinementPlan,
        limits: NativeRefinementOwnerReadLimits,
        run: impl FnOnce(&mut Self) -> Result<T, E>,
    ) -> Result<T, NativeRefinementOwnerSourceError<E>> {
        let working =
            Self::required_working_bytes(reader, parent, plan, limits.extra_payload_bytes)
                .map_err(NativeRefinementOwnerSourceError::Source)?;
        if limits.max_working_bytes == 0 || working > limits.max_working_bytes {
            return Err(NativeRefinementOwnerSourceError::Source(Error::Limit(
                "native owner decode working cap exceeded",
            )));
        }
        parent
            .verify(reader)
            .map_err(Error::from)
            .map_err(NativeRefinementOwnerSourceError::Source)?;
        // Keep the original identity outside the mutable callback loan so a
        // swap of two valid scopes cannot redirect terminal verification.
        let original_reader = reader as *const TritModelReader<R>;
        let original_parent = parent as *const NativeRefinementParentView;
        let original_plan = plan as *const NativeRefinementPlan;
        let mut source = Self {
            reader,
            parent,
            plan,
            limits,
            working,
            next: 0,
            traversals: 0,
            poisoned: false,
        };
        let result = run(&mut source).map_err(NativeRefinementOwnerSourceError::Consumer)?;
        if !std::ptr::eq(source.reader as *const _, original_reader)
            || !std::ptr::eq(source.parent as *const _, original_parent)
            || !std::ptr::eq(source.plan as *const _, original_plan)
        {
            return Err(NativeRefinementOwnerSourceError::Source(Error::Binding(
                "native owner source context changed",
            )));
        }
        if source.poisoned || source.next != 0 || source.traversals == 0 {
            return Err(NativeRefinementOwnerSourceError::Source(Error::Binding(
                "native owner traversal incomplete or poisoned",
            )));
        }
        source
            .parent
            .verify(source.reader)
            .map_err(Error::from)
            .map_err(NativeRefinementOwnerSourceError::Source)?;
        Ok(result)
    }
    /// Largest-owner logical decode requirement plus declared caller overlap.
    /// Not a process-memory measurement or a numerical optimizer-state ledger.
    #[must_use]
    pub const fn working_bytes(&self) -> u64 {
        self.working
    }
    /// Load the exact next frozen owner from actual blob bytes. Each complete
    /// inventory traversal may restart at zero. Any ignored order/source/codec
    /// failure poisons the scope; it cannot later return a successful result.
    pub fn load(&mut self, index: usize, name: &str) -> Result<AdditiveTensor, Error> {
        if self.poisoned {
            return Err(Error::Binding("native owner source is poisoned"));
        }
        self.poisoned = true;
        if index != self.next || self.plan.owners.get(index).map(String::as_str) != Some(name) {
            return Err(Error::Binding("native owner name/index/order mismatch"));
        }
        let mut stream = self.reader.tensor_stream(&self.plan.parent.level, name)?;
        let weight = stream
            .decode_additive(self.limits.max_working_bytes - self.limits.extra_payload_bytes)?;
        self.next += 1;
        if self.next == self.plan.owners.len() {
            self.next = 0;
            self.traversals = self
                .traversals
                .checked_add(1)
                .ok_or(Error::Limit("native owner traversal overflow"))?;
        }
        self.poisoned = false;
        Ok(weight)
    }
}
