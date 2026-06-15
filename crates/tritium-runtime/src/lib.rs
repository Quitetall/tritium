//! # tritium-runtime
//!
//! Backend registry and dispatch. Execution backends (`tritium-cpu`,
//! `tritium-cuda`, …) **self-register** through the [`BACKENDS`] `linkme`
//! distributed slice, so wiring a new backend into the runtime needs no central
//! edit here — the linker collects every entry at link time and [`Registry::init`]
//! discovers them at startup.
//!
//! This crate depends only on [`tritium_spec`] and `linkme`; it never references a
//! concrete backend, which keeps the dependency graph acyclic and lets backends be
//! built (and the runtime linked) in any order.
//!
//! ## Registering a backend (downstream pattern)
//!
//! A backend crate pushes one [`BackendEntry`] into the slice. The `#[used]`
//! semantics of [`linkme::distributed_slice`] keep the entry alive even though
//! nothing names it directly — but the backend crate must actually be linked into
//! the final binary (e.g. a `use tritium_cpu as _;` in the binary, or a normal
//! dependency that pulls in a referenced symbol) for its entry to appear.
//!
//! ```
//! use tritium_runtime::{BackendEntry, BACKENDS};
//! use tritium_spec::{BackendError, TernaryBackend};
//!
//! # struct CpuBackend;
//! # impl CpuBackend { fn new() -> Result<Self, BackendError> { Ok(Self) } }
//! # impl TernaryBackend for CpuBackend {
//! #     fn device_id(&self) -> &str { "cpu" }
//! #     fn capabilities(&self) -> tritium_spec::DeviceCaps {
//! #         tritium_spec::DeviceCaps::new("cpu", "host")
//! #     }
//! #     fn upload_weights(&self, _: &[u8], _: tritium_core::GemmShape, _: tritium_core::TernaryFormat)
//! #         -> Result<Box<dyn tritium_spec::DeviceBuffer>, BackendError> { unimplemented!() }
//! #     fn mpgemm(&self, _: &[f32], _: &dyn tritium_spec::DeviceBuffer, _: &[f32],
//! #         _: tritium_core::GemmShape, _: tritium_core::TernaryFormat, _: &mut [f32])
//! #         -> Result<(), BackendError> { unimplemented!() }
//! # }
//! /// One `init` constructor per backend. Failure is reported as an `Err`, which
//! /// the registry logs and skips — it never aborts discovery of other backends.
//! fn init_cpu() -> Result<Box<dyn TernaryBackend>, BackendError> {
//!     Ok(Box::new(CpuBackend::new()?))
//! }
//!
//! #[allow(unsafe_code)]
//! #[linkme::distributed_slice(BACKENDS)]
//! static CPU: BackendEntry = BackendEntry {
//!     name: "cpu",
//!     init: init_cpu,
//! };
//! ```
// `linkme`'s `distributed_slice` expands to a static with a custom
// `#[link_section]`, which the compiler's `unsafe_code` lint flags. We therefore
// `deny` (not `forbid`) unsafe code at the crate level and grant a narrowly-scoped
// `#[allow(unsafe_code)]` on exactly the `distributed_slice` declarations below —
// no hand-written `unsafe` exists in this crate.
#![deny(unsafe_code)]

use std::sync::OnceLock;

use tritium_spec::{BackendError, DeviceCaps, TernaryBackend};

/// A single registration record contributed by a backend crate.
///
/// Backends place one of these into the [`BACKENDS`] distributed slice; the
/// runtime reads the slice at [`Registry::init`] time and calls each
/// [`init`](BackendEntry::init) to instantiate the backend.
pub struct BackendEntry {
    /// Stable, human-facing name used to look a backend up with
    /// [`Registry::get`], e.g. `"cpu"` or `"cuda"`. This is the *family* name; a
    /// constructed backend may report a more specific [`TernaryBackend::device_id`]
    /// such as `"cuda:0"`.
    pub name: &'static str,

    /// Constructor for the backend. Returns the backend as a trait object, or a
    /// [`BackendError`] if the device is unavailable (no GPU, missing ISA, OOM…).
    /// A returned `Err` is logged and skipped by the registry — it does **not**
    /// prevent other backends from registering.
    pub init: fn() -> Result<Box<dyn TernaryBackend>, BackendError>,
}

// `BackendEntry` holds a `fn` pointer (Debug-able) and a `&'static str`, but the
// derive would require nothing special; keep an explicit impl so the function
// pointer is rendered as an opaque address rather than failing the
// `missing_debug_implementations` lint.
impl core::fmt::Debug for BackendEntry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BackendEntry")
            .field("name", &self.name)
            .field("init", &(self.init as *const ()))
            .finish()
    }
}

/// The distributed slice every backend crate registers into.
///
/// Each backend contributes exactly one [`BackendEntry`] with
/// `#[linkme::distributed_slice(BACKENDS)]`. The linker gathers all entries
/// across all linked crates into this single slice with no runtime cost and no
/// central registration list. See the [crate docs](crate) for the registration
/// example.
#[allow(unsafe_code)]
#[linkme::distributed_slice]
pub static BACKENDS: [BackendEntry] = [..];

/// A live set of successfully-initialised backends.
///
/// Built by [`Registry::init`], which walks [`BACKENDS`] and calls each entry's
/// constructor. Backends whose constructor returns `Err` (or panics — caught and
/// turned into a skip) are omitted; every other backend is still registered, so
/// one unavailable device never takes the whole runtime down.
pub struct Registry {
    backends: Vec<(String, Box<dyn TernaryBackend>)>,
}

// `dyn TernaryBackend` is not `Debug`, so derive cannot apply. Render the
// registry by the names of its backends — enough to satisfy
// `missing_debug_implementations` and to be useful in test output.
impl core::fmt::Debug for Registry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Registry")
            .field(
                "backends",
                &self.backends.iter().map(|(n, _)| n).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Registry {
    /// Discover and initialise every backend in the [`BACKENDS`] slice.
    ///
    /// Each entry's [`init`](BackendEntry::init) is called once. On success the
    /// backend is stored under its [`name`](BackendEntry::name). On `Err`, the
    /// failure is written to stderr and the entry is skipped — discovery of the
    /// remaining backends continues regardless. The returned `Registry` therefore
    /// contains exactly the backends that initialised cleanly, in slice order.
    ///
    /// This never fails: an empty `Registry` (no backends available) is a valid
    /// result, not an error.
    #[must_use]
    pub fn init() -> Self {
        let mut backends: Vec<(String, Box<dyn TernaryBackend>)> = Vec::new();
        for entry in BACKENDS {
            match (entry.init)() {
                Ok(backend) => backends.push((entry.name.to_owned(), backend)),
                Err(err) => {
                    // A failing backend (no GPU, missing ISA, OOM, …) is expected
                    // and must not abort the others. Log and move on.
                    eprintln!(
                        "tritium-runtime: backend `{}` failed to initialise, skipping: {err}",
                        entry.name
                    );
                }
            }
        }
        Self { backends }
    }

    /// Look up an initialised backend by its registration name.
    ///
    /// Returns the first backend registered under `name`, or `None` if no such
    /// backend initialised successfully.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn TernaryBackend> {
        self.backends
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, b)| b.as_ref())
    }

    /// The [`DeviceCaps`] of every registered backend, in registration order.
    ///
    /// Used by callers to choose a backend for a given problem without having to
    /// construct or name each one.
    #[must_use]
    pub fn enumerate(&self) -> Vec<DeviceCaps> {
        self.backends
            .iter()
            .map(|(_, b)| b.capabilities())
            .collect()
    }

    /// Number of successfully-registered backends.
    #[must_use]
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// True if no backend registered successfully.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }
}

/// Process-global registry, initialised once on first access.
static GLOBAL: OnceLock<Registry> = OnceLock::new();

/// Borrow the process-global [`Registry`], initialising it on first call.
///
/// The first caller triggers [`Registry::init`]; every later caller gets the same
/// instance. Cheap to call repeatedly. Use this for the common case of a single
/// shared backend set; construct a standalone [`Registry::init`] only if you need
/// an isolated one.
#[must_use]
pub fn global() -> &'static Registry {
    GLOBAL.get_or_init(Registry::init)
}

/// Eagerly initialise the process-global [`Registry`] and borrow it.
///
/// Equivalent to [`global`], but the name documents intent at the call site where
/// you want the one-time discovery (and its stderr diagnostics) to happen now —
/// e.g. at program startup — rather than lazily on first use.
#[must_use]
pub fn init_global() -> &'static Registry {
    global()
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::any::Any;
    use tritium_core::{GemmShape, TernaryFormat};
    use tritium_spec::DeviceBuffer;

    /// Minimal owned-bytes device buffer for the dummy backend.
    #[derive(Debug)]
    struct VecBuffer(Vec<u8>);

    impl DeviceBuffer for VecBuffer {
        fn len_bytes(&self) -> usize {
            self.0.len()
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// A trivial backend: uploads a copy of the bytes, fills `out` with zeros.
    #[derive(Debug)]
    struct DummyBackend;

    impl TernaryBackend for DummyBackend {
        fn device_id(&self) -> &str {
            "dummy"
        }

        fn capabilities(&self) -> DeviceCaps {
            DeviceCaps::new("dummy", "in-crate test backend").with_features(vec!["test".to_owned()])
        }

        fn upload_weights(
            &self,
            packed: &[u8],
            _shape: GemmShape,
            _format: TernaryFormat,
        ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
            Ok(Box::new(VecBuffer(packed.to_vec())))
        }

        fn mpgemm(
            &self,
            _act: &[f32],
            _weights: &dyn DeviceBuffer,
            _scales: &[f32],
            _shape: GemmShape,
            _format: TernaryFormat,
            out: &mut [f32],
        ) -> Result<(), BackendError> {
            out.fill(0.0);
            Ok(())
        }
    }

    fn init_dummy() -> Result<Box<dyn TernaryBackend>, BackendError> {
        Ok(Box::new(DummyBackend))
    }

    /// A backend whose init always fails — proves the registry tolerates it.
    fn init_always_fails() -> Result<Box<dyn TernaryBackend>, BackendError> {
        Err(BackendError::Backend("intentional test failure".to_owned()))
    }

    #[allow(unsafe_code)]
    #[linkme::distributed_slice(BACKENDS)]
    static DUMMY: BackendEntry = BackendEntry {
        name: "dummy",
        init: init_dummy,
    };

    #[allow(unsafe_code)]
    #[linkme::distributed_slice(BACKENDS)]
    static FAILING: BackendEntry = BackendEntry {
        name: "failing",
        init: init_always_fails,
    };

    #[test]
    fn registry_finds_dummy_and_skips_failing() {
        let reg = Registry::init();

        // The failing backend was skipped, the dummy registered.
        assert!(reg.get("dummy").is_some(), "dummy backend should register");
        assert!(
            reg.get("failing").is_none(),
            "failing backend must be skipped, not registered"
        );
        assert!(!reg.is_empty(), "at least the dummy must be present");
        assert_eq!(
            reg.len(),
            reg.enumerate().len(),
            "len agrees with enumerate"
        );
    }

    #[test]
    fn get_returns_correct_backend() {
        let reg = Registry::init();
        let backend = reg.get("dummy").expect("dummy must be present");
        assert_eq!(backend.device_id(), "dummy");
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn enumerate_includes_dummy_caps() {
        let reg = Registry::init();
        let caps = reg.enumerate();
        assert_eq!(
            caps.len(),
            reg.len(),
            "one caps entry per registered backend"
        );
        let dummy = caps
            .iter()
            .find(|c| c.backend == "dummy")
            .expect("dummy caps must be enumerated");
        assert_eq!(dummy.device_name, "in-crate test backend");
        assert!(dummy.has_feature("test"));
    }

    #[test]
    fn init_does_not_panic_with_failing_backend() {
        // Sole purpose: a failing init must not unwind. Reaching here is the pass.
        let _reg = Registry::init();
    }

    #[test]
    fn dummy_backend_roundtrips_through_registry() {
        let reg = Registry::init();
        let backend = reg.get("dummy").expect("dummy must be present");
        let shape = GemmShape { m: 1, n: 2, k: 4 };
        let buf = backend
            .upload_weights(&[1, 2, 3, 4], shape, TernaryFormat::Tq2_0)
            .expect("upload should succeed");
        assert_eq!(buf.len_bytes(), 4);

        let act = [1.0_f32; 4];
        let scales = [1.0_f32; 2];
        let mut out = [9.0_f32; 2];
        backend
            .mpgemm(
                &act,
                buf.as_ref(),
                &scales,
                shape,
                TernaryFormat::Tq2_0,
                &mut out,
            )
            .expect("mpgemm should succeed");
        assert_eq!(out, [0.0, 0.0], "dummy mpgemm fills zeros");
    }

    #[test]
    fn global_is_stable_across_calls() {
        let a = global();
        let b = init_global();
        assert!(std::ptr::eq(a, b), "global registry must be a singleton");
        assert!(a.get("dummy").is_some());
    }
}
