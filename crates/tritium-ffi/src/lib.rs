//! # tritium-ffi — C ABI for Tritium inference.
//!
//! A `cdylib` + `staticlib` exposing a small, stable C API: load a GGUF model on
//! the CPU backend and greedily generate token IDs. The committed header lives at
//! `include/tritium.h`; a drift test (`tests/header.rs`) regenerates it with
//! `cbindgen` and fails if it falls out of sync with these declarations.
//!
//! ## Safety contract (the FFI boundary)
//!
//! - **Panics never unwind across the boundary** (unwinding out of `extern "C"`
//!   is undefined behavior). Under the workspace's default `release`/`dist`
//!   profile (`panic = "abort"`) an internal panic aborts the process — the safe,
//!   defined outcome. The `catch_unwind` guards additionally turn a panic into
//!   [`TritiumStatus::Panic`] *when the crate is built with* `panic = "unwind"`
//!   (the default for `dev`/`test`); build that way if you need the host to
//!   survive an internal panic with an error code instead of an abort. (`panic`
//!   is a whole-artifact profile setting — Cargo forbids overriding it per crate.)
//! - **Null-checked.** Every pointer argument is checked; a null where one is
//!   required returns [`TritiumStatus::NullArg`] instead of dereferencing.
//! - **Ownership.** [`tritium_model_load_file`] returns an owned handle the caller
//!   must release with [`tritium_model_free`] (null-safe). Token IDs are copied
//!   into a caller-owned buffer; Tritium frees nothing the caller passes in.
//! - **Threading.** A `TritiumModel*` is `Send` (move it between threads) but **not
//!   `Sync`**: do not call [`tritium_generate`] on the *same* handle from two
//!   threads at once (it mutates the model's KV cache). Distinct handles are fully
//!   independent and may run concurrently.
#![deny(missing_docs)]

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

// Force-link the CPU backend so its `linkme` registration is present for
// `ModelRunner::load_cpu`.
use tritium_cpu as _;
use tritium_nn::ModelRunner;

/// C ABI version. Bump on any breaking change to the exported symbols/layout.
pub const TRITIUM_ABI_VERSION: u32 = 1;

thread_local! {
    /// Last error message for the current thread, returned by
    /// [`tritium_last_error`]. Set on error paths, cleared on entry to each
    /// fallible call, so it is valid until the next `tritium_*` call on this thread.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Record a thread-local error message (lossily NUL-sanitized).
fn set_last_error(msg: &str) {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_else(|_| {
        // Unreachable after the NUL replacement, but stay infallible.
        CString::new("error").expect("static literal has no interior NUL")
    });
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

/// Clear the thread-local error message (called on entry to a fallible call).
fn clear_last_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// Return a pointer to the calling thread's last error message as a
/// NUL-terminated C string, or null if there is none.
///
/// The pointer is valid until the next `tritium_*` call **on the same thread**
/// (which may overwrite or clear it); copy the string if you need it longer. The
/// storage is thread-local, so each thread has its own last error.
#[unsafe(no_mangle)]
pub extern "C" fn tritium_last_error() -> *const c_char {
    LAST_ERROR.with(|e| match &*e.borrow() {
        Some(c) => c.as_ptr(),
        None => ptr::null(),
    })
}

/// Result code returned by the C API. `Ok` is 0; all errors are non-zero.
///
/// `#[non_exhaustive]` (Rust side): future ABI revisions may add status codes, so
/// Rust callers must include a wildcard arm. C callers should treat any unknown
/// non-zero code as a generic error.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TritiumStatus {
    /// Success.
    Ok = 0,
    /// A required pointer argument was null.
    NullArg = 1,
    /// Loading/parsing the model failed.
    Load = 2,
    /// Generation failed in the backend.
    Generate = 3,
    /// The output buffer was too small; `*out_len` holds the required length.
    BufferTooSmall = 4,
    /// An internal panic was caught at the boundary. Only reachable when the
    /// crate is built with `panic = "unwind"`; under `panic = "abort"` (the
    /// default release/dist profile) a panic aborts the process instead.
    Panic = 5,
}

/// Opaque handle to a loaded model (a `ModelRunner`). Created by
/// [`tritium_model_load_file`], released by [`tritium_model_free`].
pub struct TritiumModel {
    runner: ModelRunner,
}

impl std::fmt::Debug for TritiumModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Opaque to C; don't leak model internals.
        f.debug_struct("TritiumModel").finish_non_exhaustive()
    }
}

/// Returns the C ABI version ([`TRITIUM_ABI_VERSION`]).
#[unsafe(no_mangle)]
pub extern "C" fn tritium_abi_version() -> u32 {
    TRITIUM_ABI_VERSION
}

/// Returns the crate version as a static, NUL-terminated C string (never null).
#[unsafe(no_mangle)]
pub extern "C" fn tritium_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Load a GGUF model from `path` (a NUL-terminated UTF-8 string) on the CPU
/// backend. Returns an owned handle, or null on failure; if `out_status` is
/// non-null it receives the [`TritiumStatus`]. Free the handle with
/// [`tritium_model_free`].
///
/// # Safety
/// `path` must be a valid NUL-terminated C string (or null); `out_status` must be
/// null or a valid writable `TritiumStatus*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tritium_model_load_file(
    path: *const c_char,
    out_status: *mut TritiumStatus,
) -> *mut TritiumModel {
    clear_last_error();
    let set = |s: TritiumStatus| {
        if !out_status.is_null() {
            // SAFETY: checked non-null; caller guarantees a valid writable pointer.
            unsafe { *out_status = s };
        }
    };
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if path.is_null() {
            set_last_error("path argument was null");
            return Err(TritiumStatus::NullArg);
        }
        // SAFETY: non-null per the check; caller guarantees a NUL-terminated string.
        let path_str = unsafe { CStr::from_ptr(path) }.to_str().map_err(|_| {
            set_last_error("path was not valid UTF-8");
            TritiumStatus::Load
        })?;
        let bytes = std::fs::read(path_str).map_err(|e| {
            set_last_error(&format!("read {path_str}: {e}"));
            TritiumStatus::Load
        })?;
        let runner = ModelRunner::load_cpu(&bytes).map_err(|e| {
            set_last_error(&format!("load model: {e}"));
            TritiumStatus::Load
        })?;
        Ok(Box::into_raw(Box::new(TritiumModel { runner })))
    }));
    match outcome {
        Ok(Ok(handle)) => {
            set(TritiumStatus::Ok);
            handle
        }
        Ok(Err(status)) => {
            set(status);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("internal panic while loading model");
            set(TritiumStatus::Panic);
            ptr::null_mut()
        }
    }
}

/// Load a GGUF model from an in-memory buffer (`data`, `len` bytes) on the CPU
/// backend. Mirrors [`tritium_model_load_file`] but reads from memory rather than
/// a path — useful when the model is embedded or fetched by the host. Returns an
/// owned handle, or null on failure; if `out_status` is non-null it receives the
/// [`TritiumStatus`]. Free the handle with [`tritium_model_free`]. On error, see
/// [`tritium_last_error`].
///
/// # Safety
/// `data` must point to `len` readable bytes (or be null iff `len == 0`);
/// `out_status` must be null or a valid writable `TritiumStatus*`. Tritium copies
/// what it needs and does not retain `data`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tritium_model_load_bytes(
    data: *const u8,
    len: usize,
    out_status: *mut TritiumStatus,
) -> *mut TritiumModel {
    clear_last_error();
    let set = |s: TritiumStatus| {
        if !out_status.is_null() {
            // SAFETY: checked non-null; caller guarantees a valid writable pointer.
            unsafe { *out_status = s };
        }
    };
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() && len != 0 {
            set_last_error("data argument was null with len != 0");
            return Err(TritiumStatus::NullArg);
        }
        let bytes: &[u8] = if len == 0 {
            &[]
        } else {
            // SAFETY: non-null + len checked; caller guarantees `len` readable bytes.
            unsafe { std::slice::from_raw_parts(data, len) }
        };
        let runner = ModelRunner::load_cpu(bytes).map_err(|e| {
            set_last_error(&format!("load model: {e}"));
            TritiumStatus::Load
        })?;
        Ok(Box::into_raw(Box::new(TritiumModel { runner })))
    }));
    match outcome {
        Ok(Ok(handle)) => {
            set(TritiumStatus::Ok);
            handle
        }
        Ok(Err(status)) => {
            set(status);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("internal panic while loading model");
            set(TritiumStatus::Panic);
            ptr::null_mut()
        }
    }
}

/// Versioned options for [`tritium_generate`]. Pass `NULL` for the default
/// (greedy) behavior — all existing behavior is preserved when no options are
/// given.
///
/// To use it: zero the whole struct, set `struct_size = sizeof(TritiumGenerateOptions)`,
/// then set the fields your build knows about. Tritium reads only the fields that
/// fit within the caller-provided `struct_size`, so growing this struct in a later
/// ABI revision stays forward- and backward-compatible. All currently-defined
/// fields beyond `struct_size` are reserved and must be zeroed.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TritiumGenerateOptions {
    /// `sizeof(TritiumGenerateOptions)` as seen by the caller. Tritium reads
    /// option fields only up to this many bytes. Must be `>= sizeof(usize)`.
    pub struct_size: usize,
    /// Reserved for future options (e.g. sampling temperature / top-k). Must be
    /// zeroed; ignored by this ABI revision.
    pub reserved: [u8; 32],
}

/// Greedily generate up to `max_new` tokens continuing `prompt` (`prompt_len`
/// token IDs), stopping early at `eos`. The generated count never exceeds
/// `max_new`, so the simplest correct use is to size `out` to `max_new` and call
/// once, reading the count from `*out_len`.
///
/// `opts` is an optional [`TritiumGenerateOptions`] pointer for forward-compatible
/// tuning: pass `NULL` for the current greedy default (unchanged behavior). When
/// non-null, Tritium reads its fields defensively up to the caller's `struct_size`;
/// this ABI revision defines only reserved fields, so a non-null zeroed `opts`
/// behaves identically to `NULL`.
///
/// Writes up to `out_cap` token IDs into `out` and sets `*out_len` to the number
/// generated. If the count exceeds `out_cap`, returns
/// [`TritiumStatus::BufferTooSmall`] with `*out_len` set to the required length
/// (nothing is written). When `out_len` is non-null it is *always* written: `0`
/// on the `NullArg`/`Generate` paths, the count on `Ok`/`BufferTooSmall`.
///
/// Each call re-runs generation from scratch (the KV cache is reset), so a "size
/// with `out_cap = 0`, then fill" pattern costs two full generations and is only
/// length-stable because greedy decoding is deterministic. Prefer the single
/// `max_new`-sized pass above.
///
/// # Safety
/// `model` must be a live handle from [`tritium_model_load_file`]; `prompt` must
/// point to `prompt_len` `u32`s (or be null iff `prompt_len == 0`); `out` must
/// point to `out_cap` writable `u32`s (or be null iff `out_cap == 0`); `out_len`
/// must be a valid writable `usize*`; `opts` must be null or point to a valid
/// [`TritiumGenerateOptions`] of at least `opts->struct_size` bytes. Do not call
/// concurrently on one `model`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tritium_generate(
    model: *mut TritiumModel,
    prompt: *const u32,
    prompt_len: usize,
    max_new: u32,
    eos: u32,
    out: *mut u32,
    out_cap: usize,
    out_len: *mut usize,
    opts: *const TritiumGenerateOptions,
) -> TritiumStatus {
    clear_last_error();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if out_len.is_null() {
            set_last_error("out_len argument was null");
            return TritiumStatus::NullArg;
        }
        // Read versioned options defensively: only fields within the caller's
        // `struct_size` are valid to read. This ABI revision defines no behavioral
        // fields, so a non-null `opts` currently changes nothing — but reading
        // `struct_size` validates the contract and reserves the extension point.
        if !opts.is_null() {
            // SAFETY: `struct_size` is the first field, so it is always within the
            // bytes the caller promises to back `opts` with; read unaligned to be safe.
            let struct_size = unsafe { ptr::read_unaligned(ptr::addr_of!((*opts).struct_size)) };
            let _ = struct_size;
        }
        // out_len is valid from here: define it (0) so every non-NullArg return
        // leaves it in a known state; the count overwrites it on success below.
        // SAFETY: out_len checked non-null; caller guarantees it is writable.
        unsafe { *out_len = 0 };
        if model.is_null()
            || (prompt.is_null() && prompt_len != 0)
            || (out.is_null() && out_cap != 0)
        {
            set_last_error("a required pointer argument was null");
            return TritiumStatus::NullArg;
        }
        // SAFETY: non-null per the check; caller guarantees a live, exclusively-held handle.
        let model = unsafe { &mut *model };
        let prompt_slice = if prompt_len == 0 {
            &[][..]
        } else {
            // SAFETY: non-null + len checked; caller guarantees `prompt_len` valid u32s.
            unsafe { std::slice::from_raw_parts(prompt, prompt_len) }
        };
        let tokens = match model.runner.generate(prompt_slice, max_new as usize, eos) {
            Ok(t) => t,
            Err(e) => {
                set_last_error(&format!("generate: {e}"));
                return TritiumStatus::Generate;
            }
        };
        // SAFETY: out_len checked non-null + writable per contract.
        unsafe { *out_len = tokens.len() };
        if tokens.len() > out_cap {
            return TritiumStatus::BufferTooSmall;
        }
        if !out.is_null() && !tokens.is_empty() {
            // SAFETY: out points to out_cap >= tokens.len() writable u32s (checked).
            unsafe { ptr::copy_nonoverlapping(tokens.as_ptr(), out, tokens.len()) };
        }
        TritiumStatus::Ok
    }));
    outcome.unwrap_or_else(|_| {
        set_last_error("internal panic during generation");
        TritiumStatus::Panic
    })
}

/// Free a model handle returned by [`tritium_model_load_file`]. Null-safe;
/// double-free is the caller's responsibility (do not call twice on one handle).
///
/// # Safety
/// `model` must be null or a handle from [`tritium_model_load_file`] not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tritium_model_free(model: *mut TritiumModel) {
    if model.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: non-null + caller guarantees it came from Box::into_raw and is unfreed.
        drop(unsafe { Box::from_raw(model) });
    }));
}
