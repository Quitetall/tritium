//! # tritium-ffi — C ABI for Tritium inference.
//!
//! A `cdylib` + `staticlib` exposing a small, stable C API: load a GGUF model on
//! the CPU backend and greedily generate token IDs. The committed header lives at
//! `include/tritium.h`; a drift test (`tests/header.rs`) regenerates it with
//! `cbindgen` and fails if it falls out of sync with these declarations.
//!
//! ## Safety contract (the FFI boundary)
//!
//! - **Never panics across the boundary.** Every entry point wraps its body in
//!   `catch_unwind`; a panic becomes [`TritiumStatus::Panic`], never undefined
//!   behavior.
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

use std::ffi::{CStr, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

// Force-link the CPU backend so its `linkme` registration is present for
// `ModelRunner::load_cpu`.
use tritium_cpu as _;
use tritium_nn::ModelRunner;

/// C ABI version. Bump on any breaking change to the exported symbols/layout.
pub const TRITIUM_ABI_VERSION: u32 = 1;

/// Result code returned by the C API. `Ok` is 0; all errors are non-zero.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// An internal panic was caught at the boundary (should not happen).
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
    let set = |s: TritiumStatus| {
        if !out_status.is_null() {
            // SAFETY: checked non-null; caller guarantees a valid writable pointer.
            unsafe { *out_status = s };
        }
    };
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if path.is_null() {
            return Err(TritiumStatus::NullArg);
        }
        // SAFETY: non-null per the check; caller guarantees a NUL-terminated string.
        let path_str = unsafe { CStr::from_ptr(path) }
            .to_str()
            .map_err(|_| TritiumStatus::Load)?;
        let bytes = std::fs::read(path_str).map_err(|_| TritiumStatus::Load)?;
        let runner = ModelRunner::load_cpu(&bytes).map_err(|_| TritiumStatus::Load)?;
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
            set(TritiumStatus::Panic);
            ptr::null_mut()
        }
    }
}

/// Greedily generate up to `max_new` tokens from `prompt` (`prompt_len` token
/// IDs), stopping early at `eos`. Writes up to `out_cap` token IDs into `out` and
/// sets `*out_len` to the number generated. If the generated count exceeds
/// `out_cap`, returns [`TritiumStatus::BufferTooSmall`] with `*out_len` set to the
/// required length (nothing is written).
///
/// # Safety
/// `model` must be a live handle from [`tritium_model_load_file`]; `prompt` must
/// point to `prompt_len` `u32`s (or be null iff `prompt_len == 0`); `out` must
/// point to `out_cap` writable `u32`s (or be null iff `out_cap == 0`); `out_len`
/// must be a valid writable `usize*`. Do not call concurrently on one `model`.
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
) -> TritiumStatus {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if model.is_null()
            || out_len.is_null()
            || (prompt.is_null() && prompt_len != 0)
            || (out.is_null() && out_cap != 0)
        {
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
            Err(_) => return TritiumStatus::Generate,
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
    outcome.unwrap_or(TritiumStatus::Panic)
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
