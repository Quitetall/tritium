//! ABI-level tests: every entry point's null/error handling, the ABI version,
//! and the version string — all callable in-process through the rlib. A gated
//! real-model round-trip (ignored unless `TRITIUM_FFI_MODEL=<gguf>` is set)
//! exercises load -> generate -> free end to end.

use std::ffi::CStr;
use std::ptr::{self, NonNull};
use tritium_ffi::{
    TRITIUM_ABI_VERSION, TritiumModel, TritiumStatus, tritium_abi_version, tritium_generate,
    tritium_model_free, tritium_model_load_file, tritium_version,
};

// A well-aligned, non-null `TritiumModel*` that is NEVER dereferenced. The
// guard clauses in `tritium_generate` short-circuit (`||`) on the null/length
// checks *before* the `&mut *model` deref, so this lets us exercise the
// prompt/out/out_len guards in isolation without a loaded model.
fn dangling_model() -> *mut TritiumModel {
    NonNull::<TritiumModel>::dangling().as_ptr()
}

#[test]
fn abi_version_matches_constant() {
    assert_eq!(tritium_abi_version(), TRITIUM_ABI_VERSION);
    assert_eq!(TRITIUM_ABI_VERSION, 1);
}

#[test]
fn version_is_nonnull_and_matches_crate() {
    let p = tritium_version();
    assert!(!p.is_null());
    // SAFETY: `tritium_version` returns a 'static NUL-terminated string.
    let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap();
    assert_eq!(s, env!("CARGO_PKG_VERSION"));
}

#[test]
fn generate_null_model_is_nullarg() {
    let mut out = [0u32; 4];
    let mut out_len = 0usize;
    // SAFETY: null model is the documented error path; the rest is valid.
    let st = unsafe {
        tritium_generate(
            ptr::null_mut(),
            ptr::null(),
            0,
            1,
            0,
            out.as_mut_ptr(),
            out.len(),
            &mut out_len,
        )
    };
    assert_eq!(st, TritiumStatus::NullArg);
}

#[test]
fn generate_null_out_len_is_nullarg() {
    // SAFETY: out_len guard fires before the model deref (short-circuit), so the
    // dangling model pointer is never read.
    let st = unsafe {
        tritium_generate(
            dangling_model(),
            ptr::null(),
            0,
            1,
            0,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
        )
    };
    assert_eq!(st, TritiumStatus::NullArg);
}

#[test]
fn generate_null_prompt_with_nonzero_len_is_nullarg() {
    let mut out_len = 0usize;
    // SAFETY: the (prompt null && len != 0) guard fires before any deref.
    let st = unsafe {
        tritium_generate(
            dangling_model(),
            ptr::null(),
            3,
            1,
            0,
            ptr::null_mut(),
            0,
            &mut out_len,
        )
    };
    assert_eq!(st, TritiumStatus::NullArg);
}

#[test]
fn generate_null_out_with_nonzero_cap_is_nullarg() {
    let prompt = [1u32, 2, 3];
    let mut out_len = 0usize;
    // SAFETY: the (out null && cap != 0) guard fires before any deref.
    let st = unsafe {
        tritium_generate(
            dangling_model(),
            prompt.as_ptr(),
            prompt.len(),
            1,
            0,
            ptr::null_mut(),
            4,
            &mut out_len,
        )
    };
    assert_eq!(st, TritiumStatus::NullArg);
}

#[test]
fn model_free_null_is_noop() {
    // SAFETY: documented null-safe.
    unsafe { tritium_model_free(ptr::null_mut()) };
}

#[test]
fn load_null_path_sets_nullarg() {
    let mut status = TritiumStatus::Ok;
    // SAFETY: null path is the documented error path; out_status is valid.
    let m = unsafe { tritium_model_load_file(ptr::null(), &mut status) };
    assert!(m.is_null());
    assert_eq!(status, TritiumStatus::NullArg);
}

#[test]
fn load_null_path_null_status_does_not_crash() {
    // SAFETY: both args are the documented null cases; nothing is written/read.
    let m = unsafe { tritium_model_load_file(ptr::null(), ptr::null_mut()) };
    assert!(m.is_null());
}

#[test]
fn load_missing_file_sets_load_error() {
    let path = c"/nonexistent/tritium/model.gguf";
    let mut status = TritiumStatus::Ok;
    // SAFETY: valid NUL-terminated path + valid out_status.
    let m = unsafe { tritium_model_load_file(path.as_ptr(), &mut status) };
    assert!(m.is_null());
    assert_eq!(status, TritiumStatus::Load);
}

// ---- gated real-model round-trip ----------------------------------------
// Run with: TRITIUM_FFI_MODEL=/path/to/model.gguf \
//           cargo test -p tritium-ffi -- --ignored model_roundtrip
#[test]
#[ignore = "needs TRITIUM_FFI_MODEL=<gguf path>"]
fn model_roundtrip() {
    let path = std::env::var("TRITIUM_FFI_MODEL").expect("set TRITIUM_FFI_MODEL");
    let cpath = std::ffi::CString::new(path).unwrap();
    let mut status = TritiumStatus::Panic;
    // SAFETY: valid NUL-terminated path + valid out_status.
    let model = unsafe { tritium_model_load_file(cpath.as_ptr(), &mut status) };
    assert_eq!(status, TritiumStatus::Ok, "load failed");
    assert!(!model.is_null());

    let prompt = [1u32, 2, 3];

    // First, a too-small buffer: BufferTooSmall, with the required length reported.
    let mut tiny = [0u32; 0];
    let mut need = 0usize;
    // SAFETY: live model; out is a 0-cap (null-equivalent) slice with cap 0.
    let st = unsafe {
        tritium_generate(
            model,
            prompt.as_ptr(),
            prompt.len(),
            8,
            128001,
            tiny.as_mut_ptr(),
            tiny.len(),
            &mut need,
        )
    };
    assert_eq!(st, TritiumStatus::BufferTooSmall);
    assert!(need > 0, "required length should be reported");

    // Now a buffer that fits.
    let mut out = vec![0u32; need];
    let mut got = 0usize;
    // SAFETY: live model; out has capacity `need` reported above.
    let st = unsafe {
        tritium_generate(
            model,
            prompt.as_ptr(),
            prompt.len(),
            8,
            128001,
            out.as_mut_ptr(),
            out.len(),
            &mut got,
        )
    };
    assert_eq!(st, TritiumStatus::Ok);
    assert_eq!(got, need);

    // SAFETY: handle came from load and has not been freed.
    unsafe { tritium_model_free(model) };
}
