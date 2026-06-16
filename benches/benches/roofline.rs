//! Roofline / %-of-SOL harness (v0.30 WF-E, ADR 0005) — pure host arithmetic, no
//! GPU, always-on.
//!
//! This bench computes (and prints, via divan's per-bench output) the **decode
//! tokens/sec ceiling** for BitNet 2B4T on the pinned RTX 4090 — the
//! `bandwidth / model_bytes` denominator ADR 0005's utilization gate reports the
//! achieved fraction against — and the committed competitor baselines. It needs no
//! device, so it runs on every lane and pins the ceiling math under test.
//!
//! ## What "% of SOL" means here, and how to *measure* the achieved fraction
//!
//! "SOL" = Speed-Of-Light, the hardware ceiling a kernel could hit if it were
//! perfectly bound by its limiting resource. ADR 0005 splits it by regime:
//!
//! - **Decode (batch-1, memory-bound).** The achieved ceiling is the **HBM
//!   bandwidth** one: a decode step streams every weight once, so
//!   `tok/s ≤ peak_HBM_BW / model_bytes`. The roofline number here is exactly that
//!   denominator ([`tritium_benches::bitnet_2b4t_decode_ceiling`]). Measure the
//!   achieved fraction with `ncu`'s **DRAM throughput % of SOL** (a.k.a. achieved
//!   HBM BW / peak): the gate wants decode within ~10% of peak (≥ ~90% of SOL).
//!
//! - **Prefill (large-M, compute-bound).** The ceiling is the **int8 tensor-core
//!   throughput** one: `tok/s ≤ peak_int8_TOPS / macs_per_token`. Measure with
//!   `ncu`'s **`sm__pipe_tensor_op_imma` active %** + the achieved int8 TOPS vs
//!   [`tritium_benches::RTX_4090_PEAK_INT8_TOPS`]; the gate wants tensor-op-active
//!   high.
//!
//! The exact `ncu` invocations live in `benches/README.md`; this wave provides the
//! harness + the command + the ceiling math (we do **not** run `ncu` here).

use divan::counter::ItemsCount;
use tritium_benches::{
    BITNET_2B4T_I2S_BYTES, BITNET_CPP_2B4T_DECODE, LLAMA_CPP_2B4T_DECODE,
    RTX_4090_PEAK_HBM_BW_BYTES_PER_SEC, RTX_4090_PEAK_INT8_TOPS, bitnet_2b4t_decode_ceiling,
    decode_bandwidth_ceiling,
};

fn main() {
    // Print the roofline summary once before the divan harness runs, so the numbers
    // are visible even when the bench output is terse. divan::main() then runs the
    // (trivial) timed benches below.
    print_roofline_summary();
    divan::main();
}

/// Emit the decode ceiling + the committed competitor baselines to stdout. Called
/// once from `main` so a plain `cargo bench -p tritium-benches --bench roofline`
/// shows the SOL context a `ncu` run is compared against.
fn print_roofline_summary() {
    let ceiling = bitnet_2b4t_decode_ceiling();
    let gib = BITNET_2B4T_I2S_BYTES as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("──────────────────────── BitNet 2B4T roofline (RTX 4090) ────────────────────────");
    println!(
        "peak HBM bandwidth       : {:.0} GB/s",
        RTX_4090_PEAK_HBM_BW_BYTES_PER_SEC / 1.0e9
    );
    println!("model weight bytes       : {BITNET_2B4T_I2S_BYTES} B ({gib:.3} GiB, I2_S GGUF)");
    println!("decode ceiling (mem)     : {ceiling:.1} tok/s  = peak_HBM_BW / model_bytes");
    println!(
        "peak int8 tensor TOPS    : {RTX_4090_PEAK_INT8_TOPS:.1} TOPS (prefill compute roofline)"
    );
    println!("competitor baselines:");
    for b in [&BITNET_CPP_2B4T_DECODE, &LLAMA_CPP_2B4T_DECODE] {
        println!(
            "  {:<38} {:>6.1} tok/s  [{:?}]",
            b.name, b.tokens_per_sec, b.source
        );
    }
    println!("measure achieved % of SOL with `ncu` — see benches/README.md.");
    println!("─────────────────────────────────────────────────────────────────────────────────");
}

/// Bench the decode-ceiling computation across a sweep of model sizes — trivial
/// arithmetic, but it pins the [`decode_bandwidth_ceiling`] formula under divan and
/// gives the always-on `roofline` bench something to time so the harness is real.
/// `ItemsCount` = number of sizes evaluated.
#[divan::bench(args = [1usize, 2, 4, 8])]
fn decode_ceiling_sweep(bencher: divan::Bencher, billions_of_bytes: usize) {
    // Model sizes 1..8 GB-ish, exercising the ceiling formula off the BitNet point.
    let model_bytes = (billions_of_bytes as u64) * 1_000_000_000;
    bencher.counter(ItemsCount::new(1usize)).bench(|| {
        divan::black_box(decode_bandwidth_ceiling(
            divan::black_box(RTX_4090_PEAK_HBM_BW_BYTES_PER_SEC),
            divan::black_box(model_bytes),
        ))
    });
}
