//! Count exact full-row-matrix heap buffers, not complete process RSS.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tritium_quantize::engine::*;
use tritium_schema::*;

const COLS: usize = 512;
const MATRIX_BYTES: usize = COLS * COLS * 8;
static TRACK: AtomicBool = AtomicBool::new(false);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
struct Allocator;
fn allocated(bytes: usize) {
    if bytes == MATRIX_BYTES && TRACK.load(Ordering::Relaxed) {
        let live = LIVE.fetch_add(1, Ordering::Relaxed) + 1;
        PEAK.fetch_max(live, Ordering::Relaxed);
    }
}
fn freed(bytes: usize) {
    if bytes == MATRIX_BYTES && TRACK.load(Ordering::Relaxed) {
        LIVE.fetch_sub(1, Ordering::Relaxed);
    }
}
// SAFETY: pointers, layouts and requests are forwarded unchanged to System.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: allocator caller supplies a valid layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        freed(layout.size());
        // SAFETY: allocator caller supplies the original pointer/layout.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: allocator caller supplies the original pointer/layout and valid size.
        let output = unsafe { System.realloc(pointer, layout, size) };
        if !output.is_null() {
            freed(layout.size());
            allocated(size);
        }
        output
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

#[test]
fn actual_owner_fit_streams_metric_without_full_matrix_buffers() {
    let input = vec![0.25; COLS];
    let adjoints = [0.5, 1.];
    let uses = [OwnerScoreUse::Matmul {
        samples: 1,
        inputs: &input,
        adjoints: &adjoints,
    }];
    let token = 1_i64.to_le_bytes();
    let cache = OwnerScoreCache::new(
        OwnerScoreCacheInput {
            source: OriginalSourceId::from_bytes([7; 32]),
            tensor: QuantSourceTensor {
                name: "weight".into(),
                shape: vec![2, COLS as u64],
                precision: ScalePrecision::F32,
            },
            tokenizer: b"tokenizer",
            objective: b"scalar objective",
            context: &[&token],
            meta: OwnerScoreWindowMeta {
                context: vec![OwnerContextMeta {
                    name: "input_ids".into(),
                    dtype: OwnerContextDtype::I64,
                    shape: vec![1],
                    conjugated: false,
                    negated: false,
                }],
                invocations: vec![OwnerInvocationMeta {
                    ordinal: 0,
                    paths: vec!["head".into()],
                    role: OwnerScoreRole::Matmul,
                    samples: 1,
                }],
            },
            observations: Arc::new(
                OwnerScoreWindow::new(2, COLS, &uses, 100_000)
                    .unwrap()
                    .snapshot()
                    .unwrap(),
            ),
        },
        OwnerScoreCacheLimits {
            max_payload_bytes: 100_000,
            max_context_bytes: 1000,
            max_context_tensors: 4,
            max_invocations: 8,
            max_samples: 16,
        },
    )
    .unwrap();
    let caches = [&cache];
    let windows = [WeightedOwnerWindow {
        cache: cache.id(),
        weight: 1.,
    }];
    let metrics = OwnerScoreMetrics::new(
        &caches,
        &windows,
        0.25,
        OwnerScoreMetricLimits {
            max_windows: 8,
            max_payload_bytes: 100_000,
            max_row_bytes: (COLS * 8) as u64,
            max_reduction_entries: (COLS * COLS * 32) as u64,
        },
    )
    .unwrap();
    let plan = DirectFitPlan {
        layout: AdditiveLayout {
            rows: 2,
            cols: COLS as u64,
            tile: ADDITIVE_TILE_SIZE,
            group: 32,
            max_planes: 1,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::B3,
            basis: Basis::Identity,
            transport: Transport::Raw,
            law: ScaleLaw {
                anchor: ScaleAnchor::Group,
                precision: ScalePrecision::F32,
                relation: PlaneRelation::Free,
            },
        },
        profile: DirectFitProfile::JointIndependentGroups(JointGroupFitOptions {
            max_iterations: 2,
            ridge: 1e-8,
            em_restarts: 1,
            ridge_condition_limit: 1e6,
            softened_relay: false,
            modulated_relay: false,
        }),
        levels: vec![DirectFitLevel {
            name: "full".into(),
            planes: 1,
        }],
        limits: DirectFitLimits {
            max_weights: (2 * COLS) as u64,
            max_metric_entries: (COLS * COLS) as u64,
            max_output_trits: (2 * COLS) as u64,
        },
    };
    let weights: Vec<_> = (0..2 * COLS)
        .map(|index| (index % 19) as f32 / 17. - 0.5)
        .collect();
    // Warm lazy runtime initialization outside the measured operation.
    let expected = fit_owner_score_levels(
        &plan,
        metrics.tensor(),
        metrics.source(),
        &weights,
        &metrics,
    )
    .unwrap();
    TRACK.store(true, Ordering::SeqCst);
    let actual = fit_owner_score_levels(
        &plan,
        metrics.tensor(),
        metrics.source(),
        &weights,
        &metrics,
    );
    TRACK.store(false, Ordering::SeqCst);
    assert_eq!(actual.unwrap(), expected);
    assert_eq!(LIVE.load(Ordering::Relaxed), 0);
    let peak = PEAK.load(Ordering::Relaxed);
    eprintln!("matrix_bytes={MATRIX_BYTES} peak_live_matrix_buffers={peak}");
    assert_eq!(
        peak, 0,
        "independent owner fitting should stream without a full-matrix buffer"
    );
}
