//! Largest allocator request during actual tile-wise import, not process RSS.
use half::f16;
use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SaltV2Package, SaltV2PackageReader, SaltV2Plane, SaltV2Tensor, SaltV2Tile,
    write_salt_v2_package,
};
use tritium_format::{SaltTensorSpoolLimits, import_salt_v2_tensor_to_spool};
use tritium_schema::{
    AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw,
    ScalePrecision, Transport,
};

struct TrackingAllocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static LARGEST: AtomicUsize = AtomicUsize::new(0);
// SAFETY: every layout/pointer is delegated unchanged to System.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: caller supplies a valid allocation layout.
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() && TRACK.load(Ordering::Relaxed) {
            LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        }
        result
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: caller supplies the original pointer/layout.
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: caller supplies original pointer/layout and valid replacement size.
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() && TRACK.load(Ordering::Relaxed) {
            LARGEST.fetch_max(size, Ordering::Relaxed);
        }
        result
    }
}
#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn actual_file_import_never_allocates_a_decoded_coefficient_or_scale_master() {
    struct Guard(std::path::PathBuf);
    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    for tiles in [4096usize, 65536] {
        let tile = SaltV2Tile::new(
            (0..3)
                .map(|p| {
                    SaltV2Plane::new_with_scale_group_size(
                        vec![1; 256],
                        vec![f16::from_f32((p + 1) as f32 * 0.25); 2],
                        128,
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
        let source =
            SaltV2Tensor::new("weight", vec![1, (tiles * 256) as u64], vec![tile; tiles]).unwrap();
        let encoded =
            write_salt_v2_package(&SaltV2Package::new(SaltV2Codec::B3, vec![source]).unwrap())
                .unwrap();
        let mut reader = SaltV2PackageReader::new_strict(Cursor::new(encoded.bytes)).unwrap();
        let path = std::env::temp_dir().join(format!(
            "tritium-v2-spool-memory-{}-{tiles}",
            std::process::id()
        ));
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let guard = Guard(path.clone());
        let layout = AdditiveLayout {
            rows: 1,
            cols: (tiles * 256) as u64,
            tile: 256,
            group: 128,
            max_planes: 3,
            allocation: PlaneAllocation::PerTile,
            codec: PlaneCodec::B3,
            basis: Basis::Identity,
            transport: Transport::Raw,
            law: ScaleLaw {
                anchor: ScaleAnchor::Group,
                relation: PlaneRelation::Free,
                precision: ScalePrecision::F16,
            },
        };
        let limits = SaltTensorSpoolLimits {
            max_source_bytes: 100_000_000,
            max_trits: 100_000_000,
            max_scales: 10_000_000,
            max_payload_bytes: 100_000_000,
            max_working_bytes: 2_000_000,
        };
        LARGEST.store(0, Ordering::Relaxed);
        TRACK.store(true, Ordering::SeqCst);
        let result =
            import_salt_v2_tensor_to_spool(&mut reader, "weight", layout, limits, &mut file);
        TRACK.store(false, Ordering::SeqCst);
        let largest = LARGEST.load(Ordering::Relaxed);
        drop(file);
        drop(guard);
        assert!(!path.exists());
        let staged = result.unwrap();
        assert_eq!(staged.tile_counts, vec![3; tiles]);
        assert!(staged.blob.length > 500_000);
        assert!(
            largest <= 256 * 1024,
            "{tiles} tiles requested {largest} heap bytes at once"
        );
        // Decoded master would require 3/48 MiB trits plus F32 scale arrays.
        // Caller-owned encoded source, reader construction, file cache, aggregate
        // heap/stack/capacity and complete model/process RSS are separate measures.
    }
}
