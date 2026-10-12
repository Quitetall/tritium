//! Measures encoder/envelope heap requests, not caller storage or process RSS.
use std::alloc::{GlobalAlloc, Layout, System};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tritium_format::trit_model::{decode_model_manifest, encode_model_manifest};
use tritium_format::{PackageHasher, TritBlob, TritBlobKind, write_trit_package_to};
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, BlobId, BlobReference, Level, LevelFit, ManifestAsset,
    ModelManifest, PlaneAllocation, PlaneCodec, PlaneCounts, SchemaId, TensorMeta, TensorStorage,
    Transport,
};

struct TrackingAllocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

// SAFETY: delegates the supplied layouts and pointers to System.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplies a valid allocation layout.
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() && TRACK.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        result
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller supplies the original pointer and layout.
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: the caller supplies the original pointer/layout and valid size.
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() && TRACK.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(size, Ordering::Relaxed);
        }
        result
    }
}
#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

struct HashSink {
    hash: PackageHasher,
    bytes: usize,
}
impl Write for HashSink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.hash.update(bytes);
        self.bytes += bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn reference(bytes: &[u8]) -> BlobReference {
    BlobReference {
        id: BlobId::from_bytes(*blake3::hash(bytes).as_bytes()),
        shard: None,
    }
}

fn manifest(tiles: usize) -> ModelManifest {
    ModelManifest {
        schema: SchemaId { major: 1, minor: 0 },
        architecture: "fixture-only".into(),
        assets: ["config.json", "tokenizer.json"]
            .into_iter()
            .map(|name| ManifestAsset {
                name: name.into(),
                blob: reference(b"{}"),
            })
            .collect(),
        levels: vec![Level {
            name: "imported".into(),
            fit: LevelFit::Imported,
            tensors: vec![TensorMeta {
                name: "weight".into(),
                blob: reference(&[]),
                storage: TensorStorage::Additive {
                    layout: AdditiveLayout {
                        rows: tiles as u64,
                        cols: 256,
                        tile: 256,
                        group: 128,
                        max_planes: ADMITTED_LAWS[0].max_planes,
                        allocation: PlaneAllocation::PerTile,
                        codec: PlaneCodec::B3,
                        law: ADMITTED_LAWS[0].law,
                        basis: Basis::Identity,
                        transport: Transport::Raw,
                    },
                    planes: PlaneCounts::PerTile(vec![0; tiles]),
                },
            }],
        }],
        shards: vec![],
        legacy_ids: vec![],
    }
}

#[test]
fn sorted_manifest_encode_decode_and_envelope_do_not_allocate_value_per_tile() {
    for tiles in [16 * 1024, 1024 * 1024] {
        let manifest = manifest(tiles);
        manifest.validate().unwrap();
        ALLOCATED.store(0, Ordering::Relaxed);
        TRACK.store(true, Ordering::SeqCst);
        let result = encode_model_manifest(&manifest);
        TRACK.store(false, Ordering::SeqCst);
        let allocated = ALLOCATED.load(Ordering::Relaxed);
        let bytes = result.unwrap();
        println!(
            "{tiles} tile counts: encoder requested {allocated} heap bytes for {} encoded bytes",
            bytes.len()
        );
        assert!(bytes.len() > tiles);
        assert!(
            allocated < bytes.len() * 8 + 65536,
            "encoder requested {allocated} heap bytes for {} encoded bytes",
            bytes.len()
        );

        ALLOCATED.store(0, Ordering::Relaxed);
        TRACK.store(true, Ordering::SeqCst);
        let result = decode_model_manifest(&bytes);
        TRACK.store(false, Ordering::SeqCst);
        let allocated = ALLOCATED.load(Ordering::Relaxed);
        println!("{tiles} tile counts: typed decoder requested {allocated} heap bytes");
        let decoded = result.unwrap();
        assert_eq!(decoded, manifest);
        assert!(
            allocated < bytes.len() * 10 + 65536,
            "typed decoder requested {allocated} heap bytes for {} encoded bytes",
            bytes.len()
        );
        drop(decoded);

        let blobs = [
            TritBlob {
                kind: TritBlobKind::AdditiveTensor,
                bytes: &[],
            },
            TritBlob {
                kind: TritBlobKind::Asset("config.json".into()),
                bytes: b"{}",
            },
            TritBlob {
                kind: TritBlobKind::Asset("tokenizer.json".into()),
                bytes: b"{}",
            },
        ];
        let mut sink = HashSink {
            hash: PackageHasher::new(),
            bytes: 0,
        };
        ALLOCATED.store(0, Ordering::Relaxed);
        TRACK.store(true, Ordering::SeqCst);
        let result = write_trit_package_to(&bytes, &blobs, bytes.len() + 4096, &mut sink);
        TRACK.store(false, Ordering::SeqCst);
        let allocated = ALLOCATED.load(Ordering::Relaxed);
        println!("{tiles} tile counts: envelope requested {allocated} heap bytes");
        assert_eq!(result.unwrap(), sink.hash.finalize());
        assert!(sink.bytes > bytes.len());
        assert!(
            allocated < 65536,
            "envelope requested {allocated} heap bytes for {} encoded metadata bytes",
            bytes.len()
        );
    }
}
