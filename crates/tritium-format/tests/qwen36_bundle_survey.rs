//! What the shipped Qwen3.6 SALT V2 package actually contains.
//!
//! Converting the decode path to the resident TQ2_0 GEMM (measured 1.5-2.3x
//! faster in `tritium-cuda`'s `salt_kernel_headroom`) hinges on two properties
//! of the real artifact rather than on the kernels: the scale-group width, which
//! decides whether a repack preserves numerics, and the plane distribution,
//! which decides what uniform-plane padding would cost in VRAM.
//!
//! Gated on `TRITIUM_QWEN36_PACKAGE` pointing at a `.tsalt2` file.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;

use tritium_format::salt_v2_package::SaltV2PackageReader;

fn package_path() -> Option<PathBuf> {
    match std::env::var("TRITIUM_QWEN36_PACKAGE") {
        Ok(value) if !value.is_empty() => Some(PathBuf::from(value)),
        _ => None,
    }
}

#[test]
#[ignore = "needs TRITIUM_QWEN36_PACKAGE pointing at a .tsalt2 file"]
fn survey_scale_groups_and_plane_density() {
    let Some(path) = package_path() else {
        eprintln!("skipping: set TRITIUM_QWEN36_PACKAGE to a .tsalt2 file");
        return;
    };
    let file = File::open(&path).unwrap();
    let reader = SaltV2PackageReader::new_strict(file).unwrap();

    let names: Vec<String> = reader.tensor_names().map(str::to_owned).collect();
    let mut groups: BTreeMap<usize, usize> = BTreeMap::new();
    let mut wide_k = 0usize;
    let mut total_coefficients = 0u64;
    let mut total_planes = 0u64;
    let mut total_tiles = 0u64;

    for name in &names {
        let info = reader.tensor_info(name).expect("named tensor");
        *groups.entry(info.scale_group_size()).or_default() += 1;
        let dims = info.dims();
        // Contraction width is the trailing dimension.
        if dims.last().copied().unwrap_or(0) > 8192 {
            wide_k += 1;
        }
        total_coefficients += info.logical_coefficients() as u64;
        total_planes += info.present_planes() as u64;
        total_tiles += info.tile_count() as u64;
    }

    let mean_planes = total_planes as f64 / total_tiles.max(1) as f64;
    eprintln!("codec              {:?}", reader.codec());
    eprintln!("tensors            {}", names.len());
    eprintln!("scale groups       {groups:?}");
    eprintln!("tensors with K>8192 (past TILED_K_MAX) {wide_k}");
    eprintln!("coefficients       {total_coefficients}");
    eprintln!("tiles              {total_tiles}");
    eprintln!("planes present     {total_planes}");
    eprintln!("mean planes/tile   {mean_planes:.3}");
    // TQ2_0 costs ~2.0625 bits per trit per plane, and `SaltResidentLinear`
    // pads every tensor to its own maximum plane count.
    let present_gib = total_coefficients as f64 * mean_planes * 2.0625 / 8.0 / (1 << 30) as f64;
    let padded_gib = total_coefficients as f64 * 3.0 * 2.0625 / 8.0 / (1 << 30) as f64;
    eprintln!("TQ2_0 at mean density  {present_gib:.2} GiB");
    eprintln!("TQ2_0 padded to 3      {padded_gib:.2} GiB");
    assert!(!names.is_empty());
}
