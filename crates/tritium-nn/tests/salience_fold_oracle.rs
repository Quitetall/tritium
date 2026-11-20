//! Common caller fixture for actual pre-change/current production fold comparison.
//! This contains no numerical fold implementation or identity/receipt oracle.
use tritium_nn::calibrate::{Arch, Calib, fold};

#[test]
fn actual_fold_keeps_untouched_tensors_and_exports_bits_when_requested() {
    let mut output = Vec::new();
    for layers in [1, 3] {
        let a = Arch {
            attn_norms: vec![vec![1.125; 8]; layers],
            ffn_norms: vec![vec![0.875; 8]; layers],
            qk_norms: vec![None; layers],
            out_norm: vec![1.5; 8],
            n_embd: 8,
            n_head: 4,
            n_head_kv: 2,
            head_dim: 2,
            ff: 12,
            vocab: 7,
            eps: 1e-5,
            theta: 10_000.0,
            n_layers: layers,
        };
        let mut shapes = vec![(7, 8)];
        for _ in 0..layers {
            shapes.extend([(8, 8), (4, 8), (4, 8), (8, 8), (12, 8), (12, 8), (8, 12)]);
        }
        let fp: Vec<Vec<f32>> = shapes
            .iter()
            .enumerate()
            .map(|(t, &(rows, cols))| {
                (0..rows * cols)
                    .map(|i| ((i * 17 + t * 11) % 199) as f32 * 0.03125 - 3.0)
                    .collect()
            })
            .collect();
        for mode in 0..3 {
            let moments = |width: usize, layer: usize| {
                (0..width)
                    .map(|channel| match mode {
                        0 => 0.0,
                        1 => 32.0,
                        _ => ((channel + 1 + layer) as f64).powi(4) * 0.125,
                    })
                    .collect::<Vec<_>>()
            };
            let c = Calib {
                attn_in: (0..layers).map(|l| moments(8, l)).collect(),
                ffn_in: (0..layers).map(|l| moments(8, l + 1)).collect(),
                down_in: (0..layers).map(|l| moments(12, l + 2)).collect(),
                rows: 32,
            };
            for alpha in [0.0, 0.75, 1.0] {
                let (weights, arch) = fold(&fp, &shapes, &a, &c, alpha);
                assert_eq!(weights.len(), fp.len());
                assert_eq!(weights[0], fp[0]);
                assert_eq!(arch.out_norm, a.out_norm);
                for layer in 0..layers {
                    assert_eq!(weights[1 + 7 * layer + 3], fp[1 + 7 * layer + 3]);
                }
                for values in weights
                    .iter()
                    .chain(&arch.attn_norms)
                    .chain(&arch.ffn_norms)
                    .chain(core::iter::once(&arch.out_norm))
                {
                    output.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for value in values {
                        assert!(value.is_finite());
                        output.extend_from_slice(&value.to_bits().to_le_bytes());
                    }
                }
            }
        }
    }
    if let Some(path) = std::env::var_os("TRITIUM_FOLD_ORACLE_OUTPUT") {
        std::fs::write(path, output).expect("write explicitly requested oracle bits");
    }
}
