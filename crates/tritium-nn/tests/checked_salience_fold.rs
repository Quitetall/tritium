use tritium_nn::calibrate::{Arch, Calib, SalienceFoldError as Error, try_fold};

#[test]
fn graph_geometry_and_late_pair_errors_preserve_caller_inputs() {
    let a = Arch {
        attn_norms: vec![vec![1.0; 4]; 2],
        ffn_norms: vec![vec![1.0; 4]; 2],
        qk_norms: vec![None; 2],
        out_norm: vec![1.0; 4],
        n_embd: 4,
        n_head: 2,
        n_head_kv: 1,
        head_dim: 2,
        ff: 6,
        vocab: 3,
        eps: 1e-5,
        theta: 10_000.0,
        n_layers: 2,
    };
    let mut shapes = vec![(3, 4)];
    for _ in 0..2 {
        shapes.extend([(4, 4), (2, 4), (2, 4), (4, 4), (6, 4), (6, 4), (4, 6)]);
    }
    let fp: Vec<Vec<f32>> = shapes.iter().map(|&(r, c)| vec![0.25; r * c]).collect();
    let c = Calib {
        attn_in: vec![vec![1.0, 4.0, 9.0, 16.0]; 2],
        ffn_in: vec![vec![4.0; 4]; 2],
        down_in: vec![vec![9.0; 6]; 2],
        rows: 8,
    };
    assert!(try_fold(&fp, &shapes, &a, &c, 0.75).is_ok());
    let original = fp.clone();
    let mut bad = Calib::new(&a);
    bad.rows = 8;
    bad.down_in[1][5] = f64::NAN;
    assert_eq!(
        try_fold(&fp, &shapes, &a, &bad, 0.75).unwrap_err(),
        Error::Moments
    );
    assert_eq!(fp, original);
    assert_eq!(a.attn_norms, vec![vec![1.0; 4]; 2]);
    assert_eq!(a.ffn_norms, vec![vec![1.0; 4]; 2]);

    let mut wrong_shapes = shapes.clone();
    wrong_shapes[1] = (2, 8); // Same bytes, wrong q input/output graph.
    assert_eq!(
        try_fold(&fp, &wrong_shapes, &a, &c, 0.75).unwrap_err(),
        Error::Geometry
    );
    let mut bad_arch = a.clone();
    bad_arch.attn_norms[1].pop();
    assert_eq!(
        try_fold(&fp, &shapes, &bad_arch, &c, 0.75).unwrap_err(),
        Error::Geometry
    );
    let mut bad = Calib::new(&a);
    assert_eq!(
        try_fold(&fp, &shapes, &a, &bad, 0.75).unwrap_err(),
        Error::EmptyStatistics
    );
    bad.rows = 8;
    assert_eq!(
        try_fold(&fp, &shapes, &a, &bad, f64::NAN).unwrap_err(),
        Error::Alpha
    );
    assert_eq!(fp, original);
}
