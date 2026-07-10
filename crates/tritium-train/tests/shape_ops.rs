//! Tape-level slice/concat round-trip (plan 0040 step 1): splitting a matrix into column blocks and
//! concatenating them back is the identity in the forward, and passes gradient through unchanged —
//! the differentiable reshape that multi-head/GQA attention is built from. (The op-level
//! finite-difference gradchecks live in `ops::shape`'s unit tests.)

use tritium_train::Tape;

#[test]
fn split_then_concat_is_identity_with_flowing_gradient() {
    let (rows, cols) = (3usize, 6usize);
    let x: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.5) - 4.0).collect();

    let mut t = Tape::new();
    let xid = t.leaf(x.clone());
    // Split into three 2-wide column blocks (like 3 heads of head_dim 2), then concat back.
    let a = t.slice_cols(xid, rows, cols, 0, 2);
    let b = t.slice_cols(xid, rows, cols, 2, 2);
    let c = t.slice_cols(xid, rows, cols, 4, 2);
    let joined = t.concat_cols(&[a, b, c], rows, &[2, 2, 2]);

    // Forward is the identity.
    assert_eq!(t.value(joined), x.as_slice());

    // d(sum of outputs)/dx = 1 everywhere: reduce joined to a scalar via mse against zeros so the
    // upstream cotangent is well-defined, then check the slice-boundary weights all carry gradient.
    let zeros = t.leaf(vec![0.0f32; rows * cols]);
    let loss = t.mse(joined, zeros); // mean of joined² → dL/djoined = 2·joined/N
    let grads = t.backward(loss);
    // The gradient reaching x must equal the gradient reaching joined (identity reshape), i.e. the
    // slice+concat neither dropped nor duplicated any element's gradient.
    let n = (rows * cols) as f32;
    for (i, &xv) in x.iter().enumerate() {
        let expected = 2.0 * xv / n; // dL/dx_i via the identity path
        assert!(
            (grads[xid][i] - expected).abs() < 1e-6,
            "grad[{i}] = {} but expected {expected}",
            grads[xid][i]
        );
    }
}
