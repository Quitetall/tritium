//! Autograd-op bindings: the `tritium-train` tape ops (ternary Conv1d, FSQ, STE) exposed as flat-buffer
//! forward/vjp `#[pyfunction]`s. The Python side ([`tritium.autograd`]) wraps each pair in a
//! `torch.autograd.Function`, so LamQuant drops ternary conv/FSQ layers into their PyTorch encoder in
//! place. Tensors cross the boundary as flat `f32`/`u32` lists plus explicit shape args (a numpy/dlpack
//! zero-copy bridge is a perf follow-on); every shape error is a `ValueError`, never a panic.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use tritium_train::ops::conv1d::{self, Conv1dCfg};
use tritium_train::ops::fsq::{self, FsqBound, FsqCfg, FsqSte};
use tritium_train::ops::ste;

#[allow(clippy::too_many_arguments)]
fn conv_cfg(
    batch: usize,
    c_in: usize,
    c_out: usize,
    l_in: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    pad_left: usize,
    pad_right: usize,
    groups: usize,
) -> PyResult<Conv1dCfg> {
    let cfg = Conv1dCfg {
        batch,
        c_in,
        c_out,
        l_in,
        k,
        stride,
        dilation,
        pad_left,
        pad_right,
        groups,
    };
    // buffers_fit also validates geometry (groups divisibility, l_out > 0); check with the true lengths.
    if groups == 0 || k == 0 || stride == 0 || dilation == 0 {
        return Err(PyValueError::new_err(
            "groups/k/stride/dilation must be >= 1",
        ));
    }
    if !c_in.is_multiple_of(groups) || !c_out.is_multiple_of(groups) {
        return Err(PyValueError::new_err(
            "c_in and c_out must both be divisible by groups",
        ));
    }
    if cfg.l_out() == 0 {
        return Err(PyValueError::new_err(
            "kernel (with dilation) is wider than the padded input (l_out == 0)",
        ));
    }
    Ok(cfg)
}

/// Ternary Conv1d forward `Y[B,C_out,L_out] = scale ⊙ conv1d(X, W)`, flat row-major buffers.
/// `x:[B·C_in·L_in]`, `w:[C_out·(C_in/groups)·K]`, `scale:[C_out]`. Returns `[B·C_out·L_out]`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_forward(
    x: Vec<f32>,
    w: Vec<f32>,
    scale: Vec<f32>,
    batch: usize,
    c_in: usize,
    c_out: usize,
    l_in: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    pad_left: usize,
    pad_right: usize,
    groups: usize,
) -> PyResult<Vec<f32>> {
    let cfg = conv_cfg(
        batch, c_in, c_out, l_in, k, stride, dilation, pad_left, pad_right, groups,
    )?;
    let out_len = batch * c_out * cfg.l_out();
    if !cfg.buffers_fit(x.len(), w.len(), scale.len(), out_len) {
        return Err(PyValueError::new_err(format!(
            "conv1d shape mismatch: x={}, w={}, scale={} (expected x={}, w={}, scale={})",
            x.len(),
            w.len(),
            scale.len(),
            batch * c_in * l_in,
            c_out * cfg.k_g(),
            c_out
        )));
    }
    Ok(conv1d::forward(&x, &w, &scale, &cfg))
}

/// Ternary Conv1d backward. Returns `(gX, gW, gScale)` (same shapes as `x`, `w`, `scale`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_vjp(
    x: Vec<f32>,
    w: Vec<f32>,
    scale: Vec<f32>,
    grad_out: Vec<f32>,
    batch: usize,
    c_in: usize,
    c_out: usize,
    l_in: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    pad_left: usize,
    pad_right: usize,
    groups: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let cfg = conv_cfg(
        batch, c_in, c_out, l_in, k, stride, dilation, pad_left, pad_right, groups,
    )?;
    let out_len = batch * c_out * cfg.l_out();
    if !cfg.buffers_fit(x.len(), w.len(), scale.len(), out_len) || grad_out.len() != out_len {
        return Err(PyValueError::new_err(
            "conv1d_vjp shape mismatch (x/w/scale/grad_out inconsistent with geometry)",
        ));
    }
    let mut g = conv1d::vjp(&x, &w, &scale, &cfg, &grad_out);
    let g_scale = g.pop().expect("vjp returns [gX, gW, gScale]");
    let g_w = g.pop().expect("vjp returns [gX, gW, gScale]");
    let g_x = g.pop().expect("vjp returns [gX, gW, gScale]");
    Ok((g_x, g_w, g_scale))
}

fn parse_bound(s: &str) -> PyResult<FsqBound> {
    match s {
        "tanh" => Ok(FsqBound::Tanh),
        "clamp" => Ok(FsqBound::Clamp),
        other => Err(PyValueError::new_err(format!(
            "unknown FSQ bound {other:?} (want \"tanh\" or \"clamp\")"
        ))),
    }
}

fn parse_ste(kind: &str, alpha: f32, seed: u64) -> PyResult<FsqSte> {
    match kind {
        "hard" => Ok(FsqSte::Hard),
        "soft" => Ok(FsqSte::SoftRound { alpha }),
        "stochastic" => Ok(FsqSte::Stochastic { seed }),
        other => Err(PyValueError::new_err(format!(
            "unknown FSQ STE {other:?} (want \"hard\", \"soft\", or \"stochastic\")"
        ))),
    }
}

fn fsq_cfg(channels: usize, len: usize, levels: Vec<u32>, bound: &str) -> PyResult<FsqCfg> {
    let cfg = FsqCfg {
        channels,
        len,
        levels,
        bound: parse_bound(bound)?,
    };
    if cfg.levels.len() != channels {
        return Err(PyValueError::new_err(
            "levels must have one entry per channel",
        ));
    }
    if cfg.levels.iter().any(|&l| l < 2) {
        return Err(PyValueError::new_err("every FSQ level L must be >= 2"));
    }
    Ok(cfg)
}

/// FSQ forward: round each channel of `x:[channels·len]` to its `L`-level grid. `ste_kind` is
/// `"hard"`/`"soft"`/`"stochastic"`; `bound` is `"tanh"`/`"clamp"`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fsq_forward(
    x: Vec<f32>,
    channels: usize,
    len: usize,
    levels: Vec<u32>,
    bound: &str,
    ste_kind: &str,
    alpha: f32,
    seed: u64,
) -> PyResult<Vec<f32>> {
    let cfg = fsq_cfg(channels, len, levels, bound)?;
    if !cfg.buffers_fit(x.len()) {
        return Err(PyValueError::new_err(format!(
            "fsq shape mismatch: x={} (expected {})",
            x.len(),
            channels * len
        )));
    }
    Ok(fsq::forward(&x, &cfg, parse_ste(ste_kind, alpha, seed)?))
}

/// FSQ backward `gX`. `ste_kind` selects the straight-through estimator (`stochastic` uses the hard
/// gradient); `alpha` is the soft-round annealing coefficient.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fsq_vjp(
    x: Vec<f32>,
    grad_out: Vec<f32>,
    channels: usize,
    len: usize,
    levels: Vec<u32>,
    bound: &str,
    ste_kind: &str,
    alpha: f32,
) -> PyResult<Vec<f32>> {
    let cfg = fsq_cfg(channels, len, levels, bound)?;
    if !cfg.buffers_fit(x.len()) || grad_out.len() != x.len() {
        return Err(PyValueError::new_err("fsq_vjp shape mismatch"));
    }
    let mut g = fsq::vjp(&x, &cfg, parse_ste(ste_kind, alpha, 0)?, &grad_out);
    Ok(g.pop().expect("fsq vjp returns [gX]"))
}

/// Per-row AbsMean quantizer scale for `[rows, cols]` latent weights (the ternary conv weight is
/// reshaped `[C_out, (C_in/groups)·K]`, so this is the per-output-channel scale).
#[pyfunction]
pub(crate) fn ste_absmean_scale(wf: Vec<f32>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
    if wf.len() != rows * cols {
        return Err(PyValueError::new_err("wf length must be rows*cols"));
    }
    Ok(ste::absmean_scale_per_row(&wf, rows, cols))
}

/// STE forward: `trit = round(clamp(Wf/s_q, -1, 1))` in `{-1,0,+1}` (the quantized weight before scale).
#[pyfunction]
pub(crate) fn ste_quantize_forward(
    wf: Vec<f32>,
    s_q: Vec<f32>,
    rows: usize,
    cols: usize,
) -> PyResult<Vec<f32>> {
    if wf.len() != rows * cols || s_q.len() != rows {
        return Err(PyValueError::new_err(
            "shape mismatch (wf=rows*cols, s_q=rows)",
        ));
    }
    Ok(ste::quantize_forward(&wf, &s_q, rows, cols))
}

/// STE backward: `gWf = grad/s_q` masked to `|Wf/s_q| < 1` (stop-gradient on the scale).
#[pyfunction]
pub(crate) fn ste_quantize_vjp(
    wf: Vec<f32>,
    s_q: Vec<f32>,
    rows: usize,
    cols: usize,
    grad_out: Vec<f32>,
) -> PyResult<Vec<f32>> {
    if wf.len() != rows * cols || s_q.len() != rows || grad_out.len() != rows * cols {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    let g = ste::quantize_vjp(&wf, &s_q, rows, cols, &grad_out);
    Ok(g.into_iter()
        .next()
        .expect("quantize_vjp returns [gWf, gs]"))
}
