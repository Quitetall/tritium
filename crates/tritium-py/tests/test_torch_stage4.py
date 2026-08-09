"""ADR 0037 Stage 4 gates: consolidated quantizers/FSQ/progressive/KD-cache/
GGUF-writer + the bf16 autocast mode. Parity tests import the originals by
absolute path and skip cleanly off the dev box."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from tritium.torch import quantizers as q
from tritium.torch import fsq_codec, progressive
from tritium.torch.estimators import create_estimator, registered_estimators

CODEC = Path("/mnt/4tb/LamQuant/codec-neural")
LAMU_PY = Path("/mnt/4tb/LamQuant/training/cookbooks/lamu/python/lamu")
LQ_STUDENT = Path("/mnt/4tb/LamQuant/training/cookbooks/lamquant/python")

needs_codec = pytest.mark.skipif(not CODEC.is_dir(), reason="LamQuant codec-neural absent")
needs_lamu = pytest.mark.skipif(not LAMU_PY.is_dir(), reason="blut-lamu python absent")


def _orig_blocks():
    if str(CODEC) not in sys.path:
        sys.path.insert(0, str(CODEC))
    from lamquant_neural.models import blocks

    return blocks


# ---------------------------------------------------------------- weight fns
@needs_codec
def test_lsq_tequila_parity_forward_and_grads():
    blocks = _orig_blocks()
    torch.manual_seed(0)
    w = torch.randn(6, 5, 4, requires_grad=True)
    a = (torch.rand(6, 1, 1) + 0.2).requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    a2 = a.detach().clone().requires_grad_(True)
    y1 = q._LSQTernaryFunction.apply(w, a, 0.1, 0.07)
    y2 = blocks._LSQTernaryFunction.apply(w2, a2, 0.1, 0.07)
    assert torch.equal(y1, y2)
    g = torch.randn_like(y1)
    y1.backward(g)
    y2.backward(g)
    assert torch.equal(w.grad, w2.grad) and torch.equal(a.grad, a2.grad)


@needs_codec
def test_pareto_seq_parity_forward_and_grads():
    blocks = _orig_blocks()
    torch.manual_seed(1)
    w = torch.randn(4, 7, 3, requires_grad=True)
    a = (torch.rand(4, 1, 1) + 0.2).requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    a2 = a.detach().clone().requires_grad_(True)
    y1 = q._SEQTernaryFunction.apply(w, a, 0.1)
    y2 = blocks._SEQTernaryFunction.apply(w2, a2, 0.1)
    assert torch.equal(y1, y2)
    g = torch.randn_like(y1)
    y1.backward(g)
    y2.backward(g)
    assert torch.equal(w.grad, w2.grad) and torch.equal(a.grad, a2.grad)


@needs_codec
def test_activation_quant_and_hadamard_parity():
    blocks = _orig_blocks()
    torch.manual_seed(2)
    x = torch.randn(2, 3, 96, requires_grad=True)  # 3 full 32-blocks
    x2 = x.detach().clone().requires_grad_(True)
    y1 = q._quantize_activation(x)
    y2 = blocks._quantize_activation(x2)
    assert torch.equal(y1, y2)
    y1.sum().backward()
    y2.sum().backward()
    assert torch.equal(x.grad, x2.grad)
    # Remainder path (T % 32 != 0) too.
    xr = torch.randn(1, 2, 40)
    assert torch.equal(q._quantize_activation(xr), blocks._quantize_activation(xr))


@needs_lamu
def test_a8_act_quant_parity():
    if str(LAMU_PY) not in sys.path:
        sys.path.insert(0, str(LAMU_PY))
    import bitnet_student

    torch.manual_seed(3)
    x = torch.randn(5, 11)
    assert torch.equal(q.act_quant_ste(x.clone()), bitnet_student.act_quant_ste(x.clone()))


# ---------------------------------------------------------------- FSQ codec
@needs_codec
def test_fsq_seeded_functions_parity():
    if str(CODEC) not in sys.path:
        sys.path.insert(0, str(CODEC))
    from lamquant_neural.models import quantizer as orig

    torch.manual_seed(4)
    x = torch.randn(3, 4, 16)
    seeds = [11, 22, 33]
    for name in ("fsq_dither_seeded", "fsq_dropout_ste_seeded"):
        ours = getattr(fsq_codec, name)(x.clone(), seeds)
        theirs = getattr(orig, name)(x.clone(), seeds)
        assert torch.equal(ours, theirs), name
    lvl = fsq_codec.FSQ_LEVELS[3]
    assert torch.equal(fsq_codec.fsq_infer(x, lvl), orig.fsq_infer(x, lvl))
    assert fsq_codec.FSQ_LEVELS == tuple(orig.FSQ_LEVELS)


# ---------------------------------------------------------------- progressive
@needs_codec
def test_progressive_schedule_parity():
    if str(LQ_STUDENT) not in sys.path:
        sys.path.insert(0, str(LQ_STUDENT))
    from lamquant.student import progressive_quant as orig

    ours = progressive.ProgressiveQuantSchedule(total_epochs=100)
    theirs = orig.ProgressiveQuantSchedule(total_epochs=100)
    series_ours = [ours.get_bits(e) for e in range(100)]
    series_theirs = [theirs.get_bits(e) for e in range(100)]
    assert series_ours == series_theirs
    assert set(series_ours) >= {8, 4, "ternary"}  # walks the INT-N ladder
    assert series_ours[-1] == "ternary"


# ---------------------------------------------------------------- estimators
def test_registry_has_consolidated_estimators():
    names = registered_estimators()
    assert "tequila-lsq" in names and "pareto-seq" in names


@pytest.mark.parametrize("name", ["tequila-lsq", "pareto-seq"])
def test_consolidated_estimator_projects(name):
    est = create_estimator(name)
    torch.manual_seed(5)
    master = torch.randn(8, 16, requires_grad=True)
    from tritium.torch.estimators import ProjectionContext

    proj = est.project(master, context=ProjectionContext(step=0))
    (plane,) = proj.planes
    assert plane.trits.dtype == torch.int8
    assert int(plane.trits.abs().max()) <= 1
    assert plane.scales.shape == (8, 1)
    proj.dense.sum().backward()
    assert master.grad is not None
    # alpha is learnable
    assert any(p.requires_grad for p in est.parameters())


def test_tequila_tau_anneal_changes_soft_code_only():
    est = create_estimator("tequila-lsq")
    torch.manual_seed(6)
    master = torch.randn(4, 8)
    from tritium.torch.estimators import ProjectionContext

    p1 = est.project(master, context=ProjectionContext(step=0))
    est.set_tau(0.0)
    p2 = est.project(master, context=ProjectionContext(step=1))
    assert torch.equal(p1.planes[0].trits, p2.planes[0].trits)  # hard code unchanged
    assert not torch.equal(p1.dense, p2.dense)  # deadzone bias annealed away
    assert not p1.exportable and p2.exportable  # exportable only at tau == 0


# ---------------------------------------------------------------- KD cache
@needs_lamu
def test_kd_cache_module_identity():
    ours = Path(q.__file__).parent / "kd_cache.py"
    theirs_src = (LAMU_PY / "kd_cache.py").read_text()
    assert "tritium.torch.kd_cache" in theirs_src  # cookbook is the shim now
    from tritium.torch.kd_cache import TopkCache, stream_key  # noqa: F401


# ---------------------------------------------------------------- autocast
def test_cuda_autocast_dtype_resolver(monkeypatch):
    from tritium.torch import ops

    monkeypatch.setenv("TRITIUM_CUDA_AUTOCAST", "bf16")
    assert ops._resolve_cuda_autocast_dtype() == torch.bfloat16
    monkeypatch.setenv("TRITIUM_CUDA_AUTOCAST", "fp16")
    assert ops._resolve_cuda_autocast_dtype() == torch.float16
    monkeypatch.setenv("TRITIUM_CUDA_AUTOCAST", "int8")
    with pytest.raises(ValueError):
        ops._resolve_cuda_autocast_dtype()
