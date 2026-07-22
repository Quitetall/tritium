"""Safety + correctness tests for the `tritium` Python extension.

These cover the FFI contract the wheel must uphold:

1.  Wrong dtype / wrong shape raises a *Python exception* (never a segfault or an
    abort that takes the interpreter down with it).
2.  A non-ternary weight value is rejected with a catchable exception.
3.  `ternary_matmul` computes the correct result for a hand-checked case.
4.  Loading a malformed / partial model raises (no panic across the boundary).

Run with: `pytest crates/tritium-py/tests/` after `maturin develop` (or
`pip install -e`). They need only the committed tiny GGUF fixture, not the big
model, so they run fully offline.
"""

import os

import pytest

import tritium


FIXTURE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "tritium-format",
    "tests",
    "fixtures",
    "bitnet_tiny.gguf",
)


def test_module_surface():
    """The module exposes exactly the documented public API."""
    assert hasattr(tritium, "Model")
    assert hasattr(tritium, "QwenLoadReceipt")
    assert hasattr(tritium, "QwenModel")
    assert hasattr(tritium, "QwenReferenceLanguageOutput")
    assert hasattr(tritium, "ternary_matmul")
    assert callable(tritium.ternary_matmul)
    assert tritium.compiled_backends() in (["cpu"], ["cpu", "cuda"])
    assert callable(tritium.Model.load)
    assert callable(tritium.QwenModel.load)
    assert callable(tritium.QwenModel.reference_language)


def test_ternary_matmul_correct():
    """A hand-checked 1x3 @ (2x3)^T ternary matmul with scale 2.0.

    `ternary_matmul` runs the model's real W1.58A8 path: activations are quantized
    to int8 per token (absmax) before the ternary GEMM, so the result carries a few
    percent of quantization error vs. the exact rational answer. We assert the
    correct *sign and magnitude* within that tolerance rather than bit-exactness.
    """
    # act = [[1, 2, 3]], W = [[1, -1, 0], [0, 1, 1]], scale = 2.0
    # exact row0: 2 * (1*1 + 2*-1 + 3*0)  = 2 * (1 - 2)      = -2
    # exact row1: 2 * (1*0 + 2*1  + 3*1)  = 2 * (0 + 2 + 3)  = 10
    out = tritium.ternary_matmul([[1.0, 2.0, 3.0]], [[1, -1, 0], [0, 1, 1]], 2.0)
    assert len(out) == 1
    assert len(out[0]) == 2
    # rel=0.05 comfortably covers the int8 per-token absmax quant error.
    assert out[0][0] == pytest.approx(-2.0, rel=0.05)
    assert out[0][1] == pytest.approx(10.0, rel=0.05)


def test_ternary_matmul_device_is_explicit():
    assert tritium.ternary_matmul([[1.0]], [[1]], 1.0, device="cpu") == [[1.0]]
    if "cuda" in tritium.compiled_backends():
        output = tritium.ternary_matmul([[1.0]], [[1]], 1.0, device="cuda:0")
        assert output[0][0] == pytest.approx(1.0)
    else:
        with pytest.raises(ValueError, match="not compiled with CUDA"):
            tritium.ternary_matmul([[1.0]], [[1]], 1.0, device="cuda:0")
    with pytest.raises(ValueError, match="device must be"):
        tritium.ternary_matmul([[1.0]], [[1]], 1.0, device="magic")


def test_ternary_matmul_wrong_shape_raises():
    """Mismatched K between activations and weights raises ValueError, not a crash."""
    with pytest.raises(ValueError):
        # activation width K=3, but a weight row has width 2 -> shape mismatch.
        tritium.ternary_matmul([[1.0, 2.0, 3.0]], [[1, -1]], 1.0)


def test_ternary_matmul_ragged_rows_raise():
    """Ragged activation rows raise ValueError."""
    with pytest.raises(ValueError):
        tritium.ternary_matmul([[1.0, 2.0], [3.0]], [[1, 0]], 1.0)


def test_ternary_matmul_non_ternary_weight_raises():
    """A weight value outside {-1, 0, 1} raises ValueError."""
    with pytest.raises(ValueError):
        tritium.ternary_matmul([[1.0, 2.0]], [[2, 0]], 1.0)


def test_ternary_matmul_wrong_dtype_raises():
    """Passing a string where a float row is expected raises (TypeError/ValueError),
    rather than segfaulting the interpreter."""
    with pytest.raises((TypeError, ValueError)):
        tritium.ternary_matmul("not a list of rows", [[1, 0]], 1.0)
    with pytest.raises((TypeError, ValueError)):
        # Float where an int weight is expected.
        tritium.ternary_matmul([[1.0, 2.0]], [[0.5, 0.5]], 1.0)


def test_ternary_matmul_empty_raises():
    """Empty inputs raise ValueError instead of indexing out of bounds."""
    with pytest.raises(ValueError):
        tritium.ternary_matmul([], [[1, 0]], 1.0)
    with pytest.raises(ValueError):
        tritium.ternary_matmul([[1.0]], [], 1.0)


def test_load_missing_file_raises_value_error():
    """A missing model path raises ValueError (a usage error), not a panic."""
    with pytest.raises(ValueError):
        tritium.Model.load("/nonexistent/model.gguf")


def test_load_partial_fixture_raises_runtime_error():
    """The tiny fixture parses as GGUF but lacks full weights, so load must raise a
    RuntimeError describing the missing tensor -- never crash the interpreter."""
    assert os.path.exists(FIXTURE), f"fixture not found at {FIXTURE}"
    with pytest.raises(RuntimeError):
        tritium.Model.load(FIXTURE)


def test_load_corrupt_bytes_raises_runtime_error(tmp_path):
    """A readable but non-GGUF file raises RuntimeError, not a segfault."""
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"not a gguf file at all")
    with pytest.raises(RuntimeError):
        tritium.Model.load(str(bad))
