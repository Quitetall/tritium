import struct

import pytest

from tritium import (
    KroneckerConflictError,
    KroneckerContractError,
    KroneckerEvidenceBuilder,
    KroneckerEvidenceReceipt,
    KroneckerPublicationError,
    KroneckerStateError,
    Qwen36KroneckerCaptureReceipt,
    Qwen36KroneckerCaptureSession,
    Qwen36KroneckerCaptureTask,
)
from tritium.torch import KroneckerCalibrationWriter


def _f32le(values):
    return struct.pack(f"<{len(values)}f", *values)


def _f64le(values):
    return struct.pack(f"<{len(values)}d", *values)


def _builder(
    tmp_path,
    *,
    index=0,
    name="a.weight",
    indexed_output=False,
    curvature="guided-fisher",
):
    return KroneckerEvidenceBuilder(
        str(tmp_path / "evidence"),
        index,
        name,
        2,
        128,
        curvature,
        "01" * 32,
        "02" * 32,
        "03" * 32,
        0.25,
        max_batch_bytes=4096,
        indexed_output=indexed_output,
    )


def test_qwen_capture_session_rejects_invalid_contract_before_source_io(tmp_path):
    assert Qwen36KroneckerCaptureSession.__name__ == "Qwen36KroneckerCaptureSession"
    assert Qwen36KroneckerCaptureTask.__name__ == "Qwen36KroneckerCaptureTask"
    assert Qwen36KroneckerCaptureReceipt.__name__ == "Qwen36KroneckerCaptureReceipt"
    args = [
        str(tmp_path / "model"),
        "not-the-pinned-revision",
        str(tmp_path / "work"),
        str(tmp_path / "evidence"),
        "guided-fisher",
        "02" * 32,
        "03" * 32,
        0.25,
    ]
    with pytest.raises(KroneckerContractError, match="pinned Qwen3.6 revision"):
        Qwen36KroneckerCaptureSession(*args)
    args[1] = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    with pytest.raises(KroneckerContractError, match="max_evidence_bytes must be positive"):
        Qwen36KroneckerCaptureSession(*args, max_evidence_bytes=0)
    args[-1] = float("nan")
    with pytest.raises(KroneckerContractError, match="damping must be finite"):
        Qwen36KroneckerCaptureSession(*args)
    args[-1] = 0.25
    args[5] = "not-a-digest"
    with pytest.raises(KroneckerContractError, match="activation_cache_digest"):
        Qwen36KroneckerCaptureSession(*args)
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "evidence").exists()


def test_binary_batches_publish_one_canonical_record(tmp_path):
    builder = _builder(tmp_path)
    residency = builder.append_batch(
        _f32le([1.0] * 128 + [3.0] * 128),
        2,
        output_factors_f32le=_f32le([1.0, 2.0, 3.0, 4.0]),
        token_weights_f64le=_f64le([2.0, 1.0]),
        token_mask_u8=bytes([1, 0]),
    )
    assert residency == (1, 1)
    receipt = builder.finish()
    assert isinstance(receipt, KroneckerEvidenceReceipt)
    assert receipt.tensor_index == 0
    assert receipt.bytes > 128 * 128 * 8
    assert len(receipt.record_digest) == 64
    assert (tmp_path / "evidence" / "000000.s2kf").stat().st_size == receipt.bytes
    assert not builder.active
    with pytest.raises(KroneckerStateError, match="already finished"):
        builder.finish()


def test_invalid_binary_batch_is_rejected_before_native_mutation(tmp_path):
    builder = _builder(tmp_path, index=1, name="b.weight")
    with pytest.raises(ValueError, match="activations requires"):
        builder.append_batch(
            _f32le([1.0] * 127),
            1,
            output_factors_f32le=_f32le([1.0, 2.0]),
        )
    with pytest.raises(ValueError, match="token_mask_u8"):
        builder.append_batch(
            _f32le([1.0] * 128),
            1,
            output_factors_f32le=_f32le([1.0, 2.0]),
            token_mask_u8=bytes([2]),
        )
    assert builder.append_batch(
        _f32le([1.0] * 128),
        1,
        output_factors_f32le=_f32le([1.0, 2.0]),
    ) == (1, 1)
    assert builder.finish().tensor_index == 1


def test_indexed_binary_batches_bind_rows_without_dense_factors(tmp_path):
    builder = _builder(
        tmp_path,
        index=4,
        name="model.embed_tokens.weight",
        indexed_output=True,
    )
    assert builder.append_indexed_batch(
        _f32le([1.0] * 128 + [3.0] * 128),
        struct.pack("<2Q", 1, 0),
        2,
        output_factors_f32le=_f32le([2.0, -3.0]),
    ) == (1, 1)
    assert builder.finish().tensor_index == 4

    invalid = _builder(
        tmp_path / "invalid",
        index=5,
        name="model.embed_tokens.weight",
        indexed_output=True,
    )
    with pytest.raises(KroneckerContractError, match="outside"):
        invalid.append_indexed_batch(
            _f32le([1.0] * 128), struct.pack("<Q", 2), 1
        )
    assert invalid.append_indexed_batch(
        _f32le([1.0] * 128), struct.pack("<Q", 1), 1
    ) == (1, 1)

    with pytest.raises(KroneckerContractError, match="encoding"):
        invalid.append_batch(
            _f32le([1.0] * 128),
            1,
            output_factors_f32le=_f32le([1.0, 0.0]),
        )
    dense = _builder(tmp_path / "dense", index=6)
    with pytest.raises(KroneckerContractError, match="encoding"):
        dense.append_indexed_batch(
            _f32le([1.0] * 128), struct.pack("<Q", 1), 1
        )


def test_indexed_input_hessian_identity_batch(tmp_path):
    builder = _builder(
        tmp_path,
        index=8,
        name="model.embed_tokens.weight",
        indexed_output=True,
        curvature="input-hessian",
    )
    assert builder.append_indexed_identity_batch(
        struct.pack("<2Q", 1, 0),
        2,
    ) == (1, 1)
    assert builder.finish().tensor_index == 8


def test_constructor_rejects_unbound_or_unsupported_contracts(tmp_path):
    args = [
        str(tmp_path / "evidence"),
        0,
        "a.weight",
        2,
        128,
        "guided-fisher",
        "01" * 32,
        "02" * 32,
        "03" * 32,
        0.25,
    ]
    args[6] = "00" * 32
    with pytest.raises(KroneckerContractError, match="source-model digest"):
        KroneckerEvidenceBuilder(*args)
    args[6] = "01" * 32
    args[5] = "diagonal-fisher"
    with pytest.raises(KroneckerContractError, match="curvature must be"):
        KroneckerEvidenceBuilder(*args)

    args[5] = "guided-fisher"
    args[3] = 0
    with pytest.raises(KroneckerContractError, match="tensor geometry"):
        KroneckerEvidenceBuilder(*args)
    assert not (tmp_path / "evidence").exists()

    args[0] = str(tmp_path / "out-of-range")
    args[1] = 1_000_000
    args[3] = 2
    with pytest.raises(KroneckerContractError, match="six-digit"):
        KroneckerEvidenceBuilder(*args)
    assert not (tmp_path / "out-of-range").exists()


def test_publication_failures_have_stable_retry_classes(tmp_path):
    first = _builder(tmp_path)
    first.append_batch(
        _f32le([1.0] * 128),
        1,
        output_factors_f32le=_f32le([1.0, 2.0]),
    )
    first.finish()

    conflicting = _builder(tmp_path)
    conflicting.append_batch(
        _f32le([2.0] * 128),
        1,
        output_factors_f32le=_f32le([1.0, 2.0]),
    )
    with pytest.raises(KroneckerConflictError):
        conflicting.finish()
    assert not conflicting.active
    with pytest.raises(KroneckerStateError):
        conflicting.finish()

    unavailable = _builder(tmp_path, index=1, name="b.weight")
    unavailable.append_batch(
        _f32le([1.0] * 128),
        1,
        output_factors_f32le=_f32le([1.0, 2.0]),
    )
    (tmp_path / "evidence").rename(tmp_path / "evidence-away")
    with pytest.raises(KroneckerPublicationError):
        unavailable.finish()
    assert unavailable.active

    hostile_root = tmp_path / "hostile"
    hostile_root.mkdir()
    hostile = _builder(hostile_root)
    hostile.append_batch(
        _f32le([1.0] * 128),
        1,
        output_factors_f32le=_f32le([1.0, 2.0]),
    )
    (hostile_root / "evidence" / ".staging").write_bytes(b"not a directory")
    with pytest.raises(KroneckerContractError, match="staging directory"):
        hostile.finish()
    assert not hostile.active


def test_pytorch_writer_checks_shapes_and_streams_tensor_bytes(tmp_path):
    torch = pytest.importorskip("torch")
    writer = KroneckerCalibrationWriter(
        tmp_path / "evidence",
        tensor_index=2,
        tensor_name="c.weight",
        rows=2,
        columns=128,
        curvature="guided-fisher",
        source_model_digest="01" * 32,
        activation_cache_digest="02" * 32,
        token_stream_digest="03" * 32,
        damping=0.25,
        max_batch_bytes=4096,
    )
    with pytest.raises(ValueError, match="last dimension"):
        writer.append(torch.ones(1, 127), torch.ones(1, 2))
    with pytest.raises(ValueError, match="boolean or 0/1"):
        writer.append(
            torch.ones(2, 128),
            torch.ones(2, 2),
            token_mask=torch.tensor([0, 2]),
        )
    with pytest.raises(ValueError, match="output_factors must have shape"):
        writer.append(torch.ones(2, 1, 128), torch.ones(1, 2, 2))
    assert writer.append(
        torch.ones(2, 128),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        token_weights=torch.tensor([2.0, 1.0]),
        token_mask=torch.tensor([True, False]),
    ) == (1, 1)
    assert writer.finish().tensor_index == 2
    assert not writer.active

    bounded = KroneckerCalibrationWriter(
        tmp_path / "bounded",
        tensor_index=3,
        tensor_name="d.weight",
        rows=2,
        columns=128,
        curvature="guided-fisher",
        source_model_digest="01" * 32,
        activation_cache_digest="02" * 32,
        token_stream_digest="03" * 32,
        damping=0.25,
        max_batch_bytes=16,
    )
    with pytest.raises(ValueError, match="batch requires 520 bytes"):
        bounded.append(torch.ones(1, 128), torch.ones(1, 2))


def test_pytorch_writer_streams_indexed_embedding_factors(tmp_path):
    torch = pytest.importorskip("torch")
    writer = KroneckerCalibrationWriter(
        tmp_path / "indexed",
        tensor_index=7,
        tensor_name="model.embed_tokens.weight",
        rows=4,
        columns=128,
        curvature="guided-fisher",
        source_model_digest="01" * 32,
        activation_cache_digest="02" * 32,
        token_stream_digest="03" * 32,
        damping=0.25,
        indexed_output=True,
        max_batch_bytes=4096,
    )
    assert writer.append_indexed(
        torch.ones(2, 128),
        torch.tensor([3, 1]),
        torch.tensor([2.0, -3.0]),
        token_mask=torch.tensor([True, False]),
    ) == (1, 1)
    with pytest.raises(ValueError, match="output_indices must have shape"):
        writer.append_indexed(torch.ones(2, 128), torch.tensor([[3, 1]]))
    assert writer.finish().tensor_index == 7


@pytest.mark.parametrize(
    "kind,optional,limit,required,samples",
    [
        ("dense", False, 16, 520, 1),
        ("dense", True, 520, 529, 1),
        ("indexed", False, 16, 524, 1),
        ("indexed", True, 524, 533, 1),
        ("identity", False, 4, 8, 1),
        ("identity", True, 8, 17, 1),
        ("dense", False, 16, 520 * (1 << 30), 1 << 30),
        ("indexed", False, 16, 524 * (1 << 30), 1 << 30),
        ("identity", False, 16, 8 * (1 << 30), 1 << 30),
    ],
)
def test_writer_admits_batch_before_tensor_value_operations(
    tmp_path, kind, optional, limit, required, samples
):
    torch = pytest.importorskip("torch")
    from torch.utils._python_dispatch import TorchDispatchMode

    class RefuseTensorOperations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            raise AssertionError(f"tensor operation before batch admission: {func}")

    writer = KroneckerCalibrationWriter(
        tmp_path / "evidence",
        tensor_index=0,
        tensor_name="a.weight",
        rows=4 if kind != "dense" else 2,
        columns=128,
        curvature="input-hessian" if kind == "identity" else "guided-fisher",
        source_model_digest="01" * 32,
        activation_cache_digest="02" * 32,
        token_stream_digest="03" * 32,
        damping=0.25,
        indexed_output=kind != "dense",
        max_batch_bytes=limit,
    )
    # Expanded views exercise huge logical batches without allocating their lanes.
    activations = torch.ones(1, 128).expand(samples, 128)
    indices = torch.tensor([1]).expand(samples)
    factors = (
        torch.ones(1, 2).expand(samples, 2)
        if kind == "dense"
        else torch.ones(1).expand(samples)
    )
    options = (
        {"token_weights": torch.ones(1), "token_mask": torch.ones(1, dtype=torch.bool)}
        if optional
        else {}
    )
    before = sorted((tmp_path / "evidence").rglob("*"))
    try:
        with RefuseTensorOperations(), pytest.raises(
            ValueError, match=f"batch requires {required} bytes, limit is {limit}"
        ):
            if kind == "dense":
                writer.append(activations, factors, **options)
            elif kind == "indexed":
                writer.append_indexed(activations, indices, factors, **options)
            else:
                writer.append_indexed_identity(indices, **options)
        assert writer.active
        assert sorted((tmp_path / "evidence").rglob("*")) == before
    finally:
        writer.abort()


@pytest.mark.parametrize("kind,limit", [("dense", 529), ("indexed", 533), ("identity", 17)])
def test_writer_exact_batch_limit_preserves_canonical_record(tmp_path, kind, limit):
    torch = pytest.importorskip("torch")
    receipts = []
    for name, budget in [("exact", limit), ("reference", 4096)]:
        writer = KroneckerCalibrationWriter(
            tmp_path / name,
            tensor_index=0,
            tensor_name="a.weight",
            rows=4 if kind != "dense" else 2,
            columns=128,
            curvature="input-hessian" if kind == "identity" else "guided-fisher",
            source_model_digest="01" * 32,
            activation_cache_digest="02" * 32,
            token_stream_digest="03" * 32,
            damping=0.25,
            indexed_output=kind != "dense",
            max_batch_bytes=budget,
        )
        options = {"token_weights": torch.tensor([2.0]), "token_mask": torch.tensor([True])}
        if kind == "dense":
            writer.append(torch.ones(1, 128), torch.tensor([[2.0, -3.0]]), **options)
        elif kind == "indexed":
            writer.append_indexed(
                torch.ones(1, 128), torch.tensor([1]), torch.tensor([2.0]), **options
            )
        else:
            writer.append_indexed_identity(torch.tensor([1]), **options)
        receipts.append(writer.finish())
    assert receipts[0].record_digest == receipts[1].record_digest
    assert (tmp_path / "exact/000000.s2kf").read_bytes() == (
        tmp_path / "reference/000000.s2kf"
    ).read_bytes()


def test_pytorch_writer_streams_indexed_embedding_identity_input_hessian(tmp_path):
    torch = pytest.importorskip("torch")
    writer = KroneckerCalibrationWriter(
        tmp_path / "indexed-identity",
        tensor_index=9,
        tensor_name="model.embed_tokens.weight",
        rows=4,
        columns=128,
        curvature="input-hessian",
        source_model_digest="01" * 32,
        activation_cache_digest="02" * 32,
        token_stream_digest="03" * 32,
        damping=0.25,
        objective_id="tritium.input-gram@1",
        indexed_output=True,
        max_batch_bytes=4096,
    )
    assert writer.append_indexed_identity(torch.tensor([3, 1])) == (1, 1)
    assert writer.finish().tensor_index == 9
