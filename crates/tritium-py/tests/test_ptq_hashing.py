"""Source/data identities must not clone whole strided weight masters."""
from __future__ import annotations

import hashlib
import json
import os
import struct

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tritium.torch import TernaryConfig, calibrate, convert, prepare
from tritium.torch import ptq

CHUNK_BYTES = 1024 * 1024


class BoundedClone(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.clone.default:
            size = args[0].numel() * args[0].element_size()
            assert size <= CHUNK_BYTES, f"whole tensor clone during hashing: {size} bytes"
        return func(*args, **(kwargs or {}))


def legacy_digest(value, tag="value"):
    digest = hashlib.sha256()

    def field(name, payload):
        encoded = name.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)

    field(f"{tag}:dtype", str(value.dtype).encode("ascii"))
    field(f"{tag}:shape", json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    flat = value.detach().contiguous().view(-1)
    elements = max(1, CHUNK_BYTES // flat.element_size())
    for start in range(0, flat.numel(), elements):
        field(f"{tag}:chunk", flat[start:start + elements].view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def digest(value, tag="value"):
    result = hashlib.sha256()
    with BoundedClone():
        ptq._hash_tensor(result, tag, value)
    return result.hexdigest()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64, torch.bool])
def test_strided_source_hash_is_bounded_and_bit_exact(dtype):
    value = torch.arange(1024 * 513, dtype=torch.int64).reshape(1024, 513).to(dtype).t()
    assert not value.is_contiguous()
    assert value.numel() * value.element_size() > CHUNK_BYTES or dtype == torch.bool
    expected = legacy_digest(value)
    # Same shape/values/dtype, only stride changes. This rules out chunk-size
    # and source-preparation explanations for the guarded clone failure.
    assert digest(value.contiguous()) == expected
    assert digest(value) == expected


def test_public_calibration_does_not_clone_a_whole_transposed_master(tmp_path, monkeypatch):
    model = torch.nn.Linear(1024, 513, bias=False)
    model.weight = torch.nn.Parameter(torch.arange(1024 * 513, dtype=torch.float32).reshape(1024, 513).t() / 1000000)
    prepared = prepare(model, TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)), inplace=True)
    assert not prepared.model.weight.is_contiguous()
    expected = "sha256:" + legacy_digest(prepared.model.weight, "state.weight")
    original = ptq._hash_tensor

    def guarded(*args, **kwargs):
        with BoundedClone():
            return original(*args, **kwargs)

    monkeypatch.setattr(ptq, "_hash_tensor", guarded)
    receipt = calibrate(prepared, [torch.ones(1, 1024)], evidence_dir=tmp_path / "evidence")
    assert receipt.source_model_digest == expected
    assert not prepared.model.weight.is_contiguous()
    artifact = convert(prepared, receipt, work_dir=tmp_path / "conversion", max_working_bytes=CHUNK_BYTES)
    assert artifact.source_model_digest == expected
    resumed = convert(prepared, receipt, work_dir=tmp_path / "conversion", max_working_bytes=CHUNK_BYTES)
    assert resumed.artifact_id == artifact.artifact_id


@pytest.mark.parametrize("kind", ["permute", "recursive", "wide-row", "stride-1d", "expand", "channels-last", "empty", "scalar", "offset"])
def test_logical_hash_preserves_chunk_boundaries_across_strided_slabs(kind):
    if kind == "permute":
        value = torch.arange(3 * 17 * 513 * 65, dtype=torch.float32).reshape(3, 17, 513, 65).permute(2, 0, 1, 3)
    elif kind == "recursive":
        value = torch.arange(2 * 3 * 140000, dtype=torch.float32).reshape(2, 3, 140000).permute(1, 0, 2)
    elif kind == "wide-row":
        value = torch.arange(2 * 600000, dtype=torch.float32).reshape(2, 600000)[:, ::2]
    elif kind == "stride-1d":
        value = torch.arange(1400002, dtype=torch.float32)[1::2]
    elif kind == "expand":
        value = torch.arange(11 * 43, dtype=torch.float32).reshape(1, 11, 43).expand(800, 11, 43)
    elif kind == "channels-last":
        value = torch.arange(3 * 7 * 111 * 257, dtype=torch.float32).reshape(3, 7, 111, 257).contiguous(memory_format=torch.channels_last)
    elif kind == "empty":
        value = torch.empty(3, 0, 5).permute(2, 0, 1)
    elif kind == "scalar":
        value = torch.tensor(-0.0, dtype=torch.float64)
    else:
        value = torch.arange(600123, dtype=torch.float64)[17:600017:2]
    expected = legacy_digest(value)
    assert digest(value) == expected


def test_hash_preserves_raw_float_bits_and_noncontiguous_conjugate_values():
    bits = torch.tensor([0, 0x80000000, 0x7FC00001, 0x7FC03001, 0x7F800000, 0xFF800000, 0x00000001], dtype=torch.int64).to(torch.int32)
    value = bits.repeat(80000).reshape(1120, 500).view(torch.float32).t()
    assert digest(value) == legacy_digest(value)
    complex_value = torch.complex(torch.arange(600003, dtype=torch.float32), torch.ones(600003)).reshape(3, 200001).t().conj()
    assert digest(complex_value) == legacy_digest(complex_value)
    recursive = torch.complex(torch.arange(840000, dtype=torch.float32), torch.ones(840000)).reshape(2, 3, 140000).permute(1, 0, 2).conj()
    assert digest(recursive) == legacy_digest(recursive)


def test_physical_cuda_source_hash_is_bounded_and_matches_cpu():
    if os.environ.get("TRITIUM_REQUIRE_HASH_CUDA") != "1":
        pytest.skip("requires separately invoked physical CUDA source-hash gate")
    assert torch.cuda.is_available(), "required physical CUDA hash test cannot skip"
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        cpu = torch.arange(1024 * 513, dtype=torch.int64).reshape(1024, 513).to(dtype).t()
        expected = legacy_digest(cpu)
        value = cpu.to("cuda", memory_format=torch.preserve_format)
        assert not value.is_contiguous()
        assert digest(value) == expected
        del value
    torch.cuda.synchronize()
