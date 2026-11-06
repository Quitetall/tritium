"""Shared file conversion; optional cross-frontend fixtures come from real Rust types.

Run the calibrated/Gram corpus with the ignored CLI integration test
``actual_python_file_conversion_matches_core_and_cli`` after building the abi3
extension. Ordinary pytest still executes direct conversion and rejection cases.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import struct

import pytest
import tritium


def _shard(path, name, dtype, shape, payload):
    header = json.dumps({name: {"dtype": dtype, "shape": shape,
                               "data_offsets": [0, len(payload)]}}).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def _direct_fixture(root):
    # Caller constructs an existing QuantPlan, not a Python-owned wire schema.
    layout = dict(rows=1, cols=32, tile=256, group=32, max_planes=2,
                  allocation="Uniform", codec="B3", basis="Identity", transport="Raw",
                  law=dict(anchor="Group", relation="Free", precision="F32"))
    fit = dict(layout=layout, profile={"JointIndependentGroups": dict(
        max_iterations=4, ridge=1e-8, em_restarts=1, ridge_condition_limit=1e6,
        softened_relay=False, modulated_relay=False)},
        levels=[dict(name="compact", planes=1), dict(name="full", planes=2)],
        limits=dict(max_weights=32, max_metric_entries=1024, max_output_trits=96))
    plan = dict(architecture="python-file-fixture", tensors=[
        dict(source=dict(name="norm", shape=[2], precision="F32"), action="KeepDense"),
        dict(source=dict(name="weight", shape=[1, 32], precision="F32"),
             action={"Fit": dict(plan=fit, metric="Identity")})],
        assets=["config.json", "tokenizer.json"], levels=["compact", "full"],
        limits=dict(max_source_bytes=1024, max_asset_bytes=4,
                    max_retained_blob_bytes=4096, max_package_bytes=100_000))
    _shard(root / "norm.safetensors", "norm", "F32", [2], struct.pack("<ff", -0.0, 1.0))
    _shard(root / "weight.safetensors", "weight", "F32", [1, 32],
           struct.pack("<32f", *((i - 16) / 16 for i in range(32))))
    for name in plan["assets"]:
        (root / name).write_bytes(b"{}")
    return plan


def _convert(root, plan, destination="python", **overrides):
    options = dict(asset_paths={name: str(root / name) for name in plan["assets"]})
    if root.name.endswith("-gram"):
        options["input_grams"] = {"weight": str(root / "gram.bin")}
    if root.name.endswith("-calibrated"):
        options.update(calibrated=True, calibration_caches={"weight": str(root / "cache.trit")})
    options.update(overrides)
    return tritium.convert_model_plan(
        json.dumps(plan), [str(root / f"{name}.safetensors") for name in ("norm", "weight")],
        str(root / destination), "filesystem-conversion", **options)


def test_direct_preservation_readonly_reopen_and_no_clobber(tmp_path):
    plan = _direct_fixture(tmp_path)
    result = _convert(tmp_path, plan)
    result.verify()
    for level in ("compact", "full"):
        assert result.preserved_tensor(level, "norm") == ("F32", [2], struct.pack("<ff", -0.0, 1.0))
    assert set(result.model_ids) == {"compact", "full"}
    assert len(result.package_id) == len(result.source_id) == 64
    assert result.run_id == "filesystem-conversion"
    assert result.publication_warnings == []
    with pytest.raises(AttributeError):
        result.package_id = "forged"
    with pytest.raises(TypeError):
        tritium.NativeModelConversion()
    returned = result.model_ids
    returned.clear()
    assert set(result.model_ids) == {"compact", "full"}
    for name, bound in [("norm", 1), ("norm", 0), ("weight", 65536)]:
        with pytest.raises(ValueError):
            result.preserved_tensor("full", name, max_bytes=bound)
    before = (tmp_path / "python/model.trit").read_bytes()
    with pytest.raises(ValueError):
        _convert(tmp_path, plan)
    assert (tmp_path / "python/model.trit").read_bytes() == before
    package = tmp_path / "python/model.trit"
    package.write_bytes(b"bad" + before[3:])
    with pytest.raises(ValueError):
        result.verify()
    package.write_bytes(before)
    evidence = tmp_path / "python/evidence.jsonl"
    evidence.write_bytes(evidence.read_bytes() + b"invalid\n")
    with pytest.raises(ValueError):
        result.preserved_tensor("full", "norm")


@pytest.mark.parametrize("variant", ["asset", "cache", "gram", "plan_limit", "source", "unknown_plan"])
def test_invalid_direct_inputs_leave_no_output(tmp_path, variant):
    plan = _direct_fixture(tmp_path)
    kwargs = {}
    if variant == "asset": kwargs["asset_paths"] = {}
    elif variant == "cache": kwargs["calibration_caches"] = {"weight": "missing"}
    elif variant == "gram": kwargs["input_grams"] = {"weight": "missing"}
    elif variant == "plan_limit": kwargs["max_plan_bytes"] = 1
    elif variant == "source": _shard(tmp_path / "norm.safetensors", "norm", "I32", [2], b"\0" * 8)
    else: plan["unrecognized"] = "reject"
    with pytest.raises(ValueError):
        _convert(tmp_path, plan, **kwargs)
    assert not (tmp_path / "python").exists()


def test_relative_destination_stays_bound_after_cwd_change(tmp_path, monkeypatch):
    plan = _direct_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = tritium.convert_model_plan(
        json.dumps(plan), [str(tmp_path / f"{name}.safetensors") for name in ("norm", "weight")],
        "relative-output", "relative-run",
        asset_paths={name: str(tmp_path / name) for name in plan["assets"]})
    assert Path(result.output_dir).is_absolute()
    monkeypatch.chdir(tmp_path.parent)
    result.verify()


def test_valid_foreign_package_and_log_do_not_rebind_result(tmp_path):
    plan = _direct_fixture(tmp_path)
    result = _convert(tmp_path, plan)
    plan["architecture"] = "foreign-fixture"
    other = _convert(tmp_path, plan, "other")
    other.verify()
    for filename in ("model.trit", "evidence.jsonl"):
        (tmp_path / "python" / filename).write_bytes((tmp_path / "other" / filename).read_bytes())
    with pytest.raises(ValueError):
        result.verify()


@pytest.mark.parametrize("precision", ["F16", "Bf16", "F32"])
@pytest.mark.parametrize("mode", ["direct", "gram", "calibrated"])
def test_rust_fixture_core_cli_python_exact_parity(precision, mode):
    location = os.environ.get("TRITIUM_MODEL_FIXTURES")
    if not location:
        pytest.skip("real Rust-generated cache/Gram fixture corpus not supplied")
    root = Path(location) / f"{precision}-{mode}"
    plan = json.loads((root / "plan.json").read_text())
    result = _convert(root, plan)
    result.verify()
    for filename in ("model.trit", "evidence.jsonl"):
        assert (root / "python" / filename).read_bytes() == (root / "cli" / filename).read_bytes()
    assert result.preserved_tensor("compact", "norm") == (precision, [2], (root / "norm.raw").read_bytes())
    if mode == "calibrated":
        with pytest.raises(ValueError):
            _convert(root, plan, "missing-cache", calibration_caches={})
        cache = root / "cache.trit"
        original = cache.read_bytes()
        cache.write_bytes(b"corrupt")
        with pytest.raises(ValueError):
            _convert(root, plan, "corrupt-cache")
        assert not (root / "corrupt-cache").exists()
        cache.write_bytes(original)
        shard = root / "weight.safetensors"
        payload = bytearray(shard.read_bytes())
        start = 8 + struct.unpack("<Q", payload[:8])[0]
        payload[start] ^= 1
        shard.write_bytes(payload)
        with pytest.raises(ValueError):
            _convert(root, plan, "foreign-cache-source")
        assert not (root / "foreign-cache-source").exists()
    elif mode == "gram":
        gram = root / "gram.bin"
        original = gram.read_bytes()
        gram.write_bytes(struct.pack("<d", 9.0) + original[8:])
        with pytest.raises(ValueError):
            _convert(root, plan, "changed-gram")
        assert not (root / "changed-gram").exists()
        gram.write_bytes(original[:-8])
        with pytest.raises(ValueError):
            _convert(root, plan, "short-gram")
        assert not (root / "short-gram").exists()
    wrong = copy.deepcopy(plan)
    wrong["tensors"][0]["source"]["shape"] = [3]
    with pytest.raises(ValueError):
        _convert(root, wrong, "wrong-inventory")
    assert not (root / "wrong-inventory").exists()
