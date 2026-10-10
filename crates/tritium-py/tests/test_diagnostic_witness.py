"""Draft numerical/provenance tests, explicitly not release qualification."""

import copy
import importlib.metadata
import math
import os
from pathlib import Path
import re
import runpy

import pytest


W = runpy.run_path(str(
    Path(__file__).parents[1] / "python/tritium/torch/_diagnostic_witness.py"
))


def _synthetic():
    teacher = [[0.0, W["f32"](0.4)]]
    student = [[0.0, 0.0]]
    value = {
        "schema": W["SCHEMA"],
        "binding": {"source_revision": "a" * 40, "release": "1.1.0-rc.2",
                    "run_id": "synthetic-unit-only", "wheel_sha256": "sha256:" + "b" * 64},
        "fixture": "tied-linear-4x2-cpu-f32-v1",
        "teacher_weight": W["fixture_weights"](),
        "student_weight": W["projected_weights"](),
        "inputs": [[1.0] * 4], "teacher_logits": teacher, "student_logits": student,
        "timing": {"clock": "perf_counter_ns", "warmups": 5, "samples": [
            {"start_ns": 100000 * (i + 1), "end_ns": 100000 * (i + 1) + 12000 + i,
             "student_logits": copy.deepcopy(student)} for i in range(31)
        ]},
        "memory": {"method": "linux-proc-self-statm",
                   "scope": "whole-process-after-forward-before-telemetry",
                   "raw_statm": "1000 100 50 10 0 500 0\n", "page_size": 4096},
        "metrics": {"runtime/forward_ms": 0.012015, "memory/resident_bytes": 409600.0,
                    "teacher_kl": W["teacher_kl"](teacher, student)},
    }
    value["witness_id"] = W["witness_id"](value)
    return value


def test_synthetic_arithmetic_only():
    value = _synthetic()
    assert W["validate_witness"](value, expected_binding=value["binding"]) == value["metrics"]
    assert value["metrics"]["teacher_kl"] > 0


@pytest.mark.parametrize("name,fixed", [
    ("runtime/forward_ms", 3.5), ("memory/resident_bytes", 4096.0), ("teacher_kl", 0.125),
])
def test_resigned_copied_counters_rejected(name, fixed):
    value = _synthetic()
    value["metrics"][name] = fixed
    value["witness_id"] = W["witness_id"](value)
    with pytest.raises(ValueError, match="raw evidence"):
        W["validate_witness"](value)


@pytest.mark.parametrize("path,replacement", [
    (("schema",), "tritium.installed-observability.v1"),
    (("fixture",), "full-qwen-model"),
    (("binding", "source_revision"), "unknown"),
    (("binding", "wheel_sha256"), "sha256:" + "z" * 64),
    (("binding", "run_id"), ""),
    (("teacher_weight", 0, 0), -1.0),
    (("student_weight", 0, 0), 0.0),
    (("inputs", 0, 0), 2.0),
    (("teacher_logits", 0, 1), 0.0),
    (("student_logits", 0, 0), 1.0),
    (("teacher_logits",), [[0.0]]),
    (("teacher_logits", 0, 0), True),
    (("teacher_logits", 0, 0), 1e100),
    (("timing", "clock"), "time.time"),
    (("timing", "warmups"), True),
    (("timing", "samples"), []),
    (("timing", "samples", 0, "start_ns"), 0),
    (("timing", "samples", 0, "start_ns"), 2**63),
    (("timing", "samples", 0, "end_ns"), 100000),
    (("timing", "samples", 1, "start_ns"), 100001),
    (("timing", "samples", 0, "student_logits", 0, 0), 1.0),
    (("memory", "method"), "ru_maxrss"),
    (("memory", "scope"), "model-only"),
    (("memory", "page_size"), 4095),
    (("memory", "page_size"), True),
    (("memory", "raw_statm"), "1000 2000 0 0 0 0 0"),
    (("memory", "raw_statm"), "1000 0 0 0 0 0 0"),
    (("memory", "raw_statm"), "1000 100"),
    (("memory", "raw_statm"), "1000 -1 0 0 0 0 0"),
])
def test_resigned_inconsistent_evidence_rejected(path, replacement):
    value = _synthetic()
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    value["witness_id"] = W["witness_id"](value)
    with pytest.raises(ValueError):
        W["validate_witness"](value)


def test_nonfinite_and_identity_rejected():
    for invalid in (float("nan"), float("inf"), -float("inf"), 10**1000):
        value = _synthetic()
        value["metrics"]["teacher_kl"] = invalid
        with pytest.raises(ValueError, match="finite"):
            W["validate_witness"](value)
    value = _synthetic()
    value["witness_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="identity"):
        W["validate_witness"](value)


def test_external_binding_required_when_supplied():
    value = _synthetic()
    expected = dict(value["binding"], source_revision="c" * 40)
    with pytest.raises(ValueError, match="candidate binding"):
        W["validate_witness"](value, expected_binding=expected)


def test_torch_float64_reference_and_actual_cpu_math():
    torch = pytest.importorskip("torch")
    from tritium.torch import _diagnostic_measurements as measurements
    from tritium.torch import TernaryConfig, prepare_qat

    value = _synthetic()
    teacher = torch.tensor(value["teacher_logits"], dtype=torch.float64).log_softmax(-1)
    student = torch.tensor(value["student_logits"], dtype=torch.float64).log_softmax(-1)
    expected = (teacher.exp() * (teacher - student)).sum(-1).mean().item()
    assert math.isclose(expected, value["metrics"]["teacher_kl"], abs_tol=1e-12)
    model = prepare_qat(measurements._Fixture(), TernaryConfig.qat())
    with torch.no_grad():
        model.left.weight.copy_(torch.tensor(W["WEIGHTS"]))
    model(torch.ones(1, 4)).sum().backward()
    before_weight = model.left.weight.detach().clone()
    before_gradient = model.left.weight.grad.clone()
    actual = measurements._measure_fixture(model, value["binding"])
    assert W["validate_witness"](actual) == pytest.approx(actual["metrics"], abs=1e-12)
    assert torch.equal(model.left.weight, before_weight)
    assert torch.equal(model.left.weight.grad, before_gradient)
    assert len(actual["timing"]["samples"]) == 31


def test_candidate_guard_precedes_model_and_measurement(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from tritium.torch import _diagnostic_measurements as measurements

    def reject(**kwargs):
        assert kwargs["executing_files"] == (
            Path(measurements.__file__), Path(measurements.witness.__file__),
        )
        raise ValueError("deliberate unbound candidate")

    def forbidden(*args, **kwargs):
        raise AssertionError("candidate rejection entered measurements")

    monkeypatch.setattr(measurements, "verify_installed_candidate", reject)
    monkeypatch.setattr(measurements, "_Fixture", forbidden)
    with pytest.raises(ValueError, match="unbound candidate"):
        measurements.measure_installed_fixture(
            wheel_artifact=tmp_path / "missing.whl", source_revision="a" * 40,
            release="1.1.0-rc.2", run_id="negative-unit-only",
        )


@pytest.mark.parametrize("failure", ["source", "release", "wheel", "origin"])
def test_actual_candidate_guard_rejects_before_model(monkeypatch, tmp_path, failure):
    pytest.importorskip("torch")
    from tritium import _tritium
    from tritium.torch import _diagnostic_measurements as measurements

    directory = os.environ.get("TRITIUM_TEST_CANDIDATE_WHEEL_DIR")
    if directory is None:
        pytest.skip("actual candidate rejection requires the executing wheel archive")
    wheels = list(Path(directory).resolve(strict=True).glob("pytritium-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    source = _tritium.source_identity().removeprefix("source-git:")
    release = re.sub(r"rc(\d+)$", r"-rc.\1", importlib.metadata.version("pytritium"))
    if failure == "source":
        source = "a" * 40 if source != "a" * 40 else "b" * 40
    elif failure == "release":
        release = "0.0.0"
    elif failure == "wheel":
        wheel = tmp_path / "opaque.whl"
        wheel.write_bytes(b"not a candidate wheel")
    else:
        origin = tmp_path / "foreign.py"
        origin.write_text("# outside the candidate\n")
        monkeypatch.setattr(measurements, "__file__", str(origin))

    def forbidden(*args, **kwargs):
        raise AssertionError("unbound candidate entered draft measurement")

    monkeypatch.setattr(measurements, "_Fixture", forbidden)
    with pytest.raises((ValueError, RuntimeError), match={
        "source": "native source", "release": "release", "wheel": "wheel", "origin": "owned",
    }[failure]):
        measurements.measure_installed_fixture(
            wheel_artifact=wheel, source_revision=source, release=release,
            run_id="negative-candidate-draft",
        )
