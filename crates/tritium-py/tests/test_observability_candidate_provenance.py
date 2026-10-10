"""Observability must bind native source/release before executing telemetry."""

import importlib.metadata
import os
from pathlib import Path
import re
import sys

import pytest

pytest.importorskip("torch")

from tritium import _tritium  # noqa: E402
from tritium.torch import qualify_observability  # noqa: E402


@pytest.fixture
def candidate_wheel():
    directory = os.environ.get("TRITIUM_TEST_CANDIDATE_WHEEL_DIR")
    if directory is None:
        pytest.skip("requires the executing candidate wheel directory")
    wheels = list(Path(directory).resolve(strict=True).glob("pytritium-*.whl"))
    assert len(wheels) == 1, "candidate wheel directory must contain exactly one wheel"
    return wheels[0]


def _identity(failure):
    source = _tritium.source_identity().removeprefix("source-git:")
    release = re.sub(r"rc(\d+)$", r"-rc.\1", importlib.metadata.version("pytritium"))
    if failure == "source":
        source = "a" * 40 if source != "a" * 40 else "b" * 40
    elif failure == "release":
        release = "0.0.0"
    return source, release


def _failure_input(tmp_path, monkeypatch, wheel, failure):
    if failure == "origin":
        foreign = tmp_path / "qualify_observability.py"
        foreign.write_text("# code outside candidate wheel\n")
        monkeypatch.setattr(qualify_observability, "__file__", str(foreign))
    elif failure == "wheel":
        wheel = tmp_path / "opaque-candidate.whl"
        wheel.write_bytes(b"not an executing candidate wheel")
    return wheel


def _expected_error(failure):
    return {"source": "native source", "release": "release", "origin": "owned", "wheel": "wheel"}[failure]


def _forbidden_execution(*_args, **_kwargs):
    raise AssertionError("unbound observability candidate entered model/receipt execution")


@pytest.mark.parametrize("failure", ["source", "release", "origin", "wheel"])
def test_observability_producer_rejects_before_model(
    tmp_path, monkeypatch, candidate_wheel, failure,
):
    source, release = _identity(failure)
    candidate_wheel = _failure_input(tmp_path, monkeypatch, candidate_wheel, failure)
    monkeypatch.chdir(tmp_path)
    # Admission-only test, not compiler-free empirical qualification.
    monkeypatch.setattr(qualify_observability.shutil, "which", lambda _name: None)
    monkeypatch.setattr(qualify_observability, "_TinyTiedModel", _forbidden_execution)
    output = tmp_path / "evidence"
    with pytest.raises((ValueError, RuntimeError), match=_expected_error(failure)):
        qualify_observability.run_installed_observability(
            output, wheel_artifact=candidate_wheel, source_revision=source,
            release=release, run_id="negative-observability",
        )
    assert not output.exists()


@pytest.mark.parametrize("failure", ["source", "release", "origin", "wheel"])
def test_observability_cli_replay_rejects_before_receipt(
    tmp_path, monkeypatch, candidate_wheel, failure,
):
    source, release = _identity(failure)
    candidate_wheel = _failure_input(tmp_path, monkeypatch, candidate_wheel, failure)
    monkeypatch.setattr(qualify_observability, "validate_receipt", _forbidden_execution)
    # Runtime adapter versions are irrelevant to candidate preflight admission.
    monkeypatch.setattr(qualify_observability, "_runtime_versions", lambda: {})
    monkeypatch.setattr(sys, "argv", [
        "qualify_observability", "--check-receipt", str(tmp_path / "receipt.json"),
        "--wheel-artifact", str(candidate_wheel), "--source-revision", source,
        "--release", release,
    ])
    with pytest.raises((ValueError, RuntimeError), match=_expected_error(failure)):
        qualify_observability.main()
