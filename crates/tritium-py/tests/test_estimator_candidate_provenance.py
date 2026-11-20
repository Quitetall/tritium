"""Public estimator entry points must admit candidates before model execution."""

import importlib.metadata
import os
from pathlib import Path
import re
import sys

import pytest

pytest.importorskip("torch")

from tritium import _tritium  # noqa: E402
from tritium.torch import qualify_estimators as worker  # noqa: E402


@pytest.fixture
def candidate_wheel():
    directory = os.environ.get("TRITIUM_TEST_CANDIDATE_WHEEL_DIR")
    if directory is None:
        pytest.skip("requires the executing candidate wheel directory")
    wheels = list(Path(directory).resolve(strict=True).glob("pytritium-*.whl"))
    assert len(wheels) == 1, "candidate directory must contain exactly one wheel"
    return wheels[0]


def _forbidden_execution(*_args, **_kwargs):
    raise AssertionError("unbound estimator candidate entered model execution")


@pytest.mark.parametrize("entry", ["run", "cli"])
@pytest.mark.parametrize("failure", ["source", "release", "origin", "wheel"])
def test_estimator_admission_precedes_model(
    tmp_path, monkeypatch, candidate_wheel, entry, failure,
):
    source = _tritium.source_identity().removeprefix("source-git:")
    release = re.sub(r"rc(\d+)$", r"-rc.\1", importlib.metadata.version("pytritium"))
    if failure == "source":
        source = "a" * 40 if source != "a" * 40 else "b" * 40
    elif failure == "release":
        release = "0.0.0"
    elif failure == "origin":
        foreign = tmp_path / "qualify_estimators.py"
        foreign.write_text("# outside executing candidate\n")
        monkeypatch.setattr(worker, "__file__", str(foreign))
    else:
        candidate_wheel = tmp_path / "opaque-candidate.whl"
        candidate_wheel.write_bytes(b"not an executing candidate wheel")
    monkeypatch.setattr(worker, "_case", _forbidden_execution)
    output = tmp_path / "trace.json"
    expected = {"source": "native source", "release": "release", "origin": "owned", "wheel": "wheel"}[failure]
    with pytest.raises((ValueError, RuntimeError), match=expected):
        if entry == "run":
            worker.run(wheel=candidate_wheel, source_revision=source,
                       release=release, run_id="negative-estimator")
        else:
            monkeypatch.setattr(sys, "argv", [
                "qualify_estimators", "--wheel", str(candidate_wheel),
                "--source-revision", source, "--release", release,
                "--run-id", "negative-estimator", "--output", str(output),
            ])
            worker.main()
    assert not output.exists()
