"""Installed tutorial admission must bind candidates before model execution."""

import importlib.metadata
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import tritium  # noqa: E402
from tritium import _tritium  # noqa: E402
from tritium.torch import tutorial_qat  # noqa: E402


def _identity(failure):
    revision = _tritium.source_identity().removeprefix("source-git:")
    release = re.sub(r"rc(\d+)$", r"-rc.\1", importlib.metadata.version("pytritium"))
    if failure == "source":
        revision = "a" * 40 if revision != "a" * 40 else "b" * 40
    if failure == "release":
        release = "0.0.0"
    return revision, release


def _forbidden_model(*_args, **_kwargs):
    raise AssertionError("unbound tutorial candidate entered model execution")


def _expected_error(failure):
    return {"source": "native source", "wheel": "wheel", "release": "release"}[failure]


def test_tutorial_import_is_transformers_optional(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", "-c", """
import importlib.abc
import sys
class NoTransformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] == "transformers":
            raise ImportError("Transformers deliberately absent")
        return None
sys.meta_path.insert(0, NoTransformers())
from tritium.torch.tutorial_qat import run_installed_qat_tutorial
from tritium.torch._installed_candidate import verify_installed_candidate
assert "transformers" not in sys.modules
assert callable(run_installed_qat_tutorial) and callable(verify_installed_candidate)
"""],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("failure", ["source", "wheel", "release"])
def test_tutorial_rejects_candidate_before_model(tmp_path, monkeypatch, failure):
    wheel = tmp_path / "opaque-candidate.whl"
    wheel.write_bytes(b"not an executing candidate wheel")
    revision, release = _identity(failure)
    monkeypatch.setattr(tutorial_qat, "_TinyTiedModel", _forbidden_model)
    output = tmp_path / "evidence"
    with pytest.raises((ValueError, RuntimeError), match=_expected_error(failure)):
        tutorial_qat.run_installed_qat_tutorial(
            output, device_name="cpu", wheel_artifact=wheel,
            source_revision=revision, release=release, run_id="negative-tutorial",
        )
    assert not output.exists()


@pytest.mark.parametrize("failure", ["source", "wheel"])
def test_tutorial_replay_rejects_unbound_candidate(tmp_path, monkeypatch, failure):
    # Deliberately mock only fixture production's admission, then exercise the
    # real installed validator. Synthetic fixture receipts are not qualified.
    wheel = tmp_path / "opaque-candidate.whl"
    wheel.write_bytes(b"not an executing candidate wheel")
    revision, release = _identity(failure)
    output = tmp_path / "evidence"
    with monkeypatch.context() as producer:
        producer.setattr(
            tutorial_qat, "_installed_distribution",
            lambda **_kwargs: (
                importlib.metadata.version("pytritium"),
                Path(tritium.__file__).resolve(),
            ),
        )
        receipt = tutorial_qat.run_installed_qat_tutorial(
            output, device_name="cpu", wheel_artifact=wheel,
            source_revision=revision, release=release, run_id="negative-replay",
        )
        tutorial_qat._write_receipt(output, receipt)
    monkeypatch.setattr(tutorial_qat, "load_qat_hard", _forbidden_model)
    with pytest.raises((ValueError, RuntimeError), match=_expected_error(failure)):
        tutorial_qat.validate_tutorial_receipt(
            output / "receipt.json", expected_device="cpu", expected_wheel=wheel,
        )


@pytest.mark.parametrize("failure", ["source", "wheel", "release"])
def test_smollm2_wrapper_rejects_candidate_before_demo(tmp_path, monkeypatch, failure):
    script = Path(__file__).resolve().parents[3] / "scripts/qualify-smollm2-release-tutorial.py"
    spec = importlib.util.spec_from_file_location("smollm2_candidate_admission", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wheel = tmp_path / "opaque-candidate.whl"
    wheel.write_bytes(b"not an executing candidate wheel")
    revision, release = _identity(failure)
    import tritium.torch as tt

    monkeypatch.setattr(tt, "run_smollm2_release_demo", _forbidden_model)
    output = tmp_path / "evidence"
    with pytest.raises((ValueError, RuntimeError), match=_expected_error(failure)):
        module.run(SimpleNamespace(
            wheel=wheel, output_dir=output, source_revision=revision,
            release=release, run_id="negative-smollm2",
        ))
    assert not output.exists()
