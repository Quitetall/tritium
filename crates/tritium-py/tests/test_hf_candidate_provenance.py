"""Actual HF qualification call sites must reject unbound inputs before training."""

import base64
import hashlib
import importlib.metadata
import re
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tritium import _tritium  # noqa: E402
from tritium.torch import hf_export_lifecycle, hf_lifecycle  # noqa: E402


def _installed_release():
    return re.sub(r"rc(\d+)$", r"-rc.\1", importlib.metadata.version("pytritium"))


@pytest.mark.parametrize("module", [hf_lifecycle, hf_export_lifecycle])
@pytest.mark.parametrize("failure", ["source", "wheel", "release"])
def test_candidate_mismatch_rejected_before_model_entry(tmp_path, monkeypatch, module, failure):
    wheel = tmp_path / "pytritium-1.1.0rc2-cp39-abi3-linux_x86_64.whl"
    wheel.write_bytes(b"opaque bytes are not an executing candidate wheel")
    identity = _tritium.source_identity()
    assert identity.startswith("source-git:") and len(identity) == 51
    revision = identity.removeprefix("source-git:")
    if failure == "source":
        revision = "a" * 40 if revision != "a" * 40 else "b" * 40

    def forbidden_model_entry():
        raise AssertionError("qualification entered model creation with unbound candidate inputs")

    monkeypatch.setattr(module, "_tiny_llama", forbidden_model_entry)
    output = tmp_path / "evidence"
    run = module.run_hf_lifecycle if module is hf_lifecycle else module.qualify_hf_export
    with pytest.raises((ValueError, RuntimeError), match="source|wheel|candidate|release"):
        run(
            output, wheel_artifact=wheel, source_revision=revision,
            release="0.0.0" if failure == "release" else _installed_release(),
            run_id="negative-provenance", seed=97,
        )
    assert not output.exists(), "unbound candidate must not publish evidence"


@pytest.mark.parametrize("module", [hf_lifecycle, hf_export_lifecycle])
@pytest.mark.parametrize("failure", ["source", "wheel"])
def test_installed_replay_rejects_unbound_candidate(tmp_path, monkeypatch, module, failure):
    # Make a portable, byte-valid receipt using an explicitly mocked producer
    # guard. Restore the real guard before exercising the installed validator.
    # No fixture produced by this test is release qualification.
    wheel = tmp_path / "opaque-candidate.whl"
    wheel.write_bytes(b"not an installed candidate wheel")
    revision = _tritium.source_identity().removeprefix("source-git:")
    if failure == "source":
        revision = "a" * 40 if revision != "a" * 40 else "b" * 40
    output = tmp_path / "evidence"
    with monkeypatch.context() as producer:
        producer.setattr(
            module, "_installed_distribution",
            lambda **_kwargs: (
                importlib.metadata.version("pytritium"),
                Path(hf_lifecycle.tritium.__file__).resolve(),
            ),
        )
        run = module.run_hf_lifecycle if module is hf_lifecycle else module.qualify_hf_export
        receipt = run(
            output, wheel_artifact=wheel, source_revision=revision,
            release=_installed_release(), run_id="negative-replay-provenance",
        )
        if module is hf_lifecycle:
            hf_lifecycle._write_receipt(output, receipt)

    def forbidden_replay(*_args, **_kwargs):
        raise AssertionError("installed replay entered model loading with unbound inputs")

    if module is hf_lifecycle:
        monkeypatch.setattr(module.transformers.AutoModelForCausalLM, "from_pretrained", forbidden_replay)
        validate = module.validate_hf_lifecycle_receipt
    else:
        monkeypatch.setattr(module, "load_qat_hard", forbidden_replay)
        validate = module.validate_hf_export_receipt
    with pytest.raises((ValueError, RuntimeError), match="source|wheel|candidate"):
        validate(output / "receipt.json", expected_wheel=wheel)


@pytest.fixture
def candidate_installation(tmp_path, monkeypatch):
    """Synthetic real ZIP/RECORD files, not model or release qualification."""
    payloads = {
        "tritium/__init__.py": b"package",
        "tritium/_tritium.abi3.so": b"native fixture",
        "tritium/torch/hf_lifecycle.py": b"qualification fixture",
        "tritium/torch/_telemetry_binary.py": b"telemetry",
        "tritium/torch/_wheel_identity.py": b"wheel identity",
        "tritium/torch/qualify_observability.py": b"observability",
        "tritium/torch/observability_receipt.py": b"receipt",
        "pytritium-1.1.0rc2.dist-info/METADATA": b"Name: pytritium\nVersion: 1.1.0rc2\n",
    }
    entries = []
    rows = []
    for name, payload in payloads.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        item = importlib.metadata.PackagePath(name)
        item.hash = importlib.metadata.FileHash("sha256=" + digest)
        item.size = len(payload)
        entries.append(item)
        rows.append(f"{name},sha256={digest},{len(payload)}\n")
    record = "pytritium-1.1.0rc2.dist-info/RECORD"
    payloads[record] = ("".join(rows) + f"{record},,\n").encode()
    item = importlib.metadata.PackagePath(record)
    item.hash = None
    item.size = None
    entries.append(item)
    root = tmp_path / "installed"
    for name, payload in payloads.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    wheel = tmp_path / "candidate.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    distribution = SimpleNamespace(
        version="1.1.0rc2", files=entries,
        locate_file=lambda item: root / str(item),
    )
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _name: distribution)
    monkeypatch.setattr(hf_lifecycle.tritium, "__file__", str(root / "tritium/__init__.py"))
    monkeypatch.setattr(_tritium, "__file__", str(root / "tritium/_tritium.abi3.so"))
    monkeypatch.setattr(_tritium, "source_identity", lambda: "source-git:" + "a" * 40)
    monkeypatch.setattr(hf_lifecycle, "__file__", str(root / "tritium/torch/hf_lifecycle.py"))
    return root, wheel, distribution


def _verify_candidate(wheel):
    return hf_lifecycle._installed_distribution(
        wheel_artifact=wheel, source_revision="a" * 40, release="1.1.0-rc.2"
    )


def test_matching_installed_candidate_is_accepted(candidate_installation):
    root, wheel, _ = candidate_installation
    assert _verify_candidate(wheel) == ("1.1.0rc2", root / "tritium/__init__.py")
    assert _verify_candidate(None) == ("1.1.0rc2", root / "tritium/__init__.py")


@pytest.mark.parametrize("rehashed_record", [False, True])
def test_candidate_rejects_changed_installed_bytes(candidate_installation, rehashed_record):
    root, wheel, distribution = candidate_installation
    target = root / "tritium/torch/hf_lifecycle.py"
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 1
    target.write_bytes(payload)
    if rehashed_record:
        item = next(item for item in distribution.files if str(item).endswith("hf_lifecycle.py"))
        encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        item.hash = importlib.metadata.FileHash("sha256=" + encoded)
    with pytest.raises(ValueError, match="differs from executing"):
        _verify_candidate(wheel)


def test_optional_wheel_replay_still_verifies_record(candidate_installation):
    root, _, _ = candidate_installation
    target = root / "tritium/torch/hf_lifecycle.py"
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="RECORD identity differs"):
        _verify_candidate(None)


@pytest.mark.parametrize("identity", ["source-git:" + "b" * 40, "source-dirty", "unverified"])
def test_candidate_rejects_unbound_native_source(candidate_installation, monkeypatch, identity):
    _, wheel, _ = candidate_installation
    monkeypatch.setattr(_tritium, "source_identity", lambda: identity)
    with pytest.raises(ValueError, match="native source"):
        _verify_candidate(wheel)


def test_candidate_rejects_missing_and_duplicate_record_members(candidate_installation):
    _, wheel, distribution = candidate_installation
    distribution.files.append(distribution.files[0])
    with pytest.raises(ValueError, match="duplicate"):
        _verify_candidate(wheel)
    distribution.files.pop()
    distribution.files = [item for item in distribution.files if not str(item).endswith("_telemetry_binary.py")]
    with pytest.raises(ValueError, match="absent from installed RECORD"):
        _verify_candidate(wheel)


def test_candidate_rejects_symlink_and_unowned_origin(candidate_installation, monkeypatch):
    root, wheel, _ = candidate_installation
    target = root / "tritium/torch/hf_lifecycle.py"
    alternate = root / "alternate.py"
    target.rename(alternate)
    target.symlink_to(alternate)
    with pytest.raises(RuntimeError, match="not owned"):
        _verify_candidate(wheel)
    target.unlink()
    alternate.rename(target)
    monkeypatch.setattr(hf_lifecycle.tritium, "__file__", str(root / "outside.py"))
    (root / "outside.py").write_bytes(b"package")
    with pytest.raises(RuntimeError, match="not owned"):
        _verify_candidate(wheel)
