from pathlib import Path

import pytest

import tritium.salt as salt
from tritium import _tritium


def test_python_wrapper_preserves_paths_and_explicit_recipe(monkeypatch, tmp_path):
    sentinel = object()
    captured = {}

    def fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(salt, "_reconcile_qwen36_ptq_masters", fake)
    result = salt.reconcile_qwen36_ptq_masters(
        Path("model"),
        revision="revision",
        work_dir=tmp_path / "work",
        evidence_dir=tmp_path / "evidence",
        packing="s34",
        max_evidence_bytes=1234,
    )

    assert result is sentinel
    assert captured["args"] == (
        "model",
        "revision",
        str(tmp_path / "work"),
        str(tmp_path / "evidence"),
    )
    assert captured["kwargs"] == {
        "packing": "s34",
        "max_evidence_bytes": 1234,
    }


def test_source_admission_wrapper_preserves_paths_and_revision(monkeypatch, tmp_path):
    sentinel = object()
    captured = {}

    def fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(salt, "_admit_qwen36_source", fake)
    result = salt.admit_qwen36_source(
        Path("model"),
        revision="revision",
        work_dir=tmp_path / "work",
    )

    assert result is sentinel
    assert captured["args"] == ("model", "revision", str(tmp_path / "work"))
    assert captured["kwargs"] == {}


def test_package_wrapper_preserves_exact_ceilings_and_output(monkeypatch, tmp_path):
    sentinel = object()
    captured = {}

    def fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(salt, "_reconcile_qwen36_ptq_packages", fake)
    result = salt.reconcile_qwen36_ptq_packages(
        Path("model"),
        revision="revision",
        work_dir=tmp_path / "work",
        evidence_dir=tmp_path / "evidence",
        output_dir=tmp_path / "artifact",
        compact_max_bytes=100,
        compact_max_resident_bytes=200,
        near_lossless_max_bytes=300,
        near_lossless_max_resident_bytes=400,
        packing="d2",
        max_evidence_bytes=500,
    )

    assert result is sentinel
    assert captured["args"] == (
        "model",
        "revision",
        str(tmp_path / "work"),
        str(tmp_path / "evidence"),
        str(tmp_path / "artifact"),
    )
    assert captured["kwargs"] == {
        "compact_max_bytes": 100,
        "compact_max_resident_bytes": 200,
        "near_lossless_max_bytes": 300,
        "near_lossless_max_resident_bytes": 400,
        "packing": "d2",
        "max_evidence_bytes": 500,
    }


def test_native_boundary_rejects_revision_before_source_io(tmp_path):
    with pytest.raises(ValueError, match="pinned Qwen3.6 revision"):
        _tritium.admit_qwen36_source(
            str(tmp_path / "missing-model"),
            "wrong-revision",
            str(tmp_path / "work"),
        )

    with pytest.raises(ValueError, match="pinned Qwen3.6 revision"):
        _tritium.reconcile_qwen36_ptq_masters(
            str(tmp_path / "missing-model"),
            "wrong-revision",
            str(tmp_path / "work"),
            str(tmp_path / "evidence"),
        )

    with pytest.raises(ValueError, match="pinned Qwen3.6 revision"):
        _tritium.reconcile_qwen36_ptq_packages(
            str(tmp_path / "missing-model"),
            "wrong-revision",
            str(tmp_path / "work"),
            str(tmp_path / "evidence"),
            str(tmp_path / "artifact"),
            compact_max_bytes=1,
            compact_max_resident_bytes=1,
            near_lossless_max_bytes=1,
            near_lossless_max_resident_bytes=1,
        )


def test_master_receipt_is_not_user_constructible():
    with pytest.raises(TypeError):
        _tritium.Qwen36PtqMasterReceipt()
    with pytest.raises(TypeError):
        _tritium.Qwen36PtqPackageReceipt()
    with pytest.raises(TypeError):
        _tritium.Qwen36SourceAdmissionReceipt()
