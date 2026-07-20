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


def test_native_boundary_rejects_revision_before_source_io(tmp_path):
    with pytest.raises(ValueError, match="pinned Qwen3.6 revision"):
        _tritium.reconcile_qwen36_ptq_masters(
            str(tmp_path / "missing-model"),
            "wrong-revision",
            str(tmp_path / "work"),
            str(tmp_path / "evidence"),
        )


def test_master_receipt_is_not_user_constructible():
    with pytest.raises(TypeError):
        _tritium.Qwen36PtqMasterReceipt()
