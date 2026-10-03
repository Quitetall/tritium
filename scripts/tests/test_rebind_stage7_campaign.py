from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "rebind-stage7-campaign.py"
SPEC = importlib.util.spec_from_file_location("rebind_stage7_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def campaign(revision: str = "a" * 40, evidence: list[dict] | None = None) -> dict:
    return {
        "schema": MODULE.CAMPAIGN_SCHEMA,
        "release": "1.1.0-rc.1",
        "source_revision": revision,
        "run_id": "old-run",
        "model": {},
        "smoke_model": {},
        "smoke_provenance": {},
        "provenance": {},
        "thresholds": {},
        "recipe_count": 1404,
        "recipe_grid_id": "sha256:" + "1" * 64,
        "token_evidence_pack": {
            "path": "token-evidence.json", "bytes": 1, "sha256": "0" * 64,
        },
        "evidence": [] if evidence is None else evidence,
    }


def write(path: Path, value: dict) -> None:
    path.write_bytes(MODULE.canonical(value) + b"\n")


def setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    evidence = []
    for index, kind in enumerate(("smoke", "native-kernels", "hestia-gate-c")):
        filename = f"{kind}.json"
        payload = MODULE.canonical({"kind": kind, "source_revision": "b" * 40})
        (tmp_path / filename).write_bytes(payload)
        evidence.append({
            "kind": kind,
            "path": filename,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    token = b"tokens"
    (tmp_path / "token-evidence.json").write_bytes(token)
    value = campaign(evidence=evidence)
    value["token_evidence_pack"] = {
        "path": "token-evidence.json",
        "bytes": len(token),
        "sha256": hashlib.sha256(token).hexdigest(),
    }
    template = tmp_path / "campaign.json"
    write(template, value)
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(MODULE, "_source_identity", lambda _: "b" * 40)
    return template, source


def test_rebind_updates_only_top_level_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    template, source = setup(monkeypatch, tmp_path)
    output = tmp_path / "out" / "campaign.json"
    result = MODULE.rebind(template, source_root=source, run_id="current-run", output=output)
    value = json.loads(output.read_text())
    assert result["source_revision"] == "b" * 40
    assert value["source_revision"] == "b" * 40
    assert value["run_id"] == "current-run"
    assert value["recipe_grid_id"] == campaign()["recipe_grid_id"]
    assert output.read_bytes() == MODULE.canonical(value) + b"\n"


def test_rebind_builds_prerequisite_evidence_from_receipt_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    template, source = setup(monkeypatch, tmp_path)
    value = campaign(evidence=[])
    value["token_evidence_pack"] = json.loads(template.read_text())["token_evidence_pack"]
    write(template, value)
    receipt_paths = tuple(tmp_path / f"{kind}.json" for kind in (
        "smoke", "native-kernels", "hestia-gate-c",
    ))

    output = tmp_path / "out" / "campaign.json"
    MODULE.rebind(
        template,
        source_root=source,
        run_id="current-run",
        output=output,
        smoke_receipt=receipt_paths[0],
        native_kernels_receipt=receipt_paths[1],
        hestia_gate_c_receipt=receipt_paths[2],
    )

    rebound = json.loads(output.read_text())
    assert [record["kind"] for record in rebound["evidence"]] == [
        "smoke", "native-kernels", "hestia-gate-c",
    ]
    for record, receipt_path in zip(rebound["evidence"], receipt_paths, strict=True):
        assert record["path"] == receipt_path.name
        assert record["bytes"] == receipt_path.stat().st_size
        assert record["sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()


def test_rebind_requires_all_receipt_paths_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    template, source = setup(monkeypatch, tmp_path)
    with pytest.raises(MODULE.RebindError, match="all three"):
        MODULE.rebind(
            template,
            source_root=source,
            run_id="new-run",
            output=tmp_path / "out.json",
            smoke_receipt=tmp_path / "smoke.json",
        )


def test_rebind_receipt_paths_must_stay_inside_evidence_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    template, source = setup(monkeypatch, tmp_path)
    outside = tmp_path.parent / "outside-smoke.json"
    outside.write_text('{"source_revision":"' + "b" * 40 + '"}')
    with pytest.raises(MODULE.RebindError, match="inside the campaign evidence directory"):
        MODULE.rebind(
            template,
            source_root=source,
            run_id="new-run",
            output=tmp_path / "out.json",
            smoke_receipt=outside,
            native_kernels_receipt=tmp_path / "native-kernels.json",
            hestia_gate_c_receipt=tmp_path / "hestia-gate-c.json",
        )


def test_rebind_rejects_nested_stale_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    template, source = setup(monkeypatch, tmp_path)
    value = campaign()
    value["provenance"] = {"source_revision": "a" * 40}
    write(template, value)
    with pytest.raises(MODULE.RebindError, match="outside top-level"):
        MODULE.rebind(template, source_root=source, run_id="new-run", output=tmp_path / "out.json")


def test_rebind_rejects_incomplete_prerequisites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    template, source = setup(monkeypatch, tmp_path)
    value = campaign()
    token = b"tokens"
    value["token_evidence_pack"] = {
        "path": "token-evidence.json",
        "bytes": len(token),
        "sha256": hashlib.sha256(token).hexdigest(),
    }
    write(template, value)
    with pytest.raises(MODULE.RebindError, match="inventory"):
        MODULE.rebind(template, source_root=source, run_id="new-run", output=tmp_path / "out.json")


def test_rebind_rejects_dirty_source_and_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    template, source = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(MODULE, "_source_identity", lambda _: (_ for _ in ()).throw(
        MODULE.RebindError("source repository must be clean before campaign rebind")
    ))
    with pytest.raises(MODULE.RebindError, match="clean"):
        MODULE.rebind(template, source_root=source, run_id="new-run", output=tmp_path / "out.json")
    monkeypatch.setattr(MODULE, "_source_identity", lambda _: "b" * 40)
    output = tmp_path / "out.json"
    output.write_text("existing")
    with pytest.raises(MODULE.RebindError, match="replace"):
        MODULE.rebind(template, source_root=source, run_id="new-run", output=output)
