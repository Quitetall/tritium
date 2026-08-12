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
        payload = kind.encode()
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
