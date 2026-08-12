"""Tests for resumable Stage-7 measurement orchestration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "run-stage7-recipe-freeze.py"
spec = importlib.util.spec_from_file_location("stage7_runner", SCRIPT)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_safe_digest_name_accepts_only_sha256_identity() -> None:
    candidate = "sha256:" + "a" * 64
    assert runner._safe_digest_name(candidate) == "a" * 64
    for invalid in ("", "sha256:" + "a" * 63, "sha256:" + "g" * 64, "sha512:" + "a" * 64):
        with pytest.raises(runner.Stage7RunError):
            runner._safe_digest_name(invalid)


def test_runner_response_requires_one_strict_json_object() -> None:
    command = [
        sys.executable,
        "-c",
        "import json,sys; json.dump({'ok': True}, sys.stdout)",
    ]
    assert runner._runner_response(command, {"request": 1}, timeout_seconds=2) == {"ok": True}

    bad_json = [sys.executable, "-c", "print('NaN')"]
    with pytest.raises(runner.Stage7RunError, match="strict UTF-8 JSON"):
        runner._runner_response(bad_json, {}, timeout_seconds=2)

    failed = [sys.executable, "-c", "import sys; sys.stderr.write('bad'); sys.exit(7)"]
    with pytest.raises(runner.Stage7RunError, match="exit 7"):
        runner._runner_response(failed, {}, timeout_seconds=2)


def test_capability_preflight_requires_complete_measurement_surface() -> None:
    source_revision = "a" * 40
    valid = {
        "schema": runner.CAPABILITY_SCHEMA,
        "source_revision": source_revision,
        "stages": list(runner.STAGE_NAMES),
        "codecs": list(runner.RECIPE_CODECS),
        "groups": list(runner.RECIPE_GROUPS),
        "planes": list(runner.RECIPE_PLANES),
        "rotations": list(runner.RECIPE_ROTATIONS),
        "curvatures": list(runner.RECIPE_CURVATURES),
        "solvers": list(runner.RECIPE_SOLVERS),
        "features": {
            "full_artifacts": True,
            "physical_reports": True,
            "baselines": False,
            "refinements": False,
        },
    }
    assert runner._validate_capabilities(
        valid, kind="measurement", source_revision=source_revision
    ) is valid
    valid["solvers"] = valid["solvers"][:-1]
    with pytest.raises(runner.Stage7RunError, match="solvers"):
        runner._validate_capabilities(
            valid, kind="measurement", source_revision=source_revision
        )


def test_auxiliary_capability_preflight_requires_baselines_and_refinements() -> None:
    value = {
        "schema": runner.CAPABILITY_SCHEMA,
        "source_revision": "b" * 40,
        "stages": [],
        "codecs": [],
        "groups": [],
        "planes": [],
        "rotations": [],
        "curvatures": [],
        "solvers": [],
        "features": {
            "full_artifacts": False,
            "physical_reports": False,
            "baselines": True,
            "refinements": True,
        },
    }
    assert runner._validate_capabilities(
        value, kind="auxiliary", source_revision="b" * 40
    ) is value
    value["features"]["refinements"] = False
    with pytest.raises(runner.Stage7RunError, match="required"):
        runner._validate_capabilities(
            value, kind="auxiliary", source_revision="b" * 40
        )


def test_cached_measurement_binds_request_stage_and_candidate(tmp_path: Path) -> None:
    candidate = "sha256:" + "b" * 64
    value = {
        "schema": runner.REQUEST_SCHEMA,
        "request_id": "sha256:" + "c" * 64,
        "stage": "one-layer",
        "candidate_id": candidate,
        "measurement": {"candidate_id": candidate},
    }
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert runner._cached_measurement(
        path,
        request_id=value["request_id"],
        candidate_id=candidate,
        stage="one-layer",
    ) == value["measurement"]
    with pytest.raises(runner.Stage7RunError, match="identity differs"):
        runner._cached_measurement(
            path,
            request_id="sha256:" + "d" * 64,
            candidate_id=candidate,
            stage="one-layer",
        )


def test_auxiliary_validation_precedes_cache_publication() -> None:
    valid = {
        "schema": runner.AUXILIARY_SCHEMA,
        "baselines": {},
        "refinements": [],
    }
    assert runner._validate_auxiliary(valid) is valid
    with pytest.raises(runner.Stage7RunError, match="fields differ"):
        runner._validate_auxiliary({"schema": runner.AUXILIARY_SCHEMA})
    with pytest.raises(runner.Stage7RunError, match="schema differs"):
        runner._validate_auxiliary({"schema": "wrong", "baselines": {}, "refinements": []})


def test_write_new_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    runner._write_new(path, {"schema": "test"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"schema": "test"}
    with pytest.raises(runner.Stage7RunError, match="refusing to replace"):
        runner._write_new(path, {"schema": "changed"})
