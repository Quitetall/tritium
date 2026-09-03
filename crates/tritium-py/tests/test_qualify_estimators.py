"""Installed-wheel estimator qualification worker gates."""

import json

import pytest

pytest.importorskip("torch")

from tritium.torch import qualify_estimators as worker  # noqa: E402


def test_worker_executes_complete_catalog_and_plugin_contract(tmp_path, monkeypatch):
    wheel = tmp_path / "pytritium-1.1.0rc0-py3-none-any.whl"
    wheel.write_bytes(b"candidate-wheel")
    monkeypatch.setattr(
        worker.importlib.metadata,
        "version",
        lambda distribution: "1.1.0rc0"
        if distribution == "pytritium"
        else pytest.fail(f"unexpected distribution lookup: {distribution}"),
    )

    result = worker.run(
        wheel=wheel,
        source_revision="a" * 40,
        release="1.1.0-rc.0",
        run_id="estimator-worker-test",
    )

    assert result["schema"] == worker.SCHEMA
    assert result["result"] == "pass"
    assert result["environment"]["tritium"] == "1.1.0rc0"
    assert [
        (case["name"], case["algorithm_id"], case["physical_planes"])
        for case in result["estimators"]
    ] == list(worker.ESTIMATORS)
    assert all(
        value is True
        for case in result["estimators"]
        for field, value in case.items()
        if field
        not in {"name", "algorithm_id", "schema_version", "physical_planes"}
    )
    assert all(result["external_plugin"].values())


def test_worker_write_is_atomic_json(tmp_path):
    output = tmp_path / "nested" / "trace.json"
    worker._write_atomic(output, {"schema": worker.SCHEMA, "result": "pass"})
    assert json.loads(output.read_bytes()) == {
        "schema": worker.SCHEMA,
        "result": "pass",
    }
    assert not list(output.parent.glob(f".{output.name}.*"))


def test_worker_rejects_symlinked_candidate_wheel(tmp_path):
    wheel = tmp_path / "candidate.whl"
    wheel.write_bytes(b"candidate-wheel")
    link = tmp_path / "linked.whl"
    link.symlink_to(wheel)
    with pytest.raises(ValueError, match="ordinary"):
        worker.run(
            wheel=link,
            source_revision="a" * 40,
            release="1.1.0-rc.0",
            run_id="symlink-rejection",
        )
