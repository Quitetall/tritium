"""Distributed preflight admission tests, not multi-GPU qualification."""

import argparse
import importlib.metadata
import os
from pathlib import Path
import re
import runpy
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tritium import _tritium  # noqa: E402


@pytest.fixture
def candidate_wheel():
    directory = os.environ.get("TRITIUM_TEST_CANDIDATE_WHEEL_DIR")
    if directory is None:
        pytest.skip("requires the executing candidate wheel directory")
    wheels = list(Path(directory).resolve(strict=True).glob("pytritium-*.whl"))
    assert len(wheels) == 1, "candidate directory must contain exactly one wheel"
    return wheels[0]


@pytest.fixture
def worker():
    # This frozen checkout script is intentionally outside the installed wheel.
    # The outer qualifier binds its clean revision; package bytes need a separate
    # installed-candidate check. Do not require wheel ownership of this script.
    return runpy.run_path(Path(__file__).with_name("hf_multi_gpu_worker.py"))


def candidate_args(tmp_path, wheel):
    return argparse.Namespace(
        mode="ddp", output=tmp_path / "fragment.json",
        checkpoint=tmp_path / "checkpoint", wheel=wheel,
        source_revision=_tritium.source_identity().removeprefix("source-git:"),
        release=re.sub(r"rc(\d+)$", r"-rc.\1", importlib.metadata.version("pytritium")),
    )


@pytest.mark.parametrize("entry", ["namespace", "cli"])
@pytest.mark.parametrize("failure", ["source", "release", "wheel"])
def test_distributed_admission_before_hardware(
    tmp_path, monkeypatch, candidate_wheel, worker, failure, entry,
):
    args = candidate_args(tmp_path, candidate_wheel)
    if failure == "source":
        args.source_revision = "a" * 40 if args.source_revision != "a" * 40 else "b" * 40
    elif failure == "release":
        args.release = "0.0.0"
    else:
        args.wheel = tmp_path / "opaque-candidate.whl"
        args.wheel.write_bytes(b"not an executing candidate wheel")
    if entry == "namespace":
        # Isolate real main preflight independently of CLI argument evolution.
        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _self: args)
    else:
        monkeypatch.setattr(sys, "argv", worker_argv(args))

    def forbidden_hardware():
        raise AssertionError("unbound distributed candidate entered hardware execution")

    monkeypatch.setattr(worker["torch"].cuda, "is_available", forbidden_hardware)
    expected = {"source": "native source", "release": "release", "wheel": "wheel"}[failure]
    with pytest.raises((ValueError, RuntimeError), match=expected):
        worker["main"]()
    assert not args.output.exists()
    assert not args.checkpoint.exists()


def worker_argv(args):
    return [
        "hf_multi_gpu_worker", "--mode", args.mode,
        "--output", str(args.output), "--checkpoint", str(args.checkpoint),
        "--wheel", str(args.wheel), "--source-revision", args.source_revision,
        "--release", args.release,
    ]


@pytest.mark.parametrize("mode", ["ddp", "fsdp"])
def test_bound_checkout_worker_reaches_physical_guard(
    tmp_path, monkeypatch, candidate_wheel, worker, mode,
):
    args = candidate_args(tmp_path, candidate_wheel)
    args.mode = mode
    monkeypatch.setattr(sys, "argv", worker_argv(args))
    monkeypatch.setattr(worker["torch"].cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="two visible physical CUDA devices"):
        worker["main"]()
    assert not args.output.exists()
    assert not args.checkpoint.exists()
