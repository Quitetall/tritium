"""Real multi-process Hugging Face training gates for plan 0047."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_two_rank_cpu_ddp_step_and_checkpoint(tmp_path: Path):
    worker = Path(__file__).with_name("hf_ddp_worker.py")
    env = os.environ.copy()
    env["TRITIUM_DDP_CHECKPOINT"] = str(tmp_path / "ddp-state.pt")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(worker),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout + completed.stderr
    assert "TRITIUM_DDP_OK rank=0" in output
    assert "TRITIUM_DDP_OK rank=1" in output


def test_two_rank_cpu_fsdp_step_and_sharded_state_resume(tmp_path: Path):
    worker = Path(__file__).with_name("hf_fsdp_worker.py")
    env = os.environ.copy()
    env["TRITIUM_FSDP_CHECKPOINT"] = str(tmp_path / "fsdp-state")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(worker),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout + completed.stderr
    assert "TRITIUM_FSDP_OK rank=0" in output
    assert "TRITIUM_FSDP_OK rank=1" in output


def test_accelerate_cpu_bf16_in_fresh_runtime():
    worker = Path(__file__).with_name("hf_accelerate_worker.py")
    completed = subprocess.run(
        [sys.executable, str(worker)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "TRITIUM_ACCELERATE_BF16_OK" in completed.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_accelerate_cuda_fp16_checkpoint_and_residency(tmp_path: Path):
    worker = Path(__file__).with_name("hf_cuda_worker.py")
    receipt = tmp_path / "cuda-qualification.json"
    env = os.environ.copy()
    env["TRITIUM_CUDA_CHECKPOINT"] = str(tmp_path / "cuda-state")
    env["TRITIUM_CUDA_RECEIPT"] = str(receipt)
    env["TRITIUM_SOURCE_REVISION"] = "a" * 40
    env["TRITIUM_RELEASE"] = "1.1.0-rc.0"
    env["TRITIUM_RUN_ID"] = "pytest-cuda-qualification"
    env["TRITIUM_QUALIFIED_ARTIFACT"] = str(worker)
    env["TRITIUM_ARTIFACT_KIND"] = "source-test-worker"
    completed = subprocess.run(
        [sys.executable, str(worker)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "TRITIUM_ACCELERATE_CUDA_FP16_OK" in completed.stdout
    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["schema"] == "tritium.cuda-training-qualification.v1"
    assert value["source_revision"] == "a" * 40
    assert value["workload"]["mixed_precision"] == "fp16"
    assert value["invariants"]["ternary_operator_host_transfers"] == 0
    assert value["invariants"]["checkpoint_exact"] is True
    assert value["measurements"]["steps_per_second"] > 0
    identity = dict(value)
    receipt_id = identity.pop("receipt_id")
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    assert receipt_id == "sha256:" + hashlib.sha256(canonical).hexdigest()
