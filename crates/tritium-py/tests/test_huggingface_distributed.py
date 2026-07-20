"""Real multi-process Hugging Face training gates for plan 0047."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")
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
