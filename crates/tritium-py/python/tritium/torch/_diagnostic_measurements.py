"""Private draft CPU measurements; no observability release admission."""

from __future__ import annotations

import os
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

from . import _diagnostic_witness as witness
from ._installed_candidate import verify_installed_candidate
from ._wheel_identity import file_sha256
from .config import TernaryConfig
from .conversion import prepare_qat
from .estimators import AbsMeanSTE
from .projection import ProjectionContext, validate_projection


class _Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(4, 2, bias=False, device="cpu", dtype=torch.float32)
        self.right = nn.Linear(4, 2, bias=False, device="cpu", dtype=torch.float32)
        self.right.weight = self.left.weight

    def forward(self, inputs):
        return self.left(inputs) + self.right(inputs)


def _measure_fixture(model, binding):
    """Internal arithmetic seam. A unit call is not candidate qualification."""
    from tritium.nn import TernaryLinear

    for layer in (model.left, model.right):
        if (
            not isinstance(layer, TernaryLinear)
            or type(layer.estimator) is not AbsMeanSTE
            or layer.bias is not None
            or tuple(layer.weight.shape) != (2, 4)
            or layer.weight.device.type != "cpu"
            or layer.weight.dtype != torch.float32
            or layer.weight.detach().tolist() != witness.fixture_weights()
        ):
            raise ValueError("draft measurements require the frozen CPU AbsMean fixture")
    if model.left.weight is not model.right.weight:
        raise ValueError("draft measurements require tied fixture weights")

    inputs = torch.tensor(witness.INPUTS, dtype=torch.float32, device="cpu")
    with torch.inference_mode():
        projection = model.left.estimator.project(
            model.left.weight,
            context=ProjectionContext(step=0, training=model.training, role="weight"),
        )
        validate_projection(projection, model.left.weight)
        teacher = F.linear(inputs, model.left.weight) * 2
        student = model(inputs)
        for _ in range(5):
            if not torch.equal(model(inputs), student):
                raise ValueError("draft warmup output differs from frozen student")
        samples = []
        for _ in range(31):
            start = time.perf_counter_ns()
            output = model(inputs)
            end = time.perf_counter_ns()
            if not torch.equal(output, student):
                raise ValueError("draft timed output differs from frozen student")
            samples.append({
                "start_ns": start, "end_ns": end,
                "student_logits": output.tolist(),
            })
        # Current whole-process Linux RSS, not peak/model-only/GPU bytes.
        raw_statm = Path("/proc/self/statm").read_text(encoding="ascii")
        page_size = os.sysconf("SC_PAGE_SIZE")
        p = teacher.to(torch.float64).log_softmax(-1)
        q = student.to(torch.float64).log_softmax(-1)
        kl = (p.exp() * (p - q)).sum(-1).mean().clamp_min(0).item()
        value = {
            "schema": witness.SCHEMA,
            "binding": dict(binding),
            "fixture": "tied-linear-4x2-cpu-f32-v1",
            "teacher_weight": model.left.weight.detach().tolist(),
            "student_weight": projection.dense.tolist(),
            "inputs": inputs.tolist(),
            "teacher_logits": teacher.tolist(),
            "student_logits": student.tolist(),
            "timing": {"clock": "perf_counter_ns", "warmups": 5, "samples": samples},
            "memory": {
                "method": "linux-proc-self-statm",
                "scope": "whole-process-after-forward-before-telemetry",
                "raw_statm": raw_statm, "page_size": page_size,
            },
            "metrics": {
                "runtime/forward_ms": statistics.median(
                    sample["end_ns"] - sample["start_ns"] for sample in samples
                ) / 1e6,
                "memory/resident_bytes": float(int(raw_statm.split()[1]) * page_size),
                "teacher_kl": kl,
            },
        }
    value["witness_id"] = witness.witness_id(value)
    witness.validate_witness(value, expected_binding=binding)
    return value


def measure_installed_fixture(*, wheel_artifact: Path, source_revision: str,
                              release: str, run_id: str):
    """Measure one exact installed candidate; return a NON-ADMITTED draft."""
    verify_installed_candidate(
        wheel_artifact=wheel_artifact, source_revision=source_revision,
        release=release,
        executing_files=(Path(__file__), Path(witness.__file__)),
    )
    binding = {
        "source_revision": source_revision, "release": release, "run_id": run_id,
        "wheel_sha256": file_sha256(wheel_artifact),
    }
    model = prepare_qat(_Fixture(), TernaryConfig.qat())
    with torch.no_grad():
        model.left.weight.copy_(torch.tensor(witness.WEIGHTS, dtype=torch.float32))
    # Preserve the existing diagnostic fixture's finite-gradient prerequisite.
    model(torch.ones(1, 4, dtype=torch.float32)).sum().backward()
    return _measure_fixture(model, binding)
