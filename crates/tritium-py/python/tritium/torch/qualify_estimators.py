"""Installed-wheel estimator catalog qualification worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import torch

from ..nn import TernaryLinear
from .config import TernaryConfig
from .conversion import inspect, prepare_qat
from .errors import TritiumError
from .estimators import (
    Estimator,
    create_estimator,
    register_estimator,
    registered_estimators,
)
from .observability import collect_diagnostics
from .projection import (
    ProjectionContext,
    TernaryPlane,
    TernaryProjection,
    validate_projection,
)


SCHEMA = "tritium.estimator-catalog-execution.v1"
ESTIMATORS = (
    ("absmean-ste", "tritium.absmean-ste", 1),
    ("annealed-ste", "tritium.annealed-ste", 1),
    ("lsq", "tritium.lsq", 1),
    ("salt-ste", "tritium.salt-ste", 1),
    ("sparse-ternary", "tritium.sparse-ternary", 1),
    ("ttq", "tritium.ttq", 2),
    ("twn", "tritium.twn", 1),
)


class _Tied(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Linear(4, 4, bias=False)
        self.right = torch.nn.Linear(4, 4, bias=False)
        self.right.weight = self.left.weight


class _Zeros(Estimator):
    algorithm_id = "tritium.external.zeros"
    schema_version = 1

    def project(
        self, master: torch.Tensor, *, context: ProjectionContext
    ) -> TernaryProjection:
        del context
        trits = torch.zeros_like(master, dtype=torch.int8)
        scales = torch.zeros(
            (master.shape[0], 1), dtype=master.dtype, device=master.device
        )
        return TernaryProjection(
            dense=master * 0,
            planes=(TernaryPlane(trits, scales, master.shape[1]),),
            algorithm_id=self.algorithm_id,
            schema_version=self.schema_version,
        )


class _Invalid(Estimator):
    algorithm_id = "tritium.external.invalid"
    schema_version = 1

    def project(
        self, master: torch.Tensor, *, context: ProjectionContext
    ) -> TernaryProjection:
        del context
        trits = torch.full_like(master, 2, dtype=torch.int8)
        scales = torch.ones(
            (master.shape[0], 1), dtype=master.dtype, device=master.device
        )
        return TernaryProjection(
            dense=master,
            planes=(TernaryPlane(trits, scales, master.shape[1]),),
            algorithm_id=self.algorithm_id,
            schema_version=self.schema_version,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_gradients(parameters: list[torch.Tensor]) -> bool:
    return all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def _case(name: str, algorithm_id: str, physical_planes: int) -> dict[str, Any]:
    torch.manual_seed(0x5A17 + sum(name.encode()))
    estimator = create_estimator(name)
    master = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
    context = ProjectionContext(step=17, training=True, role="weight")
    projection = estimator.project(master, context=context)
    validate_projection(
        projection,
        master,
        algorithm_id=estimator.algorithm_id,
        schema_version=estimator.schema_version,
    )
    decoded = sum(
        (
            plane.trits.to(master.dtype) * plane.scales.to(master.dtype)
            for plane in projection.planes
        ),
        torch.zeros_like(master),
    )
    hard_trits_exact = len(projection.planes) == physical_planes and all(
        set(plane.trits.unique().tolist()) <= {-1, 0, 1}
        for plane in projection.planes
    ) and torch.equal(projection.dense.detach(), decoded)
    finite_scales = all(
        bool(torch.isfinite(plane.scales).all())
        and bool((plane.scales >= 0).all())
        for plane in projection.planes
    )
    projection.dense.square().mean().backward()
    master_gradients = _finite_gradients([master])
    parameters = list(estimator.parameters())
    state_gradients = _finite_gradients(parameters) if parameters else True

    restored = create_estimator(name)
    restored.load_state_dict(estimator.state_dict())
    restored_projection = restored.project(master.detach(), context=context)
    state_roundtrip = torch.equal(
        restored_projection.dense.detach(), projection.dense.detach()
    ) and all(
        torch.equal(left.trits, right.trits)
        and torch.equal(left.scales, right.scales)
        for left, right in zip(
            restored_projection.planes, projection.planes, strict=True
        )
    )

    tied = prepare_qat(
        _Tied(),
        TernaryConfig.qat(estimator=name, planes=physical_planes),
    )
    report = inspect(tied)
    target_entries = [
        entry for entry in report.entries if entry.reason == "target_weight"
    ]
    tied_identity = (
        tied.left.weight is tied.right.weight
        and tied.left.estimator is tied.right.estimator
    )
    coverage_exact = (
        len(target_entries) == 1
        and target_entries[0].aliases == ("left.weight", "right.weight")
        and target_entries[0].numel == 16
    )
    return {
        "name": name,
        "algorithm_id": algorithm_id,
        "schema_version": estimator.schema_version,
        "physical_planes": physical_planes,
        "hard_trits_exact": hard_trits_exact,
        "finite_nonnegative_scales": finite_scales,
        "master_gradients_finite": master_gradients,
        "state_gradients_finite": state_gradients,
        "state_roundtrip_exact": state_roundtrip,
        "tied_identity_preserved": tied_identity,
        "coverage_exact": coverage_exact,
    }


def _plugin(run_id: str) -> dict[str, bool]:
    suffix = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    name = f"release-zeros-{suffix}"
    register_estimator(name, _Zeros)
    registered = name in registered_estimators() and isinstance(
        create_estimator(name), _Zeros
    )
    duplicate_rejected = False
    try:
        register_estimator(name, _Zeros)
    except TritiumError as error:
        duplicate_rejected = error.code == "estimator_registry"

    master = torch.randn(2, 4, requires_grad=True)
    estimator = create_estimator(name)
    projection = estimator.project(
        master,
        context=ProjectionContext(step=0, training=True, role="weight"),
    )
    validate_projection(
        projection,
        master,
        algorithm_id=estimator.algorithm_id,
        schema_version=estimator.schema_version,
    )
    contract_validated = torch.equal(projection.dense, torch.zeros_like(master))

    purity_required = False
    external_layer = TernaryLinear(4, 2, estimator=_Zeros())
    try:
        collect_diagnostics(external_layer)
    except ValueError as error:
        purity_required = "external estimator" in str(error)

    invalid_rejected = False
    invalid = _Invalid()
    bad = invalid.project(
        master,
        context=ProjectionContext(step=0, training=True, role="weight"),
    )
    try:
        validate_projection(
            bad,
            master,
            algorithm_id=invalid.algorithm_id,
            schema_version=invalid.schema_version,
        )
    except (TypeError, ValueError, TritiumError):
        invalid_rejected = True
    return {
        "registered": registered,
        "duplicate_rejected": duplicate_rejected,
        "contract_validated": contract_validated,
        "purity_opt_in_required": purity_required,
        "invalid_projection_rejected": invalid_rejected,
    }


def run(
    *, wheel: Path, source_revision: str, release: str, run_id: str
) -> dict[str, Any]:
    if wheel.is_symlink() or not wheel.is_file() or wheel.stat().st_size <= 0:
        raise ValueError("candidate wheel must be an ordinary nonempty file")
    wheel = wheel.resolve(strict=True)
    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise ValueError("source revision must be 40 lowercase hexadecimal")
    if not release or not run_id:
        raise ValueError("release and run id must be non-empty")
    cases = [_case(*spec) for spec in ESTIMATORS]
    plugin = _plugin(run_id)
    passed = all(
        all(
            case[field] is True
            for field in (
                "hard_trits_exact",
                "finite_nonnegative_scales",
                "master_gradients_finite",
                "state_gradients_finite",
                "state_roundtrip_exact",
                "tied_identity_preserved",
                "coverage_exact",
            )
        )
        for case in cases
    ) and all(plugin.values())
    return {
        "schema": SCHEMA,
        "result": "pass" if passed else "fail",
        "release": release,
        "source_revision": source_revision,
        "run_id": run_id,
        "wheel": {
            "name": wheel.name,
            "bytes": wheel.stat().st_size,
            "sha256": _sha256(wheel),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tritium": importlib.metadata.version("tritium-torch"),
            "device": "cpu",
        },
        "estimators": cases,
        "external_plugin": plugin,
    }


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        wheel=args.wheel,
        source_revision=args.source_revision,
        release=args.release,
        run_id=args.run_id,
    )
    _write_atomic(args.output.absolute(), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
