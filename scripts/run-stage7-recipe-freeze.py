#!/usr/bin/env python3
"""Run a source-bound Stage-7 measurement campaign.

This module is orchestration only.  It never invents quality, runtime, memory,
package, baseline, or refinement values.  A caller-supplied runner must emit
one strict measurement JSON object for every admitted recipe.  The runner is
invoked once per stage/candidate and its responses are cached under the
campaign evidence directory so an interrupted campaign can resume without
replaying completed work.

The final trace is accepted only after the existing Stage-7 qualifier validates
all rows, artifacts, physical reports, baselines, and refinements.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_RUNNER_OUTPUT_BYTES = 16 * 1024 * 1024
TRACE_SCHEMA = "tritium.stage7-execution.v1"
REQUEST_SCHEMA = "tritium.stage7-measurement-request.v1"
AUXILIARY_SCHEMA = "tritium.stage7-auxiliary-request.v1"
CAPABILITY_REQUEST_SCHEMA = "tritium.stage7-capabilities-request.v1"
CAPABILITY_SCHEMA = "tritium.stage7-capabilities.v1"
CAPABILITY_FEATURE_FIELDS = {
    "full_artifacts", "physical_reports", "baselines", "refinements",
}
CAPABILITY_FIELDS = {
    "schema", "request_id", "source_revision", "stages", "codecs", "groups",
    "planes", "rotations", "curvatures", "solvers", "features",
}
STAGE_NAMES = ("one-layer", "four-layer", "full-model")
RECIPE_CODECS = ("D2", "B3", "S34")
RECIPE_GROUPS = (64, 128, 256)
RECIPE_PLANES = (2, 3)
RECIPE_ROTATIONS = ("none", "signed-rht")
RECIPE_CURVATURES = (
    "input-hessian", "guided-fisher", "forward-kl-kronecker",
)
RECIPE_SOLVERS = (
    "greedy", "joint", "joint+feedback", "joint+feedback+output-recon",
    "+softened-relay-basin", "+modulated-basin",
)
MEASUREMENT_FIELDS = {
    "candidate_id", "track", "physical_bytes", "resident_bytes", "output_loss",
    "heldout_ppl", "task_metrics", "runtime_ms", "artifact", "physical_report",
    "correct",
}


class Stage7RunError(ValueError):
    """Campaign execution failed before a qualified trace could be published."""


def _capability_request(
    *,
    kind: str,
    campaign: dict[str, Any],
    campaign_sha256: str,
    command: list[str],
    model_root: Path,
    source_root: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    if kind not in {"measurement", "auxiliary"}:
        raise Stage7RunError("capability request kind is invalid")
    request = {
        "schema": CAPABILITY_REQUEST_SCHEMA,
        "kind": kind,
        "source_revision": campaign["source_revision"],
        "run_id": campaign["run_id"],
        "campaign_sha256": campaign_sha256,
        "runner": command,
        "model_root": str(model_root),
        "source_root": str(source_root),
        "evidence_root": str(evidence_root),
    }
    request["request_id"] = "sha256:" + hashlib.sha256(canonical(request)).hexdigest()
    return request


def _validate_capabilities(
    value: dict[str, Any],
    *,
    kind: str,
    request_id: str,
    source_revision: str,
) -> dict[str, Any]:
    """Validate runner declaration before any candidate work is started."""

    if kind not in {"measurement", "auxiliary"}:
        raise Stage7RunError("capability kind is invalid")
    if set(value) != CAPABILITY_FIELDS:
        raise Stage7RunError("runner capability fields differ")
    if value["schema"] != CAPABILITY_SCHEMA:
        raise Stage7RunError("runner capability schema differs")
    if value["request_id"] != request_id:
        raise Stage7RunError("runner capability request identity differs")
    if value["source_revision"] != source_revision:
        raise Stage7RunError("runner capability source revision differs")

    def exact_list(field: str, expected: tuple[Any, ...]) -> None:
        observed = value[field]
        if observed != list(expected):
            raise Stage7RunError(
                f"runner capability {field} does not cover frozen Stage-7 contract"
            )

    if kind == "measurement":
        exact_list("stages", STAGE_NAMES)
        exact_list("codecs", RECIPE_CODECS)
        exact_list("groups", RECIPE_GROUPS)
        exact_list("planes", RECIPE_PLANES)
        exact_list("rotations", RECIPE_ROTATIONS)
        exact_list("curvatures", RECIPE_CURVATURES)
        exact_list("solvers", RECIPE_SOLVERS)
    else:
        for field in ("stages", "codecs", "groups", "planes", "rotations", "curvatures", "solvers"):
            if value[field] != []:
                raise Stage7RunError(
                    f"auxiliary runner capability {field} must be empty"
                )
    features = value["features"]
    if not isinstance(features, dict) or set(features) != CAPABILITY_FEATURE_FIELDS:
        raise Stage7RunError("runner capability features differ")
    if any(type(flag) is not bool for flag in features.values()):
        raise Stage7RunError("runner capability features must be boolean")
    required = (
        {"full_artifacts", "physical_reports"}
        if kind == "measurement"
        else {"baselines", "refinements"}
    )
    if any(features[field] is not True for field in required):
        raise Stage7RunError(
            f"{kind} runner does not advertise required Stage-7 capabilities"
        )
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise Stage7RunError(f"{label} must be a bounded ordinary file")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise Stage7RunError(f"{label} exceeds JSON byte bound")
    try:
        value = json.loads(
            path.read_bytes(),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Stage7RunError(f"{label} must contain strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise Stage7RunError(f"{label} must contain a JSON object")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualifier(source_root: Path):
    path = source_root / "scripts" / "qualify-stage7-recipe-freeze.py"
    spec = importlib.util.spec_from_file_location("tritium_stage7_qualifier", path)
    if spec is None or spec.loader is None:
        raise Stage7RunError("cannot load Stage-7 qualifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_identity(source_root: Path) -> str:
    try:
        top = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise Stage7RunError("source repository identity probe failed") from error
    if Path(top).resolve() != source_root.resolve():
        raise Stage7RunError("source root must be repository top level")
    if dirty:
        raise Stage7RunError("source repository must be clean for Stage-7 execution")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise Stage7RunError("source HEAD is not a canonical Git revision")
    return revision


def _safe_digest_name(candidate_id: str) -> str:
    if not candidate_id.startswith("sha256:") or len(candidate_id) != 71:
        raise Stage7RunError("candidate id is not a canonical SHA-256 identity")
    suffix = candidate_id[7:]
    if any(char not in "0123456789abcdef" for char in suffix):
        raise Stage7RunError("candidate id is not a canonical SHA-256 identity")
    return suffix


def _runner_response(
    command: list[str], request: dict[str, Any], *, timeout_seconds: float
) -> dict[str, Any]:
    if not command or any("\0" in part for part in command):
        raise Stage7RunError("runner command must be a nonempty NUL-free argv")
    if timeout_seconds <= 0:
        raise Stage7RunError("runner timeout must be positive")
    try:
        completed = subprocess.run(
            command,
            input=canonical(request),
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise Stage7RunError(
            f"runner timed out after {timeout_seconds:.3f}s"
        ) from error
    except OSError as error:
        raise Stage7RunError("runner could not be started") from error
    if completed.returncode != 0:
        stderr = completed.stderr[:4096].decode("utf-8", "replace")
        raise Stage7RunError(
            f"runner failed with exit {completed.returncode}: {stderr.strip()}"
        )
    if len(completed.stdout) > MAX_RUNNER_OUTPUT_BYTES:
        raise Stage7RunError("runner response exceeds byte bound")
    try:
        value = json.loads(
            completed.stdout,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Stage7RunError("runner stdout must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise Stage7RunError("runner stdout must contain one JSON object")
    return value


def _measurement(
    qualifier: Any,
    row: dict[str, Any],
    *,
    stage: str,
    grid: dict[str, dict[str, Any]],
    campaign: dict[str, Any],
    evidence_root: Path,
    quantized: int,
    quantized_tensors: int,
    preserved: int,
) -> dict[str, Any]:
    if set(row) != MEASUREMENT_FIELDS:
        raise Stage7RunError(f"{stage} runner response fields differ")
    try:
        validated, _ = qualifier._measurement(
            row,
            full=stage == "full-model",
            grid=grid,
            campaign=campaign,
            trace_root=evidence_root,
            quantized=quantized,
            quantized_tensors=quantized_tensors,
            preserved=preserved,
            label=f"{stage}.measurement",
        )
    except (OSError, qualifier.Stage7Error) as error:
        raise Stage7RunError(str(error)) from error
    return validated


def _cached_measurement(
    path: Path,
    *,
    request_id: str,
    candidate_id: str,
    stage: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = load_json(path, f"cached {stage} measurement")
    if set(value) != {"schema", "request_id", "stage", "candidate_id", "measurement"}:
        raise Stage7RunError(f"cached {stage} measurement envelope differs")
    if (
        value["schema"] != REQUEST_SCHEMA
        or value["request_id"] != request_id
        or value["stage"] != stage
        or value["candidate_id"] != candidate_id
    ):
        raise Stage7RunError(f"cached {stage} measurement identity differs")
    measurement = value["measurement"]
    if not isinstance(measurement, dict):
        raise Stage7RunError(f"cached {stage} measurement is not an object")
    return measurement


def _validate_auxiliary(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != {"schema", "baselines", "refinements"}:
        raise Stage7RunError("auxiliary runner response fields differ")
    if value["schema"] != AUXILIARY_SCHEMA:
        raise Stage7RunError("auxiliary runner schema differs")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise Stage7RunError(f"refusing to replace existing evidence file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise Stage7RunError(f"refusing to replace existing evidence file: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _run_stage(
    qualifier: Any,
    *,
    command: list[str],
    stage: str,
    input_ids: list[str],
    campaign: dict[str, Any],
    campaign_sha256: str,
    evidence_root: Path,
    grid: dict[str, dict[str, Any]],
    model_root: Path,
    source_root: Path,
    run_id: str,
    quantized: int,
    quantized_tensors: int,
    preserved: int,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    for candidate_id in input_ids:
        recipe = grid[candidate_id]
        request = {
            "schema": REQUEST_SCHEMA,
            "source_revision": campaign["source_revision"],
            "run_id": run_id,
            "campaign_sha256": campaign_sha256,
            "stage": stage,
            "candidate_id": candidate_id,
            "recipe": recipe,
            "runner": command,
            "model_root": str(model_root),
            "source_root": str(source_root),
            "evidence_root": str(evidence_root),
        }
        request_id = "sha256:" + hashlib.sha256(canonical(request)).hexdigest()
        request["request_id"] = request_id
        cache = evidence_root / "measurements" / stage / f"{_safe_digest_name(candidate_id)}.json"
        row = _cached_measurement(
            cache,
            request_id=request_id,
            candidate_id=candidate_id,
            stage=stage,
        )
        if row is None:
            row = _runner_response(command, request, timeout_seconds=timeout_seconds)
            row = _measurement(
                qualifier,
                row,
                stage=stage,
                grid=grid,
                campaign=campaign,
                evidence_root=evidence_root,
                quantized=quantized,
                quantized_tensors=quantized_tensors,
                preserved=preserved,
            )
            _write_new(
                cache,
                {
                    "schema": REQUEST_SCHEMA,
                    "request_id": request_id,
                    "stage": stage,
                    "candidate_id": candidate_id,
                    "measurement": row,
                },
            )
        else:
            row = _measurement(
                qualifier,
                row,
                stage=stage,
                grid=grid,
                campaign=campaign,
                evidence_root=evidence_root,
                quantized=quantized,
                quantized_tensors=quantized_tensors,
                preserved=preserved,
            )
        if row["candidate_id"] != candidate_id:
            raise Stage7RunError(
                f"{stage} runner response candidate differs from requested candidate"
            )
        rows.append(row)
    promoted = qualifier._expected_promotions(stage, rows, grid)
    return rows, promoted


def run(
    campaign_path: Path,
    *,
    model_root: Path,
    smoke_model_root: Path,
    source_root: Path,
    runner: list[str],
    auxiliary_runner: list[str],
    timeout_seconds: float,
    output: Path,
) -> dict[str, Any]:
    if campaign_path.is_symlink():
        raise Stage7RunError("Stage-7 campaign must not be a symlink")
    if output.is_symlink():
        raise Stage7RunError("trace output must not be a symlink")
    source_root = source_root.resolve(strict=True)
    source_revision = _source_identity(source_root)
    campaign_path = campaign_path.resolve(strict=True)
    evidence_root = campaign_path.parent
    output = output.absolute()
    if output.parent.resolve() != evidence_root:
        raise Stage7RunError("trace output must share campaign evidence directory")
    campaign = load_json(campaign_path, "Stage-7 campaign")
    if campaign.get("source_revision") != source_revision:
        raise Stage7RunError("campaign source revision differs from clean repository HEAD")
    qualifier = _qualifier(source_root)
    try:
        validated = qualifier._validate_campaign(
            campaign_path,
            model_root=model_root.resolve(strict=True),
            smoke_model_root=smoke_model_root.resolve(strict=True),
            source_root=source_root,
        )
    except (OSError, qualifier.Stage7Error) as error:
        raise Stage7RunError(str(error)) from error
    (
        campaign, _, grid, _, quantized, preserved,
        quantized_tensors, _, prerequisite_reasons,
    ) = validated
    if prerequisite_reasons:
        raise Stage7RunError(
            "Stage-7 prerequisites failed: " + ", ".join(prerequisite_reasons)
        )
    campaign_sha256 = digest(campaign_path)
    capability_context = {
        "campaign": campaign,
        "campaign_sha256": campaign_sha256,
        "model_root": model_root.resolve(strict=True),
        "source_root": source_root,
        "evidence_root": evidence_root,
    }
    measurement_capability_request = _capability_request(
        kind="measurement", command=runner, **capability_context
    )
    _validate_capabilities(
        _runner_response(
            runner,
            measurement_capability_request,
            timeout_seconds=timeout_seconds,
        ),
        kind="measurement",
        request_id=measurement_capability_request["request_id"],
        source_revision=campaign["source_revision"],
    )
    auxiliary_capability_request = _capability_request(
        kind="auxiliary", command=auxiliary_runner, **capability_context
    )
    _validate_capabilities(
        _runner_response(
            auxiliary_runner,
            auxiliary_capability_request,
            timeout_seconds=timeout_seconds,
        ),
        kind="auxiliary",
        request_id=auxiliary_capability_request["request_id"],
        source_revision=campaign["source_revision"],
    )
    previous = sorted(grid)
    stages = []
    for name in ("one-layer", "four-layer", "full-model"):
        rows, promoted = _run_stage(
            qualifier,
            command=runner,
            stage=name,
            input_ids=previous,
            campaign=campaign,
            campaign_sha256=campaign_sha256,
            evidence_root=evidence_root,
            grid=grid,
            model_root=model_root.resolve(strict=True),
            source_root=source_root,
            run_id=campaign["run_id"],
            quantized=quantized,
            quantized_tensors=quantized_tensors,
            preserved=preserved,
            timeout_seconds=timeout_seconds,
        )
        stages.append({
            "name": name,
            "input_ids": previous,
            "measurements": rows,
            "promoted_ids": promoted,
        })
        previous = sorted(promoted)

    auxiliary_request = {
        "schema": AUXILIARY_SCHEMA,
        "source_revision": campaign["source_revision"],
        "run_id": campaign["run_id"],
        "campaign_sha256": campaign_sha256,
        "runner": auxiliary_runner,
        "model_root": str(model_root.resolve(strict=True)),
        "source_root": str(source_root),
        "evidence_root": str(evidence_root),
    }
    auxiliary_request_id = "sha256:" + hashlib.sha256(
        canonical(auxiliary_request)
    ).hexdigest()
    auxiliary_cache = evidence_root / "measurements" / "auxiliary.json"
    if auxiliary_cache.exists():
        cached = load_json(auxiliary_cache, "cached auxiliary response")
        if set(cached) != {"schema", "request_id", "response"}:
            raise Stage7RunError("cached auxiliary response envelope differs")
        if (
            cached["schema"] != AUXILIARY_SCHEMA
            or cached["request_id"] != auxiliary_request_id
        ):
            raise Stage7RunError("cached auxiliary response identity differs")
        auxiliary = cached["response"]
        if not isinstance(auxiliary, dict):
            raise Stage7RunError("cached auxiliary response is not an object")
    else:
        auxiliary = _runner_response(
            auxiliary_runner,
            auxiliary_request,
            timeout_seconds=timeout_seconds,
        )
        auxiliary = _validate_auxiliary(auxiliary)
        _write_new(
            auxiliary_cache,
            {
                "schema": AUXILIARY_SCHEMA,
                "request_id": auxiliary_request_id,
                "response": auxiliary,
            },
        )
    auxiliary = _validate_auxiliary(auxiliary)
    trace = {
        "schema": TRACE_SCHEMA,
        "release": campaign["release"],
        "source_revision": campaign["source_revision"],
        "run_id": campaign["run_id"],
        "campaign_sha256": campaign_sha256,
        "stages": stages,
        "baselines": auxiliary["baselines"],
        "refinements": auxiliary["refinements"],
    }
    temporary = evidence_root / f".{output.name}.validation"
    if temporary.exists() or temporary.is_symlink():
        raise Stage7RunError(f"temporary validation path already exists: {temporary}")
    try:
        _write_new(temporary, trace)
        try:
            qualifier._validate_trace(
                temporary,
                campaign_path=campaign_path,
                campaign=campaign,
                grid=grid,
                quantized=quantized,
                quantized_tensors=quantized_tensors,
                quantized_tensor_names=validated[7],
                preserved=preserved,
            )
        except (OSError, qualifier.Stage7Error) as error:
            raise Stage7RunError(str(error)) from error
        if output.exists() or output.is_symlink():
            raise Stage7RunError(f"trace output already exists: {output}")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--smoke-model-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--runner", required=True, nargs="+", help="argv for one candidate measurement")
    parser.add_argument("--auxiliary-runner", required=True, nargs="+", help="argv for baselines/refinements")
    parser.add_argument("--timeout-seconds", type=float, default=86_400.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        trace = run(
            args.campaign,
            model_root=args.model_root,
            smoke_model_root=args.smoke_model_root,
            source_root=args.source_root,
            runner=args.runner,
            auxiliary_runner=args.auxiliary_runner,
            timeout_seconds=args.timeout_seconds,
            output=args.output,
        )
    except (OSError, Stage7RunError) as error:
        parser.error(str(error))
    print(json.dumps({
        "schema": trace["schema"],
        "source_revision": trace["source_revision"],
        "run_id": trace["run_id"],
        "stages": [
            {"name": stage["name"], "inputs": len(stage["input_ids"]), "promoted": len(stage["promoted_ids"])}
            for stage in trace["stages"]
        ],
        "output": str(args.output),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
