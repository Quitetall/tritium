#!/usr/bin/env python3
"""Execute frozen matched-byte baseline recipes and emit raw release evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


CAMPAIGN_SCHEMA = "tritium.baseline-ablation-campaign.v1"
TRACE_SCHEMA = "tritium.baseline-ablation-execution.v1"
SAMPLE_SCHEMA = "tritium.baseline-ablation-sample.v1"
SAMPLE_COUNT = 30
MAX_CAMPAIGN_BYTES = 4 * 1024 * 1024
MAX_SAMPLE_BYTES = 1024 * 1024
CAMPAIGN_FIELDS = {
    "schema", "release", "source_revision", "run_id", "model_artifact_id",
    "evaluation_id", "target_bytes", "parameter_count", "device", "baselines",
}
BASELINE_FIELDS = {
    "method", "family", "recipe", "build_command", "evaluation_command",
    "artifact",
}
SAMPLE_FIELDS = {
    "schema", "evaluation_id", "artifact_sha256", "quality_score",
    "resident_bytes", "physical_device",
}
ENVIRONMENT_FIELDS = {"python", "torch", "tritium"}
BASELINES = (
    ("rtn-absmean", "ternary"),
    ("gptq-style", "global-low-bit"),
    ("awq-style", "global-low-bit"),
    ("salt-v1", "ternary"),
    ("no-curvature", "ablation"),
    ("no-rotation", "ablation"),
    ("greedy-salt-v1", "ablation"),
)
SAFE_ENVIRONMENT = {
    "CUDA_PATH", "CUDA_VISIBLE_DEVICES", "HF_HOME", "HIP_VISIBLE_DEVICES", "HOME",
    "LANG", "LC_ALL", "LD_LIBRARY_PATH", "PATH", "PYTHONPATH", "ROCM_PATH",
    "ROCR_VISIBLE_DEVICES", "TMPDIR", "TORCH_HOME", "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
}


class CampaignError(ValueError):
    """Baseline campaign is unsafe, incomplete, stale, or not byte matched."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise CampaignError(f"{label} fields do not match frozen schema")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise CampaignError(f"{label} must be a nonempty string without NUL")
    return value


def _integer(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CampaignError(f"{label} must be an integer at least {minimum}")
    return value


def _number(value: Any, label: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise CampaignError(f"{label} must be finite and at least {minimum}")
    return float(value)


def _digest(value: Any, label: str) -> str:
    text = _string(value, label)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise CampaignError(f"{label} must be a canonical SHA-256 digest")
    return text


def _command(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise CampaignError(f"{label} must be a nonempty argv list")
    return [_string(item, f"{label} argument") for item in value]


def _artifact_name(value: Any) -> str:
    text = _string(value, "baseline artifact path")
    logical = PurePosixPath(text)
    if logical.is_absolute() or ".." in logical.parts or "\\" in text:
        raise CampaignError("baseline artifact path is unsafe")
    return text


def _relative_artifact(root: Path, value: Any) -> Path:
    text = _artifact_name(value)
    logical = PurePosixPath(text)
    path = root.joinpath(*logical.parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _ordinary_artifact(root: Path, path: Path) -> Path:
    cursor = root.resolve(strict=True)
    relative = path.relative_to(root)
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise CampaignError("baseline artifact path traverses a symlink")
    resolved = cursor.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise CampaignError("baseline artifact escapes work directory") from error
    if not resolved.is_file():
        raise CampaignError("baseline build did not produce an ordinary artifact")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_campaign(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size <= 0
        or path.stat().st_size > MAX_CAMPAIGN_BYTES
    ):
        raise CampaignError("campaign must be a bounded ordinary file")
    try:
        value = json.loads(
            path.read_bytes(),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CampaignError("campaign must contain strict UTF-8 JSON") from error
    return _object(value, CAMPAIGN_FIELDS, "campaign")


def _validate_campaign(value: dict[str, Any]) -> list[dict[str, Any]]:
    if value["schema"] != CAMPAIGN_SCHEMA:
        raise CampaignError("campaign schema differs")
    release = _string(value["release"], "release")
    revision = _string(value["source_revision"], "source revision")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise CampaignError("source revision must be forty lowercase hexadecimal")
    _string(value["run_id"], "run id")
    _string(value["model_artifact_id"], "model artifact id")
    _digest(value["evaluation_id"], "evaluation id")
    _integer(value["target_bytes"], "target bytes")
    _integer(value["parameter_count"], "parameter count")
    _string(value["device"], "physical device")
    baselines = value["baselines"]
    if not isinstance(baselines, list) or len(baselines) != len(BASELINES):
        raise CampaignError("baseline inventory is incomplete")
    result = []
    for ordinal, expected in enumerate(BASELINES):
        row = _object(
            baselines[ordinal], BASELINE_FIELDS, f"baselines[{ordinal}]"
        )
        if (row["method"], row["family"]) != expected:
            raise CampaignError("baseline inventory identity or order differs")
        if not isinstance(row["recipe"], dict) or not row["recipe"]:
            raise CampaignError("baseline recipe must be a nonempty JSON object")
        try:
            canonical(row["recipe"])
        except (TypeError, ValueError) as error:
            raise CampaignError("baseline recipe must be canonical JSON") from error
        _command(row["build_command"], "build command")
        _command(row["evaluation_command"], "evaluation command")
        _artifact_name(row["artifact"])
        result.append(row)
    return result


def _runtime_environment() -> dict[str, str]:
    try:
        import torch
    except ImportError as error:
        raise CampaignError("PyTorch is required for baseline execution") from error
    try:
        tritium_version = importlib.metadata.version("tritium-torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise CampaignError("installed tritium-torch wheel is required") from error
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "tritium": tritium_version,
    }


def _subprocess_runner(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str]
) -> str:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=24 * 60 * 60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CampaignError(f"baseline command could not complete: {command[0]}") from error
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace").strip()
        raise CampaignError(
            f"baseline command failed ({result.returncode}): {stderr[:4096]}"
        )
    if len(result.stdout) > MAX_SAMPLE_BYTES:
        raise CampaignError("baseline command output exceeds bound")
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CampaignError("baseline command output must be UTF-8") from error


def _sample(
    raw: str,
    *,
    evaluation_id: str,
    artifact_sha256: str,
    device: str,
) -> tuple[float, int]:
    try:
        value = _object(json.loads(raw), SAMPLE_FIELDS, "evaluation sample")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CampaignError("evaluation command must emit one frozen JSON sample") from error
    if (
        value["schema"] != SAMPLE_SCHEMA
        or value["evaluation_id"] != evaluation_id
        or value["artifact_sha256"] != artifact_sha256
    ):
        raise CampaignError("evaluation sample identity differs")
    if value["physical_device"] != device:
        raise CampaignError("evaluation sample came from a different physical device")
    return (
        _number(value["quality_score"], "quality score", 1e-12),
        _integer(value["resident_bytes"], "resident bytes"),
    )


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CampaignError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise CampaignError(f"output already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def execute(
    campaign_path: Path,
    *,
    output: Path,
    work_dir: Path,
    command_runner: Callable[..., str] = _subprocess_runner,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute every frozen recipe; publish trace only after all rows reproduce."""
    campaign = _load_campaign(campaign_path)
    baselines = _validate_campaign(campaign)
    runtime = (
        _runtime_environment()
        if environment is None
        else _object(environment, ENVIRONMENT_FIELDS, "runtime environment")
    )
    expected_version = campaign["release"].replace("-rc.", "rc")
    if runtime["tritium"] != expected_version:
        raise CampaignError("installed Tritium version differs from campaign release")
    for field in ENVIRONMENT_FIELDS:
        _string(runtime[field], f"runtime {field}")
    if output.absolute() == work_dir.absolute():
        raise CampaignError("output and work directory must differ")
    if output.exists() or output.is_symlink():
        raise CampaignError(f"output already exists: {output}")
    if work_dir.exists() or work_dir.is_symlink():
        raise CampaignError(f"work directory already exists: {work_dir}")
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{work_dir.name}.", dir=work_dir.parent))
    rows = []
    built_artifacts = []
    try:
        for ordinal, row in enumerate(baselines):
            artifact = _relative_artifact(stage, row["artifact"])
            recipe_scope = {
                "method": row["method"],
                "family": row["family"],
                "recipe": row["recipe"],
                "build_command": row["build_command"],
                "evaluation_command": row["evaluation_command"],
                "artifact": row["artifact"],
            }
            recipe_id = "sha256:" + hashlib.sha256(
                canonical(recipe_scope)
            ).hexdigest()
            common_env = {
                **{
                    key: os.environ[key]
                    for key in SAFE_ENVIRONMENT
                    if key in os.environ
                },
                "TRITIUM_ABLATION_ARTIFACT": str(artifact),
                "TRITIUM_ABLATION_BASELINE_INDEX": str(ordinal),
                "TRITIUM_ABLATION_METHOD": row["method"],
                "TRITIUM_ABLATION_FAMILY": row["family"],
                "TRITIUM_ABLATION_EVALUATION_ID": campaign["evaluation_id"],
                "TRITIUM_ABLATION_TARGET_BYTES": str(campaign["target_bytes"]),
                "TRITIUM_ABLATION_PARAMETER_COUNT": str(campaign["parameter_count"]),
                "TRITIUM_ABLATION_DEVICE": campaign["device"],
                "TRITIUM_ABLATION_RECIPE_ID": recipe_id,
                "TRITIUM_ABLATION_RELEASE": campaign["release"],
                "TRITIUM_ABLATION_SOURCE_REVISION": campaign["source_revision"],
                "TRITIUM_ABLATION_RUN_ID": campaign["run_id"],
                "TRITIUM_ABLATION_MODEL_ARTIFACT_ID": campaign["model_artifact_id"],
            }
            command_runner(
                _command(row["build_command"], "build command"),
                cwd=stage,
                env={**common_env, "TRITIUM_ABLATION_PHASE": "build"},
            )
            artifact = _ordinary_artifact(stage, artifact)
            artifact_bytes = artifact.stat().st_size
            artifact_sha256 = _sha256(artifact)
            built_artifacts.append((artifact, artifact_bytes, artifact_sha256))
            target_bpw = (
                campaign["target_bytes"] * 8.0 / campaign["parameter_count"]
            )
            actual_bpw = artifact_bytes * 8.0 / campaign["parameter_count"]
            if (
                artifact_bytes <= 0
                or artifact_bytes > campaign["target_bytes"]
                or abs(actual_bpw - target_bpw) > 0.05
            ):
                raise CampaignError("baseline artifact is not physical-byte matched")
            quality = []
            elapsed = []
            resident = []
            for sample_index in range(SAMPLE_COUNT):
                before = time.perf_counter_ns()
                raw = command_runner(
                    _command(row["evaluation_command"], "evaluation command"),
                    cwd=stage,
                    env={
                        **common_env,
                        "TRITIUM_ABLATION_PHASE": "evaluate",
                        "TRITIUM_ABLATION_SAMPLE_INDEX": str(sample_index),
                    },
                )
                duration_ns = max(1, time.perf_counter_ns() - before)
                if (
                    not artifact.is_file()
                    or artifact.is_symlink()
                    or artifact.stat().st_size != artifact_bytes
                    or _sha256(artifact) != artifact_sha256
                ):
                    raise CampaignError("baseline artifact drifted during evaluation")
                score, resident_bytes = _sample(
                    raw,
                    evaluation_id=campaign["evaluation_id"],
                    artifact_sha256=artifact_sha256,
                    device=campaign["device"],
                )
                quality.append(score)
                elapsed.append(duration_ns / 1_000_000.0)
                resident.append(resident_bytes)
            reference_score = quality[0]
            if any(
                not math.isclose(
                    score, reference_score, rel_tol=1e-12, abs_tol=1e-12
                )
                for score in quality[1:]
            ):
                raise CampaignError("baseline quality score drifted across samples")
            rows.append(
                {
                    "method": row["method"], "family": row["family"],
                    "recipe": row["recipe"],
                    "build_command": row["build_command"],
                    "evaluation_command": row["evaluation_command"],
                    "artifact": row["artifact"], "recipe_id": recipe_id,
                    "artifact_bytes": artifact_bytes,
                    "artifact_sha256": artifact_sha256,
                    "parameter_count": campaign["parameter_count"],
                    "quality_score": statistics.median(quality),
                    "elapsed_samples_ms": elapsed,
                    "resident_samples_bytes": resident,
                    "physical_device": campaign["device"],
                    "reproduced": True, "publishable_recipe": True,
                }
            )
        for artifact, artifact_bytes, artifact_sha256 in built_artifacts:
            if (
                not artifact.is_file()
                or artifact.is_symlink()
                or artifact.stat().st_size != artifact_bytes
                or _sha256(artifact) != artifact_sha256
            ):
                raise CampaignError("baseline artifact drifted before publication")
        target_bpw = campaign["target_bytes"] * 8.0 / campaign["parameter_count"]
        recipe_identities = [
            {
                "method": row["method"], "family": row["family"],
                "recipe_id": row["recipe_id"],
            }
            for row in rows
        ]
        set_scope = {
            "model_artifact_id": campaign["model_artifact_id"],
            "evaluation_id": campaign["evaluation_id"],
            "target_bytes": campaign["target_bytes"],
            "target_bpw": target_bpw,
            "recipes": recipe_identities,
        }
        trace = {
            "schema": TRACE_SCHEMA, "result": "pass",
            "release": campaign["release"],
            "source_revision": campaign["source_revision"],
            "run_id": campaign["run_id"],
            "environment": {**runtime, "device": campaign["device"]},
            "model_artifact_id": campaign["model_artifact_id"],
            "evaluation_id": campaign["evaluation_id"],
            "baseline_set_id": "sha256:" + hashlib.sha256(
                canonical(set_scope)
            ).hexdigest(),
            "target_bytes": campaign["target_bytes"],
            "target_bpw": target_bpw,
            "baselines": rows,
        }
        os.replace(stage, work_dir)
        try:
            _write_atomic(output, trace)
        except Exception:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        return trace
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = execute(
        args.campaign.absolute(),
        output=args.output.absolute(),
        work_dir=args.work_dir.absolute(),
    )
    print(json.dumps(trace, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
