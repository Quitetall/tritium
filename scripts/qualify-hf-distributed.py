#!/usr/bin/env python3
"""Run and atomically publish the two-GPU Hugging Face qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile
import time
from typing import Any
import venv


VERIFIER = runpy.run_path(Path(__file__).with_name("verify-hf-distributed-receipt.py"))
validate_receipt = VERIFIER["validate"]
validate_mode = VERIFIER["_validate_mode"]
canonical = VERIFIER["canonical"]
SCHEMA = VERIFIER["SCHEMA"]
FRAGMENT_FIELDS = {
    "schema",
    "model_config_sha256",
    "model_parameters",
    "machine",
    "environment",
    "devices",
    "mode",
}
MACHINE_FIELDS = {"system", "architecture"}
ENVIRONMENT_FIELDS = VERIFIER["ENVIRONMENT_FIELDS"]
DEVICE_FIELDS = VERIFIER["DEVICE_FIELDS"]


class QualificationError(RuntimeError):
    """The hardware run or fragment aggregation is not admissible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def object_fields(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise QualificationError(f"{label} fields do not match the frozen schema")
    return value


def load_fragment(path: Path, expected_mode: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"{expected_mode} fragment is not an ordinary file")
    try:
        fragment = object_fields(
            json.loads(path.read_bytes()), FRAGMENT_FIELDS, f"{expected_mode} fragment"
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(
            f"{expected_mode} fragment is not UTF-8 JSON"
        ) from error
    if fragment["schema"] != "tritium.hf-distributed-mode.v1":
        raise QualificationError(f"{expected_mode} fragment schema mismatch")
    mode = validate_mode(fragment["mode"])
    if mode["name"] != expected_mode:
        raise QualificationError(f"expected {expected_mode} fragment")
    object_fields(fragment["machine"], MACHINE_FIELDS, f"{expected_mode}.machine")
    object_fields(
        fragment["environment"], ENVIRONMENT_FIELDS, f"{expected_mode}.environment"
    )
    devices = fragment["devices"]
    if not isinstance(devices, list) or len(devices) != 2:
        raise QualificationError(f"{expected_mode} must report two devices")
    for rank, device in enumerate(devices):
        object_fields(device, DEVICE_FIELDS, f"{expected_mode}.devices[{rank}]")
    return fragment


def assemble_receipt(
    *,
    stage: Path,
    artifact: Path,
    source_revision: str,
    release: str,
    run_id: str,
    started_at_utc: str,
    duration_ms: float,
    fragments: list[dict[str, Any]],
    checkpoint_files: dict[tuple[str, int], Path],
) -> dict[str, Any]:
    """Assemble one receipt and copy exact rank checkpoint support bytes."""

    if len(fragments) != 2 or [item["mode"]["name"] for item in fragments] != [
        "ddp",
        "fsdp",
    ]:
        raise QualificationError("fragments must be exactly ordered ddp,fsdp")
    reference = fragments[0]
    for fragment in fragments[1:]:
        for field in (
            "model_config_sha256",
            "model_parameters",
            "machine",
            "environment",
            "devices",
        ):
            if fragment[field] != reference[field]:
                raise QualificationError(f"distributed fragments disagree on {field}")
    support_dir = stage / "support"
    support_dir.mkdir()
    support = []
    for mode in ("ddp", "fsdp"):
        for rank in (0, 1):
            source = checkpoint_files.get((mode, rank))
            if source is None or source.is_symlink() or not source.is_file():
                raise QualificationError(f"{mode} rank {rank} checkpoint is missing")
            destination = support_dir / f"{mode}-rank-{rank}.checkpoint"
            shutil.copyfile(source, destination)
            fsync_file(destination)
            support.append(
                {
                    "mode": mode,
                    "rank": rank,
                    "path": destination.relative_to(stage).as_posix(),
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
    modes = [fragment["mode"] for fragment in fragments]
    for mode in modes:
        support_digests = [
            "sha256:" + item["sha256"]
            for item in support
            if item["mode"] == mode["name"]
        ]
        if mode["rank_checkpoint_sha256"] != support_digests:
            raise QualificationError(
                f"{mode['name']} fragment checkpoint digests differ from support bytes"
            )
    machine_material = {
        **reference["machine"],
        "device_uuids": [item["uuid"] for item in reference["devices"]],
    }
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "source_revision": source_revision,
        "release": release,
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "duration_ms": duration_ms,
        "source_dirty": False,
        "command_contract": "torchrun-nproc2-ddp-then-fsdp-v1",
        "artifact": {
            "kind": "python-wheel",
            "name": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        },
        "model_config_sha256": reference["model_config_sha256"],
        "model_parameters": reference["model_parameters"],
        "machine": {
            "machine_id": "sha256:"
            + hashlib.sha256(canonical(machine_material)).hexdigest(),
            **reference["machine"],
        },
        "environment": reference["environment"],
        "world_size": 2,
        "devices": reference["devices"],
        "modes": modes,
        "support_artifacts": support,
        "result": "pass",
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
    return receipt


def require_clean_revision(repo: Path, revision: str) -> None:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=30
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            text=True,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise QualificationError("cannot verify source revision") from error
    if head != revision or dirty:
        raise QualificationError(
            "qualification requires the exact clean source revision"
        )


def run_qualification(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(__file__).resolve().parent.parent
    require_clean_revision(repo, args.source_revision)
    if args.artifact.is_symlink() or not args.artifact.is_file():
        raise QualificationError("artifact must be one ordinary wheel")
    artifact = args.artifact.resolve(strict=True)
    if artifact.suffix != ".whl":
        raise QualificationError("artifact must be one ordinary wheel")
    requested_output = args.output_dir.absolute()
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    if requested_output.parent.is_symlink():
        raise QualificationError("output parent must not be a symlink")
    output = requested_output.parent.resolve(strict=True) / requested_output.name
    if output.exists() or output.is_symlink():
        raise QualificationError("output directory must not exist")
    worker = repo / "crates/tritium-py/tests/hf_multi_gpu_worker.py"
    if worker.is_symlink() or not worker.is_file():
        raise QualificationError("frozen distributed worker is unavailable")
    worker = worker.resolve(strict=True)
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.monotonic()
    stage = Path(tempfile.mkdtemp(prefix=".hf-distributed-", dir=output.parent))
    try:
        runtime = stage / "runtime"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(runtime)
        runtime_python = runtime / "bin/python"
        install = subprocess.run(
            [
                str(runtime_python),
                "-I",
                "-m",
                "pip",
                "install",
                "--isolated",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                "--only-binary=:all:",
                str(artifact),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if install.returncode != 0:
            raise QualificationError(
                f"exact wheel install failed:\n{install.stdout}\n{install.stderr}"
            )
        fragments = []
        checkpoints: dict[tuple[str, int], Path] = {}
        for mode in ("ddp", "fsdp"):
            fragment_path = stage / f"{mode}.json"
            checkpoint = stage / f"{mode}-checkpoint"
            command = [
                str(runtime_python),
                "-I",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(worker),
                "--mode",
                mode,
                "--output",
                str(fragment_path),
                "--checkpoint",
                str(checkpoint),
            ]
            completed = subprocess.run(
                command,
                cwd=repo,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            if completed.returncode != 0:
                raise QualificationError(
                    f"{mode} worker failed:\n{completed.stdout}\n{completed.stderr}"
                )
            fragments.append(load_fragment(fragment_path, mode))
            if mode == "ddp":
                for rank in (0, 1):
                    checkpoints[(mode, rank)] = checkpoint / f"rank-{rank}.pt"
            else:
                shards = sorted((checkpoint / "dcp").glob("*.distcp"))
                if len(shards) != 2:
                    raise QualificationError("FSDP did not emit two checkpoint shards")
                for rank, shard in enumerate(shards):
                    checkpoints[(mode, rank)] = shard
        receipt = assemble_receipt(
            stage=stage,
            artifact=artifact,
            source_revision=args.source_revision,
            release=args.release,
            run_id=args.run_id,
            started_at_utc=started_at_utc,
            duration_ms=(time.monotonic() - started) * 1000.0,
            fragments=fragments,
            checkpoint_files=checkpoints,
        )
        receipt_path = stage / "receipt.json"
        receipt_path.write_bytes(canonical(receipt) + b"\n")
        fsync_file(receipt_path)
        validate_receipt(receipt_path, args.source_revision, args.release, artifact)
        for disposable in (stage / "ddp.json", stage / "fsdp.json"):
            disposable.unlink()
        shutil.rmtree(stage / "ddp-checkpoint")
        shutil.rmtree(stage / "fsdp-checkpoint")
        shutil.rmtree(runtime)
        fsync_directory(stage / "support")
        fsync_directory(stage)
        os.replace(stage, output)
        fsync_directory(output.parent)
        return receipt
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=7200.0)
    args = parser.parse_args()
    receipt = run_qualification(args)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
