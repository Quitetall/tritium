#!/usr/bin/env python3
"""Capture pinned Qwen3.6 S2KF records from the verified calibration pack.

Without ``--execute`` this performs source/pack/replay preflight only. Loading
the 27B checkpoint and executing the capture requires the explicit flag.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import runpy
import subprocess
import sys
from typing import Any
from urllib.parse import unquote, urlparse


REPLAY = runpy.run_path(Path(__file__).with_name("qwen36_calibration_replay.py"))
PINNED_REVISION = REPLAY["_VERIFIER"]["REVISION"]
STAGE7 = runpy.run_path(
    Path(__file__).with_name("verify-stage7-qualification-receipt.py")
)
WHEEL_RUNTIME = runpy.run_path(Path(__file__).with_name("wheel-functional-smoke.py"))


def _parse_max_memory(values: list[str]) -> dict[Any, str]:
    result: dict[Any, str] = {}
    for value in values:
        device, separator, limit = value.partition("=")
        if not separator or not device or not limit:
            raise ValueError("--max-memory must use DEVICE=LIMIT syntax")
        key: Any = int(device) if device.isdecimal() else device
        if key in result:
            raise ValueError(f"--max-memory repeats device {device!r}")
        result[key] = limit
    return result


def _validate_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"{label} must be 64 hexadecimal characters")


def _validate_capture_recipe(args: argparse.Namespace) -> None:
    if args.curvature is None:
        raise ValueError("--execute requires --curvature")
    if args.damping is None:
        raise ValueError("--execute requires --damping")
    if args.activation_cache_digest is None:
        raise ValueError("--execute requires --activation-cache-digest")
    _validate_digest(args.activation_cache_digest, "activation-cache digest")
    if args.curvature == "guided-fisher" and args.guided_loss_reduction is None:
        raise ValueError("guided-fisher requires --guided-loss-reduction")
    if args.curvature != "guided-fisher" and args.guided_loss_reduction is not None:
        raise ValueError("--guided-loss-reduction is valid only for guided-fisher")
    if args.max_shared_modules <= 0 or min(
        args.max_evidence_bytes,
        args.max_batch_bytes,
        args.max_capture_bytes,
        args.max_objective_bytes,
    ) <= 0:
        raise ValueError("capture limits must be positive")
    if not math.isfinite(args.damping) or args.damping < 0:
        raise ValueError("damping must be finite and nonnegative")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _candidate_wheel(
    candidate_path: Path, candidate: dict[str, Any], installed_wheel: Path
) -> tuple[Path, str]:
    artifacts = candidate.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("release candidate has no artifact inventory")
    installed_wheel = installed_wheel.resolve(strict=True)
    matches: list[tuple[Path, str]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("kind") != "python-wheel":
            continue
        logical_text = artifact.get("path")
        if not isinstance(logical_text, str):
            raise ValueError("candidate Python wheel path is invalid")
        logical = PurePosixPath(logical_text)
        if (
            logical.is_absolute()
            or not logical.parts
            or ".." in logical.parts
            or "\\" in logical_text
            or logical.as_posix() != logical_text
        ):
            raise ValueError("candidate Python wheel path is unsafe")
        path = candidate_path.parent
        for part in logical.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError("candidate Python wheel path traverses a symlink")
        if path.resolve(strict=False) != installed_wheel:
            continue
        if not path.is_file():
            raise ValueError("candidate Python wheel artifact is missing")
        identity = artifact.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("candidate Python wheel identity is invalid")
        expected_bytes = identity.get("bytes")
        expected_digest = identity.get("sha256")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
        ):
            raise ValueError("candidate Python wheel identity is incomplete")
        if path.stat().st_size != expected_bytes:
            raise ValueError("installed Python wheel size differs from candidate")
        actual_digest = WHEEL_RUNTIME["_sha256"](path)
        if actual_digest != expected_digest:
            raise ValueError("installed Python wheel digest differs from candidate")
        matches.append((path.resolve(strict=True), actual_digest))
    if len(matches) != 1:
        raise ValueError("installed Tritium wheel is not a unique candidate artifact")
    return matches[0]


def _validate_capture_python_environment(
    candidate_path: Path,
    candidate: dict[str, Any],
    revision: str,
) -> Any:
    try:
        distribution = importlib.metadata.distribution("pytritium")
        direct_url = distribution.read_text("direct_url.json")
        if direct_url is None:
            raise ValueError("installed pytritium wheel has no direct_url identity")
        document = json.loads(
            direct_url,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(document, dict) or not isinstance(document.get("url"), str):
            raise ValueError("installed pytritium direct_url identity is malformed")
        parsed = urlparse(document["url"])
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise ValueError("installed pytritium wheel is not from a local candidate")
        installed_wheel = Path(unquote(parsed.path))
        wheel, wheel_digest = _candidate_wheel(
            candidate_path, candidate, installed_wheel
        )
        _version, distribution_files = WHEEL_RUNTIME[
            "installed_distribution_identity"
        ](wheel, wheel_digest)

        package_spec = importlib.util.find_spec("tritium")
        if package_spec is None or package_spec.origin is None:
            raise ValueError("candidate wheel does not provide the tritium package")
        WHEEL_RUNTIME["require_distribution_file"](
            Path(package_spec.origin), distribution_files
        )
        tritium = importlib.import_module("tritium")
        qwen36 = importlib.import_module("tritium.torch.qwen36")
        repository = Path(__file__).resolve().parent.parent
        environment = Path(sys.prefix)
        for module in (tritium, tritium._tritium, qwen36):
            WHEEL_RUNTIME["require_installed"](
                Path(module.__file__), repository, environment
            )
            WHEEL_RUNTIME["require_distribution_file"](
                Path(module.__file__), distribution_files
            )
        WHEEL_RUNTIME["require_native_source_identity"](tritium._tritium, revision)
        return qwen36
    except Exception as error:
        raise ValueError(f"capture Python environment is not candidate-bound: {error}") from error


def _validate_stage7_qualification(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if args.stage7_qualification_receipt is None:
        raise ValueError("--execute requires --stage7-qualification-receipt")
    if args.release_candidate_manifest is None:
        raise ValueError("--execute requires --release-candidate-manifest")

    candidate_path = args.release_candidate_manifest
    try:
        STAGE7["_ordinary_candidate"](candidate_path)
        raw = candidate_path.read_bytes()
        candidate = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"release candidate manifest is invalid: {error}") from error
    if (
        not isinstance(candidate, dict)
        or candidate.get("schema") != "tritium.release-candidate.v1"
    ):
        raise ValueError("release candidate manifest schema mismatch")
    release = candidate.get("release")
    revision = candidate.get("source_revision")
    if not isinstance(release, str) or not release:
        raise ValueError("release candidate manifest has no release identity")
    if not isinstance(revision, str) or len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("release candidate manifest has an invalid source revision")
    try:
        checkout_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot determine capture checkout revision") from error
    if revision != checkout_revision:
        raise ValueError(
            "release candidate source_revision differs from capture checkout HEAD"
        )
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot determine capture checkout cleanliness") from error
    if status.strip():
        raise ValueError(
            "Qwen capture requires a clean checkout; use a clean candidate worktree"
        )
    try:
        receipt = STAGE7["validate"](
            args.stage7_qualification_receipt,
            revision,
            release,
            candidate_path,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        raise ValueError(f"Stage 7 qualification receipt rejected: {error}") from error
    return candidate, revision, receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--official-source-identity", required=True, type=Path)
    parser.add_argument("--pack-receipt", required=True, type=Path)
    parser.add_argument("--replay-contract", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--capture-binding-output", required=True, type=Path)
    parser.add_argument(
        "--release-candidate-manifest",
        type=Path,
        help="candidate manifest bound by the required Stage 7 freeze receipt",
    )
    parser.add_argument(
        "--stage7-qualification-receipt",
        type=Path,
        help="candidate-bound Stage 7 recipe-freeze pass required by --execute",
    )
    parser.add_argument("--declared-revision", default=PINNED_REVISION)
    parser.add_argument(
        "--activation-cache-digest",
        help="64-hex digest from the frozen activation-cache recipe; never inferred",
    )
    parser.add_argument(
        "--curvature",
        choices=("input-hessian", "guided-fisher", "forward-kl-kronecker"),
    )
    parser.add_argument("--damping", type=float)
    parser.add_argument(
        "--guided-loss-reduction",
        choices=("sum", "mean-attention-mask", "mean-valid-causal-labels"),
    )
    parser.add_argument("--max-shared-modules", type=int, default=8)
    parser.add_argument("--max-evidence-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-batch-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-capture-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-objective-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory", action="append", default=[])
    parser.add_argument("--offload-folder", type=Path)
    parser.add_argument("--input-device")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="load the local 27B checkpoint and begin/resume empirical capture",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        replay_module = REPLAY["Qwen36CalibrationReplay"]
        replay = replay_module.open(
            args.manifest,
            args.model_dir,
            args.official_source_identity,
            args.pack_receipt,
        )
        saved_contract = REPLAY["_VERIFIER"]["validate_replay_contract"](
            args.replay_contract
        )
        if saved_contract != replay.contract:
            raise ValueError("saved replay contract differs from verified pack inputs")
        max_memory = _parse_max_memory(args.max_memory)
        if args.declared_revision != replay.receipt["revision"]:
            raise ValueError("declared revision differs from verified Qwen source")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(
        "PREFLIGHT PASS "
        f"pack={replay.receipt['receipt_id']} "
        f"replay={replay.contract['contract_id']} "
        f"batch={replay.token_stream_digest}"
    )
    if not args.execute:
        print("NOT STARTED: model load and capture require --execute")
        return 0

    try:
        candidate, source_revision, stage7 = _validate_stage7_qualification(args)
        _validate_capture_recipe(args)
        if not args.offload_folder:
            raise ValueError("--execute requires an explicit --offload-folder")
        qwen36 = _validate_capture_python_environment(
            args.release_candidate_manifest,
            candidate,
            source_revision,
        )
        print(f"STAGE 7 PASS receipt={stage7['receipt_id']}")
    except ValueError as error:
        parser.error(str(error))
    try:
        import torch
        from transformers import AutoModelForImageTextToText

        if not torch.cuda.is_available() and not args.allow_cpu:
            raise RuntimeError("no CUDA device detected; pass --allow-cpu to opt in")
        args.offload_folder.mkdir(parents=True, exist_ok=True)
        model = AutoModelForImageTextToText.from_pretrained(
            str(args.model_dir),
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
            max_memory=max_memory or None,
            offload_folder=str(args.offload_folder),
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        qwen36.attach_qwen36_mtp(model, args.model_dir)
        embedding_weight = model.model.language_model.embed_tokens.weight
        input_device = args.input_device or str(embedding_weight.device)
        if input_device == "meta":
            raise RuntimeError(
                "input embedding is disk-offloaded; specify --input-device explicitly"
            )
        tensor_factory = REPLAY["torch_int64_tensor_factory"](input_device)
        native_receipt = qwen36.capture_qwen36_components(
            model,
            replay.data_factory(tensor_factory),
            model_dir=args.model_dir,
            declared_revision=args.declared_revision,
            work_dir=args.work_dir,
            evidence_dir=args.evidence_dir,
            curvature=args.curvature,
            activation_cache_digest=args.activation_cache_digest,
            token_stream_digest=replay.token_stream_digest,
            damping=args.damping,
            guided_loss_reduction=args.guided_loss_reduction,
            max_evidence_bytes=args.max_evidence_bytes,
            max_batch_bytes=args.max_batch_bytes,
            max_capture_bytes=args.max_capture_bytes,
            max_objective_bytes=args.max_objective_bytes,
            max_shared_modules=args.max_shared_modules,
        )
        binding = replay.capture_binding(
            native_receipt,
            max_evidence_bytes=args.max_evidence_bytes,
        )
        REPLAY["write_capture_binding"](args.capture_binding_output, binding)
    except Exception as error:
        print(f"qwen36 capture failed: {error}", file=sys.stderr)
        return 1
    print(
        f"CAPTURE RECEIPT {binding['binding_id']} "
        f"evidence_set={binding['evidence_set_digest']} records={binding['records']}"
    )
    print("Next: run scripts/verify-qwen36-capture-binding.py with the same paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
