"""Source-free installed-wheel whole-Qwen ONNX qualification worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import torch

from .. import QwenModel
from .config import TernaryConfig
from .conversion import prepare_qat
from .onnx import export_onnx, load_onnx


SCHEMA = "tritium.onnx-inference-execution.v1"
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
OPERATORS = (
    "TritiumTernaryMpGemm",
    "TritiumSaltV2MpGemm",
    "TritiumSaltV2Embedding",
    "TritiumKvAttention",
    "TritiumQwenDeltaNet",
)
PROMPTS = ((1, 2, 3), (17, 29))
TOLERANCE = 1.0e-3


class OnnxQualificationError(RuntimeError):
    """The installed candidate cannot produce admissible ONNX evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ordinary(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise OnnxQualificationError(f"{label} must be an ordinary nonempty file")
    return path.resolve(strict=True)


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise OnnxQualificationError(f"{label} must be an ordinary directory")
    return path.resolve(strict=True)


def _artifact(value: Mapping[str, Any], kind: str, label: str) -> dict[str, Any]:
    fields = {"id", "kind", "name", "bytes", "sha256"}
    if set(value) != fields or value.get("kind") != kind:
        raise OnnxQualificationError(f"{label} identity differs")
    if (
        not isinstance(value.get("id"), str)
        or not value["id"]
        or not isinstance(value.get("name"), str)
        or not value["name"]
        or type(value.get("bytes")) is not int
        or value["bytes"] <= 0
        or not isinstance(value.get("sha256"), str)
        or len(value["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in value["sha256"])
    ):
        raise OnnxQualificationError(f"{label} identity is malformed")
    return dict(value)


def _last_row(values: Sequence[float], width: int, label: str) -> Sequence[float]:
    if width <= 0 or len(values) < width or len(values) % width:
        raise OnnxQualificationError(f"{label} has invalid flattened geometry")
    return values[-width:]


def _maximum_error(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise OnnxQualificationError("parity vectors differ in shape")
    left_tensor = torch.tensor(left, dtype=torch.float32)
    right_tensor = torch.tensor(right, dtype=torch.float32)
    if not bool(
        torch.isfinite(left_tensor).all() and torch.isfinite(right_tensor).all()
    ):
        raise OnnxQualificationError("parity vectors contain non-finite values")
    return float(torch.max(torch.abs(left_tensor - right_tensor)).item())


def _states_equal(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> bool:
    return len(left) == len(right) and all(
        tuple(a.shape) == tuple(b.shape) and torch.equal(a, b)
        for a, b in zip(left, right, strict=True)
    )


def _greedy(values: Sequence[float]) -> int:
    if not values:
        raise OnnxQualificationError("cannot select a token from empty logits")
    return max(range(len(values)), key=values.__getitem__)


def _case(
    kind: str,
    case_id: str,
    error: float,
    *,
    tokens_exact: bool,
    states_exact: bool,
    output_exact: bool,
) -> dict[str, Any]:
    if not math.isfinite(error) or error < 0:
        raise OnnxQualificationError("parity error must be finite and nonnegative")
    return {
        "kind": kind,
        "case_id": case_id,
        "max_abs_error": error,
        "tolerance": TOLERANCE,
        "token_ids_exact": tokens_exact,
        "states_exact": states_exact,
        "output_exact": output_exact,
    }


def _require_mtp_oracle(native: Any) -> Any:
    oracle = getattr(native, "reference_mtp", None)
    if getattr(native, "mtp_verified", False) is not True or not callable(oracle):
        raise OnnxQualificationError(
            "production Qwen MTP oracle is not promoted; whole-model ONNX evidence is blocked"
        )
    return oracle


def _language_cases(native: Any, ort_model: Any) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for ordinal, raw_prompt in enumerate(PROMPTS):
        prompt = list(raw_prompt)
        reference = native.reference_language([prompt], len(prompt) + 8)[0]
        observed = ort_model(torch.tensor([prompt], dtype=torch.int64))
        replay = ort_model(torch.tensor([prompt], dtype=torch.int64))
        logits = observed.logits[0, -1].tolist()
        error = _maximum_error(reference.last_logits, logits)
        tokens_exact = list(reference.token_ids) == prompt
        states_exact = _states_equal(observed.past_key_values, replay.past_key_values)
        output_exact = _greedy(reference.last_logits) == _greedy(logits)
        cases.append(
            _case(
                "prompt",
                f"prompt-{ordinal}",
                error,
                tokens_exact=tokens_exact,
                states_exact=states_exact,
                output_exact=output_exact,
            )
        )

        next_token = _greedy(reference.last_logits)
        native_steps = native.reference_language(
            [prompt, [next_token]], len(prompt) + 8
        )
        continuation = ort_model(
            torch.tensor([[next_token]], dtype=torch.int64),
            past_key_values=observed.past_key_values,
        )
        replay_prompt = ort_model(torch.tensor([prompt], dtype=torch.int64))
        replay_continuation = ort_model(
            torch.tensor([[next_token]], dtype=torch.int64),
            past_key_values=replay_prompt.past_key_values,
        )
        continued_logits = continuation.logits[0, -1].tolist()
        continued_error = _maximum_error(native_steps[-1].last_logits, continued_logits)
        cases.append(
            _case(
                "cached-decode",
                f"cached-decode-{ordinal}",
                continued_error,
                tokens_exact=list(native_steps[-1].token_ids) == [next_token],
                states_exact=_states_equal(
                    continuation.past_key_values,
                    replay_continuation.past_key_values,
                ),
                output_exact=(
                    _greedy(native_steps[-1].last_logits) == _greedy(continued_logits)
                ),
            )
        )

        native_tokens = list(native.generate(prompt, 4))
        generated = ort_model.generate(
            torch.tensor([prompt], dtype=torch.int64), max_new_tokens=4
        )[0].tolist()
        onnx_tokens = generated[len(prompt) :]
        cases.append(
            _case(
                "generation",
                f"generation-{ordinal}",
                max(error, continued_error),
                tokens_exact=native_tokens == onnx_tokens,
                states_exact=True,
                output_exact=native_tokens == onnx_tokens,
            )
        )
    return cases


def _mtp_cases(native: Any, ort_model: Any) -> list[dict[str, Any]]:
    oracle = _require_mtp_oracle(native)
    cases: list[dict[str, Any]] = []
    for ordinal, raw_prompt in enumerate(PROMPTS):
        prompt = list(raw_prompt)
        language = native.reference_language([prompt], len(prompt) + 8)[0]
        sampled = _greedy(language.last_logits)
        references = oracle([prompt], [sampled], len(prompt) + 8)
        if len(references) != 1:
            raise OnnxQualificationError(
                "native MTP oracle returned the wrong transaction count"
            )
        reference = references[0]
        shifted = torch.tensor([list(reference.shifted_input_ids)], dtype=torch.int64)
        hidden = torch.tensor(reference.target_hidden_states, dtype=torch.float32).view(
            len(reference.shifted_input_ids), reference.hidden_size
        )
        observed = ort_model.draft(shifted, hidden)
        replay = ort_model.draft(shifted, hidden)
        logits = observed.logits[0, -1].tolist()
        final_hidden = observed.final_hidden.reshape(-1).tolist()
        error = max(
            _maximum_error(reference.last_logits, logits),
            _maximum_error(reference.final_hidden_states, final_hidden),
        )
        cases.append(
            _case(
                "mtp",
                f"mtp-{ordinal}",
                error,
                tokens_exact=list(reference.shifted_input_ids) == shifted[0].tolist(),
                states_exact=_states_equal(
                    observed.past_key_values, replay.past_key_values
                ),
                output_exact=_greedy(reference.last_logits) == _greedy(logits),
            )
        )
    return cases


def _inspect_graphs(bundle: Path) -> tuple[int, list[int], int]:
    try:
        import onnx
    except ImportError as error:  # pragma: no cover - release environment gate
        raise OnnxQualificationError(
            "onnx is absent from qualification environment"
        ) from error
    standard: set[int] = set()
    tritium: set[int] = set()
    dense = 0
    float_types = {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.BFLOAT16,
    }
    for name in ("language.onnx", "mtp.onnx"):
        graph = onnx.load(bundle / name, load_external_data=False)
        for imported in graph.opset_import:
            if imported.domain in ("", "ai.onnx"):
                standard.add(int(imported.version))
            elif imported.domain == "com.tritium":
                tritium.add(int(imported.version))
        for initializer in graph.graph.initializer:
            elements = math.prod(initializer.dims)
            if (
                initializer.data_type in float_types
                and len(initializer.dims) >= 2
                and elements >= 1_048_576
            ):
                dense += 1
    if len(standard) != 1:
        raise OnnxQualificationError("language/MTP standard opsets differ")
    return standard.pop(), sorted(tritium), dense


def _physical_cpu_id() -> str:
    identity = [platform.machine(), platform.processor()]
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(("vendor_id", "model name")):
                identity.append(line.partition(":")[2].strip())
    value = "|".join(item for item in identity if item)
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _error_code(error: BaseException) -> str:
    code = getattr(error, "code", None)
    return code if isinstance(code, str) and code else type(error).__name__


def _copy_bundle(
    source: Path, destination: Path, *, weights: bytes | None = None
) -> None:
    destination.mkdir()
    for name in ("language.onnx", "mtp.onnx", "tritium-onnx-manifest.json"):
        shutil.copyfile(source / name, destination / name)
    if weights is None:
        os.link(source / "weights.bin", destination / "weights.bin")
    else:
        (destination / "weights.bin").write_bytes(weights)


def _rejected(
    kind: str,
    operation: Any,
    *,
    codes: Sequence[str] = (),
    message_tokens: Sequence[str] = (),
) -> dict[str, Any]:
    try:
        operation()
    except BaseException as error:
        code = _error_code(error)
        message = str(error).lower()
        if codes and code not in codes:
            raise OnnxQualificationError(
                f"{kind} fault returned unexpected error code {code}"
            ) from error
        if message_tokens and not any(token in message for token in message_tokens):
            raise OnnxQualificationError(
                f"{kind} fault was rejected for an unrelated reason"
            ) from error
        return {"kind": kind, "rejected": True, "error_code": code}
    raise OnnxQualificationError(f"{kind} fault was accepted")


def _unknown_operator_fault() -> None:
    import onnx
    import onnxruntime

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "UnknownQualificationOp", ["x"], ["y"], domain="com.tritium"
            )
        ],
        "unknown-operator",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 21),
            onnx.helper.make_opsetid("com.tritium", 2),
        ],
    )
    onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def _faults(bundle: Path) -> list[dict[str, Any]]:
    # Keep the fault workspace on the bundle filesystem so the potentially
    # tens-of-gigabytes external arena can be hard-linked, never copied.
    with tempfile.TemporaryDirectory(
        prefix=".tritium-onnx-faults-", dir=bundle.parent
    ) as raw:
        root = Path(raw)
        graph = root / "graph"
        _copy_bundle(bundle, graph)
        language = graph / "language.onnx"
        with language.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            value = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(bytes([value[0] ^ 0x01]))

        weights = root / "weights"
        _copy_bundle(bundle, weights, weights=b"corrupt")

        traversal = root / "traversal"
        _copy_bundle(bundle, traversal)
        manifest = json.loads((traversal / "tritium-onnx-manifest.json").read_bytes())
        manifest["language"]["file"] = "../language.onnx"
        (traversal / "tritium-onnx-manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

        checkpoint = root / "trainable-import"
        checkpoint.mkdir()
        (checkpoint / "optimizer.pt").write_bytes(b"training state")

        return [
            _rejected(
                "graph-corruption",
                lambda: load_onnx(graph),
                message_tokens=("digest", "hash"),
            ),
            _rejected(
                "weights-corruption",
                lambda: load_onnx(weights),
                message_tokens=("digest", "hash", "length", "bytes"),
            ),
            _rejected(
                "path-traversal",
                lambda: load_onnx(traversal),
                message_tokens=("path", "unsafe", "canonical", "file"),
            ),
            _rejected(
                "unknown-operator",
                _unknown_operator_fault,
                message_tokens=("unknownqualificationop",),
            ),
            _rejected(
                "trainable-export",
                lambda: export_onnx(
                    prepare_qat(torch.nn.Linear(4, 4), TernaryConfig.qat()),
                    root / "trainable-export",
                ),
                codes=("trainable_onnx_requires_v1_3",),
            ),
            _rejected(
                "trainable-import",
                lambda: load_onnx(checkpoint),
                codes=("trainable_onnx_requires_v1_3",),
            ),
        ]


def _source_free_environment() -> tuple[bool, bool]:
    roots = (Path.cwd(), Path(__file__).resolve())
    repository_absent = not any(
        (parent / ".git").exists() for root in roots for parent in (root, *root.parents)
    )
    compilers = ("cc", "gcc", "clang", "rustc", "cargo", "nvcc")
    compiler_absent = all(shutil.which(name) is None for name in compilers)
    return repository_absent, compiler_absent


def run(
    *,
    wheel: Path,
    wheel_record: Mapping[str, Any],
    artifact_record: Mapping[str, Any],
    model_record: Mapping[str, Any],
    model_artifact_id: str,
    onnx_bundle: Path,
    native_bundle: Path,
    profile: str,
    conversion_mode: str,
    source_revision: str,
    release: str,
    run_id: str,
    candidate_manifest_sha256: str,
) -> dict[str, Any]:
    wheel = _ordinary(wheel, "candidate wheel")
    onnx_bundle = _directory(onnx_bundle, "ONNX bundle")
    native_bundle = _directory(native_bundle, "native bundle")
    wheel_identity = _artifact(wheel_record, "python-wheel", "wheel")
    artifact_identity = _artifact(artifact_record, "onnx-bundle", "ONNX artifact")
    model_identity = _artifact(model_record, "model-bundle", "native model artifact")
    if model_identity["id"] != model_artifact_id:
        raise OnnxQualificationError("native model artifact ID differs")
    if (
        wheel_identity["name"] != wheel.name
        or wheel_identity["bytes"] != wheel.stat().st_size
        or wheel_identity["sha256"] != _sha256(wheel)
    ):
        raise OnnxQualificationError("installed wheel differs from candidate identity")
    if profile not in {"compact-v1", "near-lossless-v1"}:
        raise OnnxQualificationError("unsupported qualification profile")
    if conversion_mode not in {"ptq", "refined"}:
        raise OnnxQualificationError("unsupported conversion mode")
    if not model_artifact_id or not run_id or not release:
        raise OnnxQualificationError("qualification identity is incomplete")
    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise OnnxQualificationError("source revision must be 40 lowercase hexadecimal")
    if len(candidate_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in candidate_manifest_sha256
    ):
        raise OnnxQualificationError("candidate manifest digest is malformed")

    native = QwenModel.load(str(native_bundle), profile=profile, device="cpu")
    _require_mtp_oracle(native)
    ort_model = load_onnx(onnx_bundle, device="cpu")
    native_receipt = native.receipt
    onnx_receipt = ort_model._runtime.receipt
    if (
        native_receipt.declared_source_model_id != MODEL_ID
        or native_receipt.source_revision != MODEL_REVISION
        or native_receipt.package_id != onnx_receipt.package_id
        or native.profile != profile
        or onnx_receipt.conversion_mode != conversion_mode
    ):
        raise OnnxQualificationError("native and ONNX package lineage differs")

    cases = _language_cases(native, ort_model)
    cases.extend(_mtp_cases(native, ort_model))
    if any(
        case["max_abs_error"] > case["tolerance"]
        or not all(
            case[field] for field in ("token_ids_exact", "states_exact", "output_exact")
        )
        for case in cases
    ):
        raise OnnxQualificationError("whole-model ONNX parity failed")

    standard_opset, tritium_opsets, dense = _inspect_graphs(onnx_bundle)
    counts = ort_model.operator_counts()
    if set(counts) != set(OPERATORS) or any(counts[name] <= 0 for name in OPERATORS):
        raise OnnxQualificationError("required custom operators did not execute")
    manifest = json.loads((onnx_bundle / "tritium-onnx-manifest.json").read_bytes())
    weights = _ordinary(onnx_bundle / "weights.bin", "ONNX external data")
    repository_absent, compiler_absent = _source_free_environment()
    try:
        import onnx
        import onnxruntime
    except ImportError as error:  # pragma: no cover - release environment gate
        raise OnnxQualificationError(
            "ONNX qualification dependencies are absent"
        ) from error
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "tritium_distribution": importlib.metadata.version("pytritium"),
        "repository_absent": repository_absent,
        "compiler_absent": compiler_absent,
    }
    if not repository_absent or not compiler_absent:
        raise OnnxQualificationError("worker is not source/compiler-free")
    return {
        "schema": SCHEMA,
        "result": "pass",
        "release": release,
        "source_revision": source_revision,
        "run_id": run_id,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "wheel": wheel_identity,
        "artifact": artifact_identity,
        "model_artifact_id": model_artifact_id,
        "environment": environment,
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "scope": "language+mtp",
            "profile": profile,
            "conversion_mode": conversion_mode,
            "package_id": "sha256:" + model_identity["sha256"],
        },
        "session": {
            "provider": "CPUExecutionProvider",
            "physical_cpu_id": _physical_cpu_id(),
            "bundle_schema": manifest["schema"],
            "sequence_mode": manifest["sequence_mode"],
            "standard_opset": standard_opset,
            "tritium_opsets": tritium_opsets,
            "custom_operator_calls": [
                {"operator": name, "calls": counts[name]} for name in OPERATORS
            ],
            "external_data_files": [
                {
                    "file": "weights.bin",
                    "bytes": weights.stat().st_size,
                    "sha256": _sha256(weights),
                    "authenticated": True,
                }
            ],
            "dense_weight_initializers": dense,
            "persistent_dense_shadows": 0,
        },
        "cases": cases,
        "faults": _faults(onnx_bundle),
    }


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
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
    parser.add_argument("--wheel-record", type=Path, required=True)
    parser.add_argument("--artifact-record", type=Path, required=True)
    parser.add_argument("--model-record", type=Path, required=True)
    parser.add_argument("--model-artifact-id", required=True)
    parser.add_argument("--onnx-bundle", type=Path, required=True)
    parser.add_argument("--native-bundle", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--conversion-mode", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(
        wheel=args.wheel,
        wheel_record=json.loads(args.wheel_record.read_bytes()),
        artifact_record=json.loads(args.artifact_record.read_bytes()),
        model_record=json.loads(args.model_record.read_bytes()),
        model_artifact_id=args.model_artifact_id,
        onnx_bundle=args.onnx_bundle,
        native_bundle=args.native_bundle,
        profile=args.profile,
        conversion_mode=args.conversion_mode,
        source_revision=args.source_revision,
        release=args.release,
        run_id=args.run_id,
        candidate_manifest_sha256=args.candidate_manifest_sha256,
    )
    _write_atomic(args.output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
