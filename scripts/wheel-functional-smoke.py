#!/usr/bin/env python3
"""Run differentiable Tritium smoke from an installed binary wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import unquote, urlparse


SCHEMA = "tritium.wheel-functional-smoke.v1"
RECEIPT_SCHEMA = "tritium.wheel-functional-qualification.v1"
RECEIPT_FIELDS = {
    "schema", "receipt_id", "release", "run_id", "started_at_utc",
    "duration_ms", "machine", "artifact", "evidence", "result",
}
MACHINE_FIELDS = {"machine_id", "system", "architecture"}
ARTIFACT_FIELDS = {"kind", "name", "bytes", "sha256"}
EVIDENCE_FIELDS = {
    "schema", "source_revision", "passed", "wheel", "wheel_sha256",
    "distribution_version", "python_version", "torch_version",
    "transformers_version", "safetensors_version", "native_device",
    "compiled_backends", "tritium_module", "converted_parameters", "operations",
}
REQUIRED_OPERATIONS = frozenset({
    "native_ternary_matmul", "qat_forward_backward", "optimizer_step",
    "optimizer_checkpoint_resume", "hf_safetensors_save_reload",
    "tied_weight_identity",
})
MAX_RECEIPT_BYTES = 1024 * 1024


class SmokeError(RuntimeError):
    """Installed wheel failed functional qualification."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_wheel(path: Path) -> Path:
    if path.is_dir():
        wheels = sorted(path.glob("*.whl"))
        if len(wheels) != 1:
            raise SmokeError(f"wheel directory must contain exactly one wheel; found {len(wheels)}")
        path = wheels[0]
    if path.is_symlink() or not path.is_file():
        raise SmokeError("wheel must be a regular non-symlink file")
    return path.resolve()


def require_installed(
    module_path: Path, forbidden_root: Path, environment_root: Path
) -> None:
    module = module_path.resolve(strict=True)
    forbidden = forbidden_root.resolve(strict=True)
    if module == forbidden or forbidden in module.parents:
        raise SmokeError(f"tritium imported from forbidden source checkout: {module}")
    environment = environment_root.resolve(strict=True)
    if environment != module and environment not in module.parents:
        raise SmokeError(f"tritium imported from outside smoke environment: {module}")


def validate_direct_url(document: object, wheel: Path, digest: str) -> None:
    if not isinstance(document, dict) or set(document) != {"archive_info", "url"}:
        raise SmokeError("installed distribution lacks canonical wheel direct_url metadata")
    archive = document["archive_info"]
    if not isinstance(archive, dict):
        raise SmokeError("installed distribution direct_url archive_info is invalid")
    hashes = archive.get("hashes")
    if not isinstance(hashes, dict) or hashes.get("sha256") != digest:
        raise SmokeError("installed distribution digest does not match candidate wheel")
    parsed = urlparse(document["url"] if isinstance(document["url"], str) else "")
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise SmokeError("installed distribution does not reference a local wheel")
    installed_from = Path(unquote(parsed.path)).resolve(strict=True)
    if installed_from != wheel:
        raise SmokeError("installed distribution does not reference the candidate wheel path")


def installed_distribution_identity(
    wheel: Path, digest: str
) -> tuple[str, frozenset[Path]]:
    distribution = importlib.metadata.distribution("tritium-torch")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        raise SmokeError("installed distribution lacks direct_url.json")
    try:
        document = json.loads(direct_url)
    except json.JSONDecodeError as error:
        raise SmokeError("installed distribution has invalid direct_url.json") from error
    validate_direct_url(document, wheel, digest)
    match = re.fullmatch(r"tritium_torch-([^-]+)-cp39-abi3-[^-]+\.whl", wheel.name)
    if match is None or match.group(1) != distribution.version:
        raise SmokeError("installed distribution version does not match candidate wheel")
    if distribution.files is None:
        raise SmokeError("installed distribution has no file inventory")
    files = frozenset(distribution.locate_file(item).resolve() for item in distribution.files)
    return distribution.version, files


def require_distribution_file(path: Path, files: frozenset[Path]) -> None:
    resolved = path.resolve(strict=True)
    if resolved not in files:
        raise SmokeError(f"imported Tritium file is not owned by candidate distribution: {resolved}")


def validate_native_result(value: object) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], list)
        or len(value[0]) != 1
        or not isinstance(value[0][0], float)
        or not math.isfinite(value[0][0])
        or not math.isclose(value[0][0], -126.0 / 127.0, rel_tol=0.0, abs_tol=1e-7)
    ):
        raise SmokeError(f"native ternary kernel returned invalid result: {value!r}")


def require_native_source_identity(native: object, revision: str) -> None:
    """Reject wheels whose compiled extension was built from another revision."""

    expected = f"source-git:{revision}"
    source_identity = getattr(native, "source_identity", None)
    if not callable(source_identity):
        raise SmokeError("candidate wheel does not expose native source identity")
    try:
        observed = source_identity()
    except Exception as error:  # noqa: BLE001 - preserve a typed smoke failure
        raise SmokeError("candidate wheel native source identity probe failed") from error
    if observed != expected:
        raise SmokeError(
            "candidate wheel native source identity mismatch: "
            f"expected {expected}, got {observed!r}"
        )


def _tiny_llama(transformers):
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    return transformers.LlamaForCausalLM(config)


def run_smoke(
    wheel: Path, forbidden_root: Path, revision: str, device: str = "cpu"
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise SmokeError("source revision must be a full lowercase Git object ID")
    wheel = resolve_wheel(wheel)
    wheel_digest = _sha256(wheel)
    distribution_version, distribution_files = installed_distribution_identity(
        wheel, wheel_digest
    )
    package_spec = importlib.util.find_spec("tritium")
    if package_spec is None or package_spec.origin is None:
        raise SmokeError("candidate distribution does not provide tritium package")
    require_distribution_file(Path(package_spec.origin), distribution_files)

    import torch
    import transformers
    import safetensors
    import tritium
    from tritium.nn import TernaryEmbedding, TernaryLinear
    from tritium.torch import TernaryConfig, inspect, prepare_qat

    environment_root = Path(os.sys.prefix)
    require_installed(Path(tritium.__file__), forbidden_root, environment_root)
    require_installed(Path(tritium._tritium.__file__), forbidden_root, environment_root)
    require_distribution_file(Path(tritium.__file__), distribution_files)
    require_distribution_file(Path(tritium._tritium.__file__), distribution_files)
    require_native_source_identity(tritium._tritium, revision)
    if device not in {"cpu", "cuda:0"}:
        raise SmokeError("device must be cpu or cuda:0")
    backend = device.split(":", 1)[0]
    if backend not in tritium.compiled_backends():
        raise SmokeError(f"candidate wheel does not contain requested {backend} backend")
    native = tritium.ternary_matmul(
        [[1.0, 2.0]], [[1, -1]], 1.0, device=device
    )
    validate_native_result(native)

    torch.manual_seed(73)
    recipe = TernaryConfig.qat(
        estimator="salt-ste", target_modules=("Linear", "Embedding"), planes=1
    )
    model = prepare_qat(_tiny_llama(transformers), recipe)
    if not isinstance(model.model.embed_tokens, TernaryEmbedding):
        raise SmokeError("Hugging Face embedding was not ternarized")
    if not isinstance(model.lm_head, TernaryLinear):
        raise SmokeError("Hugging Face LM head was not ternarized")
    if model.model.embed_tokens.weight is not model.lm_head.weight:
        raise SmokeError("tied embedding/head identity was not preserved")
    coverage = inspect(model)
    if coverage.converted_parameters <= 0:
        raise SmokeError("QAT coverage contains no converted parameters")

    tokens = torch.tensor([[1, 2, 3, 4]])
    tracked = model.model.embed_tokens.weight
    before = tracked.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    loss = model(input_ids=tokens, labels=tokens).loss
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise SmokeError("QAT backward did not produce finite gradients")
    if not any(torch.count_nonzero(gradient).item() for gradient in gradients):
        raise SmokeError("QAT backward produced only zero gradients")
    if (
        tracked.grad is None
        or not torch.isfinite(tracked.grad).all().item()
        or torch.count_nonzero(tracked.grad).item() == 0
    ):
        raise SmokeError("ternary tied weight did not receive a finite nonzero STE gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if torch.equal(before, tracked.detach()):
        raise SmokeError("optimizer step did not update latent master weights")
    if not math.isfinite(float(loss.detach())):
        raise SmokeError("QAT loss is not finite")

    model.eval()
    expected = model(input_ids=tokens).logits.detach()
    with tempfile.TemporaryDirectory(prefix="tritium-hf-smoke-") as raw:
        checkpoint = Path(raw)
        model.save_pretrained(checkpoint, safe_serialization=True)
        optimizer_path = checkpoint / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
        if not (checkpoint / "model.safetensors").is_file():
            raise SmokeError("Hugging Face checkpoint did not use safetensors")
        restored = transformers.AutoModelForCausalLM.from_pretrained(checkpoint)
        restored.eval()
        if not isinstance(restored.model.embed_tokens, TernaryEmbedding):
            raise SmokeError("reloaded embedding lost Tritium type")
        if not isinstance(restored.lm_head, TernaryLinear):
            raise SmokeError("reloaded LM head lost Tritium type")
        if restored.model.embed_tokens.weight is not restored.lm_head.weight:
            raise SmokeError("reloaded checkpoint lost tied-weight identity")
        observed = restored(input_ids=tokens).logits.detach()
        if not torch.equal(observed, expected):
            raise SmokeError("Hugging Face checkpoint changed QAT logits")
        restored_optimizer = torch.optim.AdamW(
            restored.parameters(), lr=1e-4, weight_decay=0.0
        )
        restored_optimizer.load_state_dict(
            torch.load(optimizer_path, map_location="cpu", weights_only=True)
        )
        if not restored_optimizer.state:
            raise SmokeError("optimizer checkpoint restored no parameter state")
        resume_before = restored.model.embed_tokens.weight.detach().clone()
        resume_loss = restored(input_ids=tokens, labels=tokens).loss
        resume_loss.backward()
        restored_optimizer.step()
        if torch.equal(resume_before, restored.model.embed_tokens.weight.detach()):
            raise SmokeError("resumed optimizer step did not update latent master weights")

    return {
        "schema": SCHEMA,
        "source_revision": revision,
        "passed": True,
        "wheel": wheel.name,
        "wheel_sha256": wheel_digest,
        "distribution_version": distribution_version,
        "python_version": ".".join(map(str, os.sys.version_info[:3])),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "safetensors_version": safetensors.__version__,
        "native_device": device,
        "compiled_backends": tritium.compiled_backends(),
        "tritium_module": str(Path(tritium.__file__).resolve()),
        "converted_parameters": coverage.converted_parameters,
        "operations": [
            "native_ternary_matmul",
            "qat_forward_backward",
            "optimizer_step",
            "optimizer_checkpoint_resume",
            "hf_safetensors_save_reload",
            "tied_weight_identity",
        ],
    }


def _atomic_write(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_receipt(
    evidence: dict[str, object], wheel: Path, release: str, run_id: str,
    started_at_utc: str, duration_ms: float,
) -> dict[str, object]:
    if re.fullmatch(r"1\.1\.0-rc\.(0|[1-9][0-9]*)", release) is None:
        raise SmokeError("release must be a canonical v1.1 candidate")
    if not run_id:
        raise SmokeError("run id must be non-empty")
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise SmokeError("qualification duration must be finite and positive")
    wheel = resolve_wheel(wheel)
    machine_material = {
        "node": platform.node(),
        "system": platform.system(),
        "architecture": platform.machine(),
    }
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "release": release,
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "duration_ms": duration_ms,
        "machine": {
            "machine_id": "sha256:" + hashlib.sha256(_canonical(machine_material)).hexdigest(),
            "system": platform.system(),
            "architecture": platform.machine(),
        },
        "artifact": {
            "kind": "python-wheel",
            "name": wheel.name,
            "bytes": wheel.stat().st_size,
            "sha256": _sha256(wheel),
        },
        "evidence": evidence,
        "result": "pass",
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(_canonical(receipt)).hexdigest()
    return receipt


def validate_receipt(
    path: Path, source_revision: str, release: str, wheel: Path,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise SmokeError("functional receipt must be a regular non-symlink file")
    if path.stat().st_size > MAX_RECEIPT_BYTES:
        raise SmokeError("functional receipt exceeds metadata size limit")
    try:
        receipt = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SmokeError("functional receipt must contain UTF-8 JSON") from error
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_FIELDS:
        raise SmokeError("functional receipt fields do not match frozen schema")
    if receipt["schema"] != RECEIPT_SCHEMA or receipt["result"] != "pass":
        raise SmokeError("functional receipt is not passed qualification evidence")
    if receipt["release"] != release or not isinstance(receipt["run_id"], str) or not receipt["run_id"]:
        raise SmokeError("functional receipt release or run identity mismatch")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", str(receipt["started_at_utc"])) is None:
        raise SmokeError("functional receipt timestamp is not canonical UTC")
    duration = receipt["duration_ms"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or duration <= 0:
        raise SmokeError("functional receipt duration is invalid")
    machine = receipt["machine"]
    if not isinstance(machine, dict) or set(machine) != MACHINE_FIELDS or re.fullmatch(
        r"sha256:[0-9a-f]{64}", str(machine.get("machine_id", ""))
    ) is None or any(not isinstance(machine[field], str) or not machine[field] for field in ("system", "architecture")):
        raise SmokeError("functional receipt machine identity is invalid")
    wheel = resolve_wheel(wheel)
    artifact = receipt["artifact"]
    expected_artifact = {
        "kind": "python-wheel", "name": wheel.name,
        "bytes": wheel.stat().st_size, "sha256": _sha256(wheel),
    }
    if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_FIELDS or artifact != expected_artifact:
        raise SmokeError("functional receipt does not bind exact candidate wheel")
    evidence = receipt["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != EVIDENCE_FIELDS:
        raise SmokeError("functional evidence fields do not match frozen schema")
    if evidence["schema"] != SCHEMA or evidence["passed"] is not True:
        raise SmokeError("functional evidence did not pass")
    if evidence["source_revision"] != source_revision:
        raise SmokeError("functional evidence source revision mismatch")
    if evidence["wheel"] != wheel.name or evidence["wheel_sha256"] != expected_artifact["sha256"]:
        raise SmokeError("functional evidence wheel identity mismatch")
    expected_version = release.replace("1.1.0-rc.", "1.1.0rc")
    if evidence["distribution_version"] != expected_version:
        raise SmokeError("functional evidence package version mismatch")
    for field in (
        "python_version", "torch_version", "transformers_version",
        "safetensors_version", "tritium_module",
    ):
        if not isinstance(evidence[field], str) or not evidence[field]:
            raise SmokeError(f"functional evidence {field} is invalid")
    if not Path(evidence["tritium_module"]).is_absolute():
        raise SmokeError("functional evidence module path is not absolute")
    compiled = evidence["compiled_backends"]
    if (
        not isinstance(compiled, list)
        or any(not isinstance(backend, str) or not backend for backend in compiled)
        or len(set(compiled)) != len(compiled)
    ):
        raise SmokeError("functional evidence backend inventory is invalid")
    device = evidence["native_device"]
    if device not in {"cpu", "cuda:0"} or device.split(":", 1)[0] not in compiled:
        raise SmokeError("functional evidence device was not compiled into wheel")
    operations = evidence["operations"]
    if (
        not isinstance(operations, list)
        or any(not isinstance(operation, str) for operation in operations)
        or len(operations) != len(REQUIRED_OPERATIONS)
        or set(operations) != REQUIRED_OPERATIONS
    ):
        raise SmokeError("functional evidence operation coverage is incomplete")
    if type(evidence["converted_parameters"]) is not int or evidence["converted_parameters"] <= 0:
        raise SmokeError("functional evidence converted no parameters")
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    expected_id = "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()
    if receipt_id != expected_id:
        raise SmokeError("functional receipt identity mismatch")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--forbidden-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        started = time.monotonic()
        evidence = run_smoke(
            args.wheel, args.forbidden_root, args.source_revision, args.device
        )
        receipt = build_receipt(
            evidence, args.wheel, args.release, args.run_id,
            started_at_utc, (time.monotonic() - started) * 1000.0,
        )
        _atomic_write(args.output, receipt)
        validate_receipt(
            args.output, args.source_revision, args.release, resolve_wheel(args.wheel)
        )
    except (OSError, SmokeError) as error:
        parser.error(str(error))
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
