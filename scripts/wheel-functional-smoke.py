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
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse


SCHEMA = "tritium.wheel-functional-smoke.v1"


class SmokeError(RuntimeError):
    """Installed wheel failed functional qualification."""


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


def run_smoke(wheel: Path, forbidden_root: Path, revision: str) -> dict[str, object]:
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
    import tritium
    from tritium.nn import TernaryEmbedding, TernaryLinear
    from tritium.torch import TernaryConfig, inspect, prepare_qat

    environment_root = Path(os.sys.prefix)
    require_installed(Path(tritium.__file__), forbidden_root, environment_root)
    require_installed(Path(tritium._tritium.__file__), forbidden_root, environment_root)
    require_distribution_file(Path(tritium.__file__), distribution_files)
    require_distribution_file(Path(tritium._tritium.__file__), distribution_files)
    native = tritium.ternary_matmul([[1.0, 2.0]], [[1, -1]], 1.0)
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
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--forbidden-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        evidence = run_smoke(args.wheel, args.forbidden_root, args.source_revision)
        _atomic_write(args.output, evidence)
    except (OSError, SmokeError) as error:
        parser.error(str(error))
    print(json.dumps(evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
