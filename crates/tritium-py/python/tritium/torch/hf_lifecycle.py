"""Installed-wheel Hugging Face QAT lifecycle qualification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
from pathlib import Path

import torch
import transformers

import tritium

from .config import TernaryConfig
from .conversion import inspect, prepare_qat
from .tutorial_receipt import (
    HF_SCHEMA,
    canonical,
    receipt_id,
    tree_identity,
    validate_hf_receipt,
)


def _installed_distribution() -> tuple[str, Path]:
    try:
        distribution = importlib.metadata.distribution("tritium-torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("qualification requires installed tritium-torch") from error
    module = Path(tritium.__file__).resolve(strict=True)
    if distribution.files is None:
        raise RuntimeError("installed tritium-torch has no file inventory")
    owned = {distribution.locate_file(item).resolve() for item in distribution.files}
    if module not in owned:
        raise RuntimeError("imported tritium package is not owned by tritium-torch")
    return distribution.version, module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    descriptor = canonical({"dtype": str(value.dtype), "shape": list(value.shape)})
    digest = hashlib.sha256(descriptor)
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _tiny_llama() -> transformers.LlamaForCausalLM:
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


def _recipe_dict(model: torch.nn.Module) -> dict[str, object]:
    recipe = model.config.quantization_config
    if hasattr(recipe, "to_dict"):
        recipe = recipe.to_dict()
    if not isinstance(recipe, dict):
        raise RuntimeError("Hugging Face quantization recipe is not serializable")
    return recipe


def run_hf_lifecycle(
    output_dir: Path,
    *,
    wheel_artifact: Path,
    source_revision: str,
    release: str,
    run_id: str,
    seed: int = 97,
) -> dict[str, object]:
    """Train, save, and AutoModel-reload a tied additive-ternary Llama."""

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if wheel_artifact.is_symlink() or not wheel_artifact.is_file():
        raise ValueError("wheel artifact must be an ordinary file")
    wheel_artifact = wheel_artifact.resolve(strict=True)
    if not wheel_artifact.name.endswith(".whl"):
        raise ValueError("wheel artifact must have a .whl filename")
    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise ValueError("source revision must be 40 lowercase hexadecimal characters")
    if not release or not run_id:
        raise ValueError("release and run id must be non-empty")
    version, module_path = _installed_distribution()

    torch.manual_seed(seed)
    model = prepare_qat(
        _tiny_llama(),
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Linear", "Embedding"),
            planes=2,
        ),
    )
    tied_before = model.model.embed_tokens.weight is model.lm_head.weight
    if not tied_before:
        raise RuntimeError("Hugging Face prepare lost tied input/output weights")
    coverage = inspect(model)
    if coverage.converted_parameters <= 0:
        raise RuntimeError("Hugging Face prepare converted no parameters")
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    loss.backward()
    gradients = [
        parameter.grad.detach().float().norm().square()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    gradient_norm = math.sqrt(float(torch.stack(gradients).sum()))
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError("Hugging Face lifecycle produced no finite gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    expected = model(input_ids=input_ids, use_cache=False).logits.detach()

    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "hf-checkpoint"
    model.save_pretrained(checkpoint, safe_serialization=True)
    if (
        not (checkpoint / "model.safetensors").is_file()
        or (checkpoint / "pytorch_model.bin").exists()
    ):
        raise RuntimeError("Hugging Face checkpoint is not safe serialization")
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(checkpoint)
    reloaded.eval()
    tied_after = reloaded.model.embed_tokens.weight is reloaded.lm_head.weight
    if not tied_after:
        raise RuntimeError("AutoModel reload lost tied input/output weights")
    observed = reloaded(input_ids=input_ids, use_cache=False).logits.detach()
    if not torch.equal(observed, expected):
        raise RuntimeError("AutoModel reload changed exact logits")
    observed_coverage = inspect(reloaded)
    if observed_coverage.converted_parameters != coverage.converted_parameters:
        raise RuntimeError("AutoModel reload changed conversion coverage")
    recipe = _recipe_dict(reloaded)
    recipe_sha256 = "sha256:" + hashlib.sha256(canonical(recipe)).hexdigest()
    checkpoint_tree = tree_identity(checkpoint)
    receipt: dict[str, object] = {
        "schema": HF_SCHEMA,
        "passed": True,
        "device": "cpu",
        "seed": seed,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "distribution_version": version,
        "tritium_module": str(module_path),
        "source_revision": source_revision,
        "release": release,
        "run_id": run_id,
        "wheel_name": wheel_artifact.name,
        "wheel_bytes": wheel_artifact.stat().st_size,
        "wheel_sha256": _sha256_file(wheel_artifact),
        "input_ids": input_ids.tolist(),
        "initial_loss": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "optimizer_steps": 1,
        "converted_parameters": coverage.converted_parameters,
        "recipe": recipe,
        "recipe_sha256": recipe_sha256,
        "tied_before_save": tied_before,
        "tied_after_reload": tied_after,
        "safe_serialization": True,
        "checkpoint_dir": "hf-checkpoint",
        "checkpoint_bytes": checkpoint_tree["bytes"],
        "checkpoint_file_count": checkpoint_tree["file_count"],
        "checkpoint_tree_sha256": checkpoint_tree["sha256"],
        "logits_sha256": _tensor_sha256(observed),
    }
    receipt["receipt_id"] = receipt_id(receipt)
    return receipt


def validate_hf_lifecycle_receipt(
    receipt_path: Path,
    *,
    expected_wheel: Path | None = None,
    expected_source_revision: str | None = None,
    expected_release: str | None = None,
) -> dict[str, object]:
    """Validate portable bytes, then replay the installed AutoModel seam."""

    receipt = validate_hf_receipt(
        receipt_path,
        expected_wheel=expected_wheel,
        expected_source_revision=expected_source_revision,
        expected_release=expected_release,
    )
    version, module_path = _installed_distribution()
    if receipt["distribution_version"] != version:
        raise ValueError("Hugging Face lifecycle distribution version mismatch")
    if receipt["tritium_module"] != str(module_path):
        raise ValueError("Hugging Face lifecycle package path mismatch")
    if receipt["torch_version"] != torch.__version__:
        raise ValueError("Hugging Face lifecycle torch version mismatch")
    if receipt["transformers_version"] != transformers.__version__:
        raise ValueError("Hugging Face lifecycle transformers version mismatch")
    checkpoint = receipt_path.parent.resolve() / str(receipt["checkpoint_dir"])
    model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint)
    model.eval()
    if model.model.embed_tokens.weight is not model.lm_head.weight:
        raise ValueError("Hugging Face lifecycle replay lost tied weights")
    if inspect(model).converted_parameters != receipt["converted_parameters"]:
        raise ValueError("Hugging Face lifecycle replay changed coverage")
    recipe_sha256 = (
        "sha256:" + hashlib.sha256(canonical(_recipe_dict(model))).hexdigest()
    )
    if recipe_sha256 != receipt["recipe_sha256"]:
        raise ValueError("Hugging Face lifecycle replay changed recipe")
    input_ids = torch.tensor(receipt["input_ids"], dtype=torch.int64)
    logits = model(input_ids=input_ids, use_cache=False).logits
    if _tensor_sha256(logits) != receipt["logits_sha256"]:
        raise ValueError("Hugging Face lifecycle replay changed logits")
    return receipt


def _write_receipt(output_dir: Path, receipt: dict[str, object]) -> Path:
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".receipt.", suffix=".json", dir=output_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        destination = output_dir / "receipt.json"
        os.replace(temporary, destination)
        if os.name != "nt":
            directory = os.open(output_dir, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--check-receipt", type=Path)
    parser.add_argument("--wheel-artifact", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--release")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.check_receipt is not None:
        receipt = validate_hf_lifecycle_receipt(
            args.check_receipt.absolute(),
            expected_wheel=(
                args.wheel_artifact.absolute()
                if args.wheel_artifact is not None
                else None
            ),
            expected_source_revision=args.source_revision,
            expected_release=args.release,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if None in (args.wheel_artifact, args.source_revision, args.release, args.run_id):
        parser.error(
            "--wheel-artifact, --source-revision, --release and --run-id are required "
            "with --output-dir"
        )
    receipt = run_hf_lifecycle(
        args.output_dir.absolute(),
        wheel_artifact=args.wheel_artifact.absolute(),
        source_revision=args.source_revision,
        release=args.release,
        run_id=args.run_id,
    )
    _write_receipt(args.output_dir.absolute(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_hf_lifecycle", "validate_hf_lifecycle_receipt"]
