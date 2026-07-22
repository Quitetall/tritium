"""Installed-wheel whole-model hard export/reload qualification."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import torch
import transformers

from ..nn import AdditiveTernaryEmbedding, AdditiveTernaryLinear
from .config import TernaryConfig
from .conversion import prepare
from .hf_lifecycle import _installed_distribution, _sha256_file, _tensor_sha256
from .qat import convert_qat_hard
from .qat_artifacts import export_qat_hard, load_qat_hard
from .tutorial_receipt import (
    EXPORT_SCHEMA,
    receipt_id,
    tree_identity,
    validate_export_receipt,
)


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
        pad_token_id=2,
        eos_token_id=2,
    )
    return transformers.LlamaForCausalLM(config)


def _no_dense_weight_shadows(model: torch.nn.Module) -> bool:
    converted = 0
    for module in model.modules():
        if isinstance(module, (AdditiveTernaryLinear, AdditiveTernaryEmbedding)):
            converted += 1
            if "weight" in module._parameters:
                return False
            module.packed_weight.validate_buffers()
    return converted > 0


def _run(
    stage: Path,
    *,
    wheel_artifact: Path,
    source_revision: str,
    release: str,
    run_id: str,
    seed: int,
) -> dict[str, object]:
    version, module_path = _installed_distribution()
    torch.manual_seed(seed)
    prepared = prepare(
        _tiny_llama(),
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Linear", "Embedding"),
            planes=2,
        ),
        inplace=True,
    )
    model = prepared.model
    tied_before = model.model.embed_tokens.weight is model.lm_head.weight
    if not tied_before:
        raise RuntimeError("Hugging Face prepare lost tied input/output weights")
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        use_cache=False,
    ).loss
    loss.backward()
    gradients = [
        parameter.grad.detach().float().norm().square()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    gradient_norm = math.sqrt(float(torch.stack(gradients).sum()))
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError("Hugging Face export lifecycle produced no finite gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()

    hard = convert_qat_hard(prepared)
    expected_logits = hard.model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    ).logits.detach()
    expected_tokens = hard.model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=2,
    )
    artifact = export_qat_hard(hard, stage / "qat-hard")
    metadata = load_qat_hard(artifact.artifact_dir)
    if metadata != artifact:
        raise RuntimeError("strict metadata reload changed the hard artifact")
    reloaded = load_qat_hard(artifact.artifact_dir, _tiny_llama(), inplace=True).eval()
    tied_after = (
        reloaded.model.embed_tokens.packed_weight is reloaded.lm_head.packed_weight
    )
    if not tied_after:
        raise RuntimeError("strict reload lost tied packed input/output weights")
    observed_logits = reloaded(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    ).logits.detach()
    observed_tokens = reloaded.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=2,
    )
    if not torch.equal(observed_logits, expected_logits):
        raise RuntimeError("strict reload changed exact whole-model logits")
    if not torch.equal(observed_tokens, expected_tokens):
        raise RuntimeError("strict reload changed greedy generation")
    no_shadows = _no_dense_weight_shadows(reloaded)
    if not no_shadows:
        raise RuntimeError("strict reload retained a dense converted-weight shadow")
    artifact_tree = tree_identity(artifact.artifact_dir)
    return {
        "schema": EXPORT_SCHEMA,
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
        "generated_ids": observed_tokens.tolist(),
        "initial_loss": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "optimizer_steps": 1,
        "converted_parameters": hard.source_coverage.converted_parameters,
        "planes": 2,
        "artifact_dir": "qat-hard",
        "artifact_id": artifact.artifact_id,
        "conversion_artifact_id": artifact.conversion_artifact_id,
        "source_checkpoint_digest": artifact.source_checkpoint_digest,
        "hard_state_digest": artifact.hard_state_digest,
        "state_sha256": artifact.state_digest,
        "state_bytes": artifact.state_bytes,
        "state_tensors": artifact.state_tensors,
        "artifact_bytes": artifact_tree["bytes"],
        "artifact_file_count": artifact_tree["file_count"],
        "artifact_tree_sha256": artifact_tree["sha256"],
        "tied_before_export": tied_before,
        "tied_after_reload": tied_after,
        "no_dense_weight_shadows": no_shadows,
        "logits_sha256": _tensor_sha256(observed_logits),
    }


def qualify_hf_export(
    output_dir: Path,
    *,
    wheel_artifact: Path,
    source_revision: str,
    release: str,
    run_id: str,
    seed: int = 101,
) -> dict[str, object]:
    """Publish one atomic candidate-bound whole-model export receipt."""

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
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        receipt = _run(
            stage,
            wheel_artifact=wheel_artifact,
            source_revision=source_revision,
            release=release,
            run_id=run_id,
            seed=seed,
        )
        receipt["receipt_id"] = receipt_id(receipt)
        receipt_path = stage / "receipt.json"
        receipt_path.write_bytes(
            (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
        )
        validate_hf_export_receipt(
            receipt_path,
            expected_wheel=wheel_artifact,
            expected_source_revision=source_revision,
            expected_release=release,
        )
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        if os.name != "nt":
            directories = [path for path in stage.rglob("*") if path.is_dir()]
            for path in sorted(
                directories, key=lambda item: len(item.parts), reverse=True
            ):
                directory = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            directory = os.open(stage, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        os.replace(stage, output_dir)
        if os.name != "nt":
            directory = os.open(output_dir.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return receipt
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def validate_hf_export_receipt(
    receipt_path: Path,
    *,
    expected_wheel: Path | None = None,
    expected_source_revision: str | None = None,
    expected_release: str | None = None,
) -> dict[str, object]:
    """Validate portable bytes, then replay strict reload in the installed wheel."""

    receipt = validate_export_receipt(
        receipt_path,
        expected_wheel=expected_wheel,
        expected_source_revision=expected_source_revision,
        expected_release=expected_release,
    )
    version, module_path = _installed_distribution()
    if receipt["distribution_version"] != version:
        raise ValueError("Hugging Face export distribution version mismatch")
    if receipt["tritium_module"] != str(module_path):
        raise ValueError("Hugging Face export package path mismatch")
    if receipt["torch_version"] != torch.__version__:
        raise ValueError("Hugging Face export torch version mismatch")
    if receipt["transformers_version"] != transformers.__version__:
        raise ValueError("Hugging Face export transformers version mismatch")
    artifact = load_qat_hard(receipt_path.parent / "qat-hard")
    for field, observed in (
        ("artifact_id", artifact.artifact_id),
        ("conversion_artifact_id", artifact.conversion_artifact_id),
        ("source_checkpoint_digest", artifact.source_checkpoint_digest),
        ("hard_state_digest", artifact.hard_state_digest),
        ("state_sha256", artifact.state_digest),
    ):
        if receipt[field] != observed:
            raise ValueError(f"Hugging Face export {field} mismatch")
    if (
        receipt["state_bytes"] != artifact.state_bytes
        or receipt["state_tensors"] != artifact.state_tensors
    ):
        raise ValueError("Hugging Face export state ledger mismatch")
    model = load_qat_hard(artifact.artifact_dir, _tiny_llama(), inplace=True).eval()
    if model.model.embed_tokens.packed_weight is not model.lm_head.packed_weight:
        raise ValueError("Hugging Face export replay lost tied packed weights")
    if not _no_dense_weight_shadows(model):
        raise ValueError("Hugging Face export replay retained dense weight shadows")
    input_ids = torch.tensor(receipt["input_ids"], dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    ).logits
    if _tensor_sha256(logits) != receipt["logits_sha256"]:
        raise ValueError("Hugging Face export replay changed logits")
    generated = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=2,
    )
    if generated.tolist() != receipt["generated_ids"]:
        raise ValueError("Hugging Face export replay changed generation")
    return receipt


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
        receipt = validate_hf_export_receipt(
            args.check_receipt.absolute(),
            expected_wheel=(
                args.wheel_artifact.absolute()
                if args.wheel_artifact is not None
                else None
            ),
            expected_source_revision=args.source_revision,
            expected_release=args.release,
        )
    else:
        if None in (
            args.wheel_artifact,
            args.source_revision,
            args.release,
            args.run_id,
        ):
            parser.error(
                "--wheel-artifact, --source-revision, --release and --run-id are "
                "required with --output-dir"
            )
        receipt = qualify_hf_export(
            args.output_dir.absolute(),
            wheel_artifact=args.wheel_artifact.absolute(),
            source_revision=args.source_revision,
            release=args.release,
            run_id=args.run_id,
        )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["qualify_hf_export", "validate_hf_export_receipt"]
