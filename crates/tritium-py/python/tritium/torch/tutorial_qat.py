"""Source-free installed-wheel PyTorch QAT tutorial."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import tempfile
from pathlib import Path

import torch
from torch import nn

import tritium

from .config import TernaryConfig
from .conversion import inspect, prepare
from .ptq import convert
from .qat_artifacts import export_qat_hard, load_qat_hard
from .tutorial_receipt import (
    SCHEMA as RECEIPT_SCHEMA,
    receipt_id,
    tree_identity,
    validate_receipt,
)


class _TinyTiedModel(nn.Module):
    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8, device=device)
        self.head = nn.Linear(8, 16, bias=False, device=device)
        self.head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(tokens))


def _installed_distribution() -> tuple[str, Path]:
    try:
        distribution = importlib.metadata.distribution("pytritium")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("tutorial requires an installed pytritium distribution") from error
    module = Path(tritium.__file__).resolve(strict=True)
    if distribution.files is None:
        raise RuntimeError("installed pytritium distribution has no file inventory")
    owned = {
        distribution.locate_file(item).resolve()
        for item in distribution.files
    }
    if module not in owned:
        raise RuntimeError("imported tritium package is not owned by pytritium")
    return distribution.version, module


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def run_installed_qat_tutorial(
    output_dir: Path,
    *,
    device_name: str,
    wheel_artifact: Path,
    source_revision: str,
    release: str,
    run_id: str,
    seed: int = 73,
) -> dict[str, object]:
    """Run one complete latent-QAT to strict-hard-artifact lifecycle."""

    if device_name not in {"cpu", "cuda:0"}:
        raise ValueError("device must be cpu or cuda:0")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if wheel_artifact.is_symlink() or not wheel_artifact.is_file():
        raise ValueError("wheel artifact must be an ordinary file")
    wheel_artifact = wheel_artifact.resolve(strict=True)
    if not wheel_artifact.name.endswith(".whl"):
        raise ValueError("wheel artifact must have a .whl filename")
    if (
        len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise ValueError("source revision must be 40 lowercase hexadecimal characters")
    if not release or not run_id:
        raise ValueError("release and run id must be non-empty")
    distribution_version, module_path = _installed_distribution()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA tutorial requested but torch.cuda.is_available() is false")

    torch.manual_seed(seed)
    config = TernaryConfig.qat(
        estimator="salt-ste",
        target_modules=("Linear", "Embedding"),
        planes=2,
    )
    prepared = prepare(
        _TinyTiedModel(device=device),
        config,
        inplace=True,
    )
    coverage = inspect(prepared.model)
    if coverage.converted_parameters != 1:
        raise RuntimeError("tutorial must convert one deduplicated tied parameter")
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64, device=device)
    optimizer = torch.optim.AdamW(prepared.model.parameters(), lr=1e-3)
    loss = prepared.model(tokens).square().mean()
    loss.backward()
    gradients = [
        parameter.grad.detach().float().norm().square()
        for parameter in prepared.model.parameters()
        if parameter.grad is not None
    ]
    gradient_norm = math.sqrt(float(torch.stack(gradients).sum()))
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError("QAT backward produced no finite nonzero gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    from safetensors.torch import load_model, save_model

    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "latent-checkpoint"
    checkpoint.mkdir()
    model_checkpoint = checkpoint / "model.safetensors"
    optimizer_checkpoint = checkpoint / "optimizer.pt"
    save_model(
        prepared.model,
        model_checkpoint,
        metadata={"format": "pt", "tritium_mode": "qat-latent"},
    )
    torch.save(optimizer.state_dict(), optimizer_checkpoint)

    resumed = prepare(_TinyTiedModel(device=device), config, inplace=True)
    missing, unexpected = load_model(resumed.model, model_checkpoint, strict=True)
    if missing or unexpected:
        raise RuntimeError("latent checkpoint did not strictly restore model state")
    if resumed.model.embed.weight is not resumed.model.head.weight:
        raise RuntimeError("latent checkpoint restore lost tied master identity")
    resumed_optimizer = torch.optim.AdamW(resumed.model.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(
        torch.load(optimizer_checkpoint, map_location=device, weights_only=True)
    )
    optimizer_state_entries = len(resumed_optimizer.state)
    if optimizer_state_entries != 1:
        raise RuntimeError("latent checkpoint restored unexpected optimizer state")
    prepared.model.eval()
    resumed.model.eval()
    torch.testing.assert_close(resumed.model(tokens), prepared.model(tokens), rtol=0, atol=0)

    prepared.model.train()
    resumed.model.train()
    prepared.model(tokens).square().mean().backward()
    resumed.model(tokens).square().mean().backward()
    optimizer.step()
    resumed_optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    resumed_optimizer.zero_grad(set_to_none=True)
    expected_state = prepared.model.state_dict()
    observed_state = resumed.model.state_dict()
    if expected_state.keys() != observed_state.keys() or any(
        not torch.equal(expected_state[name], observed_state[name])
        for name in expected_state
    ):
        raise RuntimeError("resumed optimizer step diverged from uninterrupted QAT")

    resumed.model.eval()
    expected = resumed.model(tokens).detach()
    hard = convert(resumed)
    torch.testing.assert_close(hard.model(tokens), expected, rtol=0, atol=0)

    artifact = export_qat_hard(hard, output_dir / "qat-hard")
    if load_qat_hard(artifact.artifact_dir) != artifact:
        raise RuntimeError("strict QAT-hard artifact reload changed its identity")
    loaded = load_qat_hard(
        artifact.artifact_dir, _TinyTiedModel(device=device), inplace=True
    )
    if loaded.embed.packed_weight is not loaded.head.packed_weight:
        raise RuntimeError("strict reload lost tied packed-weight identity")
    torch.testing.assert_close(loaded(tokens), expected, rtol=0, atol=0)

    if len(hard.weights) != 1:
        raise RuntimeError("hard conversion must contain one deduplicated weight")
    weight = hard.weights[0]
    if weight.aliases != ("embed.weight", "head.weight"):
        raise RuntimeError("hard conversion lost tied weight aliases")
    if weight.algorithm_id != "tritium.additive-2/tritium.salt-ste@1":
        raise RuntimeError("hard conversion used unexpected estimator identity")
    if weight.planes != 2:
        raise RuntimeError("hard conversion did not preserve two additive planes")
    hard_tree = tree_identity(artifact.artifact_dir)
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "passed": True,
        "device": device_name,
        "seed": seed,
        "torch_version": torch.__version__,
        "distribution_version": distribution_version,
        "tritium_module": str(module_path),
        "source_revision": source_revision,
        "release": release,
        "run_id": run_id,
        "wheel_name": wheel_artifact.name,
        "wheel_bytes": wheel_artifact.stat().st_size,
        "wheel_sha256": _sha256_file(wheel_artifact),
        "loss": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "converted_parameters": coverage.converted_parameters,
        "aliases": list(weight.aliases),
        "algorithm_id": weight.algorithm_id,
        "planes": weight.planes,
        "artifact_id": artifact.artifact_id,
        "hard_state_digest": artifact.hard_state_digest,
        "artifact_dir": "qat-hard",
        "hard_artifact_bytes": hard_tree["bytes"],
        "hard_artifact_file_count": hard_tree["file_count"],
        "hard_artifact_tree_sha256": hard_tree["sha256"],
        "checkpoint_model_bytes": model_checkpoint.stat().st_size,
        "checkpoint_model_sha256": _sha256_file(model_checkpoint),
        "checkpoint_optimizer_bytes": optimizer_checkpoint.stat().st_size,
        "checkpoint_optimizer_sha256": _sha256_file(optimizer_checkpoint),
        "optimizer_state_entries": optimizer_state_entries,
        "resume_steps": 1,
    }
    receipt["receipt_id"] = receipt_id(receipt)
    return receipt


def validate_tutorial_receipt(
    receipt_path: Path,
    *,
    expected_device: str,
    expected_wheel: Path | None = None,
    expected_source_revision: str | None = None,
    expected_release: str | None = None,
) -> dict[str, object]:
    """Strictly validate one tutorial result and its hard artifact."""
    receipt = validate_receipt(
        receipt_path,
        expected_device=expected_device,
        expected_wheel=expected_wheel,
        expected_source_revision=expected_source_revision,
        expected_release=expected_release,
    )
    version, module_path = _installed_distribution()
    if receipt["distribution_version"] != version:
        raise ValueError("tutorial distribution version mismatch")
    if receipt["tritium_module"] != str(module_path):
        raise ValueError("tutorial package path mismatch")
    artifact_dir = receipt_path.parent.resolve() / str(receipt["artifact_dir"])
    artifact = load_qat_hard(artifact_dir)
    if (
        artifact.artifact_id != receipt["artifact_id"]
        or artifact.hard_state_digest != receipt["hard_state_digest"]
    ):
        raise ValueError("tutorial hard artifact identity mismatch")
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
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--check-receipt", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--wheel-artifact", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--release")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.check_receipt is not None:
        receipt = validate_tutorial_receipt(
            args.check_receipt.absolute(),
            expected_device=args.device,
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
    assert args.output_dir is not None
    if None in (args.wheel_artifact, args.source_revision, args.release, args.run_id):
        parser.error(
            "--wheel-artifact, --source-revision, --release and --run-id are required "
            "with --output-dir"
        )
    output = args.output_dir.absolute()
    receipt = run_installed_qat_tutorial(
        output,
        device_name=args.device,
        wheel_artifact=args.wheel_artifact.absolute(),
        source_revision=args.source_revision,
        release=args.release,
        run_id=args.run_id,
        seed=args.seed,
    )
    _write_receipt(output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_installed_qat_tutorial", "validate_tutorial_receipt"]
