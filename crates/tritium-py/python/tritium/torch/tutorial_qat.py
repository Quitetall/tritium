"""Source-free installed-wheel PyTorch QAT tutorial."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import tempfile
from pathlib import Path

import torch
from torch import nn

import tritium

from .config import TernaryConfig
from .conversion import inspect, prepare
from .ptq import convert
from .qat_artifacts import export_qat_hard, load_qat_hard


_RECEIPT_FIELDS = {
    "schema",
    "receipt_id",
    "passed",
    "device",
    "seed",
    "torch_version",
    "distribution_version",
    "tritium_module",
    "loss",
    "gradient_norm",
    "converted_parameters",
    "aliases",
    "algorithm_id",
    "planes",
    "artifact_id",
    "hard_state_digest",
    "artifact_dir",
}
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


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
        distribution = importlib.metadata.distribution("tritium-torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("tutorial requires an installed tritium-torch distribution") from error
    module = Path(tritium.__file__).resolve(strict=True)
    if distribution.files is None:
        raise RuntimeError("installed tritium-torch distribution has no file inventory")
    owned = {
        distribution.locate_file(item).resolve()
        for item in distribution.files
    }
    if module not in owned:
        raise RuntimeError("imported tritium package is not owned by tritium-torch")
    return distribution.version, module


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _receipt_id(receipt: dict[str, object]) -> str:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    return "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()


def run_installed_qat_tutorial(
    output_dir: Path, *, device_name: str, seed: int = 73
) -> dict[str, object]:
    """Run one complete latent-QAT to strict-hard-artifact lifecycle."""

    if device_name not in {"cpu", "cuda:0"}:
        raise ValueError("device must be cpu or cuda:0")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    distribution_version, module_path = _installed_distribution()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA tutorial requested but torch.cuda.is_available() is false")

    torch.manual_seed(seed)
    prepared = prepare(
        _TinyTiedModel(device=device),
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Linear", "Embedding"),
            planes=2,
        ),
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

    prepared.model.eval()
    expected = prepared.model(tokens).detach()
    hard = convert(prepared)
    torch.testing.assert_close(hard.model(tokens), expected, rtol=0, atol=0)

    output_dir.mkdir(parents=True)
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
    receipt: dict[str, object] = {
        "schema": "tritium.installed-qat-tutorial.v1",
        "passed": True,
        "device": device_name,
        "seed": seed,
        "torch_version": torch.__version__,
        "distribution_version": distribution_version,
        "tritium_module": str(module_path),
        "loss": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "converted_parameters": coverage.converted_parameters,
        "aliases": list(weight.aliases),
        "algorithm_id": weight.algorithm_id,
        "planes": weight.planes,
        "artifact_id": artifact.artifact_id,
        "hard_state_digest": artifact.hard_state_digest,
        "artifact_dir": str(artifact.artifact_dir.resolve()),
    }
    receipt["receipt_id"] = _receipt_id(receipt)
    return receipt


def validate_tutorial_receipt(
    receipt_path: Path, *, expected_device: str
) -> dict[str, object]:
    """Strictly validate one tutorial result and its hard artifact."""

    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("tutorial receipt must be a regular non-symlink file")
    if receipt_path.stat().st_size > 1024 * 1024:
        raise ValueError("tutorial receipt exceeds metadata size limit")
    try:
        receipt = json.loads(receipt_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("tutorial receipt must contain UTF-8 JSON") from error
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("tutorial receipt fields do not match schema version 1")
    if receipt["schema"] != "tritium.installed-qat-tutorial.v1":
        raise ValueError("unsupported tutorial receipt schema")
    if receipt["passed"] is not True or receipt["device"] != expected_device:
        raise ValueError("tutorial receipt result or device mismatch")
    if type(receipt["seed"]) is not int:
        raise ValueError("tutorial seed must be an integer")
    for field in ("loss", "gradient_norm"):
        value = receipt[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"tutorial {field} must be finite and positive")
    if receipt["converted_parameters"] != 1:
        raise ValueError("tutorial converted-parameter coverage mismatch")
    if receipt["aliases"] != ["embed.weight", "head.weight"]:
        raise ValueError("tutorial tied aliases mismatch")
    if receipt["algorithm_id"] != "tritium.additive-2/tritium.salt-ste@1":
        raise ValueError("tutorial estimator identity mismatch")
    if receipt["planes"] != 2:
        raise ValueError("tutorial plane count mismatch")
    for field in ("artifact_id", "hard_state_digest", "receipt_id"):
        if not isinstance(receipt[field], str) or _DIGEST.fullmatch(receipt[field]) is None:
            raise ValueError(f"tutorial {field} is not a canonical digest")
    if receipt["receipt_id"] != _receipt_id(receipt):
        raise ValueError("tutorial receipt identity mismatch")
    version, module_path = _installed_distribution()
    if receipt["distribution_version"] != version:
        raise ValueError("tutorial distribution version mismatch")
    if receipt["tritium_module"] != str(module_path):
        raise ValueError("tutorial package path mismatch")
    artifact_dir = Path(str(receipt["artifact_dir"]))
    if not artifact_dir.is_absolute():
        raise ValueError("tutorial artifact path must be absolute")
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
    args = parser.parse_args()
    if args.check_receipt is not None:
        receipt = validate_tutorial_receipt(
            args.check_receipt.absolute(), expected_device=args.device
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    assert args.output_dir is not None
    output = args.output_dir.absolute()
    receipt = run_installed_qat_tutorial(
        output, device_name=args.device, seed=args.seed
    )
    _write_receipt(output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_installed_qat_tutorial", "validate_tutorial_receipt"]
