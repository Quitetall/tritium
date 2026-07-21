"""Receipt-backed five-minute SmolLM2 release tutorial."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Union

import torch

from .. import _tritium
from ..nn import AdditiveTernaryWeight
from .config import TernaryConfig
from .conversion import prepare, prepare_qat
from .module_onnx import export_module_onnx, load_module_onnx
from .ptq import calibrate, convert, load_quantized_module

Pathish = Union[str, os.PathLike[str]]
SMOLLM2_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMOLLM2_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
_SCHEMA = "tritium.smollm2-five-minute.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _state_bytes(model: torch.nn.Module) -> int:
    return sum(value.numel() * value.element_size() for value in model.state_dict().values())


def _trit_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    seen = set()
    planes = []
    for module in model.modules():
        if not isinstance(module, AdditiveTernaryWeight) or id(module) in seen:
            continue
        seen.add(id(module))
        planes.extend(module.trit_counts())
    negative = sum(counts[0] for counts in planes)
    zero = sum(counts[1] for counts in planes)
    positive = sum(counts[2] for counts in planes)
    total = negative + zero + positive
    if total == 0:
        raise RuntimeError("compact model exposed no ternary values")
    return {
        "negative": negative,
        "zero": zero,
        "positive": positive,
        "zero_rate": zero / total,
        "planes": len(planes),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_smollm2_release_demo(
    output_dir: Pathish,
    *,
    model_id: str = SMOLLM2_MODEL_ID,
    revision: str = SMOLLM2_REVISION,
    device: str = "cuda",
    max_seconds: float = 300.0,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Run PTQ, QAT, checkpoint, generation and real-ORT gates on pinned SmolLM2."""

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    if len(revision) != 40 or any(value not in "0123456789abcdef" for value in revision):
        raise ValueError("revision must be a full lowercase Git object ID")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA tutorial requested but torch.cuda.is_available() is false")
    if device == "cuda" and "cuda" not in _tritium.compiled_backends():
        raise RuntimeError("CUDA tutorial requires a CUDA-enabled Tritium wheel")
    target = Path(output_dir).absolute()
    target.mkdir(parents=True, exist_ok=False)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("SmolLM2 tutorial requires transformers") from error

    # First model/tokenizer download is deliberately outside measured wall time.
    source = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float32,
        local_files_only=local_files_only,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
    )
    batch = tokenizer("Ternary models make efficient inference", return_tensors="pt")
    tokens = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    started = time.monotonic()

    recipe = TernaryConfig.ptq(
        profile="compact-v1", target_modules=("Linear", "Embedding")
    )
    prepared = prepare(source, recipe, inplace=True)
    calibration = calibrate(
        prepared,
        [{"input_ids": tokens, "attention_mask": attention_mask, "use_cache": False}],
        evidence_dir=target / "calibration",
    )
    conversion = convert(
        prepared,
        calibration,
        work_dir=target / "conversion",
        max_working_bytes=256 * 1024 * 1024,
    )
    compact = load_quantized_module(prepared.model, conversion, inplace=True).eval()
    physical_bytes = _state_bytes(compact)
    diagnostics = _trit_diagnostics(compact)

    native_dir = target / "native-hf"
    compact.save_pretrained(native_dir, safe_serialization=True)
    tokenizer.save_pretrained(native_dir)
    restored = AutoModelForCausalLM.from_pretrained(native_dir).eval()
    restored.config.use_cache = False
    with torch.no_grad():
        expected = compact(
            input_ids=tokens, attention_mask=attention_mask, use_cache=False
        ).logits
        observed = restored(
            input_ids=tokens, attention_mask=attention_mask, use_cache=False
        ).logits
    torch.testing.assert_close(observed, expected)
    generated = restored.generate(
        tokens,
        attention_mask=attention_mask,
        max_new_tokens=8,
        do_sample=False,
        use_cache=False,
    )

    onnx_artifact = export_module_onnx(
        restored,
        tokens,
        target / "onnx",
        input_names=("input_ids",),
        output_names=("logits",),
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
    )
    replay = torch.cat((tokens, tokens[:, :2]), dim=1)
    with torch.no_grad():
        replay_expected = restored(input_ids=replay, use_cache=False).logits
    replay_observed = load_module_onnx(onnx_artifact.artifact_dir)(replay)
    torch.testing.assert_close(
        replay_observed, replay_expected, rtol=1e-4, atol=1e-5
    )

    del source, prepared, compact, restored
    qat_source = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float32,
        local_files_only=True,
    )
    qat = prepare_qat(
        qat_source,
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Linear", "Embedding"),
            planes=1,
        ),
    ).to(device)
    qat_tokens = tokens.to(device)
    qat_mask = attention_mask.to(device)
    optimizer = torch.optim.AdamW(qat.parameters(), lr=1e-5, weight_decay=0.0)
    loss = qat(
        input_ids=qat_tokens,
        attention_mask=qat_mask,
        labels=qat_tokens,
        use_cache=False,
    ).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if not math.isfinite(float(loss.detach())):
        raise RuntimeError("SmolLM2 QAT step produced non-finite loss")
    qat_dir = target / "qat-checkpoint"
    qat.save_pretrained(qat_dir, safe_serialization=True)
    optimizer_path = qat_dir / "optimizer.pt"
    torch.save(optimizer.state_dict(), optimizer_path)
    del optimizer, qat
    if device == "cuda":
        torch.cuda.empty_cache()
    resumed = AutoModelForCausalLM.from_pretrained(qat_dir).to(device)
    resumed_optimizer = torch.optim.AdamW(
        resumed.parameters(), lr=1e-5, weight_decay=0.0
    )
    resumed_optimizer.load_state_dict(
        torch.load(optimizer_path, map_location="cpu", weights_only=True)
    )
    if not resumed_optimizer.state:
        raise RuntimeError("SmolLM2 QAT optimizer resumed no state")

    elapsed = time.monotonic() - started
    if elapsed >= max_seconds:
        raise RuntimeError(
            f"SmolLM2 tutorial exceeded wall-time budget: {elapsed:.3f}s >= {max_seconds:.3f}s"
        )
    selected_bytes = sum(
        entry.logical_bytes
        for entry in conversion.coverage.entries
        if entry.disposition == "selected"
    )
    receipt = {
        "schema": _SCHEMA,
        "passed": True,
        "model_id": model_id,
        "source_revision": revision,
        "distribution_version": importlib.metadata.version("tritium-torch"),
        "device": device,
        "elapsed_seconds_excluding_download": elapsed,
        "max_seconds": max_seconds,
        "coverage": {
            "selected_parameters": conversion.coverage.selected_parameters,
            "preserved_parameters": conversion.coverage.preserved_parameters,
            "total_numel": conversion.coverage.total_numel,
        },
        "storage": {
            "selected_dense_bytes": selected_bytes,
            "compact_checkpoint_bytes": physical_bytes,
            "selected_dense_to_checkpoint_ratio": selected_bytes / physical_bytes,
        },
        "trits": diagnostics,
        "ptq_artifact_id": conversion.artifact_id,
        "native_checkpoint_digest": _sha256(native_dir / "model.safetensors"),
        "onnx_artifact_id": onnx_artifact.artifact_id,
        "qat_loss": float(loss.detach().cpu()),
        "qat_optimizer_state_entries": len(resumed_optimizer.state),
        "generated_token_ids": generated[0].tolist(),
        "generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
    }
    _atomic_json(target / "receipt.json", receipt)
    return receipt


__all__ = [
    "SMOLLM2_MODEL_ID",
    "SMOLLM2_REVISION",
    "run_smollm2_release_demo",
]
