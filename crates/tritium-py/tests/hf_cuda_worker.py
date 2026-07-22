"""Single-device CUDA/fp16 Accelerate qualification with a durable receipt."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import time

import accelerate
import torch
import transformers
from accelerate import Accelerator

from tritium.nn import TernaryLinear
from tritium.torch import TernaryConfig, prepare_qat


def _model():
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            tie_word_embeddings=True,
        )
    )


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _artifact_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return {"name": path.name, "bytes": size, "sha256": digest.hexdigest()}


def _write_receipt(path: Path, value) -> None:
    identity = dict(value)
    identity["receipt_id"] = "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".cuda-receipt-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(identity) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    source_revision = os.environ.get("TRITIUM_SOURCE_REVISION", "")
    release = os.environ.get("TRITIUM_RELEASE", "")
    run_id = os.environ.get("TRITIUM_RUN_ID", "")
    artifact_path = Path(os.environ.get("TRITIUM_QUALIFIED_ARTIFACT", ""))
    artifact_kind = os.environ.get("TRITIUM_ARTIFACT_KIND", "")
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise RuntimeError("TRITIUM_SOURCE_REVISION must be a full Git object ID")
    if re.fullmatch(r"1\.1\.0-rc\.(0|[1-9][0-9]*)", release) is None:
        raise RuntimeError("TRITIUM_RELEASE must be a canonical v1.1 candidate")
    if not run_id:
        raise RuntimeError("TRITIUM_RUN_ID must identify this independent run")
    if not artifact_kind or not artifact_path.is_file() or artifact_path.is_symlink():
        raise RuntimeError("qualified artifact must be an ordinary named file")
    artifact = {"kind": artifact_kind, **_artifact_identity(artifact_path)}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA qualification requires a physical CUDA device")
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    qualification_started = time.monotonic()
    accelerator = Accelerator(mixed_precision="fp16")
    if accelerator.device.type != "cuda":
        raise RuntimeError("Accelerate did not select CUDA")
    torch.manual_seed(401)
    torch.cuda.manual_seed_all(401)
    dense_model = _model()
    model_config_sha256 = hashlib.sha256(
        _canonical(dense_model.config.to_dict())
    ).hexdigest()
    model = prepare_qat(
        dense_model,
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Linear", "Embedding"),
            planes=1,
        ),
    )
    observed_dtypes = []
    first = next(
        module for module in model.modules() if isinstance(module, TernaryLinear)
    )
    first.register_forward_hook(
        lambda _module, _inputs, output: observed_dtypes.append(str(output.dtype))
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model, optimizer = accelerator.prepare(model, optimizer)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=accelerator.device)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            loss = model(input_ids=tokens, labels=tokens).loss
        accelerator.backward(loss)
        optimizer.step()

    step()
    torch.cuda.synchronize()
    observed_dtypes.clear()
    probe = torch.randn(
        8,
        first.in_features,
        device=accelerator.device,
        dtype=torch.float32,
        requires_grad=True,
    )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        with accelerator.autocast():
            first(probe).square().mean().backward()
    event_names = tuple(sorted({event.key.lower() for event in profile.key_averages()}))
    forbidden = tuple(
        name for name in event_names if "memcpy dtoh" in name or "memcpy htod" in name
    )
    if forbidden:
        raise AssertionError(f"steady-state host transfer observed: {forbidden}")
    if "torch.float16" not in observed_dtypes:
        raise AssertionError(
            f"ternary kernels did not execute in fp16: {observed_dtypes}"
        )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        step()
    end.record()
    end.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    if elapsed_ms <= 0:
        raise AssertionError("CUDA timing interval is not positive")

    checkpoint = Path(os.environ["TRITIUM_CUDA_CHECKPOINT"])
    accelerator.save_state(checkpoint)
    unwrapped = accelerator.unwrap_model(model)
    expected = {
        name: value.detach().clone() for name, value in unwrapped.state_dict().items()
    }
    with torch.no_grad():
        next(unwrapped.parameters()).add_(1)
    accelerator.load_state(checkpoint)
    if any(
        not torch.equal(unwrapped.state_dict()[name], value)
        for name, value in expected.items()
    ):
        raise AssertionError("Accelerate CUDA checkpoint did not restore exact state")

    device_index = accelerator.device.index or 0
    properties = torch.cuda.get_device_properties(device_index)
    device_uuid = str(getattr(properties, "uuid", ""))
    if not device_uuid:
        raise RuntimeError("CUDA device UUID is unavailable")
    machine_material = {
        "architecture": platform.machine(),
        "device_uuid": device_uuid,
        "node": platform.node(),
        "system": platform.system(),
    }
    machine_id = "sha256:" + hashlib.sha256(_canonical(machine_material)).hexdigest()
    cuda_driver = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            f"--id={device_index}",
        ],
        text=True,
        timeout=30,
    ).strip()
    if not cuda_driver:
        raise RuntimeError("CUDA driver version is unavailable")
    receipt = {
        "schema": "tritium.cuda-training-qualification.v1",
        "source_revision": source_revision,
        "release": release,
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "duration_ms": (time.monotonic() - qualification_started) * 1000.0,
        "command": ["python", "hf_cuda_worker.py"],
        "artifact": artifact,
        "machine": {
            "machine_id": machine_id,
            "system": platform.system(),
            "architecture": platform.machine(),
        },
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "accelerate_version": accelerate.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_driver": cuda_driver,
        },
        "device": {
            "index": device_index,
            "uuid": device_uuid,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
        },
        "workload": {
            "seed": 401,
            "mixed_precision": "fp16",
            "steps": 5,
            "batch_size": 1,
            "sequence_length": 8,
            "model_config_sha256": model_config_sha256,
        },
        "measurements": {
            "elapsed_ms": elapsed_ms,
            "steps_per_second": 5000.0 / elapsed_ms,
        },
        "invariants": {
            "ternary_operator_host_transfers": 0,
            "ternary_operator_dtype": "torch.float16",
            "checkpoint_exact": True,
        },
        "result": "pass",
    }
    _write_receipt(Path(os.environ["TRITIUM_CUDA_RECEIPT"]), receipt)
    print("TRITIUM_ACCELERATE_CUDA_FP16_OK", flush=True)


if __name__ == "__main__":
    main()
