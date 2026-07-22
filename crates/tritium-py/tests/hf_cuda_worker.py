"""Single-device CUDA/fp16 Accelerate qualification with a durable receipt."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

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
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA qualification requires a physical CUDA device")
    accelerator = Accelerator(mixed_precision="fp16")
    if accelerator.device.type != "cuda":
        raise RuntimeError("Accelerate did not select CUDA")
    torch.manual_seed(401)
    torch.cuda.manual_seed_all(401)
    model = prepare_qat(
        _model(),
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
    if not elapsed_ms > 0:
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

    properties = torch.cuda.get_device_properties(accelerator.device)
    receipt = {
        "schema_version": 1,
        "artifact_kind": "tritium-torch-cuda-qualification",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": properties.name,
        "device_total_memory": properties.total_memory,
        "mixed_precision": "fp16",
        "steps": 5,
        "elapsed_ms": elapsed_ms,
        "steps_per_second": 5000.0 / elapsed_ms,
        "ternary_operator_host_transfers": 0,
        "checkpoint_exact": True,
    }
    _write_receipt(Path(os.environ["TRITIUM_CUDA_RECEIPT"]), receipt)
    print("TRITIUM_ACCELERATE_CUDA_FP16_OK", flush=True)


if __name__ == "__main__":
    main()
