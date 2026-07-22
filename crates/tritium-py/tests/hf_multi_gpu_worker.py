"""Two-physical-GPU DDP/FSDP worker for the v1.1 release qualification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from functools import lru_cache

import accelerate
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.nn.parallel import DistributedDataParallel
import transformers

import tritium
from tritium.nn import TernaryLinear
from tritium.torch import TernaryConfig, prepare_qat


SEED = 1201
STEPS = 20
SEQUENCE_LENGTH = 128
MODEL_CONFIG = {
    "vocab_size": 32768,
    "hidden_size": 1024,
    "intermediate_size": 2816,
    "num_hidden_layers": 8,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "max_position_embeddings": 256,
    "tie_word_embeddings": True,
}


def require_installed_distribution() -> None:
    try:
        distribution = importlib.metadata.distribution("tritium-torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "distributed worker requires installed tritium-torch"
        ) from error
    module = Path(tritium.__file__).resolve(strict=True)
    if distribution.files is None or module not in {
        distribution.locate_file(item).resolve() for item in distribution.files
    }:
        raise RuntimeError("imported tritium package is not owned by tritium-torch")


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def model() -> transformers.LlamaForCausalLM:
    return transformers.LlamaForCausalLM(transformers.LlamaConfig(**MODEL_CONFIG))


@lru_cache(maxsize=1)
def model_parameters() -> int:
    dense = model()
    return sum(parameter.numel() for parameter in dense.parameters())


def model_config_digest() -> str:
    config = transformers.LlamaConfig(**MODEL_CONFIG).to_dict()
    return "sha256:" + hashlib.sha256(canonical(config)).hexdigest()


def prepared_model(device: torch.device) -> torch.nn.Module:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return prepare_qat(
        model(),
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Linear", "Embedding"),
            planes=2,
        ),
    ).to(device)


def tokens(rank: int, device: torch.device) -> torch.Tensor:
    values = (torch.arange(SEQUENCE_LENGTH, device=device) + 1 + rank) % 32768
    return values.unsqueeze(0).to(torch.int64)


def step(
    wrapped: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = wrapped(input_ids=batch, labels=batch, use_cache=False).loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.detach()


def measured_steps(
    wrapped: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: torch.Tensor,
    *,
    global_batch_size: int,
) -> tuple[float, float, float]:
    for _ in range(2):
        step(wrapped, optimizer, scaler, batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(batch.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    initial_loss = None
    final_loss = None
    for index in range(STEPS):
        loss = step(wrapped, optimizer, scaler, batch)
        if index == 0:
            initial_loss = float(loss)
        final_loss = float(loss)
    end.record()
    end.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise RuntimeError("distributed CUDA timing interval is invalid")
    assert initial_loss is not None and final_loss is not None
    del global_batch_size
    return initial_loss, final_loss, elapsed_ms


def state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        value = parameter.detach().contiguous().view(torch.uint8).cpu()
        digest.update(name.encode())
        digest.update(
            canonical({"shape": list(parameter.shape), "dtype": str(parameter.dtype)})
        )
        digest.update(value.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def structured_digest(value: object) -> str:
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().view(torch.uint8).cpu()
            digest.update(b"tensor")
            digest.update(
                canonical({"shape": list(item.shape), "dtype": str(item.dtype)})
            )
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence")
            for child in item:
                update(child)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(canonical(item))
        else:
            raise TypeError(f"unsupported checkpoint state type: {type(item).__name__}")

    update(value)
    return "sha256:" + digest.hexdigest()


def scaler_tensor(scaler: torch.amp.GradScaler, device: torch.device) -> torch.Tensor:
    state = scaler.state_dict()
    fields = (
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    )
    if set(state) != set(fields):
        raise RuntimeError("GradScaler state fields changed")
    return torch.tensor([float(state[field]) for field in fields], device=device)


def load_scaler_tensor(scaler: torch.amp.GradScaler, value: torch.Tensor) -> None:
    fields = (
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    )
    values = value.detach().cpu().tolist()
    state = dict(zip(fields, values, strict=True))
    state["growth_interval"] = int(state["growth_interval"])
    state["_growth_tracker"] = int(state["_growth_tracker"])
    scaler.load_state_dict(state)


def host_transfer_count(module: torch.nn.Module, device: torch.device) -> int:
    first = next(item for item in module.modules() if isinstance(item, TernaryLinear))
    probe = torch.randn(
        8, first.in_features, device=device, dtype=torch.float32, requires_grad=True
    )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            first(probe).square().mean().backward()
    names = (event.key.lower() for event in profile.key_averages())
    return sum("memcpy dtoh" in name or "memcpy htod" in name for name in names)


def device_record(rank: int, device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    uuid = str(getattr(properties, "uuid", ""))
    if not uuid:
        raise RuntimeError("CUDA physical device UUID is unavailable")
    return {
        "rank": rank,
        "uuid": uuid,
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
    }


def restore_ddp(
    checkpoint: Path,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[bool, bool]:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    module.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["cpu_rng"].cpu())
    torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
    observed_cpu = torch.rand(8)
    observed_cuda = torch.rand(8, device=device)
    return (
        state_digest(module) == state["state_digest"]
        and structured_digest(optimizer.state_dict()) == state["optimizer_digest"]
        and structured_digest(scaler.state_dict()) == state["scaler_digest"],
        torch.equal(observed_cpu, state["next_cpu"])
        and torch.equal(observed_cuda, state["next_cuda"]),
    )


def save_ddp_checkpoint(
    checkpoint: Path,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> None:
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device)
    next_cpu = torch.rand(8)
    next_cuda = torch.rand(8, device=device)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device)
    torch.save(
        {
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "cpu_rng": cpu_rng,
            "cuda_rng": cuda_rng,
            "next_cpu": next_cpu,
            "next_cuda": next_cuda,
            "state_digest": state_digest(module),
            "optimizer_digest": structured_digest(optimizer.state_dict()),
            "scaler_digest": structured_digest(scaler.state_dict()),
        },
        checkpoint,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ddp", "fsdp"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    require_installed_distribution()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("qualification requires two visible physical CUDA devices")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError("qualification requires exactly two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    local_device = device_record(rank, device)
    devices: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(devices, local_device)
    concrete_devices = [item for item in devices if item is not None]
    if len({str(item["uuid"]) for item in concrete_devices}) != 2:
        raise RuntimeError("ranks do not map to distinct physical GPU UUIDs")

    baseline = 0.0
    if rank == 0:
        baseline_model = prepared_model(device)
        baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-4)
        baseline_scaler = torch.amp.GradScaler("cuda")
        _, _, baseline_elapsed = measured_steps(
            baseline_model,
            baseline_optimizer,
            baseline_scaler,
            tokens(rank, device),
            global_batch_size=1,
        )
        baseline = STEPS * SEQUENCE_LENGTH / (baseline_elapsed / 1000.0)
        del baseline_model, baseline_optimizer, baseline_scaler
        torch.cuda.empty_cache()
    baseline_tensor = torch.tensor([baseline], device=device, dtype=torch.float64)
    dist.broadcast(baseline_tensor, src=0)
    baseline = float(baseline_tensor.item())
    dist.barrier()

    unwrapped = prepared_model(device)
    transfers = host_transfer_count(unwrapped, device)
    unwrapped.zero_grad(set_to_none=True)
    if args.mode == "ddp":
        wrapped = DistributedDataParallel(
            unwrapped,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
        )
    else:
        wrapped = FullyShardedDataParallel(
            unwrapped,
            device_id=device,
            use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            ),
        )
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    initial_loss, final_loss, elapsed_ms = measured_steps(
        wrapped, optimizer, scaler, tokens(rank, device), global_batch_size=world_size
    )
    loss_pair = torch.tensor(
        [initial_loss, final_loss], device=device, dtype=torch.float64
    )
    dist.all_reduce(loss_pair, op=dist.ReduceOp.SUM)
    initial_loss, final_loss = (float(value / world_size) for value in loss_pair)
    elapsed = torch.tensor([elapsed_ms], device=device, dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    elapsed_ms = float(elapsed.item())
    throughput = STEPS * world_size * SEQUENCE_LENGTH / (elapsed_ms / 1000.0)
    peak = torch.tensor(
        [torch.cuda.max_memory_allocated(device)], device=device, dtype=torch.int64
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    peak_memory = int(peak.item())
    digest_target = wrapped.module if args.mode == "ddp" else wrapped
    local_digest = state_digest(digest_target)
    gathered_digests: list[str | None] = [None] * world_size
    dist.all_gather_object(gathered_digests, local_digest)
    global_digest = "sha256:" + hashlib.sha256(canonical(gathered_digests)).hexdigest()

    args.checkpoint.mkdir(parents=True, exist_ok=True)
    if args.mode == "ddp":
        rank_checkpoint = args.checkpoint / f"rank-{rank}.pt"
        save_ddp_checkpoint(rank_checkpoint, wrapped.module, optimizer, scaler, device)
        restored = prepared_model(device)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
        restored_scaler = torch.amp.GradScaler("cuda")
        checkpoint_exact, rng_exact = restore_ddp(
            rank_checkpoint,
            restored,
            restored_optimizer,
            restored_scaler,
            device,
        )
    else:
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device)
        next_cpu = torch.rand(8)
        next_cuda = torch.rand(8, device=device)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        model_state, optimizer_state = get_state_dict(wrapped, optimizer)
        saved_scaler = scaler_tensor(scaler, device)
        dcp.save(
            {
                "model": model_state,
                "optimizer": optimizer_state,
                "scaler": saved_scaler,
                "cpu_rng": cpu_rng,
                "cuda_rng": cuda_rng,
            },
            checkpoint_id=args.checkpoint / "dcp",
        )
        restored_model = prepared_model(device)
        restored_wrapped = FullyShardedDataParallel(
            restored_model,
            device_id=device,
            use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            ),
        )
        restored_optimizer = torch.optim.AdamW(restored_wrapped.parameters(), lr=1e-4)
        restored_scaler = torch.amp.GradScaler("cuda")
        restored_model_state, restored_optimizer_state = get_state_dict(
            restored_wrapped, restored_optimizer
        )
        loaded = {
            "model": restored_model_state,
            "optimizer": restored_optimizer_state,
            "scaler": torch.empty_like(saved_scaler),
            "cpu_rng": torch.empty_like(cpu_rng),
            "cuda_rng": torch.empty_like(cuda_rng),
        }
        dcp.load(loaded, checkpoint_id=args.checkpoint / "dcp")
        set_state_dict(
            restored_wrapped,
            restored_optimizer,
            model_state_dict=loaded["model"],
            optim_state_dict=loaded["optimizer"],
        )
        load_scaler_tensor(restored_scaler, loaded["scaler"])
        checkpoint_exact = (
            state_digest(restored_wrapped) == local_digest
            and structured_digest(restored_optimizer.state_dict())
            == structured_digest(optimizer.state_dict())
            and structured_digest(restored_scaler.state_dict())
            == structured_digest(scaler.state_dict())
        )
        torch.set_rng_state(loaded["cpu_rng"].cpu())
        torch.cuda.set_rng_state(loaded["cuda_rng"].cpu(), device)
        rng_exact = torch.equal(torch.rand(8), next_cpu) and torch.equal(
            torch.rand(8, device=device), next_cuda
        )
    dist.barrier()
    if args.mode == "fsdp":
        shards = sorted((args.checkpoint / "dcp").glob("*.distcp"))
        if len(shards) != world_size:
            raise RuntimeError("FSDP checkpoint did not emit one shard per rank")
        rank_checkpoint = shards[rank]
    rank_checkpoint_digest = sha256_file(rank_checkpoint)
    checkpoint_digests: list[str | None] = [None] * world_size
    dist.all_gather_object(checkpoint_digests, rank_checkpoint_digest)
    if not checkpoint_exact or not rng_exact or transfers != 0:
        raise RuntimeError("distributed checkpoint, RNG, or residency invariant failed")

    config_digest = model_config_digest()
    parameter_count = model_parameters()
    driver_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
        text=True,
        timeout=30,
    ).splitlines()
    drivers = {
        uuid.strip(): driver.strip()
        for uuid, driver in (row.split(",", maxsplit=1) for row in driver_rows)
    }
    cuda_driver = drivers.get(str(local_device["uuid"]), "")
    if not cuda_driver:
        raise RuntimeError("CUDA driver version is unavailable for the rank device")
    if rank == 0:
        nccl = torch.cuda.nccl.version()
        fragment = {
            "schema": "tritium.hf-distributed-mode.v1",
            "model_config_sha256": config_digest,
            "model_parameters": parameter_count,
            "machine": {
                "system": platform.system(),
                "architecture": platform.machine(),
            },
            "environment": {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "accelerate_version": accelerate.__version__,
                "cuda_runtime": str(torch.version.cuda),
                "cuda_driver": cuda_driver,
                "nccl_version": ".".join(map(str, nccl)),
            },
            "devices": concrete_devices,
            "mode": {
                "name": args.mode,
                "backend": "nccl",
                "mixed_precision": "fp16",
                "world_size": world_size,
                "steps": STEPS,
                "global_batch_size": world_size,
                "sequence_length": SEQUENCE_LENGTH,
                "measured_tokens": STEPS * world_size * SEQUENCE_LENGTH,
                "elapsed_ms": elapsed_ms,
                "tokens_per_second": throughput,
                "single_device_tokens_per_second": baseline,
                "scaling_efficiency": throughput / (baseline * world_size),
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "checkpoint_exact": checkpoint_exact,
                "rng_exact": rng_exact,
                "host_transfers": transfers,
                "peak_memory_bytes": peak_memory,
                "global_state_sha256": global_digest,
                "rank_checkpoint_sha256": checkpoint_digests,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical(fragment) + b"\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
