from __future__ import annotations

import hashlib
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "verify-hf-distributed-receipt.py")
ReceiptError = MODULE["ReceiptError"]
canonical = MODULE["canonical"]
validate = MODULE["validate"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mode(name: str) -> dict:
    steps = 20
    batch = 2
    sequence = 128
    tokens = steps * batch * sequence
    elapsed = 20_000.0 if name == "ddp" else 25_000.0
    throughput = tokens / (elapsed / 1000.0)
    single_device = 160.0
    return {
        "name": name,
        "backend": "nccl",
        "mixed_precision": "fp16",
        "world_size": 2,
        "steps": steps,
        "global_batch_size": batch,
        "sequence_length": sequence,
        "measured_tokens": tokens,
        "elapsed_ms": elapsed,
        "tokens_per_second": throughput,
        "single_device_tokens_per_second": single_device,
        "scaling_efficiency": throughput / (single_device * 2.0),
        "peak_memory_bytes": 12_000_000_000,
        "initial_loss": 3.5,
        "final_loss": 3.25,
        "checkpoint_exact": True,
        "rng_exact": True,
        "host_transfers": 0,
        "global_state_sha256": "sha256:" + ("1" if name == "ddp" else "2") * 64,
        "rank_checkpoint_sha256": ["sha256:" + "3" * 64, "sha256:" + "4" * 64],
    }


def receipt(artifact: Path, root: Path) -> dict:
    support = []
    for mode_name in ("ddp", "fsdp"):
        for rank in (0, 1):
            path = root / f"{mode_name}-rank-{rank}.checkpoint"
            path.write_bytes(f"{mode_name} rank {rank} checkpoint".encode())
            support.append(
                {
                    "mode": mode_name,
                    "rank": rank,
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    modes = [mode("ddp"), mode("fsdp")]
    for value in modes:
        value["rank_checkpoint_sha256"] = [
            "sha256:" + item["sha256"]
            for item in support
            if item["mode"] == value["name"]
        ]
    value = {
        "schema": "tritium.hf-distributed-qualification.v1",
        "source_revision": "a" * 40,
        "release": "1.1.0-rc.0",
        "run_id": "two-gpu-run-1",
        "started_at_utc": "2026-07-22T12:00:00Z",
        "duration_ms": 60000.0,
        "source_dirty": False,
        "command_contract": "torchrun-nproc2-ddp-then-fsdp-v1",
        "artifact": {
            "kind": "python-wheel",
            "name": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": sha256(artifact),
        },
        "model_config_sha256": "sha256:" + "6" * 64,
        "model_parameters": 1_000_000_000,
        "machine": {
            "machine_id": "sha256:" + "5" * 64,
            "system": "Linux",
            "architecture": "x86_64",
        },
        "environment": {
            "python_version": "3.13.5",
            "torch_version": "2.11.0",
            "transformers_version": "5.5.3",
            "accelerate_version": "1.10.0",
            "cuda_runtime": "13.0",
            "cuda_driver": "610.43.03",
            "nccl_version": "2.27.3",
        },
        "world_size": 2,
        "devices": [
            {
                "rank": 0,
                "uuid": "GPU-physical-0",
                "name": "NVIDIA A100-SXM4-80GB",
                "compute_capability": "8.0",
                "total_memory_bytes": 80_000_000_000,
            },
            {
                "rank": 1,
                "uuid": "GPU-physical-1",
                "name": "NVIDIA A100-SXM4-80GB",
                "compute_capability": "8.0",
                "total_memory_bytes": 80_000_000_000,
            },
        ],
        "modes": modes,
        "support_artifacts": support,
        "result": "pass",
    }
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    return value


class VerifyHfDistributedReceiptTests(unittest.TestCase):
    def test_accepts_exact_two_device_ddp_and_fsdp_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifact = root / "tritium_torch.whl"
            artifact.write_bytes(b"candidate wheel")
            value = receipt(artifact, root)
            path = root / "receipt.json"
            path.write_bytes(canonical(value) + b"\n")

            self.assertEqual(validate(path, "a" * 40, "1.1.0-rc.0", artifact), value)

    def test_rejects_shared_device_missing_mode_bad_arithmetic_and_wheel_drift(self):
        mutations = (
            "shared-device",
            "missing-mode",
            "arithmetic",
            "scaling",
            "wheel",
            "support",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                artifact = root / "tritium_torch.whl"
                artifact.write_bytes(b"candidate wheel")
                value = receipt(artifact, root)
                if mutation == "shared-device":
                    value["devices"][1]["uuid"] = value["devices"][0]["uuid"]
                elif mutation == "missing-mode":
                    value["modes"].pop()
                elif mutation == "arithmetic":
                    value["modes"][0]["tokens_per_second"] += 1.0
                elif mutation == "scaling":
                    value["modes"][0]["single_device_tokens_per_second"] = 256.0
                    value["modes"][0]["scaling_efficiency"] = 0.5
                value["receipt_id"] = (
                    "sha256:"
                    + hashlib.sha256(
                        canonical(
                            {
                                key: item
                                for key, item in value.items()
                                if key != "receipt_id"
                            }
                        )
                    ).hexdigest()
                )
                path = root / "receipt.json"
                path.write_bytes(canonical(value) + b"\n")
                if mutation == "wheel":
                    artifact.write_bytes(b"different wheel")
                elif mutation == "support":
                    support = root / value["support_artifacts"][0]["path"]
                    payload = bytearray(support.read_bytes())
                    payload[-1] ^= 1
                    support.write_bytes(payload)

                with self.assertRaises(ReceiptError):
                    validate(path, "a" * 40, "1.1.0-rc.0", artifact)


if __name__ == "__main__":
    unittest.main()
