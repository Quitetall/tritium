from __future__ import annotations

import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-hf-distributed.py")
QualificationError = MODULE["QualificationError"]
assemble = MODULE["assemble_receipt"]
canonical = MODULE["canonical"]
sha256_file = MODULE["sha256_file"]
validate = MODULE["validate_receipt"]


def mode(name: str) -> dict:
    elapsed = 20_000.0 if name == "ddp" else 25_000.0
    throughput = 5120 / (elapsed / 1000.0)
    baseline = 160.0
    return {
        "name": name,
        "backend": "nccl",
        "mixed_precision": "fp16",
        "world_size": 2,
        "steps": 20,
        "global_batch_size": 2,
        "sequence_length": 128,
        "measured_tokens": 5120,
        "elapsed_ms": elapsed,
        "tokens_per_second": throughput,
        "single_device_tokens_per_second": baseline,
        "scaling_efficiency": throughput / (baseline * 2),
        "peak_memory_bytes": 12_000_000_000,
        "initial_loss": 3.5,
        "final_loss": 3.25,
        "checkpoint_exact": True,
        "rng_exact": True,
        "host_transfers": 0,
        "global_state_sha256": "sha256:" + ("1" if name == "ddp" else "2") * 64,
        "rank_checkpoint_sha256": ["sha256:" + "0" * 64] * 2,
    }


def fragments() -> list[dict]:
    common = {
        "schema": "tritium.hf-distributed-mode.v1",
        "model_config_sha256": "sha256:" + "6" * 64,
        "model_parameters": 127_943_680,
        "machine": {"system": "Linux", "architecture": "x86_64"},
        "environment": {
            "python_version": "3.13.5",
            "torch_version": "2.11.0",
            "transformers_version": "5.5.3",
            "accelerate_version": "1.10.0",
            "cuda_runtime": "13.0",
            "cuda_driver": "610.43.03",
            "nccl_version": "2.27.3",
        },
        "devices": [
            {
                "rank": 0,
                "uuid": "GPU-physical-0",
                "name": "A100",
                "compute_capability": "8.0",
                "total_memory_bytes": 80_000_000_000,
            },
            {
                "rank": 1,
                "uuid": "GPU-physical-1",
                "name": "A100",
                "compute_capability": "8.0",
                "total_memory_bytes": 80_000_000_000,
            },
        ],
    }
    return [{**common, "mode": mode(name)} for name in ("ddp", "fsdp")]


class QualifyHfDistributedTests(unittest.TestCase):
    def test_assembly_copies_and_validates_all_rank_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stage = root / "stage"
            stage.mkdir()
            artifact = root / "pytritium.whl"
            artifact.write_bytes(b"candidate wheel")
            checkpoints = {}
            for mode_name in ("ddp", "fsdp"):
                for rank in (0, 1):
                    path = root / f"{mode_name}-{rank}.checkpoint"
                    path.write_bytes(f"{mode_name}-{rank}".encode())
                    checkpoints[(mode_name, rank)] = path
            values = fragments()
            for value in values:
                value["mode"]["rank_checkpoint_sha256"] = [
                    "sha256:" + sha256_file(checkpoints[(value["mode"]["name"], rank)])
                    for rank in (0, 1)
                ]
            receipt = assemble(
                stage=stage,
                artifact=artifact,
                source_revision="a" * 40,
                release="1.1.0-rc.0",
                run_id="two-gpu-run-1",
                started_at_utc="2026-07-22T12:00:00Z",
                duration_ms=60_000.0,
                fragments=values,
                checkpoint_files=checkpoints,
            )
            receipt_path = stage / "receipt.json"
            receipt_path.write_bytes(canonical(receipt) + b"\n")

            self.assertEqual(
                validate(receipt_path, "a" * 40, "1.1.0-rc.0", artifact),
                receipt,
            )
            self.assertEqual(len(receipt["support_artifacts"]), 4)

    def test_assembly_rejects_fragment_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stage = root / "stage"
            stage.mkdir()
            artifact = root / "pytritium.whl"
            artifact.write_bytes(b"candidate wheel")
            values = fragments()
            values[1] = json.loads(json.dumps(values[1]))
            values[1]["devices"][1]["uuid"] = "GPU-other"

            with self.assertRaisesRegex(QualificationError, "devices"):
                assemble(
                    stage=stage,
                    artifact=artifact,
                    source_revision="a" * 40,
                    release="1.1.0-rc.0",
                    run_id="two-gpu-run-1",
                    started_at_utc="2026-07-22T12:00:00Z",
                    duration_ms=60_000.0,
                    fragments=values,
                    checkpoint_files={},
                )

    def test_runner_uses_isolated_two_rank_torchrun(self):
        source = (ROOT / "scripts" / "qualify-hf-distributed.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"-I"', source)
        self.assertIn('"--nproc_per_node=2"', source)
        self.assertIn('for mode in ("ddp", "fsdp")', source)
        self.assertIn(
            "venv.EnvBuilder(with_pip=True, system_site_packages=True)", source
        )
        self.assertIn('"--no-index"', source)
        self.assertIn('"--no-deps"', source)

    def test_worker_freezes_physical_and_training_contracts(self):
        source = (ROOT / "crates/tritium-py/tests/hf_multi_gpu_worker.py").read_text(
            encoding="utf-8"
        )
        for contract in (
            "STEPS = 20",
            "SEQUENCE_LENGTH = 128",
            'dist.init_process_group("nccl")',
            "torch.cuda.device_count() < 2",
            'getattr(properties, "uuid", "")',
            "DistributedDataParallel(",
            "FullyShardedDataParallel(",
            "dcp.save(",
            "dcp.load(",
            "torch.profiler.profile(",
        ):
            self.assertIn(contract, source)

    def test_wheel_ci_runs_structural_qualification_tests(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
        self.assertIn("scripts.tests.test_qualify_hf_distributed", workflow)
        self.assertIn("scripts.tests.test_verify_hf_distributed_receipt", workflow)


if __name__ == "__main__":
    unittest.main()
