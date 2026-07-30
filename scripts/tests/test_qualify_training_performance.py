from pathlib import Path
import json
import runpy
import tempfile
import unittest

from scripts.tests.test_verify_training_backend_receipt import fixture as backend_fixture


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-training-performance.py")
assemble = MODULE["assemble"]
parse_traces = MODULE["parse_traces"]
QualificationError = MODULE["QualificationError"]


def traces(root: Path, *, host_transfer_family: str | None = None):
    result = {}
    physical = {
        "cpu": "cpu:x86_64", "cuda": "cuda:0:GPU-1", "rocm": "rocm:0:GPU-2",
        "metal": "metal:Apple-M4", "wgpu": "vulkan:GPU-1",
        "wasi": "wasmtime:46.0.0:x86_64", "mcu": "stm32h7:board-1",
    }
    for ordinal, family in enumerate(MODULE["FAMILIES"]):
        median = 100.0 / (ordinal + 1)
        path = root / f"{family}.trace.json"
        path.write_text(json.dumps({
            "schema": MODULE["TRACE_SCHEMA"], "family": family,
            "artifact_id": f"training-{family}",
            "physical_device": physical[family],
            "workload_id": MODULE["WORKLOAD_ID"],
            "budget_id": "sha256:" + "9" * 64,
            "warmups_ms": [median] * 10,
            "samples": [
                {
                    "elapsed_ms": median if sample < 28 else median * 1.2,
                    "cases": 117, "peak_resident_bytes": 1000,
                    "peak_scratch_bytes": 100,
                    "host_transfers": 1
                    if family == host_transfer_family and sample == 0 else 0,
                    "global_synchronizations": 0, "native_execution": True,
                    "budget_pass": True, "energy_joules": None,
                }
                for sample in range(30)
            ],
        }, sort_keys=True), encoding="utf-8")
        result[family] = path
    return result


class QualifyTrainingPerformanceTests(unittest.TestCase):
    def test_aggregates_raw_traces_and_self_validates(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, backend_path, backend = backend_fixture(root)
            output = root / "performance"
            receipt = assemble(
                output, repo=ROOT, candidate=candidate,
                backend_receipt_path=backend_path, trace_paths=traces(root),
                source_revision="a" * 40, release="1.1.0-rc.0",
                run_id="performance-physical-1",
            )
            self.assertEqual(
                receipt["backend_manifest_receipt_id"], backend["receipt_id"]
            )
            self.assertEqual(len(receipt["measurements"]), 7)
            self.assertTrue((output / "traces/00-cpu.json").is_file())

    def test_rejects_raw_host_transfer(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, backend_path, _ = backend_fixture(root)
            with self.assertRaisesRegex(
                MODULE["TrainingPerformanceError"], "residency"
            ):
                assemble(
                    root / "performance", repo=ROOT, candidate=candidate,
                    backend_receipt_path=backend_path,
                    trace_paths=traces(root, host_transfer_family="cuda"),
                    source_revision="a" * 40, release="1.1.0-rc.0",
                    run_id="performance-physical-1",
                )

    def test_trace_cli_bindings_require_frozen_order(self):
        values = [f"{family}=/{family}.json" for family in MODULE["FAMILIES"]]
        self.assertEqual(tuple(parse_traces(values)), MODULE["FAMILIES"])
        with self.assertRaisesRegex(QualificationError, "all seven"):
            parse_traces(values[:-1])


if __name__ == "__main__":
    unittest.main()
