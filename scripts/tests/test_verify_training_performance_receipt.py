import hashlib
from pathlib import Path
import runpy
import tempfile
import unittest


MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "verify-training-performance-receipt.py"
)
canonical = MODULE["canonical"]
validate = MODULE["validate"]
TrainingPerformanceError = MODULE["TrainingPerformanceError"]


def fixture(root: Path):
    candidate_artifacts = []
    records = []
    for ordinal, family in enumerate(MODULE["FAMILIES"]):
        path = root / f"{family}.json"
        path.write_bytes(canonical({"physical_device": f"{family}:device-{ordinal}"}))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        blake3 = f"{ordinal + 1:x}" * 64
        artifact_id = f"training-{family}"
        candidate_artifacts.append(
            {
                "id": artifact_id, "kind": "training-receipt-bundle", "path": path.name,
                "identity": {"bytes": path.stat().st_size, "sha256": sha, "blake3": blake3},
            }
        )
        records.append(
            {
                "id": artifact_id, "kind": "training-receipt-bundle", "name": path.name,
                "bytes": path.stat().st_size, "sha256": sha, "blake3": blake3,
            }
        )
    candidate = root / "manifest.json"
    candidate.write_bytes(canonical({"artifacts": candidate_artifacts}))
    measurements = []
    traces = root / "traces"
    traces.mkdir()
    cpu_median = 100.0
    for ordinal, (family, artifact) in enumerate(
        zip(MODULE["FAMILIES"], records, strict=True)
    ):
        median = cpu_median / (ordinal + 1)
        p95 = median * 1.2
        trace_path = traces / f"{family}.json"
        trace_path.write_bytes(canonical({
            "schema": MODULE["TRACE_SCHEMA"], "family": family,
            "artifact_id": artifact["id"],
            "physical_device": f"{family}:device-{ordinal}",
            "workload_id": "training-manifest-v2-full-117",
            "budget_id": "sha256:" + "9" * 64,
            "warmups_ms": [median] * 10,
            "samples": [
                {
                    "elapsed_ms": value, "cases": 117,
                    "peak_resident_bytes": 1000, "peak_scratch_bytes": 100,
                    "host_transfers": 0, "global_synchronizations": 0,
                    "native_execution": True, "budget_pass": True,
                    "energy_joules": None,
                }
                for value in [*([median] * 28), p95, p95]
            ],
        }))
        measurements.append(
            {
                "family": family,
                "tier": "throughput" if ordinal < 5 else "bounded-latency",
                "artifact": artifact, "physical_device": f"{family}:device-{ordinal}",
                "warmup_iterations": 10, "sample_count": 30,
                "cases_per_sample": 117, "median_ms": median,
                "p95_ms": p95, "cases_per_second": 117000 / median,
                "cpu_relative_speed": cpu_median / median,
                "peak_resident_bytes": 1000, "peak_scratch_bytes": 100,
                "host_transfers": 0, "global_synchronizations": 0,
                "native_execution": True, "budget_pass": True,
                "energy_joules": None,
                "trace": {
                    "path": trace_path.relative_to(root).as_posix(),
                    "bytes": trace_path.stat().st_size,
                    "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                },
            }
        )
    receipt = {
        "schema": MODULE["SCHEMA"], "result": "pass",
        "release": "1.1.0-rc.0", "source_revision": "a" * 40,
        "run_id": "training-performance-1",
        "candidate_manifest_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "backend_manifest_receipt_id": "sha256:" + "8" * 64,
        "workload_id": "training-manifest-v2-full-117",
        "budget_id": "sha256:" + "9" * 64, "measurements": measurements,
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
    path = root / "receipt.json"
    path.write_bytes(canonical(receipt))
    return candidate, path, receipt


class TrainingPerformanceReceiptTests(unittest.TestCase):
    def test_accepts_all_physical_performance_tiers(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, path, receipt = fixture(Path(raw))
            self.assertEqual(
                validate(path, "a" * 40, "1.1.0-rc.0", candidate), receipt
            )

    def test_rejects_host_transfer(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, path, receipt = fixture(Path(raw))
            receipt["measurements"][1]["host_transfers"] = 1
            unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
            path.write_bytes(canonical(receipt))
            with self.assertRaisesRegex(TrainingPerformanceError, "residency"):
                validate(path, "a" * 40, "1.1.0-rc.0", candidate)


if __name__ == "__main__":
    unittest.main()
