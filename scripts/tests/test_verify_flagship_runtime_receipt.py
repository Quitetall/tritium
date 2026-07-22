import hashlib
from pathlib import Path
import runpy
import tempfile
import unittest


MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "verify-flagship-runtime-receipt.py"
)
canonical = MODULE["canonical"]
validate_runtime = MODULE["validate_runtime"]
validate_physical = MODULE["validate_physical"]
FlagshipRuntimeError = MODULE["FlagshipRuntimeError"]


def seal(value):
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    return value


def fixture(root: Path):
    sizes = (200, 400, 450)
    candidate_artifacts = []
    records = []
    for ordinal, track_id in enumerate(MODULE["TRACKS"]):
        path = root / f"{track_id}.salt"
        path.write_bytes(bytes([ordinal + 1]) * sizes[ordinal])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact_id = f"qwen-{track_id}"
        candidate_artifacts.append(
            {
                "id": artifact_id, "kind": "model-bundle", "path": path.name,
                "identity": {"bytes": path.stat().st_size, "sha256": digest},
            }
        )
        records.append(
            {
                "id": artifact_id, "kind": "model-bundle", "name": path.name,
                "bytes": path.stat().st_size, "sha256": digest,
            }
        )
    candidate = root / "manifest.json"
    candidate.write_bytes(canonical({"artifacts": candidate_artifacts}))
    common = {
        "result": "pass", "release": "1.1.0-rc.0",
        "source_revision": "a" * 40, "run_id": "runtime-run",
        "candidate_manifest_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "artifact": records[2], "model_id": MODULE["MODEL_ID"],
        "model_revision": MODULE["MODEL_REVISION"], "scope": "language+mtp",
    }
    return candidate, common, records


class FlagshipRuntimeReceiptTests(unittest.TestCase):
    def test_accepts_complete_native_runtime_and_physical_ledgers(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common, records = fixture(root)
            measurements = []
            for track_id in MODULE["TRACKS"]:
                for phase in ("prefill", "decode"):
                    for context, batch in MODULE["WORKLOADS"]:
                        tokens = context * batch if phase == "prefill" else batch
                        measurements.append(
                            {
                                "track_id": track_id, "phase": phase,
                                "context_tokens": context, "batch_size": batch,
                                "iterations": 20, "median_ms": 10.0,
                                "tokens_per_second": tokens * 100.0,
                            }
                        )
            runtime = seal(
                {
                    **common, "schema": MODULE["RUNTIME_SCHEMA"],
                    "device": {
                        "backend": "cuda", "physical": True, "uuid": "GPU-1",
                        "name": "RTX", "driver": "610.1", "total_bytes": 24_000,
                    },
                    "direct_ternary_kernel": True, "dense_materialization": False,
                    "host_transfers": 0, "measurements": measurements,
                    "claimed_regime": {
                        "context_tokens": 2048, "batch_size": 4,
                        "salt_v1_decode_ms": 10.0, "ptq_decode_ms": 10.0,
                        "slowdown_pct": 0.0,
                    },
                    "mtp": {
                        "acceptance_rate": 0.5, "baseline_tokens_per_second": 100.0,
                        "mtp_tokens_per_second": 120.0, "speedup": 1.2,
                    },
                }
            )
            runtime_path = root / "runtime.json"
            runtime_path.write_bytes(canonical(runtime))
            self.assertEqual(
                validate_runtime(runtime_path, "a" * 40, "1.1.0-rc.0", candidate),
                runtime,
            )

            weights = 1000
            dense_artifact = 4000
            dense_resident = 5000
            tracks = []
            for track_id, artifact in zip(MODULE["TRACKS"], records, strict=True):
                matrix_bytes = artifact["bytes"]
                resident = artifact["bytes"] + 50
                tracks.append(
                    {
                        "track_id": track_id, "artifact": artifact,
                        "matrix_bytes": matrix_bytes,
                        "matrix_bpw": matrix_bytes * 8 / weights,
                        "metadata_bpw": 0.005,
                        "whole_artifact_bytes": artifact["bytes"],
                        "whole_artifact_bpw": artifact["bytes"] * 8 / weights,
                        "resident_bytes": resident, "resident_bpw": resident * 8 / weights,
                        "peak_host_bytes": 1000, "peak_device_bytes": 1000,
                        "peak_transient_bytes": 100,
                        "artifact_reduction": dense_artifact / artifact["bytes"],
                        "resident_reduction": dense_resident / resident,
                    }
                )
            physical = seal(
                {
                    **common, "schema": MODULE["PHYSICAL_SCHEMA"],
                    "run_id": "physical-run", "quantized_weights": weights,
                    "dense_artifact_bytes": dense_artifact,
                    "dense_resident_bytes": dense_resident, "tracks": tracks,
                }
            )
            physical_path = root / "physical.json"
            physical_path.write_bytes(canonical(physical))
            self.assertEqual(
                validate_physical(physical_path, "a" * 40, "1.1.0-rc.0", candidate),
                physical,
            )

    def test_rejects_hidden_transfer(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common, records = fixture(root)
            runtime = {
                **common, "schema": MODULE["RUNTIME_SCHEMA"], "receipt_id": "bad",
                "device": {
                    "backend": "cuda", "physical": True, "uuid": "GPU-1",
                    "name": "RTX", "driver": "610.1", "total_bytes": 24_000,
                },
                "direct_ternary_kernel": True, "dense_materialization": False,
                "host_transfers": 1, "measurements": [], "claimed_regime": {}, "mtp": {},
            }
            path = root / "bad-runtime.json"
            path.write_bytes(canonical(runtime))
            with self.assertRaisesRegex(FlagshipRuntimeError, "host transfer"):
                validate_runtime(path, "a" * 40, "1.1.0-rc.0", candidate)


if __name__ == "__main__":
    unittest.main()
