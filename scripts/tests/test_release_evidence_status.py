from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "release-evidence-status.py")
EvidenceError = MODULE["EvidenceError"]
evaluate = MODULE["evaluate"]
render = MODULE["render"]
gate_row = MODULE["_gate_row"]


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_fixture(root: Path) -> tuple[Path, dict, Path, Path]:
    candidate_root = root / "candidate"
    candidate_root.mkdir()
    artifact = candidate_root / "candidate.whl"
    artifact.write_bytes(b"qualified wheel bytes")
    candidate = candidate_root / "manifest.json"
    document = {
        "schema": "tritium.release-candidate.v1",
        "release": "1.1.0-rc.0",
        "source_revision": "a" * 40,
        "artifacts": [
            {
                "id": "cuda-wheel",
                "kind": "python-wheel",
                "path": artifact.name,
                "identity": {},
                "sbom": {},
                "provenance": {},
            }
        ],
    }
    candidate.write_bytes(canonical(document) + b"\n")
    evidence_root = root / "evidence"
    evidence_root.mkdir()
    return candidate, document, artifact, evidence_root


def cuda_receipt(path: Path, artifact: Path, *, run_id: str = "run-17") -> dict:
    value = {
        "schema": "tritium.cuda-training-qualification.v1",
        "source_revision": "a" * 40,
        "release": "1.1.0-rc.0",
        "run_id": run_id,
        "started_at_utc": "2026-07-21T12:00:00Z",
        "duration_ms": 4000.0,
        "command": ["python", "hf_cuda_worker.py"],
        "artifact": {
            "kind": "python-wheel",
            "name": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": sha256(artifact),
        },
        "machine": {
            "machine_id": "sha256:" + "b" * 64,
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
        },
        "device": {
            "index": 0,
            "uuid": "GPU-physical",
            "name": "NVIDIA GeForce RTX 4090",
            "compute_capability": "8.9",
            "total_memory_bytes": 25_000_000_000,
        },
        "workload": {
            "seed": 401,
            "mixed_precision": "fp16",
            "steps": 5,
            "batch_size": 1,
            "sequence_length": 8,
            "model_config_sha256": "c" * 64,
        },
        "measurements": {"elapsed_ms": 250.0, "steps_per_second": 20.0},
        "invariants": {
            "ternary_operator_host_transfers": 0,
            "ternary_operator_dtype": "torch.float16",
            "checkpoint_exact": True,
        },
        "result": "pass",
    }
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    path.write_bytes(canonical(value) + b"\n")
    return value


def registry(
    path: Path, candidate: Path, receipts: list[dict]
) -> None:
    path.write_bytes(
        canonical(
            {
                "schema": "tritium.release-evidence-registry.v1",
                "release": "1.1.0-rc.0",
                "source_revision": "a" * 40,
                "candidate_manifest_sha256": sha256(candidate),
                "receipts": receipts,
            }
        )
        + b"\n"
    )


def entry(receipt_path: Path, receipt: dict, *, parents: list[str] | None = None) -> dict:
    return {
        "id": receipt["receipt_id"],
        "kind": "cuda-training",
        "path": receipt_path.name,
        "sha256": sha256(receipt_path),
        "artifact_id": "cuda-wheel",
        "parents": parents or [],
    }


class ReleaseEvidenceStatusTests(unittest.TestCase):
    def test_gate_status_distinguishes_pass_missing_and_structural(self):
        required = ("quality", "runtime")
        self.assertEqual(gate_row("gate", required, {})["status"], "MISSING")
        self.assertEqual(
            gate_row("gate", required, {"quality": "empirical"})["status"],
            "MISSING",
        )
        self.assertEqual(
            gate_row(
                "gate", required, {"quality": "empirical", "runtime": "structural"}
            )["status"],
            "STRUCTURAL_ONLY",
        )
        self.assertEqual(
            gate_row(
                "gate", required, {"quality": "empirical", "runtime": "empirical"}
            )["status"],
            "PASS",
        )

    def test_empty_registry_reports_every_gate_missing(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, _, evidence_root = release_fixture(Path(raw))
            registry_path = evidence_root / "registry.json"
            registry(registry_path, candidate, [])
            report = evaluate(registry_path, candidate, document)
            self.assertFalse(report["ready"])
            self.assertTrue(all(row["status"] == "MISSING" for row in report["rows"]))
            rendered = render(report)
            self.assertIn("MISSING", rendered)
            self.assertIn("EXTERNAL_AUTH_REQUIRED", rendered)

    def test_cuda_receipt_is_empirical_but_does_not_green_backend_gate(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, artifact, evidence_root = release_fixture(Path(raw))
            receipt_path = evidence_root / "cuda.json"
            receipt = cuda_receipt(receipt_path, artifact)
            registry_path = evidence_root / "registry.json"
            registry(registry_path, candidate, [entry(receipt_path, receipt)])
            report = evaluate(registry_path, candidate, document)
            backend = next(row for row in report["rows"] if row["id"] == "native-backends")
            self.assertEqual(backend["status"], "MISSING")
            self.assertEqual(backend["satisfied_kinds"], ["cuda-training"])
            self.assertEqual(backend["missing_kinds"], ["backend-manifest", "performance"])

    def test_rejects_unknown_kind_stale_candidate_and_artifact_drift(self):
        for mutation in ("kind", "candidate", "artifact"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                candidate, document, artifact, evidence_root = release_fixture(Path(raw))
                receipt_path = evidence_root / "cuda.json"
                receipt = cuda_receipt(receipt_path, artifact)
                receipt_entry = entry(receipt_path, receipt)
                if mutation == "kind":
                    receipt_entry["kind"] = "self-asserted-pass"
                registry_path = evidence_root / "registry.json"
                registry(registry_path, candidate, [receipt_entry])
                if mutation == "candidate":
                    candidate.write_bytes(candidate.read_bytes() + b" ")
                elif mutation == "artifact":
                    artifact.write_bytes(b"changed")
                with self.assertRaises(EvidenceError):
                    evaluate(registry_path, candidate, document)

    def test_rejects_missing_parent_and_duplicate_run_id(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, artifact, evidence_root = release_fixture(Path(raw))
            first_path = evidence_root / "first.json"
            first = cuda_receipt(first_path, artifact)
            registry_path = evidence_root / "registry.json"
            registry(
                registry_path,
                candidate,
                [entry(first_path, first, parents=["sha256:" + "f" * 64])],
            )
            with self.assertRaisesRegex(EvidenceError, "unknown parent"):
                evaluate(registry_path, candidate, document)

            second_path = evidence_root / "second.json"
            second = cuda_receipt(second_path, artifact, run_id=first["run_id"])
            second["duration_ms"] = 5000.0
            unsigned = dict(second)
            unsigned.pop("receipt_id")
            second["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
            second_path.write_bytes(canonical(second) + b"\n")
            registry(
                registry_path,
                candidate,
                [entry(first_path, first), entry(second_path, second)],
            )
            with self.assertRaisesRegex(EvidenceError, "duplicate run id"):
                evaluate(registry_path, candidate, document)


if __name__ == "__main__":
    unittest.main()
