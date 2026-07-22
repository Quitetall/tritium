from __future__ import annotations

from pathlib import Path
import runpy
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-oci-runtime.py")
QualificationError = MODULE["QualificationError"]
atomic_create = MODULE["atomic_create"]
canonical = MODULE["canonical"]
validate_ready = MODULE["validate_ready"]
validate_receipt = MODULE["validate_receipt"]
request_json = MODULE["request_json"]
request_error = MODULE["request_error"]


def readiness(flavor: str = "cpu"):
    return {
        "status": "ready",
        "release_gate": "production_artifact_admitted",
        "startup_receipt": {
            "schema_version": 1,
            "artifact_kind": "qwen3.6-language-mtp-salt-v2-hf-bundle",
            "server_source_revision": "a" * 40,
            "server_build_id": "tritium-serve:1.1.0-rc.0:" + "a" * 40,
            "model_source_revision": "b" * 40,
            "manifest_package_id": "c" * 64,
            "salt_package_id": "d" * 64,
            "preserved_package_id": "e" * 64,
            "config_package_id": "f" * 64,
            "profile": "compact-v1",
            "codec": "b3",
            "backend_policy": flavor,
            "effective_backend": flavor,
            "physical_device_id": (
                "cpu" if flavor == "cpu" else "cuda:0:GPU-physical"
            ),
            "loaded_bundle_bytes": 100,
            "resident_bytes": 80,
            "self_test_digest": "1" * 64,
        },
    }


def runtime_receipt(artifact: Path, flavor: str = "cpu") -> dict:
    import hashlib

    value = {
        "schema": MODULE["SCHEMA"], "release": "1.1.0-rc.0",
        "source_revision": "a" * 40, "run_id": f"{flavor}-1", "flavor": flavor,
        "image": "example@sha256:" + "b" * 64,
        "image_id": "sha256:" + "c" * 64,
        "image_manifest_digest": "sha256:" + "b" * 64,
        "artifact": {"kind": "oci-image", "name": artifact.name,
                     "bytes": artifact.stat().st_size,
                     "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        "manifest": {"schema": "tritium.file-identity.v1", "bytes": 42,
                     "sha256": "2" * 64, "blake3": "c" * 64},
        "profile": "compact-v1", "startup_receipt": readiness(flavor)["startup_receipt"],
        "faults": {
            "unauthenticated_status": 401, "wrong_token_status": 401,
            "malformed_json_status": 400, "rate_limited_status": 429,
            "retry_after_seconds": 60, "rate_rejections_before": 0,
            "rate_rejections_after": 1, "replacement_principal_status": 400,
            "queue_flood_clients": 3, "slow_reader_tokens": 32,
            "queue_rejections_before": 0, "queue_rejections_after": 1,
            "disconnects_before": 0, "disconnects_after": 1,
            "accepted_streams": 2, "rejected_streams": 1,
            "settled_queue_depth": 0, "worker_alive": 1,
            "queue_capacity": 1, "saturated_queue_depth": 1, "slow_hold_ms": 1000,
            "tokens_out_before_hold": 0, "tokens_out_after_hold": 1,
            "recovery_status": 200, "recovery_ms": 1,
            "recovery_timeout_ms": 60000,
        },
        "checks": list(MODULE["CHECKS"]),
        "started_at_utc": "2026-07-21T00:00:00+00:00",
        "timing": {"startup_ms": 10.0, "shutdown_ms": 5.0},
        "machine": {"id": "sha256:" + "5" * 64, "system": "Linux",
                    "architecture": "x86_64", "docker_server": "28.0.0",
                    "gpu": None if flavor == "cpu" else {
                        "uuid": "GPU-physical", "name": "RTX 4090",
                        "driver_version": "610.43.03",
                    }},
        "result": "pass",
    }
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    return value


class QualifyOciRuntimeTests(unittest.TestCase):
    def test_error_client_requires_stable_json_type_and_headers(self):
        class Response:
            status = 429
            headers = {"Retry-After": "60"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return (
                    b'{"error":{"message":"principal request rate exceeded; retry later",'
                    b'"type":"rate_limit_exceeded"}}'
                )

        with mock.patch.object(MODULE["urllib"].request, "urlopen", return_value=Response()):
            _, headers = request_error(
                "http://127.0.0.1/v1/chat/completions", 429,
                "rate_limit_exceeded",
                "principal request rate exceeded; retry later",
                token="token", body=b"{",
            )
        self.assertEqual(headers["retry-after"], "60")

    def test_json_client_rejects_oversized_response(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return b"x" * (MODULE["MAX_JSON_RESPONSE_BYTES"] + 1)

        with mock.patch.object(MODULE["urllib"].request, "urlopen", return_value=Response()):
            with self.assertRaisesRegex(QualificationError, "byte limit"):
                request_json("http://127.0.0.1/readyz", "token")

    def test_accepts_exact_production_readiness(self):
        receipt = validate_ready(
            readiness(), "a" * 40, "cpu", "compact-v1", "c" * 64, "1.1.0-rc.0"
        )
        self.assertEqual(receipt["resident_bytes"], 80)

    def test_rejects_legacy_and_cross_artifact_readiness(self):
        value = readiness()
        value["release_gate"] = "legacy_compatibility"
        with self.assertRaisesRegex(QualificationError, "production artifact"):
            validate_ready(value, "a" * 40, "cpu", "compact-v1", "c" * 64)
        value = readiness()
        with self.assertRaisesRegex(QualificationError, "artifact identity"):
            validate_ready(value, "a" * 40, "cpu", "compact-v1", "0" * 64)

    def test_atomic_receipt_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "receipt.json"
            atomic_create(path, b"first\n")
            with self.assertRaisesRegex(QualificationError, "overwrite"):
                atomic_create(path, b"second\n")
            self.assertEqual(path.read_bytes(), b"first\n")

    def test_receipt_validator_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as raw:
            artifact = Path(raw) / "image.oci.tar"
            artifact.write_bytes(b"qualified OCI bytes")
            receipt = runtime_receipt(artifact)
            validate_receipt(
                receipt, revision="a" * 40, release="1.1.0-rc.0",
                artifact_path=artifact,
            )
            receipt["run_id"] = "tampered"
            with self.assertRaisesRegex(QualificationError, "content digest"):
                validate_receipt(receipt)

    def test_receipt_validator_rejects_noncausal_fault_evidence(self):
        with tempfile.TemporaryDirectory() as raw:
            artifact = Path(raw) / "image.oci.tar"
            artifact.write_bytes(b"qualified OCI bytes")
            receipt = runtime_receipt(artifact)
            receipt["faults"]["rate_rejections_after"] = 2
            import hashlib
            del receipt["receipt_id"]
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
            with self.assertRaisesRegex(QualificationError, "fault evidence"):
                validate_receipt(receipt)

            receipt = runtime_receipt(artifact)
            receipt["faults"]["disconnects_after"] = 0
            del receipt["receipt_id"]
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
            with self.assertRaisesRegex(QualificationError, "queue/disconnect evidence"):
                validate_receipt(receipt)

            for field, value in (
                ("queue_capacity", 2),
                ("tokens_out_after_hold", 0),
                ("recovery_ms", 60001),
            ):
                receipt = runtime_receipt(artifact)
                receipt["faults"][field] = value
                del receipt["receipt_id"]
                receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                    canonical(receipt)
                ).hexdigest()
                with self.assertRaisesRegex(
                    QualificationError, "queue/disconnect evidence"
                ):
                    validate_receipt(receipt)

    def test_receipt_validator_rejects_cross_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as raw:
            artifact = Path(raw) / "image.oci.tar"
            artifact.write_bytes(b"wrong bytes")
            receipt = runtime_receipt(artifact)
            receipt["artifact"]["bytes"] = 4
            receipt["artifact"]["sha256"] = "0" * 64
            import hashlib
            del receipt["receipt_id"]
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
            with self.assertRaisesRegex(QualificationError, "candidate OCI bytes"):
                validate_receipt(receipt, artifact_path=artifact)

    def test_cuda_receipt_binds_startup_to_physical_gpu_uuid(self):
        with tempfile.TemporaryDirectory() as raw:
            artifact = Path(raw) / "image.oci.tar"
            artifact.write_bytes(b"qualified OCI bytes")
            receipt = runtime_receipt(artifact, "cuda")
            validate_receipt(receipt, artifact_path=artifact)
            receipt["startup_receipt"]["physical_device_id"] = "cuda:0:GPU-other"
            import hashlib
            del receipt["receipt_id"]
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
            with self.assertRaisesRegex(QualificationError, "physical NVIDIA UUID"):
                validate_receipt(receipt, artifact_path=artifact)


if __name__ == "__main__":
    unittest.main()
