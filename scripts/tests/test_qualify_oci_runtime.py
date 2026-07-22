from __future__ import annotations

from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-oci-runtime.py")
QualificationError = MODULE["QualificationError"]
atomic_create = MODULE["atomic_create"]
canonical = MODULE["canonical"]
validate_ready = MODULE["validate_ready"]
validate_receipt = MODULE["validate_receipt"]


def readiness():
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
            "backend_policy": "cpu",
            "effective_backend": "cpu",
            "physical_device_id": "cpu",
            "loaded_bundle_bytes": 100,
            "resident_bytes": 80,
            "self_test_digest": "1" * 64,
        },
    }


def runtime_receipt(artifact: Path) -> dict:
    import hashlib

    value = {
        "schema": "tritium.oci-runtime-qualification.v1", "release": "1.1.0-rc.0",
        "source_revision": "a" * 40, "run_id": "cpu-1", "flavor": "cpu",
        "image": "example@sha256:" + "b" * 64,
        "image_id": "sha256:" + "c" * 64,
        "image_manifest_digest": "sha256:" + "b" * 64,
        "artifact": {"kind": "oci-image", "name": artifact.name,
                     "bytes": artifact.stat().st_size,
                     "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        "manifest": {"schema": "tritium.file-identity.v1", "bytes": 42,
                     "sha256": "2" * 64, "blake3": "c" * 64},
        "profile": "compact-v1", "startup_receipt": readiness()["startup_receipt"],
        "checks": list(MODULE["CHECKS"]),
        "started_at_utc": "2026-07-21T00:00:00+00:00",
        "timing": {"startup_ms": 10.0, "shutdown_ms": 5.0},
        "machine": {"id": "sha256:" + "5" * 64, "system": "Linux",
                    "architecture": "x86_64", "docker_server": "28.0.0", "gpu": None},
        "result": "pass",
    }
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    return value


class QualifyOciRuntimeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
