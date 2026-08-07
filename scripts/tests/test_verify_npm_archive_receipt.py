from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "verify-npm-archive-receipt.py")
NpmReceiptError = MODULE["NpmReceiptError"]
validate_receipt = MODULE["validate_receipt"]


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def receipt_for(archive: Path, *, dirty: bool = False) -> dict:
    revision = "a" * 40
    payload = archive.read_bytes()
    value = {
        "schema": MODULE["SCHEMA"],
        "release": "1.1.0-rc.0",
        "source_revision": revision,
        "run_id": "npm-run-1",
        "started_at_utc": "2026-07-21T12:00:00Z",
        "duration_ms": 125.5,
        "machine": {
            "machine_id": "sha256:" + "b" * 64,
            "system": "linux",
            "architecture": "x64",
        },
        "toolchain": {"node": "v22.18.0", "npm": "11.5.2"},
        "artifact": {
            "kind": "npm-archive",
            "name": archive.name,
            "package": "@tritium-ai/web@1.1.0-rc.0",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "integrity": "sha512-" + base64.b64encode(
                hashlib.sha512(payload).digest()
            ).decode("ascii"),
        },
        "evidence": {
            "source_dirty": dirty,
            "entry_count": 16,
            "source_free": True,
            "installed_offline": True,
            "strict_typescript": True,
            "wasm_build_id": f"tritium-wasm@1.1.0-rc.0+source-git:{revision}",
            "wasm_guest_digest": "c" * 64,
        },
        "result": "pass",
    }
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    return value


class VerifyNpmArchiveReceiptTests(unittest.TestCase):
    def test_valid_receipt_binds_exact_clean_archive(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive = root / "tritium-ai-web-1.1.0-rc.0.tgz"
            archive.write_bytes(b"exact npm archive")
            receipt = receipt_for(archive)
            path = root / "receipt.json"
            path.write_bytes(canonical(receipt) + b"\n")
            self.assertEqual(
                validate_receipt(path, archive, "a" * 40, "1.1.0-rc.0"),
                receipt,
            )

    def test_rejects_dirty_source_archive_drift_and_receipt_tampering(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive = root / "tritium-ai-web-1.1.0-rc.0.tgz"
            archive.write_bytes(b"exact npm archive")
            path = root / "receipt.json"
            dirty = receipt_for(archive, dirty=True)
            path.write_bytes(canonical(dirty))
            with self.assertRaisesRegex(NpmReceiptError, "clean qualified install"):
                validate_receipt(path, archive, "a" * 40, "1.1.0-rc.0")
            clean = receipt_for(archive)
            path.write_bytes(canonical(clean))
            archive.write_bytes(b"tampered")
            with self.assertRaisesRegex(NpmReceiptError, "byte count|digest"):
                validate_receipt(path, archive, "a" * 40, "1.1.0-rc.0")
            archive.write_bytes(b"exact npm archive")
            clean["run_id"] = "tampered-run"
            path.write_bytes(canonical(clean))
            with self.assertRaisesRegex(NpmReceiptError, "receipt identity"):
                validate_receipt(path, archive, "a" * 40, "1.1.0-rc.0")


if __name__ == "__main__":
    unittest.main()
