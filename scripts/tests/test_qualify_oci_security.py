from __future__ import annotations

import hashlib
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-oci-security.py")
SecurityScanError = MODULE["SecurityScanError"]
canonical = MODULE["canonical"]
report_findings = MODULE["report_findings"]
validate_receipt = MODULE["validate_receipt"]


def receipt(artifact: Path) -> dict:
    command = [
        "/usr/bin/trivy", "--cache-dir", "/cache", "image", "--input",
        str(artifact.resolve()), "--format", "json", "--offline-scan",
        "--skip-db-update", "--skip-java-db-update", "--skip-check-update",
    ]
    value = {
        "schema": MODULE["SCHEMA"], "release": "1.1.0-rc.0",
        "source_revision": "a" * 40, "run_id": "security-cpu-1", "flavor": "cpu",
        "started_at_utc": "2026-07-21T12:00:00+00:00", "duration_ms": 100.0,
        "artifact": {"kind": "oci-image", "name": artifact.name,
                     "bytes": artifact.stat().st_size,
                     "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        "scanner": {"name": "trivy", "version": "0.69.1",
                    "executable_sha256": "b" * 64,
                    "commands": [
                        command + ["--scanners", "vuln", "--severity", "HIGH,CRITICAL",
                                   "--output", "/tmp/vulnerability.json"],
                        command + ["--scanners", "secret", "--output", "/tmp/secret.json"],
                    ]},
        "database": {"updated_at": "2026-07-21T06:00:00Z",
                     "downloaded_at": "2026-07-21T06:01:00Z",
                     "next_update": "2026-07-21T12:00:01Z",
                     "trivy_db_sha256": "c" * 64, "metadata_sha256": "d" * 64,
                     "max_age_hours": 24.0},
        "findings": {"high_or_critical_vulnerabilities": 0, "secret_findings": 0},
        "result": "pass",
    }
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    return value


class QualifyOciSecurityTests(unittest.TestCase):
    def test_report_parser_counts_each_scanner(self):
        vulnerability = {"SchemaVersion": 2, "ArtifactType": "container_image",
                         "Results": [{"Vulnerabilities": [{"Severity": "HIGH"}]}]}
        secret = {"SchemaVersion": 2, "ArtifactType": "container_image",
                  "Results": [{"Secrets": [{"RuleID": "aws-access-key-id"}]}]}
        self.assertEqual(report_findings(vulnerability, "vulnerability"), 1)
        self.assertEqual(report_findings(secret, "secret"), 1)

    def test_validator_accepts_zero_finding_candidate_bound_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            artifact = Path(raw) / "image.oci.tar"
            artifact.write_bytes(b"qualified image")
            value = receipt(artifact)
            self.assertEqual(
                validate_receipt(
                    value, artifact_path=artifact, revision="a" * 40,
                    release="1.1.0-rc.0",
                )["result"],
                "pass",
            )

    def test_validator_rejects_findings_stale_db_and_artifact_drift(self):
        for mutation in ("finding", "database", "artifact"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                artifact = Path(raw) / "image.oci.tar"
                artifact.write_bytes(b"qualified image")
                value = receipt(artifact)
                if mutation == "finding":
                    value["findings"]["secret_findings"] = 1
                elif mutation == "database":
                    value["database"]["updated_at"] = "2026-07-01T00:00:00Z"
                else:
                    artifact.write_bytes(b"different image")
                unsigned = dict(value)
                unsigned.pop("receipt_id")
                value["receipt_id"] = "sha256:" + hashlib.sha256(
                    canonical(unsigned)
                ).hexdigest()
                with self.assertRaises(SecurityScanError):
                    validate_receipt(
                        value, artifact_path=artifact, revision="a" * 40,
                        release="1.1.0-rc.0",
                    )

    def test_loader_rejects_non_json(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifact = root / "image.oci.tar"
            artifact.write_bytes(b"qualified image")
            path = root / "receipt.json"
            path.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(SecurityScanError, "UTF-8 JSON"):
                MODULE["load_receipt"](
                    path, artifact_path=artifact, revision="a" * 40,
                    release="1.1.0-rc.0",
                )


if __name__ == "__main__":
    unittest.main()
