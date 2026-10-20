from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-oci-security.py")
SecurityScanError = MODULE["SecurityScanError"]
canonical = MODULE["canonical"]
report_findings = MODULE["report_findings"]
validate_receipt = MODULE["validate_receipt"]


class ScannerExecutionTests(unittest.TestCase):
    def test_real_scanner_failure_withholds_untrusted_stderr(self):
        marker = "fixture-only-sensitive-scanner-output"
        command = [sys.executable, "-c", "import sys; sys.stderr.write(sys.argv[1]); sys.exit(7)", marker]
        with self.assertRaises(SecurityScanError) as caught:
            MODULE["run"](command, timeout=5)
        self.assertIn("7", str(caught.exception))
        self.assertNotIn(marker, "".join(traceback.format_exception(caught.exception)))

    def test_launch_and_timeout_errors_withhold_exception_context(self):
        marker = "fixture-only-sensitive-scanner-exception"
        command = ["/private/fixture-only-scanner", marker]
        failures = (
            OSError(marker),
            subprocess.TimeoutExpired(command, 1, output=marker.encode(), stderr=marker.encode()),
            subprocess.SubprocessError(marker),
        )
        for failure in failures:
            with self.subTest(kind=type(failure).__name__):
                with mock.patch.object(MODULE["subprocess"], "run", side_effect=failure):
                    with self.assertRaises(SecurityScanError) as caught:
                        MODULE["run"](command, timeout=1)
                diagnostic = "".join(traceback.format_exception(caught.exception))
                self.assertNotIn(marker, diagnostic)
                self.assertNotIn(command[0], diagnostic)

    def test_real_timeout_is_bounded_and_withholds_captured_output(self):
        marker = "fixture-only-sensitive-timeout-output"
        command = [
            sys.executable, "-c",
            "import sys,time; print(sys.argv[1], flush=True); "
            "sys.stderr.write(sys.argv[1]); sys.stderr.flush(); time.sleep(10)",
            marker,
        ]
        started = time.monotonic()
        with self.assertRaises(SecurityScanError) as caught:
            MODULE["run"](command, timeout=0.1)
        self.assertLess(time.monotonic() - started, 3)
        self.assertNotIn(marker, "".join(traceback.format_exception(caught.exception)))

    def test_successful_scanner_metadata_is_unchanged(self):
        self.assertEqual(
            MODULE["run"]([sys.executable, "-c", "print('fixture-scanner-metadata')"], timeout=5),
            "fixture-scanner-metadata\n",
        )

    def test_successful_scanner_report_is_unchanged(self):
        with tempfile.TemporaryDirectory() as raw:
            report = Path(raw) / "report.json"
            content = '{"SchemaVersion":2,"Results":[]}\n'
            report.write_text(content, encoding="utf-8")
            command = ["fixture-scanner", "--output", str(report)]
            result = subprocess.CompletedProcess(command, 0, None, "")
            with mock.patch.object(MODULE["subprocess"], "run", return_value=result) as launch:
                self.assertEqual(MODULE["run"](command, timeout=5, output=report), content)
            self.assertEqual(launch.call_args.kwargs["stdout"], subprocess.DEVNULL)

    def test_invalid_timeout_is_rejected_before_launch(self):
        for timeout in (float("nan"), float("inf"), float("-inf"), 0, -1, True):
            with self.subTest(timeout=timeout):
                with mock.patch.object(MODULE["subprocess"], "run") as launch:
                    with self.assertRaisesRegex(SecurityScanError, "timeout"):
                        MODULE["run"](["fixture-scanner"], timeout=timeout)
                    launch.assert_not_called()

    def test_invalid_qualification_limits_fail_before_artifact_or_scanner_work(self):
        qualify = MODULE["qualify"]
        for field, values in (
            ("timeout", (float("nan"), float("inf"), float("-inf"), 0, -1, True)),
            ("max_db_age_hours", (float("nan"), float("inf"), 0, -1, 25, True)),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    args = argparse.Namespace(
                        flavor="cpu", release="1.1.0-rc.2", run_id="fixture-only-scan",
                        source_revision="a" * 40, timeout=1, max_db_age_hours=24,
                        archive=Path("/fixture-only-not-read.oci.tar"),
                    )
                    setattr(args, field, value)
                    ordinary = mock.Mock(side_effect=SecurityScanError("unexpected artifact access"))
                    with mock.patch.dict(qualify.__globals__, {"ordinary": ordinary}):
                        with self.assertRaisesRegex(SecurityScanError, "scan limits"):
                            qualify(args)
                        ordinary.assert_not_called()


def receipt(artifact: Path) -> dict:
    command = [
        "/usr/bin/trivy", "--cache-dir", "/cache", "image", "--input",
        "<layout>", "--format", "json", "--offline-scan",
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
