from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "verify-api-signature-receipt.py")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def make_receipt(root: Path):
    wheel = root / "candidate.whl"
    wheel.write_bytes(b"candidate wheel")
    report_path = ROOT / "docs/generated/api-diff-v1.0-v1.1.json"
    report = json.loads(report_path.read_bytes())
    python = report["python"]
    api_report = {
        "baseline": report["baseline"],
        "candidate_version": report["candidate_version"],
        "report_id": report["report_id"],
        "root_exports": sorted([*python["retained"], *python["added"]]),
    }
    revision = "a" * 40
    release = "1.1.0-rc.1"
    runtime = {
        "python_version": "3.13.5",
        "distribution_version": "1.1.0rc1",
        "source_identity": f"source-git:{revision}",
        "module_path": "/venv/site-packages/tritium/__init__.py",
        "native_module_path": "/venv/site-packages/tritium/_tritium.abi3.so",
        "wheel_file_count": 3,
        "installed_file_count": 3,
        "installed_tree_sha256": "sha256:" + "1" * 64,
    }
    environment = {
        "source_tree_absent": True,
        "compiler_absent": True,
        "network_mode": "offline",
    }
    signature = {
        "root_exports": api_report["root_exports"],
        "callable_signatures": {},
        "opaque_callables": [],
    }
    trace = {
        "schema": "tritium.installed-api-signature-trace.v1",
        "result": "complete",
        "release": release,
        "source_revision": revision,
        "run_id": "api-run-1",
        "wheel": {
            "name": wheel.name,
            "bytes": wheel.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest(),
        },
        "api_report": api_report,
        "runtime": runtime,
        "environment": environment,
        "signature": signature,
    }
    trace_path = root / "trace.json"
    trace_path.write_bytes(canonical(trace) + b"\n")
    receipt = {
        "schema": "tritium.installed-api-signature.v1",
        "receipt_id": "",
        "result": "pass",
        "release": release,
        "source_revision": revision,
        "run_id": trace["run_id"],
        "wheel": trace["wheel"],
        "api_report": api_report,
        "runtime": runtime,
        "environment": environment,
        "signature": signature,
        "trace": {
            "path": trace_path.name,
            "bytes": trace_path.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        },
    }
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    receipt_path = root / "receipt.json"
    receipt_path.write_bytes(canonical(receipt) + b"\n")
    return receipt_path, wheel, report_path


class ApiSignatureReceiptTests(unittest.TestCase):
    def test_release_registry_dispatches_api_signature(self):
        status = runpy.run_path(ROOT / "scripts/release-evidence-status.py")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            receipt, wheel, report = make_receipt(root)
            candidate = root / "manifest.json"
            document = {
                "schema": "tritium.release-candidate.v1",
                "release": "1.1.0-rc.1",
                "source_revision": "a" * 40,
                "artifacts": [{
                    "id": "cuda-wheel", "kind": "python-wheel", "path": wheel.name,
                    "identity": {"sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                                  "bytes": wheel.stat().st_size},
                    "sbom": {}, "provenance": {},
                }],
            }
            candidate.write_bytes(canonical(document) + b"\n")
            registry = root / "registry.json"
            record = json.loads(receipt.read_bytes())
            registry.write_bytes(canonical({
                "schema": "tritium.release-evidence-registry.v1",
                "release": document["release"],
                "source_revision": document["source_revision"],
                "candidate_manifest_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "receipts": [{
                    "id": record["receipt_id"], "kind": "api-signature",
                    "path": receipt.name,
                    "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                    "artifact_id": "cuda-wheel", "parents": [],
                }],
            }) + b"\n")
            report = status["evaluate"](registry, candidate, document)
            frontend = next(row for row in report["rows"] if row["id"] == "pytorch-hf")
            self.assertEqual(frontend["satisfied_kinds"], ["api-signature"])

    def test_accepts_exact_bound_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            receipt, wheel, report = make_receipt(Path(raw))
            value = MODULE["validate"](
                receipt,
                expected_revision="a" * 40,
                expected_release="1.1.0-rc.1",
                expected_wheel=wheel,
                expected_api_report=report,
            )
            self.assertEqual(value["result"], "pass")

    def test_rejects_root_namespace_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            receipt, wheel, report = make_receipt(Path(raw))
            value = json.loads(receipt.read_bytes())
            value["signature"]["root_exports"] = value["signature"]["root_exports"][:-1]
            value["receipt_id"] = "sha256:" + hashlib.sha256(
                canonical({key: item for key, item in value.items() if key != "receipt_id"})
            ).hexdigest()
            receipt.write_bytes(canonical(value) + b"\n")
            with self.assertRaisesRegex(MODULE["ApiSignatureError"], "namespace"):
                MODULE["validate"](
                    receipt, expected_revision="a" * 40, expected_release="1.1.0-rc.1",
                    expected_wheel=wheel, expected_api_report=report,
                )

    def test_rejects_trace_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            receipt, wheel, report = make_receipt(Path(raw))
            trace = Path(raw) / "trace.json"
            trace.write_bytes(trace.read_bytes() + b"drift")
            with self.assertRaisesRegex(MODULE["ApiSignatureError"], "trace bytes"):
                MODULE["validate"](
                    receipt, expected_revision="a" * 40, expected_release="1.1.0-rc.1",
                    expected_wheel=wheel, expected_api_report=report,
                )

    def test_rejects_non_offline_environment(self):
        with tempfile.TemporaryDirectory() as raw:
            receipt, wheel, report = make_receipt(Path(raw))
            value = json.loads(receipt.read_bytes())
            value["environment"]["network_mode"] = "online"
            value["receipt_id"] = "sha256:" + hashlib.sha256(
                canonical({key: item for key, item in value.items() if key != "receipt_id"})
            ).hexdigest()
            receipt.write_bytes(canonical(value) + b"\n")
            with self.assertRaisesRegex(MODULE["ApiSignatureError"], "offline"):
                MODULE["validate"](
                    receipt, expected_revision="a" * 40, expected_release="1.1.0-rc.1",
                    expected_wheel=wheel, expected_api_report=report,
                )

    def test_rejects_malformed_api_report_with_typed_error(self):
        with tempfile.TemporaryDirectory() as raw:
            report = json.loads(
                (ROOT / "docs/generated/api-diff-v1.0-v1.1.json").read_bytes()
            )
            report["python"]["added"] = "not-a-list"
            report["report_id"] = "sha256:" + hashlib.sha256(
                canonical({key: item for key, item in report.items() if key != "report_id"})
            ).hexdigest()
            path = Path(raw) / "api.json"
            path.write_bytes(canonical(report) + b"\n")
            with self.assertRaisesRegex(MODULE["ApiSignatureError"], "lists"):
                MODULE["_report"](path, "1.1.0-rc.1")


if __name__ == "__main__":
    unittest.main()
