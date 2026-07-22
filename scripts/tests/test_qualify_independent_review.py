from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-independent-review.py")
assemble = MODULE["assemble"]
QualificationError = MODULE["QualificationError"]


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def fixture(root: Path):
    wheel = root / "candidate.whl"
    wheel.write_bytes(b"wheel")
    candidate = root / "manifest.json"
    write(candidate, {
        "release": "1.1.0-rc.0", "source_revision": "a" * 40,
        "artifacts": [{
            "id": "wheel", "kind": "python-wheel", "path": wheel.name,
            "identity": {
                "bytes": wheel.stat().st_size,
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            },
        }],
    })
    reviewed = ["sha256:" + f"{ordinal:064x}" for ordinal in range(1, 32)]
    scope = "9" * 64
    registry = root / "registry.json"
    write(registry, {"placeholder": True})
    attestation = root / "attestation.json"
    value = {
        "schema": MODULE["REVIEW_ATTESTATION_SCHEMA"],
        "release": "1.1.0-rc.0", "source_revision": "a" * 40,
        "run_id": "independent-review-3",
        "reviewer": {
            "id": "reviewer-3", "organization": "Independent Audit Lab",
            "independent": True, "tool": "human+static", "model": "none",
        },
        "candidate_manifest_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "review_scope_sha256": scope, "reviewed_receipt_ids": reviewed,
        "scopes": ["code", "security", "evidence"],
        "findings": {
            "total": 3, "verified": 2, "fixed": 2,
            "false_positive": 1, "open": 0,
        },
        "verdict": "pass",
    }
    write(attestation, value)
    return candidate, wheel, registry, attestation, reviewed, scope


class QualifyIndependentReviewTests(unittest.TestCase):
    def setUp(self):
        self.original = assemble.__globals__["validate_registry"]
        self.original_evaluate = self.original.__globals__["evaluate"]

    def tearDown(self):
        assemble.__globals__["validate_registry"] = self.original
        self.original.__globals__["evaluate"] = self.original_evaluate

    def test_pre_review_registry_requires_all_31_kinds(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, _, registry, _, _, _ = fixture(root)
            entries = []
            for ordinal, kind in enumerate(
                sorted(MODULE["KNOWN_KINDS"] - {"independent-review"}), start=1
            ):
                entries.append({
                    "id": "sha256:" + f"{ordinal:064x}", "kind": kind,
                    "path": f"{ordinal}.json", "sha256": f"{ordinal:064x}",
                    "artifact_id": "wheel", "parents": [],
                })
            write(registry, {
                "schema": MODULE["REGISTRY_SCHEMA"], "release": "1.1.0-rc.0",
                "source_revision": "a" * 40,
                "candidate_manifest_sha256": hashlib.sha256(
                    candidate.read_bytes()
                ).hexdigest(),
                "receipts": entries,
            })
            rows = [
                {
                    "id": gate[0],
                    "status": "MISSING" if gate[0] == "reproduction-signoff" else "PASS",
                    "satisfied_kinds": ["second-machine"]
                    if gate[0] == "reproduction-signoff" else [],
                    "missing_kinds": ["independent-review"]
                    if gate[0] == "reproduction-signoff" else [],
                    "structural_kinds": [],
                }
                for gate in MODULE["STATUS"]["GATES"]
            ]
            self.original.__globals__["evaluate"] = (
                lambda *args, **kwargs: {"rows": rows}
            )
            _, ids, scope = self.original(
                registry, candidate, revision="a" * 40,
                release="1.1.0-rc.0", digest_tool="tritium",
            )
            self.assertEqual(len(ids), 31)
            self.assertEqual(len(scope), 64)
            document = json.loads(registry.read_bytes())
            document["receipts"].pop()
            write(registry, document)
            with self.assertRaisesRegex(QualificationError, "every non-review kind"):
                self.original(
                    registry, candidate, revision="a" * 40,
                    release="1.1.0-rc.0", digest_tool="tritium",
                )

    def test_seals_exact_review_scope_and_retains_attestation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, wheel, registry, attestation, reviewed, scope = fixture(root)
            assemble.__globals__["validate_registry"] = (
                lambda *args, **kwargs: ({}, reviewed, scope)
            )
            output = root / "output"
            receipt = assemble(
                output, candidate=candidate, anchor=wheel,
                registry_path=registry, attestation_path=attestation,
                source_revision="a" * 40, release="1.1.0-rc.0",
                digest_tool="tritium",
            )
            self.assertEqual(receipt["reviewed_receipt_ids"], reviewed)
            self.assertEqual(receipt["review_scope_sha256"], scope)
            self.assertTrue((output / "support/review-attestation.json").is_file())

    def test_rejects_open_findings(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, wheel, registry, attestation, reviewed, scope = fixture(root)
            document = json.loads(attestation.read_bytes())
            document["findings"]["open"] = 1
            write(attestation, document)
            assemble.__globals__["validate_registry"] = (
                lambda *args, **kwargs: ({}, reviewed, scope)
            )
            with self.assertRaisesRegex(QualificationError, "unresolved findings"):
                assemble(
                    root / "output", candidate=candidate, anchor=wheel,
                    registry_path=registry, attestation_path=attestation,
                    source_revision="a" * 40, release="1.1.0-rc.0",
                    digest_tool="tritium",
                )


if __name__ == "__main__":
    unittest.main()
