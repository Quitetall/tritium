from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-zoo-community.py")
assemble = MODULE["assemble"]
QualificationError = MODULE["QualificationError"]


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def fixture(root: Path):
    repo = root / "repo"
    repo.mkdir()
    for relative in (
        *MODULE["CLAIM_DOCUMENTS"], *MODULE["GOVERNANCE_FILES"],
        "scripts/generate-release-claims.py",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")

    candidate_root = root / "candidate"
    candidate_root.mkdir()
    wheel = candidate_root / "tritium.whl"
    wheel.write_bytes(b"wheel")
    artifacts = [{
        "id": "anchor-wheel", "kind": "python-wheel", "path": wheel.name,
        "identity": {
            "bytes": wheel.stat().st_size,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        },
    }]
    for ordinal in range(4):
        artifacts.append({
            "id": f"model-{ordinal}", "kind": "model-bundle",
            "path": f"model-{ordinal}.salt", "identity": {},
        })
    candidate = candidate_root / "manifest.json"
    write(candidate, {
        "schema": "tritium.release-candidate.v1", "release": "1.1.0-rc.0",
        "source_revision": "a" * 40, "artifacts": artifacts,
    })

    evidence_root = root / "evidence"
    evidence_root.mkdir()
    entries = []
    evidence_ids = []
    for ordinal in range(4):
        evidence_id = digest(f"evidence-{ordinal}")
        evidence_ids.append(evidence_id)
        receipt = evidence_root / f"receipt-{ordinal}.json"
        write(receipt, {
            "receipt_id": evidence_id, "result": "pass",
            "release": "1.1.0-rc.0", "source_revision": "a" * 40,
        })
        entries.append({
            "id": evidence_id, "kind": "clean-install",
            "path": receipt.name,
            "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
            "artifact_id": f"model-{ordinal}", "parents": [],
        })
    registry = evidence_root / "registry.json"
    write(registry, {
        "schema": MODULE["REGISTRY_SCHEMA"], "release": "1.1.0-rc.0",
        "source_revision": "a" * 40,
        "candidate_manifest_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "receipts": entries,
    })

    model_root = root / "models"
    model_root.mkdir()
    models = []
    for ordinal, (tier, role, model_id) in enumerate(MODULE["EXPECTED_MODELS"]):
        card = model_root / f"card-{ordinal}.md"
        card.write_text(f"# {model_id}\n", encoding="utf-8")
        models.append({
            "tier": tier, "role": role, "model_id": model_id,
            "revision": f"pinned-{ordinal}",
            "tokenizer_sha256": digest(f"tokenizer-{ordinal}"),
            "license": "Apache-2.0", "card": card.name,
            "artifact_ids": [f"model-{ordinal}"],
            "evidence_receipt_ids": [evidence_ids[ordinal]],
        })
    models_path = model_root / "models.json"
    write(models_path, {
        "schema": MODULE["SOURCE_SCHEMA"], "release": "1.1.0-rc.0",
        "source_revision": "a" * 40, "models": models,
    })

    review = root / "governance-review.json"
    write(review, {
        "schema": MODULE["REVIEW_SCHEMA"], "release": "1.1.0-rc.0",
        "source_revision": "a" * 40,
        "reviewed_at_utc": "2026-07-22T12:00:00Z",
        "reviewer": {"id": "reviewer-1", "organization": "Independent Lab"},
        "reviewed_files": list(MODULE["GOVERNANCE_FILES"]),
        "repository_links_checked": True, "contacts_checked": True,
        "independent_from_maintainers": True,
        "unstaffed_channels_advertised": False, "result": "pass",
    })
    return repo, candidate, wheel, registry, models_path, review


class QualifyZooCommunityTests(unittest.TestCase):
    def setUp(self):
        self.original_check = assemble.__globals__["check_claim_documents"]
        assemble.__globals__["check_claim_documents"] = lambda repo: None

    def tearDown(self):
        assemble.__globals__["check_claim_documents"] = self.original_check

    def test_assembles_three_self_validating_receipts(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo, candidate, wheel, registry, models, review = fixture(root)
            output = root / "output"
            receipts = assemble(
                output, repo=repo, candidate=candidate, anchor=wheel,
                registry_path=registry, models_path=models, review_path=review,
                source_revision="a" * 40, release="1.1.0-rc.0", run_id="run-7",
            )
            self.assertEqual(set(receipts), {"model_zoo", "generated_claims", "governance"})
            self.assertEqual(
                receipts["generated_claims"]["source_receipt_ids"][0],
                receipts["model_zoo"]["receipt_id"],
            )
            self.assertTrue((output / "support/source-registry.json").is_file())
            self.assertTrue((output / "support/governance-review.json").is_file())

    def test_rejects_evidence_absent_from_registry(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo, candidate, wheel, registry, models, review = fixture(root)
            document = json.loads(models.read_bytes())
            document["models"][0]["evidence_receipt_ids"] = [digest("absent")]
            write(models, document)
            with self.assertRaisesRegex(QualificationError, "absent from source registry"):
                assemble(
                    root / "output", repo=repo, candidate=candidate, anchor=wheel,
                    registry_path=registry, models_path=models, review_path=review,
                    source_revision="a" * 40, release="1.1.0-rc.0", run_id="run-7",
                )

    def test_rejects_unreviewed_governance(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo, candidate, wheel, registry, models, review = fixture(root)
            document = json.loads(review.read_bytes())
            document["independent_from_maintainers"] = False
            write(review, document)
            with self.assertRaisesRegex(QualificationError, "independent_from_maintainers"):
                assemble(
                    root / "output", repo=repo, candidate=candidate, anchor=wheel,
                    registry_path=registry, models_path=models, review_path=review,
                    source_revision="a" * 40, release="1.1.0-rc.0", run_id="run-7",
                )


if __name__ == "__main__":
    unittest.main()
