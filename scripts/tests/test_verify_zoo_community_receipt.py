import hashlib
from pathlib import Path
import runpy
import tempfile
import unittest


MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "verify-zoo-community-receipt.py"
)
canonical = MODULE["canonical"]
validate_zoo = MODULE["validate_zoo"]
validate_claims = MODULE["validate_claims"]
validate_governance = MODULE["validate_governance"]
ZooCommunityError = MODULE["ZooCommunityError"]


def file_record(root: Path, relative: str):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(relative + "\n", encoding="utf-8")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def base(root: Path):
    wheel = root / "candidate.whl"
    wheel.write_bytes(b"wheel")
    candidate = root / "manifest.json"
    candidate.write_bytes(b'{"candidate":true}\n')
    common = {
        "result": "pass",
        "release": "1.1.0-rc.0",
        "source_revision": "a" * 40,
        "run_id": "run-1",
        "candidate_manifest_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "anchor_artifact": {
            "id": "wheel",
            "kind": "python-wheel",
            "name": wheel.name,
            "bytes": wheel.stat().st_size,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        },
    }
    return candidate, wheel, common


def seal(value):
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    return value


def write(path: Path, value):
    path.write_bytes(canonical(value) + b"\n")


def claim_snapshot(root: Path, evidence_ids):
    support = root / "support"
    evidence = support / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    entries = []
    for ordinal, evidence_id in enumerate(evidence_ids):
        receipt = evidence / f"{ordinal:02d}.json"
        write(receipt, {
            "receipt_id": evidence_id, "result": "pass",
            "release": "1.1.0-rc.0", "source_revision": "a" * 40,
        })
        entries.append({
            "id": evidence_id, "kind": "model-evidence",
            "artifact_id": f"artifact-{ordinal}",
            "receipt": {
                "path": f"evidence/{ordinal:02d}.json",
                "bytes": receipt.stat().st_size,
                "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
            },
        })
    snapshot = support / "source-registry.json"
    write(snapshot, {
        "schema": MODULE["CLAIM_SOURCE_SCHEMA"], "release": "1.1.0-rc.0",
        "source_revision": "a" * 40,
        "candidate_manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
        "registry_sha256": "9" * 64, "entries": entries,
    })
    return file_record(root, "support/source-registry.json")


def governance_review(root: Path, *, independent=True):
    path = root / "support/governance-review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, {
        "schema": MODULE["REVIEW_SCHEMA"], "release": "1.1.0-rc.0",
        "source_revision": "a" * 40,
        "reviewed_at_utc": "2026-07-22T12:00:00Z",
        "reviewer": {"id": "reviewer-1", "organization": "Independent Lab"},
        "reviewed_files": list(MODULE["GOVERNANCE_FILES"]),
        "repository_links_checked": True, "contacts_checked": True,
        "independent_from_maintainers": independent,
        "unstaffed_channels_advertised": False, "result": "pass",
    })
    return file_record(root, "support/governance-review.json")


class ZooCommunityReceiptTests(unittest.TestCase):
    def test_accepts_frozen_zoo_claim_and_governance_inventories(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, wheel, common = base(root)
            models = []
            for ordinal, (tier, role, model_id) in enumerate(MODULE["EXPECTED_MODELS"]):
                models.append(
                    {
                        "tier": tier,
                        "role": role,
                        "model_id": model_id,
                        "revision": f"revision-{ordinal}",
                        "tokenizer_sha256": "sha256:" + f"{ordinal + 1:x}" * 64,
                        "license": "Apache-2.0",
                        "card": file_record(root, f"cards/model-{ordinal}.md"),
                        "artifact_ids": [f"artifact-{ordinal}"],
                        "evidence_receipt_ids": ["sha256:" + f"{ordinal + 5:x}" * 64],
                    }
                )
            zoo = seal({**common, "schema": MODULE["ZOO_SCHEMA"], "models": models})
            zoo_path = root / "model-zoo.json"
            write(zoo_path, zoo)
            self.assertEqual(
                validate_zoo(zoo_path, "a" * 40, "1.1.0-rc.0", candidate, wheel), zoo
            )

            repo = root / "repo"
            repo.mkdir()
            documents = [file_record(repo, path) for path in MODULE["CLAIM_DOCUMENTS"]]
            generator_file = file_record(repo, "scripts/qualify-zoo-community.py")
            evidence_ids = [model["evidence_receipt_ids"][0] for model in models]
            source_registry = claim_snapshot(root, evidence_ids)
            claims = seal(
                {
                    **common,
                    "schema": MODULE["CLAIMS_SCHEMA"],
                    "run_id": "claims-1",
                    "generator_id": MODULE["GENERATOR_ID"],
                    "generator_file": generator_file,
                    "source_registry": source_registry,
                    "documents": documents,
                    "source_receipt_ids": [zoo["receipt_id"], *evidence_ids],
                }
            )
            claims_path = root / "claims.json"
            write(claims_path, claims)
            self.assertEqual(
                validate_claims(
                    claims_path, "a" * 40, "1.1.0-rc.0", candidate, wheel, repo
                ),
                claims,
            )
            (root / "support/evidence/00.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ZooCommunityError, "drifted"):
                validate_claims(
                    claims_path, "a" * 40, "1.1.0-rc.0", candidate, wheel, repo
                )

            files = [file_record(repo, path) for path in MODULE["GOVERNANCE_FILES"]]
            review_attestation = governance_review(root)
            governance = seal(
                {
                    **common,
                    "schema": MODULE["GOVERNANCE_SCHEMA"],
                    "run_id": "governance-1",
                    "files": files,
                    "review_attestation": review_attestation,
                    "repository_links_checked": True,
                    "contacts_checked": True,
                    "independent_policy_review": True,
                    "unstaffed_channels_advertised": False,
                }
            )
            governance_path = root / "governance.json"
            write(governance_path, governance)
            self.assertEqual(
                validate_governance(
                    governance_path, "a" * 40, "1.1.0-rc.0", candidate, wheel, repo
                ),
                governance,
            )

    def test_rejects_missing_model_evidence_claim_drift_and_unreviewed_policy(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, wheel, common = base(root)
            models = []
            for ordinal, (tier, role, model_id) in enumerate(MODULE["EXPECTED_MODELS"]):
                models.append(
                    {
                        "tier": tier,
                        "role": role,
                        "model_id": model_id,
                        "revision": f"revision-{ordinal}",
                        "tokenizer_sha256": "sha256:" + f"{ordinal + 1:x}" * 64,
                        "license": "Apache-2.0",
                        "card": file_record(root, f"cards/model-{ordinal}.md"),
                        "artifact_ids": [f"artifact-{ordinal}"],
                        "evidence_receipt_ids": ["sha256:" + f"{ordinal + 5:x}" * 64],
                    }
                )
            models[-1]["evidence_receipt_ids"] = []
            zoo = seal({**common, "schema": MODULE["ZOO_SCHEMA"], "models": models})
            zoo_path = root / "model-zoo.json"
            write(zoo_path, zoo)
            with self.assertRaisesRegex(ZooCommunityError, "evidence_receipt_ids"):
                validate_zoo(zoo_path, "a" * 40, "1.1.0-rc.0", candidate, wheel)

            models[-1]["evidence_receipt_ids"] = ["sha256:" + "8" * 64]
            zoo = seal({**common, "schema": MODULE["ZOO_SCHEMA"], "models": models})
            write(zoo_path, zoo)

            repo = root / "repo"
            repo.mkdir()
            documents = [file_record(repo, path) for path in MODULE["CLAIM_DOCUMENTS"]]
            generator_file = file_record(repo, "scripts/qualify-zoo-community.py")
            evidence_ids = [model["evidence_receipt_ids"][0] for model in models]
            source_registry = claim_snapshot(root, evidence_ids)
            claims = seal(
                {
                    **common,
                    "schema": MODULE["CLAIMS_SCHEMA"],
                    "run_id": "claims-1",
                    "generator_id": MODULE["GENERATOR_ID"],
                    "generator_file": generator_file,
                    "source_registry": source_registry,
                    "documents": documents,
                    "source_receipt_ids": [zoo["receipt_id"], *evidence_ids],
                }
            )
            claims_path = root / "claims.json"
            write(claims_path, claims)
            (repo / MODULE["CLAIM_DOCUMENTS"][0]).write_text(
                "drift\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ZooCommunityError, "drifted"):
                validate_claims(
                    claims_path, "a" * 40, "1.1.0-rc.0", candidate, wheel, repo
                )

            files = [file_record(repo, path) for path in MODULE["GOVERNANCE_FILES"]]
            review_attestation = governance_review(root)
            governance = seal(
                {
                    **common,
                    "schema": MODULE["GOVERNANCE_SCHEMA"],
                    "run_id": "governance-1",
                    "files": files,
                    "review_attestation": review_attestation,
                    "repository_links_checked": True,
                    "contacts_checked": True,
                    "independent_policy_review": False,
                    "unstaffed_channels_advertised": False,
                }
            )
            governance_path = root / "governance.json"
            write(governance_path, governance)
            with self.assertRaisesRegex(ZooCommunityError, "independent_policy_review"):
                validate_governance(
                    governance_path, "a" * 40, "1.1.0-rc.0", candidate, wheel, repo
                )


if __name__ == "__main__":
    unittest.main()
