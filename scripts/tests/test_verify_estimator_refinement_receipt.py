import hashlib
from pathlib import Path
import runpy
import tempfile
import unittest


MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "verify-estimator-refinement-receipt.py"
)
canonical = MODULE["canonical"]
validate_estimators = MODULE["validate_estimators"]
validate_refinement = MODULE["validate_refinement"]
validate_ablation = MODULE["validate_ablation"]
EstimatorRefinementError = MODULE["EstimatorRefinementError"]


def seal(value):
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    return value


def fixture(root: Path):
    artifacts = []
    records = {}
    for ordinal, (artifact_id, kind) in enumerate(
        (
            ("wheel", "python-wheel"), ("parent", "model-bundle"),
            ("scale", "model-bundle"), ("pv", "model-bundle"),
            ("s34", "model-bundle"),
        )
    ):
        path = root / f"{artifact_id}.bin"
        path.write_bytes(bytes([ordinal + 1]) * (ordinal + 2))
        record = {
            "id": artifact_id, "kind": kind, "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        records[artifact_id] = record
        artifacts.append(
            {
                "id": artifact_id, "kind": kind, "path": path.name,
                "identity": {"bytes": record["bytes"], "sha256": record["sha256"]},
            }
        )
    candidate = root / "manifest.json"
    candidate.write_bytes(canonical({"artifacts": artifacts}))
    common = {
        "result": "pass", "release": "1.1.0-rc.0",
        "source_revision": "a" * 40, "run_id": "run-1",
        "candidate_manifest_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    }
    return candidate, common, records


class EstimatorRefinementReceiptTests(unittest.TestCase):
    def test_accepts_catalog_refinement_and_matched_ablation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common, records = fixture(root)
            estimators = seal(
                {
                    **common, "schema": MODULE["ESTIMATOR_SCHEMA"],
                    "anchor_artifact": records["wheel"],
                    "estimators": [
                        {
                            "name": name, "algorithm_id": algorithm,
                            "schema_version": 1, "physical_planes": planes,
                            "hard_trits_exact": True, "finite_nonnegative_scales": True,
                            "master_gradients_finite": True, "state_gradients_finite": True,
                            "state_roundtrip_exact": True, "tied_identity_preserved": True,
                            "coverage_exact": True,
                        }
                        for name, algorithm, planes in MODULE["ESTIMATORS"]
                    ],
                    "external_plugin": {
                        "registered": True, "duplicate_rejected": True,
                        "contract_validated": True, "purity_opt_in_required": True,
                        "invalid_projection_rejected": True,
                    },
                }
            )
            estimator_path = root / "estimators.json"
            estimator_path.write_bytes(canonical(estimators))
            self.assertEqual(
                validate_estimators(
                    estimator_path, "a" * 40, "1.1.0-rc.0", candidate
                ),
                estimators,
            )

            child_ids = ("scale", "pv", "s34")
            parents = ("parent", "parent", "parent")
            refinement = seal(
                {
                    **common, "schema": MODULE["REFINEMENT_SCHEMA"],
                    "run_id": "refinement-1", "anchor_artifact": records["s34"],
                    "parent_artifact_id": "parent",
                    "training_set_id": "sha256:" + "1" * 64,
                    "validation_set_id": "sha256:" + "2" * 64,
                    "splits_disjoint": True,
                    "children": [
                        {
                            "mode": mode, "artifact": records[artifact_id],
                            "parent_artifact_id": parent,
                            "work_id": "sha256:" + f"{ordinal + 3:x}" * 64,
                            "recipe_id": "sha256:" + f"{ordinal + 6:x}" * 64,
                            "ancestry_id": "sha256:" + f"{ordinal + 9:x}" * 64,
                            "trits_frozen": ordinal == 0,
                            "allocation_frozen": ordinal == 0,
                            "hard_candidates_held_out": True, "g128_aligned": True,
                            "native_salt_package": True, "strict_reload": True,
                            "latent_residuals": 0,
                        }
                        for ordinal, (mode, artifact_id, parent) in enumerate(
                            zip(MODULE["CHILDREN"], child_ids, parents, strict=True)
                        )
                    ],
                }
            )
            refinement_path = root / "refinement.json"
            refinement_path.write_bytes(canonical(refinement))
            self.assertEqual(
                validate_refinement(
                    refinement_path, "a" * 40, "1.1.0-rc.0", candidate
                ),
                refinement,
            )

            ablation = seal(
                {
                    **common, "schema": MODULE["ABLATION_SCHEMA"],
                    "run_id": "ablation-1", "anchor_artifact": records["s34"],
                    "model_artifact_id": "s34",
                    "evaluation_id": "sha256:" + "3" * 64,
                    "baseline_set_id": "sha256:" + "4" * 64,
                    "inventory_complete": True,
                    "baselines": [
                        {
                            "method": method, "family": family,
                            "artifact_bytes": 100, "target_bytes": 100,
                            "rate_gap_bpw": 0.0, "quality_score": 1.0,
                            "runtime_ms": 1.0, "resident_bytes": 100,
                            "reproduced": True, "same_box": True,
                            "publishable_recipe": True, "eligible_for_claim": True,
                        }
                        for method, family in MODULE["BASELINES"]
                    ],
                }
            )
            ablation_path = root / "ablation.json"
            ablation_path.write_bytes(canonical(ablation))
            self.assertEqual(
                validate_ablation(
                    ablation_path, "a" * 40, "1.1.0-rc.0", candidate
                ),
                ablation,
            )

    def test_rejects_overlapping_refinement_data(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common, records = fixture(root)
            receipt = seal(
                {
                    **common, "schema": MODULE["REFINEMENT_SCHEMA"],
                    "anchor_artifact": records["s34"], "parent_artifact_id": "parent",
                    "training_set_id": "sha256:" + "1" * 64,
                    "validation_set_id": "sha256:" + "1" * 64,
                    "splits_disjoint": False, "children": [],
                }
            )
            path = root / "bad.json"
            path.write_bytes(canonical(receipt))
            with self.assertRaisesRegex(EstimatorRefinementError, "overlap"):
                validate_refinement(path, "a" * 40, "1.1.0-rc.0", candidate)


if __name__ == "__main__":
    unittest.main()
