import hashlib
import json
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


def refinement_trace(root: Path, common):
    training = ["sha256:" + "1" * 64, "sha256:" + "2" * 64]
    validation = ["sha256:" + "3" * 64]
    candidates = ["sha256:" + "4" * 64]
    trace = {
        "schema": MODULE["REFINEMENT_TRACE_SCHEMA"], "result": "pass",
        "release": common["release"],
        "source_revision": common["source_revision"],
        "run_id": "refinement-1",
        "environment": {
            "python": "3.13.5", "torch": "2.7.1",
            "tritium": "1.1.0rc0", "device": "cuda:0:GPU-1",
        },
        "parent_artifact_id": "parent",
        "training_set_id": MODULE["ledger_id"](training, "training")[0],
        "training_members": training,
        "validation_set_id": MODULE["ledger_id"](validation, "validation")[0],
        "validation_members": validation,
        "hard_candidate_set_id": MODULE["ledger_id"](candidates, "candidates")[0],
        "hard_candidate_members": candidates,
        "children": [
            {
                "mode": mode, "artifact_id": artifact_id,
                "parent_artifact_id": "parent",
                "work_id": "sha256:" + f"{ordinal + 5:x}" * 64,
                "recipe_id": "sha256:" + f"{ordinal + 8:x}" * 64,
                "ancestry": ["parent"],
                "group_sizes": [128],
                "packing": "s34" if mode == "s34" else "b3",
                "package_artifact_id": "sha256:" + f"{ordinal + 11:x}" * 64,
                "trits_changed": 0 if mode == "scale-only" else ordinal + 2,
                "allocations_changed": 2 if mode == "s34" else 0,
                "reload_samples": 32, "reload_max_abs_error": 0.0,
                "reload_tolerance": 1e-4, "latent_residuals": 0,
                "validation_loss_before": 1.0,
                "validation_loss_after": 0.9,
            }
            for ordinal, (mode, artifact_id) in enumerate(
                zip(MODULE["CHILDREN"], ("scale", "pv", "s34"), strict=True)
            )
        ],
    }
    path = root / "refinement-execution.json"
    path.write_bytes(canonical(trace) + b"\n")
    return path


def ablation_trace(root: Path, common):
    evaluation_id = "sha256:" + "3" * 64
    identities = [
        {
            "method": method, "family": family,
            "recipe_id": "sha256:" + f"{ordinal + 5:x}" * 64,
        }
        for ordinal, (method, family) in enumerate(MODULE["BASELINES"])
    ]
    trace = {
        "schema": MODULE["ABLATION_TRACE_SCHEMA"], "result": "pass",
        "release": common["release"],
        "source_revision": common["source_revision"],
        "run_id": "ablation-1",
        "environment": {
            "python": "3.13.5", "torch": "2.7.1",
            "tritium": "1.1.0rc0", "device": "cuda:0:GPU-1",
        },
        "model_artifact_id": "s34",
        "evaluation_id": evaluation_id,
        "baseline_set_id": "sha256:"
        + hashlib.sha256(
            canonical(
                {
                    "model_artifact_id": "s34",
                    "evaluation_id": evaluation_id,
                    "target_bytes": 100,
                    "target_bpw": 2.0,
                    "recipes": identities,
                }
            )
        ).hexdigest(),
        "target_bytes": 100, "target_bpw": 2.0,
        "baselines": [
            {
                **identity,
                "artifact_bytes": 100, "parameter_count": 400,
                "quality_score": 1.0,
                "elapsed_samples_ms": [float(ordinal + 1)] * 30,
                "resident_samples_bytes": [100 + ordinal] * 30,
                "physical_device": "cuda:0:GPU-1",
                "reproduced": True, "publishable_recipe": True,
            }
            for ordinal, identity in enumerate(identities)
        ],
    }
    path = root / "baseline-ablation-execution.json"
    path.write_bytes(canonical(trace) + b"\n")
    return path


class EstimatorRefinementReceiptTests(unittest.TestCase):
    def test_accepts_catalog_refinement_and_matched_ablation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common, records = fixture(root)
            cases = [
                {
                    "name": name, "algorithm_id": algorithm,
                    "schema_version": 1, "physical_planes": planes,
                    "hard_trits_exact": True, "finite_nonnegative_scales": True,
                    "master_gradients_finite": True, "state_gradients_finite": True,
                    "state_roundtrip_exact": True, "tied_identity_preserved": True,
                    "coverage_exact": True,
                }
                for name, algorithm, planes in MODULE["ESTIMATORS"]
            ]
            plugin = {
                "registered": True, "duplicate_rejected": True,
                "contract_validated": True, "purity_opt_in_required": True,
                "invalid_projection_rejected": True,
            }
            trace = {
                "schema": MODULE["TRACE_SCHEMA"], "result": "pass",
                "release": common["release"],
                "source_revision": common["source_revision"],
                "run_id": common["run_id"],
                "wheel": {
                    field: records["wheel"][field]
                    for field in ("name", "bytes", "sha256")
                },
                "environment": {
                    "python": "3.13.5", "torch": "2.7.1",
                    "tritium": "1.1.0rc0", "device": "cpu",
                },
                "estimators": cases, "external_plugin": plugin,
            }
            trace_path = root / "estimator-execution.json"
            trace_path.write_text(
                json.dumps(trace, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            estimators = seal(
                {
                    **common, "schema": MODULE["ESTIMATOR_SCHEMA"],
                    "anchor_artifact": records["wheel"],
                    "estimators": cases,
                    "external_plugin": plugin,
                    "trace": {
                        "path": trace_path.name,
                        "bytes": trace_path.stat().st_size,
                        "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
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

            refinement_trace_path = refinement_trace(root, common)
            derived = MODULE["derive_refinement_trace"](
                refinement_trace_path,
                receipt={"run_id": "refinement-1", "parent_artifact_id": "parent"},
                artifacts=MODULE["inventory"](candidate),
                revision="a" * 40,
                release="1.1.0-rc.0",
            )
            refinement = seal(
                {
                    **common, "schema": MODULE["REFINEMENT_SCHEMA"],
                    "run_id": "refinement-1", "anchor_artifact": records["s34"],
                    "parent_artifact_id": "parent",
                    **derived,
                    "trace": {
                        "path": refinement_trace_path.name,
                        "bytes": refinement_trace_path.stat().st_size,
                        "sha256": hashlib.sha256(
                            refinement_trace_path.read_bytes()
                        ).hexdigest(),
                    },
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

            ablation_trace_path = ablation_trace(root, common)
            ablation_derived = MODULE["derive_ablation_trace"](
                ablation_trace_path,
                receipt={"run_id": "ablation-1", "model_artifact_id": "s34"},
                revision="a" * 40,
                release="1.1.0-rc.0",
            )
            ablation = seal(
                {
                    **common, "schema": MODULE["ABLATION_SCHEMA"],
                    "run_id": "ablation-1", "anchor_artifact": records["s34"],
                    "model_artifact_id": "s34",
                    **ablation_derived,
                    "trace": {
                        "path": ablation_trace_path.name,
                        "bytes": ablation_trace_path.stat().st_size,
                        "sha256": hashlib.sha256(
                            ablation_trace_path.read_bytes()
                        ).hexdigest(),
                    },
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
                    "splits_disjoint": False, "children": [], "trace": {},
                }
            )
            path = root / "bad.json"
            path.write_bytes(canonical(receipt))
            with self.assertRaisesRegex(EstimatorRefinementError, "overlap"):
                validate_refinement(path, "a" * 40, "1.1.0-rc.0", candidate)


if __name__ == "__main__":
    unittest.main()
