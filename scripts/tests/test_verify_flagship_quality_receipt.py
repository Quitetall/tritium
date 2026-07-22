import hashlib
from pathlib import Path
import runpy
import tempfile
import unittest


MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "verify-flagship-quality-receipt.py"
)
canonical = MODULE["canonical"]
validate_quality = MODULE["validate_quality"]
validate_tasks = MODULE["validate_tasks"]
FlagshipQualityError = MODULE["FlagshipQualityError"]


def seal(value):
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    return value


def fixture(root: Path):
    artifact = root / "qwen-refined.salt"
    artifact.write_bytes(b"refined model bundle")
    identity = {
        "id": "qwen-refined", "kind": "model-bundle", "name": artifact.name,
        "bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    candidate = root / "manifest.json"
    candidate.write_bytes(
        canonical(
            {
                "artifacts": [
                    {
                        "id": identity["id"], "kind": identity["kind"],
                        "path": artifact.name,
                        "identity": {
                            "bytes": identity["bytes"], "sha256": identity["sha256"],
                        },
                    }
                ]
            }
        )
    )
    common = {
        "result": "pass", "release": "1.1.0-rc.0",
        "source_revision": "a" * 40, "run_id": "quality-run",
        "candidate_manifest_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "artifact": identity, "model_id": MODULE["MODEL_ID"],
        "model_revision": MODULE["MODEL_REVISION"], "scope": "language+mtp",
        "evaluation_id": "sha256:" + "1" * 64,
        "recipe_id": "sha256:" + "2" * 64,
    }
    return candidate, common


class FlagshipQualityReceiptTests(unittest.TestCase):
    def test_accepts_preregistered_quality_and_six_task_bounds(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common = fixture(root)
            quality = seal(
                {
                    **common, "schema": MODULE["QUALITY_SCHEMA"],
                    "dense_perplexity": 10.0, "salt_v1_perplexity": 12.0,
                    "ptq_perplexity": 11.0, "refined_perplexity": 10.05,
                    "ptq_gap_closed_fraction": 0.5,
                    "refined_relative_increase_pct": 0.5,
                    "refined_relative_ci95_upper_pct": 0.8,
                    "baseline_set_id": "sha256:" + "3" * 64,
                    "baseline_inventory_complete": True,
                    "baseline_comparisons": [
                        {
                            "method": "salt-v1", "family": "additive-ternary",
                            "artifact_bytes": 100, "resident_bytes": 120,
                            "quality_score": 12.0, "runtime_ms": 10.0,
                            "reproduced": True, "matched_physical_bytes": True,
                            "comparison_result": "tritium-win",
                        },
                        {
                            "method": "gptq-style", "family": "global-low-bit",
                            "artifact_bytes": 100, "resident_bytes": 120,
                            "quality_score": 10.04, "runtime_ms": 9.0,
                            "reproduced": True, "matched_physical_bytes": True,
                            "comparison_result": "tradeoff",
                        },
                    ],
                    "near_zero_divergence": True, "additive_ternary_sota": True,
                    "global_low_bit_pareto": True,
                }
            )
            quality_path = root / "quality.json"
            quality_path.write_bytes(canonical(quality))
            self.assertEqual(
                validate_quality(quality_path, "a" * 40, "1.1.0-rc.0", candidate),
                quality,
            )

            tasks = seal(
                {
                    **common, "schema": MODULE["TASK_SCHEMA"], "run_id": "tasks-run",
                    "tasks": [
                        {
                            "name": f"task-{ordinal}", "dense_accuracy_pct": 70.0,
                            "refined_accuracy_pct": 69.6, "delta_pp": 0.4,
                            "ci95_upper_pp": 0.9,
                        }
                        for ordinal in range(6)
                    ],
                    "mean_delta_pp": 0.4, "mean_ci95_upper_pp": 0.5,
                }
            )
            tasks_path = root / "tasks.json"
            tasks_path.write_bytes(canonical(tasks))
            self.assertEqual(
                validate_tasks(tasks_path, "a" * 40, "1.1.0-rc.0", candidate),
                tasks,
            )

    def test_rejects_confidence_boundary(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common = fixture(root)
            quality = seal(
                {
                    **common, "schema": MODULE["QUALITY_SCHEMA"],
                    "dense_perplexity": 10.0, "salt_v1_perplexity": 12.0,
                    "ptq_perplexity": 11.0, "refined_perplexity": 10.05,
                    "ptq_gap_closed_fraction": 0.5,
                    "refined_relative_increase_pct": 0.5,
                    "refined_relative_ci95_upper_pct": 1.01,
                    "baseline_set_id": "sha256:" + "3" * 64,
                    "baseline_inventory_complete": True,
                    "baseline_comparisons": [],
                    "near_zero_divergence": True, "additive_ternary_sota": True,
                    "global_low_bit_pareto": True,
                }
            )
            path = root / "quality.json"
            path.write_bytes(canonical(quality))
            with self.assertRaisesRegex(FlagshipQualityError, "one percent"):
                validate_quality(path, "a" * 40, "1.1.0-rc.0", candidate)


if __name__ == "__main__":
    unittest.main()
