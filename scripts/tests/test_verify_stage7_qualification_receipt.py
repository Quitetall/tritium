import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "verify-stage7-qualification-receipt.py"
)
validate = MODULE["validate"]
canonical = MODULE["canonical"]
Stage7QualificationError = MODULE["Stage7QualificationError"]
RELEASE_MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "release-evidence-status.py"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(seed: int) -> str:
    return "sha256:" + f"{seed:064x}"[-64:]


def file_record(path: str, seed: int = 1) -> dict:
    return {"path": path, "bytes": 1, "sha256": f"{seed:064x}"[-64:]}


def materialize(root: Path, path: str, seed: int = 1) -> dict:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bytes([seed % 256]))
    return {"path": path, "bytes": 1, "sha256": sha256(target)}


def fixture(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / "manifest.json"
    candidate.write_bytes(b'{"release":"1.1.0-rc.1","source_revision":"' + b"a" * 40 + b'"}\n')
    campaign = root / "campaign.json"
    model_files = [file_record(name, index + 1) for index, name in enumerate(sorted(MODULE["MODEL_FILE_NAMES"]))]
    members = [digest(index + 100) for index in range(512)]
    provenance = {}
    for index, name in enumerate(("calibration", "refinement", "validation", "evaluation")):
        rotated = members[index:] + members[:index]
        provenance[name] = {
            "id": digest(index + 1_000), "members": rotated,
            "datasets": [
                {"repo_id": repo, "revision": revision, "fraction_ppm": fraction}
                for repo, revision, fraction in MODULE["DATASETS"]
            ],
            "sampling_seed": index + 1, "tokenizer_digest": digest(2_000 + index),
            "ordered_token_digest": "sha256:" + hashlib.sha256(canonical(rotated)).hexdigest(),
            "sequence_count": 512, "tokens_per_sequence": 2048,
        }
        scope = {key: value for key, value in provenance[name].items() if key != "id"}
        provenance[name]["id"] = "sha256:" + hashlib.sha256(canonical(scope)).hexdigest()
    model = {"repo_id": "HuggingFaceTB/SmolLM2-1.7B", "revision": MODULE["MODEL_REVISION"], "files": model_files}
    smoke_model = {"repo_id": "HuggingFaceTB/SmolLM2-135M", "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2", "files": model_files}
    MODULE["MODEL_ID"] = "sha256:" + hashlib.sha256(canonical(model)).hexdigest()
    MODULE["SMOKE_MODEL_ID"] = "sha256:" + hashlib.sha256(canonical(smoke_model)).hexdigest()
    validate.__globals__["MODEL_ID"] = MODULE["MODEL_ID"]
    validate.__globals__["SMOKE_MODEL_ID"] = MODULE["SMOKE_MODEL_ID"]
    campaign_value = {
        "schema": "tritium.stage7-campaign.v1", "release": "1.1.0-rc.1",
        "source_revision": "a" * 40, "run_id": "stage7-run-1",
        "model": model,
        "smoke_model": smoke_model,
        "smoke_provenance": {
            "evaluation_id": "sha256:" + hashlib.sha256(canonical(members[:128])).hexdigest(), "evaluation_members": members[:128],
            "calibration_id": digest(1_000), "dataset_repo_id": "allenai/c4",
            "dataset_revision": MODULE["DATASETS"][0][1], "sampling_seed": 1,
            "tokenizer_digest": digest(2_000), "ordered_token_digest": "sha256:" + hashlib.sha256(canonical(members[:128])).hexdigest(),
            "sequence_count": 128, "tokens_per_sequence": 2048,
            "prefix_start": 0, "prefix_end": 128,
        },
        "provenance": provenance,
        "thresholds": {"r3_gap_closure_min": 0.25, "metadata_bpw_max": 0.01, "scale_only_token_cap": 8_000_000, "short_pv_token_cap": 32_000_000},
        "recipe_count": 1404, "recipe_grid_id": digest(5_000),
        "token_evidence_pack": materialize(root, "token-evidence/manifest.json", 5_001),
        "evidence": [
            {**materialize(root, "smoke/smoke-receipt.json", 5_010), "kind": "smoke"},
            {**materialize(root, "native/native-receipt.json", 5_011), "kind": "native-kernels"},
            {**materialize(root, "hestia-gate-c.json", 5_012), "kind": "hestia-gate-c"},
        ],
    }
    campaign.write_bytes(canonical(campaign_value) + b"\n")
    trace = root / "trace.json"
    ids = [digest(index + 10_000) for index in range(1404)]
    artifact = materialize(root, "artifacts/shared.tsalt2", 6_000)
    physical = materialize(root, "artifacts/shared-physical.json", 6_001)
    def measurement(candidate_id: str, full: bool) -> dict:
        return {
            "candidate_id": candidate_id, "track": "ptq", "physical_bytes": 1,
            "resident_bytes": 1, "output_loss": 1.0, "heldout_ppl": 2.0 if full else None,
            "task_metrics": {"mmlu": 1.0, "arc_challenge": 1.0, "hellaswag": 1.0, "boolq": 1.0, "gsm8k": 1.0, "math": 1.0} if full else {},
            "runtime_ms": 1.0, "artifact": artifact if full else None,
            "physical_report": physical if full else None, "correct": True,
        }
    stages = [
        {"name": "one-layer", "input_ids": ids, "measurements": [measurement(item, False) for item in ids], "promoted_ids": ids[:2]},
        {"name": "four-layer", "input_ids": ids[:2], "measurements": [measurement(item, False) for item in ids[:2]], "promoted_ids": ids[:2]},
        {"name": "full-model", "input_ids": ids[:2], "measurements": [measurement(item, True) for item in ids[:2]], "promoted_ids": ids[:2]},
    ]
    refinements = []
    for index in range(5):
        mode = "scale-only" if index < 3 else "short-pv"
        soft_method = None if index < 3 else ("ste-soft" if index == 3 else "hestia-relaxation")
        refinements.append({
            "refinement_id": digest(20_000 + index), "mode": "scale-only", "parent_candidate_id": ids[0],
            "rate": ("R2", "R3", "R4", "R2", "R3")[index], "soft_method": None,
            "soft_policy": {"kind": "none"}, "refinement_corpus_id": digest(30_000),
            "validation_id": digest(30_001), "parent_validation_ppl": 2.0,
            "checkpoints": [{"tokens": 1}, {"tokens": 2}, {"tokens": 3}],
        })
        refinements[-1]["mode"] = mode
        refinements[-1]["soft_method"] = soft_method
        refinements[-1]["soft_policy"] = {"kind": "none"} if soft_method is None else {"kind": soft_method}
    trace_value = {
        "schema": "tritium.stage7-execution.v1", "release": "1.1.0-rc.1", "source_revision": "a" * 40,
        "run_id": "stage7-run-1", "campaign_sha256": sha256(campaign), "stages": stages,
        "baselines": {
            "bf16": {"heldout_ppl": 2.0, "task_metrics": {"mmlu": 1.0}},
            "salt_v1": [{"rate": rate, "artifact": artifact, "physical_report": physical} for rate in ("R2", "R3", "R4")],
        },
        "refinements": refinements,
    }
    trace.write_bytes(canonical(trace_value) + b"\n")
    receipt = {
        "schema": "tritium.stage7-qualification.v1",
        "result": "pass",
        "release": "1.1.0-rc.1",
        "source_revision": "a" * 40,
        "run_id": "stage7-run-1",
        "model_id": MODULE["MODEL_ID"],
        "model_revision": MODULE["MODEL_REVISION"],
        "candidate_manifest_sha256": sha256(candidate),
        "campaign": {
            "path": campaign.name,
            "bytes": campaign.stat().st_size,
            "sha256": sha256(campaign),
        },
        "trace": {
            "path": trace.name,
            "bytes": trace.stat().st_size,
            "sha256": sha256(trace),
        },
        "freeze_authorized": True,
        "freeze_reasons": [],
        "frozen_ptq_recipe_ids": {"R2": digest(10_000), "R3": digest(10_001)},
        "r4_control_recipe_id": digest(10_000),
        "frozen_refined_recipe_id": digest(20_000),
        "frozen_refined_checkpoint": {
            "refinement_id": digest(20_000),
            "tokens": 8_000_000,
            "package_id": "trp1_" + "3" * 64,
            "artifact_sha256": "sha256:" + "4" * 64,
            "evaluation_sha256": "sha256:" + "5" * 64,
        },
        "soft_method_ab": {
            "outcome": "hestia-win",
            "winner": "hestia-relaxation",
            "ste_refinement_id": digest(20_001),
            "hestia_refinement_id": digest(20_002),
        },
    }
    unsigned = dict(receipt)
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    receipt_path = root / "qualification.json"
    receipt_path.write_bytes(canonical(receipt) + b"\n")
    return candidate, receipt_path, receipt


class Stage7QualificationTests(unittest.TestCase):
    def test_accepts_authorized_candidate_bound_freeze(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, receipt_path, receipt = fixture(Path(raw))
            self.assertEqual(
                validate(receipt_path, "a" * 40, "1.1.0-rc.1", candidate), receipt
            )

    def test_rejects_missing_freeze_or_mutated_trace(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, receipt_path, receipt = fixture(root)
            receipt["freeze_authorized"] = False
            receipt["freeze_reasons"] = ["missing-nondominated-ptq-rate"]
            unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
            receipt_path.write_bytes(canonical(receipt) + b"\n")
            with self.assertRaisesRegex(Stage7QualificationError, "authorized"):
                validate(receipt_path, "a" * 40, "1.1.0-rc.1", candidate)

            candidate, receipt_path, _ = fixture(root / "fresh")
            candidate.write_bytes(b"mutated")
            with self.assertRaisesRegex(Stage7QualificationError, "candidate manifest"):
                validate(receipt_path, "a" * 40, "1.1.0-rc.1", candidate)

            candidate, receipt_path, _ = fixture(root / "trace-drift")
            (candidate.parent / "trace.json").write_bytes(b"mutated")
            with self.assertRaisesRegex(Stage7QualificationError, "trace"):
                validate(receipt_path, "a" * 40, "1.1.0-rc.1", candidate)

    def test_rejects_intermediate_symlink_escape(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, receipt_path, _ = fixture(root)
            outside = root / "outside"
            outside.mkdir()
            (outside / "campaign.json").write_bytes((root / "campaign.json").read_bytes())
            (root / "link").symlink_to(outside, target_is_directory=True)
            receipt = json.loads(receipt_path.read_text())
            receipt["campaign"]["path"] = "link/campaign.json"
            receipt["campaign"]["sha256"] = sha256(outside / "campaign.json")
            unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
            receipt_path.write_bytes(canonical(receipt) + b"\n")
            with self.assertRaisesRegex(Stage7QualificationError, "intermediate symlink"):
                validate(receipt_path, "a" * 40, "1.1.0-rc.1", candidate)

    def test_release_registry_dispatches_stage7_artifact_branch(self):
        evaluate = RELEASE_MODULE["evaluate"]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifact = root / "model.tsalt2"
            artifact.write_bytes(b"artifact")
            candidate = root / "manifest.json"
            document = {
                "schema": "tritium.release-candidate.v1", "release": "1.1.0-rc.1",
                "source_revision": "a" * 40, "artifacts": [{
                    "id": "stage7-model", "kind": "model-bundle", "path": artifact.name,
                    "identity": {"bytes": artifact.stat().st_size, "sha256": sha256(artifact)},
                    "sbom": {}, "provenance": {},
                }],
            }
            candidate.write_bytes(canonical(document) + b"\n")
            receipt_path = root / "stage7.json"
            receipt_path.write_bytes(b"stage7")
            registry = root / "registry.json"
            registry_value = {
                "schema": "tritium.release-evidence-registry.v1", "release": "1.1.0-rc.1",
                "source_revision": "a" * 40, "candidate_manifest_sha256": sha256(candidate),
                "receipts": [{
                    "id": "stage7-receipt", "kind": "stage7-recipe-freeze", "path": receipt_path.name,
                    "sha256": sha256(receipt_path), "artifact_id": "stage7-model", "parents": [],
                }],
            }
            registry.write_bytes(canonical(registry_value) + b"\n")
            evaluate_globals = evaluate.__globals__
            old_validator = evaluate_globals["validate_stage7_freeze"]
            evaluate_globals["validate_stage7_freeze"] = lambda *args: {
                "run_id": "stage7-run", "receipt_id": "stage7-receipt"
            }
            try:
                report = evaluate(registry, candidate, document)
            finally:
                evaluate_globals["validate_stage7_freeze"] = old_validator
            self.assertEqual(report["schema"], "tritium.release-gate-report.v1")


if __name__ == "__main__":
    unittest.main()
