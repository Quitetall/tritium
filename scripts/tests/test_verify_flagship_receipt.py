import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "verify-flagship-receipt.py"
)
validate = MODULE["validate"]
canonical = MODULE["canonical"]
FlagshipReceiptError = MODULE["FlagshipReceiptError"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    artifacts = []
    records = []
    for ordinal, (track_id, _, _) in enumerate(MODULE["EXPECTED_TRACKS"]):
        path = root / f"{track_id}.salt"
        path.write_bytes(f"package-{ordinal}".encode())
        artifact_id = f"qwen-{track_id}"
        artifacts.append(
            {
                "id": artifact_id, "kind": "model-bundle", "path": path.name,
                "identity": {"bytes": path.stat().st_size, "sha256": sha256(path)},
            }
        )
        records.append(
            {
                "id": artifact_id, "kind": "model-bundle", "name": path.name,
                "bytes": path.stat().st_size, "sha256": sha256(path),
            }
        )
    candidate = root / "manifest.json"
    candidate.write_bytes(json.dumps({"artifacts": artifacts}).encode())
    tracks = []
    for ordinal, (track_id, mode, profile) in enumerate(MODULE["EXPECTED_TRACKS"]):
        tracks.append(
            {
                "track_id": track_id, "mode": mode, "profile": profile,
                "artifact": records[ordinal],
                "parent_artifact_id": records[1]["id"] if ordinal == 2 else None,
                "work_id": "sha256:" + f"{ordinal + 1:x}" * 64,
                "recipe_sha256": "sha256:" + f"{ordinal + 4:x}" * 64,
                "package_id": "sha256:" + f"{ordinal + 7:x}" * 64,
                "complete": True, "strict_reload": True,
            }
        )
    receipt = {
        "schema": MODULE["SCHEMA"], "result": "pass",
        "release": "1.1.0-rc.0", "source_revision": "a" * 40,
        "run_id": "qwen-conversion-1",
        "candidate_manifest_sha256": sha256(candidate),
        "source": {
            "model_id": MODULE["MODEL_ID"],
            "revision": MODULE["MODEL_REVISION"], "scope": "language+mtp",
        },
        "tracks": tracks,
        "coverage": {
            "total_tensors": 1199, "additive_matrices": 506,
            "preserved_tensors": 360, "deferred_vision_tensors": 333,
            "unknown_tensors": 0, "duplicate_tensors": 0, "missing_tensors": 0,
            "vision_identity_bound": True,
        },
        "parity": {
            "language_layers": 64, "host_parity": True, "cuda_parity": True,
            "mtp_oracle_parity": True,
        },
        "determinism": {
            "package_repeat_exact": True, "evaluation_repeat_exact": True,
        },
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
    receipt_path = root / "receipt.json"
    receipt_path.write_bytes(canonical(receipt) + b"\n")
    return candidate, receipt_path, receipt


class FlagshipReceiptTests(unittest.TestCase):
    def test_accepts_complete_separate_candidate_bound_tracks(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, receipt_path, receipt = fixture(Path(raw))
            self.assertEqual(
                validate(
                    receipt_path, "a" * 40, "1.1.0-rc.0", candidate
                ),
                receipt,
            )

    def test_rejects_lineage_coverage_and_candidate_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, receipt_path, receipt = fixture(root)
            receipt["tracks"][2]["parent_artifact_id"] = None
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                canonical({key: value for key, value in receipt.items() if key != "receipt_id"})
            ).hexdigest()
            receipt_path.write_bytes(canonical(receipt))
            with self.assertRaisesRegex(FlagshipReceiptError, "lineage"):
                validate(receipt_path, "a" * 40, "1.1.0-rc.0", candidate)

            candidate, receipt_path, _ = fixture(root / "fresh")
            (candidate.parent / "compact-ptq.salt").write_bytes(b"drift")
            with self.assertRaisesRegex(FlagshipReceiptError, "contradict"):
                validate(receipt_path, "a" * 40, "1.1.0-rc.0", candidate)


if __name__ == "__main__":
    unittest.main()
