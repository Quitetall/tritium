import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-estimator-catalog.py")
canonical = MODULE["canonical"]
seal = MODULE["seal"]
QualificationError = MODULE["QualificationError"]


def fixture(root: Path):
    wheel = root / "tritium_torch-1.1.0rc0-py3-none-any.whl"
    wheel.write_bytes(b"candidate-wheel")
    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    candidate = root / "manifest.json"
    candidate.write_bytes(
        canonical(
            {
                "artifacts": [
                    {
                        "id": "python-wheel",
                        "kind": "python-wheel",
                        "path": wheel.name,
                        "identity": {
                            "bytes": wheel.stat().st_size,
                            "sha256": wheel_sha256,
                        },
                    }
                ]
            }
        )
    )
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
        "release": "1.1.0-rc.0", "source_revision": "a" * 40,
        "run_id": "estimator-physical-1",
        "wheel": {
            "name": wheel.name, "bytes": wheel.stat().st_size,
            "sha256": wheel_sha256,
        },
        "environment": {
            "python": "3.13.5", "torch": "2.7.1",
            "tritium": "1.1.0rc0", "device": "cpu",
        },
        "estimators": cases, "external_plugin": plugin,
    }
    trace_path = root / "worker.json"
    trace_path.write_bytes(canonical(trace) + b"\n")
    return candidate, wheel, trace_path


class QualifyEstimatorCatalogTests(unittest.TestCase):
    def test_preserves_venv_style_python_entrypoint(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "python-real"
            target.write_bytes(b"binary")
            target.chmod(0o755)
            link = root / "python"
            link.symlink_to(target)
            self.assertEqual(MODULE["executable"](link, "Python"), link.absolute())

    def test_seals_trace_and_self_validates(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, wheel, trace = fixture(root)
            receipt = seal(
                root / "qualification", candidate=candidate, wheel=wheel,
                trace_path=trace, source_revision="a" * 40,
                release="1.1.0-rc.0", run_id="estimator-physical-1",
            )
            self.assertEqual(receipt["result"], "pass")
            self.assertEqual(receipt["anchor_artifact"]["id"], "python-wheel")
            self.assertTrue((root / "qualification/estimator-execution.json").is_file())

    def test_rejects_trace_bound_to_different_wheel(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, wheel, trace_path = fixture(root)
            trace = json.loads(trace_path.read_bytes())
            trace["wheel"]["sha256"] = "0" * 64
            trace_path.write_bytes(canonical(trace) + b"\n")
            with self.assertRaisesRegex(QualificationError, "identity"):
                seal(
                    root / "qualification", candidate=candidate, wheel=wheel,
                    trace_path=trace_path, source_revision="a" * 40,
                    release="1.1.0-rc.0", run_id="estimator-physical-1",
                )


if __name__ == "__main__":
    unittest.main()
