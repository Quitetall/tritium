import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "wheel-functional-smoke.py"
SPEC = importlib.util.spec_from_file_location("wheel_functional_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WheelFunctionalSmokeTests(unittest.TestCase):
    def _evidence(self, wheel: Path) -> dict:
        return {
            "schema": MODULE.SCHEMA,
            "source_revision": "a" * 40,
            "passed": True,
            "wheel": wheel.name,
            "wheel_sha256": MODULE._sha256(wheel),
            "distribution_version": "1.1.0rc0",
            "python_version": "3.13.5",
            "torch_version": "2.11.0",
            "transformers_version": "5.5.3",
            "safetensors_version": "0.8.0",
            "native_device": "cpu",
            "compiled_backends": ["cpu"],
            "tritium_module": "/venv/tritium/__init__.py",
            "converted_parameters": 256,
            "operations": sorted(MODULE.REQUIRED_OPERATIONS),
        }

    def test_functional_receipt_binds_wheel_revision_run_and_coverage(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wheel = root / "candidate.whl"
            wheel.write_bytes(b"wheel")
            receipt = MODULE.build_receipt(
                self._evidence(wheel), wheel, "1.1.0-rc.0", "run-1",
                "2026-07-21T12:00:00Z", 100.0,
            )
            path = root / "receipt.json"
            MODULE._atomic_write(path, receipt)
            self.assertEqual(
                MODULE.validate_receipt(path, "a" * 40, "1.1.0-rc.0", wheel),
                receipt,
            )

    def test_functional_receipt_rejects_artifact_and_coverage_drift(self):
        for mutation in (
            "artifact", "coverage", "identity", "backend", "backend-type",
            "coverage-type", "version",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                wheel = root / "candidate.whl"
                wheel.write_bytes(b"wheel")
                receipt = MODULE.build_receipt(
                    self._evidence(wheel), wheel, "1.1.0-rc.0", "run-1",
                    "2026-07-21T12:00:00Z", 100.0,
                )
                path = root / "receipt.json"
                if mutation == "artifact":
                    wheel.write_bytes(b"changed")
                elif mutation == "coverage":
                    receipt["evidence"]["operations"].pop()
                elif mutation == "backend":
                    receipt["evidence"]["compiled_backends"] = []
                elif mutation == "backend-type":
                    receipt["evidence"]["compiled_backends"] = [{}]
                elif mutation == "coverage-type":
                    receipt["evidence"]["operations"] = [{}]
                elif mutation == "version":
                    receipt["evidence"]["distribution_version"] = "1.1.0rc1"
                else:
                    receipt["receipt_id"] = "sha256:" + "0" * 64
                path.write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaises(MODULE.SmokeError):
                    MODULE.validate_receipt(path, "a" * 40, "1.1.0-rc.0", wheel)

    def test_resolve_wheel_requires_exactly_one_regular_file(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaisesRegex(MODULE.SmokeError, "exactly one wheel"):
                MODULE.resolve_wheel(root)
            wheel = root / "candidate.whl"
            wheel.write_bytes(b"wheel")
            self.assertEqual(MODULE.resolve_wheel(root), wheel.resolve())

    def test_source_checkout_import_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            module = root / "tritium" / "__init__.py"
            module.parent.mkdir()
            module.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.SmokeError, "forbidden source checkout"):
                MODULE.require_installed(module, root, root)

    def test_import_outside_smoke_environment_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            forbidden = root / "source"
            environment = root / "environment"
            module = root / "global" / "tritium" / "__init__.py"
            forbidden.mkdir()
            environment.mkdir()
            module.parent.mkdir(parents=True)
            module.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.SmokeError, "outside smoke environment"):
                MODULE.require_installed(module, forbidden, environment)

    def test_direct_url_binds_exact_wheel_path_and_digest(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel = Path(raw) / "candidate.whl"
            wheel.write_bytes(b"wheel")
            digest = MODULE._sha256(wheel)
            document = {
                "url": wheel.resolve().as_uri(),
                "archive_info": {"hashes": {"sha256": digest}},
            }
            MODULE.validate_direct_url(document, wheel.resolve(), digest)
            document["archive_info"]["hashes"]["sha256"] = "00" * 32
            with self.assertRaisesRegex(MODULE.SmokeError, "digest does not match"):
                MODULE.validate_direct_url(document, wheel.resolve(), digest)

    def test_native_result_rejects_wrong_negative_and_nonfinite(self):
        MODULE.validate_native_result([[-126.0 / 127.0]])
        for value in ([[-0.5]], [[float("nan")]], [[float("-inf")]]):
            with self.subTest(value=value), self.assertRaisesRegex(
                MODULE.SmokeError, "native ternary kernel"
            ):
                MODULE.validate_native_result(value)

    def test_native_source_identity_is_required_and_exact(self):
        revision = "a" * 40
        MODULE.require_native_source_identity(
            SimpleNamespace(source_identity=lambda: f"source-git:{revision}"),
            revision,
        )
        with self.assertRaisesRegex(MODULE.SmokeError, "does not expose native source identity"):
            MODULE.require_native_source_identity(SimpleNamespace(), revision)
        with self.assertRaisesRegex(MODULE.SmokeError, "source identity mismatch"):
            MODULE.require_native_source_identity(
                SimpleNamespace(source_identity=lambda: "source-git:" + "b" * 40),
                revision,
            )

    def test_smoke_device_choices_are_closed(self):
        self.assertEqual(MODULE.run_smoke.__defaults__, ("cpu",))


if __name__ == "__main__":
    unittest.main()
