import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "wheel-functional-smoke.py"
SPEC = importlib.util.spec_from_file_location("wheel_functional_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WheelFunctionalSmokeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
