import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "aggregate-wheel-smoke.py"
SPEC = importlib.util.spec_from_file_location("aggregate_wheel_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
REVISION = "ab" * 20
RELEASE = "1.1.0-rc.0"


def write_matrix(root):
    for target, minors in MODULE.VERSIONS.items():
        for minor in minors:
            cell = f"{target}-cp3.{minor}"
            if target.startswith("linux"):
                host_os, host_arch, platform_tag = "linux", "x86_64", "manylinux_2_28_x86_64"
            elif target.startswith("windows"):
                host_os, host_arch, platform_tag = "win32", "amd64", "win_amd64"
            else:
                host_os, host_arch, platform_tag = "darwin", "arm64", "macosx_11_0_universal2"
            document = {
                "schema": MODULE.SCHEMA,
                "cell_id": cell,
                "target_id": target,
                "source_revision": REVISION,
                "passed": True,
                "python_implementation": "CPython",
                "python_version": f"3.{minor}.7",
                "host_os": host_os,
                "host_arch": host_arch,
                "wheel": f"tritium_torch-1.1.0rc0-cp39-abi3-{platform_tag}.whl",
                "sha256": {"linux": "11", "win32": "22", "darwin": "33"}[host_os] * 32,
                "bytes": 1234,
                "version": "1.1.0rc0",
                "platform_tag": platform_tag,
            }
            (root / f"{cell}.json").write_text(json.dumps(document), encoding="utf-8")


class AggregateWheelSmokeTests(unittest.TestCase):
    def test_complete_exact_matrix_passes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_matrix(root)
            receipt = MODULE.aggregate(root, REVISION, RELEASE, "run-1")
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["target_id"], MODULE.TARGET_ID)
            self.assertEqual(len(receipt["cells"]), len(MODULE.expected_cells()))

    def test_native_macos_arm64_platform_passes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_matrix(root)
            for path in root.glob("macos-arm64-cpu-*.json"):
                value = json.loads(path.read_text())
                value["platform_tag"] = "macosx_11_0_arm64"
                value["wheel"] = value["wheel"].replace(
                    "macosx_11_0_universal2", "macosx_11_0_arm64"
                )
                path.write_text(json.dumps(value), encoding="utf-8")
            receipt = MODULE.aggregate(root, REVISION, RELEASE, "run-arm64")
            self.assertTrue(receipt["passed"])

    def test_missing_cell_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_matrix(root)
            next(root.glob("linux-x86_64-cpu-cp3.9.json")).unlink()
            with self.assertRaisesRegex(MODULE.AggregateError, "matrix mismatch"):
                MODULE.aggregate(root, REVISION, RELEASE, "run-1")

    def test_forged_runtime_cell_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_matrix(root)
            path = root / "linux-x86_64-cpu-cp3.9.json"
            value = json.loads(path.read_text())
            value["cell_id"] = "linux-x86_64-cpu-cp3.14"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.AggregateError, "runtime-derived"):
                MODULE.aggregate(root, REVISION, RELEASE, "run-1")

    def test_target_must_reuse_exact_wheel(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_matrix(root)
            path = root / "windows-x86_64-cpu-cp3.14.json"
            value = json.loads(path.read_text())
            value["sha256"] = "44" * 32
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.AggregateError, "one exact wheel"):
                MODULE.aggregate(root, REVISION, RELEASE, "run-1")

    def test_target_revalidates_host_and_platform(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_matrix(root)
            path = root / "linux-x86_64-cpu-cp3.9.json"
            value = json.loads(path.read_text())
            value["host_os"] = "win32"
            value["host_arch"] = "amd64"
            value["platform_tag"] = "win_amd64"
            value["wheel"] = "tritium_torch-1.1.0rc0-cp39-abi3-win_amd64.whl"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.AggregateError, "host does not match"):
                MODULE.aggregate(root, REVISION, RELEASE, "run-1")

    def test_all_targets_require_one_candidate_version(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_matrix(root)
            for path in root.glob("windows-*.json"):
                value = json.loads(path.read_text())
                value["version"] = "1.1.0rc1"
                value["wheel"] = value["wheel"].replace("1.1.0rc0", "1.1.0rc1")
                path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.AggregateError, "one candidate version"):
                MODULE.aggregate(root, REVISION, RELEASE, "run-1")

    def test_receipt_strict_reload_rejects_identity_and_cell_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            evidence = root / "evidence"
            evidence.mkdir()
            write_matrix(evidence)
            receipt = MODULE.aggregate(evidence, REVISION, RELEASE, "run-1")
            path = root / "receipt.json"
            MODULE._atomic_write(path, receipt)
            self.assertEqual(MODULE.validate_receipt(path, REVISION, RELEASE), receipt)
            for mutation in ("identity", "cell"):
                changed = json.loads(path.read_text())
                if mutation == "identity":
                    changed["receipt_id"] = "sha256:" + "0" * 64
                else:
                    changed["cells"][0]["wheel_sha256"] = "0" * 64
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(MODULE.AggregateError):
                    MODULE.validate_receipt(path, REVISION, RELEASE)
                MODULE._atomic_write(path, receipt)


if __name__ == "__main__":
    unittest.main()
