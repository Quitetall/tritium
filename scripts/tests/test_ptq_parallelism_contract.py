"""Fail-closed installed-native provenance for the public PTQ performance probe."""

import base64
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("test_ptq_parallelism.py")
SPEC = importlib.util.spec_from_file_location("ptq_parallelism_probe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Record(str):
    def __new__(cls, name, payload):
        value = super().__new__(cls, name)
        value.hash = SimpleNamespace(
            mode="sha256",
            value=base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode(),
        )
        value.size = len(payload)
        return value


class PtqParallelismContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tritium-ptq-probe-contract-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.package = ModuleType("tritium")
        self.native = ModuleType("tritium._tritium")
        self.frontend = ModuleType("tritium.torch.ptq")
        self.native.source_identity = lambda: "source-git:" + "a" * 40
        self.native._fit_joint_ternary_diagonal_groups_with_objective = lambda: None
        self.records = []
        for name, module in (
            ("tritium/__init__.py", self.package),
            ("tritium/_tritium.abi3.so", self.native),
            ("tritium/torch/ptq.py", self.frontend),
        ):
            payload = name.encode()
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            module.__file__ = str(path)
            self.records.append(Record(name, payload))
        self.package._tritium = self.native
        torch_frontend = ModuleType("tritium.torch")
        torch_frontend.ptq = self.frontend
        self.distribution = SimpleNamespace(
            version="1.1.0rc2", files=self.records,
            locate_file=lambda name: self.root / str(name),
        )
        modules = mock.patch.dict("sys.modules", {
            "tritium": self.package, "tritium._tritium": self.native,
            "tritium.torch": torch_frontend, "tritium.torch.ptq": self.frontend,
        })
        modules.start()
        self.addCleanup(modules.stop)
        distribution = mock.patch("importlib.metadata.distribution", return_value=self.distribution)
        self.lookup = distribution.start()
        self.addCleanup(distribution.stop)

    def guard(self, expected="a" * 40):
        return MODULE._native_probe_installation(expected)

    def test_installed_native_has_exact_source_and_record_custody(self):
        result = self.guard()
        self.lookup.assert_called_once_with("pytritium")
        self.assertEqual(result["distribution"], "pytritium")
        self.assertEqual(result["version"], "1.1.0rc2")
        self.assertEqual(result["source_identity"], "source-git:" + "a" * 40)
        self.assertEqual(set(result["module_sha256"]), set(self.records))

    def test_old_native_without_current_fitter_is_not_benchmarked(self):
        del self.native._fit_joint_ternary_diagonal_groups_with_objective
        with self.assertRaisesRegex(RuntimeError, "current native.*fitter"):
            self.guard()

    def test_missing_pytritium_distribution_is_not_benchmarked(self):
        from importlib.metadata import PackageNotFoundError

        self.lookup.side_effect = PackageNotFoundError("pytritium")
        with self.assertRaisesRegex(RuntimeError, "installed pytritium wheel"):
            self.guard()

    def test_dirty_unverified_and_wrong_revision_are_rejected(self):
        for identity in ("unverified:no-git-metadata", "source-git:" + "a" * 40 + "+dirty",
                         "source-git:" + "b" * 40, None):
            with self.subTest(identity=identity):
                self.native.source_identity = lambda: identity
                with self.assertRaisesRegex(RuntimeError, "source identity"):
                    self.guard()

    def test_invalid_expected_revision_is_rejected(self):
        for expected in ("", "a" * 39, "A" * 40, True):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, "expected source revision"):
                    self.guard(expected)

    def test_namespace_shadowing_is_rejected(self):
        for module in (self.package, self.native, self.frontend):
            original = module.__file__
            module.__file__ = str(self.root / "shadow" / Path(original).name)
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(RuntimeError, "not owned by pytritium"):
                    self.guard()
            module.__file__ = original

    def test_missing_record_hash_size_or_member_is_rejected(self):
        record = self.records[-1]
        for mutation in ("hash", "size", "member", "algorithm"):
            old_hash, old_size = record.hash, record.size
            with self.subTest(mutation=mutation):
                if mutation == "hash":
                    record.hash = None
                elif mutation == "size":
                    record.size += 1
                elif mutation == "member":
                    self.distribution.files = self.records[:-1]
                else:
                    record.hash = SimpleNamespace(mode="md5", value="not-trusted")
                with self.assertRaisesRegex(RuntimeError, "RECORD"):
                    self.guard()
            record.hash, record.size = old_hash, old_size
            self.distribution.files = self.records

    def test_installed_byte_tampering_is_rejected(self):
        path = Path(self.frontend.__file__)
        payload = path.read_bytes()
        path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
        with self.assertRaisesRegex(RuntimeError, "RECORD"):
            self.guard()

    def test_symlinked_installed_member_is_rejected(self):
        path = Path(self.frontend.__file__)
        original = path.with_name("original.py")
        path.rename(original)
        path.symlink_to(original.name)
        with self.assertRaisesRegex(RuntimeError, "ordinary file"):
            self.guard()

    def test_subprocess_uses_the_same_guard(self):
        namespace = {}
        exec(MODULE._PROBE.split("\nimport torch\n", 1)[0], namespace)
        self.assertEqual(namespace["_native_probe_installation"]("a" * 40), self.guard())

    def test_fast_python_fallback_cannot_satisfy_native_performance_guard(self):
        case = MODULE.PublicPtqParallelismTests(
            "test_public_convert_parallelizes_rows_without_changing_fitted_artifact"
        )
        probe = {
            "algorithm_id": "fixture", "fit_digest": "fixture", "weighted_mse": 0,
            "installation": {}, "native_work": {"calls": 0, "rows": 2048},
        }
        with mock.patch.object(case, "_convert", return_value=probe):
            with self.assertRaisesRegex(AssertionError, "not greater than 0"):
                case.test_public_convert_parallelizes_rows_without_changing_fitted_artifact()

    def test_partial_native_row_coverage_is_rejected(self):
        case = MODULE.PublicPtqParallelismTests(
            "test_public_convert_parallelizes_rows_without_changing_fitted_artifact"
        )
        probe = {
            "algorithm_id": "fixture", "fit_digest": "fixture", "weighted_mse": 0,
            "installation": {}, "native_work": {"calls": 1, "rows": 1},
        }
        with mock.patch.object(case, "_convert", return_value=probe):
            with self.assertRaisesRegex(AssertionError, "1 != 2048"):
                case.test_public_convert_parallelizes_rows_without_changing_fitted_artifact()


class PtqPerformanceDiagnosticsTests(unittest.TestCase):
    def test_inherited_cpu_quota_is_observed_without_reporting_private_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group = root / "private-owner" / "worker"
            group.mkdir(parents=True)
            (root / "cpu.max").write_text("max 100000\n")
            (group.parent / "cpu.max").write_text("150000 100000\n")
            (group / "cpu.max").write_text("400000 100000\n")
            membership = root / "membership"
            membership.write_text("0::/private-owner/worker\n")
            result = MODULE._cpu_quota_diagnostics(root, membership)
            self.assertTrue(result["complete"])
            self.assertEqual(result["observed_quota_cores"], 1.5)
            self.assertEqual(result["scopes_read"], 3)
            self.assertNotIn("private-owner", str(result))

    def test_missing_and_malformed_quota_are_unknown_not_unlimited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            membership = root / "membership"
            membership.write_text("0::/\n")
            for index, value in enumerate((None, "max 0", "-1 100000", "4", "4 2 extra", "x" * 4097)):
                with self.subTest(index=index):
                    path = root / "cpu.max"
                    if value is None:
                        path.unlink(missing_ok=True)
                    else:
                        path.write_text(value)
                    result = MODULE._cpu_quota_diagnostics(root, membership)
                    self.assertFalse(result["complete"])
                    self.assertIsNone(result["observed_quota_cores"])

    def test_unlimited_and_partial_ancestry_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            membership = root / "membership"
            membership.write_text("0::/worker\n")
            (root / "worker").mkdir()
            (root / "worker" / "cpu.max").write_text("max 100000\n")
            partial = MODULE._cpu_quota_diagnostics(root, membership)
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["scopes_read"], 1)
            (root / "cpu.max").write_text("max 100000\n")
            complete = MODULE._cpu_quota_diagnostics(root, membership)
            self.assertTrue(complete["complete"])
            self.assertIsNone(complete["observed_quota_cores"])

    def test_unsafe_or_overdeep_membership_is_not_followed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            membership = root / "membership"
            for index, value in enumerate(("0::/../outside", "0::relative", "0::/" + "/".join(["x"] * 65),
                                           "0::/\n0::/second", "x" * 4097)):
                with self.subTest(index=index):
                    membership.write_text(value)
                    self.assertEqual(MODULE._cpu_quota_diagnostics(root, membership), {
                        "complete": False, "observed_quota_cores": None, "scopes_read": 0,
                    })

    def test_failed_speedup_remains_a_failure_with_diagnostics(self):
        case = MODULE.PublicPtqParallelismTests(
            "test_public_convert_parallelizes_rows_without_changing_fitted_artifact"
        )
        common = {
            "algorithm_id": "fixture", "fit_digest": "fixture", "weighted_mse": 0,
            "installation": {"source_identity": "fixture"},
            "native_work": {"calls": 1, "rows": 2048},
            "benchmark_environment": {}, "timing": {},
        }
        serial = {**common, "elapsed_seconds": 1.39}
        parallel = {**common, "elapsed_seconds": 1.0}
        with mock.patch.object(case, "_convert", side_effect=(serial, parallel)):
            with self.assertRaisesRegex(AssertionError, "not greater than or equal to 1.5"):
                case.test_public_convert_parallelizes_rows_without_changing_fitted_artifact()


if __name__ == "__main__":
    unittest.main()
