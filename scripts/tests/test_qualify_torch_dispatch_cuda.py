from __future__ import annotations

import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
VERIFY = runpy.run_path(ROOT / "scripts" / "verify-torch-dispatch-cuda-receipt.py")
QUALIFY = runpy.run_path(ROOT / "scripts" / "qualify-torch-dispatch-cuda.py")
QualificationError = QUALIFY["QualificationError"]
assemble = QUALIFY["assemble"]
parse_sanitizer_version = QUALIFY["parse_sanitizer_version"]
validate = VERIFY["validate"]
CUDA_TESTS = VERIFY["CUDA_TESTS"]
MEMCHECK_TESTS = VERIFY["MEMCHECK_TESTS"]


def junit(names: tuple[str, ...]) -> str:
    cases = "".join(f"<testcase name='{name}'/>" for name in names)
    count = len(names)
    return (
        f"<testsuites tests='{count}' failures='0' errors='0' skipped='0'>"
        f"<testsuite tests='{count}' failures='0' errors='0' skipped='0'>"
        f"{cases}</testsuite></testsuites>"
    )


class QualifyTorchDispatchCudaTests(unittest.TestCase):
    def test_assembles_self_validating_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stage = root / "stage"
            stage.mkdir()
            wheel = root / "tritium_torch-1.1.0rc0-cp39-abi3-linux_x86_64.whl"
            wheel.write_bytes(b"wheel")
            source = root / "test_torch_dispatch.py"
            source.write_text("def test_native_cuda(): pass\n", encoding="utf-8")
            suite = root / "suite.xml"
            suite.write_text(junit(CUDA_TESTS), encoding="utf-8")
            memcheck = root / "memcheck.xml"
            memcheck.write_text(junit(MEMCHECK_TESTS), encoding="utf-8")
            log = root / "memcheck.log"
            log.write_text("========= ERROR SUMMARY: 0 errors\n", encoding="utf-8")
            probe = {
                "python_version": "3.14.6",
                "torch_version": "2.11.0+cu130",
                "tritium_version": "1.1.0rc0",
                "cuda_runtime": "13.0",
                "cuda_driver": "610.57.04",
                "source_identity": "source-git:" + "a" * 40,
                "device": {
                    "index": 0,
                    "uuid": "GPU-physical",
                    "name": "NVIDIA GeForce RTX 4090",
                    "compute_capability": "8.9",
                    "total_memory_bytes": 25_000_000_000,
                },
            }
            receipt = assemble(
                stage,
                wheel=wheel,
                source=source,
                source_revision="a" * 40,
                release="1.1.0-rc.0",
                run_id="cuda-dispatch-17",
                probe=probe,
                suite_junit=suite,
                memcheck_junit=memcheck,
                sanitizer_log=log,
                sanitizer_version="2026.2.1.0",
            )
            self.assertEqual(
                validate(stage / "receipt.json", "a" * 40, "1.1.0-rc.0", wheel),
                receipt,
            )

    def test_rejects_nonzero_or_ambiguous_sanitizer_log(self):
        for content in (
            "========= ERROR SUMMARY: 1 error\n",
            "========= ERROR SUMMARY: 0 errors\n========= ERROR SUMMARY: 0 errors\n",
        ):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as raw:
                path = Path(raw) / "memcheck.log"
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(QualificationError):
                    QUALIFY["require_zero_sanitizer_errors"](path)

    def test_parses_only_canonical_compute_sanitizer_version(self):
        output = "NVIDIA Compute Sanitizer\nVersion 2026.2.1.0 (build 1)\n"
        self.assertEqual(parse_sanitizer_version(output), "2026.2.1.0")
        for malformed in ("", "Version latest", "Version 2026.2"):
            with self.subTest(malformed=malformed), self.assertRaises(QualificationError):
                parse_sanitizer_version(malformed)

    def test_rejects_candidate_drift_from_private_snapshot(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate = root / "candidate.whl"
            snapshot = root / "snapshot.whl"
            candidate.write_bytes(b"wheel")
            snapshot.write_bytes(b"wheel")
            QUALIFY["require_same_file_identity"](candidate, snapshot)
            candidate.write_bytes(b"drift")
            with self.assertRaises(QualificationError):
                QUALIFY["require_same_file_identity"](candidate, snapshot)


if __name__ == "__main__":
    unittest.main()
