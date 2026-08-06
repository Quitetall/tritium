from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "verify-torch-dispatch-cuda-receipt.py")
ReceiptError = MODULE["ReceiptError"]
canonical = MODULE["canonical"]
validate = MODULE["validate"]

CUDA_TESTS = MODULE["CUDA_TESTS"]
MEMCHECK_TESTS = MODULE["MEMCHECK_TESTS"]


def record(path: Path) -> dict[str, object]:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def junit(names: tuple[str, ...]) -> str:
    cases = "".join(f"<testcase name='{name}'/>" for name in names)
    count = len(names)
    return (
        f"<testsuites tests='{count}' failures='0' errors='0' skipped='0'>"
        f"<testsuite tests='{count}' failures='0' errors='0' skipped='0'>"
        f"{cases}</testsuite></testsuites>"
    )


def fixture(root: Path) -> tuple[Path, Path, dict[str, object]]:
    root.mkdir(parents=True, exist_ok=True)
    wheel = root / "tritium_torch-1.1.0rc0-cp39-abi3-linux_x86_64.whl"
    wheel.write_bytes(b"wheel")
    source = root / "test_torch_dispatch.py"
    source.write_text("def test_native_cuda(): pass\n", encoding="utf-8")
    suite = root / "suite-junit.xml"
    suite.write_text(junit(CUDA_TESTS), encoding="utf-8")
    memcheck = root / "memcheck-junit.xml"
    memcheck.write_text(junit(MEMCHECK_TESTS), encoding="utf-8")
    log = root / "compute-sanitizer.log"
    log.write_text("========= ERROR SUMMARY: 0 errors\n", encoding="utf-8")
    receipt = root / "receipt.json"
    value: dict[str, object] = {
        "schema": "tritium.torch-dispatch-cuda-qualification.v1",
        "receipt_id": "",
        "result": "pass",
        "release": "1.1.0-rc.0",
        "source_revision": "a" * 40,
        "run_id": "cuda-dispatch-17",
        "artifact": {"kind": "python-wheel", **record(wheel)},
        "environment": {
            "python_version": "3.14.6",
            "torch_version": "2.11.0+cu130",
            "tritium_version": "1.1.0rc0",
            "cuda_runtime": "13.0",
            "cuda_driver": "610.57.04",
            "source_identity": "source-git:" + "a" * 40,
        },
        "device": {
            "index": 0,
            "uuid": "GPU-physical",
            "name": "NVIDIA GeForce RTX 4090",
            "compute_capability": "8.9",
            "total_memory_bytes": 25_000_000_000,
        },
        "source": {
            "path": "crates/tritium-py/tests/test_torch_dispatch.py",
            "git_blob": hashlib.sha1(
                f"blob {source.stat().st_size}\0".encode() + source.read_bytes()
            ).hexdigest(),
            **record(source),
        },
        "suite": {
            "selector": "native_cuda",
            "tests": list(CUDA_TESTS),
            "passed": len(CUDA_TESTS),
            "junit": record(suite),
        },
        "sanitizer": {
            "tool": "compute-sanitizer",
            "version": "2026.2.1.0",
            "error_summary": 0,
            "tests": list(MEMCHECK_TESTS),
            "passed": len(MEMCHECK_TESTS),
            "junit": record(memcheck),
            "log": record(log),
        },
    }
    unsigned = {key: item for key, item in value.items() if key != "receipt_id"}
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    receipt.write_bytes(canonical(value) + b"\n")
    return receipt, wheel, value


class VerifyTorchDispatchCudaReceiptTests(unittest.TestCase):
    def test_accepts_exact_physical_cuda_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            receipt, wheel, expected = fixture(Path(raw))
            self.assertEqual(
                validate(receipt, "a" * 40, "1.1.0-rc.0", wheel), expected
            )

    def test_rejects_identity_and_result_drift(self):
        mutations = (
            ("source_revision", "c" * 40),
            ("result", "fail"),
            ("receipt_id", "sha256:" + "0" * 64),
        )
        for field, replacement in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                receipt, wheel, value = fixture(Path(raw))
                value[field] = replacement
                receipt.write_bytes(canonical(value) + b"\n")
                with self.assertRaises(ReceiptError):
                    validate(receipt, "a" * 40, "1.1.0-rc.0", wheel)

    def test_rejects_installed_distribution_version_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            receipt, wheel, value = fixture(Path(raw))
            value["environment"]["tritium_version"] = "1.1.0rc1"
            unsigned = {key: item for key, item in value.items() if key != "receipt_id"}
            value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
            receipt.write_bytes(canonical(value) + b"\n")
            with self.assertRaises(ReceiptError):
                validate(receipt, "a" * 40, "1.1.0-rc.0", wheel)

    def test_rejects_test_set_and_sanitizer_claim_drift(self):
        mutations = (
            ("suite", "tests", [*CUDA_TESTS[:-1]]),
            ("suite", "passed", 6),
            ("sanitizer", "tests", [*MEMCHECK_TESTS, "extra"]),
            ("sanitizer", "error_summary", 1),
        )
        for section, field, replacement in mutations:
            with self.subTest(section=section, field=field), tempfile.TemporaryDirectory() as raw:
                receipt, wheel, value = fixture(Path(raw))
                value[section][field] = replacement
                unsigned = {key: item for key, item in value.items() if key != "receipt_id"}
                value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
                receipt.write_bytes(canonical(value) + b"\n")
                with self.assertRaises(ReceiptError):
                    validate(receipt, "a" * 40, "1.1.0-rc.0", wheel)

    def test_rejects_retained_file_and_wheel_drift(self):
        for target in ("suite-junit.xml", "compute-sanitizer.log", "wheel"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as raw:
                receipt, wheel, _ = fixture(Path(raw))
                path = wheel if target == "wheel" else Path(raw) / target
                path.write_bytes(path.read_bytes() + b"drift")
                with self.assertRaises(ReceiptError):
                    validate(receipt, "a" * 40, "1.1.0-rc.0", wheel)

    def test_rejects_unknown_fields_and_symlinked_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            receipt, wheel, value = fixture(root)
            value["unknown"] = True
            unsigned = {key: item for key, item in value.items() if key != "receipt_id"}
            value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
            receipt.write_bytes(canonical(value) + b"\n")
            with self.assertRaises(ReceiptError):
                validate(receipt, "a" * 40, "1.1.0-rc.0", wheel)

            receipt, wheel, _ = fixture(root)
            link = root / "link.json"
            link.symlink_to(receipt)
            with self.assertRaises(ReceiptError):
                validate(link, "a" * 40, "1.1.0-rc.0", wheel)


if __name__ == "__main__":
    unittest.main()
