from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(
    ROOT / "scripts" / "verify-torch-dispatch-overhead-receipt.py"
)
canonical = MODULE["canonical"]
aggregate_case = MODULE["aggregate_case"]
validate = MODULE["validate"]
validate_trace = MODULE["validate_trace"]
DispatchOverheadError = MODULE["DispatchOverheadError"]


def _cache(hits: int) -> dict[str, int]:
    return {
        "capacity": 4096,
        "entries": 1,
        "hits": hits,
        "invalidations": 0,
        "misses": 1,
    }


def fixture(root: Path, *, wrapper_ratio: float = 1.04):
    wheel = root / "tritium_torch-1.1.0rc0-cp39-abi3-linux_x86_64.whl"
    wheel.write_bytes(b"exact wheel bytes")
    cases = []
    for ordinal, policy in enumerate(MODULE["POLICY_CASES"]):
        repetitions = policy["repetitions"]
        expected_hits = 2 + (
            MODULE["WARMUP_COUNT"] + MODULE["SAMPLE_COUNT"]
        ) * 2 * repetitions
        samples = [
            {
                "ordinal": sample,
                "order": "direct-first" if sample % 2 == 0 else "wrapper-first",
                "direct_total_ns": 1_000_000 * repetitions + sample,
                "wrapper_total_ns": round(
                    (1_000_000 * repetitions + sample) * wrapper_ratio
                ),
            }
            for sample in range(MODULE["SAMPLE_COUNT"])
        ]
        warmups = [
            {
                "ordinal": sample,
                "order": "direct-first" if sample % 2 == 0 else "wrapper-first",
                "direct_total_ns": 1_000_000 * repetitions,
                "wrapper_total_ns": round(
                    1_000_000 * repetitions * wrapper_ratio
                ),
            }
            for sample in range(MODULE["WARMUP_COUNT"])
        ]
        cases.append(
            {
                **policy,
                "parity_exact": True,
                "cache_before": _cache(0),
                "cache_after": _cache(expected_hits),
                "warmups": warmups,
                "samples": samples,
            }
        )
    trace = {
        "schema": MODULE["TRACE_SCHEMA"],
        "release": "1.1.0-rc.0",
        "source_revision": "a" * 40,
        "run_id": "dispatch-overhead-physical-1",
        "result": "complete",
        "wheel": {
            "name": wheel.name,
            "bytes": wheel.stat().st_size,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        },
        "runtime": {
            "python": "3.14.0",
            "torch": "2.11.0",
            "tritium": "1.1.0rc0",
            "source_identity": "source-git:" + "a" * 40,
            "module_path": "/venv/site-packages/tritium/__init__.py",
            "native_module_path": "/venv/site-packages/tritium/_tritium.abi3.so",
            "wheel_file_count": 12,
            "verified_installed_file_count": 12,
            "installed_tree_sha256": "sha256:" + "b" * 64,
        },
        "environment": {
            "system": "Linux",
            "machine": "x86_64",
            "cpu_model": "fixture cpu",
            "logical_cpu_count": 32,
            "affinity_before": [4, 5],
            "affinity_used": 4,
            "rayon_threads": 1,
            "torch_threads": 1,
            "torch_interop_threads": 1,
            "omp_threads": 1,
            "mkl_threads": 1,
            "source_tree_absent": True,
            "clock": "perf_counter_ns",
        },
        "policy": {
            "policy_id": MODULE["POLICY_ID"],
            "warmup_count": MODULE["WARMUP_COUNT"],
            "sample_count": MODULE["SAMPLE_COUNT"],
            "bootstrap_resamples": MODULE["BOOTSTRAP_RESAMPLES"],
            "bootstrap_confidence": MODULE["BOOTSTRAP_CONFIDENCE"],
            "overhead_limit_ratio": MODULE["OVERHEAD_LIMIT_RATIO"],
        },
        "cases": cases,
    }
    trace_path = root / "raw-trace.json"
    trace_path.write_bytes(canonical(trace) + b"\n")
    measurements = [
        aggregate_case(case, ordinal) for ordinal, case in enumerate(cases)
    ]
    receipt = {
        "schema": MODULE["SCHEMA"],
        "receipt_id": "",
        "result": "pass",
        "release": trace["release"],
        "source_revision": trace["source_revision"],
        "run_id": trace["run_id"],
        "wheel": trace["wheel"],
        "policy": trace["policy"],
        "environment": trace["environment"],
        "measurements": measurements,
        "trace": {
            "path": trace_path.name,
            "bytes": trace_path.stat().st_size,
            "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        },
    }
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    receipt_path = root / "receipt.json"
    receipt_path.write_bytes(canonical(receipt) + b"\n")
    return wheel, trace_path, receipt_path, trace, receipt


class DispatchOverheadReceiptTests(unittest.TestCase):
    def test_accepts_complete_paired_trace_under_five_percent(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel, trace_path, receipt_path, trace, receipt = fixture(Path(raw))
            self.assertEqual(
                validate_trace(
                    trace_path,
                    expected_revision="a" * 40,
                    expected_release="1.1.0-rc.0",
                    expected_wheel=wheel,
                ),
                trace,
            )
            self.assertEqual(
                validate(
                    receipt_path,
                    expected_revision="a" * 40,
                    expected_release="1.1.0-rc.0",
                    expected_wheel=wheel,
                ),
                receipt,
            )

    def test_rejects_six_percent_wrapper_overhead(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel, trace_path, _receipt_path, trace, _receipt = fixture(Path(raw))
            for case in trace["cases"]:
                for sample in case["samples"]:
                    sample["wrapper_total_ns"] = round(
                        sample["direct_total_ns"] * 1.06
                    )
            trace_path.write_bytes(canonical(trace) + b"\n")
            with self.assertRaisesRegex(DispatchOverheadError, "five-percent"):
                validate_trace(
                    trace_path,
                    expected_revision="a" * 40,
                    expected_release="1.1.0-rc.0",
                    expected_wheel=wheel,
                )

    def test_rejects_reordered_sample_schedule(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel, trace_path, _receipt_path, trace, _receipt = fixture(Path(raw))
            trace["cases"][0]["samples"][1]["order"] = "direct-first"
            trace_path.write_bytes(canonical(trace) + b"\n")
            with self.assertRaisesRegex(DispatchOverheadError, "alternating"):
                validate_trace(
                    trace_path,
                    expected_revision="a" * 40,
                    expected_release="1.1.0-rc.0",
                    expected_wheel=wheel,
                )

    def test_rejects_wheel_source_identity_from_other_revision(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel, trace_path, _receipt_path, trace, _receipt = fixture(Path(raw))
            trace["runtime"]["source_identity"] = "source-git:" + "b" * 40
            trace_path.write_bytes(canonical(trace) + b"\n")
            with self.assertRaisesRegex(DispatchOverheadError, "source identity"):
                validate_trace(
                    trace_path,
                    expected_revision="a" * 40,
                    expected_release="1.1.0-rc.0",
                    expected_wheel=wheel,
                )

    def test_rejects_installed_wheel_version_from_other_release(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel, trace_path, _receipt_path, trace, _receipt = fixture(Path(raw))
            trace["runtime"]["tritium"] = "1.1.0rc999"
            trace_path.write_bytes(canonical(trace) + b"\n")
            with self.assertRaisesRegex(DispatchOverheadError, "version"):
                validate_trace(
                    trace_path,
                    expected_revision="a" * 40,
                    expected_release="1.1.0-rc.0",
                    expected_wheel=wheel,
                )


if __name__ == "__main__":
    unittest.main()
