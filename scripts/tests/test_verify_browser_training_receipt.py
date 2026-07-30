import copy
import hashlib
from pathlib import Path
import runpy
import tempfile
import unittest


MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "verify-browser-training-receipt.py"
)
validate = MODULE["validate"]
canonical = MODULE["canonical"]
BrowserReceiptError = MODULE["BrowserReceiptError"]


class BrowserTrainingReceiptTests(unittest.TestCase):
    def fixture(self, root: Path):
        archive = root / "tritium-ai-web-1.1.0-rc.0.tgz"
        archive.write_bytes(b"exact npm archive")
        lanes = []
        for engine, os_name in (
            ("chrome", "Linux"),
            ("firefox", "Windows"),
            ("safari", "macOS"),
        ):
            trace = root / f"{engine}.trace.json"
            trace.write_bytes((engine + " physical trace").encode())
            lanes.append(
                {
                    "engine": engine,
                    "browser_version": "140.0.1",
                    "os": {"name": os_name, "version": "26.0", "architecture": "arm64"},
                    "adapter": {
                        "vendor": "NVIDIA" if engine != "safari" else "Apple",
                        "architecture": "Ada" if engine != "safari" else "M4",
                        "device": "RTX 4090" if engine != "safari" else "Apple M4 Max",
                        "description": "physical WebGPU adapter",
                        "software": False,
                    },
                    "limits": {
                        "max_buffer_size": 1 << 30,
                        "max_storage_buffer_binding_size": 1 << 29,
                        "max_compute_workgroups_per_dimension": 65535,
                        "max_storage_buffers_per_shader_stage": 10,
                    },
                    "case_counts": {"valid": 72, "invalid": 45, "skipped": 0},
                    "lifecycle": {
                        "prepare": True,
                        "forward": True,
                        "backward": True,
                        "optimizer_step": True,
                        "checkpoint_resume": True,
                        "export_reload": True,
                        "native_artifact_parity": True,
                    },
                    "faults": {
                        "device_loss": True,
                        "allocation_failure": True,
                        "malformed_checkpoint": True,
                        "malformed_salt": True,
                        "cancellation": True,
                        "out_of_order": True,
                    },
                    "trace": {
                        "file": trace.name,
                        "bytes": trace.stat().st_size,
                        "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                        "steady_state_readbacks": 0,
                        "wasm_dispatches": 0,
                        "explicit_readbacks": 3,
                        "peak_buffer_bytes": 4096,
                    },
                }
            )
        receipt = {
            "schema": MODULE["SCHEMA"],
            "result": "pass",
            "release": "1.1.0-rc.0",
            "source_revision": "a" * 40,
            "run_id": "physical-browser-run-1",
            "artifact": {
                "kind": "npm-archive",
                "name": archive.name,
                "bytes": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            },
            "manifest_digest": MODULE["MANIFEST_DIGEST"],
            "vector_digest": MODULE["VECTOR_DIGEST"],
            "lanes": lanes,
        }
        receipt["receipt_id"] = (
            "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
        )
        path = root / "receipt.json"
        path.write_bytes(canonical(receipt) + b"\n")
        return path, archive, receipt

    def test_accepts_complete_three_engine_physical_evidence(self):
        with tempfile.TemporaryDirectory() as raw:
            path, archive, receipt = self.fixture(Path(raw))
            self.assertEqual(validate(path, "a" * 40, "1.1.0-rc.0", archive), receipt)

    def test_rejects_synthetic_partial_and_fallback_claims(self):
        mutations = (
            (
                lambda value: value["lanes"][0]["adapter"].__setitem__(
                    "software", True
                ),
                "physical",
            ),
            (
                lambda value: value["lanes"][1]["case_counts"].__setitem__(
                    "invalid", 43
                ),
                "117",
            ),
            (
                lambda value: value["lanes"][2]["faults"].__setitem__(
                    "device_loss", False
                ),
                "fault",
            ),
            (
                lambda value: value["lanes"][0]["trace"].__setitem__(
                    "wasm_dispatches", 1
                ),
                "fallback",
            ),
            (
                lambda value: value["lanes"][1].__setitem__(
                    "trace", copy.deepcopy(value["lanes"][0]["trace"])
                ),
                "distinct trace",
            ),
            (lambda value: value["lanes"].pop(), "three lanes"),
        )
        for mutate, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as raw:
                path, archive, receipt = self.fixture(Path(raw))
                changed = copy.deepcopy(receipt)
                mutate(changed)
                changed["receipt_id"] = (
                    "sha256:"
                    + hashlib.sha256(
                        canonical(
                            {
                                key: value
                                for key, value in changed.items()
                                if key != "receipt_id"
                            }
                        )
                    ).hexdigest()
                )
                path.write_bytes(canonical(changed) + b"\n")
                with self.assertRaisesRegex(BrowserReceiptError, message):
                    validate(path, "a" * 40, "1.1.0-rc.0", archive)

    def test_rejects_trace_tampering_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            path, archive, receipt = self.fixture(Path(raw))
            (Path(raw) / "chrome.trace.json").write_bytes(b"tampered trace bytes")
            with self.assertRaisesRegex(BrowserReceiptError, "trace bytes"):
                validate(path, "a" * 40, "1.1.0-rc.0", archive)

            receipt["lanes"][0]["trace"]["file"] = "../outside.trace"
            receipt["receipt_id"] = (
                "sha256:"
                + hashlib.sha256(
                    canonical(
                        {
                            key: value
                            for key, value in receipt.items()
                            if key != "receipt_id"
                        }
                    )
                ).hexdigest()
            )
            path.write_bytes(canonical(receipt) + b"\n")
            with self.assertRaisesRegex(BrowserReceiptError, "unsafe"):
                validate(path, "a" * 40, "1.1.0-rc.0", archive)


if __name__ == "__main__":
    unittest.main()
