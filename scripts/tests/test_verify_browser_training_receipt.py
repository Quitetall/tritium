import base64
import copy
import hashlib
import json
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


def npm_receipt(archive: Path) -> dict:
    revision = "a" * 40
    payload = archive.read_bytes()
    unsigned = {
        "schema": "tritium.npm-archive-qualification.v1",
        "release": "1.1.0-rc.0",
        "source_revision": revision,
        "run_id": "npm-physical-1",
        "started_at_utc": "2026-08-07T12:00:00Z",
        "duration_ms": 1,
        "machine": {
            "machine_id": "sha256:" + "1" * 64,
            "system": "linux",
            "architecture": "x86_64",
        },
        "toolchain": {"node": "v24.18.1", "npm": "12.0.0"},
        "artifact": {
            "kind": "npm-archive",
            "name": archive.name,
            "package": "@tritium-ai/web@1.1.0-rc.0",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "integrity": "sha512-"
            + base64.b64encode(hashlib.sha512(payload).digest()).decode(),
        },
        "evidence": {
            "source_dirty": False,
            "entry_count": 16,
            "source_free": True,
            "installed_offline": True,
            "strict_typescript": True,
            "wasm_build_id": f"tritium-wasm@1.1.0-rc.0+source-git:{revision}",
            "wasm_guest_digest": "2" * 64,
        },
        "result": "pass",
    }
    return {
        **unsigned,
        "receipt_id": "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest(),
    }


def native_receipt(artifact_digest: str) -> dict:
    revision = "a" * 40

    def lifecycle(operation: str) -> dict:
        return {
            "result": "pass",
            "operation": operation,
            "artifact_sha256": artifact_digest,
            "input_digest": "1" * 64,
            "output_digest": "2" * 64,
            "peak_resident_bytes": 448,
            "scratch_bytes": 131296,
            "host_transfers": 0,
            "device_resident": True,
        }

    unsigned = {
        "schema": "tritium.browser-native-reference.v1",
        "result": "pass",
        "scenario_id": "salt-ste-sgd-256-v1",
        "source_revision": revision,
        "backend": "cpu",
        "backend_id": "cpu.reference.v1",
        "backend_build": f"tritium-train@1.1.0-rc.0+source-git:{revision}",
        "physical_device": "cpu:test",
        "manifest_digest": MODULE["MANIFEST_DIGEST"],
        "vector_digest": MODULE["VECTOR_DIGEST"],
        "artifact": {
            "name": "native.salt",
            "bytes": len(b"native artifact"),
            "sha256": artifact_digest,
        },
        "export": lifecycle("lifecycle.export"),
        "reload": {
            **lifecycle("lifecycle.reload"),
            "reloaded_sha256": artifact_digest,
        },
    }
    return {
        **unsigned,
        "receipt_id": "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest(),
    }


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
            adapter = {
                "vendor": "NVIDIA" if engine != "safari" else "Apple",
                "architecture": "Ada" if engine != "safari" else "M4",
                "device": "RTX 4090" if engine != "safari" else "Apple M4 Max",
                "description": "physical WebGPU adapter",
                "software": False,
            }
            limits = {
                "max_buffer_size": 1 << 30,
                "max_storage_buffer_binding_size": 1 << 29,
                "max_compute_workgroups_per_dimension": 65535,
                "max_storage_buffers_per_shader_stage": 10,
            }
            physical_device = ":".join(
                [adapter["vendor"], adapter["architecture"], adapter["device"], adapter["description"]]
            )
            cases = [
                {
                    "caseId": expected["caseId"],
                    "implementation": expected["implementation"],
                    "outputDigest": "0" * 64,
                    "scratchBytes": None
                    if expected["implementation"] == "wasm-validation"
                    else 0,
                    "scratchBytesMax": expected["scratchBytesMax"],
                }
                for expected in MODULE["CANONICAL_VECTOR_CASES"]
            ]
            vector = {
                "schemaId": "tritium.webgpu_vector_conformance_trace",
                "schemaVersion": 1,
                "implementation": "webgpu",
                "manifestDigest": MODULE["MANIFEST_DIGEST"],
                "vectorDigest": MODULE["VECTOR_DIGEST"],
                "caseCounts": {"valid": 72, "invalid": 45, "skipped": 0},
                "webgpuCaseTransactions": 68,
                "webgpuDispatches": 100,
                "wasmDispatches": 0,
                "wasmCodecCalls": 4,
                "wasmValidationCalls": 45,
                "explicitReadbacks": 80,
                "peakBufferBytes": 2048,
                "executionDigest": hashlib.sha256(MODULE["canonical"](cases)).hexdigest(),
                "cases": cases,
            }
            native_digest = hashlib.sha256(b"native artifact").hexdigest()
            raw_native_receipt = native_receipt(native_digest)
            reference_digest = hashlib.sha256(
                canonical(raw_native_receipt)
            ).hexdigest()
            native_lifecycle = lambda operation: {
                "result": "pass",
                "operation": operation,
                "artifactSha256": native_digest,
                "inputDigest": "1" * 64,
                "outputDigest": "2" * 64,
                "peakResidentBytes": 448,
                "scratchBytes": 131296,
                "hostTransfers": 0,
                "deviceResident": True,
            }
            receipts = [
                {
                    "operation": operation,
                    "completedSteps": 0 if operation in {"session.forward", "session.backward"} else 1,
                    "peakResidentBytes": 4096,
                    "buildId": "wgsl:" + "9" * 64 + ":browser-qualification:salt-ste-sgd-256-v1",
                    "physicalDevice": physical_device,
                }
                for operation in (
                    "session.forward",
                    "session.backward",
                    "session.step",
                    "session.checkpoint",
                    "session.resume",
                    "session.export",
                )
            ]
            browser_trace = {
                "schemaId": "tritium.physical_browser_training_lane_trace",
                "schemaVersion": 1,
                "scenarioId": "salt-ste-sgd-256-v1",
                "implementation": "webgpu",
                "manifestDigest": MODULE["MANIFEST_DIGEST"],
                "vectorDigest": MODULE["VECTOR_DIGEST"],
                "physicalDevice": physical_device,
                "buildId": "wgsl:" + "9" * 64 + ":browser-qualification:salt-ste-sgd-256-v1",
                "adapter": adapter,
                "limits": {
                    "maxBufferSize": limits["max_buffer_size"],
                    "maxStorageBufferBindingSize": limits[
                        "max_storage_buffer_binding_size"
                    ],
                    "maxComputeWorkgroupsPerDimension": limits[
                        "max_compute_workgroups_per_dimension"
                    ],
                    "maxStorageBuffersPerShaderStage": limits[
                        "max_storage_buffers_per_shader_stage"
                    ],
                },
                "vector": vector,
                "lifecycle": {
                    "prepare": True,
                    "forward": True,
                    "backward": True,
                    "optimizerStep": True,
                    "checkpointResume": True,
                    "exportReload": True,
                    "nativeArtifactParity": True,
                    "completedSteps": 1,
                    "checkpointSha256": "1" * 64,
                    "artifactSha256": native_digest,
                    "nativeArtifactSha256": native_digest,
                    "nativeReferenceDigest": reference_digest,
                    "receipts": receipts,
                },
                "faults": {
                    field: {
                        "passed": True,
                        "errorCode": (
                            "injected_allocation_failure"
                            if field == "allocationFailure"
                            else "cancelled"
                            if field == "cancellation"
                            else "expected"
                        ),
                        "stateAfter": None,
                        **(
                            {"observedEvents": 1}
                            if field in {"allocationFailure", "cancellation"}
                            else {}
                        ),
                    }
                    for field in MODULE["FAULT_TRACE_FIELDS"]
                },
                "explicitReadbacks": 87,
                "steadyStateReadbacks": 0,
                "wasmDispatches": 0,
                "peakBufferBytes": 4096,
            }
            browser_trace["executionDigest"] = hashlib.sha256(
                MODULE["canonical"](browser_trace)
            ).hexdigest()
            trace_evidence = {
                "schema": "tritium.browser-training-lane-evidence.v1",
                "run_id": f"{engine}-physical-1",
                "engine": engine,
                "source_revision": "a" * 40,
                "archive": {
                    "name": archive.name,
                    "bytes": archive.stat().st_size,
                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                },
                "npm_receipt": npm_receipt(archive),
                "native_receipt": raw_native_receipt,
                "native_reference": {
                    "schema": "tritium.browser-native-reference.v1",
                    "scenarioId": "salt-ste-sgd-256-v1",
                    "sourceRevision": "a" * 40,
                    "backend": "cpu",
                    "backendId": "cpu.reference.v1",
                    "backendBuild": "tritium-train@1.1.0-rc.0+source-git:" + "a" * 40,
                    "physicalDevice": "cpu:test",
                    "artifactName": "native.salt",
                    "artifactBytes": len(b"native artifact"),
                    "artifactSha256": native_digest,
                    "receiptId": raw_native_receipt["receipt_id"],
                    "receiptDigest": reference_digest,
                    "export": native_lifecycle("lifecycle.export"),
                    "reload": {
                        **native_lifecycle("lifecycle.reload"),
                        "reloadedSha256": native_digest,
                    },
                },
                "webdriver_capabilities": {
                    "browserName": engine,
                    "browserVersion": "140.0.1",
                    "platformName": (
                        "linux"
                        if os_name == "Linux"
                        else "windows"
                        if os_name == "Windows"
                        else "mac"
                    ),
                },
                "browser_trace": browser_trace,
            }
            trace = root / f"{engine}.trace.json"
            trace.write_bytes(MODULE["canonical"](trace_evidence) + b"\n")
            lanes.append(
                {
                    "engine": engine,
                    "browser_version": "140.0.1",
                    "os": {"name": os_name, "version": "26.0", "architecture": "arm64"},
                    "adapter": adapter,
                    "limits": limits,
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
                        "explicit_readbacks": 87,
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
                lambda value: value["lanes"][0]["adapter"].__setitem__(
                    "description", "Microsoft WARP adapter"
                ),
                "physical",
            ),
            (
                lambda value: value["lanes"][0]["adapter"].__setitem__(
                    "description", "Mesa lavapipe"
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
                "identity|distinct trace",
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

    def test_rejects_rehashed_forged_browser_execution_trace(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path, archive, receipt = self.fixture(root)
            trace_path = root / receipt["lanes"][0]["trace"]["file"]
            evidence = json.loads(trace_path.read_bytes())
            evidence["browser_trace"]["faults"]["deviceLoss"]["passed"] = False
            browser_trace = evidence["browser_trace"]
            browser_trace["executionDigest"] = hashlib.sha256(
                canonical(
                    {
                        key: value
                        for key, value in browser_trace.items()
                        if key != "executionDigest"
                    }
                )
            ).hexdigest()
            trace_path.write_bytes(canonical(evidence) + b"\n")
            receipt["lanes"][0]["trace"]["bytes"] = trace_path.stat().st_size
            receipt["lanes"][0]["trace"]["sha256"] = hashlib.sha256(
                trace_path.read_bytes()
            ).hexdigest()
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                canonical(
                    {
                        key: value
                        for key, value in receipt.items()
                        if key != "receipt_id"
                    }
                )
            ).hexdigest()
            path.write_bytes(canonical(receipt) + b"\n")
            with self.assertRaisesRegex(BrowserReceiptError, "fault trace"):
                validate(path, "a" * 40, "1.1.0-rc.0", archive)

    def test_rejects_rehashed_noncanonical_vector_inventory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path, archive, receipt = self.fixture(root)
            trace_path = root / receipt["lanes"][0]["trace"]["file"]
            evidence = json.loads(trace_path.read_bytes())
            vector = evidence["browser_trace"]["vector"]
            vector["cases"][0]["caseId"] = "fabricated.vector.case"
            vector["executionDigest"] = hashlib.sha256(
                canonical(vector["cases"])
            ).hexdigest()
            browser_trace = evidence["browser_trace"]
            browser_trace["executionDigest"] = hashlib.sha256(
                canonical(
                    {
                        key: value
                        for key, value in browser_trace.items()
                        if key != "executionDigest"
                    }
                )
            ).hexdigest()
            trace_path.write_bytes(canonical(evidence) + b"\n")
            receipt["lanes"][0]["trace"]["bytes"] = trace_path.stat().st_size
            receipt["lanes"][0]["trace"]["sha256"] = hashlib.sha256(
                trace_path.read_bytes()
            ).hexdigest()
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                canonical(
                    {
                        key: value
                        for key, value in receipt.items()
                        if key != "receipt_id"
                    }
                )
            ).hexdigest()
            path.write_bytes(canonical(receipt) + b"\n")
            with self.assertRaisesRegex(BrowserReceiptError, "canonical"):
                validate(path, "a" * 40, "1.1.0-rc.0", archive)

    def test_rejects_rehashed_native_build_prefix_or_suffix(self):
        exact_build = "tritium-train@1.1.0-rc.0+source-git:" + "a" * 40
        for forged_build in (exact_build + "-suffix", "prefix-" + exact_build):
            with self.subTest(forged_build=forged_build), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                path, archive, receipt = self.fixture(root)
                trace_path = root / receipt["lanes"][0]["trace"]["file"]
                evidence = json.loads(trace_path.read_bytes())
                evidence["native_reference"]["backendBuild"] = forged_build
                trace_path.write_bytes(canonical(evidence) + b"\n")
                receipt["lanes"][0]["trace"]["bytes"] = trace_path.stat().st_size
                receipt["lanes"][0]["trace"]["sha256"] = hashlib.sha256(
                    trace_path.read_bytes()
                ).hexdigest()
                receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                    canonical(
                        {
                            key: value
                            for key, value in receipt.items()
                            if key != "receipt_id"
                        }
                    )
                ).hexdigest()
                path.write_bytes(canonical(receipt) + b"\n")
                with self.assertRaisesRegex(BrowserReceiptError, "native reference identity"):
                    validate(path, "a" * 40, "1.1.0-rc.0", archive)

    def test_rejects_rehashed_dirty_retained_npm_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path, archive, receipt = self.fixture(root)
            trace_path = root / receipt["lanes"][0]["trace"]["file"]
            evidence = json.loads(trace_path.read_bytes())
            npm = evidence["npm_receipt"]
            npm["evidence"]["source_dirty"] = True
            npm["receipt_id"] = "sha256:" + hashlib.sha256(
                canonical(
                    {
                        key: value
                        for key, value in npm.items()
                        if key != "receipt_id"
                    }
                )
            ).hexdigest()
            trace_path.write_bytes(canonical(evidence) + b"\n")
            receipt["lanes"][0]["trace"]["bytes"] = trace_path.stat().st_size
            receipt["lanes"][0]["trace"]["sha256"] = hashlib.sha256(
                trace_path.read_bytes()
            ).hexdigest()
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                canonical(
                    {
                        key: value
                        for key, value in receipt.items()
                        if key != "receipt_id"
                    }
                )
            ).hexdigest()
            path.write_bytes(canonical(receipt) + b"\n")
            with self.assertRaisesRegex(BrowserReceiptError, "npm receipt"):
                validate(path, "a" * 40, "1.1.0-rc.0", archive)

    def test_rejects_rehashed_raw_native_receipt_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path, archive, receipt = self.fixture(root)
            trace_path = root / receipt["lanes"][0]["trace"]["file"]
            evidence = json.loads(trace_path.read_bytes())
            native = evidence["native_receipt"]
            native["backend_build"] += "-suffix"
            native["receipt_id"] = "sha256:" + hashlib.sha256(
                canonical(
                    {
                        key: value
                        for key, value in native.items()
                        if key != "receipt_id"
                    }
                )
            ).hexdigest()
            trace_path.write_bytes(canonical(evidence) + b"\n")
            receipt["lanes"][0]["trace"]["bytes"] = trace_path.stat().st_size
            receipt["lanes"][0]["trace"]["sha256"] = hashlib.sha256(
                trace_path.read_bytes()
            ).hexdigest()
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                canonical(
                    {
                        key: value
                        for key, value in receipt.items()
                        if key != "receipt_id"
                    }
                )
            ).hexdigest()
            path.write_bytes(canonical(receipt) + b"\n")
            with self.assertRaisesRegex(BrowserReceiptError, "native raw receipt"):
                validate(path, "a" * 40, "1.1.0-rc.0", archive)

    def test_rejects_unobserved_physical_fault_injection(self):
        for field in ("cancellation", "allocationFailure"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                path, archive, receipt = self.fixture(root)
                trace_path = root / receipt["lanes"][0]["trace"]["file"]
                evidence = json.loads(trace_path.read_bytes())
                browser_trace = evidence["browser_trace"]
                browser_trace["faults"][field]["observedEvents"] = 0
                browser_trace["executionDigest"] = hashlib.sha256(
                    canonical(
                        {
                            key: value
                            for key, value in browser_trace.items()
                            if key != "executionDigest"
                        }
                    )
                ).hexdigest()
                trace_path.write_bytes(canonical(evidence) + b"\n")
                receipt["lanes"][0]["trace"]["bytes"] = trace_path.stat().st_size
                receipt["lanes"][0]["trace"]["sha256"] = hashlib.sha256(
                    trace_path.read_bytes()
                ).hexdigest()
                receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                    canonical(
                        {
                            key: value
                            for key, value in receipt.items()
                            if key != "receipt_id"
                        }
                    )
                ).hexdigest()
                path.write_bytes(canonical(receipt) + b"\n")
                with self.assertRaisesRegex(BrowserReceiptError, "fault trace"):
                    validate(path, "a" * 40, "1.1.0-rc.0", archive)


if __name__ == "__main__":
    unittest.main()
