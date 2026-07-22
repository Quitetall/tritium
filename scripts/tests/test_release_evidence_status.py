from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "release-evidence-status.py")
EvidenceError = MODULE["EvidenceError"]
evaluate = MODULE["evaluate"]
render = MODULE["render"]
gate_row = MODULE["_gate_row"]
STATUS_MODULE = runpy.run_path(ROOT / "scripts" / "release-status")
status_main = STATUS_MODULE["main"]
WHEEL_MODULE = runpy.run_path(ROOT / "scripts" / "wheel-functional-smoke.py")
MATRIX_MODULE = runpy.run_path(ROOT / "scripts" / "aggregate-wheel-smoke.py")
CRATE_MODULE = runpy.run_path(ROOT / "scripts" / "qualify-crate-archives.py")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_fixture(root: Path) -> tuple[Path, dict, Path, Path]:
    candidate_root = root / "candidate"
    candidate_root.mkdir()
    artifact = candidate_root / "candidate.whl"
    artifact.write_bytes(b"qualified wheel bytes")
    candidate = candidate_root / "manifest.json"
    document = {
        "schema": "tritium.release-candidate.v1",
        "release": "1.1.0-rc.0",
        "source_revision": "a" * 40,
        "artifacts": [
            {
                "id": "cuda-wheel",
                "kind": "python-wheel",
                "path": artifact.name,
                "identity": {},
                "sbom": {},
                "provenance": {},
            }
        ],
    }
    candidate.write_bytes(canonical(document) + b"\n")
    evidence_root = root / "evidence"
    evidence_root.mkdir()
    return candidate, document, artifact, evidence_root


def cuda_receipt(path: Path, artifact: Path, *, run_id: str = "run-17") -> dict:
    value = {
        "schema": "tritium.cuda-training-qualification.v1",
        "source_revision": "a" * 40,
        "release": "1.1.0-rc.0",
        "run_id": run_id,
        "started_at_utc": "2026-07-21T12:00:00Z",
        "duration_ms": 4000.0,
        "command": ["python", "hf_cuda_worker.py"],
        "artifact": {
            "kind": "python-wheel",
            "name": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": sha256(artifact),
        },
        "machine": {
            "machine_id": "sha256:" + "b" * 64,
            "system": "Linux",
            "architecture": "x86_64",
        },
        "environment": {
            "python_version": "3.13.5",
            "torch_version": "2.11.0",
            "transformers_version": "5.5.3",
            "accelerate_version": "1.10.0",
            "cuda_runtime": "13.0",
            "cuda_driver": "610.43.03",
        },
        "device": {
            "index": 0,
            "uuid": "GPU-physical",
            "name": "NVIDIA GeForce RTX 4090",
            "compute_capability": "8.9",
            "total_memory_bytes": 25_000_000_000,
        },
        "workload": {
            "seed": 401,
            "mixed_precision": "fp16",
            "steps": 5,
            "batch_size": 1,
            "sequence_length": 8,
            "model_config_sha256": "c" * 64,
        },
        "measurements": {"elapsed_ms": 250.0, "steps_per_second": 20.0},
        "invariants": {
            "ternary_operator_host_transfers": 0,
            "ternary_operator_dtype": "torch.float16",
            "checkpoint_exact": True,
        },
        "result": "pass",
    }
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    path.write_bytes(canonical(value) + b"\n")
    return value


def registry(
    path: Path, candidate: Path, receipts: list[dict]
) -> None:
    path.write_bytes(
        canonical(
            {
                "schema": "tritium.release-evidence-registry.v1",
                "release": "1.1.0-rc.0",
                "source_revision": "a" * 40,
                "candidate_manifest_sha256": sha256(candidate),
                "receipts": receipts,
            }
        )
        + b"\n"
    )


def entry(
    receipt_path: Path, receipt: dict, *, kind: str = "cuda-training",
    parents: list[str] | None = None,
) -> dict:
    return {
        "id": receipt["receipt_id"],
        "kind": kind,
        "path": receipt_path.name,
        "sha256": sha256(receipt_path),
        "artifact_id": "cuda-wheel",
        "parents": parents or [],
    }


class ReleaseEvidenceStatusTests(unittest.TestCase):
    def test_crate_archive_receipt_binds_complete_candidate_inventory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, document, old_artifact, evidence_root = release_fixture(root)
            old_artifact.unlink()
            document["artifacts"] = []
            packages = []
            for name in ("alpha", "beta"):
                archive = candidate.parent / f"{name}-1.1.0-rc.0.crate"
                archive.write_bytes(name.encode())
                identity = {"sha256": sha256(archive), "bytes": archive.stat().st_size}
                artifact_id = f"crate-{name}"
                document["artifacts"].append(
                    {
                        "id": artifact_id, "kind": "rust-crate", "path": archive.name,
                        "identity": identity, "sbom": {}, "provenance": {},
                    }
                )
                packages.append(
                    {
                        "artifact_id": artifact_id, "name": name,
                        "version": "1.1.0-rc.0", "archive": archive.name,
                        "bytes": identity["bytes"], "sha256": identity["sha256"],
                    }
                )
            candidate.write_bytes(canonical(document) + b"\n")
            receipt = {
                "schema": CRATE_MODULE["SCHEMA"], "release": "1.1.0-rc.0",
                "source_revision": "a" * 40, "run_id": "crate-run-1",
                "started_at_utc": "2026-07-21T12:00:00Z", "duration_ms": 100.0,
                "machine": {
                    "machine_id": "sha256:" + "b" * 64,
                    "system": "Linux", "architecture": "x86_64",
                },
                "toolchain": {"cargo": "cargo 1.89.0", "rustc": "rustc 1.89.0"},
                "command_contract": (
                    "vendor-locked_then_empty-cargo-home_offline-locked-all-targets-v1"
                ),
                "dependency_lock_sha256": sha256(ROOT / "Cargo.lock"),
                "offline": True, "isolated_cargo_home": True, "packages": packages,
                "compiled_packages": ["alpha", "beta"], "result": "pass",
            }
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                canonical(receipt)
            ).hexdigest()
            receipt_path = evidence_root / "crates.json"
            CRATE_MODULE["_atomic_write"](receipt_path, receipt)
            registry_path = evidence_root / "registry.json"
            registry(
                registry_path, candidate,
                [{**entry(receipt_path, receipt, kind="crate-archive"), "artifact_id": "crate-alpha"}],
            )
            report = evaluate(registry_path, candidate, document)
            row = next(item for item in report["rows"] if item["id"] == "packages")
            self.assertEqual(row["satisfied_kinds"], ["crate-archive"])
            self.assertEqual(
                row["missing_kinds"],
                ["clean-install", "compatibility-matrix", "npm-archive"],
            )
            (candidate.parent / "beta-1.1.0-rc.0.crate").write_bytes(b"tampered")
            with self.assertRaises(EvidenceError):
                evaluate(registry_path, candidate, document)

    def test_complete_abi3_matrix_binds_three_candidate_wheels(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, document, old_artifact, evidence_root = release_fixture(root)
            old_artifact.unlink()
            document["artifacts"] = []
            identities = {}
            platforms = {
                "linux-x86_64-cpu": ("linux", "x86_64", "manylinux_2_28_x86_64"),
                "windows-x86_64-cpu": ("win32", "amd64", "win_amd64"),
                "macos-arm64-cpu": ("darwin", "arm64", "macosx_11_0_universal2"),
            }
            for target, (_, _, platform_tag) in platforms.items():
                wheel = candidate.parent / (
                    f"tritium_torch-1.1.0rc0-cp39-abi3-{platform_tag}.whl"
                )
                wheel.write_bytes(target.encode())
                identity = (wheel.name, sha256(wheel), wheel.stat().st_size)
                identities[target] = identity
                document["artifacts"].append(
                    {
                        "id": f"wheel-{target}",
                        "kind": "python-wheel",
                        "path": wheel.name,
                        "identity": {"sha256": identity[1], "bytes": identity[2]},
                        "sbom": {},
                        "provenance": {},
                    }
                )
            candidate.write_bytes(canonical(document) + b"\n")
            cells = root / "cells"
            cells.mkdir()
            for target, minors in MATRIX_MODULE["VERSIONS"].items():
                host_os, host_arch, platform_tag = platforms[target]
                wheel_name, wheel_sha, wheel_bytes = identities[target]
                for minor in minors:
                    cell_id = f"{target}-cp3.{minor}"
                    value = {
                        "schema": MATRIX_MODULE["SCHEMA"],
                        "cell_id": cell_id,
                        "target_id": target,
                        "source_revision": "a" * 40,
                        "passed": True,
                        "python_implementation": "CPython",
                        "python_version": f"3.{minor}.7",
                        "host_os": host_os,
                        "host_arch": host_arch,
                        "wheel": wheel_name,
                        "sha256": wheel_sha,
                        "bytes": wheel_bytes,
                        "version": "1.1.0rc0",
                        "platform_tag": platform_tag,
                    }
                    (cells / f"{cell_id}.json").write_bytes(canonical(value) + b"\n")
            receipt = MATRIX_MODULE["aggregate"](
                cells, "a" * 40, "1.1.0-rc.0", "matrix-run-1"
            )
            receipt_path = evidence_root / "matrix.json"
            MATRIX_MODULE["_atomic_write"](receipt_path, receipt)
            registry_path = evidence_root / "registry.json"
            registry(
                registry_path,
                candidate,
                [
                    {
                        **entry(
                            receipt_path, receipt, kind="compatibility-matrix"
                        ),
                        "artifact_id": "wheel-linux-x86_64-cpu",
                    }
                ],
            )
            report = evaluate(registry_path, candidate, document)
            packages = next(row for row in report["rows"] if row["id"] == "packages")
            self.assertEqual(packages["satisfied_kinds"], ["compatibility-matrix"])
            self.assertEqual(
                packages["missing_kinds"],
                ["clean-install", "crate-archive", "npm-archive"],
            )
            (candidate.parent / identities["windows-x86_64-cpu"][0]).write_bytes(
                b"tampered"
            )
            with self.assertRaisesRegex(EvidenceError, "candidate wheel bytes"):
                evaluate(registry_path, candidate, document)

    def test_clean_install_receipt_advances_package_gate_without_overclaim(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, artifact, evidence_root = release_fixture(Path(raw))
            evidence = {
                "schema": WHEEL_MODULE["SCHEMA"],
                "source_revision": "a" * 40,
                "passed": True,
                "wheel": artifact.name,
                "wheel_sha256": sha256(artifact),
                "distribution_version": "1.1.0rc0",
                "python_version": "3.13.5",
                "torch_version": "2.11.0",
                "transformers_version": "5.5.3",
                "safetensors_version": "0.8.0",
                "native_device": "cpu",
                "compiled_backends": ["cpu"],
                "tritium_module": "/venv/tritium/__init__.py",
                "converted_parameters": 256,
                "operations": sorted(WHEEL_MODULE["REQUIRED_OPERATIONS"]),
            }
            receipt = WHEEL_MODULE["build_receipt"](
                evidence, artifact, "1.1.0-rc.0", "clean-run-1",
                "2026-07-21T12:00:00Z", 100.0,
            )
            receipt_path = evidence_root / "clean.json"
            WHEEL_MODULE["_atomic_write"](receipt_path, receipt)
            registry_path = evidence_root / "registry.json"
            registry(
                registry_path, candidate,
                [entry(receipt_path, receipt, kind="clean-install")],
            )
            report = evaluate(registry_path, candidate, document)
            packages = next(row for row in report["rows"] if row["id"] == "packages")
            self.assertEqual(packages["status"], "MISSING")
            self.assertEqual(packages["satisfied_kinds"], ["clean-install"])
            self.assertEqual(
                packages["missing_kinds"],
                ["compatibility-matrix", "crate-archive", "npm-archive"],
            )

    def test_gate_status_distinguishes_pass_missing_and_structural(self):
        required = ("quality", "runtime")
        self.assertEqual(gate_row("gate", required, {})["status"], "MISSING")
        self.assertEqual(
            gate_row("gate", required, {"quality": "empirical"})["status"],
            "MISSING",
        )
        self.assertEqual(
            gate_row(
                "gate", required, {"quality": "empirical", "runtime": "structural"}
            )["status"],
            "STRUCTURAL_ONLY",
        )
        self.assertEqual(
            gate_row(
                "gate", required, {"quality": "empirical", "runtime": "empirical"}
            )["status"],
            "PASS",
        )

    def test_empty_registry_reports_every_gate_missing(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, _, evidence_root = release_fixture(Path(raw))
            registry_path = evidence_root / "registry.json"
            registry(registry_path, candidate, [])
            report = evaluate(registry_path, candidate, document)
            self.assertFalse(report["ready"])
            self.assertTrue(all(row["status"] == "MISSING" for row in report["rows"]))
            rendered = render(report)
            self.assertIn("MISSING", rendered)
            self.assertIn("EXTERNAL_AUTH_REQUIRED", rendered)

    def test_cuda_receipt_is_empirical_but_does_not_green_backend_gate(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, artifact, evidence_root = release_fixture(Path(raw))
            receipt_path = evidence_root / "cuda.json"
            receipt = cuda_receipt(receipt_path, artifact)
            registry_path = evidence_root / "registry.json"
            registry(registry_path, candidate, [entry(receipt_path, receipt)])
            report = evaluate(registry_path, candidate, document)
            backend = next(row for row in report["rows"] if row["id"] == "native-backends")
            self.assertEqual(backend["status"], "MISSING")
            self.assertEqual(backend["satisfied_kinds"], ["cuda-training"])
            self.assertEqual(backend["missing_kinds"], ["backend-manifest", "performance"])

    def test_rejects_unknown_kind_stale_candidate_and_artifact_drift(self):
        for mutation in ("kind", "candidate", "artifact"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                candidate, document, artifact, evidence_root = release_fixture(Path(raw))
                receipt_path = evidence_root / "cuda.json"
                receipt = cuda_receipt(receipt_path, artifact)
                receipt_entry = entry(receipt_path, receipt)
                if mutation == "kind":
                    receipt_entry["kind"] = "self-asserted-pass"
                registry_path = evidence_root / "registry.json"
                registry(registry_path, candidate, [receipt_entry])
                if mutation == "candidate":
                    candidate.write_bytes(candidate.read_bytes() + b" ")
                elif mutation == "artifact":
                    artifact.write_bytes(b"changed")
                with self.assertRaises(EvidenceError):
                    evaluate(registry_path, candidate, document)

    def test_rejects_missing_parent_and_duplicate_run_id(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, artifact, evidence_root = release_fixture(Path(raw))
            first_path = evidence_root / "first.json"
            first = cuda_receipt(first_path, artifact)
            registry_path = evidence_root / "registry.json"
            registry(
                registry_path,
                candidate,
                [entry(first_path, first, parents=["sha256:" + "f" * 64])],
            )
            with self.assertRaisesRegex(EvidenceError, "unknown parent"):
                evaluate(registry_path, candidate, document)

            second_path = evidence_root / "second.json"
            second = cuda_receipt(second_path, artifact, run_id=first["run_id"])
            second["duration_ms"] = 5000.0
            unsigned = dict(second)
            unsigned.pop("receipt_id")
            second["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
            second_path.write_bytes(canonical(second) + b"\n")
            registry(
                registry_path,
                candidate,
                [entry(first_path, first), entry(second_path, second)],
            )
            with self.assertRaisesRegex(EvidenceError, "duplicate run id"):
                evaluate(registry_path, candidate, document)

    def test_rejects_leaf_symlink_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, artifact, evidence_root = release_fixture(Path(raw))
            target = evidence_root / "actual.json"
            receipt = cuda_receipt(target, artifact)
            linked = evidence_root / "linked.json"
            linked.symlink_to(target.name)
            receipt_entry = entry(target, receipt)
            receipt_entry["path"] = linked.name
            registry_path = evidence_root / "registry.json"
            registry(registry_path, candidate, [receipt_entry])
            with self.assertRaisesRegex(EvidenceError, "symlink"):
                evaluate(registry_path, candidate, document)

    def test_release_status_registry_wiring_emits_partial_json_and_nonzero(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, artifact, evidence_root = release_fixture(Path(raw))
            receipt_path = evidence_root / "cuda.json"
            receipt = cuda_receipt(receipt_path, artifact)
            registry_path = evidence_root / "registry.json"
            registry(registry_path, candidate, [entry(receipt_path, receipt)])
            output = evidence_root / "status.json"
            globals_ = status_main.__globals__
            replacements = {
                "validate": lambda _candidate, _tool: document,
                "_git_gate": lambda _root, _revision: None,
                "_version_gate": lambda _root, _release: None,
            }
            original = {name: globals_[name] for name in replacements}
            globals_.update(replacements)
            stdout = io.StringIO()
            stderr = io.StringIO()
            try:
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "release-status",
                        "--candidate",
                        str(candidate),
                        "--registry",
                        str(registry_path),
                        "--json-output",
                        str(output),
                    ],
                ), mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
                    result = status_main()
            finally:
                globals_.update(original)
            self.assertEqual(result, 1)
            self.assertIn("LOCAL_RC_BLOCKED", stderr.getvalue())
            self.assertIn("native-backends", stdout.getvalue())
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["ready"])


if __name__ == "__main__":
    unittest.main()
