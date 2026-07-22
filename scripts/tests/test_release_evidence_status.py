from __future__ import annotations

import base64
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
NPM_MODULE = runpy.run_path(ROOT / "scripts" / "verify-npm-archive-receipt.py")
OCI_RUNTIME_MODULE = runpy.run_path(ROOT / "scripts" / "qualify-oci-runtime.py")
OCI_SECURITY_MODULE = runpy.run_path(ROOT / "scripts" / "qualify-oci-security.py")
TUTORIAL_RECEIPT_MODULE = runpy.run_path(
    ROOT / "crates/tritium-py/python/tritium/torch/tutorial_receipt.py"
)


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


def oci_runtime_receipt(path: Path, artifact: Path, *, flavor: str = "cpu") -> dict:
    startup = {
        "schema_version": 1,
        "artifact_kind": "qwen3.6-language-mtp-salt-v2-hf-bundle",
        "server_source_revision": "a" * 40,
        "server_build_id": "tritium-serve:1.1.0-rc.0:" + "a" * 40,
        "model_source_revision": "b" * 40,
        "manifest_package_id": "c" * 64,
        "salt_package_id": "d" * 64,
        "preserved_package_id": "e" * 64,
        "config_package_id": "f" * 64,
        "profile": "compact-v1",
        "codec": "b3",
        "backend_policy": flavor,
        "effective_backend": flavor,
        "physical_device_id": "cpu" if flavor == "cpu" else "cuda:0:GPU-physical",
        "loaded_bundle_bytes": 100,
        "resident_bytes": 80,
        "self_test_digest": "1" * 64,
    }
    startup_sha256 = hashlib.sha256(
        OCI_RUNTIME_MODULE["canonical"](startup)
    ).hexdigest()
    value = {
        "schema": OCI_RUNTIME_MODULE["SCHEMA"], "release": "1.1.0-rc.0",
        "source_revision": "a" * 40, "run_id": f"oci-{flavor}-1", "flavor": flavor,
        "image": "registry.example/tritium@sha256:" + "2" * 64,
        "image_id": "sha256:" + "3" * 64,
        "image_manifest_digest": "sha256:" + "2" * 64,
        "artifact": {"kind": "oci-image", "name": artifact.name,
                     "bytes": artifact.stat().st_size, "sha256": sha256(artifact)},
        "manifest": {"schema": "tritium.file-identity.v1", "bytes": 42,
                     "sha256": "4" * 64, "blake3": "c" * 64},
        "profile": "compact-v1", "startup_receipt": startup,
        "faults": {
            "unauthenticated_status": 401, "wrong_token_status": 401,
            "malformed_json_status": 400, "rate_limited_status": 429,
            "retry_after_seconds": 60, "rate_rejections_before": 0,
            "rate_rejections_after": 1, "replacement_principal_status": 400,
            "queue_flood_clients": 3, "slow_reader_tokens": 32,
            "queue_rejections_before": 0, "queue_rejections_after": 1,
            "disconnects_before": 0, "disconnects_after": 1,
            "accepted_streams": 2, "rejected_streams": 1,
            "settled_queue_depth": 0, "worker_alive": 1,
            "queue_capacity": 1, "saturated_queue_depth": 1, "slow_hold_ms": 1000,
            "tokens_out_before_hold": 0, "tokens_out_after_hold": 1,
            "recovery_status": 200, "recovery_ms": 1,
            "recovery_timeout_ms": 60000,
        },
        "shutdown_scenarios": [
            {"phase": "queue", "observed_worker_phase": "decode", "queue_depth": 1,
             "signal": "SIGTERM", "container_id": "1" * 64,
             "image_id": "sha256:" + "3" * 64,
             "startup_receipt_sha256": startup_sha256, "prompt_sha256": "4" * 64,
             "prompt_bytes": 4, "prompt_repetitions": 1, "max_tokens": 32,
             "observation_to_signal_ms": 5, "observation_budget_ms": 2000,
             "exit_code": 0, "shutdown_ms": 10, "budget_ms": 35000},
            {"phase": "prefill", "observed_worker_phase": "prefill", "queue_depth": 0,
             "signal": "SIGTERM", "container_id": "2" * 64,
             "image_id": "sha256:" + "3" * 64,
             "startup_receipt_sha256": startup_sha256, "prompt_sha256": "5" * 64,
             "prompt_bytes": 40, "prompt_repetitions": 256, "max_tokens": 1,
             "observation_to_signal_ms": 5, "observation_budget_ms": 2000,
             "exit_code": 0, "shutdown_ms": 20, "budget_ms": 35000},
            {"phase": "decode", "observed_worker_phase": "decode", "queue_depth": 0,
             "signal": "SIGTERM", "container_id": "3" * 64,
             "image_id": "sha256:" + "3" * 64,
             "startup_receipt_sha256": startup_sha256, "prompt_sha256": "4" * 64,
             "prompt_bytes": 4, "prompt_repetitions": 1, "max_tokens": 32,
             "observation_to_signal_ms": 5, "observation_budget_ms": 2000,
             "exit_code": 0, "shutdown_ms": 10, "budget_ms": 35000},
        ],
        "checks": list(OCI_RUNTIME_MODULE["CHECKS"]),
        "started_at_utc": "2026-07-21T12:00:00+00:00",
        "timing": {"startup_ms": 100.0, "shutdown_ms": 20},
        "machine": {"id": "sha256:" + "5" * 64, "system": "Linux",
                    "architecture": "x86_64", "docker_server": "28.0.0",
                    "gpu": None if flavor == "cpu" else {
                        "uuid": "GPU-physical", "name": "RTX 4090",
                        "driver_version": "610.43.03",
                    }},
        "result": "pass",
    }
    value["receipt_id"] = "sha256:" + hashlib.sha256(
        OCI_RUNTIME_MODULE["canonical"](value)
    ).hexdigest()
    path.write_bytes(OCI_RUNTIME_MODULE["canonical"](value))
    return value


def oci_security_receipt(path: Path, artifact: Path, *, flavor: str = "cpu") -> dict:
    common = [
        "/usr/bin/trivy", "--cache-dir", "/cache", "image", "--input",
        str(artifact.resolve()), "--format", "json", "--offline-scan",
        "--skip-db-update", "--skip-java-db-update", "--skip-check-update",
    ]
    value = {
        "schema": OCI_SECURITY_MODULE["SCHEMA"], "release": "1.1.0-rc.0",
        "source_revision": "a" * 40, "run_id": f"security-{flavor}-1",
        "flavor": flavor, "started_at_utc": "2026-07-21T12:00:00+00:00",
        "duration_ms": 100.0,
        "artifact": {"kind": "oci-image", "name": artifact.name,
                     "bytes": artifact.stat().st_size, "sha256": sha256(artifact)},
        "scanner": {"name": "trivy", "version": "0.69.1",
                    "executable_sha256": "6" * 64,
                    "commands": [
                        common + ["--scanners", "vuln", "--severity", "HIGH,CRITICAL",
                                  "--output", "/tmp/vulnerability.json"],
                        common + ["--scanners", "secret", "--output", "/tmp/secret.json"],
                    ]},
        "database": {"updated_at": "2026-07-21T06:00:00Z",
                     "downloaded_at": "2026-07-21T06:01:00Z",
                     "next_update": "2026-07-21T12:00:01Z",
                     "trivy_db_sha256": "7" * 64, "metadata_sha256": "8" * 64,
                     "max_age_hours": 24.0},
        "findings": {"high_or_critical_vulnerabilities": 0, "secret_findings": 0},
        "result": "pass",
    }
    value["receipt_id"] = "sha256:" + hashlib.sha256(
        OCI_SECURITY_MODULE["canonical"](value)
    ).hexdigest()
    path.write_bytes(OCI_SECURITY_MODULE["canonical"](value))
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
    def test_distributed_receipt_advances_frontend_gate_through_strict_dispatch(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, artifact, evidence_root = release_fixture(Path(raw))
            receipt_path = evidence_root / "distributed.json"
            receipt_path.write_bytes(b"{}\n")
            receipt = {
                "receipt_id": "sha256:" + "9" * 64,
                "run_id": "two-gpu-run-1",
                "artifact": {"kind": "python-wheel"},
            }
            registry_path = evidence_root / "registry.json"
            registry(
                registry_path, candidate,
                [{
                    "id": receipt["receipt_id"], "kind": "distributed-training",
                    "path": receipt_path.name, "sha256": sha256(receipt_path),
                    "artifact_id": "cuda-wheel", "parents": [],
                }],
            )
            loader = mock.Mock(return_value=receipt)
            with mock.patch.dict(
                evaluate.__globals__, {"validate_distributed_receipt": loader}
            ):
                report = evaluate(registry_path, candidate, document)
            frontend = next(row for row in report["rows"] if row["id"] == "pytorch-hf")
            self.assertEqual(frontend["satisfied_kinds"], ["distributed-training"])
            loader.assert_called_once_with(
                receipt_path.resolve(), "a" * 40, "1.1.0-rc.0", artifact
            )

    def test_hf_lifecycle_binds_candidate_wheel_and_advances_frontend_gate(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, artifact, evidence_root = release_fixture(Path(raw))
            document["artifacts"][0]["identity"] = {
                "sha256": sha256(artifact), "bytes": artifact.stat().st_size,
            }
            candidate.write_bytes(canonical(document) + b"\n")
            checkpoint = evidence_root / "hf-checkpoint"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"HF checkpoint")
            tree = TUTORIAL_RECEIPT_MODULE["tree_identity"](checkpoint)
            receipt = {
                "schema": "tritium.hf-lifecycle.v1", "passed": True,
                "device": "cpu", "seed": 97, "torch_version": "2.11.0",
                "transformers_version": "5.5.3",
                "distribution_version": "1.1.0rc0",
                "tritium_module": "/venv/tritium/__init__.py",
                "source_revision": "a" * 40, "release": "1.1.0-rc.0",
                "run_id": "hf-lifecycle-run-1", "wheel_name": artifact.name,
                "wheel_bytes": artifact.stat().st_size,
                "wheel_sha256": "sha256:" + sha256(artifact),
                "input_ids": [[1, 2, 3, 4]], "initial_loss": 1.0,
                "gradient_norm": 1.0, "optimizer_steps": 1,
                "converted_parameters": 8,
                "recipe": TUTORIAL_RECEIPT_MODULE["EXPECTED_HF_RECIPE"],
                "recipe_sha256": "sha256:" + hashlib.sha256(canonical(
                    TUTORIAL_RECEIPT_MODULE["EXPECTED_HF_RECIPE"]
                )).hexdigest(),
                "tied_before_save": True, "tied_after_reload": True,
                "safe_serialization": True, "checkpoint_dir": "hf-checkpoint",
                "checkpoint_bytes": tree["bytes"],
                "checkpoint_file_count": tree["file_count"],
                "checkpoint_tree_sha256": tree["sha256"],
                "logits_sha256": "sha256:" + "2" * 64,
            }
            receipt["receipt_id"] = TUTORIAL_RECEIPT_MODULE["receipt_id"](receipt)
            receipt_path = evidence_root / "hf-lifecycle.json"
            receipt_path.write_bytes(canonical(receipt) + b"\n")
            registry_path = evidence_root / "registry.json"
            registry(
                registry_path, candidate,
                [entry(receipt_path, receipt, kind="frontend-lifecycle")],
            )

            report = evaluate(registry_path, candidate, document)
            frontend = next(row for row in report["rows"] if row["id"] == "pytorch-hf")
            self.assertEqual(frontend["satisfied_kinds"], ["frontend-lifecycle"])
            self.assertIn("installed-qat-tutorial", frontend["missing_kinds"])

    def test_installed_qat_tutorial_binds_candidate_wheel_and_advances_frontend_gate(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, artifact, evidence_root = release_fixture(Path(raw))
            document["artifacts"][0]["identity"] = {
                "sha256": sha256(artifact),
                "bytes": artifact.stat().st_size,
            }
            candidate.write_bytes(canonical(document) + b"\n")
            artifact_dir = evidence_root / "qat-hard"
            artifact_dir.mkdir()
            (artifact_dir / "manifest.json").write_bytes(b"strict hard artifact")
            checkpoint = evidence_root / "latent-checkpoint"
            checkpoint.mkdir()
            model = checkpoint / "model.safetensors"
            optimizer = checkpoint / "optimizer.pt"
            model.write_bytes(b"latent model")
            optimizer.write_bytes(b"optimizer state")
            tree = TUTORIAL_RECEIPT_MODULE["tree_identity"](artifact_dir)
            receipt = {
                "schema": "tritium.installed-qat-tutorial.v3",
                "passed": True,
                "device": "cpu",
                "seed": 73,
                "torch_version": "2.11.0",
                "distribution_version": "1.1.0rc0",
                "tritium_module": "/venv/tritium/__init__.py",
                "source_revision": "a" * 40,
                "release": "1.1.0-rc.0",
                "run_id": "tutorial-run-1",
                "wheel_name": artifact.name,
                "wheel_bytes": artifact.stat().st_size,
                "wheel_sha256": "sha256:" + sha256(artifact),
                "loss": 1.0,
                "gradient_norm": 1.0,
                "converted_parameters": 1,
                "aliases": ["embed.weight", "head.weight"],
                "algorithm_id": "tritium.additive-2/tritium.salt-ste@1",
                "planes": 2,
                "artifact_id": "sha256:" + "1" * 64,
                "hard_state_digest": "sha256:" + "2" * 64,
                "artifact_dir": "qat-hard",
                "hard_artifact_bytes": tree["bytes"],
                "hard_artifact_file_count": tree["file_count"],
                "hard_artifact_tree_sha256": tree["sha256"],
                "checkpoint_model_bytes": model.stat().st_size,
                "checkpoint_model_sha256": "sha256:" + sha256(model),
                "checkpoint_optimizer_bytes": optimizer.stat().st_size,
                "checkpoint_optimizer_sha256": "sha256:" + sha256(optimizer),
                "optimizer_state_entries": 1,
                "resume_steps": 1,
            }
            receipt["receipt_id"] = TUTORIAL_RECEIPT_MODULE["receipt_id"](receipt)
            receipt_path = evidence_root / "tutorial.json"
            receipt_path.write_bytes(canonical(receipt) + b"\n")
            registry_path = evidence_root / "registry.json"
            registry(
                registry_path,
                candidate,
                [entry(receipt_path, receipt, kind="installed-qat-tutorial")],
            )

            report = evaluate(registry_path, candidate, document)
            frontend = next(row for row in report["rows"] if row["id"] == "pytorch-hf")
            self.assertEqual(frontend["satisfied_kinds"], ["installed-qat-tutorial"])
            artifact.write_bytes(b"different candidate wheel")
            with self.assertRaisesRegex(EvidenceError, "tutorial|wheel"):
                evaluate(registry_path, candidate, document)

    def test_cpu_deployment_requires_exact_image_parents_and_support_bytes(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, old_artifact, evidence_root = release_fixture(Path(raw))
            old_artifact.unlink()
            image = candidate.parent / "tritium-cpu.oci.tar"
            chart = candidate.parent / "tritium-1.1.0-rc.0.tgz"
            image.write_bytes(b"qualified OCI archive")
            chart.write_bytes(b"qualified Helm chart")
            document["artifacts"] = [
                {
                    "id": "oci-cpu", "kind": "oci-image", "path": image.name,
                    "identity": {"sha256": sha256(image), "bytes": image.stat().st_size},
                    "sbom": {}, "provenance": {},
                },
                {
                    "id": "helm-chart", "kind": "helm-chart", "path": chart.name,
                    "identity": {"sha256": sha256(chart), "bytes": chart.stat().st_size},
                    "sbom": {}, "provenance": {},
                },
            ]
            candidate.write_bytes(canonical(document) + b"\n")
            runtime_path = evidence_root / "runtime-cpu.json"
            runtime = oci_runtime_receipt(runtime_path, image)
            security_path = evidence_root / "security-cpu.json"
            security = oci_security_receipt(security_path, image)
            manifest = evidence_root / "bundle.manifest.json"
            build = evidence_root / "oci-build-cpu.json"
            manifest.write_bytes(b"bundle manifest")
            build.write_bytes(b"OCI build receipt")
            deployment_path = evidence_root / "deployment-cpu.json"
            deployment_raw = {
                "chart_artifact": {"artifact_id": "helm-chart"},
                "bundle_manifest_artifact": {
                    "name": manifest.name, "bytes": manifest.stat().st_size,
                    "sha256": sha256(manifest),
                },
                "build_receipt_artifact": {
                    "name": build.name, "bytes": build.stat().st_size,
                    "sha256": sha256(build),
                },
            }
            deployment_id = "sha256:" + hashlib.sha256(canonical(deployment_raw)).hexdigest()
            deployment_raw["receipt_id"] = deployment_id
            deployment_path.write_bytes(canonical(deployment_raw) + b"\n")
            deployment = {
                "receipt_id": deployment_id, "run_id": "deployment-cpu-1",
                "flavor": "cpu",
                "image_artifact": {
                    "name": image.name, "sha256": sha256(image),
                    "bytes": image.stat().st_size,
                },
                "workload": {"startup_receipt": runtime["startup_receipt"]},
            }
            runtime_entry = {
                **entry(runtime_path, runtime, kind="oci-runtime-cpu"),
                "artifact_id": "oci-cpu",
            }
            security_entry = {
                **entry(security_path, security, kind="oci-security-cpu"),
                "artifact_id": "oci-cpu",
            }
            deployment_entry = {
                **entry(deployment_path, deployment_raw, kind="serving-deployment-cpu"),
                "artifact_id": "oci-cpu",
                "parents": [runtime["receipt_id"], security["receipt_id"]],
            }
            registry_path = evidence_root / "registry.json"
            registry(
                registry_path, candidate,
                [deployment_entry, security_entry, runtime_entry],
            )
            loader = mock.Mock(return_value=deployment)
            with mock.patch.dict(evaluate.__globals__, {"load_deployment_receipt": loader}):
                report = evaluate(registry_path, candidate, document, "tritium-test")
            serving = next(row for row in report["rows"] if row["id"] == "serving")
            self.assertIn("serving-deployment-cpu", serving["satisfied_kinds"])
            self.assertIn("serving-deployment-cuda", serving["missing_kinds"])
            self.assertEqual(loader.call_args.kwargs["manifest_path"], manifest.resolve())
            self.assertEqual(loader.call_args.kwargs["build_receipt"], build.resolve())

            manifest.write_bytes(b"tampered bundle manifest")
            with self.assertRaisesRegex(EvidenceError, "support|bytes differ"):
                evaluate(registry_path, candidate, document, "tritium-test")
            manifest.write_bytes(b"bundle manifest")

            manifest.rename(evidence_root / "real-bundle.manifest.json")
            manifest.symlink_to("real-bundle.manifest.json")
            with self.assertRaisesRegex(EvidenceError, "symlink"):
                evaluate(registry_path, candidate, document, "tritium-test")
            manifest.unlink()
            (evidence_root / "real-bundle.manifest.json").rename(manifest)

            mismatched = {**deployment, "workload": {
                "startup_receipt": {**runtime["startup_receipt"], "self_test_digest": "9" * 64}
            }}
            with mock.patch.dict(
                evaluate.__globals__,
                {"load_deployment_receipt": mock.Mock(return_value=mismatched)},
            ), self.assertRaisesRegex(EvidenceError, "startup receipt differs"):
                evaluate(registry_path, candidate, document, "tritium-test")

            deployment_entry["parents"] = [runtime["receipt_id"]]
            registry(
                registry_path, candidate,
                [deployment_entry, security_entry, runtime_entry],
            )
            with mock.patch.dict(evaluate.__globals__, {"load_deployment_receipt": loader}), \
                    self.assertRaisesRegex(EvidenceError, "exact matching runtime"):
                evaluate(registry_path, candidate, document, "tritium-test")

    def test_cpu_oci_security_advances_serving_without_waiving_other_gates(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, old_artifact, evidence_root = release_fixture(Path(raw))
            old_artifact.unlink()
            artifact = candidate.parent / "tritium-cpu.oci.tar"
            artifact.write_bytes(b"qualified OCI archive")
            document["artifacts"] = [{
                "id": "oci-cpu", "kind": "oci-image", "path": artifact.name,
                "identity": {"sha256": sha256(artifact), "bytes": artifact.stat().st_size},
                "sbom": {}, "provenance": {},
            }]
            candidate.write_bytes(canonical(document) + b"\n")
            receipt_path = evidence_root / "security-cpu.json"
            receipt = oci_security_receipt(receipt_path, artifact)
            receipt_entry = entry(receipt_path, receipt, kind="oci-security-cpu")
            receipt_entry["artifact_id"] = "oci-cpu"
            registry_path = evidence_root / "registry.json"
            registry(registry_path, candidate, [receipt_entry])
            report = evaluate(registry_path, candidate, document)
            serving = next(row for row in report["rows"] if row["id"] == "serving")
            self.assertEqual(serving["satisfied_kinds"], ["oci-security-cpu"])
            self.assertEqual(
                serving["missing_kinds"],
                ["oci-runtime-cpu", "oci-runtime-cuda", "oci-security-cuda",
                 "serving-deployment-cpu", "serving-deployment-cuda"],
            )
            receipt_entry["kind"] = "oci-security-cuda"
            registry(registry_path, candidate, [receipt_entry])
            with self.assertRaisesRegex(EvidenceError, "flavor differs"):
                evaluate(registry_path, candidate, document)

    def test_cpu_oci_runtime_advances_serving_without_overclaiming_cuda_or_deployment(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, old_artifact, evidence_root = release_fixture(Path(raw))
            old_artifact.unlink()
            artifact = candidate.parent / "tritium-cpu.oci.tar"
            artifact.write_bytes(b"qualified OCI archive")
            document["artifacts"] = [{
                "id": "oci-cpu", "kind": "oci-image", "path": artifact.name,
                "identity": {"sha256": sha256(artifact), "bytes": artifact.stat().st_size},
                "sbom": {}, "provenance": {},
            }]
            candidate.write_bytes(canonical(document) + b"\n")
            receipt_path = evidence_root / "oci-cpu.json"
            receipt = oci_runtime_receipt(receipt_path, artifact)
            receipt_entry = entry(receipt_path, receipt, kind="oci-runtime-cpu")
            receipt_entry["artifact_id"] = "oci-cpu"
            registry_path = evidence_root / "registry.json"
            registry(registry_path, candidate, [receipt_entry])
            report = evaluate(registry_path, candidate, document)
            serving = next(row for row in report["rows"] if row["id"] == "serving")
            self.assertEqual(serving["satisfied_kinds"], ["oci-runtime-cpu"])
            self.assertEqual(
                serving["missing_kinds"],
                ["oci-runtime-cuda", "oci-security-cpu", "oci-security-cuda",
                 "serving-deployment-cpu", "serving-deployment-cuda"],
            )
            receipt_entry["kind"] = "oci-runtime-cuda"
            registry(registry_path, candidate, [receipt_entry])
            with self.assertRaisesRegex(EvidenceError, "flavor differs"):
                evaluate(registry_path, candidate, document)

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

    def test_npm_receipt_binds_candidate_archive_and_advances_package_gate(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, old_artifact, evidence_root = release_fixture(Path(raw))
            old_artifact.unlink()
            archive = candidate.parent / "tritium-ai-web-1.1.0-rc.0.tgz"
            archive.write_bytes(b"qualified npm bytes")
            archive_sha = sha256(archive)
            document["artifacts"] = [{
                "id": "tritium-web-node22", "kind": "npm-archive",
                "path": archive.name,
                "identity": {"sha256": archive_sha, "bytes": archive.stat().st_size},
                "sbom": {}, "provenance": {},
            }]
            candidate.write_bytes(canonical(document) + b"\n")
            receipt = {
                "schema": NPM_MODULE["SCHEMA"], "release": "1.1.0-rc.0",
                "source_revision": "a" * 40, "run_id": "npm-run-1",
                "started_at_utc": "2026-07-21T12:00:00Z", "duration_ms": 100.0,
                "machine": {
                    "machine_id": "sha256:" + "b" * 64,
                    "system": "linux", "architecture": "x64",
                },
                "toolchain": {"node": "v22.18.0", "npm": "11.5.2"},
                "artifact": {
                    "kind": "npm-archive", "name": archive.name,
                    "package": "@tritium-ai/web@1.1.0-rc.0",
                    "bytes": archive.stat().st_size, "sha256": archive_sha,
                    "integrity": "sha512-" + base64.b64encode(
                        hashlib.sha512(archive.read_bytes()).digest()
                    ).decode("ascii"),
                },
                "evidence": {
                    "source_dirty": False, "entry_count": 13,
                    "source_free": True, "installed_offline": True,
                    "strict_typescript": True,
                    "wasm_build_id": "tritium-wasm@1.1.0-rc.0+source-git:" + "a" * 40,
                    "wasm_guest_digest": "c" * 64,
                },
                "result": "pass",
            }
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                canonical(receipt)
            ).hexdigest()
            receipt_path = evidence_root / "npm.json"
            receipt_path.write_bytes(canonical(receipt) + b"\n")
            registry_path = evidence_root / "registry.json"
            registry(
                registry_path, candidate,
                [{
                    **entry(receipt_path, receipt, kind="npm-archive"),
                    "artifact_id": "tritium-web-node22",
                }],
            )
            report = evaluate(registry_path, candidate, document)
            packages = next(row for row in report["rows"] if row["id"] == "packages")
            self.assertEqual(packages["satisfied_kinds"], ["npm-archive"])
            archive.write_bytes(b"tampered")
            with self.assertRaisesRegex(EvidenceError, "npm-archive validation"):
                evaluate(registry_path, candidate, document)

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

            second["run_id"] = "run-18"
            unsigned = dict(second)
            unsigned.pop("receipt_id")
            second["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
            second_path.write_bytes(canonical(second) + b"\n")
            registry(
                registry_path,
                candidate,
                [entry(first_path, first), entry(second_path, second)],
            )
            with self.assertRaisesRegex(EvidenceError, "duplicate evidence kind"):
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

    def test_release_status_never_calls_unsigned_evidence_local_rc_ready(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, document, _, evidence_root = release_fixture(Path(raw))
            registry_path = evidence_root / "registry.json"
            registry(registry_path, candidate, [])
            report = {
                "schema": "tritium.release-gate-report.v1",
                "release": document["release"],
                "source_revision": document["source_revision"],
                "candidate_manifest_sha256": sha256(candidate),
                "evidence_registry_sha256": sha256(registry_path),
                "ready": True,
                "rows": [],
                "external_activation": "EXTERNAL_AUTH_REQUIRED",
            }
            globals_ = status_main.__globals__
            replacements = {
                "validate": lambda _candidate, _tool: document,
                "_git_gate": lambda _root, _revision: None,
                "_version_gate": lambda _root, _release: None,
            }
            original = {name: globals_[name] for name in replacements}
            globals_.update(replacements)
            stdout = io.StringIO()
            try:
                with mock.patch.object(
                    sys, "argv",
                    ["release-status", "--candidate", str(candidate),
                     "--registry", str(registry_path)],
                ), mock.patch("sys.stdout", stdout), mock.patch.object(
                    globals_["runpy"], "run_path",
                    return_value={"evaluate": lambda *_args: report,
                                  "render": lambda _report: "ALL GATES PASS"},
                ):
                    result = status_main()
            finally:
                globals_.update(original)
            self.assertEqual(result, 2)
            self.assertIn("LOCAL_RC_EVIDENCE_READY_UNSIGNED", stdout.getvalue())
            self.assertNotIn("LOCAL_RC_READY", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
