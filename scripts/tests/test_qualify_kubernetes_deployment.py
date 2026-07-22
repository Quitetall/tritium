from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import runpy
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-kubernetes-deployment.py")
DeploymentError = MODULE["DeploymentError"]
canonical = MODULE["canonical"]
pod_snapshot = MODULE["pod_snapshot"]
container_identity = MODULE["container_identity"]
validate_deployment = MODULE["validate_deployment"]
validate_receipt = MODULE["validate_receipt"]
metrics_snapshot = MODULE["metrics_snapshot"]
validate_helm_history = MODULE["validate_helm_history"]
scale_snapshot = MODULE["scale_snapshot"]
prometheus_url = MODULE["prometheus_url"]
deployment_update_identity = MODULE["deployment_update_identity"]
validate_scale_contract = MODULE["validate_scale_contract"]
WATCHDOG_FAULT_COMMAND = MODULE["WATCHDOG_FAULT_COMMAND"]
watchdog_contract = MODULE["watchdog_contract"]
deployment_service_port = MODULE["deployment_service_port"]
validate_service_port = MODULE["validate_service_port"]
artifact_volume_contract = MODULE["artifact_volume_contract"]
memory_limit_contract = MODULE["memory_limit_contract"]
observed_oom_failure = MODULE["observed_oom_failure"]
oom_replica_set_lineage = MODULE["oom_replica_set_lineage"]
replace_pre_oom_pods = MODULE["replace_pre_oom_pods"]
pending_artifact_volume_failure = MODULE["pending_artifact_volume_failure"]
pvc_identity = MODULE["pvc_identity"]
qualify_artifact_volume_loss = MODULE["qualify_artifact_volume_loss"]
qualify_memory_oom = MODULE["qualify_memory_oom"]
prove_absent = MODULE["prove_absent"]
resource_quantity = MODULE["resource_quantity"]
resource_usage_sample = MODULE["resource_usage_sample"]
request_evidence = MODULE["request_evidence"]
qualify_metrics_scrape_flood = MODULE["qualify_metrics_scrape_flood"]
validate_metrics_flood = MODULE["validate_metrics_flood"]
qualify_slow_collector = MODULE["qualify_slow_collector"]
validate_slow_collector = MODULE["validate_slow_collector"]
auth_secret_contract = MODULE["auth_secret_contract"]
missing_secret_failure = MODULE["missing_secret_failure"]
restore_missing_secret_refs = MODULE["restore_missing_secret_refs"]
startup_argument_contract = MODULE["startup_argument_contract"]
startup_process_failure = MODULE["startup_process_failure"]
startup_error_line = MODULE["startup_error_line"]
restore_startup_argument = MODULE["restore_startup_argument"]
startup_log_command = MODULE["startup_log_command"]
prometheus_target_absence = MODULE["prometheus_target_absence"]
qualify_collector_outage = MODULE["qualify_collector_outage"]
collect_rollback_runtime = MODULE["collect_rollback_runtime"]
validate_rollback_runtime = MODULE["validate_rollback_runtime"]


def startup(flavor: str = "cpu") -> dict:
    return {
        "schema_version": 1,
        "artifact_kind": "qwen3.6-language-mtp-salt-v2-hf-bundle",
        "server_source_revision": "a" * 40,
        "server_build_id": "tritium-serve:1.1.0-rc.0:" + "a" * 40,
        "model_source_revision": "b" * 40,
        "manifest_package_id": "c" * 64,
        "salt_package_id": "d" * 64,
        "preserved_package_id": "e" * 64,
        "config_package_id": "f" * 64,
        "profile": "compact-v1", "codec": "b3",
        "backend_policy": flavor, "effective_backend": flavor,
        "physical_device_id": (
            "cpu" if flavor == "cpu"
            else "cuda:0:GPU-12345678-1234-1234-1234-123456789abc"
        ),
        "loaded_bundle_bytes": 100, "resident_bytes": 80,
        "self_test_digest": "1" * 64,
    }


def startup_argument_receipt(scenario: str, flavor: str,
                             startup_receipt: dict) -> dict:
    if scenario == "invalid_config":
        flag, source, fault, index = "--max-new", "256", "0", 13
        versions = ("30", "31", "32")
        error_line = 'Error: "all request and prompt limits must be >= 1"'
        elapsed = (3000.0, 8100.0)
    else:
        flag, source, fault, index = (
            "--backend", flavor, "tritium-unavailable", 5
        )
        versions = ("40", "41", "42")
        backends = "cpu" if flavor == "cpu" else "cpu, cuda"
        error_line = (
            'Error: "backend `tritium-unavailable` is not in the registry '
            f'(linked backends: {backends}); for cuda, build with `--features cuda`"'
        )
        elapsed = (4000.0, 9100.0)
    path = f"/spec/template/spec/containers/0/args/{index}"
    args = [
        "--bundle", "/artifacts/bundle", "--profile", "compact-v1",
        "--backend", flavor, "--host", "0.0.0.0", "--port", "8080",
        "--model-id", "tritium", "--max-new", "256", "--max-messages", "128",
        "--max-prompt-bytes", "1048576", "--max-prompt-tokens", "131072",
        "--max-completion-tokens", "4096", "--max-total-tokens", "131072",
        "--rate-limit-rpm", "120", "--rate-limit-burst", "8",
    ]
    binding = {
        "deployment_uid": "deployment-uid", "resource_version": versions[0],
        "container": "tritium", "container_index": 0, "flag": flag,
        "flag_index": index - 1,
        "flag_path": f"/spec/template/spec/containers/0/args/{index - 1}",
        "value_index": index, "source_value": source, "args": args, "path": path,
    }
    fault_patch = [
        {"op": "test", "path": "/metadata/uid", "value": "deployment-uid"},
        {"op": "test", "path": "/metadata/resourceVersion", "value": versions[0]},
        {"op": "test", "path": binding["flag_path"], "value": flag},
        {"op": "test", "path": path, "value": source},
        {"op": "replace", "path": path, "value": fault},
    ]
    restore_patch = [
        {"op": "test", "path": "/metadata/uid", "value": "deployment-uid"},
        {"op": "test", "path": binding["flag_path"], "value": flag},
        {"op": "test", "path": path, "value": fault},
        {"op": "replace", "path": path, "value": source},
    ]
    return {
        "scenario": scenario, "deployment_uid": "deployment-uid",
        "baseline_resource_version": versions[0],
        "fault_resource_version": versions[1],
        "restored_resource_version": versions[2],
        "binding": binding, "fault_value": fault,
        "fault_patch_sha256": hashlib.sha256(
            canonical(fault_patch).decode().strip().encode()
        ).hexdigest(),
        "restore_patch_sha256": hashlib.sha256(
            canonical(restore_patch).decode().strip().encode()
        ).hexdigest(),
        "observation_budget_ms": 120000, "duration_ms": 5000.0,
        "started_elapsed_ms": elapsed[0], "completed_elapsed_ms": elapsed[1],
        "failure": {
            "pod_name": f"pod-{scenario}", "pod_uid": f"{scenario}-pod-uid",
            "container": "tritium", "exit_code": 1, "reason": "Error",
            "restart_count": 1, "termination_source": "last_state",
            "replica_set_name": f"qualification-tritium-{scenario}",
            "replica_set_uid": f"{scenario}-rs-uid",
            "replica_set_owner": {
                "kind": "Deployment", "name": "qualification-tritium",
                "uid": "deployment-uid",
            },
            "error_line": error_line, "normalized_log_sha256": "d" * 64,
        },
        "recovered": {"pods": [{
            "name": f"pod-{scenario}-recovered",
            "uid": f"{scenario}-recovered-uid", "node": "node-1", "restarts": 0,
        }]},
        "startup_receipt": dict(startup_receipt),
        "generation_response_sha256": "e" * 64,
        "request": request_evidence(
            "tritium", "Hello", temperature=0, max_tokens=1
        ),
        "metrics": {"sha256": "f" * 64, "values": {
            "tritium_chat_requests_total": 1.0,
            "tritium_tokens_out_total": 1.0,
            "tritium_worker_alive": 1.0,
            "tritium_backend_faults_total": 0.0,
            "tritium_backend_faulted": 0.0,
        }},
        "cleanup": {"status": "restored"},
        "transitions": [
            {"state": state, "elapsed_ms": float(index * 1000),
             "observed_at_utc": f"2026-07-21T12:00:3{index}+00:00"}
            for index, state in enumerate(MODULE["STARTUP_ARGUMENT_TRANSITIONS"])
        ],
    }


def receipt(chart: Path, image_archive: Path, candidate: Path,
            flavor: str = "cpu") -> dict:
    startup_receipt = startup(flavor)
    recovered_name = "pod-artifact-recovered" if flavor == "cuda" else "pod-new"

    def resource_sample(name: str | None, timestamp: str, seed: str) -> dict:
        return {
            "sample_sha256": seed * 64, "sampled_at_utc": timestamp,
            "pod_names": ([] if name is None else [name]),
            "pod_count": 0 if name is None else 1,
            "container_count": 0 if name is None else 2,
            "cpu_nanocores": 0 if name is None else 10_000_000,
            "memory_bytes": 0 if name is None else 1_000_000_000,
        }
    manifest = candidate.parent / "tritium.json"
    build_receipt = candidate.parent / "build-receipt.json"
    value = {
        "schema": MODULE["SCHEMA"], "release": "1.1.0-rc.0",
        "source_revision": "a" * 40, "run_id": f"kubernetes-{flavor}-1",
        "flavor": flavor, "profile": "compact-v1",
        "started_at_utc": "2026-07-21T12:00:00+00:00", "duration_ms": 10_000.0,
        "chart_artifact": {"kind": "helm-chart", "name": chart.name,
                           "bytes": chart.stat().st_size,
                           "sha256": hashlib.sha256(chart.read_bytes()).hexdigest(),
                           "artifact_id": "helm-chart",
                           "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()},
        "image_artifact": {"kind": "oci-image", "name": image_archive.name,
                           "bytes": image_archive.stat().st_size,
                           "sha256": hashlib.sha256(image_archive.read_bytes()).hexdigest()},
        "bundle_manifest_artifact": {
            "kind": "bundle-manifest", "name": manifest.name,
            "bytes": manifest.stat().st_size,
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
        "build_receipt_artifact": {
            "kind": "oci-build-receipt", "name": build_receipt.name,
            "bytes": build_receipt.stat().st_size,
            "sha256": hashlib.sha256(build_receipt.read_bytes()).hexdigest(),
        },
        "image": "registry.example/tritium@sha256:" + "2" * 64,
        "manifest": {"schema": "tritium.file-identity.v1", "bytes": 42,
                     "sha256": "3" * 64, "blake3": "c" * 64},
        "cluster": {"context": "kind-tritium", "namespace": "tritium-test",
                    "namespace_uid": "namespace-uid", "server_git_version": "v1.34.0",
                    "server_platform": "linux/amd64", "nodes": [{
                        "name": "node-1", "uid": "node-uid", "provider_id": "kind://node-1",
                        "kernel_version": "6.12.0", "os_image": "Tritium Test Linux",
                        "architecture": "amd64", "container_runtime": "containerd://2.0",
                    }], "cuda_node": None},
        "tools": {"kubectl_sha256": "4" * 64, "helm_sha256": "5" * 64,
                  "helm_version": "v3.18.0"},
        "workload": {
            "release_name": "qualification", "deployment_uid": "deployment-uid",
            "qualification_lock_uid": "lock-uid", "source_pvc": "tritium-artifact",
            "model_id": "tritium", "service_port": 8080,
            "source_pvc_identity": {
                "name": "tritium-artifact", "uid": "pvc-uid",
                "volume_name": "pv-artifact", "storage_class": "standard",
                "access_modes": ["ReadWriteOnce"], "capacity": "64Gi",
                "phase": "Bound",
            },
            "initial": {"pods": [{"name": "pod-old", "uid": "pod-old-uid",
                                    "node": "node-1", "restarts": 1}]},
            "restarted": {"pods": [{"name": "pod-new", "uid": "pod-new-uid",
                                      "node": "node-1", "restarts": 0}]},
            "updated": {"pods": [{"name": "pod-updated", "uid": "pod-updated-uid",
                                    "node": "node-1", "restarts": 0}]},
            "startup_receipt": startup_receipt,
            "watchdog_replacement": {
                "pod_uid": "pod-old-uid",
                "container_id_before": "containerd://old",
                "container_id_after": "containerd://new",
                "restart_count_before": 0,
                "restart_count_after": 1,
                "last_exit_code": 137,
                "fault_command_sha256": hashlib.sha256(
                    WATCHDOG_FAULT_COMMAND.encode()
                ).hexdigest(),
                "replacement_ms": 32000.0,
                "watchdog": {
                    "startup_period_seconds": 5, "startup_timeout_seconds": 2,
                    "startup_failure_threshold": 60,
                    "startup_probe_window_ms": 300000,
                    "startup_wait_period_seconds": 5,
                    "startup_gate": MODULE["health_gate_descriptor"](
                        8080, "before_failure_monitoring"
                    ),
                    "startup_probe_handlers": MODULE["startup_probe_handlers"](8080),
                    "period_seconds": 10, "timeout_seconds": 2,
                    "failure_threshold": 3, "escalation_seconds": 2,
                    "scheduling_allowance_ms": 60000, "budget_ms": 98000,
                    "monitor_gate": MODULE["health_gate_descriptor"](
                        8080, "steady_state_monitoring"
                    ),
                },
                "startup_receipt": dict(startup_receipt),
                "generation_response_sha256": "a" * 64,
                "metrics": {"sha256": "b" * 64, "values": {
                    "tritium_chat_requests_total": 1.0,
                    "tritium_tokens_out_total": 1.0,
                    "tritium_worker_alive": 1.0,
                    "tritium_backend_faults_total": 0.0,
                    "tritium_backend_faulted": 0.0,
                }},
            },
            "artifact_volume_loss": {
                "source_claim": "tritium-artifact",
                "missing_claim": "tritium-missing-123456789abc",
                "volume_index": 0,
                "absence": {
                    "status": "NotFound",
                    "output_sha256": hashlib.sha256(b"").hexdigest(),
                },
                "fault_patch_sha256": hashlib.sha256(canonical([{
                    "op": "replace",
                    "path": "/spec/template/spec/volumes/0/persistentVolumeClaim/claimName",
                    "value": "tritium-missing-123456789abc",
                }]).decode().strip().encode()).hexdigest(),
                "observation_budget_ms": 120000,
                "observation_ms": 5000.0,
                "pending": {
                    "pod_name": "pod-missing", "pod_uid": "pod-missing-uid",
                    "reason": "Unschedulable", "message_sha256": "d" * 64,
                },
                "recovered": {"pods": [{
                    "name": recovered_name,
                    "uid": ("pod-artifact-recovered-uid" if flavor == "cuda"
                            else "pod-new-uid"),
                    "node": "node-1", "restarts": 0,
                }]},
                "cleanup": {"status": "restored", "source_claim": "tritium-artifact"},
                "startup_receipt": dict(startup_receipt),
                "generation_response_sha256": "e" * 64,
                "request": request_evidence(
                    "tritium", "Hello", temperature=0, max_tokens=1
                ),
                "transitions": [
                    {"state": state, "elapsed_ms": float(index),
                     "observed_at_utc": f"2026-07-21T12:00:0{index}+00:00"}
                    for index, state in enumerate(MODULE["ARTIFACT_VOLUME_TRANSITIONS"])
                ],
                "metrics": {"sha256": "f" * 64, "values": {
                    "tritium_chat_requests_total": 1.0,
                    "tritium_tokens_out_total": 1.0,
                    "tritium_worker_alive": 1.0,
                    "tritium_backend_faults_total": 0.0,
                    "tritium_backend_faulted": 0.0,
                }},
                "resources": {
                    "baseline": resource_sample(
                        "pod-new", "2026-07-21T12:00:01+00:00", "1"
                    ),
                    "failure": resource_sample(
                        None, "2026-07-21T12:00:02+00:00", "2"
                    ),
                    "recovered": resource_sample(
                        recovered_name, "2026-07-21T12:00:03+00:00", "3"
                    ),
                    "high_water": {
                        "cpu_nanocores": 10_000_000,
                        "memory_bytes": 1_000_000_000,
                    },
                },
            },
            "memory_oom_recovery": {
                "container_index": 0,
                "source_limit": "32Gi", "source_limit_bytes": 32 * 1024 ** 3,
                "fault_limit": "16Mi", "fault_limit_bytes": 16 * 1024 ** 2,
                "fault_patch_sha256": hashlib.sha256(canonical([{
                    "op": "replace",
                    "path": "/spec/template/spec/containers/0/resources/limits/memory",
                    "value": "16Mi",
                }]).decode().strip().encode()).hexdigest(),
                "observation_budget_ms": 120000, "observation_ms": 6000.0,
                "terminated": {
                    "pod_name": "pod-oom", "pod_uid": "pod-oom-uid",
                    "node": "node-1", "restart_count": 1,
                    "reason": "OOMKilled", "last_exit_code": 137,
                    "memory_limit": "16Mi", "memory_limit_bytes": 16 * 1024 ** 2,
                    "replica_set_name": "qualification-tritium-fault",
                    "replica_set_uid": "oom-rs-uid",
                    "template_hash": "fault-hash",
                },
                "replica_set": {
                    "name": "qualification-tritium-fault", "uid": "oom-rs-uid",
                    "deployment_name": "qualification-tritium",
                    "deployment_uid": "deployment-uid", "template_hash": "fault-hash",
                    "memory_limit": "16Mi", "memory_limit_bytes": 16 * 1024 ** 2,
                },
                "cleanup": {"status": "restored", "memory_limit": "32Gi"},
                "pre_fault_cleanup": {
                    "mode": ("deleted_after_restore" if flavor == "cpu"
                             else "already_absent"),
                    "pod_names": [recovered_name],
                    "pod_uids": [("pod-artifact-recovered-uid" if flavor == "cuda"
                                  else "pod-new-uid")],
                },
                "transitions": [
                    {"state": state, "elapsed_ms": float(index),
                     "observed_at_utc": f"2026-07-21T12:01:0{index}+00:00"}
                    for index, state in enumerate(MODULE["OOM_TRANSITIONS"])
                ],
                "recovered": {"pods": [{
                    "name": "pod-oom-recovered", "uid": "pod-oom-recovered-uid",
                    "node": "node-1", "restarts": 0,
                }]},
                "startup_receipt": dict(startup_receipt),
                "generation_response_sha256": "1" * 64,
                "request": request_evidence(
                    "tritium", "Hello", temperature=0, max_tokens=1
                ),
                "metrics": {"sha256": "2" * 64, "values": {
                    "tritium_chat_requests_total": 1.0,
                    "tritium_tokens_out_total": 1.0,
                    "tritium_worker_alive": 1.0,
                    "tritium_backend_faults_total": 0.0,
                    "tritium_backend_faulted": 0.0,
                }},
                "resources": {
                    "baseline": resource_sample(
                        recovered_name, "2026-07-21T12:01:01+00:00", "4"
                    ),
                    "failure": resource_sample(
                        None, "2026-07-21T12:01:02+00:00", "5"
                    ),
                    "recovered": resource_sample(
                        "pod-oom-recovered", "2026-07-21T12:01:03+00:00", "6"
                    ),
                    "high_water": {
                        "cpu_nanocores": 10_000_000,
                        "memory_bytes": 1_000_000_000,
                    },
                },
            },
            "missing_secret_startup": {
                "deployment_uid": "deployment-uid",
                "baseline_resource_version": "20",
                "fault_resource_version": "21",
                "restored_resource_version": "22",
                "secret_name": "tritium-auth", "secret_key": "token",
                "bindings": [
                    {"container": "tritium", "container_index": 0, "env_index": 0,
                     "secret_name": "tritium-auth", "secret_key": "token",
                     "path": ("/spec/template/spec/containers/0/env/0/"
                              "valueFrom/secretKeyRef/name")},
                    {"container": "authenticated-probe", "container_index": 1,
                     "env_index": 0, "secret_name": "tritium-auth",
                     "secret_key": "token",
                     "path": ("/spec/template/spec/containers/1/env/0/"
                              "valueFrom/secretKeyRef/name")},
                ],
                "missing_secret": "tritium-missing-auth-0123456789ab",
                "absence": {"status": "NotFound",
                            "output_sha256": hashlib.sha256(b"").hexdigest()},
                "fault_patch_sha256": hashlib.sha256(canonical([
                    {"op": "test", "path": "/metadata/uid",
                     "value": "deployment-uid"},
                    {"op": "test", "path": "/metadata/resourceVersion",
                     "value": "20"},
                    {"op": "test", "path": ("/spec/template/spec/containers/0/"
                     "env/0/valueFrom/secretKeyRef/name"), "value": "tritium-auth"},
                    {"op": "replace", "path": ("/spec/template/spec/containers/0/"
                     "env/0/valueFrom/secretKeyRef/name"),
                     "value": "tritium-missing-auth-0123456789ab"},
                    {"op": "test", "path": ("/spec/template/spec/containers/1/"
                     "env/0/valueFrom/secretKeyRef/name"), "value": "tritium-auth"},
                    {"op": "replace", "path": ("/spec/template/spec/containers/1/"
                     "env/0/valueFrom/secretKeyRef/name"),
                     "value": "tritium-missing-auth-0123456789ab"},
                ]).decode().strip().encode()).hexdigest(),
                "restore_patch_sha256": hashlib.sha256(canonical([
                    {"op": "test", "path": "/metadata/uid",
                     "value": "deployment-uid"},
                    {"op": "test", "path": ("/spec/template/spec/containers/0/"
                     "env/0/valueFrom/secretKeyRef/name"),
                     "value": "tritium-missing-auth-0123456789ab"},
                    {"op": "replace", "path": ("/spec/template/spec/containers/0/"
                     "env/0/valueFrom/secretKeyRef/name"), "value": "tritium-auth"},
                    {"op": "test", "path": ("/spec/template/spec/containers/1/"
                     "env/0/valueFrom/secretKeyRef/name"),
                     "value": "tritium-missing-auth-0123456789ab"},
                    {"op": "replace", "path": ("/spec/template/spec/containers/1/"
                     "env/0/valueFrom/secretKeyRef/name"), "value": "tritium-auth"},
                ]).decode().strip().encode()).hexdigest(),
                "observation_budget_ms": 120000, "duration_ms": 5000.0,
                "started_elapsed_ms": 2000.0, "completed_elapsed_ms": 7100.0,
                "failure": {
                    "pod_name": "pod-missing-secret", "pod_uid": "missing-secret-uid",
                    "container": "tritium", "reason": "CreateContainerConfigError",
                    "message_sha256": hashlib.sha256(
                        b'secret "tritium-missing-auth-0123456789ab" not found'
                    ).hexdigest(),
                    "replica_set_name": "qualification-tritium-missing",
                    "replica_set_uid": "missing-secret-rs-uid",
                    "replica_set_owner": {
                        "kind": "Deployment", "name": "qualification-tritium",
                        "uid": "deployment-uid",
                    },
                },
                "recovered": {"pods": [{
                    "name": "pod-secret-recovered", "uid": "secret-recovered-uid",
                    "node": "node-1", "restarts": 0,
                }]},
                "startup_receipt": dict(startup_receipt),
                "generation_response_sha256": "b" * 64,
                "request": request_evidence(
                    "tritium", "Hello", temperature=0, max_tokens=1
                ),
                "metrics": {"sha256": "c" * 64, "values": {
                    "tritium_chat_requests_total": 1.0,
                    "tritium_tokens_out_total": 1.0,
                    "tritium_worker_alive": 1.0,
                    "tritium_backend_faults_total": 0.0,
                    "tritium_backend_faulted": 0.0,
                }},
                "cleanup": {"status": "restored"},
                "transitions": [
                    {"state": state, "elapsed_ms": float(index * 1000),
                     "observed_at_utc": f"2026-07-21T12:00:2{index}+00:00"}
                    for index, state in enumerate(MODULE["MISSING_SECRET_TRANSITIONS"])
                ],
            },
            "invalid_config_startup": startup_argument_receipt(
                "invalid_config", flavor, startup_receipt
            ),
            "unavailable_backend_startup": startup_argument_receipt(
                "unavailable_backend", flavor, startup_receipt
            ),
            "restart_startup_receipt": dict(startup_receipt),
            "update_startup_receipt": dict(startup_receipt),
            "update_strategy": "Recreate" if flavor == "cuda" else "RollingUpdate",
            "update_config": {
                "before": {"generation": 1, "rate_limit_burst": 8},
                "after": {"generation": 2, "rate_limit_burst": 9},
            },
            "rollback_startup_receipt": dict(startup_receipt),
            "rollback_runtime": {
                "image": "registry.example/tritium@sha256:" + "2" * 64,
                "image_digest": "sha256:" + "2" * 64,
                "deployment_name": "qualification-tritium",
                "deployment_uid": "deployment-uid",
                "elapsed_ms": 75.0,
                "observed_at_utc": "2026-07-21T12:03:00+00:00",
                "pods": [{
                    "pod_name": "qualification-tritium-rollback",
                    "pod_uid": "rollback-pod-uid",
                    "replica_set_name": "qualification-tritium-abc123",
                    "replica_set_uid": "rollback-rs-uid",
                    "replica_set_owner": {
                        "kind": "Deployment", "name": "qualification-tritium",
                        "uid": "deployment-uid",
                    },
                    "image": "registry.example/tritium@sha256:" + "2" * 64,
                    "image_id": "containerd://sha256:" + "2" * 64,
                }],
            },
            "metrics": {"sha256": "6" * 64, "values": {
                "tritium_chat_requests_total": 1.0,
                "tritium_tokens_out_total": 1.0,
                "tritium_worker_alive": 1.0,
                "tritium_backend_faults_total": 0.0,
                "tritium_backend_faulted": 0.0,
            }},
            "metrics_scrape_flood": {
                "endpoint": "/metrics", "authentication": "bearer",
                "requests": 64, "concurrency": 8, "peak_client_tasks": 8,
                "budget_ms": 60000, "duration_ms": 1000.0,
                "max_response_bytes": 1024 * 1024,
                "response_bytes": [1024] * 64,
                "response_sha256s": [f"{index:064x}" for index in range(64)],
                "response_latency_ms": [500.0] * 64,
                "max_scrape_latency_ms": 500.0, "generation_latency_ms": 750.0,
                "generation_response_sha256": "3" * 64,
                "request": request_evidence(
                    "tritium", "Hello", temperature=0, max_tokens=1
                ),
                "baseline_metrics": {"sha256": "5" * 64, "values": {
                    "tritium_chat_requests_total": 1.0,
                    "tritium_tokens_out_total": 1.0,
                    "tritium_worker_alive": 1.0,
                    "tritium_backend_faults_total": 0.0,
                    "tritium_backend_faulted": 0.0,
                }},
                "metrics": {"sha256": "4" * 64, "values": {
                    "tritium_chat_requests_total": 2.0,
                    "tritium_tokens_out_total": 2.0,
                    "tritium_worker_alive": 1.0,
                    "tritium_backend_faults_total": 0.0,
                    "tritium_backend_faulted": 0.0,
                }},
            },
            "slow_collector": ({
                "endpoint": "/metrics", "authentication": "bearer",
                "connections": 8, "hold_ms": 3000, "budget_ms": 30000,
                "duration_ms": 4000.0, "generation_latency_ms": 500.0,
                "started_elapsed_ms": 1000.0, "completed_elapsed_ms": 5100.0,
                "generation_response_sha256": "9" * 64,
                "request": request_evidence(
                    "tritium", "Hello", temperature=0, max_tokens=1
                ),
                "response_bytes": [1024] * 8,
                "response_sha256s": [f"{index + 20:064x}" for index in range(8)],
                "transitions": [
                    {"state": state, "elapsed_ms": elapsed}
                    for state, elapsed in zip((
                        "partial_headers_open", "generation_complete",
                        "hold_complete", "scrapes_complete",
                    ), (100.0, 600.0, 3100.0, 3900.0), strict=True)
                ],
            } if flavor == "cpu" else None),
            "collector_outage": ({
                "service_monitor_name": "qualification-tritium",
                "service_monitor_uid": "monitor-uid", "service_uid": "service-uid",
                "baseline_resource_version": "10", "fault_resource_version": "11",
                "restored_resource_version": "12",
                "fault_label": "tritium-telemetry-fault",
                "fault_value": "0123456789abcdef",
                "fault_patch_sha256": hashlib.sha256(canonical([
                    {"op": "test", "path": "/metadata/uid",
                     "value": "monitor-uid"},
                    {"op": "test", "path": "/metadata/resourceVersion",
                     "value": "10"},
                    {"op": "add",
                     "path": "/spec/selector/matchLabels/tritium-telemetry-fault",
                     "value": "0123456789abcdef"},
                ]).decode().strip().encode()).hexdigest(),
                "restore_patch_sha256": hashlib.sha256(canonical([
                    {"op": "test", "path": "/metadata/uid",
                     "value": "monitor-uid"},
                    {"op": "test",
                     "path": "/spec/selector/matchLabels/tritium-telemetry-fault",
                     "value": "0123456789abcdef"},
                    {"op": "remove",
                     "path": "/spec/selector/matchLabels/tritium-telemetry-fault"},
                ]).decode().strip().encode()).hexdigest(),
                "observation_budget_ms": 120000, "duration_ms": 50.0,
                "baseline_target": {
                    "scrape_url": "http://10.0.0.2:8080/metrics",
                    "last_scrape_utc": "2026-07-21T12:01:59+00:00",
                },
                "absence": {
                    "active_matches": 0, "response_sha256": "6" * 64,
                    "observed_at_utc": "2026-07-21T12:02:02+00:00",
                },
                "startup_receipt": dict(startup_receipt),
                "generation_response_sha256": "7" * 64,
                "request": request_evidence(
                    "tritium", "Hello", temperature=0, max_tokens=1
                ),
                "metrics": {"sha256": "8" * 64, "values": {
                    "tritium_chat_requests_total": 2.0,
                    "tritium_tokens_out_total": 2.0,
                    "tritium_worker_alive": 1.0,
                    "tritium_backend_faults_total": 0.0,
                    "tritium_backend_faulted": 0.0,
                }},
                "recovered_target": {
                    "scrape_url": "http://10.0.0.2:8080/metrics",
                    "last_scrape_utc": "2026-07-21T12:02:05+00:00",
                },
                "cleanup": {"status": "restored", "selector": {
                    "app.kubernetes.io/name": "tritium",
                    "app.kubernetes.io/instance": "qualification",
                }},
                "transitions": [
                    {"state": state, "elapsed_ms": float(index),
                     "observed_at_utc": f"2026-07-21T12:02:0{index}+00:00"}
                    for index, state in enumerate(MODULE["COLLECTOR_OUTAGE_TRANSITIONS"])
                ],
            } if flavor == "cpu" else None),
            "restart_recovery": {
                "generation_response_sha256": "7" * 64,
                "metrics": {"sha256": "8" * 64, "values": {
                    "tritium_chat_requests_total": 1.0,
                    "tritium_tokens_out_total": 1.0,
                    "tritium_worker_alive": 1.0,
                    "tritium_backend_faults_total": 0.0,
                    "tritium_backend_faulted": 0.0,
                }},
            },
            "scale": ({
                "scaled_object_uid": "scaled-object-uid", "hpa_uid": "hpa-uid",
                "external_metric": "s0-prometheus-tritium_queue_pressure",
                "scaled_replicas": 2, "settled_replicas": 1,
                "load_requests": 8, "load_concurrency": 8, "max_tokens": 256,
                "prometheus_server": "http://prometheus.monitoring.svc:9090",
                "prometheus_service": "prometheus", "prometheus_port": 9090,
                "monitoring_namespace": "monitoring",
                "service_monitor_label": "release=kube-prometheus-stack",
                "query": ('max(tritium_queue_depth{namespace="tritium-test",'
                          'service="qualification-tritium"})'),
                "target": {"scrape_url": "http://10.0.0.2:8080/metrics",
                           "last_scrape_utc": "1970-01-01T00:16:35+00:00"},
                "final_target": {"scrape_url": "http://10.0.0.2:8080/metrics",
                                 "last_scrape_utc": "1970-01-01T00:17:15+00:00"},
                "baseline_sample": {"timestamp": 990.0, "value": 0.0},
                "peak_sample": {"timestamp": 1020.0, "value": 3.0},
                "settled_sample": {"timestamp": 1030.0, "value": 0.0},
                "load_started_unix": 1000.0, "load_finished_unix": 1030.0,
                "observation_started_unix": 980.0,
                "observation_finished_unix": 1040.0,
                "scaled_pods": {"pods": [
                    {"name": "pod-scale-1", "uid": "pod-scale-uid-1",
                     "node": "node-1", "restarts": 0},
                    {"name": "pod-scale-2", "uid": "pod-scale-uid-2",
                     "node": "node-1", "restarts": 0},
                ]},
                "settled_active": False, "settled_hpa_current": 1,
                "settled_hpa_desired": 1,
            } if flavor == "cpu" else None),
            "helm_history": [
                {"revision": 1, "status": "superseded"},
                {"revision": 2, "status": "failed"},
                {"revision": 3, "status": "deployed"},
            ],
            "prior_helm_revision": 1,
            "failed_manifest_sha256": "0" * 64,
            "failed_image_digest": "sha256:" + "0" * 64,
            "failed_upgrade_output_sha256": "7" * 64,
        },
        "checks": MODULE["expected_checks"](flavor), "result": "pass",
    }
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    return value


def candidate_inputs(raw: str, flavor: str = "cpu") -> tuple[Path, Path, Path, Path, Path]:
    root = Path(raw)
    chart = root / "tritium-1.1.0-rc.0.tgz"
    image = root / f"tritium-{flavor}.oci.tar"
    manifest = root / "tritium.json"
    build = root / "build-receipt.json"
    candidate = root / "manifest.json"
    chart.write_bytes(b"chart")
    image.write_bytes(b"image")
    manifest.write_bytes(b"bundle manifest")
    build.write_text("{}\n", encoding="utf-8")
    candidate.write_text(
        __import__("json").dumps({"artifacts": [{
            "id": "helm-chart", "kind": "helm-chart", "identity": {
                "bytes": chart.stat().st_size,
                "sha256": hashlib.sha256(chart.read_bytes()).hexdigest(),
            },
        }]}) + "\n",
        encoding="utf-8",
    )
    return chart, image, manifest, build, candidate


def validate(value: dict, chart: Path, image: Path, manifest: Path,
             build: Path, candidate: Path) -> dict:
    archive_result = {
        "image_manifest_digest": value["image"].rpartition("@")[2],
        "release": value["release"], "source_revision": value["source_revision"],
        "flavor": value["flavor"],
    }
    with mock.patch.dict(validate_receipt.__globals__, {
        "validate_oci_archive": mock.Mock(return_value=archive_result),
        "manifest_identity": mock.Mock(return_value=value["manifest"]),
    }):
        return validate_receipt(
            value, chart_path=chart, image_path=image, manifest_path=manifest,
            build_receipt=build, package_candidate=candidate, digest_tool="tritium",
            revision="a" * 40, release="1.1.0-rc.0",
        )


class QualifyKubernetesDeploymentTests(unittest.TestCase):
    def test_prometheus_target_absence_requires_zero_matching_active_targets(self):
        response = {"status": "success", "data": {"activeTargets": [{
            "labels": {"namespace": "other", "service": "other-service"},
        }]}}
        with mock.patch.dict(prometheus_target_absence.__globals__, {
            "public_json": mock.Mock(return_value=response),
        }):
            observed = prometheus_target_absence(
                "http://prometheus", namespace="tritium-test",
                service="qualification-tritium", timeout=1,
            )
        self.assertEqual(observed["active_matches"], 0)
        response["data"]["activeTargets"][0]["labels"] = {
            "namespace": "tritium-test", "service": "qualification-tritium",
        }
        with mock.patch.dict(prometheus_target_absence.__globals__, {
            "public_json": mock.Mock(return_value=response),
        }), self.assertRaisesRegex(DeploymentError, "still has an active"):
            prometheus_target_absence(
                "http://prometheus", namespace="tritium-test",
                service="qualification-tritium", timeout=1,
            )

    def test_metrics_scrape_flood_runs_synchronized_generation(self):
        baseline = "\n".join([
            "tritium_chat_requests_total 1",
            "tritium_tokens_out_total 1",
            "tritium_worker_alive 1",
            "tritium_backend_faults_total 0",
            "tritium_backend_faulted 0",
            "",
        ]).encode()
        metrics = "\n".join([
            "tritium_chat_requests_total 2",
            "tritium_tokens_out_total 2",
            "tritium_worker_alive 1",
            "tritium_backend_faults_total 0",
            "tritium_backend_faulted 0",
            "",
        ]).encode()
        with mock.patch.dict(qualify_metrics_scrape_flood.__globals__, {
            "request": mock.Mock(side_effect=[baseline, *([metrics] * 65)]),
            "request_json": mock.Mock(return_value={"choices": [{"message": {
                "role": "assistant", "content": "ok",
            }}]}),
        }):
            result = qualify_metrics_scrape_flood(
                "http://127.0.0.1:8080", "token", model_id="tritium",
                prompt="Hello", request_timeout=5,
            )
        self.assertEqual(result["requests"], 64)
        self.assertEqual(result["peak_client_tasks"], 8)
        self.assertEqual(len(result["response_sha256s"]), 64)

    def test_metrics_scrape_flood_bounds_stalled_wave_by_one_deadline(self):
        metrics = "\n".join([
            "tritium_chat_requests_total 1", "tritium_tokens_out_total 1",
            "tritium_worker_alive 1", "tritium_backend_faults_total 0",
            "tritium_backend_faulted 0", "",
        ]).encode()
        calls = 0
        lock = __import__("threading").Lock()

        def stalled(_url, _token, *, timeout):
            nonlocal calls
            with lock:
                calls += 1
                call = calls
            if call == 1:
                return metrics
            del timeout
            __import__("threading").Event().wait()
            raise AssertionError("unreachable")

        began = time.monotonic()
        with mock.patch.dict(qualify_metrics_scrape_flood.__globals__, {
            "METRICS_FLOOD_BUDGET_MS": 50,
            "request": stalled,
            "request_json": mock.Mock(return_value={"choices": [{}]}),
        }), self.assertRaisesRegex(DeploymentError, "wall deadline"):
            qualify_metrics_scrape_flood(
                "http://127.0.0.1:8080", "token", model_id="tritium",
                prompt="Hello", request_timeout=5,
            )
        self.assertLess(time.monotonic() - began, 0.5)

    def test_metrics_scrape_flood_final_scrape_uses_remaining_deadline(self):
        baseline = "\n".join([
            "tritium_chat_requests_total 1", "tritium_tokens_out_total 1",
            "tritium_worker_alive 1", "tritium_backend_faults_total 0",
            "tritium_backend_faulted 0", "",
        ]).encode()
        final = baseline.replace(b"total 1", b"total 2")
        calls = 0
        lock = __import__("threading").Lock()

        def final_stall(_url, _token, *, timeout):
            nonlocal calls
            with lock:
                calls += 1
                call = calls
            if call == 1:
                return baseline
            if call <= 65:
                return final
            time.sleep(timeout + 0.02)
            raise DeploymentError("final metrics timeout")

        began = time.monotonic()
        with mock.patch.dict(qualify_metrics_scrape_flood.__globals__, {
            "METRICS_FLOOD_BUDGET_MS": 100,
            "request": final_stall,
            "request_json": mock.Mock(return_value={"choices": [{}]}),
        }), self.assertRaisesRegex(DeploymentError, "wall deadline"):
            qualify_metrics_scrape_flood(
                "http://127.0.0.1:8080", "token", model_id="tritium",
                prompt="Hello", request_timeout=5,
            )
        self.assertLess(time.monotonic() - began, 0.5)

    def test_slow_collector_has_absolute_process_deadline(self):
        def stalled(*_args, **_kwargs):
            __import__("threading").Event().wait()

        began = time.monotonic()
        with mock.patch.dict(qualify_slow_collector.__globals__, {
            "SLOW_COLLECTOR_BUDGET_MS": 50,
            "_qualify_slow_collector": stalled,
        }), self.assertRaisesRegex(DeploymentError, "wall deadline"):
            qualify_slow_collector(
                "http://127.0.0.1:8080", "token", model_id="tritium",
                prompt="Hello", request_timeout=5,
                qualification_started=time.monotonic() - 0.01,
            )
        self.assertLess(time.monotonic() - began, 0.5)

    def test_slow_collector_validator_binds_hold_and_transition_causality(self):
        value = {
            "endpoint": "/metrics", "authentication": "bearer",
            "connections": 8, "hold_ms": 3000, "budget_ms": 30000,
            "duration_ms": 4000.0, "generation_latency_ms": 500.0,
            "started_elapsed_ms": 1000.0, "completed_elapsed_ms": 5100.0,
            "generation_response_sha256": "9" * 64,
            "request": request_evidence(
                "tritium", "Hello", temperature=0, max_tokens=1
            ),
            "response_bytes": [1024] * 8,
            "response_sha256s": [f"{index + 20:064x}" for index in range(8)],
            "transitions": [
                {"state": state, "elapsed_ms": elapsed}
                for state, elapsed in zip((
                    "partial_headers_open", "generation_complete",
                    "hold_complete", "scrapes_complete",
                ), (100.0, 600.0, 3100.0, 3900.0), strict=True)
            ],
        }
        self.assertEqual(validate_slow_collector(
            value, model_id="tritium", run_duration_ms=6000.0
        ), value)
        value["transitions"][2]["elapsed_ms"] = 3099.0
        with self.assertRaisesRegex(DeploymentError, "transition causality"):
            validate_slow_collector(
                value, model_id="tritium", run_duration_ms=6000.0
            )
        value["transitions"][2]["elapsed_ms"] = 3100.0
        value["completed_elapsed_ms"] = 6001.0
        with self.assertRaisesRegex(DeploymentError, "bounds differ"):
            validate_slow_collector(
                value, model_id="tritium", run_duration_ms=6000.0
            )
        value["completed_elapsed_ms"] = 5100.0
        value["request"]["temperature"] = False
        descriptor = {
            key: value["request"][key]
            for key in value["request"] if key != "descriptor_sha256"
        }
        value["request"]["descriptor_sha256"] = hashlib.sha256(
            canonical(descriptor)
        ).hexdigest()
        with self.assertRaisesRegex(DeploymentError, "request differs"):
            validate_slow_collector(
                value, model_id="tritium", run_duration_ms=6000.0
            )

    def test_deployment_update_identity_binds_exact_rate_limit_delta(self):
        document = {
            "metadata": {"generation": 4},
            "spec": {"template": {"spec": {"containers": [
                {"name": "tritium", "args": ["--rate-limit-burst", "9"]},
                {"name": "authenticated-probe"},
            ]}}},
        }
        self.assertEqual(
            deployment_update_identity(document),
            {"generation": 4, "rate_limit_burst": 9},
        )
        document["spec"]["template"]["spec"]["containers"][0]["args"] += [
            "--rate-limit-burst", "10"
        ]
        with self.assertRaisesRegex(DeploymentError, "absent or duplicated"):
            deployment_update_identity(document)

    def test_startup_secret_contract_binds_both_container_paths(self):
        document = {
            "metadata": {"uid": "deployment-uid", "resourceVersion": "20"},
            "spec": {"template": {"spec": {"containers": [
                {"name": "tritium", "env": [{
                    "name": "TRITIUM_AUTH_TOKEN", "valueFrom": {
                        "secretKeyRef": {"name": "tritium-auth", "key": "token"}
                    },
                }]},
                {"name": "authenticated-probe", "env": [{
                    "name": "TRITIUM_AUTH_TOKEN", "valueFrom": {
                        "secretKeyRef": {"name": "tritium-auth", "key": "token"}
                    },
                }]},
            ]}}},
        }
        contract = auth_secret_contract(document)
        self.assertEqual(contract["secret_name"], "tritium-auth")
        self.assertEqual([item["container_index"] for item in contract["bindings"]], [0, 1])
        document["spec"]["template"]["spec"]["containers"][1]["env"][0][
            "valueFrom"
        ]["secretKeyRef"]["name"] = "other-auth"
        with self.assertRaisesRegex(DeploymentError, "do not share"):
            auth_secret_contract(document)

    def test_missing_secret_failure_requires_new_exact_config_error(self):
        document = {"items": [{
            "metadata": {"name": "pod-fault", "uid": "fault-uid",
                         "ownerReferences": [{
                             "kind": "ReplicaSet", "name": "tritium-rs",
                             "uid": "rs-uid", "controller": True,
                         }]},
            "status": {"containerStatuses": [{
                "name": "tritium", "state": {"waiting": {
                    "reason": "CreateContainerConfigError",
                    "message": 'secret "tritium-missing-auth-0123456789ab" not found',
                }},
            }]},
        }]}
        failure = missing_secret_failure(
            document, baseline_uids={"baseline-uid"},
            missing_secret="tritium-missing-auth-0123456789ab",
        )
        self.assertEqual(failure["pod_uid"], "fault-uid")
        self.assertEqual(failure["reason"], "CreateContainerConfigError")
        self.assertIsNone(missing_secret_failure(
            document, baseline_uids={"fault-uid"},
            missing_secret="tritium-missing-auth-0123456789ab",
        ))

    def test_missing_secret_cleanup_removes_owned_ref_and_preserves_foreign_drift(self):
        baseline = {
            "metadata": {"uid": "deployment-uid", "resourceVersion": "20"},
            "spec": {"template": {"spec": {"containers": [
                {"name": "tritium", "env": [{
                    "name": "TRITIUM_AUTH_TOKEN", "valueFrom": {"secretKeyRef": {
                        "name": "tritium-auth", "key": "token",
                    }},
                }]},
                {"name": "authenticated-probe", "env": [{
                    "name": "TRITIUM_AUTH_TOKEN", "valueFrom": {"secretKeyRef": {
                        "name": "tritium-auth", "key": "token",
                    }},
                }]},
            ]}}},
        }
        contract = auth_secret_contract(baseline)
        mixed = copy.deepcopy(baseline)
        mixed["spec"]["template"]["spec"]["containers"][0]["env"][0][
            "valueFrom"
        ]["secretKeyRef"]["name"] = "tritium-missing-auth-0123456789ab"
        mixed["spec"]["template"]["spec"]["containers"][1]["env"][0][
            "valueFrom"
        ]["secretKeyRef"]["name"] = "foreign-auth"
        cleaned = copy.deepcopy(mixed)
        cleaned["spec"]["template"]["spec"]["containers"][0]["env"][0][
            "valueFrom"
        ]["secretKeyRef"]["name"] = "tritium-auth"
        run_mock = mock.Mock(return_value="")
        with mock.patch.dict(restore_missing_secret_refs.__globals__, {
            "run": run_mock,
            "run_json": mock.Mock(side_effect=[mixed, cleaned]),
        }), self.assertRaisesRegex(DeploymentError, "preserved foreign drift"):
            restore_missing_secret_refs(
                ["kubectl"], service="qualification-tritium", contract=contract,
                missing_secret="tritium-missing-auth-0123456789ab", timeout=10,
            )
        patch_payload = run_mock.call_args.args[0][-1]
        self.assertIn("containers/0/env/0", patch_payload)
        self.assertNotIn("containers/1/env/0", patch_payload)

    def test_startup_argument_contract_binds_exact_chart_positions(self):
        args = [
            "--bundle", "/artifacts/bundle", "--profile", "compact-v1",
            "--backend", "cpu", "--host", "0.0.0.0", "--port", "8080",
            "--model-id", "tritium", "--max-new", "256",
        ]
        document = {
            "metadata": {"uid": "deployment-uid", "resourceVersion": "30"},
            "spec": {"template": {"spec": {"containers": [
                {"name": "tritium", "args": args},
                {"name": "authenticated-probe"},
            ]}}},
        }
        backend = startup_argument_contract(
            document, flag="--backend", source_value="cpu"
        )
        invalid = startup_argument_contract(
            document, flag="--max-new", source_value="256"
        )
        self.assertEqual((backend["value_index"], invalid["value_index"]), (5, 13))
        args[13] = "255"
        with self.assertRaisesRegex(DeploymentError, "binding differs"):
            startup_argument_contract(
                document, flag="--max-new", source_value="256"
            )

    def test_startup_error_line_binds_scenario_and_linked_backends(self):
        invalid = 'Error: "all request and prompt limits must be >= 1"\n'
        self.assertEqual(startup_error_line(
            invalid, scenario="invalid_config", flavor="cpu"
        ), invalid.strip())
        cuda = (
            'Error: "backend `tritium-unavailable` is not in the registry '
            '(linked backends: cuda, cpu); for cuda, build with `--features cuda`"\n'
        )
        self.assertEqual(startup_error_line(
            cuda, scenario="unavailable_backend", flavor="cuda"
        ), cuda.strip())
        with self.assertRaisesRegex(DeploymentError, "error line differs"):
            startup_error_line(
                cuda, scenario="unavailable_backend", flavor="cpu"
            )
        duplicate = cuda.replace("cuda, cpu", "cuda, cpu, cpu")
        with self.assertRaisesRegex(DeploymentError, "error line differs"):
            startup_error_line(
                duplicate, scenario="unavailable_backend", flavor="cuda"
            )

    def test_startup_log_command_tracks_terminated_container_instance(self):
        current = startup_log_command(
            ["kubectl"], pod_name="pod-fault", termination_source="current"
        )
        previous = startup_log_command(
            ["kubectl"], pod_name="pod-fault", termination_source="last_state"
        )
        self.assertNotIn("--previous", current)
        self.assertEqual(previous[-1], "--previous")

    def test_startup_process_failure_binds_new_pod_controller(self):
        document = {"items": [{
            "metadata": {"name": "pod-fault", "uid": "fault-uid",
                         "ownerReferences": [{
                             "kind": "ReplicaSet", "name": "tritium-rs",
                             "uid": "rs-uid", "controller": True,
                         }]},
            "status": {"containerStatuses": [{
                "name": "tritium", "restartCount": 1,
                "state": {"waiting": {"reason": "CrashLoopBackOff"}},
                "lastState": {"terminated": {"exitCode": 1, "reason": "Error"}},
            }]},
        }]}
        failure = startup_process_failure(document, baseline_uids={"baseline-uid"})
        self.assertEqual(failure["termination_source"], "last_state")
        self.assertEqual(failure["replica_set_uid"], "rs-uid")

    def test_startup_argument_cleanup_handles_admitted_response_loss(self):
        args = [
            "--bundle", "/artifacts/bundle", "--profile", "compact-v1",
            "--backend", "tritium-unavailable",
        ]
        faulted = {
            "metadata": {"uid": "deployment-uid", "resourceVersion": "31"},
            "spec": {"template": {"spec": {"containers": [
                {"name": "tritium", "args": args},
                {"name": "authenticated-probe"},
            ]}}},
        }
        contract = {
            "deployment_uid": "deployment-uid", "resource_version": "30",
            "container": "tritium", "container_index": 0, "flag": "--backend",
            "flag_index": 4,
            "flag_path": "/spec/template/spec/containers/0/args/4",
            "value_index": 5, "source_value": "cpu",
            "args": [
                "--bundle", "/artifacts/bundle", "--profile", "compact-v1",
                "--backend", "cpu",
            ],
            "path": "/spec/template/spec/containers/0/args/5",
        }
        restored = copy.deepcopy(faulted)
        restored["metadata"]["resourceVersion"] = "32"
        restored["spec"]["template"]["spec"]["containers"][0]["args"][5] = "cpu"
        with mock.patch.dict(restore_startup_argument.__globals__, {
            "run": mock.Mock(side_effect=DeploymentError("response lost")),
            "run_json": mock.Mock(side_effect=[faulted, restored]),
        }):
            self.assertEqual(restore_startup_argument(
                ["kubectl"], service="qualification-tritium", contract=contract,
                fault_value="tritium-unavailable", timeout=10,
            ), restored)

    def test_startup_argument_cleanup_restores_injection_then_rejects_vector_drift(self):
        baseline_args = [
            "--bundle", "/artifacts/bundle", "--profile", "compact-v1",
            "--backend", "cpu",
        ]
        contract = {
            "deployment_uid": "deployment-uid", "resource_version": "30",
            "container": "tritium", "container_index": 0, "flag": "--backend",
            "flag_index": 4,
            "flag_path": "/spec/template/spec/containers/0/args/4",
            "value_index": 5, "source_value": "cpu", "args": baseline_args,
            "path": "/spec/template/spec/containers/0/args/5",
        }
        cases = [
            ["--bundle", "/artifacts/bundle", "--profile", "foreign-profile",
             "--backend", "tritium-unavailable"],
            ["--backend", "tritium-unavailable", "--bundle", "/artifacts/bundle",
             "--profile", "compact-v1"],
        ]
        for faulted_args in cases:
            with self.subTest(args=faulted_args):
                faulted = {
                    "metadata": {"uid": "deployment-uid", "resourceVersion": "31"},
                    "spec": {"template": {"spec": {"containers": [
                        {"name": "tritium", "args": faulted_args},
                        {"name": "authenticated-probe"},
                    ]}}},
                }
                cleaned = copy.deepcopy(faulted)
                clean_args = cleaned["spec"]["template"]["spec"]["containers"][0]["args"]
                injected = clean_args.index("tritium-unavailable")
                clean_args[injected] = "cpu"
                run_mock = mock.Mock(return_value="")
                with mock.patch.dict(restore_startup_argument.__globals__, {
                    "run": run_mock,
                    "run_json": mock.Mock(side_effect=[faulted, cleaned]),
                }), self.assertRaisesRegex(DeploymentError, "foreign drift"):
                    restore_startup_argument(
                        ["kubectl"], service="qualification-tritium",
                        contract=contract, fault_value="tritium-unavailable", timeout=10,
                    )
                payload = run_mock.call_args.args[0][-1]
                expected_value_index = faulted_args.index("tritium-unavailable")
                self.assertIn(f"args/{expected_value_index}", payload)

    def test_scale_contract_binds_prometheus_trigger_and_authenticated_monitor(self):
        query = 'max(tritium_queue_depth{namespace="ns",service="qualification-tritium"})'
        scaled = {"spec": {
            "scaleTargetRef": {"name": "qualification-tritium"},
            "pollingInterval": 5, "cooldownPeriod": 30,
            "minReplicaCount": 1, "maxReplicaCount": 2,
            "triggers": [{"type": "prometheus", "metadata": {
                "serverAddress": "http://prometheus.monitoring.svc:9090",
                "metricName": "tritium_queue_pressure", "threshold": "1",
                "query": query,
            }}],
        }}
        monitor = {
            "metadata": {"labels": {"release": "stack"}},
            "spec": {
                "selector": {"matchLabels": {
                    "app.kubernetes.io/name": "tritium",
                    "app.kubernetes.io/instance": "qualification",
                }},
                "endpoints": [{"path": "/metrics", "port": "http", "interval": "30s",
                    "authorization": {
                    "credentials": {"name": "auth", "key": "token"},
                }}],
            },
        }
        validate_scale_contract(
            scaled, monitor, service="qualification-tritium",
            server="http://prometheus.monitoring.svc:9090", query=query,
            auth_secret="auth", auth_key="token", monitor_label=("release", "stack"),
        )
        scaled["spec"]["triggers"][0]["metadata"]["query"] = "vector(100)"
        with self.assertRaisesRegex(DeploymentError, "KEDA trigger"):
            validate_scale_contract(
                scaled, monitor, service="qualification-tritium",
                server="http://prometheus.monitoring.svc:9090", query=query,
                auth_secret="auth", auth_key="token",
                monitor_label=("release", "stack"),
            )

    def test_prometheus_url_rejects_helm_value_injection(self):
        self.assertEqual(
            prometheus_url("http://prometheus.monitoring.svc:9090"),
            "http://prometheus.monitoring.svc:9090",
        )
        for value in (
            "http://prometheus:9090,networkPolicy.enabled=false",
            "http://user:password@prometheus:9090",
            "http://prometheus:9090/?query=unsafe",
        ):
            with self.subTest(value=value), self.assertRaises(DeploymentError):
                prometheus_url(value)

    def test_scale_snapshot_requires_active_ready_keda_and_two_ready_replicas(self):
        scaled = {
            "metadata": {"uid": "scaled-uid"},
            "spec": {"minReplicaCount": 1, "maxReplicaCount": 2},
            "status": {
                "conditions": [{"type": "Ready", "status": "True"},
                               {"type": "Active", "status": "True"}],
                "externalMetricNames": ["s0-prometheus-tritium_queue_pressure"],
            },
        }
        hpa = {"metadata": {"uid": "hpa-uid", "ownerReferences": [{
                    "kind": "ScaledObject", "uid": "scaled-uid", "controller": True,
                }]},
               "status": {"currentReplicas": 2, "desiredReplicas": 2}}
        deployment = {"spec": {"replicas": 2}, "status": {"readyReplicas": 2}}
        self.assertEqual(scale_snapshot(scaled, hpa, deployment)["scaled_replicas"], 2)
        scaled["status"]["conditions"][1]["status"] = "False"
        with self.assertRaisesRegex(DeploymentError, "not active"):
            scale_snapshot(scaled, hpa, deployment)

    def test_metrics_snapshot_requires_observed_generation_and_live_worker(self):
        metrics = (
            "tritium_chat_requests_total 1\n"
            "tritium_tokens_out_total 1\n"
            "tritium_worker_alive 1\n"
            "tritium_backend_faults_total 0\n"
            "tritium_backend_faulted 0\n"
        )
        self.assertEqual(
            metrics_snapshot(metrics)["values"]["tritium_tokens_out_total"], 1.0
        )
        with self.assertRaisesRegex(DeploymentError, "did not observe generation"):
            metrics_snapshot(metrics.replace("tokens_out_total 1", "tokens_out_total 0"))
        with self.assertRaisesRegex(DeploymentError, "backend fault"):
            metrics_snapshot(metrics.replace("backend_faulted 0", "backend_faulted 1"))

    def test_helm_history_proves_atomic_failure_and_recovery(self):
        history = [
            {"revision": 1, "status": "superseded"},
            {"revision": 2, "status": "failed"},
            {"revision": 3, "status": "deployed"},
        ]
        self.assertEqual(validate_helm_history(history, 1), history)
        history[1]["status"] = "superseded"
        with self.assertRaisesRegex(DeploymentError, "failed upgrade and rollback"):
            validate_helm_history(history, 1)

    def test_rollback_runtime_binds_cri_digest_and_controller_lineage(self):
        image = "registry.example/tritium@sha256:" + "2" * 64
        pod = {
            "metadata": {
                "name": "pod", "uid": "pod-uid",
                "ownerReferences": [{
                    "kind": "ReplicaSet", "name": "tritium-rs", "uid": "rs-uid",
                    "controller": True,
                }],
            },
            "spec": {"nodeName": "node-1", "containers": [
                {"name": "tritium", "image": image},
                {"name": "authenticated-probe"},
            ]},
            "status": {
                "conditions": [{"type": "Ready", "status": "True"}],
                "containerStatuses": [
                    {"name": "tritium", "image": image,
                     "imageID": "containerd://sha256:" + "2" * 64,
                     "ready": True, "restartCount": 0},
                    {"name": "authenticated-probe", "restartCount": 0},
                ],
            },
        }
        replica_set = {
            "metadata": {
                "name": "tritium-rs", "uid": "rs-uid",
                "ownerReferences": [{
                    "kind": "Deployment", "name": "tritium",
                    "uid": "deployment-uid", "controller": True,
                }],
            },
        }
        with mock.patch.dict(collect_rollback_runtime.__globals__, {
            "run_json": mock.Mock(side_effect=[{"items": [pod]}, replica_set]),
        }):
            evidence = collect_rollback_runtime(
                ["kubectl"], selector="app=tritium", image=image,
                deployment_name="tritium", deployment_uid="deployment-uid",
                flavor="cpu", run_elapsed_ms=50.0,
            )
        self.assertEqual(evidence["image_digest"], "sha256:" + "2" * 64)
        self.assertEqual(evidence["pods"][0]["replica_set_uid"], "rs-uid")
        self.assertEqual(
            validate_rollback_runtime(
                evidence, image=image, deployment_name="tritium",
                deployment_uid="deployment-uid", run_duration_ms=100.0,
            ), evidence,
        )
        evidence["pods"][0]["image_id"] = "containerd://sha256:" + "3" * 64
        with self.assertRaisesRegex(DeploymentError, "runtime pod identity"):
            validate_rollback_runtime(
                evidence, image=image, deployment_name="tritium",
                deployment_uid="deployment-uid", run_duration_ms=100.0,
            )
        evidence["pods"][0]["image_id"] = "containerd://sha256:" + "2" * 64
        evidence["pods"][0]["replica_set_owner"]["uid"] = "foreign-uid"
        with self.assertRaisesRegex(DeploymentError, "runtime pod identity"):
            validate_rollback_runtime(
                evidence, image=image, deployment_name="tritium",
                deployment_uid="deployment-uid", run_duration_ms=100.0,
            )
        evidence["pods"][0]["replica_set_owner"]["uid"] = "deployment-uid"
        evidence["elapsed_ms"] = 101.0
        with self.assertRaisesRegex(DeploymentError, "identity fields"):
            validate_rollback_runtime(
                evidence, image=image, deployment_name="tritium",
                deployment_uid="deployment-uid", run_duration_ms=100.0,
            )

    def test_rollback_runtime_rejects_foreign_deployment_owner(self):
        image = "registry.example/tritium@sha256:" + "2" * 64
        pod = {
            "metadata": {"name": "pod", "uid": "pod-uid", "ownerReferences": [{
                "kind": "ReplicaSet", "name": "tritium-rs", "uid": "rs-uid",
                "controller": True,
            }]},
            "spec": {"nodeName": "node-1", "containers": [
                {"name": "tritium", "image": image}, {"name": "authenticated-probe"},
            ]},
            "status": {
                "conditions": [{"type": "Ready", "status": "True"}],
                "containerStatuses": [
                    {"name": "tritium", "image": image,
                     "imageID": "containerd://sha256:" + "2" * 64,
                     "ready": True, "restartCount": 0},
                    {"name": "authenticated-probe", "restartCount": 0},
                ],
            },
        }
        replica_set = {"metadata": {"name": "tritium-rs", "uid": "rs-uid",
                                    "ownerReferences": [{
            "kind": "Deployment", "name": "foreign", "uid": "foreign-uid",
            "controller": True,
        }]}}
        with mock.patch.dict(collect_rollback_runtime.__globals__, {
            "run_json": mock.Mock(side_effect=[{"items": [pod]}, replica_set]),
        }), self.assertRaisesRegex(DeploymentError, "another Deployment"):
            collect_rollback_runtime(
                ["kubectl"], selector="app=tritium", image=image,
                deployment_name="tritium", deployment_uid="deployment-uid",
                flavor="cpu", run_elapsed_ms=50.0,
            )

    def test_deployment_snapshot_binds_digest_manifest_replicas_and_strategy(self):
        image = "registry.example/tritium@sha256:" + "2" * 64
        document = {
            "metadata": {"uid": "deployment-uid"},
            "spec": {"replicas": 1, "strategy": {"type": "RollingUpdate"},
                     "template": {"metadata": {"annotations": {
                         "tritium.ai/image-digest": "sha256:" + "2" * 64,
                         "tritium.ai/manifest-sha256": "3" * 64,
                     }}, "spec": {"containers": [
                         {"name": "tritium", "image": image},
                         {"name": "authenticated-probe"},
                     ]}}},
            "status": {"readyReplicas": 1},
        }
        self.assertEqual(
            validate_deployment(document, image=image, manifest_sha256="3" * 64,
                                flavor="cpu"),
            "deployment-uid",
        )
        document["spec"]["template"]["metadata"]["annotations"][
            "tritium.ai/manifest-sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(DeploymentError, "manifest annotation"):
            validate_deployment(document, image=image, manifest_sha256="3" * 64,
                                flavor="cpu")

    def test_cuda_pod_snapshot_requires_ready_two_container_gpu_pod(self):
        document = {"items": [{
            "metadata": {"name": "pod", "uid": "uid"},
            "spec": {"nodeName": "gpu-node", "containers": [
                {"resources": {"limits": {"nvidia.com/gpu": "1"}}}, {}
            ]},
            "status": {"conditions": [{"type": "Ready", "status": "True"}],
                       "containerStatuses": [{"restartCount": 0}, {"restartCount": 0}]},
        }]}
        self.assertEqual(pod_snapshot(document, "cuda")["pods"][0]["node"], "gpu-node")
        document["items"][0]["spec"]["containers"][0]["resources"]["limits"].clear()
        with self.assertRaisesRegex(DeploymentError, "NVIDIA GPU"):
            pod_snapshot(document, "cuda")

    def test_watchdog_container_identity_binds_restart_and_exit(self):
        document = {
            "metadata": {"name": "pod", "uid": "pod-uid"},
            "status": {"containerStatuses": [{
                "name": "tritium", "containerID": "containerd://new",
                "restartCount": 1,
                "lastState": {"terminated": {"exitCode": 137}},
            }]},
        }
        self.assertEqual(container_identity(document, "pod", "tritium"), {
            "pod_uid": "pod-uid", "container_id": "containerd://new",
            "restart_count": 1, "last_exit_code": 137,
        })
        document["status"]["containerStatuses"].append(
            dict(document["status"]["containerStatuses"][0])
        )
        with self.assertRaisesRegex(DeploymentError, "status differs"):
            container_identity(document, "pod", "tritium")

    def test_watchdog_contract_derives_bounded_escalation_budget(self):
        script = (
            'terminate_tritium() { pid="$(pidof tritium-serve)"; kill $pid; sleep 2; '
            'kill -0 $pid && kill -KILL $pid || true; }; '
            'until wget --timeout=2 '
            '--header="Authorization: Bearer ${TRITIUM_AUTH_TOKEN}" '
            'http://127.0.0.1:8080/healthz | grep -q \'\\"status\\":\\"ok\\"\'; do '
            'sleep 5; done; failures=0; '
            'while sleep 10; do if wget --timeout=2 '
            '--header="Authorization: Bearer ${TRITIUM_AUTH_TOKEN}" '
            'http://127.0.0.1:8080/healthz | grep -q \'\\"status\\":\\"ok\\"\'; '
            'then failures=0; else failures=$((failures + 1)); fi; '
            'if [ "$failures" -ge 3 ]; then pid="$(pidof tritium-serve)"; '
            'terminate_tritium; fi; done'
        )
        document = {"spec": {"template": {"spec": {"containers": [
            {"name": "tritium", "ports": [{"name": "http", "containerPort": 8080}],
             "startupProbe": {"failureThreshold": 60, "periodSeconds": 5,
                              "timeoutSeconds": 2, "tcpSocket": {"port": "http"}}},
            {"name": "authenticated-probe", "args": [script],
             "startupProbe": {"failureThreshold": 60, "periodSeconds": 5,
                              "timeoutSeconds": 2, "exec": {"command": [
                                  "sh", "-ec",
                                  'wget -qO- --header="Authorization: Bearer '
                                  '${TRITIUM_AUTH_TOKEN}" '
                                  'http://127.0.0.1:8080/healthz | grep -q '
                                  '\'"status":"ok"\'',
                              ]}}},
        ]}}}}
        self.assertEqual(watchdog_contract(document)["budget_ms"], 98000)
        wrong_probe = copy.deepcopy(document)
        wrong_probe["spec"]["template"]["spec"]["containers"][0]["startupProbe"][
            "failureThreshold"
        ] = 1
        with self.assertRaisesRegex(DeploymentError, "startup probes differ"):
            watchdog_contract(wrong_probe)
        wrong_main_handler = copy.deepcopy(document)
        wrong_main_handler["spec"]["template"]["spec"]["containers"][0][
            "startupProbe"
        ] = {"failureThreshold": 60, "periodSeconds": 5, "timeoutSeconds": 2,
             "exec": {"command": ["true"]}}
        with self.assertRaisesRegex(DeploymentError, "main startup probe handler"):
            watchdog_contract(wrong_main_handler)
        wrong_watchdog_handler = copy.deepcopy(document)
        wrong_watchdog_handler["spec"]["template"]["spec"]["containers"][1][
            "startupProbe"
        ]["exec"]["command"] = ["true"]
        with self.assertRaisesRegex(DeploymentError, "watchdog startup probe handler"):
            watchdog_contract(wrong_watchdog_handler)
        wrong_auth = copy.deepcopy(document)
        wrong_auth["spec"]["template"]["spec"]["containers"][1]["args"][0] = (
            script.replace("Authorization: Bearer", "Authorization: Basic")
        )
        with self.assertRaisesRegex(DeploymentError, "gate semantics differ"):
            watchdog_contract(wrong_auth)
        wrong_monitor_auth = copy.deepcopy(document)
        monitor_prefix, monitor_suffix = script.rsplit("Authorization: Bearer", 1)
        wrong_monitor_auth["spec"]["template"]["spec"]["containers"][1]["args"][0] = (
            monitor_prefix + "Authorization: Basic" + monitor_suffix
        )
        with self.assertRaisesRegex(DeploymentError, "monitor gate semantics differ"):
            watchdog_contract(wrong_monitor_auth)
        terminating_startup = copy.deepcopy(document)
        terminating_startup["spec"]["template"]["spec"]["containers"][1]["args"][0] = (
            script.replace("sleep 5; done", "terminate_tritium; sleep 5; done")
        )
        with self.assertRaisesRegex(DeploymentError, "can terminate Tritium"):
            watchdog_contract(terminating_startup)
        dynamic_port = copy.deepcopy(document)
        dynamic_port["spec"]["template"]["spec"]["containers"][0]["ports"][0][
            "containerPort"
        ] = 9090
        dynamic_port["spec"]["template"]["spec"]["containers"][1]["args"][0] = (
            script.replace(":8080/healthz", ":9090/healthz")
        )
        dynamic_port["spec"]["template"]["spec"]["containers"][1]["startupProbe"][
            "exec"
        ]["command"][2] = dynamic_port["spec"]["template"]["spec"]["containers"][1][
            "startupProbe"
        ]["exec"]["command"][2].replace(":8080/healthz", ":9090/healthz")
        self.assertEqual(
            watchdog_contract(dynamic_port)["startup_gate"]["url"],
            "http://127.0.0.1:9090/healthz",
        )
        document["spec"]["template"]["spec"]["containers"][1]["args"][0] = (
            script.replace("sleep 2; kill -0", "sleep 3; kill -0")
        )
        with self.assertRaisesRegex(DeploymentError, "bounds differ"):
            watchdog_contract(document)

    def test_service_port_contract_binds_named_service_target(self):
        deployment = {"spec": {"template": {"spec": {"containers": [{
            "name": "tritium",
            "ports": [{"name": "http", "containerPort": 9090}],
        }]}}}}
        service = {"spec": {"ports": [{
            "name": "http", "port": 9090, "targetPort": "http",
        }]}}
        self.assertEqual(deployment_service_port(deployment), 9090)
        validate_service_port(service, 9090)
        service["spec"]["ports"][0]["targetPort"] = 9090
        with self.assertRaisesRegex(DeploymentError, "Service port differs"):
            validate_service_port(service, 9090)

    def test_artifact_volume_contract_and_pending_failure_are_exact(self):
        deployment = {"spec": {"template": {"spec": {"volumes": [
            {"name": "source-artifact", "persistentVolumeClaim": {
                "claimName": "tritium-artifact"
            }},
        ]}}}}
        self.assertEqual(artifact_volume_contract(deployment), {
            "volume_index": 0, "claim_name": "tritium-artifact",
        })
        pods = {"items": [{
            "metadata": {"name": "pending", "uid": "new-uid"},
            "status": {"conditions": [{
                "type": "PodScheduled", "status": "False",
                "reason": "Unschedulable",
                "message": 'persistentvolumeclaim "tritium-missing-abc" not found',
            }]},
        }]}
        observed = pending_artifact_volume_failure(
            pods, missing_claim="tritium-missing-abc", previous_uids={"old-uid"}
        )
        self.assertEqual(observed["pod_uid"], "new-uid")
        self.assertIsNone(pending_artifact_volume_failure(
            pods, missing_claim="other-missing", previous_uids={"old-uid"}
        ))

    def test_memory_oom_contract_and_observation_are_exact(self):
        deployment = {"spec": {"template": {"spec": {"containers": [
            {"name": "tritium", "resources": {"limits": {"memory": "32Gi"}}},
            {"name": "authenticated-probe"},
        ]}}}}
        self.assertEqual(memory_limit_contract(deployment), {
            "container_index": 0, "source_limit": "32Gi",
            "source_limit_bytes": 32 * 1024 ** 3,
            "fault_limit": "16Mi", "fault_limit_bytes": 16 * 1024 ** 2,
        })
        pods = {"items": [{
            "metadata": {
                "name": "pod-oom", "uid": "oom-uid",
                "labels": {"pod-template-hash": "fault-hash"},
                "ownerReferences": [{
                    "kind": "ReplicaSet", "name": "tritium-fault",
                    "uid": "rs-uid", "controller": True,
                }],
            },
            "spec": {"nodeName": "node-1", "containers": [{
                "name": "tritium", "resources": {"limits": {"memory": "16Mi"}},
            }]},
            "status": {"containerStatuses": [{
                "name": "tritium", "restartCount": 2,
                "lastState": {"terminated": {"reason": "OOMKilled", "exitCode": 137}},
            }]},
        }]}
        self.assertEqual(observed_oom_failure(pods, previous_uids=set()), {
            "pod_name": "pod-oom", "pod_uid": "oom-uid", "node": "node-1",
            "restart_count": 2, "reason": "OOMKilled", "last_exit_code": 137,
            "memory_limit": "16Mi", "memory_limit_bytes": 16 * 1024 ** 2,
            "replica_set_name": "tritium-fault", "replica_set_uid": "rs-uid",
            "template_hash": "fault-hash",
        })
        self.assertIsNone(observed_oom_failure(pods, previous_uids={"oom-uid"}))
        wrong_limit = copy.deepcopy(pods)
        wrong_limit["items"][0]["spec"]["containers"][0]["resources"]["limits"][
            "memory"
        ] = "32Gi"
        self.assertIsNone(observed_oom_failure(wrong_limit, previous_uids=set()))
        missing_owner = copy.deepcopy(pods)
        missing_owner["items"][0]["metadata"]["ownerReferences"] = []
        self.assertIsNone(observed_oom_failure(missing_owner, previous_uids=set()))

    def test_oom_replica_set_must_belong_to_qualified_deployment(self):
        observed = {
            "replica_set_name": "tritium-fault", "replica_set_uid": "rs-uid",
            "template_hash": "fault-hash",
        }
        replica_set = {
            "metadata": {
                "name": "tritium-fault", "uid": "rs-uid",
                "ownerReferences": [{
                    "kind": "Deployment", "name": "qualification-tritium",
                    "uid": "deployment-uid", "controller": True,
                }],
            },
            "spec": {"template": {
                "metadata": {"labels": {"pod-template-hash": "fault-hash"}},
                "spec": {"containers": [{
                    "name": "tritium",
                    "resources": {"limits": {"memory": "16Mi"}},
                }]},
            }},
        }
        self.assertEqual(oom_replica_set_lineage(
            replica_set, observed=observed, deployment_name="qualification-tritium",
            deployment_uid="deployment-uid",
        )["deployment_uid"], "deployment-uid")
        replica_set["metadata"]["ownerReferences"][0]["uid"] = "foreign-uid"
        with self.assertRaisesRegex(DeploymentError, "qualified Deployment"):
            oom_replica_set_lineage(
                replica_set, observed=observed,
                deployment_name="qualification-tritium",
                deployment_uid="deployment-uid",
            )

    def test_source_pvc_identity_requires_bound_storage(self):
        document = {
            "metadata": {"name": "tritium-artifact", "uid": "pvc-uid"},
            "spec": {"volumeName": "pv-artifact", "storageClassName": "standard",
                     "accessModes": ["ReadWriteOnce"]},
            "status": {"phase": "Bound", "capacity": {"storage": "64Gi"}},
        }
        self.assertEqual(
            pvc_identity(document, "tritium-artifact")["volume_name"], "pv-artifact"
        )
        document["status"]["phase"] = "Pending"
        with self.assertRaisesRegex(DeploymentError, "not fully bound"):
            pvc_identity(document, "tritium-artifact")

    def test_artifact_volume_fault_always_restores_source_claim(self):
        commands = []

        def fake_run(command, _timeout):
            commands.append(command)
            return ""

        with mock.patch.dict(qualify_artifact_volume_loss.__globals__, {
            "prove_absent": mock.Mock(return_value={
                "status": "NotFound", "output_sha256": hashlib.sha256(b"").hexdigest()
            }),
            "collect_resource_usage": mock.Mock(),
            "run": fake_run,
            "run_json": mock.Mock(return_value={"items": [None]}),
        }):
            with self.assertRaisesRegex(DeploymentError, "pod entry"):
                qualify_artifact_volume_loss(
                    ["kubectl"], service="tritium", namespace="tritium-test",
                    selector="app=tritium",
                    contract={"volume_index": 0, "claim_name": "tritium-artifact"},
                    previous_uids={"old-uid"}, timeout=10,
                )
        self.assertEqual(len(commands), 2)
        self.assertIn("tritium-missing-", commands[0][-1])
        self.assertIn("tritium-artifact", commands[1][-1])

    def test_memory_oom_fault_always_restores_source_limit(self):
        commands = []

        def fake_run(command, _timeout):
            commands.append(command)
            return ""

        with mock.patch.dict(qualify_memory_oom.__globals__, {
            "run": fake_run,
            "run_json": mock.Mock(return_value={"items": [None]}),
            "collect_resource_usage": mock.Mock(),
        }):
            with self.assertRaisesRegex(DeploymentError, "pod entry"):
                qualify_memory_oom(
                    ["kubectl"], service="tritium", namespace="tritium-test",
                    selector="app=tritium",
                    contract={
                        "container_index": 0, "source_limit": "32Gi",
                        "source_limit_bytes": 32 * 1024 ** 3,
                        "fault_limit": "16Mi", "fault_limit_bytes": 16 * 1024 ** 2,
                    },
                    previous_uids={"old-uid"}, deployment_uid="deployment-uid",
                    timeout=10,
                )
        self.assertEqual(len(commands), 2)
        self.assertIn('"value":"16Mi"', commands[0][-1])
        self.assertIn('"value":"32Gi"', commands[1][-1])

    def test_cpu_oom_cleanup_deletes_retained_baseline_pod(self):
        listing = {"items": [{"metadata": {"name": "pod-old", "uid": "old-uid"}}]}
        commands = []
        with mock.patch.dict(replace_pre_oom_pods.__globals__, {
            "run_json": mock.Mock(side_effect=[listing, {"items": []}]),
            "run": mock.Mock(side_effect=lambda command, _timeout: commands.append(command)),
        }):
            result = replace_pre_oom_pods(
                ["kubectl"], selector="app=tritium",
                previous={"pods": [{"name": "pod-old", "uid": "old-uid"}]},
                flavor="cpu", timeout=10,
            )
        self.assertEqual(result["mode"], "deleted_after_restore")
        self.assertEqual(commands[0][2:4], ["pod/pod-old", "--wait=true"])

    def test_cuda_oom_cleanup_requires_baseline_pod_already_absent(self):
        with mock.patch.dict(replace_pre_oom_pods.__globals__, {
            "run_json": mock.Mock(side_effect=[{"items": []}, {"items": []}]),
            "run": mock.Mock(),
        }):
            result = replace_pre_oom_pods(
                ["kubectl"], selector="app=tritium",
                previous={"pods": [{"name": "pod-old", "uid": "old-uid"}]},
                flavor="cuda", timeout=10,
            )
        self.assertEqual(result["mode"], "already_absent")

    def test_collector_fault_always_restores_service_monitor_selector(self):
        commands = []
        monitor = {
            "metadata": {"name": "qualification-tritium", "uid": "monitor-uid",
                         "resourceVersion": "10"},
            "spec": {"selector": {"matchLabels": {
                "app.kubernetes.io/name": "tritium",
                "app.kubernetes.io/instance": "qualification",
            }}},
        }
        faulted = copy.deepcopy(monitor)
        faulted["metadata"]["resourceVersion"] = "11"
        faulted["spec"]["selector"]["matchLabels"] = {"wrong": "selector"}
        injected = copy.deepcopy(monitor)
        injected["metadata"]["resourceVersion"] = "11"
        injected["spec"]["selector"]["matchLabels"][
            "tritium-telemetry-fault"
        ] = "0123456789abcdef"
        restored = copy.deepcopy(monitor)
        restored["metadata"]["resourceVersion"] = "12"

        def fake_run(command, _timeout):
            commands.append(command)
            return ""

        with mock.patch.dict(qualify_collector_outage.__globals__, {
            "run": fake_run,
            "run_json": mock.Mock(side_effect=[
                {"metadata": {"uid": "service-uid", "labels": {
                    "app.kubernetes.io/name": "tritium",
                }}},
                faulted,
                injected,
                restored,
            ]),
            "secrets": mock.Mock(token_hex=mock.Mock(
                return_value="0123456789abcdef"
            )),
            "prometheus_target": mock.Mock(return_value={
                "scrape_url": "http://pod:8080/metrics",
                "last_scrape_utc": datetime.now(timezone.utc).isoformat(),
            }),
        }), self.assertRaisesRegex(DeploymentError, "not admitted exactly"):
            qualify_collector_outage(
                ["kubectl"], service="qualification-tritium",
                namespace="tritium-test", service_port=8080,
                service_monitor=monitor, prometheus_base_url="http://prometheus",
                token="token", model_id="tritium", prompt="Hello", timeout=10,
                request_timeout=1, revision="a" * 40, profile="compact-v1",
                manifest_blake3="c" * 64, release="1.1.0-rc.0",
                expected_startup=startup(),
            )
        self.assertEqual(len(commands), 2)
        self.assertIn('"op":"add"', commands[0][-1])
        self.assertIn('"op":"remove"', commands[1][-1])
        self.assertIn('"path":"/metadata/uid"', commands[0][-1])
        self.assertIn('"path":"/metadata/resourceVersion"', commands[0][-1])
        self.assertIn('"value":"0123456789abcdef"', commands[1][-1])

    def test_collector_fault_rejects_reserved_selector_without_patch(self):
        monitor = {
            "metadata": {"name": "qualification-tritium", "uid": "monitor-uid",
                         "resourceVersion": "10"},
            "spec": {"selector": {"matchLabels": {
                "app.kubernetes.io/name": "tritium",
                "app.kubernetes.io/instance": "qualification",
                "tritium-telemetry-fault": "owned-by-someone-else",
            }}},
        }
        run_mock = mock.Mock()
        with mock.patch.dict(qualify_collector_outage.__globals__, {
            "run": run_mock,
            "run_json": mock.Mock(return_value={
                "metadata": {"uid": "service-uid", "labels": {
                    "app.kubernetes.io/name": "tritium",
                }},
            }),
        }), self.assertRaisesRegex(DeploymentError, "selector precondition"):
            qualify_collector_outage(
                ["kubectl"], service="qualification-tritium",
                namespace="tritium-test", service_port=8080,
                service_monitor=monitor, prometheus_base_url="http://prometheus",
                token="token", model_id="tritium", prompt="Hello", timeout=10,
                request_timeout=1, revision="a" * 40, profile="compact-v1",
                manifest_blake3="c" * 64, release="1.1.0-rc.0",
                expected_startup=startup(),
            )
        run_mock.assert_not_called()

    def test_collector_fault_restores_after_admitted_patch_response_loss(self):
        original_selector = {
            "app.kubernetes.io/name": "tritium",
            "app.kubernetes.io/instance": "qualification",
        }
        monitor = {
            "metadata": {"name": "qualification-tritium", "uid": "monitor-uid",
                         "resourceVersion": "10"},
            "spec": {"selector": {"matchLabels": original_selector}},
        }
        injected = copy.deepcopy(monitor)
        injected["metadata"]["resourceVersion"] = "11"
        injected["spec"]["selector"]["matchLabels"] = {
            **original_selector, "tritium-telemetry-fault": "0123456789abcdef",
        }
        restored = copy.deepcopy(monitor)
        restored["metadata"]["resourceVersion"] = "12"
        commands = []

        def fake_run(command, _timeout):
            commands.append(command)
            if len(commands) == 1:
                raise DeploymentError("patch response lost")
            return ""

        with mock.patch.dict(qualify_collector_outage.__globals__, {
            "run": fake_run,
            "run_json": mock.Mock(side_effect=[
                {"metadata": {"uid": "service-uid", "labels": {}}},
                injected,
                restored,
            ]),
            "secrets": mock.Mock(token_hex=mock.Mock(
                return_value="0123456789abcdef"
            )),
            "prometheus_target": mock.Mock(return_value={
                "scrape_url": "http://pod:8080/metrics",
                "last_scrape_utc": datetime.now(timezone.utc).isoformat(),
            }),
        }), self.assertRaisesRegex(DeploymentError, "patch response lost"):
            qualify_collector_outage(
                ["kubectl"], service="qualification-tritium",
                namespace="tritium-test", service_port=8080,
                service_monitor=monitor, prometheus_base_url="http://prometheus",
                token="token", model_id="tritium", prompt="Hello", timeout=10,
                request_timeout=1, revision="a" * 40, profile="compact-v1",
                manifest_blake3="c" * 64, release="1.1.0-rc.0",
                expected_startup=startup(),
            )
        self.assertEqual(len(commands), 2)
        self.assertIn('"op":"remove"', commands[1][-1])

    def test_collector_fault_does_not_remove_concurrent_value(self):
        original_selector = {
            "app.kubernetes.io/name": "tritium",
            "app.kubernetes.io/instance": "qualification",
        }
        monitor = {
            "metadata": {"name": "qualification-tritium", "uid": "monitor-uid",
                         "resourceVersion": "10"},
            "spec": {"selector": {"matchLabels": original_selector}},
        }
        foreign = copy.deepcopy(monitor)
        foreign["metadata"]["resourceVersion"] = "11"
        foreign["spec"]["selector"]["matchLabels"] = {
            **original_selector, "tritium-telemetry-fault": "foreign-value",
        }
        run_mock = mock.Mock(side_effect=DeploymentError("patch conflict"))
        with mock.patch.dict(qualify_collector_outage.__globals__, {
            "run": run_mock,
            "run_json": mock.Mock(side_effect=[
                {"metadata": {"uid": "service-uid", "labels": {}}}, foreign,
            ]),
            "secrets": mock.Mock(token_hex=mock.Mock(
                return_value="0123456789abcdef"
            )),
            "prometheus_target": mock.Mock(return_value={
                "scrape_url": "http://pod:8080/metrics",
                "last_scrape_utc": datetime.now(timezone.utc).isoformat(),
            }),
        }), self.assertRaisesRegex(DeploymentError, "cleanup state is ambiguous"):
            qualify_collector_outage(
                ["kubectl"], service="qualification-tritium",
                namespace="tritium-test", service_port=8080,
                service_monitor=monitor, prometheus_base_url="http://prometheus",
                token="token", model_id="tritium", prompt="Hello", timeout=10,
                request_timeout=1, revision="a" * 40, profile="compact-v1",
                manifest_blake3="c" * 64, release="1.1.0-rc.0",
                expected_startup=startup(),
            )
        self.assertEqual(run_mock.call_count, 1)

    def test_collector_fault_cleanup_preserves_concurrent_selector(self):
        original_selector = {
            "app.kubernetes.io/name": "tritium",
            "app.kubernetes.io/instance": "qualification",
        }
        monitor = {
            "metadata": {"name": "qualification-tritium", "uid": "monitor-uid",
                         "resourceVersion": "10"},
            "spec": {"selector": {"matchLabels": original_selector}},
        }
        injected = copy.deepcopy(monitor)
        injected["metadata"]["resourceVersion"] = "11"
        injected["spec"]["selector"]["matchLabels"] = {
            **original_selector, "tritium-telemetry-fault": "0123456789abcdef",
            "concurrent-owner": "preserve-me",
        }
        cleaned = copy.deepcopy(injected)
        cleaned["metadata"]["resourceVersion"] = "12"
        del cleaned["spec"]["selector"]["matchLabels"][
            "tritium-telemetry-fault"
        ]
        commands = []

        def fake_run(command, _timeout):
            commands.append(command)
            if len(commands) == 1:
                raise DeploymentError("patch response lost")
            return ""

        with mock.patch.dict(qualify_collector_outage.__globals__, {
            "run": fake_run,
            "run_json": mock.Mock(side_effect=[
                {"metadata": {"uid": "service-uid", "labels": {}}},
                injected,
                cleaned,
            ]),
            "secrets": mock.Mock(token_hex=mock.Mock(
                return_value="0123456789abcdef"
            )),
            "prometheus_target": mock.Mock(return_value={
                "scrape_url": "http://pod:8080/metrics",
                "last_scrape_utc": datetime.now(timezone.utc).isoformat(),
            }),
        }), self.assertRaisesRegex(DeploymentError, "patch response lost"):
            qualify_collector_outage(
                ["kubectl"], service="qualification-tritium",
                namespace="tritium-test", service_port=8080,
                service_monitor=monitor, prometheus_base_url="http://prometheus",
                token="token", model_id="tritium", prompt="Hello", timeout=10,
                request_timeout=1, revision="a" * 40, profile="compact-v1",
                manifest_blake3="c" * 64, release="1.1.0-rc.0",
                expected_startup=startup(),
            )
        self.assertEqual(len(commands), 2)
        self.assertIn('"op":"remove"', commands[1][-1])
        self.assertEqual(
            cleaned["spec"]["selector"]["matchLabels"]["concurrent-owner"],
            "preserve-me",
        )

    def test_absence_proof_rejects_rbac_and_accepts_only_empty_success(self):
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(prove_absent.__globals__["subprocess"], "run",
                               return_value=completed):
            self.assertEqual(prove_absent(["kubectl", "get", "pvc/x"], 1)["status"],
                             "NotFound")
        completed.returncode = 1
        completed.stderr = "Error from server (Forbidden)"
        with mock.patch.object(prove_absent.__globals__["subprocess"], "run",
                               return_value=completed), self.assertRaisesRegex(
                                   DeploymentError, "exact empty success"
                               ):
            prove_absent(["kubectl", "get", "pvc/x"], 1)

    def test_resource_usage_sample_normalizes_cpu_and_memory(self):
        self.assertEqual(resource_quantity("12m", cpu=True), 12_000_000)
        self.assertEqual(resource_quantity("2Mi", cpu=False), 2 * 1024 * 1024)
        document = {"items": [{
            "metadata": {"name": "pod"},
            "containers": [
                {"usage": {"cpu": "12m", "memory": "2Mi"}},
                {"usage": {"cpu": "500u", "memory": "1Mi"}},
            ],
        }]}
        sample = resource_usage_sample(document)
        self.assertEqual(sample["cpu_nanocores"], 12_500_000)
        self.assertEqual(sample["memory_bytes"], 3 * 1024 * 1024)

    def test_receipt_validator_binds_both_candidate_artifacts_and_restart(self):
        with tempfile.TemporaryDirectory() as raw:
            chart, image, manifest, build, candidate = candidate_inputs(raw)
            value = receipt(chart, image, candidate)
            validate(value, chart, image, manifest, build, candidate)
            value["workload"]["restarted"]["pods"][0]["uid"] = "pod-old-uid"
            del value["receipt_id"]
            value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
            with self.assertRaisesRegex(DeploymentError, "retains old pod UID"):
                validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_binds_missing_secret_startup_failure(self):
        for target in ("patch", "reason", "failure_uid", "transition", "uid"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as raw:
                chart, image, manifest, build, candidate = candidate_inputs(raw)
                value = receipt(chart, image, candidate)
                fault = value["workload"]["missing_secret_startup"]
                if target == "patch":
                    fault["fault_patch_sha256"] = "0" * 64
                elif target == "reason":
                    fault["failure"]["reason"] = "ImagePullBackOff"
                elif target == "failure_uid":
                    fault["recovered"]["pods"][0]["uid"] = fault["failure"]["pod_uid"]
                elif target == "uid":
                    fault["deployment_uid"] = "foreign-deployment-uid"
                    fault_patch = [
                        {"op": "test", "path": "/metadata/uid",
                         "value": fault["deployment_uid"]},
                        {"op": "test", "path": "/metadata/resourceVersion",
                         "value": fault["baseline_resource_version"]},
                    ]
                    restore_patch = [{
                        "op": "test", "path": "/metadata/uid",
                        "value": fault["deployment_uid"],
                    }]
                    for binding in fault["bindings"]:
                        fault_patch.extend([
                            {"op": "test", "path": binding["path"],
                             "value": fault["secret_name"]},
                            {"op": "replace", "path": binding["path"],
                             "value": fault["missing_secret"]},
                        ])
                        restore_patch.extend([
                            {"op": "test", "path": binding["path"],
                             "value": fault["missing_secret"]},
                            {"op": "replace", "path": binding["path"],
                             "value": fault["secret_name"]},
                        ])
                    fault["fault_patch_sha256"] = hashlib.sha256(
                        canonical(fault_patch).decode().strip().encode()
                    ).hexdigest()
                    fault["restore_patch_sha256"] = hashlib.sha256(
                        canonical(restore_patch).decode().strip().encode()
                    ).hexdigest()
                else:
                    fault["transitions"][1], fault["transitions"][2] = (
                        fault["transitions"][2], fault["transitions"][1]
                    )
                del value["receipt_id"]
                value["receipt_id"] = "sha256:" + hashlib.sha256(
                    canonical(value)
                ).hexdigest()
                with self.assertRaisesRegex(
                    DeploymentError,
                    "missing-Secret patch|missing-Secret failure|survived|"
                    "transition sequence|identity or bounds",
                ):
                    validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_binds_startup_argument_failures(self):
        for scenario in ("invalid_config", "unavailable_backend"):
            for target in ("patch", "error", "owner", "timing"):
                with (self.subTest(scenario=scenario, target=target),
                      tempfile.TemporaryDirectory() as raw):
                    chart, image, manifest, build, candidate = candidate_inputs(raw)
                    value = receipt(chart, image, candidate)
                    evidence = value["workload"][f"{scenario}_startup"]
                    if target == "patch":
                        evidence["fault_patch_sha256"] = "0" * 64
                    elif target == "error":
                        evidence["failure"]["error_line"] = "Error: unrelated"
                    elif target == "owner":
                        evidence["failure"]["replica_set_owner"]["uid"] = "foreign-uid"
                    else:
                        evidence["completed_elapsed_ms"] = value["duration_ms"] + 1
                    del value["receipt_id"]
                    value["receipt_id"] = "sha256:" + hashlib.sha256(
                        canonical(value)
                    ).hexdigest()
                    with self.assertRaisesRegex(
                        DeploymentError,
                        f"{scenario} patch|{scenario} startup error|"
                        f"{scenario} process failure|{scenario} identity or bounds",
                    ):
                        validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_requires_clean_post_restart_recovery(self):
        with tempfile.TemporaryDirectory() as raw:
            chart, image, manifest, build, candidate = candidate_inputs(raw)
            value = receipt(chart, image, candidate)
            value["workload"]["restart_recovery"]["metrics"]["values"][
                "tritium_backend_faulted"
            ] = 1.0
            del value["receipt_id"]
            value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
            with self.assertRaisesRegex(DeploymentError, "restart metrics"):
                validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_binds_watchdog_process_replacement(self):
        with tempfile.TemporaryDirectory() as raw:
            chart, image, manifest, build, candidate = candidate_inputs(raw)
            value = receipt(chart, image, candidate)
            value["workload"]["watchdog_replacement"]["last_exit_code"] = 143
            del value["receipt_id"]
            value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
            with self.assertRaisesRegex(DeploymentError, "watchdog replacement"):
                validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_binds_exact_watchdog_fault_command(self):
        with tempfile.TemporaryDirectory() as raw:
            chart, image, manifest, build, candidate = candidate_inputs(raw)
            value = receipt(chart, image, candidate)
            value["workload"]["watchdog_replacement"]["fault_command_sha256"] = "0" * 64
            del value["receipt_id"]
            value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
            with self.assertRaisesRegex(DeploymentError, "fault command differs"):
                validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_rejects_over_budget_watchdog_replacement(self):
        with tempfile.TemporaryDirectory() as raw:
            chart, image, manifest, build, candidate = candidate_inputs(raw)
            value = receipt(chart, image, candidate)
            value["workload"]["watchdog_replacement"]["replacement_ms"] = 98001
            del value["receipt_id"]
            value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
            with self.assertRaisesRegex(DeploymentError, "exceeded its budget"):
                validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_rejects_coordinated_watchdog_policy_tamper(self):
        with tempfile.TemporaryDirectory() as raw:
            chart, image, manifest, build, candidate = candidate_inputs(raw)
            value = receipt(chart, image, candidate)
            policy = value["workload"]["watchdog_replacement"]["watchdog"]
            policy["period_seconds"] = 60
            policy["budget_ms"] = 248000
            del value["receipt_id"]
            value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
            with self.assertRaisesRegex(DeploymentError, "policy differs"):
                validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_binds_artifact_volume_fault_patch(self):
        with tempfile.TemporaryDirectory() as raw:
            chart, image, manifest, build, candidate = candidate_inputs(raw)
            value = receipt(chart, image, candidate)
            value["workload"]["artifact_volume_loss"]["fault_patch_sha256"] = "0" * 64
            del value["receipt_id"]
            value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
            with self.assertRaisesRegex(DeploymentError, "fault patch differs"):
                validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_rejects_artifact_observation_over_budget(self):
        with tempfile.TemporaryDirectory() as raw:
            chart, image, manifest, build, candidate = candidate_inputs(raw)
            value = receipt(chart, image, candidate)
            value["workload"]["artifact_volume_loss"]["observation_ms"] = 120001
            del value["receipt_id"]
            value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
            with self.assertRaisesRegex(DeploymentError, "artifact-volume evidence"):
                validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_recomputes_artifact_resource_high_water(self):
        with tempfile.TemporaryDirectory() as raw:
            chart, image, manifest, build, candidate = candidate_inputs(raw)
            value = receipt(chart, image, candidate)
            value["workload"]["artifact_volume_loss"]["resources"]["high_water"][
                "memory_bytes"
            ] += 1
            del value["receipt_id"]
            value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
            with self.assertRaisesRegex(DeploymentError, "high-water evidence"):
                validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_binds_artifact_request_and_transition_order(self):
        for target in ("request", "boolean_request", "transition"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as raw:
                chart, image, manifest, build, candidate = candidate_inputs(raw)
                value = receipt(chart, image, candidate)
                artifact = value["workload"]["artifact_volume_loss"]
                if target == "request":
                    artifact["request"]["max_tokens"] = 2
                elif target == "boolean_request":
                    request = artifact["request"]
                    request["temperature"] = False
                    request["max_tokens"] = True
                    descriptor = {
                        key: value for key, value in request.items()
                        if key != "descriptor_sha256"
                    }
                    request["descriptor_sha256"] = hashlib.sha256(
                        canonical(descriptor)
                    ).hexdigest()
                else:
                    artifact["transitions"][2], artifact["transitions"][3] = (
                        artifact["transitions"][3], artifact["transitions"][2]
                    )
                del value["receipt_id"]
                value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
                with self.assertRaisesRegex(
                    DeploymentError, "request evidence|transition sequence"
                ):
                    validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_binds_real_oom_termination_and_patch(self):
        for target in ("reason", "patch", "lineage"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as raw:
                chart, image, manifest, build, candidate = candidate_inputs(raw)
                value = receipt(chart, image, candidate)
                oom = value["workload"]["memory_oom_recovery"]
                if target == "reason":
                    oom["terminated"]["reason"] = "Error"
                elif target == "patch":
                    oom["fault_patch_sha256"] = "0" * 64
                else:
                    oom["replica_set"]["deployment_uid"] = "foreign-uid"
                del value["receipt_id"]
                value["receipt_id"] = "sha256:" + hashlib.sha256(
                    canonical(value)
                ).hexdigest()
                with self.assertRaisesRegex(
                    DeploymentError,
                    "OOM termination evidence|OOM fault patch|OOM ReplicaSet lineage",
                ):
                    validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_recomputes_oom_high_water(self):
        with tempfile.TemporaryDirectory() as raw:
            chart, image, manifest, build, candidate = candidate_inputs(raw)
            value = receipt(chart, image, candidate)
            value["workload"]["memory_oom_recovery"]["resources"]["high_water"][
                "memory_bytes"
            ] += 1
            del value["receipt_id"]
            value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
            with self.assertRaisesRegex(DeploymentError, "OOM high-water evidence"):
                validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_binds_metrics_flood_concurrency_and_responses(self):
        for target in ("peak", "response"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as raw:
                chart, image, manifest, build, candidate = candidate_inputs(raw)
                value = receipt(chart, image, candidate)
                flood = value["workload"]["metrics_scrape_flood"]
                if target == "peak":
                    flood["peak_client_tasks"] = 7
                else:
                    flood["response_sha256s"].pop()
                del value["receipt_id"]
                value["receipt_id"] = "sha256:" + hashlib.sha256(
                    canonical(value)
                ).hexdigest()
                with self.assertRaisesRegex(
                    DeploymentError, "metrics-flood bounds|metrics-flood responses"
                ):
                    validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_binds_collector_patch_and_recovery_causality(self):
        for target in ("patch", "recovery"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as raw:
                chart, image, manifest, build, candidate = candidate_inputs(raw)
                value = receipt(chart, image, candidate)
                collector = value["workload"]["collector_outage"]
                if target == "patch":
                    collector["fault_patch_sha256"] = "0" * 64
                else:
                    collector["recovered_target"]["last_scrape_utc"] = (
                        "2026-07-21T12:01:00+00:00"
                    )
                del value["receipt_id"]
                value["receipt_id"] = "sha256:" + hashlib.sha256(
                    canonical(value)
                ).hexdigest()
                with self.assertRaisesRegex(
                    DeploymentError, "collector-outage patch|collector transition causality"
                ):
                    validate(value, chart, image, manifest, build, candidate)

    def test_receipt_validator_rejects_chart_or_image_drift(self):
        for target in ("chart", "image"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as raw:
                chart, image, manifest, build, candidate = candidate_inputs(raw)
                value = receipt(chart, image, candidate)
                (chart if target == "chart" else image).write_bytes(b"drift")
                with self.assertRaisesRegex(DeploymentError, "candidate"):
                    validate(value, chart, image, manifest, build, candidate)

    def test_offline_validator_reestablishes_manifest_and_oci_lineage(self):
        with tempfile.TemporaryDirectory() as raw:
            chart, image, manifest, build, candidate = candidate_inputs(raw)
            value = receipt(chart, image, candidate)
            admitted_manifest = dict(value["manifest"])
            admitted_oci = {
                "image_manifest_digest": value["image"].rpartition("@")[2],
                "release": value["release"], "source_revision": value["source_revision"],
                "flavor": value["flavor"],
            }
            value["manifest"]["blake3"] = "9" * 64
            value["workload"]["startup_receipt"]["manifest_package_id"] = "9" * 64
            value["workload"]["restart_startup_receipt"]["manifest_package_id"] = "9" * 64
            del value["receipt_id"]
            value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
            with mock.patch.dict(validate_receipt.__globals__, {
                "validate_oci_archive": mock.Mock(return_value=admitted_oci),
                "manifest_identity": mock.Mock(return_value=admitted_manifest),
            }), self.assertRaisesRegex(DeploymentError, "candidate bytes"):
                validate_receipt(
                    value, chart_path=chart, image_path=image, manifest_path=manifest,
                    build_receipt=build, package_candidate=candidate,
                    digest_tool="tritium", revision="a" * 40,
                    release="1.1.0-rc.0",
                )

    def test_cuda_receipt_binds_gpu_uuid_driver_runtime_and_host_node(self):
        with tempfile.TemporaryDirectory() as raw:
            chart, image, manifest, build, candidate = candidate_inputs(raw, "cuda")
            value = receipt(chart, image, candidate, "cuda")
            value["cluster"]["cuda_node"] = {
                "node_name": "node-1",
                "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
                "gpu_name": "NVIDIA Test GPU",
                "driver_version": "999.0",
                "cuda_runtime": "13.0.1",
                "probe_image": next(iter(MODULE["CUDA_PROBE_IMAGES"])),
                "probe_pod_uid": "probe-pod-uid",
                "output_sha256": "8" * 64,
            }
            del value["receipt_id"]
            value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
            validate(value, chart, image, manifest, build, candidate)
            value["cluster"]["cuda_node"]["gpu_uuid"] = (
                "GPU-00000000-1234-1234-1234-123456789abc"
            )
            del value["receipt_id"]
            value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
            with self.assertRaisesRegex(DeploymentError, "deployed hardware"):
                validate(value, chart, image, manifest, build, candidate)


if __name__ == "__main__":
    unittest.main()
