from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import runpy
import tempfile
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
pending_artifact_volume_failure = MODULE["pending_artifact_volume_failure"]
pvc_identity = MODULE["pvc_identity"]
qualify_artifact_volume_loss = MODULE["qualify_artifact_volume_loss"]
prove_absent = MODULE["prove_absent"]
resource_quantity = MODULE["resource_quantity"]
resource_usage_sample = MODULE["resource_usage_sample"]
request_evidence = MODULE["request_evidence"]


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
        "started_at_utc": "2026-07-21T12:00:00+00:00", "duration_ms": 100.0,
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
            "restart_startup_receipt": dict(startup_receipt),
            "update_startup_receipt": dict(startup_receipt),
            "update_strategy": "Recreate" if flavor == "cuda" else "RollingUpdate",
            "update_config": {
                "before": {"generation": 1, "rate_limit_burst": 8},
                "after": {"generation": 2, "rate_limit_burst": 9},
            },
            "rollback_startup_receipt": dict(startup_receipt),
            "metrics": {"sha256": "6" * 64, "values": {
                "tritium_chat_requests_total": 1.0,
                "tritium_tokens_out_total": 1.0,
                "tritium_worker_alive": 1.0,
                "tritium_backend_faults_total": 0.0,
                "tritium_backend_faulted": 0.0,
            }},
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
