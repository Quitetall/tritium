from __future__ import annotations

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
validate_deployment = MODULE["validate_deployment"]
validate_receipt = MODULE["validate_receipt"]
metrics_snapshot = MODULE["metrics_snapshot"]
validate_helm_history = MODULE["validate_helm_history"]
scale_snapshot = MODULE["scale_snapshot"]
prometheus_url = MODULE["prometheus_url"]
deployment_update_identity = MODULE["deployment_update_identity"]
validate_scale_contract = MODULE["validate_scale_contract"]


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
            "qualification_lock_uid": "lock-uid",
            "initial": {"pods": [{"name": "pod-old", "uid": "pod-old-uid",
                                    "node": "node-1", "restarts": 0}]},
            "restarted": {"pods": [{"name": "pod-new", "uid": "pod-new-uid",
                                      "node": "node-1", "restarts": 0}]},
            "updated": {"pods": [{"name": "pod-updated", "uid": "pod-updated-uid",
                                    "node": "node-1", "restarts": 0}]},
            "startup_receipt": startup_receipt,
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
