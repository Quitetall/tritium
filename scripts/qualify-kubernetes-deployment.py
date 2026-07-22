#!/usr/bin/env python3
"""Install, exercise, restart, fail, roll back, and seal one Tritium Helm release."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


RUNTIME = runpy.run_path(Path(__file__).with_name("qualify-oci-runtime.py"))
validate_ready = RUNTIME["validate_ready"]
manifest_identity = RUNTIME["manifest_identity"]
OCI_ARCHIVE = runpy.run_path(Path(__file__).with_name("verify-oci-archive.py"))
validate_oci_archive = OCI_ARCHIVE["validate"]

SCHEMA = "tritium.kubernetes-deployment-qualification.v1"
HEX = frozenset("0123456789abcdef")
SAFE_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?")
SAFE_SECRET_KEY = re.compile(r"[A-Za-z0-9](?:[-._A-Za-z0-9]{0,251}[A-Za-z0-9])?")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
RELEASE = re.compile(r"1\.1\.0-rc\.(0|[1-9][0-9]*)")
GPU_UUID = re.compile(
    r"cuda:[0-9]+:GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
CUDA_PROBE_IMAGES = {
    "docker.io/nvidia/cuda:13.0.1-devel-ubuntu22.04@sha256:"
    "bb3de902bc1b522231cffea98a4d25a16ddb9fc8685a958b48d83036f46fd0c2": "13.0.1",
}
CHECKS = (
    "namespace-preflight", "secret-binding", "pvc-preflight", "helm-install",
    "rollout-ready", "production-readiness", "buffered-generation", "metrics",
    "pod-restart", "failed-upgrade", "atomic-rollback", "rollback-readiness",
    "rollback-generation", "release-cleanup",
)
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class DeploymentError(ValueError):
    """Kubernetes deployment evidence is malformed, incomplete, or non-green."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, kind: str) -> dict[str, Any]:
    return {
        "kind": kind, "name": path.name, "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def ordinary(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise DeploymentError(f"{label} must be an ordinary file")
    return path.resolve(strict=True)


def exact_hex(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(c not in HEX for c in value):
        raise DeploymentError(f"{label} must be {length} lowercase hexadecimal characters")
    return value


def bind_chart_to_candidate(candidate: Path, chart: Path) -> dict[str, Any]:
    try:
        document = json.loads(candidate.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeploymentError("package candidate must contain UTF-8 JSON") from error
    artifacts = document.get("artifacts") if isinstance(document, dict) else None
    if not isinstance(artifacts, list):
        raise DeploymentError("package candidate artifacts are malformed")
    chart_sha = sha256(chart)
    matches = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("kind") != "helm-chart":
            continue
        identity = artifact.get("identity")
        if not isinstance(identity, dict):
            continue
        if (identity.get("bytes"), identity.get("sha256")) == (
            chart.stat().st_size, chart_sha,
        ):
            matches.append(artifact)
    if len(matches) != 1:
        raise DeploymentError("package candidate does not bind exactly one Helm chart")
    artifact = matches[0]
    if not isinstance(artifact.get("id"), str) or not artifact["id"]:
        raise DeploymentError("package candidate Helm chart ID is malformed")
    return {"artifact_id": artifact["id"], "candidate_sha256": sha256(candidate)}


def executable(value: str, label: str) -> Path:
    found = shutil.which(value)
    if found is None:
        raise DeploymentError(f"{label} executable is unavailable")
    resolved = Path(found).resolve(strict=True)
    if not resolved.is_file():
        raise DeploymentError(f"{label} executable must resolve to a file")
    return resolved


def run(command: list[str], timeout: float = 120.0) -> str:
    try:
        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DeploymentError(f"command failed: {command[0]}: {error}") from error
    if result.returncode != 0:
        raise DeploymentError(
            f"command failed ({result.returncode}): {' '.join(command)}: "
            f"{result.stderr.strip()[-2000:]}"
        )
    if len(result.stdout.encode()) > MAX_RESPONSE_BYTES:
        raise DeploymentError("command output exceeds byte limit")
    return result.stdout.strip()


def run_json(command: list[str], timeout: float = 120.0) -> dict[str, Any]:
    try:
        value = json.loads(run(command, timeout))
    except json.JSONDecodeError as error:
        raise DeploymentError(f"command returned malformed JSON: {command[0]}") from error
    if not isinstance(value, dict):
        raise DeploymentError(f"command did not return a JSON object: {command[0]}")
    return value


def run_json_value(command: list[str], timeout: float = 120.0) -> Any:
    try:
        return json.loads(run(command, timeout))
    except json.JSONDecodeError as error:
        raise DeploymentError(f"command returned malformed JSON: {command[0]}") from error


def expect_failure(command: list[str], timeout: float) -> str:
    try:
        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DeploymentError(f"expected-failure command could not run: {error}") from error
    if result.returncode == 0:
        raise DeploymentError("invalid Helm upgrade unexpectedly succeeded")
    output = result.stdout + result.stderr
    if len(output.encode()) > MAX_RESPONSE_BYTES:
        raise DeploymentError("failed-upgrade output exceeds byte limit")
    return hashlib.sha256(output.encode()).hexdigest()


def request(url: str, token: str, *, body: dict[str, Any] | None = None,
            timeout: float = 30.0) -> bytes:
    payload = None if body is None else canonical(body)
    req = urllib.request.Request(url, data=payload)
    req.add_header("Authorization", f"Bearer {token}")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = response.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, urllib.error.URLError) as error:
        raise DeploymentError(f"request failed: {url}: {error}") from error
    if len(result) > MAX_RESPONSE_BYTES:
        raise DeploymentError(f"response exceeds byte limit: {url}")
    return result


def request_json(url: str, token: str, *, body: dict[str, Any] | None = None,
                 timeout: float = 30.0) -> dict[str, Any]:
    try:
        value = json.loads(request(url, token, body=body, timeout=timeout))
    except json.JSONDecodeError as error:
        raise DeploymentError(f"response is malformed JSON: {url}") from error
    if not isinstance(value, dict):
        raise DeploymentError(f"response is not a JSON object: {url}")
    return value


def free_port() -> int:
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def kube(kubectl: Path, context: str, namespace: str) -> list[str]:
    return [str(kubectl), "--context", context, "--namespace", namespace]


def helm(helm_bin: Path, context: str, namespace: str) -> list[str]:
    return [str(helm_bin), "--kube-context", context, "--namespace", namespace]


def pod_snapshot(document: dict[str, Any], flavor: str) -> dict[str, Any]:
    items = document.get("items")
    if not isinstance(items, list) or len(items) != 1:
        raise DeploymentError("deployment must have exactly one pod")
    pods = []
    for item in items:
        if not isinstance(item, dict):
            raise DeploymentError("pod listing is malformed")
        metadata = item.get("metadata", {})
        spec = item.get("spec", {})
        status = item.get("status", {})
        conditions = status.get("conditions", [])
        ready = any(
            isinstance(condition, dict) and condition.get("type") == "Ready"
            and condition.get("status") == "True"
            for condition in conditions
        )
        containers = status.get("containerStatuses", [])
        if not ready or not isinstance(containers, list) or len(containers) != 2 or any(
            not isinstance(container, dict) for container in containers
        ):
            raise DeploymentError("pod is not fully ready")
        restart_counts = [container.get("restartCount") for container in containers]
        if any(type(count) is not int or count < 0 for count in restart_counts):
            raise DeploymentError("pod restart accounting is malformed")
        restarts = sum(restart_counts)
        node = spec.get("nodeName")
        uid = metadata.get("uid")
        name = metadata.get("name")
        if not all(isinstance(value, str) and value for value in (node, uid, name)):
            raise DeploymentError("pod identity is incomplete")
        pod_containers = spec.get("containers")
        if not isinstance(pod_containers, list) or len(pod_containers) != 2 or any(
            not isinstance(container, dict) for container in pod_containers
        ):
            raise DeploymentError("pod container specification differs")
        limits = pod_containers[0].get("resources", {}).get("limits", {})
        if flavor == "cuda" and limits.get("nvidia.com/gpu") != "1":
            raise DeploymentError("CUDA pod does not bind one NVIDIA GPU")
        pods.append({"name": name, "uid": uid, "node": node, "restarts": restarts})
    return {"pods": sorted(pods, key=lambda item: item["uid"])}


def validate_receipt_snapshot(snapshot: Any, label: str) -> dict[str, Any]:
    if not isinstance(snapshot, dict) or set(snapshot) != {"pods"}:
        raise DeploymentError(f"{label} pod snapshot fields differ")
    pods = snapshot.get("pods")
    if not isinstance(pods, list) or len(pods) != 1:
        raise DeploymentError(f"{label} must contain exactly one pod")
    uids = set()
    for pod in pods:
        if not isinstance(pod, dict) or set(pod) != {"name", "uid", "node", "restarts"}:
            raise DeploymentError(f"{label} pod fields differ")
        if any(not isinstance(pod.get(key), str) or not pod[key] for key in ("name", "uid", "node")):
            raise DeploymentError(f"{label} pod identity is malformed")
        if type(pod.get("restarts")) is not int or pod["restarts"] < 0:
            raise DeploymentError(f"{label} pod restart count is malformed")
        if pod["uid"] in uids:
            raise DeploymentError(f"{label} pod UIDs are duplicated")
        uids.add(pod["uid"])
    if pods != sorted(pods, key=lambda item: item["uid"]):
        raise DeploymentError(f"{label} pod snapshot is not canonical")
    return snapshot


def validate_deployment(document: dict[str, Any], *, image: str,
                        manifest_sha256: str, flavor: str) -> str:
    metadata = document.get("metadata", {})
    spec = document.get("spec", {})
    status = document.get("status", {})
    annotations = spec.get("template", {}).get("metadata", {}).get("annotations", {})
    containers = spec.get("template", {}).get("spec", {}).get("containers", [])
    if annotations.get("tritium.ai/image-digest") != image.rpartition("@")[2]:
        raise DeploymentError("deployment image annotation differs")
    if annotations.get("tritium.ai/manifest-sha256") != manifest_sha256:
        raise DeploymentError("deployment manifest annotation differs")
    if (
        not isinstance(containers, list)
        or len(containers) != 2
        or any(not isinstance(container, dict) for container in containers)
        or [container.get("name") for container in containers]
        != ["tritium", "authenticated-probe"]
        or containers[0].get("image") != image
    ):
        raise DeploymentError("deployment container image differs")
    replicas = spec.get("replicas")
    if type(replicas) is not int or replicas < 1 or status.get("readyReplicas") != replicas:
        raise DeploymentError("deployment replica readiness differs")
    uid = metadata.get("uid")
    if not isinstance(uid, str) or not uid:
        raise DeploymentError("deployment UID is absent")
    strategy = spec.get("strategy", {}).get("type")
    expected_strategy = "Recreate" if flavor == "cuda" else "RollingUpdate"
    if strategy != expected_strategy:
        raise DeploymentError("deployment strategy differs")
    return uid


def port_forward(command: list[str], deadline: float, url: str, token: str) -> tuple[subprocess.Popen, dict]:
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
        )
    except OSError as error:
        raise DeploymentError(f"port-forward failed to start: {error}") from error
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise DeploymentError("port-forward exited before readiness")
            try:
                ready = request_json(url + "/readyz", token, timeout=5)
                return process, ready
            except DeploymentError:
                time.sleep(1)
        raise DeploymentError("port-forward readiness deadline expired")
    except BaseException:
        stop_forward(process)
        raise


def stop_forward(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def node_snapshot(document: dict[str, Any]) -> dict[str, str]:
    metadata = document.get("metadata", {})
    status = document.get("status", {})
    node_info = status.get("nodeInfo", {})
    result = {
        "name": metadata.get("name"),
        "uid": metadata.get("uid"),
        "provider_id": document.get("spec", {}).get("providerID"),
        "kernel_version": node_info.get("kernelVersion"),
        "os_image": node_info.get("osImage"),
        "architecture": node_info.get("architecture"),
        "container_runtime": node_info.get("containerRuntimeVersion"),
    }
    if any(not isinstance(value, str) or not value for value in result.values()):
        raise DeploymentError("Kubernetes node identity is incomplete")
    return result


def validate_node_snapshot(value: Any) -> dict[str, str]:
    fields = {
        "name", "uid", "provider_id", "kernel_version", "os_image",
        "architecture", "container_runtime",
    }
    if not isinstance(value, dict) or set(value) != fields or any(
        not isinstance(value.get(key), str) or not value[key] for key in fields
    ):
        raise DeploymentError("deployment node identity is malformed")
    return value


def collect_cuda_node_evidence(kubectl_base: list[str], *, node: str, image: str,
                               startup: dict[str, Any], timeout: float) -> dict[str, str]:
    cuda_runtime = CUDA_PROBE_IMAGES.get(image)
    if cuda_runtime is None:
        raise DeploymentError("CUDA probe image is not admitted by release policy")
    name = "tritium-gpu-evidence-" + secrets.token_hex(4)
    overrides = {
        "apiVersion": "v1", "spec": {"nodeName": node,
            "automountServiceAccountToken": False, "restartPolicy": "Never",
            "containers": [{"name": "probe", "image": image,
                "imagePullPolicy": "IfNotPresent",
                "resources": {"limits": {"nvidia.com/gpu": "1"}},
                "securityContext": {"allowPrivilegeEscalation": False,
                    "runAsNonRoot": True, "runAsUser": 65532,
                    "readOnlyRootFilesystem": True,
                    "capabilities": {"drop": ["ALL"]},
                    "seccompProfile": {"type": "RuntimeDefault"}},
                "command": ["/bin/sh", "-ceu"],
                "args": ["nvidia-smi --query-gpu=uuid,name,driver_version "
                         "--format=csv,noheader"]}]}}
    attempted = False
    try:
        attempted = True
        run(kubectl_base + ["run", name, f"--image={image}", "--restart=Never",
                            f"--overrides={canonical(overrides).decode().strip()}"], timeout)
        run(kubectl_base + ["wait", "--for=jsonpath={.status.phase}=Succeeded",
                            f"pod/{name}", f"--timeout={math.ceil(timeout)}s"], timeout + 30)
        pod = run_json(kubectl_base + ["get", f"pod/{name}", "-o", "json"])
        metadata = pod.get("metadata", {})
        containers = pod.get("spec", {}).get("containers", [])
        if (pod.get("spec", {}).get("nodeName") != node or len(containers) != 1
                or containers[0].get("image") != image
                or containers[0].get("resources", {}).get("limits", {}).get("nvidia.com/gpu") != "1"):
            raise DeploymentError("CUDA probe pod identity differs")
        uid = metadata.get("uid")
        if not isinstance(uid, str) or not uid:
            raise DeploymentError("CUDA probe pod UID is absent")
        output = run(kubectl_base + ["logs", f"pod/{name}", "--container=probe"], timeout)
        if len(output.encode()) > 1024 * 1024:
            raise DeploymentError("CUDA probe output exceeds byte limit")
        rows = [row.split(", ", 2) for row in output.splitlines() if row]
        if len(rows) != 1 or len(rows[0]) != 3:
            raise DeploymentError("CUDA probe must report exactly one GPU")
        value = {"gpu_uuid": rows[0][0], "gpu_name": rows[0][1],
                 "driver_version": rows[0][2], "cuda_runtime": cuda_runtime}
    finally:
        active_error = sys.exc_info()[0] is not None
        if attempted:
            try:
                run(kubectl_base + ["delete", f"pod/{name}", "--wait=true",
                                    "--ignore-not-found=true"], timeout)
            except DeploymentError:
                if not active_error:
                    raise
    fields = {"gpu_uuid", "gpu_name", "driver_version", "cuda_runtime"}
    if not isinstance(value, dict) or set(value) != fields or any(
        not isinstance(value.get(key), str) or not value[key] for key in fields
    ):
        raise DeploymentError("CUDA probe output fields differ")
    physical = startup.get("physical_device_id")
    if GPU_UUID.fullmatch(physical or "") is None or physical.rsplit(":", 1)[-1] != value["gpu_uuid"]:
        raise DeploymentError("CUDA probe does not bind the startup GPU UUID")
    return {**value, "node_name": node, "probe_image": image, "probe_pod_uid": uid,
            "output_sha256": hashlib.sha256(output.encode()).hexdigest()}


def metrics_snapshot(metrics: str) -> dict[str, Any]:
    values = {}
    for name in ("tritium_chat_requests_total", "tritium_tokens_out_total",
                 "tritium_worker_alive"):
        match = re.search(rf"(?m)^{name} ([0-9]+(?:\.[0-9]+)?)$", metrics)
        if match is None:
            raise DeploymentError(f"cluster metrics lack {name}")
        values[name] = float(match.group(1))
    if values["tritium_chat_requests_total"] < 1 or values["tritium_tokens_out_total"] < 1:
        raise DeploymentError("cluster metrics did not observe generation")
    if values["tritium_worker_alive"] != 1:
        raise DeploymentError("cluster metrics report a dead worker")
    return {"sha256": hashlib.sha256(metrics.encode()).hexdigest(), "values": values}


def deployed_revision(value: Any) -> int:
    if not isinstance(value, list) or not value:
        raise DeploymentError("Helm deployed history is absent")
    latest = value[-1]
    if (not isinstance(latest, dict) or type(latest.get("revision")) is not int
            or str(latest.get("status", "")).lower() != "deployed"):
        raise DeploymentError("Helm latest revision is not deployed")
    return latest["revision"]


def validate_helm_history(value: Any, prior_revision: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) < 3:
        raise DeploymentError("Helm rollback history is incomplete")
    revisions = []
    statuses = []
    for entry in value:
        if not isinstance(entry, dict) or type(entry.get("revision")) is not int:
            raise DeploymentError("Helm rollback history is malformed")
        revisions.append(entry["revision"])
        status = entry.get("status")
        if not isinstance(status, str):
            raise DeploymentError("Helm rollback history status is malformed")
        statuses.append(status.lower())
    if revisions != sorted(revisions) or len(set(revisions)) != len(revisions):
        raise DeploymentError("Helm rollback revisions are not strictly ordered")
    if (
        revisions[-3:] != [prior_revision, prior_revision + 1, prior_revision + 2]
        or statuses[-3:] != ["superseded", "failed", "deployed"]
    ):
        raise DeploymentError("Helm history does not prove the exact failed upgrade and rollback")
    return value


def atomic_create(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise DeploymentError("refusing to overwrite output")
    parent = path.parent.resolve(strict=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    if args.flavor not in {"cpu", "cuda"} or args.profile not in {
        "compact-v1", "near-lossless-v1"
    }:
        raise DeploymentError("flavor or profile is not admitted")
    if RELEASE.fullmatch(args.release) is None or RUN_ID.fullmatch(args.run_id) is None:
        raise DeploymentError("release or run ID is malformed")
    for label, value in (
        ("namespace", args.namespace), ("release name", args.release_name),
        ("source PVC", args.source_pvc), ("auth secret", args.auth_secret),
    ):
        if SAFE_NAME.fullmatch(value) is None:
            raise DeploymentError(f"{label} is not a safe Kubernetes name")
    if len(args.release_name) > 55:
        raise DeploymentError("release name is too long for the Tritium chart fullname")
    if SAFE_SECRET_KEY.fullmatch(args.auth_key) is None:
        raise DeploymentError("auth key is not a safe Secret key")
    if not args.context or len(args.context) > 256 or "\x00" in args.context:
        raise DeploymentError("Kubernetes context is malformed")
    revision = exact_hex(args.source_revision, 40, "source revision")
    manifest_sha = exact_hex(args.manifest_sha256, 64, "bundle manifest SHA-256")
    repository, separator, image_digest = args.image.rpartition("@")
    if separator != "@" or not repository or not image_digest.startswith("sha256:"):
        raise DeploymentError("image must be an exact repository digest")
    exact_hex(image_digest[7:], 64, "image digest")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 7200:
        raise DeploymentError("timeout must be finite and in (0, 7200]")
    if not math.isfinite(args.request_timeout) or not 0 < args.request_timeout <= 1800:
        raise DeploymentError("request timeout must be finite and in (0, 1800]")
    if args.flavor == "cuda" and args.cuda_probe_image is None:
        raise DeploymentError("CUDA qualification requires a pinned probe image")
    if args.flavor == "cpu" and args.cuda_probe_image is not None:
        raise DeploymentError("CPU qualification cannot accept a CUDA probe image")
    chart = ordinary(args.chart_archive, "Helm chart archive")
    image_archive = ordinary(args.image_archive, "OCI image archive")
    build_receipt = ordinary(args.build_receipt, "OCI build receipt")
    package_candidate = ordinary(args.package_candidate, "package candidate")
    initial_files = {
        "chart": file_record(chart, "helm-chart"),
        "image": file_record(image_archive, "oci-image"),
        "build_receipt": file_record(build_receipt, "oci-build-receipt"),
        "package_candidate": file_record(package_candidate, "release-candidate"),
    }
    chart_binding = bind_chart_to_candidate(package_candidate, chart)
    token_file = ordinary(args.auth_token_file, "auth token file")
    token = token_file.read_text(encoding="utf-8").strip()
    if not token or len(token.encode()) > 4096:
        raise DeploymentError("auth token is empty or oversized")
    manifest = ordinary(args.bundle_manifest, "bundle manifest")
    initial_files["manifest"] = file_record(manifest, "bundle-manifest")
    if sha256(manifest) != manifest_sha:
        raise DeploymentError("bundle manifest SHA-256 differs")
    manifest_id = manifest_identity(manifest, args.digest_tool)
    archive_result = validate_oci_archive(
        image_archive, build_receipt, package_candidate
    )
    if (archive_result["image_manifest_digest"], archive_result["flavor"],
            archive_result["release"], archive_result["source_revision"]) != (
        image_digest, args.flavor, args.release, revision,
    ):
        raise DeploymentError("image archive lineage differs from deployment")
    kubectl = executable(args.kubectl, "kubectl")
    helm_bin = executable(args.helm, "Helm")
    kubectl_base = kube(kubectl, args.context, args.namespace)
    helm_base = helm(helm_bin, args.context, args.namespace)
    namespace = run_json(kubectl_base + ["get", "namespace", args.namespace, "-o", "json"])
    namespace_uid = namespace.get("metadata", {}).get("uid")
    if not isinstance(namespace_uid, str) or not namespace_uid:
        raise DeploymentError("namespace UID is absent")
    if run(helm_base + ["list", "--filter", f"^{args.release_name}$", "-q"]):
        raise DeploymentError("Helm release already exists")
    run_json(kubectl_base + ["get", "persistentvolumeclaim", args.source_pvc, "-o", "json"])
    secret_doc = run_json(kubectl_base + ["get", "secret", args.auth_secret, "-o", "json"])
    encoded_token = secret_doc.get("data", {}).get(args.auth_key)
    try:
        cluster_token = base64.b64decode(encoded_token, validate=True).decode("utf-8")
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        raise DeploymentError("cluster auth secret is malformed") from error
    if not secrets.compare_digest(cluster_token, token):
        raise DeploymentError("cluster auth secret differs from qualification token")

    service = f"{args.release_name}-tritium"
    selector = f"app.kubernetes.io/name=tritium,app.kubernetes.io/instance={args.release_name}"
    timeout_value = f"{math.ceil(args.timeout)}s"
    values = [
        "--set-string", f"image.repository={repository}",
        "--set-string", f"image.digest={image_digest}",
        "--set-string", f"backend={args.flavor}",
        "--set-string", f"artifact.sourcePvc.claimName={args.source_pvc}",
        "--set-string", f"artifact.profile={args.profile}",
        "--set-string", f"artifact.expectedManifestSha256={manifest_sha}",
        "--set-string", f"auth.existingSecret={args.auth_secret}",
        "--set-string", f"auth.key={args.auth_key}",
    ]
    if args.flavor == "cuda":
        values += ["--set", "gpu.enabled=true"]
    install = helm_base + [
        "install", args.release_name, str(chart), "--atomic", "--wait",
        "--timeout", timeout_value,
    ] + values
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.monotonic()
    install_attempted = False
    cleanup_passed = False
    try:
        install_attempted = True
        run(install, args.timeout + 60)
        run(kubectl_base + ["rollout", "status", f"deployment/{service}",
                            f"--timeout={timeout_value}"], args.timeout + 30)
        deployment = run_json(kubectl_base + ["get", f"deployment/{service}", "-o", "json"])
        deployment_uid = validate_deployment(
            deployment, image=args.image, manifest_sha256=manifest_sha, flavor=args.flavor
        )
        initial = pod_snapshot(
            run_json(kubectl_base + ["get", "pods", "-l", selector, "-o", "json"]),
            args.flavor,
        )
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        process, ready = port_forward(
            kubectl_base + ["port-forward", f"service/{service}", f"{port}:8080",
                            "--address", "127.0.0.1"],
            time.monotonic() + args.timeout, url, token,
        )
        try:
            startup = validate_ready(
                ready, revision, args.flavor, args.profile, manifest_id["blake3"], args.release
            )
            models = request_json(url + "/v1/models", token, timeout=args.request_timeout)
            entries = models.get("data")
            if (
                not isinstance(entries, list) or len(entries) != 1
                or not isinstance(entries[0], dict)
                or not isinstance(entries[0].get("id"), str)
                or not entries[0]["id"]
            ):
                raise DeploymentError("cluster model listing is not singular")
            generation = request_json(
                url + "/v1/chat/completions", token,
                body={"model": entries[0].get("id"),
                      "messages": [{"role": "user", "content": args.prompt}],
                      "temperature": 0, "max_tokens": 1},
                timeout=args.request_timeout,
            )
            if not isinstance(generation.get("choices"), list) or len(generation["choices"]) != 1:
                raise DeploymentError("cluster generation did not return one choice")
            metrics = request(url + "/metrics", token, timeout=args.request_timeout).decode("utf-8")
            metric_evidence = metrics_snapshot(metrics)
        finally:
            stop_forward(process)
        first_pod = initial["pods"][0]["name"]
        run(kubectl_base + ["delete", f"pod/{first_pod}", "--wait=true"], args.timeout)
        run(kubectl_base + ["rollout", "status", f"deployment/{service}",
                            f"--timeout={timeout_value}"], args.timeout + 30)
        restarted = pod_snapshot(
            run_json(kubectl_base + ["get", "pods", "-l", selector, "-o", "json"]),
            args.flavor,
        )
        if {pod["uid"] for pod in initial["pods"]} & {pod["uid"] for pod in restarted["pods"]}:
            raise DeploymentError("pod restart retained an old pod UID")
        restart_port = free_port()
        restart_url = f"http://127.0.0.1:{restart_port}"
        restart_process, restart_ready = port_forward(
            kubectl_base + ["port-forward", f"service/{service}", f"{restart_port}:8080",
                            "--address", "127.0.0.1"],
            time.monotonic() + args.timeout, restart_url, token,
        )
        try:
            restart_startup = validate_ready(
                restart_ready, revision, args.flavor, args.profile,
                manifest_id["blake3"], args.release,
            )
        finally:
            stop_forward(restart_process)
        if restart_startup != startup:
            raise DeploymentError("restart changed immutable startup receipt")
        node_names = {pod["node"] for pod in initial["pods"] + restarted["pods"]}
        nodes = [
            node_snapshot(run_json(kubectl_base + ["get", f"node/{name}", "-o", "json"]))
            for name in sorted(node_names)
        ]
        cuda_node = None
        if args.flavor == "cuda":
            if len(node_names) != 1:
                raise DeploymentError("CUDA qualification changed physical nodes")
            cuda_node = collect_cuda_node_evidence(
                kubectl_base, node=next(iter(node_names)), image=args.cuda_probe_image,
                startup=startup, timeout=args.timeout,
            )
        prior_revision = deployed_revision(run_json_value(
            helm_base + ["history", args.release_name, "-o", "json"]
        ))
        wrong_manifest = "0" * 64 if manifest_sha != "0" * 64 else "f" * 64
        wrong_image = "0" * 64 if image_digest[7:] != "0" * 64 else "f" * 64
        failure_sha = expect_failure(
            helm_base + ["upgrade", args.release_name, str(chart), "--atomic", "--wait",
                         "--timeout", timeout_value] + values + [
                             "--set-string", f"artifact.expectedManifestSha256={wrong_manifest}",
                             "--set-string", f"image.digest=sha256:{wrong_image}",
                         ],
            args.timeout + 60,
        )
        run(kubectl_base + ["rollout", "status", f"deployment/{service}",
                            f"--timeout={timeout_value}"], args.timeout + 30)
        rolled_back = run_json(
            kubectl_base + ["get", f"deployment/{service}", "-o", "json"]
        )
        rollback_uid = validate_deployment(
            rolled_back, image=args.image, manifest_sha256=manifest_sha, flavor=args.flavor
        )
        if rollback_uid != deployment_uid:
            raise DeploymentError("atomic rollback replaced deployment identity")
        rollback_port = free_port()
        rollback_url = f"http://127.0.0.1:{rollback_port}"
        rollback_process, rollback_ready = port_forward(
            kubectl_base + ["port-forward", f"service/{service}",
                            f"{rollback_port}:8080", "--address", "127.0.0.1"],
            time.monotonic() + args.timeout, rollback_url, token,
        )
        try:
            rollback_startup = validate_ready(
                rollback_ready, revision, args.flavor, args.profile,
                manifest_id["blake3"], args.release,
            )
            rollback_models = request_json(
                rollback_url + "/v1/models", token, timeout=args.request_timeout
            ).get("data")
            if (not isinstance(rollback_models, list) or len(rollback_models) != 1
                    or not isinstance(rollback_models[0], dict)
                    or not isinstance(rollback_models[0].get("id"), str)):
                raise DeploymentError("rollback model listing differs")
            rollback_generation = request_json(
                rollback_url + "/v1/chat/completions", token,
                body={"model": rollback_models[0]["id"],
                      "messages": [{"role": "user", "content": args.prompt}],
                      "temperature": 0, "max_tokens": 1},
                timeout=args.request_timeout,
            )
            if (not isinstance(rollback_generation.get("choices"), list)
                    or len(rollback_generation["choices"]) != 1):
                raise DeploymentError("rollback generation did not return one choice")
        finally:
            stop_forward(rollback_process)
        if rollback_startup != startup:
            raise DeploymentError("rollback changed immutable startup receipt")
        history = validate_helm_history(run_json_value(
            helm_base + ["history", args.release_name, "-o", "json"]
        ), prior_revision)
    finally:
        active_error = sys.exc_info()[0] is not None
        if install_attempted:
            try:
                run(helm_base + ["uninstall", args.release_name, "--wait",
                                 "--ignore-not-found", "--timeout", timeout_value],
                    args.timeout + 60)
                cleanup_passed = not run(
                    helm_base + ["list", "--filter", f"^{args.release_name}$", "-q"]
                )
                remaining = run(kubectl_base + [
                    "get", "deployment,service,pod,pdb,networkpolicy", "-l", selector,
                    "-o", "name", "--ignore-not-found=true",
                ])
                cleanup_passed = cleanup_passed and not remaining
                if not cleanup_passed and not active_error:
                    raise DeploymentError("Helm release cleanup is incomplete")
            except DeploymentError:
                if not active_error:
                    raise
    if not cleanup_passed:
        raise DeploymentError("Helm release cleanup did not pass")
    current_files = {
        "chart": file_record(chart, "helm-chart"),
        "image": file_record(image_archive, "oci-image"),
        "build_receipt": file_record(build_receipt, "oci-build-receipt"),
        "package_candidate": file_record(package_candidate, "release-candidate"),
        "manifest": file_record(manifest, "bundle-manifest"),
    }
    if current_files != initial_files:
        raise DeploymentError("qualification input changed during the cluster run")
    cluster_version = run_json([str(kubectl), "--context", args.context, "version", "-o", "json"])
    server_version = cluster_version.get("serverVersion", {})
    receipt = {
        "schema": SCHEMA, "release": args.release, "source_revision": revision,
        "run_id": args.run_id, "flavor": args.flavor, "profile": args.profile,
        "started_at_utc": started_at, "duration_ms": (time.monotonic() - started) * 1000,
        "chart_artifact": {**initial_files["chart"], **chart_binding},
        "image_artifact": initial_files["image"],
        "image": args.image, "manifest": manifest_id,
        "cluster": {"context": args.context, "namespace": args.namespace,
                    "namespace_uid": namespace_uid,
                    "server_git_version": server_version.get("gitVersion"),
                    "server_platform": server_version.get("platform"),
                    "nodes": nodes, "cuda_node": cuda_node},
        "tools": {"kubectl_sha256": sha256(kubectl), "helm_sha256": sha256(helm_bin),
                  "helm_version": run([str(helm_bin), "version", "--short"])},
        "workload": {"release_name": args.release_name, "deployment_uid": deployment_uid,
                     "initial": initial, "restarted": restarted,
                     "startup_receipt": startup, "restart_startup_receipt": restart_startup,
                     "rollback_startup_receipt": rollback_startup,
                     "metrics": metric_evidence, "helm_history": history,
                     "prior_helm_revision": prior_revision,
                     "failed_manifest_sha256": wrong_manifest,
                     "failed_image_digest": "sha256:" + wrong_image,
                     "failed_upgrade_output_sha256": failure_sha},
        "checks": list(CHECKS), "result": "pass",
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
    validate_receipt(
        receipt, chart_path=chart, image_path=image_archive,
        manifest_path=manifest, build_receipt=build_receipt,
        package_candidate=package_candidate, digest_tool=args.digest_tool,
        revision=revision, release=args.release,
    )
    return receipt


def validate_receipt(receipt: dict[str, Any], *, chart_path: Path, image_path: Path,
                     manifest_path: Path, build_receipt: Path,
                     package_candidate: Path, digest_tool: str,
                     revision: str, release: str) -> dict[str, Any]:
    fields = {
        "schema", "receipt_id", "release", "source_revision", "run_id", "flavor",
        "profile", "started_at_utc", "duration_ms", "chart_artifact", "image_artifact",
        "image", "manifest", "cluster", "tools", "workload", "checks", "result",
    }
    if not isinstance(receipt, dict) or set(receipt) != fields:
        raise DeploymentError("deployment receipt fields differ")
    if receipt.get("schema") != SCHEMA or receipt.get("result") != "pass":
        raise DeploymentError("deployment receipt schema or result differs")
    if receipt.get("release") != release or receipt.get("source_revision") != revision:
        raise DeploymentError("deployment receipt release identity differs")
    if receipt.get("checks") != list(CHECKS) or receipt.get("flavor") not in {"cpu", "cuda"}:
        raise DeploymentError("deployment receipt checks or flavor differ")
    if receipt.get("profile") not in {"compact-v1", "near-lossless-v1"}:
        raise DeploymentError("deployment receipt profile differs")
    if not isinstance(receipt.get("run_id"), str) or RUN_ID.fullmatch(receipt["run_id"]) is None:
        raise DeploymentError("deployment receipt run ID is malformed")
    duration = receipt.get("duration_ms")
    if type(duration) not in {int, float} or not math.isfinite(duration) or duration <= 0:
        raise DeploymentError("deployment receipt duration is malformed")
    started_at = receipt.get("started_at_utc")
    if not isinstance(started_at, str):
        raise DeploymentError("deployment receipt timestamp is malformed")
    try:
        parsed_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise DeploymentError("deployment receipt timestamp is malformed") from error
    if parsed_at.tzinfo is None or parsed_at.utcoffset() != timedelta(0):
        raise DeploymentError("deployment receipt timestamp must be UTC")
    package_candidate = ordinary(package_candidate, "candidate package manifest")
    build_receipt = ordinary(build_receipt, "candidate OCI build receipt")
    for key, path, kind in (("image_artifact", image_path, "oci-image"),):
        path = ordinary(path, f"candidate {kind}")
        artifact = receipt.get(key)
        if not isinstance(artifact, dict) or set(artifact) != {"kind", "name", "bytes", "sha256"}:
            raise DeploymentError(f"deployment {kind} fields differ")
        if (artifact.get("kind"), artifact.get("name"), artifact.get("bytes"),
                artifact.get("sha256")) != (
            kind, path.name, path.stat().st_size, sha256(path),
        ):
            raise DeploymentError(f"deployment receipt does not bind candidate {kind}")
    chart_path = ordinary(chart_path, "candidate Helm chart")
    chart = receipt.get("chart_artifact")
    if not isinstance(chart, dict) or set(chart) != {
        "kind", "name", "bytes", "sha256", "artifact_id", "candidate_sha256"
    }:
        raise DeploymentError("deployment helm-chart fields differ")
    binding = bind_chart_to_candidate(package_candidate, chart_path)
    if chart != {**file_record(chart_path, "helm-chart"), **binding}:
        raise DeploymentError("deployment receipt does not bind candidate helm-chart")
    image = receipt.get("image")
    if not isinstance(image, str) or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image):
        raise DeploymentError("deployment image reference is mutable")
    manifest = receipt.get("manifest")
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "bytes", "sha256", "blake3"}:
        raise DeploymentError("deployment manifest identity fields differ")
    if manifest.get("schema") != "tritium.file-identity.v1":
        raise DeploymentError("deployment manifest identity schema differs")
    if type(manifest.get("bytes")) is not int or manifest["bytes"] <= 0:
        raise DeploymentError("deployment manifest byte count is invalid")
    exact_hex(manifest.get("sha256"), 64, "deployment manifest SHA-256")
    exact_hex(manifest.get("blake3"), 64, "deployment manifest BLAKE3")
    manifest_path = ordinary(manifest_path, "candidate bundle manifest")
    if manifest_identity(manifest_path, digest_tool) != manifest:
        raise DeploymentError("deployment manifest identity differs from candidate bytes")
    archive_result = validate_oci_archive(image_path, build_receipt, package_candidate)
    if (
        archive_result.get("image_manifest_digest") != image.rpartition("@")[2]
        or archive_result.get("release") != release
        or archive_result.get("source_revision") != revision
        or archive_result.get("flavor") != receipt["flavor"]
    ):
        raise DeploymentError("deployment image lineage differs from receipt")
    cluster = receipt.get("cluster")
    if not isinstance(cluster, dict) or set(cluster) != {
        "context", "namespace", "namespace_uid", "server_git_version", "server_platform",
        "nodes", "cuda_node",
    } or any(
        not isinstance(cluster.get(key), str) or not cluster[key]
        for key in ("context", "namespace", "namespace_uid", "server_git_version", "server_platform")
    ):
        raise DeploymentError("deployment cluster identity is malformed")
    nodes = cluster.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise DeploymentError("deployment node identities are absent")
    validated_nodes = [validate_node_snapshot(node) for node in nodes]
    if validated_nodes != sorted(validated_nodes, key=lambda node: node["name"]):
        raise DeploymentError("deployment node identities are not canonical")
    tools = receipt.get("tools")
    if not isinstance(tools, dict) or set(tools) != {
        "kubectl_sha256", "helm_sha256", "helm_version"
    } or not isinstance(tools.get("helm_version"), str) or not tools["helm_version"]:
        raise DeploymentError("deployment tool identity is malformed")
    exact_hex(tools.get("kubectl_sha256"), 64, "kubectl SHA-256")
    exact_hex(tools.get("helm_sha256"), 64, "Helm SHA-256")
    workload = receipt.get("workload")
    if not isinstance(workload, dict) or set(workload) != {
        "release_name", "deployment_uid", "initial", "restarted", "startup_receipt",
        "restart_startup_receipt", "rollback_startup_receipt", "metrics", "helm_history",
        "prior_helm_revision", "failed_manifest_sha256", "failed_image_digest",
        "failed_upgrade_output_sha256",
    }:
        raise DeploymentError("deployment workload fields differ")
    for key in ("deployment_uid", "release_name"):
        if not isinstance(workload.get(key), str) or not workload[key]:
            raise DeploymentError("deployment workload identity is malformed")
    exact_hex(workload.get("failed_upgrade_output_sha256"), 64, "failed-upgrade SHA-256")
    exact_hex(workload.get("failed_manifest_sha256"), 64, "failed manifest SHA-256")
    failed_image = workload.get("failed_image_digest")
    if not isinstance(failed_image, str) or not failed_image.startswith("sha256:"):
        raise DeploymentError("failed image digest is malformed")
    exact_hex(failed_image[7:], 64, "failed image digest")
    if workload["failed_manifest_sha256"] == manifest["sha256"]:
        raise DeploymentError("failed upgrade did not use a different manifest identity")
    if failed_image == image.rpartition("@")[2]:
        raise DeploymentError("failed upgrade did not use a different image digest")
    if type(workload.get("prior_helm_revision")) is not int or workload["prior_helm_revision"] < 1:
        raise DeploymentError("prior Helm revision is malformed")
    initial = validate_receipt_snapshot(workload.get("initial"), "initial")
    restarted = validate_receipt_snapshot(workload.get("restarted"), "restarted")
    if {pod["uid"] for pod in initial["pods"]} & {
        pod.get("uid") for pod in restarted["pods"] if isinstance(pod, dict)
    }:
        raise DeploymentError("deployment restart evidence retains old pod UID")
    startup = validate_ready(
        {"status": "ready", "release_gate": "production_artifact_admitted",
         "startup_receipt": workload.get("startup_receipt")},
        revision, receipt["flavor"], receipt["profile"], manifest["blake3"], release,
    )
    validate_ready(
        {"status": "ready", "release_gate": "production_artifact_admitted",
         "startup_receipt": workload.get("restart_startup_receipt")},
        revision, receipt["flavor"], receipt["profile"], manifest["blake3"], release,
    )
    if workload["restart_startup_receipt"] != workload["startup_receipt"]:
        raise DeploymentError("deployment restart changed startup receipt")
    validate_ready(
        {"status": "ready", "release_gate": "production_artifact_admitted",
         "startup_receipt": workload.get("rollback_startup_receipt")},
        revision, receipt["flavor"], receipt["profile"], manifest["blake3"], release,
    )
    if workload["rollback_startup_receipt"] != workload["startup_receipt"]:
        raise DeploymentError("deployment rollback changed startup receipt")
    metrics = workload.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {"sha256", "values"}:
        raise DeploymentError("deployment metrics evidence fields differ")
    exact_hex(metrics.get("sha256"), 64, "metrics SHA-256")
    metric_values = metrics.get("values")
    expected_metrics = {
        "tritium_chat_requests_total", "tritium_tokens_out_total", "tritium_worker_alive"
    }
    if not isinstance(metric_values, dict) or set(metric_values) != expected_metrics or any(
        type(value) not in {int, float} or not math.isfinite(value) or value < 0
        for value in metric_values.values()
    ) or metric_values["tritium_chat_requests_total"] < 1 or (
        metric_values["tritium_tokens_out_total"] < 1
    ) or metric_values["tritium_worker_alive"] != 1:
        raise DeploymentError("deployment metrics evidence is malformed")
    validate_helm_history(workload.get("helm_history"), workload["prior_helm_revision"])
    node_names = {pod["node"] for pod in initial["pods"] + restarted["pods"]}
    if node_names != {node["name"] for node in validated_nodes}:
        raise DeploymentError("deployment node identities differ from pod placement")
    cuda_node = cluster.get("cuda_node")
    if receipt["flavor"] == "cpu":
        if cuda_node is not None or startup.get("physical_device_id") != "cpu":
            raise DeploymentError("CPU deployment contains CUDA identity")
    else:
        fields = {
            "node_name", "gpu_uuid", "gpu_name", "driver_version", "cuda_runtime",
            "probe_image", "probe_pod_uid", "output_sha256",
        }
        if not isinstance(cuda_node, dict) or set(cuda_node) != fields or any(
            not isinstance(cuda_node.get(key), str) or not cuda_node[key] for key in fields
        ):
            raise DeploymentError("CUDA node evidence is malformed")
        exact_hex(cuda_node.get("output_sha256"), 64, "CUDA probe output SHA-256")
        expected_runtime = CUDA_PROBE_IMAGES.get(cuda_node["probe_image"])
        if expected_runtime is None or cuda_node["cuda_runtime"] != expected_runtime:
            raise DeploymentError("CUDA probe image is not admitted by release policy")
        physical = startup.get("physical_device_id")
        if (GPU_UUID.fullmatch(physical or "") is None
                or physical.rsplit(":", 1)[-1] != cuda_node["gpu_uuid"]
                or cuda_node["node_name"] not in node_names):
            raise DeploymentError("CUDA node evidence differs from deployed hardware")
    supplied = receipt.get("receipt_id")
    if not isinstance(supplied, str) or not supplied.startswith("sha256:"):
        raise DeploymentError("deployment receipt ID is malformed")
    exact_hex(supplied[7:], 64, "deployment receipt ID")
    unsigned = dict(receipt)
    del unsigned["receipt_id"]
    expected = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if not secrets.compare_digest(supplied, expected):
        raise DeploymentError("deployment receipt content digest differs")
    return receipt


def load_receipt(path: Path, *, chart_path: Path, image_path: Path,
                 manifest_path: Path, build_receipt: Path,
                 package_candidate: Path, digest_tool: str,
                 revision: str, release: str) -> dict[str, Any]:
    path = ordinary(path, "deployment receipt")
    if path.stat().st_size > 32 * 1024 * 1024:
        raise DeploymentError("deployment receipt exceeds byte limit")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeploymentError("deployment receipt must contain UTF-8 JSON") from error
    return validate_receipt(
        value, chart_path=chart_path, image_path=image_path,
        manifest_path=manifest_path, build_receipt=build_receipt,
        package_candidate=package_candidate, digest_tool=digest_tool,
        revision=revision, release=release,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flavor", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--release-name", required=True)
    parser.add_argument("--chart-archive", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-archive", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--package-candidate", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-pvc", required=True)
    parser.add_argument("--profile", default="compact-v1")
    parser.add_argument("--auth-secret", required=True)
    parser.add_argument("--auth-key", default="token")
    parser.add_argument("--auth-token-file", type=Path, required=True)
    parser.add_argument("--cuda-probe-image")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--helm", default="helm")
    parser.add_argument("--digest-tool", default=os.environ.get("TRITIUM_BIN", "tritium"))
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--prompt", default="Hello")
    args = parser.parse_args()
    try:
        receipt = qualify(args)
        atomic_create(args.output, canonical(receipt))
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as error:
        print(f"qualify-kubernetes-deployment: BLOCKED: {error}", file=sys.stderr)
        return 1
    print(f"qualify-kubernetes-deployment: PASS: {receipt['receipt_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
