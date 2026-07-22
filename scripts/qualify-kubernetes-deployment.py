#!/usr/bin/env python3
"""Install, exercise, restart, fail, roll back, and seal one Tritium Helm release."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
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
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


RUNTIME = runpy.run_path(Path(__file__).with_name("qualify-oci-runtime.py"))
validate_ready = RUNTIME["validate_ready"]
manifest_identity = RUNTIME["manifest_identity"]
OCI_ARCHIVE = runpy.run_path(Path(__file__).with_name("verify-oci-archive.py"))
validate_oci_archive = OCI_ARCHIVE["validate"]

SCHEMA = "tritium.kubernetes-deployment-qualification.v5"
HEX = frozenset("0123456789abcdef")
SAFE_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?")
SAFE_SECRET_KEY = re.compile(r"[A-Za-z0-9](?:[-._A-Za-z0-9]{0,251}[A-Za-z0-9])?")
LABEL_KEY = re.compile(
    r"(?:[a-z0-9](?:[-a-z0-9.]{0,251}[a-z0-9])?/)?"
    r"[A-Za-z0-9](?:[-._A-Za-z0-9]{0,61}[A-Za-z0-9])?"
)
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
RELEASE = re.compile(r"1\.1\.0-rc\.(0|[1-9][0-9]*)")
GPU_UUID = re.compile(
    r"cuda:[0-9]+:GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
OCI_REPOSITORY = re.compile(
    r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?(?::[0-9]{1,5})?"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+"
)
CUDA_PROBE_IMAGES = {
    "docker.io/nvidia/cuda:13.0.1-devel-ubuntu22.04@sha256:"
    "bb3de902bc1b522231cffea98a4d25a16ddb9fc8685a958b48d83036f46fd0c2": "13.0.1",
}
CHECKS = (
    "namespace-preflight", "secret-binding", "pvc-preflight", "helm-install",
    "rollout-ready", "production-readiness", "buffered-generation", "metrics",
    "watchdog-replacement", "pod-restart", "artifact-volume-loss",
    "successful-update", "update-readiness", "failed-upgrade",
    "atomic-rollback", "rollback-readiness", "rollback-generation", "release-cleanup",
)
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
WATCHDOG_FAULT_COMMAND = (
    'set -- $(pidof tritium-serve); [ "$#" -eq 1 ]; kill -STOP "$1"'
)
WATCHDOG_SCHEDULING_ALLOWANCE_MS = 60_000
WATCHDOG_POLICY = {
    "period_seconds": 10,
    "timeout_seconds": 2,
    "failure_threshold": 3,
    "escalation_seconds": 2,
    "scheduling_allowance_ms": WATCHDOG_SCHEDULING_ALLOWANCE_MS,
    "budget_ms": 98_000,
}
ARTIFACT_VOLUME_OBSERVATION_BUDGET_MS = 120_000
QUALIFICATION_METRICS = {
    "tritium_chat_requests_total", "tritium_tokens_out_total",
    "tritium_worker_alive", "tritium_backend_faults_total",
    "tritium_backend_faulted",
}
ARTIFACT_VOLUME_TRANSITIONS = [
    "baseline_ready", "absence_verified", "fault_applied",
    "unschedulable_observed", "source_restored", "recovery_ready",
]


def expected_checks(flavor: str) -> list[str]:
    return [*CHECKS, *(["keda-scale-signal"] if flavor == "cpu" else [])]


class DeploymentError(ValueError):
    """Kubernetes deployment evidence is malformed, incomplete, or non-green."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def request_evidence(model_id: str, prompt: str, *, temperature: int,
                     max_tokens: int) -> dict[str, Any]:
    descriptor = {
        "model": model_id, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_bytes": len(prompt.encode()), "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return {**descriptor, "descriptor_sha256": hashlib.sha256(canonical(descriptor)).hexdigest()}


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


def label_assignment(value: str) -> tuple[str, str]:
    key, separator, label_value = value.partition("=")
    if (separator != "=" or LABEL_KEY.fullmatch(key) is None
            or not label_value or len(label_value) > 63
            or re.fullmatch(r"[A-Za-z0-9](?:[-._A-Za-z0-9]{0,61}[A-Za-z0-9])?", label_value) is None):
        raise DeploymentError("ServiceMonitor label must be a Kubernetes key=value")
    return key, label_value


def helm_key(value: str) -> str:
    return value.replace(".", "\\.")


def prometheus_url(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 512:
        raise DeploymentError("Prometheus URL is malformed")
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise DeploymentError("Prometheus URL is malformed") from error
    if (parsed.scheme not in {"http", "https"} or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or not parsed.hostname or port is None or not 1 <= port <= 65535
            or re.fullmatch(r"[a-z0-9](?:[-a-z0-9.]{0,251}[a-z0-9])?", parsed.hostname) is None
            or not re.fullmatch(r"(?:/[A-Za-z0-9._~/-]*)?", parsed.path)):
        raise DeploymentError("Prometheus URL is malformed")
    return value


def bind_prometheus_endpoint(value: Any, service: str, namespace: str, port: int) -> str:
    server = prometheus_url(value)
    parsed = urllib.parse.urlsplit(server)
    admitted_hosts = {
        f"{service}.{namespace}.svc",
        f"{service}.{namespace}.svc.cluster.local",
    }
    if parsed.hostname not in admitted_hosts or parsed.port != port or parsed.path not in {"", "/"}:
        raise DeploymentError("KEDA Prometheus URL differs from observed Service")
    return server


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


def prove_absent(command: list[str], timeout: float) -> dict[str, str]:
    try:
        result = subprocess.run(
            command + ["--ignore-not-found=true"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DeploymentError(f"absence check could not run: {error}") from error
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0 or stdout or stderr:
        raise DeploymentError("resource absence check did not return exact empty success")
    return {
        "status": "NotFound",
        "output_sha256": hashlib.sha256(b"").hexdigest(),
    }


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


def public_json(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, urllib.error.URLError) as error:
        raise DeploymentError(f"public request failed: {url}: {error}") from error
    if len(payload) > MAX_RESPONSE_BYTES:
        raise DeploymentError(f"public response exceeds byte limit: {url}")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise DeploymentError(f"public response is malformed JSON: {url}") from error
    if not isinstance(value, dict):
        raise DeploymentError(f"public response is not an object: {url}")
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


def pod_snapshot(document: dict[str, Any], flavor: str,
                 expected_count: int = 1) -> dict[str, Any]:
    items = document.get("items")
    if not isinstance(items, list) or len(items) != expected_count:
        raise DeploymentError(f"deployment must have exactly {expected_count} pod(s)")
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


def container_identity(document: dict[str, Any], pod_name: str,
                       container_name: str) -> dict[str, Any]:
    metadata = document.get("metadata", {})
    if metadata.get("name") != pod_name or not isinstance(metadata.get("uid"), str):
        raise DeploymentError("watchdog pod identity differs")
    statuses = document.get("status", {}).get("containerStatuses", [])
    matches = [status for status in statuses if isinstance(status, dict)
               and status.get("name") == container_name]
    if len(matches) != 1:
        raise DeploymentError("watchdog target container status differs")
    status = matches[0]
    container_id = status.get("containerID")
    restarts = status.get("restartCount")
    if (not isinstance(container_id, str) or not container_id
            or type(restarts) is not int or restarts < 0):
        raise DeploymentError("watchdog target container identity is malformed")
    terminated = status.get("lastState", {}).get("terminated")
    exit_code = terminated.get("exitCode") if isinstance(terminated, dict) else None
    return {
        "pod_uid": metadata["uid"], "container_id": container_id,
        "restart_count": restarts, "last_exit_code": exit_code,
    }


def qualify_watchdog_restart(kubectl_base: list[str], *, pod_name: str,
                             timeout: float,
                             contract: dict[str, int]) -> dict[str, Any]:
    before = container_identity(
        run_json(kubectl_base + ["get", f"pod/{pod_name}", "-o", "json"]),
        pod_name, "tritium",
    )
    fault_command = WATCHDOG_FAULT_COMMAND
    started = time.monotonic()
    run(kubectl_base + ["exec", f"pod/{pod_name}", "-c", "authenticated-probe",
                        "--", "sh", "-ec", fault_command], min(timeout, 30.0))
    deadline = started + min(timeout, contract["budget_ms"] / 1000)
    while time.monotonic() < deadline:
        current = container_identity(
            run_json(kubectl_base + ["get", f"pod/{pod_name}", "-o", "json"]),
            pod_name, "tritium",
        )
        if current["pod_uid"] != before["pod_uid"]:
            raise DeploymentError("watchdog fault replaced the pod instead of the process")
        if current["restart_count"] > before["restart_count"] + 1:
            raise DeploymentError("watchdog fault caused repeated process restarts")
        if current["restart_count"] == before["restart_count"] + 1:
            if (current["container_id"] == before["container_id"]
                    or current["last_exit_code"] != 137):
                raise DeploymentError("watchdog process replacement identity differs")
            return {
                "pod_uid": before["pod_uid"],
                "container_id_before": before["container_id"],
                "container_id_after": current["container_id"],
                "restart_count_before": before["restart_count"],
                "restart_count_after": current["restart_count"],
                "last_exit_code": current["last_exit_code"],
                "fault_command_sha256": hashlib.sha256(
                    fault_command.encode()
                ).hexdigest(),
                "replacement_ms": (time.monotonic() - started) * 1000,
                "watchdog": contract,
            }
        time.sleep(1)
    raise DeploymentError("watchdog did not replace the stopped serving process")


def validate_receipt_snapshot(snapshot: Any, label: str,
                              expected_count: int = 1) -> dict[str, Any]:
    if not isinstance(snapshot, dict) or set(snapshot) != {"pods"}:
        raise DeploymentError(f"{label} pod snapshot fields differ")
    pods = snapshot.get("pods")
    if not isinstance(pods, list) or len(pods) != expected_count:
        raise DeploymentError(f"{label} must contain exactly {expected_count} pod(s)")
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


def pvc_identity(document: dict[str, Any], expected_name: str) -> dict[str, Any]:
    metadata = document.get("metadata", {})
    spec = document.get("spec", {})
    status = document.get("status", {})
    modes = spec.get("accessModes")
    result = {
        "name": metadata.get("name"), "uid": metadata.get("uid"),
        "volume_name": spec.get("volumeName"),
        "storage_class": spec.get("storageClassName"),
        "access_modes": sorted(modes) if isinstance(modes, list) else modes,
        "capacity": status.get("capacity", {}).get("storage"),
        "phase": status.get("phase"),
    }
    if (result["name"] != expected_name
            or any(not isinstance(result.get(key), str) or not result[key]
                   for key in ("uid", "volume_name", "storage_class", "capacity"))
            or not isinstance(result["access_modes"], list) or not result["access_modes"]
            or any(not isinstance(mode, str) or not mode for mode in result["access_modes"])
            or result["phase"] != "Bound"):
        raise DeploymentError("source artifact PVC is not fully bound")
    return result


def validate_pvc_identity(value: Any, expected_name: str) -> dict[str, Any]:
    fields = {
        "name", "uid", "volume_name", "storage_class", "access_modes",
        "capacity", "phase",
    }
    if (not isinstance(value, dict) or set(value) != fields
            or value.get("name") != expected_name or value.get("phase") != "Bound"
            or any(not isinstance(value.get(key), str) or not value[key]
                   for key in ("uid", "volume_name", "storage_class", "capacity"))
            or not isinstance(value.get("access_modes"), list) or not value["access_modes"]
            or value["access_modes"] != sorted(value["access_modes"])
            or any(not isinstance(mode, str) or not mode for mode in value["access_modes"])):
        raise DeploymentError("deployment source PVC identity is malformed")
    return value


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


def watchdog_contract(document: dict[str, Any]) -> dict[str, int]:
    containers = document.get("spec", {}).get("template", {}).get("spec", {}).get(
        "containers", []
    )
    matches = [container for container in containers if isinstance(container, dict)
               and container.get("name") == "authenticated-probe"]
    args = matches[0].get("args") if len(matches) == 1 else None
    if not isinstance(args, list) or len(args) != 1 or not isinstance(args[0], str):
        raise DeploymentError("deployed watchdog command is malformed")
    script = args[0]
    patterns = {
        "period_seconds": r"while sleep ([0-9]+)",
        "timeout_seconds": r"wget .*?--timeout=([0-9]+)",
        "failure_threshold": r'failures" -ge ([0-9]+)',
        "escalation_seconds": r"kill \$pid;\s*sleep ([0-9]+)",
    }
    values = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, script, re.DOTALL)
        if match is None:
            raise DeploymentError(f"deployed watchdog {field} is absent")
        values[field] = int(match.group(1))
    if (not 1 <= values["period_seconds"] <= 60
            or not 1 <= values["timeout_seconds"] <= 30
            or not 1 <= values["failure_threshold"] <= 10
            or values["escalation_seconds"] != values["timeout_seconds"]
            or "kill -KILL $pid" not in script):
        raise DeploymentError("deployed watchdog bounds differ")
    values["scheduling_allowance_ms"] = WATCHDOG_SCHEDULING_ALLOWANCE_MS
    values["budget_ms"] = (
        values["failure_threshold"]
        * (values["period_seconds"] + values["timeout_seconds"])
        + values["escalation_seconds"]
    ) * 1000 + WATCHDOG_SCHEDULING_ALLOWANCE_MS
    if values != WATCHDOG_POLICY:
        raise DeploymentError("deployed watchdog differs from release policy")
    return values


def artifact_volume_contract(document: dict[str, Any]) -> dict[str, Any]:
    volumes = document.get("spec", {}).get("template", {}).get("spec", {}).get(
        "volumes", []
    )
    matches = [(index, volume) for index, volume in enumerate(volumes)
               if isinstance(volume, dict) and volume.get("name") == "source-artifact"]
    if len(matches) != 1:
        raise DeploymentError("source artifact volume contract differs")
    index, volume = matches[0]
    claim = volume.get("persistentVolumeClaim", {}).get("claimName")
    if not isinstance(claim, str) or SAFE_NAME.fullmatch(claim) is None:
        raise DeploymentError("source artifact PVC identity is malformed")
    return {"volume_index": index, "claim_name": claim}


def pending_artifact_volume_failure(document: dict[str, Any], *, missing_claim: str,
                                    previous_uids: set[str]) -> dict[str, str] | None:
    items = document.get("items")
    if not isinstance(items, list):
        raise DeploymentError("artifact-volume pod listing is malformed")
    for pod in items:
        if not isinstance(pod, dict):
            raise DeploymentError("artifact-volume pod entry is malformed")
        metadata = pod.get("metadata", {})
        uid = metadata.get("uid")
        name = metadata.get("name")
        if uid in previous_uids:
            continue
        conditions = pod.get("status", {}).get("conditions", [])
        failures = [condition for condition in conditions if isinstance(condition, dict)
                    and condition.get("type") == "PodScheduled"
                    and condition.get("status") == "False"
                    and condition.get("reason") == "Unschedulable"
                    and isinstance(condition.get("message"), str)
                    and missing_claim in condition["message"]]
        if len(failures) == 1 and all(isinstance(value, str) and value
                                      for value in (uid, name)):
            return {
                "pod_name": name, "pod_uid": uid, "reason": "Unschedulable",
                "message_sha256": hashlib.sha256(
                    failures[0]["message"].encode()
                ).hexdigest(),
            }
    return None


def resource_quantity(value: Any, *, cpu: bool) -> int:
    if not isinstance(value, str):
        raise DeploymentError("Kubernetes resource quantity is malformed")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([A-Za-z]*)", value)
    if match is None:
        raise DeploymentError("Kubernetes resource quantity is malformed")
    factors = (
        {"": 1_000_000_000, "m": 1_000_000, "u": 1_000, "n": 1}
        if cpu else
        {"": 1, "K": 1_000, "M": 1_000_000, "G": 1_000_000_000,
         "Ki": 1024, "Mi": 1024 ** 2, "Gi": 1024 ** 3}
    )
    suffix = match.group(2)
    if suffix not in factors:
        raise DeploymentError("Kubernetes resource quantity suffix is unsupported")
    try:
        scaled = Decimal(match.group(1)) * factors[suffix]
    except InvalidOperation as error:
        raise DeploymentError("Kubernetes resource quantity is malformed") from error
    if scaled != scaled.to_integral_value() or scaled < 0:
        raise DeploymentError("Kubernetes resource quantity is not integral")
    return int(scaled)


def resource_usage_sample(document: dict[str, Any], *, allow_empty: bool = False) -> dict[str, Any]:
    items = document.get("items")
    if not isinstance(items, list) or len(items) > 16:
        raise DeploymentError("Kubernetes resource metrics pod set is malformed")
    cpu_nanocores = 0
    memory_bytes = 0
    containers = 0
    pod_names = []
    for pod in items:
        name = pod.get("metadata", {}).get("name") if isinstance(pod, dict) else None
        if not isinstance(name, str) or not name or name in pod_names:
            raise DeploymentError("Kubernetes resource metrics pod identity differs")
        pod_names.append(name)
        entries = pod.get("containers") if isinstance(pod, dict) else None
        if not isinstance(entries, list) or len(entries) > 8:
            raise DeploymentError("Kubernetes resource metrics containers differ")
        for container in entries:
            usage = container.get("usage") if isinstance(container, dict) else None
            if not isinstance(usage, dict) or set(usage) != {"cpu", "memory"}:
                raise DeploymentError("Kubernetes resource metrics usage differs")
            cpu_nanocores += resource_quantity(usage["cpu"], cpu=True)
            memory_bytes += resource_quantity(usage["memory"], cpu=False)
            containers += 1
    if ((not allow_empty and (not items or containers == 0))
            or (items and containers == 0)
            or (containers > 0 and (cpu_nanocores <= 0 or memory_bytes <= 0))):
        raise DeploymentError("Kubernetes resource metrics sample is empty")
    return {
        "sample_sha256": hashlib.sha256(canonical(document)).hexdigest(),
        "sampled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pod_names": sorted(pod_names), "pod_count": len(items),
        "container_count": containers,
        "cpu_nanocores": cpu_nanocores, "memory_bytes": memory_bytes,
    }


def validate_resource_sample(value: Any, label: str,
                             *, allow_empty: bool = False) -> dict[str, Any]:
    fields = {
        "sample_sha256", "sampled_at_utc", "pod_names", "pod_count",
        "container_count", "cpu_nanocores", "memory_bytes",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise DeploymentError(f"artifact-volume {label} resource fields differ")
    exact_hex(value.get("sample_sha256"), 64, f"artifact-volume {label} sample SHA-256")
    try:
        timestamp = datetime.fromisoformat(value["sampled_at_utc"])
    except (TypeError, ValueError) as error:
        raise DeploymentError(f"artifact-volume {label} timestamp is malformed") from error
    names = value.get("pod_names")
    if (timestamp.tzinfo != timezone.utc or not isinstance(names, list)
            or names != sorted(set(names))
            or any(not isinstance(name, str) or not name for name in names)
            or type(value.get("pod_count")) is not int
            or value["pod_count"] != len(names)
            or type(value.get("container_count")) is not int
            or type(value.get("cpu_nanocores")) is not int
            or type(value.get("memory_bytes")) is not int
            or min(value["container_count"], value["cpu_nanocores"],
                   value["memory_bytes"]) < 0
            or not allow_empty and (
                value["pod_count"] < 1 or value["container_count"] < 1
                or value["cpu_nanocores"] < 1 or value["memory_bytes"] < 1
            )
            or value["container_count"] == 0
            and (value["cpu_nanocores"] != 0 or value["memory_bytes"] != 0)):
        raise DeploymentError(f"artifact-volume {label} resource evidence is malformed")
    return value


def validate_transition_trace(value: Any) -> list[dict[str, Any]]:
    if (not isinstance(value, list) or len(value) != len(ARTIFACT_VOLUME_TRANSITIONS)
            or [item.get("state") if isinstance(item, dict) else None for item in value]
            != ARTIFACT_VOLUME_TRANSITIONS):
        raise DeploymentError("artifact-volume transition sequence differs")
    elapsed = []
    observed = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "state", "elapsed_ms", "observed_at_utc"
        } or type(item.get("elapsed_ms")) not in {int, float} or not math.isfinite(
            item["elapsed_ms"]
        ) or item["elapsed_ms"] < 0:
            raise DeploymentError("artifact-volume transition evidence is malformed")
        try:
            timestamp = datetime.fromisoformat(item["observed_at_utc"])
        except (TypeError, ValueError) as error:
            raise DeploymentError("artifact-volume transition timestamp is malformed") from error
        if timestamp.tzinfo != timezone.utc:
            raise DeploymentError("artifact-volume transition timestamp is not UTC")
        elapsed.append(item["elapsed_ms"])
        observed.append(timestamp)
    if elapsed != sorted(elapsed) or observed != sorted(observed):
        raise DeploymentError("artifact-volume transition timing is not ordered")
    return value


def collect_resource_usage(kubectl_base: list[str], *, namespace: str,
                           selector: str, timeout: float,
                           allow_empty: bool = False,
                           expected_pods: set[str] | None = None) -> dict[str, Any]:
    encoded = urllib.parse.quote(selector, safe="")
    path = f"/apis/metrics.k8s.io/v1beta1/namespaces/{namespace}/pods?labelSelector={encoded}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sample = resource_usage_sample(
            run_json(kubectl_base + ["get", "--raw", path], min(timeout, 30)),
            allow_empty=allow_empty,
        )
        if expected_pods is None or set(sample["pod_names"]) == expected_pods:
            return sample
        time.sleep(1)
    raise DeploymentError("Kubernetes resource metrics pod set did not converge")


def qualify_artifact_volume_loss(kubectl_base: list[str], *, service: str,
                                 namespace: str, selector: str, contract: dict[str, Any],
                                 previous_uids: set[str], timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    transitions = []

    def record_transition(state: str) -> None:
        transitions.append({
            "state": state,
            "elapsed_ms": (time.monotonic() - started) * 1000,
            "observed_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        })

    record_transition("baseline_ready")
    missing_claim = f"tritium-missing-{secrets.token_hex(6)}"
    absence = prove_absent(
        kubectl_base + ["get", f"pvc/{missing_claim}", "-o", "name"], min(timeout, 30)
    )
    record_transition("absence_verified")
    path = (
        f'/spec/template/spec/volumes/{contract["volume_index"]}'
        "/persistentVolumeClaim/claimName"
    )
    fault_patch = [{"op": "replace", "path": path, "value": missing_claim}]
    restore_patch = [{"op": "replace", "path": path, "value": contract["claim_name"]}]
    fault_payload = canonical(fault_patch).decode().strip()
    patched = False
    observation = None
    observation_ms = None
    failure_resources = None
    try:
        run(kubectl_base + ["patch", f"deployment/{service}", "--type=json",
                            f"--patch={fault_payload}"], min(timeout, 60))
        patched = True
        record_transition("fault_applied")
        deadline = started + min(timeout, ARTIFACT_VOLUME_OBSERVATION_BUDGET_MS / 1000)
        while time.monotonic() < deadline:
            observation = pending_artifact_volume_failure(
                run_json(kubectl_base + ["get", "pods", "-l", selector, "-o", "json"]),
                missing_claim=missing_claim, previous_uids=previous_uids,
            )
            if observation is not None:
                observation_ms = (time.monotonic() - started) * 1000
                record_transition("unschedulable_observed")
                failure_resources = collect_resource_usage(
                    kubectl_base, namespace=namespace, selector=selector,
                    timeout=min(timeout, 30), allow_empty=True,
                )
                break
            time.sleep(1)
        if observation is None:
            raise DeploymentError("missing artifact PVC did not produce Unschedulable pod")
    finally:
        if patched:
            run(kubectl_base + ["patch", f"deployment/{service}", "--type=json",
                                f"--patch={canonical(restore_patch).decode().strip()}"],
                min(timeout, 60))
            record_transition("source_restored")
    return {
        "source_claim": contract["claim_name"], "missing_claim": missing_claim,
        "volume_index": contract["volume_index"],
        "absence": absence,
        "fault_patch_sha256": hashlib.sha256(fault_payload.encode()).hexdigest(),
        "observation_budget_ms": ARTIFACT_VOLUME_OBSERVATION_BUDGET_MS,
        "observation_ms": observation_ms,
        "pending": observation,
        "failure_resources": failure_resources,
        "transitions": transitions,
        "_scenario_started_monotonic": started,
    }


def deployment_update_identity(document: dict[str, Any]) -> dict[str, int]:
    metadata = document.get("metadata", {})
    generation = metadata.get("generation")
    containers = document.get("spec", {}).get("template", {}).get("spec", {}).get(
        "containers", []
    )
    if type(generation) is not int or generation < 1 or not isinstance(containers, list):
        raise DeploymentError("deployment update identity is malformed")
    tritium = next(
        (container for container in containers
         if isinstance(container, dict) and container.get("name") == "tritium"), None
    )
    args = tritium.get("args") if isinstance(tritium, dict) else None
    if not isinstance(args, list):
        raise DeploymentError("deployment Tritium arguments are malformed")
    indices = [index for index, value in enumerate(args) if value == "--rate-limit-burst"]
    if len(indices) != 1 or indices[0] + 1 >= len(args):
        raise DeploymentError("deployment rate-limit argument is absent or duplicated")
    try:
        burst = int(args[indices[0] + 1])
    except (TypeError, ValueError) as error:
        raise DeploymentError("deployment rate-limit burst is malformed") from error
    return {"generation": generation, "rate_limit_burst": burst}


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


def public_port_forward(command: list[str], deadline: float, url: str) -> subprocess.Popen:
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
        )
    except OSError as error:
        raise DeploymentError(f"public port-forward failed to start: {error}") from error
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise DeploymentError("public port-forward exited before readiness")
            try:
                value = public_json(url + "/api/v1/status/buildinfo", 5)
                if value.get("status") == "success":
                    return process
            except DeploymentError:
                time.sleep(1)
        raise DeploymentError("public port-forward readiness deadline expired")
    except BaseException:
        stop_forward(process)
        raise


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
    for name in sorted(QUALIFICATION_METRICS):
        match = re.search(rf"(?m)^{name} ([0-9]+(?:\.[0-9]+)?)$", metrics)
        if match is None:
            raise DeploymentError(f"cluster metrics lack {name}")
        values[name] = float(match.group(1))
    if values["tritium_chat_requests_total"] < 1 or values["tritium_tokens_out_total"] < 1:
        raise DeploymentError("cluster metrics did not observe generation")
    if values["tritium_worker_alive"] != 1:
        raise DeploymentError("cluster metrics report a dead worker")
    if (values["tritium_backend_faults_total"] != 0
            or values["tritium_backend_faulted"] != 0):
        raise DeploymentError("cluster metrics report a backend fault")
    return {"sha256": hashlib.sha256(metrics.encode()).hexdigest(), "values": values}


def validate_metrics_evidence(value: Any, label: str) -> dict[str, float]:
    descriptor = f"{label} metrics" if label else "metrics"
    if not isinstance(value, dict) or set(value) != {"sha256", "values"}:
        raise DeploymentError(f"deployment {descriptor} fields differ")
    exact_hex(value.get("sha256"), 64, f"{descriptor} SHA-256")
    values = value.get("values")
    if not isinstance(values, dict) or set(values) != QUALIFICATION_METRICS or any(
        type(metric) not in {int, float} or not math.isfinite(metric) or metric < 0
        for metric in values.values()
    ) or values["tritium_chat_requests_total"] < 1 or (
        values["tritium_tokens_out_total"] < 1
    ) or values["tritium_worker_alive"] != 1 or any(
        values[name] != 0
        for name in ("tritium_backend_faults_total", "tritium_backend_faulted")
    ):
        raise DeploymentError(f"deployment {descriptor} evidence is malformed")
    return values


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


def scale_snapshot(scaled_object: dict[str, Any], hpa: dict[str, Any],
                   deployment: dict[str, Any]) -> dict[str, Any]:
    scaled_metadata = scaled_object.get("metadata", {})
    scaled_spec = scaled_object.get("spec", {})
    conditions = scaled_object.get("status", {}).get("conditions", [])
    active = any(
        isinstance(condition, dict) and condition.get("type") == "Active"
        and condition.get("status") == "True"
        for condition in conditions
    )
    ready = any(
        isinstance(condition, dict) and condition.get("type") == "Ready"
        and condition.get("status") == "True"
        for condition in conditions
    )
    external_metrics = scaled_object.get("status", {}).get("externalMetricNames")
    if not active or not ready or external_metrics != ["s0-prometheus-tritium_queue_pressure"]:
        raise DeploymentError("KEDA scale signal is not active and ready")
    if (scaled_spec.get("minReplicaCount"), scaled_spec.get("maxReplicaCount")) != (1, 2):
        raise DeploymentError("KEDA replica bounds differ")
    hpa_metadata = hpa.get("metadata", {})
    hpa_status = hpa.get("status", {})
    deployment_status = deployment.get("status", {})
    deployment_spec = deployment.get("spec", {})
    scaled_uid = scaled_metadata.get("uid")
    owners = hpa_metadata.get("ownerReferences")
    if not isinstance(owners, list) or not any(
        isinstance(owner, dict) and owner.get("kind") == "ScaledObject"
        and owner.get("uid") == scaled_uid and owner.get("controller") is True
        for owner in owners
    ):
        raise DeploymentError("KEDA HPA is not owned by the qualified ScaledObject")
    replicas = deployment_status.get("readyReplicas")
    if (type(replicas) is not int or replicas < 2
            or deployment_spec.get("replicas") != replicas
            or hpa_status.get("currentReplicas") != replicas
            or hpa_status.get("desiredReplicas") != replicas):
        raise DeploymentError("KEDA did not produce a ready scale-out")
    result = {
        "scaled_object_uid": scaled_uid,
        "hpa_uid": hpa_metadata.get("uid"),
        "external_metric": external_metrics[0],
        "scaled_replicas": replicas,
    }
    if any(not isinstance(result.get(key), str) or not result[key]
           for key in ("scaled_object_uid", "hpa_uid", "external_metric")):
        raise DeploymentError("KEDA scale object identity is incomplete")
    return result


def condition(document: dict[str, Any], kind: str) -> bool:
    return any(
        isinstance(item, dict) and item.get("type") == kind and item.get("status") == "True"
        for item in document.get("status", {}).get("conditions", [])
    )


def validate_scale_contract(scaled_object: dict[str, Any], service_monitor: dict[str, Any],
                            *, service: str, server: str, query: str,
                            auth_secret: str, auth_key: str,
                            monitor_label: tuple[str, str]) -> None:
    spec = scaled_object.get("spec", {})
    triggers = spec.get("triggers")
    expected_trigger = {"type": "prometheus", "metadata": {
        "serverAddress": server, "metricName": "tritium_queue_pressure",
        "threshold": "1", "query": query,
    }}
    if (spec.get("scaleTargetRef", {}).get("name") != service
            or spec.get("pollingInterval") != 5 or spec.get("cooldownPeriod") != 30
            or spec.get("minReplicaCount") != 1 or spec.get("maxReplicaCount") != 2
            or triggers != [expected_trigger]):
        raise DeploymentError("rendered KEDA trigger differs from qualification contract")
    metadata = service_monitor.get("metadata", {})
    monitor_spec = service_monitor.get("spec", {})
    endpoints = monitor_spec.get("endpoints")
    key, value = monitor_label
    if (metadata.get("labels", {}).get(key) != value
            or monitor_spec.get("selector", {}).get("matchLabels") != {
                "app.kubernetes.io/name": "tritium",
                "app.kubernetes.io/instance": service.removesuffix("-tritium"),
            }
            or not isinstance(endpoints, list) or len(endpoints) != 1
            or endpoints[0].get("path") != "/metrics"
            or endpoints[0].get("port") != "http"
            or endpoints[0].get("interval") != "30s"
            or endpoints[0].get("authorization", {}).get("credentials") != {
                "name": auth_secret, "key": auth_key,
            }):
        raise DeploymentError("rendered ServiceMonitor differs from qualification contract")


def prometheus_sample(base_url: str, query: str, timeout: float) -> dict[str, float]:
    url = base_url + "/api/v1/query?" + urllib.parse.urlencode({"query": query})
    value = public_json(url, timeout)
    result = value.get("data", {}).get("result") if value.get("status") == "success" else None
    if not isinstance(result, list) or len(result) != 1:
        raise DeploymentError("Prometheus query did not return one series")
    sample = result[0].get("value") if isinstance(result[0], dict) else None
    try:
        timestamp = float(sample[0])
        number = float(sample[1])
    except (TypeError, ValueError, IndexError) as error:
        raise DeploymentError("Prometheus sample is malformed") from error
    if not math.isfinite(timestamp) or not math.isfinite(number) or number < 0:
        raise DeploymentError("Prometheus sample is non-finite or negative")
    return {"timestamp": timestamp, "value": number}


def prometheus_peak(base_url: str, query: str, start: float, end: float,
                    timeout: float) -> dict[str, float]:
    url = base_url + "/api/v1/query_range?" + urllib.parse.urlencode({
        "query": query, "start": str(start), "end": str(end), "step": "5s",
    })
    value = public_json(url, timeout)
    result = value.get("data", {}).get("result") if value.get("status") == "success" else None
    if not isinstance(result, list) or len(result) != 1:
        raise DeploymentError("Prometheus range query did not return one series")
    samples = result[0].get("values") if isinstance(result[0], dict) else None
    if not isinstance(samples, list) or not samples:
        raise DeploymentError("Prometheus range query contains no samples")
    parsed = []
    try:
        for sample in samples:
            parsed.append((float(sample[0]), float(sample[1])))
    except (TypeError, ValueError, IndexError) as error:
        raise DeploymentError("Prometheus range sample is malformed") from error
    if any(not math.isfinite(ts) or not math.isfinite(number) or number < 0
           for ts, number in parsed):
        raise DeploymentError("Prometheus range sample is non-finite or negative")
    timestamp, number = max(parsed, key=lambda item: item[1])
    return {"timestamp": timestamp, "value": number}


def prometheus_target(base_url: str, *, namespace: str, service: str,
                      timeout: float) -> dict[str, Any]:
    value = public_json(base_url + "/api/v1/targets?state=active", timeout)
    targets = value.get("data", {}).get("activeTargets") if value.get("status") == "success" else None
    matches = [target for target in targets or [] if isinstance(target, dict)
               and target.get("labels", {}).get("namespace") == namespace
               and target.get("labels", {}).get("service") == service]
    if len(matches) != 1 or matches[0].get("health") != "up":
        raise DeploymentError("Prometheus does not have one healthy Tritium target")
    target = matches[0]
    try:
        last_scrape = datetime.fromisoformat(target["lastScrape"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as error:
        raise DeploymentError("Prometheus target scrape timestamp is malformed") from error
    age = datetime.now(timezone.utc) - last_scrape.astimezone(timezone.utc)
    scrape_url = target.get("scrapeUrl")
    if (age < timedelta(seconds=-5) or age > timedelta(seconds=120)
            or not isinstance(scrape_url, str) or not scrape_url.endswith("/metrics")):
        raise DeploymentError("Prometheus target scrape is stale or wrong")
    return {"scrape_url": scrape_url, "last_scrape_utc": last_scrape.isoformat()}


def scale_baseline(scaled_object: dict[str, Any], hpa: dict[str, Any],
                   deployment: dict[str, Any]) -> bool:
    hpa_status = hpa.get("status", {})
    return (condition(scaled_object, "Ready") and not condition(scaled_object, "Active")
            and hpa_status.get("currentReplicas") == 1
            and hpa_status.get("desiredReplicas") == 1
            and deployment.get("spec", {}).get("replicas") == 1
            and deployment.get("status", {}).get("readyReplicas") == 1)


def validate_scale_receipt(value: Any) -> dict[str, Any]:
    fields = {
        "scaled_object_uid", "hpa_uid", "external_metric", "scaled_replicas",
        "settled_replicas", "load_requests", "load_concurrency", "max_tokens",
        "prometheus_server", "monitoring_namespace", "service_monitor_label",
        "query", "target", "baseline_sample", "peak_sample", "settled_sample",
        "load_started_unix", "load_finished_unix", "scaled_pods",
        "settled_active", "settled_hpa_current", "settled_hpa_desired",
        "prometheus_service", "prometheus_port", "observation_started_unix",
        "observation_finished_unix", "final_target",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise DeploymentError("KEDA scale evidence fields differ")
    for key in ("scaled_object_uid", "hpa_uid", "external_metric"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise DeploymentError("KEDA scale identity is malformed")
    prometheus_url(value.get("prometheus_server"))
    if (not isinstance(value.get("monitoring_namespace"), str)
            or SAFE_NAME.fullmatch(value["monitoring_namespace"]) is None
            or not isinstance(value.get("service_monitor_label"), str)):
        raise DeploymentError("KEDA Prometheus binding is malformed")
    label_assignment(value["service_monitor_label"])
    if (not isinstance(value.get("prometheus_service"), str)
            or SAFE_NAME.fullmatch(value["prometheus_service"]) is None
            or type(value.get("prometheus_port")) is not int):
        raise DeploymentError("KEDA observed Prometheus Service is malformed")
    bind_prometheus_endpoint(
        value["prometheus_server"], value["prometheus_service"],
        value["monitoring_namespace"], value["prometheus_port"],
    )
    if not isinstance(value.get("query"), str) or not value["query"]:
        raise DeploymentError("KEDA Prometheus query is malformed")
    for key in ("scaled_replicas", "settled_replicas", "load_requests",
                "load_concurrency", "max_tokens"):
        if type(value.get(key)) is not int or value[key] < 1:
            raise DeploymentError("KEDA scale counts are malformed")
    if (value["external_metric"] != "s0-prometheus-tritium_queue_pressure"
            or value["scaled_replicas"] < 2 or value["settled_replicas"] != 1
            or value["load_requests"] < value["load_concurrency"]):
        raise DeploymentError("KEDA scale transition is incomplete")
    started = value.get("load_started_unix")
    finished = value.get("load_finished_unix")
    observed_started = value.get("observation_started_unix")
    observed_finished = value.get("observation_finished_unix")
    if (any(type(item) not in {int, float} or not math.isfinite(item)
            for item in (started, finished, observed_started, observed_finished))
            or not observed_started <= started <= finished <= observed_finished):
        raise DeploymentError("KEDA load timestamps are malformed")
    samples = []
    for key in ("baseline_sample", "peak_sample", "settled_sample"):
        sample = value.get(key)
        if not isinstance(sample, dict) or set(sample) != {"timestamp", "value"}:
            raise DeploymentError("KEDA Prometheus sample fields differ")
        if any(type(sample.get(field)) not in {int, float}
               or not math.isfinite(sample[field]) for field in sample) or sample["value"] < 0:
            raise DeploymentError("KEDA Prometheus sample is malformed")
        samples.append(sample)
    baseline, peak, settled = samples
    if (not observed_started <= baseline["timestamp"] <= started + 5
            or baseline["value"] > 1 or not started - 5 <= peak["timestamp"] <= finished + 5
            or peak["value"] <= 1 or settled["timestamp"] < finished - 5
            or settled["timestamp"] > observed_finished + 5 or settled["value"] > 1):
        raise DeploymentError("KEDA Prometheus samples do not prove scale causality")
    target = value.get("target")
    final_target = value.get("final_target")
    for observed_target in (target, final_target):
        if (not isinstance(observed_target, dict)
                or set(observed_target) != {"scrape_url", "last_scrape_utc"}
                or not isinstance(observed_target.get("scrape_url"), str)
                or not observed_target["scrape_url"].endswith("/metrics")
                or not isinstance(observed_target.get("last_scrape_utc"), str)):
            raise DeploymentError("KEDA Prometheus target is malformed")
    validate_receipt_snapshot(value.get("scaled_pods"), "scaled", value["scaled_replicas"])
    if (value.get("settled_active") is not False
            or value.get("settled_hpa_current") != 1
            or value.get("settled_hpa_desired") != 1):
        raise DeploymentError("KEDA controller did not prove settled scale-down")
    return value


def exercise_keda_scale(*, kubectl_base: list[str], service: str, token: str,
                        model_id: str, prompt: str, concurrency: int,
                        max_tokens: int, timeout: float,
                        request_timeout: float, prometheus_base_url: str,
                        prometheus_query: str, namespace: str, selector: str,
                        flavor: str) -> dict[str, Any]:
    stop = threading.Event()
    count_lock = threading.Lock()
    completed = 0
    observation_started = time.time()
    baseline = None
    target = None
    baseline_deadline = time.monotonic() + timeout
    while time.monotonic() < baseline_deadline:
        try:
            scaled = run_json(
                kubectl_base + ["get", f"scaledobject/{service}", "-o", "json"]
            )
            hpa = run_json(kubectl_base + ["get", f"hpa/keda-hpa-{service}", "-o", "json"])
            deployment = run_json(
                kubectl_base + ["get", f"deployment/{service}", "-o", "json"]
            )
            candidate_target = prometheus_target(
                prometheus_base_url, namespace=namespace, service=service,
                timeout=request_timeout,
            )
            candidate_sample = prometheus_sample(
                prometheus_base_url, prometheus_query, request_timeout
            )
            target_time = datetime.fromisoformat(
                candidate_target["last_scrape_utc"]
            ).timestamp()
            if (scale_baseline(scaled, hpa, deployment)
                    and candidate_sample["value"] <= 1
                    and candidate_sample["timestamp"] >= observation_started
                    and target_time >= observation_started):
                baseline = candidate_sample
                target = candidate_target
                break
        except DeploymentError:
            pass
        time.sleep(5)
    if baseline is None or target is None:
        raise DeploymentError("fresh Prometheus/KEDA baseline deadline expired")

    def generate() -> None:
        nonlocal completed
        while not stop.is_set():
            port = free_port()
            url = f"http://127.0.0.1:{port}"
            process, _ = port_forward(
                kubectl_base + ["port-forward", f"service/{service}", f"{port}:8080",
                                "--address", "127.0.0.1"],
                time.monotonic() + min(timeout, 60), url, token,
            )
            try:
                result = request_json(
                    url + "/v1/chat/completions", token,
                    body={"model": model_id,
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0, "max_tokens": max_tokens},
                    timeout=request_timeout,
                )
                if not isinstance(result.get("choices"), list) or len(result["choices"]) != 1:
                    raise DeploymentError("KEDA load request returned malformed generation")
                with count_lock:
                    completed += 1
            finally:
                stop_forward(process)

    load_started = time.time()
    deadline = time.monotonic() + timeout
    observed = None
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(generate) for _ in range(concurrency)]
        try:
            while time.monotonic() < deadline:
                try:
                    observed = scale_snapshot(
                        run_json(kubectl_base + ["get", f"scaledobject/{service}", "-o", "json"]),
                        run_json(kubectl_base + ["get", f"hpa/keda-hpa-{service}", "-o", "json"]),
                        run_json(kubectl_base + ["get", f"deployment/{service}", "-o", "json"]),
                    )
                    observed["scaled_pods"] = pod_snapshot(
                        run_json(kubectl_base + ["get", "pods", "-l", selector, "-o", "json"]),
                        flavor, observed["scaled_replicas"],
                    )
                    break
                except DeploymentError:
                    time.sleep(5)
            if observed is None:
                raise DeploymentError("KEDA scale-out deadline expired")
        finally:
            stop.set()
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as error:
                errors.append(error)
    if errors:
        raise DeploymentError(f"KEDA load generation failed: {errors[0]}") from errors[0]
    load_finished = time.time()
    peak = prometheus_peak(
        prometheus_base_url, prometheus_query, load_started, load_finished,
        request_timeout,
    )
    if peak["value"] <= 1 or peak["timestamp"] < load_started - 5:
        raise DeploymentError("Prometheus did not record queue pressure above threshold")
    settle_deadline = time.monotonic() + timeout
    while time.monotonic() < settle_deadline:
        scaled = run_json(
            kubectl_base + ["get", f"scaledobject/{service}", "-o", "json"]
        )
        hpa = run_json(kubectl_base + ["get", f"hpa/keda-hpa-{service}", "-o", "json"])
        deployment = run_json(
            kubectl_base + ["get", f"deployment/{service}", "-o", "json"]
        )
        try:
            settled = prometheus_sample(
                prometheus_base_url, prometheus_query, request_timeout
            )
        except DeploymentError:
            time.sleep(5)
            continue
        hpa_status = hpa.get("status", {})
        if (scale_baseline(scaled, hpa, deployment)
                and settled["timestamp"] >= load_finished
                and settled["value"] <= 1):
            final_target = prometheus_target(
                prometheus_base_url, namespace=namespace, service=service,
                timeout=request_timeout,
            )
            final_target_time = datetime.fromisoformat(
                final_target["last_scrape_utc"]
            ).timestamp()
            if final_target_time < load_finished:
                time.sleep(5)
                continue
            observation_finished = time.time()
            return {
                **observed, "settled_replicas": 1, "load_requests": completed,
                "load_concurrency": concurrency, "max_tokens": max_tokens,
                "query": prometheus_query, "target": target,
                "baseline_sample": baseline, "peak_sample": peak,
                "settled_sample": settled, "load_started_unix": load_started,
                "load_finished_unix": load_finished,
                "settled_active": condition(scaled, "Active"),
                "settled_hpa_current": hpa_status.get("currentReplicas"),
                "settled_hpa_desired": hpa_status.get("desiredReplicas"),
                "observation_started_unix": observation_started,
                "observation_finished_unix": observation_finished,
                "final_target": final_target,
            }
        time.sleep(5)
    raise DeploymentError("KEDA scale-down deadline expired")


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
    if len(args.release_name) > 46:
        raise DeploymentError("release name is too long for Tritium KEDA resource names")
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
    if OCI_REPOSITORY.fullmatch(repository) is None:
        raise DeploymentError("image repository is not canonical or Helm-safe")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 7200:
        raise DeploymentError("timeout must be finite and in (0, 7200]")
    if not math.isfinite(args.request_timeout) or not 0 < args.request_timeout <= 1800:
        raise DeploymentError("request timeout must be finite and in (0, 1800]")
    if args.flavor == "cuda" and args.cuda_probe_image is None:
        raise DeploymentError("CUDA qualification requires a pinned probe image")
    if args.flavor == "cpu" and args.cuda_probe_image is not None:
        raise DeploymentError("CPU qualification cannot accept a CUDA probe image")
    if args.flavor == "cpu":
        prometheus_url(args.keda_prometheus_server)
        if SAFE_NAME.fullmatch(args.monitoring_namespace or "") is None:
            raise DeploymentError("CPU qualification requires a safe monitoring namespace")
        if SAFE_NAME.fullmatch(args.prometheus_service or "") is None:
            raise DeploymentError("CPU qualification requires a safe Prometheus service")
        if type(args.prometheus_port) is not int or not 1 <= args.prometheus_port <= 65535:
            raise DeploymentError("Prometheus port must be in [1, 65535]")
        bind_prometheus_endpoint(
            args.keda_prometheus_server, args.prometheus_service,
            args.monitoring_namespace, args.prometheus_port,
        )
        monitor_label_key, monitor_label_value = label_assignment(
            args.service_monitor_label or ""
        )
    elif any(value is not None for value in (
        args.keda_prometheus_server, args.monitoring_namespace,
        args.service_monitor_label, args.prometheus_service,
    )):
        raise DeploymentError("CUDA qualification cannot claim CPU KEDA scale evidence")
    if type(args.scale_concurrency) is not int or not 2 <= args.scale_concurrency <= 32:
        raise DeploymentError("scale concurrency must be in [2, 32]")
    if type(args.scale_max_tokens) is not int or not 2 <= args.scale_max_tokens <= 256:
        raise DeploymentError("scale max tokens must be in [2, 256]")
    if not math.isfinite(args.scale_timeout) or not 60 <= args.scale_timeout <= 1800:
        raise DeploymentError("scale timeout must be finite and in [60, 1800]")
    if not isinstance(args.prompt, str) or not args.prompt or len(args.prompt.encode()) > 1024 * 1024:
        raise DeploymentError("prompt must be non-empty and at most 1 MiB")
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
    run(kubectl_base + ["get", "--raw", "/apis/metrics.k8s.io/v1beta1"])
    if args.flavor == "cpu":
        run_json(kubectl_base + ["get", "customresourcedefinition/scaledobjects.keda.sh",
                                  "-o", "json"])
        run_json(kubectl_base + ["get", "customresourcedefinition/servicemonitors.monitoring.coreos.com",
                                  "-o", "json"])
        run_json(kubectl_base + ["get", "namespace", args.monitoring_namespace, "-o", "json"])
        run(kubectl_base + ["get", "--raw", "/apis/external.metrics.k8s.io/v1beta1"])
    source_pvc_identity = pvc_identity(
        run_json(kubectl_base + [
            "get", "persistentvolumeclaim", args.source_pvc, "-o", "json"
        ]),
        args.source_pvc,
    )
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
    prometheus_query_text = (
        f'max(tritium_queue_depth{{namespace="{args.namespace}",service="{service}"}})'
    )
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
    else:
        values += [
            "--set", "keda.enabled=true",
            "--set", "keda.minReplicaCount=1",
            "--set", "keda.maxReplicaCount=2",
            "--set", "keda.pollingInterval=5",
            "--set", "keda.cooldownPeriod=30",
            "--set", "keda.stabilizationWindowSeconds=30",
            "--set-string", f"keda.prometheus.serverAddress={args.keda_prometheus_server}",
            "--set-string", "keda.prometheus.threshold=1",
            "--set", "serviceMonitor.enabled=true",
            "--set-string", f"serviceMonitor.labels.{helm_key(monitor_label_key)}={monitor_label_value}",
            "--set-string", "serviceMonitor.ingressNamespaceSelector.matchLabels."
                            f"kubernetes\\.io/metadata\\.name={args.monitoring_namespace}",
        ]
    install = helm_base + [
        "install", args.release_name, str(chart), "--atomic", "--wait",
        "--timeout", timeout_value,
    ] + values
    lock_name = "tritium-qualify-" + hashlib.sha256(
        f"{args.context}\0{args.namespace}\0{args.release_name}".encode()
    ).hexdigest()[:16]
    lock = run_json(kubectl_base + [
        "create", "configmap", lock_name, f"--from-literal=run-id={args.run_id}",
        "-o", "json",
    ], 30)
    lock_uid = lock.get("metadata", {}).get("uid")
    if (not isinstance(lock_uid, str) or not lock_uid
            or lock.get("data") != {"run-id": args.run_id}):
        try:
            run(kubectl_base + ["delete", f"configmap/{lock_name}", "--wait=true"], 30)
        except DeploymentError:
            pass
        raise DeploymentError("qualification lock identity is malformed")
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
        watchdog_policy = watchdog_contract(deployment)
        artifact_volume = artifact_volume_contract(deployment)
        update_before = deployment_update_identity(deployment)
        if update_before["rate_limit_burst"] != 8:
            raise DeploymentError("initial deployment rate-limit burst differs")
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
            model_id = entries[0]["id"]
            generation = request_json(
                url + "/v1/chat/completions", token,
                body={"model": model_id,
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
        scale = None
        if args.flavor == "cpu":
            scaled_object = run_json(
                kubectl_base + ["get", f"scaledobject/{service}", "-o", "json"]
            )
            service_monitor = run_json(
                kubectl_base + ["get", f"servicemonitor/{service}", "-o", "json"]
            )
            validate_scale_contract(
                scaled_object, service_monitor, service=service,
                server=args.keda_prometheus_server, query=prometheus_query_text,
                auth_secret=args.auth_secret, auth_key=args.auth_key,
                monitor_label=(monitor_label_key, monitor_label_value),
            )
            prometheus_kubectl = kube(kubectl, args.context, args.monitoring_namespace)
            prometheus_port = free_port()
            prometheus_local = f"http://127.0.0.1:{prometheus_port}"
            prometheus_process = public_port_forward(
                prometheus_kubectl + [
                    "port-forward", f"service/{args.prometheus_service}",
                    f"{prometheus_port}:{args.prometheus_port}", "--address", "127.0.0.1",
                ],
                time.monotonic() + args.timeout, prometheus_local,
            )
            try:
                scale = exercise_keda_scale(
                    kubectl_base=kubectl_base, service=service, token=token,
                    model_id=model_id, prompt=args.prompt,
                    concurrency=args.scale_concurrency, max_tokens=args.scale_max_tokens,
                    timeout=args.scale_timeout, request_timeout=args.request_timeout,
                    prometheus_base_url=prometheus_local,
                    prometheus_query=prometheus_query_text,
                    namespace=args.namespace, selector=selector, flavor=args.flavor,
                )
            finally:
                stop_forward(prometheus_process)
            scale.update({
                "prometheus_server": args.keda_prometheus_server,
                "prometheus_service": args.prometheus_service,
                "prometheus_port": args.prometheus_port,
                "monitoring_namespace": args.monitoring_namespace,
                "service_monitor_label": args.service_monitor_label,
            })
            initial = pod_snapshot(
                run_json(kubectl_base + ["get", "pods", "-l", selector, "-o", "json"]),
                args.flavor,
            )
        first_pod = initial["pods"][0]["name"]
        watchdog = qualify_watchdog_restart(
            kubectl_base, pod_name=first_pod, timeout=args.timeout,
            contract=watchdog_policy,
        )
        run(kubectl_base + ["rollout", "status", f"deployment/{service}",
                            f"--timeout={timeout_value}"], args.timeout + 30)
        watchdog_port = free_port()
        watchdog_url = f"http://127.0.0.1:{watchdog_port}"
        watchdog_process, watchdog_ready = port_forward(
            kubectl_base + ["port-forward", f"service/{service}",
                            f"{watchdog_port}:8080", "--address", "127.0.0.1"],
            time.monotonic() + args.timeout, watchdog_url, token,
        )
        try:
            watchdog_startup = validate_ready(
                watchdog_ready, revision, args.flavor, args.profile,
                manifest_id["blake3"], args.release,
            )
            watchdog_generation = request_json(
                watchdog_url + "/v1/chat/completions", token,
                body={"model": model_id,
                      "messages": [{"role": "user", "content": args.prompt}],
                      "temperature": 0, "max_tokens": 1},
                timeout=args.request_timeout,
            )
            if (not isinstance(watchdog_generation.get("choices"), list)
                    or len(watchdog_generation["choices"]) != 1):
                raise DeploymentError("watchdog recovery generation did not return one choice")
            watchdog_metrics = metrics_snapshot(request(
                watchdog_url + "/metrics", token, timeout=args.request_timeout
            ).decode("utf-8"))
        finally:
            stop_forward(watchdog_process)
        if watchdog_startup != startup:
            raise DeploymentError("watchdog replacement changed immutable startup receipt")
        watchdog["startup_receipt"] = watchdog_startup
        watchdog["generation_response_sha256"] = hashlib.sha256(
            canonical(watchdog_generation)
        ).hexdigest()
        watchdog["metrics"] = watchdog_metrics
        initial = pod_snapshot(
            run_json(kubectl_base + ["get", "pods", "-l", selector, "-o", "json"]),
            args.flavor,
        )
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
            restart_generation = request_json(
                restart_url + "/v1/chat/completions", token,
                body={"model": model_id,
                      "messages": [{"role": "user", "content": args.prompt}],
                      "temperature": 0, "max_tokens": 1},
                timeout=args.request_timeout,
            )
            if (not isinstance(restart_generation.get("choices"), list)
                    or len(restart_generation["choices"]) != 1):
                raise DeploymentError("restart generation did not return one choice")
            restart_metrics_text = request(
                restart_url + "/metrics", token, timeout=args.request_timeout
            ).decode("utf-8")
            restart_recovery = {
                "generation_response_sha256": hashlib.sha256(
                    canonical(restart_generation)
                ).hexdigest(),
                "metrics": metrics_snapshot(restart_metrics_text),
            }
        finally:
            stop_forward(restart_process)
        if restart_startup != startup:
            raise DeploymentError("restart changed immutable startup receipt")
        artifact_baseline_resources = collect_resource_usage(
            kubectl_base, namespace=args.namespace, selector=selector,
            timeout=min(args.timeout, 30),
            expected_pods={pod["name"] for pod in restarted["pods"]},
        )
        artifact_fault = qualify_artifact_volume_loss(
            kubectl_base, service=service, namespace=args.namespace, selector=selector,
            contract=artifact_volume,
            previous_uids={pod["uid"] for pod in restarted["pods"]},
            timeout=args.timeout,
        )
        run(kubectl_base + ["rollout", "status", f"deployment/{service}",
                            f"--timeout={timeout_value}"], args.timeout + 30)
        artifact_recovered = pod_snapshot(
            run_json(kubectl_base + ["get", "pods", "-l", selector, "-o", "json"]),
            args.flavor,
        )
        artifact_recovered_resources = collect_resource_usage(
            kubectl_base, namespace=args.namespace, selector=selector,
            timeout=min(args.timeout, 30),
            expected_pods={pod["name"] for pod in artifact_recovered["pods"]},
        )
        artifact_port = free_port()
        artifact_url = f"http://127.0.0.1:{artifact_port}"
        artifact_process, artifact_ready = port_forward(
            kubectl_base + ["port-forward", f"service/{service}",
                            f"{artifact_port}:8080", "--address", "127.0.0.1"],
            time.monotonic() + args.timeout, artifact_url, token,
        )
        artifact_request = {
            "model": model_id,
            "messages": [{"role": "user", "content": args.prompt}],
            "temperature": 0,
            "max_tokens": 1,
        }
        try:
            artifact_startup = validate_ready(
                artifact_ready, revision, args.flavor, args.profile,
                manifest_id["blake3"], args.release,
            )
            artifact_generation = request_json(
                artifact_url + "/v1/chat/completions", token,
                body=artifact_request,
                timeout=args.request_timeout,
            )
            if (not isinstance(artifact_generation.get("choices"), list)
                    or len(artifact_generation["choices"]) != 1):
                raise DeploymentError("artifact-volume recovery generation differs")
            artifact_metrics = metrics_snapshot(request(
                artifact_url + "/metrics", token, timeout=args.request_timeout
            ).decode("utf-8"))
        finally:
            stop_forward(artifact_process)
        if artifact_startup != startup:
            raise DeploymentError("artifact-volume recovery changed startup receipt")
        artifact_scenario_started = artifact_fault.pop("_scenario_started_monotonic")
        artifact_fault["transitions"].append({
            "state": "recovery_ready",
            "elapsed_ms": (time.monotonic() - artifact_scenario_started) * 1000,
            "observed_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        })
        artifact_failure_resources = artifact_fault.pop("failure_resources")
        artifact_fault.update({
            "recovered": artifact_recovered,
            "cleanup": {"status": "restored", "source_claim": args.source_pvc},
            "startup_receipt": artifact_startup,
            "generation_response_sha256": hashlib.sha256(
                canonical(artifact_generation)
            ).hexdigest(),
            "request": request_evidence(
                model_id, args.prompt, temperature=0, max_tokens=1
            ),
            "metrics": artifact_metrics,
            "resources": {
                "baseline": artifact_baseline_resources,
                "failure": artifact_failure_resources,
                "recovered": artifact_recovered_resources,
                "high_water": {
                    "cpu_nanocores": max(
                        artifact_baseline_resources["cpu_nanocores"],
                        artifact_failure_resources["cpu_nanocores"],
                        artifact_recovered_resources["cpu_nanocores"],
                    ),
                    "memory_bytes": max(
                        artifact_baseline_resources["memory_bytes"],
                        artifact_failure_resources["memory_bytes"],
                        artifact_recovered_resources["memory_bytes"],
                    ),
                },
            },
        })
        update_before = deployment_update_identity(run_json(
            kubectl_base + ["get", f"deployment/{service}", "-o", "json"]
        ))
        update_values = ["--set", "requestLimits.rateLimitBurst=9"]
        run(helm_base + ["upgrade", args.release_name, str(chart), "--atomic", "--wait",
                         "--timeout", timeout_value] + values + update_values,
            args.timeout + 60)
        run(kubectl_base + ["rollout", "status", f"deployment/{service}",
                            f"--timeout={timeout_value}"], args.timeout + 30)
        updated_deployment = run_json(
            kubectl_base + ["get", f"deployment/{service}", "-o", "json"]
        )
        if validate_deployment(
            updated_deployment, image=args.image, manifest_sha256=manifest_sha,
            flavor=args.flavor,
        ) != deployment_uid:
            raise DeploymentError("successful update replaced deployment identity")
        update_after = deployment_update_identity(updated_deployment)
        if (update_after["rate_limit_burst"] != 9
                or update_after["generation"] <= update_before["generation"]):
            raise DeploymentError("successful update did not apply intended config delta")
        updated = pod_snapshot(
            run_json(kubectl_base + ["get", "pods", "-l", selector, "-o", "json"]),
            args.flavor,
        )
        if {pod["uid"] for pod in artifact_recovered["pods"]} & {
            pod["uid"] for pod in updated["pods"]
        }:
            raise DeploymentError("successful update retained the previous pod UID")
        update_port = free_port()
        update_url = f"http://127.0.0.1:{update_port}"
        update_process, update_ready = port_forward(
            kubectl_base + ["port-forward", f"service/{service}", f"{update_port}:8080",
                            "--address", "127.0.0.1"],
            time.monotonic() + args.timeout, update_url, token,
        )
        try:
            update_startup = validate_ready(
                update_ready, revision, args.flavor, args.profile,
                manifest_id["blake3"], args.release,
            )
        finally:
            stop_forward(update_process)
        if update_startup != startup:
            raise DeploymentError("successful update changed immutable startup receipt")
        observed_pods = (initial["pods"] + restarted["pods"]
                         + artifact_recovered["pods"] + updated["pods"])
        if scale is not None:
            observed_pods += scale["scaled_pods"]["pods"]
        node_names = {pod["node"] for pod in observed_pods}
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
                         "--timeout", timeout_value] + values + update_values + [
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
        owns_lock = False
        try:
            current_lock = run_json(
                kubectl_base + ["get", f"configmap/{lock_name}", "-o", "json"]
            )
            owns_lock = (current_lock.get("metadata", {}).get("uid") == lock_uid
                         and current_lock.get("data") == {"run-id": args.run_id})
        except DeploymentError:
            owns_lock = False
        if not owns_lock and not active_error:
            raise DeploymentError("qualification lock ownership was lost")
        if install_attempted and owns_lock:
            try:
                run(helm_base + ["uninstall", args.release_name, "--wait",
                                 "--ignore-not-found", "--timeout", timeout_value],
                    args.timeout + 60)
                cleanup_passed = not run(
                    helm_base + ["list", "--filter", f"^{args.release_name}$", "-q"]
                )
                exact_resources = [
                    f"deployment/{service}", f"service/{service}", f"pdb/{service}",
                    f"networkpolicy/{service}",
                ]
                if args.flavor == "cpu":
                    exact_resources += [
                        f"hpa/keda-hpa-{service}", f"scaledobject/{service}",
                        f"servicemonitor/{service}",
                    ]
                remaining_exact = run(
                    kubectl_base + ["get", *exact_resources, "-o", "name",
                                    "--ignore-not-found=true"]
                )
                remaining_pods = run(kubectl_base + [
                    "get", "pods", "-l", selector, "-o", "name",
                    "--ignore-not-found=true",
                ])
                cleanup_passed = cleanup_passed and not remaining_exact and not remaining_pods
                if not cleanup_passed and not active_error:
                    raise DeploymentError("Helm release cleanup is incomplete")
            except DeploymentError:
                if not active_error:
                    raise
        if owns_lock:
            try:
                run(kubectl_base + ["delete", f"configmap/{lock_name}", "--wait=true"], 30)
                lock_remaining = run(kubectl_base + [
                    "get", f"configmap/{lock_name}", "-o", "name",
                    "--ignore-not-found=true",
                ])
                cleanup_passed = cleanup_passed and not lock_remaining
            except DeploymentError:
                cleanup_passed = False
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
        "bundle_manifest_artifact": initial_files["manifest"],
        "build_receipt_artifact": initial_files["build_receipt"],
        "image": args.image, "manifest": manifest_id,
        "cluster": {"context": args.context, "namespace": args.namespace,
                    "namespace_uid": namespace_uid,
                    "server_git_version": server_version.get("gitVersion"),
                    "server_platform": server_version.get("platform"),
                    "nodes": nodes, "cuda_node": cuda_node},
        "tools": {"kubectl_sha256": sha256(kubectl), "helm_sha256": sha256(helm_bin),
                  "helm_version": run([str(helm_bin), "version", "--short"])},
        "workload": {"release_name": args.release_name, "deployment_uid": deployment_uid,
                     "qualification_lock_uid": lock_uid, "source_pvc": args.source_pvc,
                     "source_pvc_identity": source_pvc_identity,
                     "model_id": model_id,
                     "initial": initial, "restarted": restarted, "updated": updated,
                     "watchdog_replacement": watchdog,
                     "artifact_volume_loss": artifact_fault,
                     "startup_receipt": startup, "restart_startup_receipt": restart_startup,
                     "update_startup_receipt": update_startup,
                     "update_strategy": "Recreate" if args.flavor == "cuda" else "RollingUpdate",
                     "update_config": {"before": update_before, "after": update_after},
                     "rollback_startup_receipt": rollback_startup,
                     "metrics": metric_evidence, "restart_recovery": restart_recovery,
                     "scale": scale, "helm_history": history,
                     "prior_helm_revision": prior_revision,
                     "failed_manifest_sha256": wrong_manifest,
                     "failed_image_digest": "sha256:" + wrong_image,
                     "failed_upgrade_output_sha256": failure_sha},
        "checks": expected_checks(args.flavor), "result": "pass",
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
        "bundle_manifest_artifact", "build_receipt_artifact", "image", "manifest",
        "cluster", "tools", "workload", "checks", "result",
    }
    if not isinstance(receipt, dict) or set(receipt) != fields:
        raise DeploymentError("deployment receipt fields differ")
    if receipt.get("schema") != SCHEMA or receipt.get("result") != "pass":
        raise DeploymentError("deployment receipt schema or result differs")
    if receipt.get("release") != release or receipt.get("source_revision") != revision:
        raise DeploymentError("deployment receipt release identity differs")
    if (receipt.get("flavor") not in {"cpu", "cuda"}
            or receipt.get("checks") != expected_checks(receipt["flavor"])):
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
    for key, path, kind in (
        ("image_artifact", image_path, "oci-image"),
        ("bundle_manifest_artifact", manifest_path, "bundle-manifest"),
        ("build_receipt_artifact", build_receipt, "oci-build-receipt"),
    ):
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
    receipt_repository = image.rpartition("@")[0]
    if OCI_REPOSITORY.fullmatch(receipt_repository) is None:
        raise DeploymentError("deployment image repository is not canonical")
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
        "release_name", "deployment_uid", "qualification_lock_uid",
        "source_pvc", "source_pvc_identity", "model_id",
        "initial", "restarted", "updated",
        "watchdog_replacement", "artifact_volume_loss",
        "startup_receipt", "restart_startup_receipt", "update_startup_receipt",
        "update_strategy", "update_config", "rollback_startup_receipt", "metrics",
        "restart_recovery", "scale", "helm_history",
        "prior_helm_revision", "failed_manifest_sha256", "failed_image_digest",
        "failed_upgrade_output_sha256",
    }:
        raise DeploymentError("deployment workload fields differ")
    for key in (
        "deployment_uid", "qualification_lock_uid", "release_name", "source_pvc", "model_id"
    ):
        if not isinstance(workload.get(key), str) or not workload[key]:
            raise DeploymentError("deployment workload identity is malformed")
    if SAFE_NAME.fullmatch(workload["source_pvc"]) is None:
        raise DeploymentError("deployment source PVC identity is malformed")
    validate_pvc_identity(workload.get("source_pvc_identity"), workload["source_pvc"])
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
    updated = validate_receipt_snapshot(workload.get("updated"), "updated")
    if {pod["uid"] for pod in initial["pods"]} & {
        pod.get("uid") for pod in restarted["pods"] if isinstance(pod, dict)
    }:
        raise DeploymentError("deployment restart evidence retains old pod UID")
    if {pod["uid"] for pod in restarted["pods"]} & {pod["uid"] for pod in updated["pods"]}:
        raise DeploymentError("deployment update evidence retains old pod UID")
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
    watchdog = workload.get("watchdog_replacement")
    watchdog_fields = {
        "pod_uid", "container_id_before", "container_id_after",
        "restart_count_before", "restart_count_after", "last_exit_code",
        "fault_command_sha256", "replacement_ms", "startup_receipt",
        "generation_response_sha256", "metrics", "watchdog",
    }
    if (not isinstance(watchdog, dict) or set(watchdog) != watchdog_fields
            or not isinstance(watchdog.get("pod_uid"), str) or not watchdog["pod_uid"]
            or not isinstance(watchdog.get("container_id_before"), str)
            or not watchdog["container_id_before"]
            or not isinstance(watchdog.get("container_id_after"), str)
            or not watchdog["container_id_after"]
            or watchdog["container_id_before"] == watchdog["container_id_after"]
            or type(watchdog.get("restart_count_before")) is not int
            or type(watchdog.get("restart_count_after")) is not int
            or watchdog["restart_count_before"] < 0
            or watchdog["restart_count_after"] != watchdog["restart_count_before"] + 1
            or watchdog.get("last_exit_code") != 137
            or type(watchdog.get("replacement_ms")) not in {int, float}
            or not math.isfinite(watchdog["replacement_ms"])
            or watchdog["replacement_ms"] <= 0):
        raise DeploymentError("deployment watchdog replacement evidence is malformed")
    watchdog_policy = watchdog.get("watchdog")
    policy_fields = {
        "period_seconds", "timeout_seconds", "failure_threshold",
        "escalation_seconds", "scheduling_allowance_ms", "budget_ms",
    }
    if (not isinstance(watchdog_policy, dict) or set(watchdog_policy) != policy_fields
            or any(type(watchdog_policy.get(field)) is not int for field in policy_fields)
            or not 1 <= watchdog_policy["period_seconds"] <= 60
            or not 1 <= watchdog_policy["timeout_seconds"] <= 30
            or not 1 <= watchdog_policy["failure_threshold"] <= 10
            or watchdog_policy["escalation_seconds"] != watchdog_policy["timeout_seconds"]
            or watchdog_policy["scheduling_allowance_ms"]
            != WATCHDOG_SCHEDULING_ALLOWANCE_MS):
        raise DeploymentError("deployment watchdog policy evidence is malformed")
    expected_watchdog_budget = (
        watchdog_policy["failure_threshold"]
        * (watchdog_policy["period_seconds"] + watchdog_policy["timeout_seconds"])
        + watchdog_policy["escalation_seconds"]
    ) * 1000 + WATCHDOG_SCHEDULING_ALLOWANCE_MS
    if watchdog_policy != WATCHDOG_POLICY:
        raise DeploymentError("deployment watchdog policy differs from release policy")
    if (watchdog_policy["budget_ms"] != expected_watchdog_budget
            or watchdog["replacement_ms"] > expected_watchdog_budget):
        raise DeploymentError("deployment watchdog replacement exceeded its budget")
    fault_command_sha256 = exact_hex(
        watchdog.get("fault_command_sha256"), 64, "watchdog fault command SHA-256"
    )
    if fault_command_sha256 != hashlib.sha256(WATCHDOG_FAULT_COMMAND.encode()).hexdigest():
        raise DeploymentError("deployment watchdog fault command differs")
    exact_hex(
        watchdog.get("generation_response_sha256"), 64,
        "watchdog generation response SHA-256",
    )
    validate_ready(
        {"status": "ready", "release_gate": "production_artifact_admitted",
         "startup_receipt": watchdog.get("startup_receipt")},
        revision, receipt["flavor"], receipt["profile"], manifest["blake3"], release,
    )
    if watchdog["startup_receipt"] != workload["startup_receipt"]:
        raise DeploymentError("deployment watchdog changed startup receipt")
    initial_uids = {pod["uid"] for pod in initial["pods"]}
    if (watchdog["pod_uid"] not in initial_uids
            or initial["pods"][0]["restarts"] < watchdog["restart_count_after"]):
        raise DeploymentError("deployment watchdog pod differs from initial snapshot")
    artifact_fault = workload.get("artifact_volume_loss")
    artifact_fields = {
        "source_claim", "missing_claim", "volume_index", "absence",
        "fault_patch_sha256", "observation_budget_ms", "observation_ms", "pending",
        "recovered", "cleanup", "startup_receipt", "generation_response_sha256",
        "metrics", "resources", "request", "transitions",
    }
    if (not isinstance(artifact_fault, dict) or set(artifact_fault) != artifact_fields
            or artifact_fault.get("source_claim") != workload["source_pvc"]
            or not isinstance(artifact_fault.get("missing_claim"), str)
            or SAFE_NAME.fullmatch(artifact_fault["missing_claim"]) is None
            or not artifact_fault["missing_claim"].startswith("tritium-missing-")
            or artifact_fault["missing_claim"] == artifact_fault["source_claim"]
            or artifact_fault.get("volume_index") != 0
            or artifact_fault.get("observation_budget_ms")
            != ARTIFACT_VOLUME_OBSERVATION_BUDGET_MS
            or type(artifact_fault.get("observation_ms")) not in {int, float}
            or not math.isfinite(artifact_fault["observation_ms"])
            or not 0 < artifact_fault["observation_ms"]
            <= ARTIFACT_VOLUME_OBSERVATION_BUDGET_MS):
        raise DeploymentError("deployment artifact-volume evidence is malformed")
    absence = artifact_fault.get("absence")
    if absence != {
        "status": "NotFound", "output_sha256": hashlib.sha256(b"").hexdigest()
    }:
        raise DeploymentError("deployment missing artifact PVC evidence differs")
    patch_payload = canonical([{
        "op": "replace",
        "path": "/spec/template/spec/volumes/0/persistentVolumeClaim/claimName",
        "value": artifact_fault["missing_claim"],
    }]).decode().strip()
    if artifact_fault.get("fault_patch_sha256") != hashlib.sha256(
        patch_payload.encode()
    ).hexdigest():
        raise DeploymentError("deployment artifact-volume fault patch differs")
    pending = artifact_fault.get("pending")
    if (not isinstance(pending, dict) or set(pending) != {
            "pod_name", "pod_uid", "reason", "message_sha256"
        } or any(not isinstance(pending.get(key), str) or not pending[key]
                 for key in ("pod_name", "pod_uid"))
            or pending.get("reason") != "Unschedulable"
            or pending["pod_uid"] in {pod["uid"] for pod in restarted["pods"]}):
        raise DeploymentError("deployment artifact-volume pending pod differs")
    exact_hex(pending.get("message_sha256"), 64, "artifact-volume scheduler message SHA-256")
    artifact_recovered = validate_receipt_snapshot(
        artifact_fault.get("recovered"), "artifact-volume recovered"
    )
    if pending["pod_uid"] in {pod["uid"] for pod in artifact_recovered["pods"]}:
        raise DeploymentError("deployment retained artifact-volume failure pod")
    if (receipt["flavor"] == "cuda"
            and {pod["uid"] for pod in restarted["pods"]}
            & {pod["uid"] for pod in artifact_recovered["pods"]}):
        raise DeploymentError("CUDA artifact-volume recovery retained old pod")
    validate_ready(
        {"status": "ready", "release_gate": "production_artifact_admitted",
         "startup_receipt": artifact_fault.get("startup_receipt")},
        revision, receipt["flavor"], receipt["profile"], manifest["blake3"], release,
    )
    if artifact_fault["startup_receipt"] != workload["startup_receipt"]:
        raise DeploymentError("deployment artifact-volume recovery changed startup receipt")
    if artifact_fault.get("cleanup") != {
        "status": "restored", "source_claim": workload["source_pvc"]
    }:
        raise DeploymentError("deployment artifact-volume cleanup differs")
    exact_hex(
        artifact_fault.get("generation_response_sha256"), 64,
        "artifact-volume generation response SHA-256",
    )
    request = artifact_fault.get("request")
    request_fields = {
        "model", "prompt_sha256", "prompt_bytes", "temperature", "max_tokens",
        "descriptor_sha256",
    }
    if (not isinstance(request, dict) or set(request) != request_fields
            or request.get("model") != workload["model_id"]
            or type(request.get("prompt_bytes")) is not int
            or not 0 < request["prompt_bytes"] <= 1024 * 1024
            or type(request.get("temperature")) is not int
            or type(request.get("max_tokens")) is not int
            or request["temperature"] != 0 or request["max_tokens"] != 1):
        raise DeploymentError("deployment artifact-volume request evidence is malformed")
    exact_hex(request.get("prompt_sha256"), 64, "artifact-volume prompt SHA-256")
    descriptor = {key: request[key] for key in request_fields - {"descriptor_sha256"}}
    if request.get("descriptor_sha256") != hashlib.sha256(canonical(descriptor)).hexdigest():
        raise DeploymentError("deployment artifact-volume request descriptor differs")
    transitions = validate_transition_trace(artifact_fault.get("transitions"))
    if transitions[-1]["elapsed_ms"] > receipt["duration_ms"]:
        raise DeploymentError("deployment artifact-volume transitions exceed run duration")
    resources = artifact_fault.get("resources")
    if not isinstance(resources, dict) or set(resources) != {
        "baseline", "failure", "recovered", "high_water"
    }:
        raise DeploymentError("deployment artifact-volume resource fields differ")
    baseline_resources = validate_resource_sample(resources["baseline"], "baseline")
    failure_resources = validate_resource_sample(
        resources["failure"], "failure", allow_empty=True
    )
    recovered_resources = validate_resource_sample(resources["recovered"], "recovered")
    if (set(baseline_resources["pod_names"])
            != {pod["name"] for pod in restarted["pods"]}
            or not set(failure_resources["pod_names"]).issubset(
                {pod["name"] for pod in restarted["pods"]}
            )
            or set(recovered_resources["pod_names"])
            != {pod["name"] for pod in artifact_recovered["pods"]}):
        raise DeploymentError("deployment artifact-volume resource pod identity differs")
    sample_times = [datetime.fromisoformat(sample["sampled_at_utc"]) for sample in (
        baseline_resources, failure_resources, recovered_resources
    )]
    if sample_times != sorted(sample_times):
        raise DeploymentError("deployment artifact-volume resource timing differs")
    expected_high_water = {
        name: max(sample[name] for sample in (
            baseline_resources, failure_resources, recovered_resources
        ))
        for name in ("cpu_nanocores", "memory_bytes")
    }
    if resources["high_water"] != expected_high_water:
        raise DeploymentError("deployment artifact-volume high-water evidence differs")
    validate_ready(
        {"status": "ready", "release_gate": "production_artifact_admitted",
         "startup_receipt": workload.get("update_startup_receipt")},
        revision, receipt["flavor"], receipt["profile"], manifest["blake3"], release,
    )
    expected_strategy = "Recreate" if receipt["flavor"] == "cuda" else "RollingUpdate"
    if (workload["update_startup_receipt"] != workload["startup_receipt"]
            or workload.get("update_strategy") != expected_strategy):
        raise DeploymentError("deployment successful-update evidence differs")
    update_config = workload.get("update_config")
    if not isinstance(update_config, dict) or set(update_config) != {"before", "after"}:
        raise DeploymentError("deployment update config fields differ")
    before = update_config.get("before")
    after = update_config.get("after")
    if (not isinstance(before, dict) or not isinstance(after, dict)
            or set(before) != {"generation", "rate_limit_burst"}
            or set(after) != {"generation", "rate_limit_burst"}
            or before.get("rate_limit_burst") != 8 or after.get("rate_limit_burst") != 9
            or type(before.get("generation")) is not int
            or type(after.get("generation")) is not int
            or after["generation"] <= before["generation"]):
        raise DeploymentError("deployment intended update delta is malformed")
    validate_ready(
        {"status": "ready", "release_gate": "production_artifact_admitted",
         "startup_receipt": workload.get("rollback_startup_receipt")},
        revision, receipt["flavor"], receipt["profile"], manifest["blake3"], release,
    )
    if workload["rollback_startup_receipt"] != workload["startup_receipt"]:
        raise DeploymentError("deployment rollback changed startup receipt")
    metrics = workload.get("metrics")
    validate_metrics_evidence(metrics, "")
    recovery = workload.get("restart_recovery")
    if not isinstance(recovery, dict) or set(recovery) != {
        "generation_response_sha256", "metrics"
    }:
        raise DeploymentError("deployment restart recovery fields differ")
    exact_hex(
        recovery.get("generation_response_sha256"), 64,
        "restart generation response SHA-256",
    )
    validate_metrics_evidence(recovery.get("metrics"), "restart")
    validate_metrics_evidence(watchdog.get("metrics"), "watchdog")
    validate_metrics_evidence(artifact_fault.get("metrics"), "artifact-volume")
    if receipt["flavor"] == "cpu":
        scale_receipt = validate_scale_receipt(workload.get("scale"))
        expected_query = (
            f'max(tritium_queue_depth{{namespace="{cluster["namespace"]}",'
            f'service="{workload["release_name"]}-tritium"}})'
        )
        if scale_receipt["query"] != expected_query:
            raise DeploymentError("KEDA query does not bind deployed namespace and service")
        try:
            target_scrape = datetime.fromisoformat(
                scale_receipt["target"]["last_scrape_utc"]
            ).timestamp()
            final_target_scrape = datetime.fromisoformat(
                scale_receipt["final_target"]["last_scrape_utc"]
            ).timestamp()
        except ValueError as error:
            raise DeploymentError("KEDA target timestamp is malformed") from error
        if (not scale_receipt["observation_started_unix"] <= target_scrape
                <= scale_receipt["load_started_unix"] + 5
                or not scale_receipt["load_finished_unix"] <= final_target_scrape
                <= scale_receipt["observation_finished_unix"] + 5):
            raise DeploymentError("KEDA target timestamp is outside qualification window")
    elif workload.get("scale") is not None:
        raise DeploymentError("CUDA deployment cannot contain CPU KEDA scale evidence")
    validate_helm_history(workload.get("helm_history"), workload["prior_helm_revision"])
    observed_pods = (initial["pods"] + restarted["pods"]
                     + artifact_recovered["pods"] + updated["pods"])
    if receipt["flavor"] == "cpu":
        observed_pods += workload["scale"]["scaled_pods"]["pods"]
    node_names = {pod["node"] for pod in observed_pods}
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
    parser.add_argument("--keda-prometheus-server")
    parser.add_argument("--monitoring-namespace")
    parser.add_argument("--prometheus-service")
    parser.add_argument("--prometheus-port", type=int, default=9090)
    parser.add_argument("--service-monitor-label")
    parser.add_argument("--scale-timeout", type=float, default=600)
    parser.add_argument("--scale-concurrency", type=int, default=8)
    parser.add_argument("--scale-max-tokens", type=int, default=256)
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
