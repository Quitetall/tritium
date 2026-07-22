#!/usr/bin/env python3
"""Run and seal exact-image Tritium production serving evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


SCHEMA = "tritium.oci-runtime-qualification.v1"
HEX = frozenset("0123456789abcdef")
CHECKS = (
    "production-readiness", "models", "buffered-generation", "sse-generation",
    "readonly-rootfs", "drop-all-capabilities", "no-new-privileges",
    "readonly-bundle", "sigterm-drain",
)
REQUIRED_STARTUP = {
    "schema_version", "artifact_kind", "server_source_revision", "server_build_id",
    "model_source_revision", "manifest_package_id", "salt_package_id",
    "preserved_package_id", "config_package_id", "profile", "codec",
    "backend_policy", "effective_backend", "physical_device_id",
    "loaded_bundle_bytes", "resident_bytes", "self_test_digest",
}


class QualificationError(ValueError):
    """Runtime qualification failed closed."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exact_hex(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(c not in HEX for c in value):
        raise QualificationError(f"{label} must be {length} lowercase hexadecimal characters")
    return value


def run(command: list[str], *, env: dict[str, str] | None = None,
        timeout: float = 120.0) -> str:
    try:
        result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        raise QualificationError(f"command failed: {command[0]}: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip()[-2000:]
        raise QualificationError(f"command failed ({result.returncode}): {' '.join(command)}: {detail}")
    return result.stdout.strip()


def request_json(url: str, token: str, body: dict[str, Any] | None = None,
                 timeout: float = 30.0) -> dict[str, Any]:
    data = None if body is None else canonical(body)
    request = urllib.request.Request(url, data=data)
    request.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise QualificationError(f"request failed: {url}: {error}") from error
    if not isinstance(value, dict):
        raise QualificationError(f"response is not a JSON object: {url}")
    return value


def validate_ready(value: dict[str, Any], revision: str, flavor: str,
                   profile: str, manifest_blake3: str,
                   release: str | None = None) -> dict[str, Any]:
    if value.get("status") != "ready" or value.get("release_gate") != "production_artifact_admitted":
        raise QualificationError("readiness did not admit a production artifact")
    receipt = value.get("startup_receipt")
    if not isinstance(receipt, dict) or set(receipt) != REQUIRED_STARTUP:
        raise QualificationError("startup receipt fields do not match schema v1")
    if receipt.get("schema_version") != 1 or receipt.get("server_source_revision") != revision:
        raise QualificationError("startup receipt source identity differs")
    if release is not None and receipt.get("server_build_id") != f"tritium-serve:{release}:{revision}":
        raise QualificationError("startup receipt release build identity differs")
    if receipt.get("backend_policy") != flavor or receipt.get("effective_backend") != flavor:
        raise QualificationError("startup receipt backend policy differs")
    if receipt.get("profile") != profile or receipt.get("manifest_package_id") != manifest_blake3:
        raise QualificationError("startup receipt artifact identity differs")
    if type(receipt.get("loaded_bundle_bytes")) is not int or receipt["loaded_bundle_bytes"] <= 0:
        raise QualificationError("startup receipt loaded byte ledger is invalid")
    if type(receipt.get("resident_bytes")) is not int or receipt["resident_bytes"] <= 0:
        raise QualificationError("startup receipt resident byte ledger is invalid")
    exact_hex(receipt.get("self_test_digest"), 64, "self-test digest")
    return receipt


def manifest_identity(path: Path, digest_tool: str) -> dict[str, Any]:
    value = json.loads(run([digest_tool, "release", "digest", str(path)], timeout=300))
    if value.get("schema") != "tritium.file-identity.v1":
        raise QualificationError("digest tool returned wrong schema")
    exact_hex(value.get("sha256"), 64, "manifest SHA-256")
    exact_hex(value.get("blake3"), 64, "manifest BLAKE3")
    return value


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def atomic_create(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise QualificationError("refusing to overwrite output")
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
    if args.flavor not in {"cpu", "cuda"}:
        raise QualificationError("flavor must be cpu or cuda")
    if args.profile not in {"compact-v1", "near-lossless-v1"}:
        raise QualificationError("profile is not admitted")
    if re.fullmatch(r"1\.1\.0-rc\.(0|[1-9][0-9]*)", args.release) is None:
        raise QualificationError("release must be a canonical 1.1.0 release candidate")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.run_id) is None:
        raise QualificationError("run ID is not a safe canonical identifier")
    if min(args.startup_timeout, args.request_timeout, args.shutdown_timeout) <= 0:
        raise QualificationError("timeouts must be positive")
    revision = exact_hex(args.source_revision, 40, "source revision")
    image_name, separator, image_digest = args.image.rpartition("@")
    if separator != "@" or not image_name or not image_digest.startswith("sha256:"):
        raise QualificationError("image must be an exact repository digest reference")
    exact_hex(image_digest[7:], 64, "image digest")
    bundle = args.bundle.resolve(strict=True)
    if args.bundle.is_symlink() or not bundle.is_dir():
        raise QualificationError("bundle must be an ordinary directory")
    identity = manifest_identity(bundle / "tritium.json", args.digest_tool)
    inspect = json.loads(run(["docker", "image", "inspect", args.image]))
    if not isinstance(inspect, list) or len(inspect) != 1:
        raise QualificationError("image reference did not resolve exactly once")
    exact_hex(str(inspect[0].get("Id", "")).removeprefix("sha256:"), 64, "image ID")
    repo_digests = inspect[0].get("RepoDigests", [])
    if args.image not in repo_digests:
        raise QualificationError("Docker image metadata does not contain requested digest")

    project = "tritium-qualify-" + secrets.token_hex(6)
    token = secrets.token_urlsafe(32)
    port = free_port()
    environment = os.environ.copy()
    environment.update({
        "TRITIUM_IMAGE": args.image,
        "TRITIUM_BUNDLE": str(bundle),
        "TRITIUM_PROFILE": args.profile,
        "TRITIUM_AUTH_TOKEN": token,
        "TRITIUM_PORT": str(port),
    })
    compose = [str(Path(__file__).with_name("run-oci-compose")), args.flavor,
               "-p", project]
    started_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.monotonic()
    try:
        run(compose + ["up", "-d"], env=environment, timeout=args.startup_timeout)
        deadline = time.monotonic() + args.startup_timeout
        ready = None
        while time.monotonic() < deadline:
            try:
                ready = request_json(f"http://127.0.0.1:{port}/readyz", token, timeout=5)
                break
            except QualificationError:
                time.sleep(1)
        if ready is None:
            raise QualificationError("production readiness deadline expired")
        startup = validate_ready(
            ready, revision, args.flavor, args.profile, identity["blake3"], args.release
        )
        ready_at = time.monotonic()
        models = request_json(f"http://127.0.0.1:{port}/v1/models", token)
        entries = models.get("data")
        if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
            raise QualificationError("model listing is not singular")
        model_id = entries[0].get("id")
        payload = {"model": model_id, "messages": [{"role": "user", "content": args.prompt}],
                   "temperature": 0, "max_tokens": 1}
        buffered = request_json(f"http://127.0.0.1:{port}/v1/chat/completions", token, payload,
                                timeout=args.request_timeout)
        if not isinstance(buffered.get("choices"), list) or len(buffered["choices"]) != 1:
            raise QualificationError("buffered generation did not return one choice")
        stream_request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=canonical({**payload, "stream": True}),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(stream_request, timeout=args.request_timeout) as response:
            stream = response.read().decode("utf-8")
        if "data: [DONE]" not in stream or '"choices"' not in stream:
            raise QualificationError("streaming generation lacked choice data or terminal marker")
        container = run(compose + ["ps", "-q", "tritium"], env=environment)
        container_inspect = json.loads(run(["docker", "inspect", container]))[0]
        host_config = container_inspect.get("HostConfig", {})
        container_config = container_inspect.get("Config", {})
        mounts = container_inspect.get("Mounts", [])
        if host_config.get("ReadonlyRootfs") is not True:
            raise QualificationError("container root filesystem is writable")
        if container_config.get("User") in {None, "", "0", "0:0"}:
            raise QualificationError("container runtime user is root")
        if "ALL" not in (host_config.get("CapDrop") or []):
            raise QualificationError("container does not drop all capabilities")
        if not any("no-new-privileges" in item for item in host_config.get("SecurityOpt") or []):
            raise QualificationError("container lacks no-new-privileges")
        if not any(item.get("Destination") == "/models/bundle" and item.get("RW") is False for item in mounts):
            raise QualificationError("bundle mount is not read-only")
        stopped = time.monotonic()
        run(["docker", "kill", "--signal", "TERM", container])
        exit_code = run(["docker", "wait", container], timeout=args.shutdown_timeout)
        shutdown_ms = (time.monotonic() - stopped) * 1000
        if exit_code != "0" or shutdown_ms > args.shutdown_timeout * 1000:
            raise QualificationError("SIGTERM shutdown exceeded budget or exited unsuccessfully")
    finally:
        try:
            subprocess.run(compose + ["down", "--volumes", "--remove-orphans"], env=environment,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
                           check=False)
        except (OSError, subprocess.SubprocessError):
            pass

    machine_source = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
    machine = hashlib.sha256(machine_source.encode()).hexdigest()
    gpu = None
    if args.flavor == "cuda":
        fields = run([
            "nvidia-smi", "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader", "--id=0",
        ]).split(", ")
        if len(fields) != 3 or not fields[0].startswith("GPU-"):
            raise QualificationError("CUDA qualification lacks physical NVIDIA identity")
        gpu = {"uuid": fields[0], "name": fields[1], "driver_version": fields[2]}
    receipt = {
        "schema": SCHEMA, "release": args.release, "source_revision": revision,
        "run_id": args.run_id, "flavor": args.flavor, "image": args.image,
        "image_id": inspect[0].get("Id"), "manifest": identity,
        "profile": args.profile, "startup_receipt": startup,
        "checks": list(CHECKS),
        "started_at_utc": started_at_utc,
        "timing": {"startup_ms": (ready_at - started) * 1000, "shutdown_ms": shutdown_ms},
        "machine": {"id": "sha256:" + machine, "system": platform.system(),
                    "architecture": platform.machine(),
                    "docker_server": run(["docker", "version", "--format", "{{.Server.Version}}"]),
                    "gpu": gpu},
        "result": "pass",
    }
    unsigned = canonical(receipt)
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(unsigned).hexdigest()
    validate_receipt(receipt)
    return receipt


def validate_receipt(receipt: dict[str, Any]) -> None:
    expected = {
        "schema", "release", "source_revision", "run_id", "flavor", "image", "image_id",
        "manifest", "profile", "startup_receipt", "checks", "started_at_utc", "timing",
        "machine", "result", "receipt_id",
    }
    if set(receipt) != expected or receipt.get("schema") != SCHEMA or receipt.get("result") != "pass":
        raise QualificationError("runtime receipt fields or disposition differ")
    if receipt.get("checks") != list(CHECKS):
        raise QualificationError("runtime receipt checks differ")
    supplied = receipt.get("receipt_id")
    if not isinstance(supplied, str) or not supplied.startswith("sha256:"):
        raise QualificationError("runtime receipt ID is malformed")
    exact_hex(supplied[7:], 64, "runtime receipt ID")
    unsigned = dict(receipt)
    del unsigned["receipt_id"]
    expected_id = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if not secrets.compare_digest(supplied, expected_id):
        raise QualificationError("runtime receipt content digest differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flavor", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--profile", default="compact-v1")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--digest-tool", default=os.environ.get("TRITIUM_BIN", "tritium"))
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--shutdown-timeout", type=float, default=35)
    args = parser.parse_args()
    try:
        receipt = qualify(args)
        atomic_create(args.output, canonical(receipt))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"qualify-oci-runtime: BLOCKED: {error}", file=sys.stderr)
        return 1
    print(f"qualify-oci-runtime: PASS: {receipt['receipt_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
