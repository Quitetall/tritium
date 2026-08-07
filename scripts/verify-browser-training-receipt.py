#!/usr/bin/env python3
"""Strict physical Chrome/Firefox/Safari WebGPU training receipt validator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import runpy
from typing import Any


SCHEMA = "tritium.browser-training-qualification.v1"
MANIFEST_DIGEST = "9093a1a7f9a3422c399943782aadf4df6b11833cf2253db0db56ff2d9dedb098"
VECTOR_DIGEST = "38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7"
VECTOR_SOURCE_SHA256 = "eae8fab4778ad00f32a5ee0984ae6620960751029118dd9d7b97ffd36a502d0d"
VECTOR_METADATA_SHA256 = "b889648bd7fe39a4413aefcf4a4d77e17e2ba81e4f69b91c703c4d2483323536"
VECTOR_CORPUS_PATH = (
    Path(__file__).resolve().parents[1]
    / "crates/tritium-spec/data/training/v2/vectors/v2.json"
)
VECTOR_CORPUS_MIRROR_PATH = (
    Path(__file__).resolve().parents[1] / "spec/training/v2/vectors/v2.json"
)
VECTOR_METADATA_PATH = (
    Path(__file__).resolve().parent
    / "data/browser-training-vector-inventory-v1.json"
)
NPM_VERIFIER = runpy.run_path(
    Path(__file__).with_name("verify-npm-archive-receipt.py")
)
validate_npm_receipt_value = NPM_VERIFIER["validate_receipt_value"]
NpmReceiptError = NPM_VERIFIER["NpmReceiptError"]
TOP_FIELDS = {
    "schema",
    "receipt_id",
    "result",
    "release",
    "source_revision",
    "run_id",
    "artifact",
    "manifest_digest",
    "vector_digest",
    "lanes",
}
ARTIFACT_FIELDS = {"kind", "name", "bytes", "sha256"}
LANE_FIELDS = {
    "engine",
    "browser_version",
    "os",
    "adapter",
    "limits",
    "case_counts",
    "lifecycle",
    "faults",
    "trace",
}
OS_FIELDS = {"name", "version", "architecture"}
ADAPTER_FIELDS = {"vendor", "architecture", "device", "description", "software"}
LIMIT_FIELDS = {
    "max_buffer_size",
    "max_storage_buffer_binding_size",
    "max_compute_workgroups_per_dimension",
    "max_storage_buffers_per_shader_stage",
}
CASE_FIELDS = {"valid", "invalid", "skipped"}
LIFECYCLE_FIELDS = {
    "prepare",
    "forward",
    "backward",
    "optimizer_step",
    "checkpoint_resume",
    "export_reload",
    "native_artifact_parity",
}
FAULT_FIELDS = {
    "device_loss",
    "allocation_failure",
    "malformed_checkpoint",
    "malformed_salt",
    "cancellation",
    "out_of_order",
}
TRACE_FIELDS = {
    "file",
    "bytes",
    "sha256",
    "steady_state_readbacks",
    "wasm_dispatches",
    "explicit_readbacks",
    "peak_buffer_bytes",
}
ENGINES = ("chrome", "firefox", "safari")
HEX = frozenset("0123456789abcdef")
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_TRACE_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
TRACE_EVIDENCE_FIELDS = {
    "schema",
    "run_id",
    "engine",
    "source_revision",
    "archive",
    "npm_receipt",
    "native_receipt",
    "native_reference",
    "webdriver_capabilities",
    "browser_trace",
}
TRACE_ARCHIVE_FIELDS = {"name", "bytes", "sha256"}
NATIVE_REFERENCE_FIELDS = {
    "schema",
    "scenarioId",
    "sourceRevision",
    "backend",
    "backendId",
    "backendBuild",
    "physicalDevice",
    "artifactName",
    "artifactBytes",
    "artifactSha256",
    "receiptId",
    "receiptDigest",
    "export",
    "reload",
}
NATIVE_LIFECYCLE_FIELDS = {
    "result",
    "operation",
    "artifactSha256",
    "inputDigest",
    "outputDigest",
    "peakResidentBytes",
    "scratchBytes",
    "hostTransfers",
    "deviceResident",
}
NATIVE_RECEIPT_FIELDS = {
    "schema",
    "result",
    "scenario_id",
    "source_revision",
    "backend",
    "backend_id",
    "backend_build",
    "physical_device",
    "manifest_digest",
    "vector_digest",
    "artifact",
    "export",
    "reload",
    "receipt_id",
}
NATIVE_ARTIFACT_FIELDS = {"name", "bytes", "sha256"}
NATIVE_RAW_LIFECYCLE_FIELDS = {
    "result",
    "operation",
    "artifact_sha256",
    "input_digest",
    "output_digest",
    "peak_resident_bytes",
    "scratch_bytes",
    "host_transfers",
    "device_resident",
}
BROWSER_TRACE_FIELDS = {
    "schemaId",
    "schemaVersion",
    "scenarioId",
    "implementation",
    "manifestDigest",
    "vectorDigest",
    "physicalDevice",
    "buildId",
    "adapter",
    "limits",
    "vector",
    "lifecycle",
    "faults",
    "explicitReadbacks",
    "steadyStateReadbacks",
    "wasmDispatches",
    "peakBufferBytes",
    "executionDigest",
}
VECTOR_TRACE_FIELDS = {
    "schemaId",
    "schemaVersion",
    "implementation",
    "manifestDigest",
    "vectorDigest",
    "caseCounts",
    "webgpuCaseTransactions",
    "webgpuDispatches",
    "wasmDispatches",
    "wasmCodecCalls",
    "wasmValidationCalls",
    "explicitReadbacks",
    "peakBufferBytes",
    "executionDigest",
    "cases",
}
LIFECYCLE_TRACE_FIELDS = {
    "prepare",
    "forward",
    "backward",
    "optimizerStep",
    "checkpointResume",
    "exportReload",
    "nativeArtifactParity",
    "completedSteps",
    "checkpointSha256",
    "artifactSha256",
    "nativeArtifactSha256",
    "nativeReferenceDigest",
    "receipts",
}
FAULT_TRACE_FIELDS = {
    "deviceLoss",
    "allocationFailure",
    "malformedCheckpoint",
    "malformedSalt",
    "cancellation",
    "outOfOrder",
}


class BrowserReceiptError(ValueError):
    """Physical-browser evidence is malformed, stale, partial, or synthetic."""


def load_canonical_vector_cases() -> tuple[dict[str, Any], ...]:
    try:
        source = VECTOR_CORPUS_PATH.read_bytes()
        mirrored = VECTOR_CORPUS_MIRROR_PATH.read_bytes()
        metadata_bytes = VECTOR_METADATA_PATH.read_bytes()
        metadata = json.loads(metadata_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"canonical vector metadata cannot be loaded: {error}") from error
    if (
        source != mirrored
        or hashlib.sha256(source).hexdigest() != VECTOR_SOURCE_SHA256
        or hashlib.sha256(metadata_bytes).hexdigest() != VECTOR_METADATA_SHA256
        or not isinstance(metadata, dict)
        or set(metadata) != {
            "schema", "manifestDigest", "vectorDigest", "sourceSha256", "cases"
        }
        or metadata.get("schema") != "tritium.browser-vector-inventory.v1"
        or metadata.get("manifestDigest") != MANIFEST_DIGEST
        or metadata.get("vectorDigest") != VECTOR_DIGEST
        or metadata.get("sourceSha256") != VECTOR_SOURCE_SHA256
        or not isinstance(metadata.get("cases"), list)
        or len(metadata["cases"]) != 117
    ):
        raise RuntimeError("canonical vector source or metadata identity differs")
    result = []
    for index, item in enumerate(metadata["cases"]):
        if (
            not isinstance(item, dict)
            or set(item) != {"caseId", "implementation", "scratchBytesMax"}
            or not isinstance(item.get("caseId"), str)
            or item.get("implementation")
            not in {"webgpu", "wasm-codec", "wasm-validation"}
        ):
            raise RuntimeError(f"canonical vector metadata case {index} is malformed")
        scratch_bytes_max = item["scratchBytesMax"]
        if scratch_bytes_max is not None and (
            type(scratch_bytes_max) is not int or scratch_bytes_max < 0
        ):
            raise RuntimeError(
                f"canonical vector case {item['caseId']} scratch bound is malformed"
            )
        if (item["implementation"] == "wasm-validation") != (
            scratch_bytes_max is None
        ):
            raise RuntimeError(
                f"canonical vector case {item['caseId']} validity metadata differs"
            )
        result.append(
            {
                "caseId": item["caseId"],
                "implementation": item["implementation"],
                "scratchBytesMax": scratch_bytes_max,
            }
        )
    if (
        len({item["caseId"] for item in result}) != 117
        or sum(item["implementation"] == "webgpu" for item in result) != 68
        or sum(item["implementation"] == "wasm-codec" for item in result) != 4
        or sum(item["implementation"] == "wasm-validation" for item in result) != 45
    ):
        raise RuntimeError("canonical vector corpus inventory differs")
    return tuple(result)


CANONICAL_VECTOR_CASES = load_canonical_vector_cases()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def object_(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise BrowserReceiptError(f"{label} fields do not match the frozen schema")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BrowserReceiptError(f"{label} must be a non-empty string")
    return value


def hex_(value: Any, label: str) -> str:
    text = string(value, label)
    if len(text) != 64 or any(character not in HEX for character in text):
        raise BrowserReceiptError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return text


def positive(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise BrowserReceiptError(f"{label} must be a positive integer")
    return value


def contained_file(root: Path, value: Any, label: str, maximum: int) -> Path:
    logical = PurePosixPath(string(value, label))
    if (
        logical.is_absolute()
        or "\\" in str(value)
        or "\0" in str(value)
        or any(part in {"", ".", ".."} for part in logical.parts)
    ):
        raise BrowserReceiptError(f"{label} is unsafe")
    candidate = root.joinpath(*logical.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise BrowserReceiptError(f"{label} must be an ordinary file")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise BrowserReceiptError(f"{label} escapes the evidence directory") from error
    if resolved.stat().st_size <= 0 or resolved.stat().st_size > maximum:
        raise BrowserReceiptError(f"{label} exceeds its byte ceiling")
    return resolved


def validate_native_receipt(
    value: Any,
    summary: dict[str, Any],
    revision: str,
    release: str,
) -> None:
    receipt = object_(value, NATIVE_RECEIPT_FIELDS, "browser native receipt")
    artifact = object_(
        receipt["artifact"], NATIVE_ARTIFACT_FIELDS, "browser native receipt artifact"
    )
    if (
        receipt["schema"] != "tritium.browser-native-reference.v1"
        or receipt["result"] != "pass"
        or receipt["scenario_id"] != "salt-ste-sgd-256-v1"
        or receipt["source_revision"] != revision
        or receipt["backend"] != "cpu"
        or receipt["backend_id"] != "cpu.reference.v1"
        or receipt["backend_build"]
        != f"tritium-train@{release}+source-git:{revision}"
        or receipt["physical_device"] != summary["physicalDevice"]
        or receipt["manifest_digest"] != MANIFEST_DIGEST
        or receipt["vector_digest"] != VECTOR_DIGEST
        or artifact["name"] != summary["artifactName"]
        or artifact["bytes"] != summary["artifactBytes"]
        or artifact["sha256"] != summary["artifactSha256"]
    ):
        raise BrowserReceiptError("browser native raw receipt identity mismatch")
    for name, operation in (("export", "lifecycle.export"), ("reload", "lifecycle.reload")):
        fields = set(NATIVE_RAW_LIFECYCLE_FIELDS)
        if name == "reload":
            fields.add("reloaded_sha256")
        raw = object_(
            receipt[name], fields, f"browser native raw {name} receipt"
        )
        normalized = summary[name]
        if (
            raw["result"] != normalized["result"]
            or raw["operation"] != operation
            or raw["artifact_sha256"] != normalized["artifactSha256"]
            or raw["input_digest"] != normalized["inputDigest"]
            or raw["output_digest"] != normalized["outputDigest"]
            or raw["peak_resident_bytes"] != normalized["peakResidentBytes"]
            or raw["scratch_bytes"] != normalized["scratchBytes"]
            or raw["host_transfers"] != normalized["hostTransfers"]
            or raw["device_resident"] is not normalized["deviceResident"]
            or (
                name == "reload"
                and raw["reloaded_sha256"] != normalized["reloadedSha256"]
            )
        ):
            raise BrowserReceiptError("browser native raw lifecycle differs")
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    expected_id = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if (
        receipt_id != expected_id
        or receipt_id != summary["receiptId"]
        or sha256_bytes(canonical(receipt)) != summary["receiptDigest"]
    ):
        raise BrowserReceiptError("browser native raw receipt identity differs")


def validate_trace_evidence(
    path: Path,
    lane: dict[str, Any],
    revision: str,
    release: str,
    archive: Path,
) -> None:
    try:
        raw = path.read_bytes()
        evidence = object_(json.loads(raw), TRACE_EVIDENCE_FIELDS, "browser trace evidence")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BrowserReceiptError("browser trace must contain UTF-8 JSON") from error
    if raw != canonical(evidence) + b"\n":
        raise BrowserReceiptError("browser trace evidence is not canonical JSON")
    if (
        evidence["schema"] != "tritium.browser-training-lane-evidence.v1"
        or evidence["engine"] != lane["engine"]
    ):
        raise BrowserReceiptError("browser trace evidence identity mismatch")
    string(evidence["run_id"], "browser trace run_id")
    trace_revision = string(evidence["source_revision"], "browser trace source_revision")
    if len(trace_revision) != 40 or any(character not in HEX for character in trace_revision):
        raise BrowserReceiptError("browser trace source revision is invalid")
    if revision is not None and trace_revision != revision:
        raise BrowserReceiptError("browser trace source revision is stale")

    trace_archive = object_(evidence["archive"], TRACE_ARCHIVE_FIELDS, "browser trace archive")
    hex_(trace_archive["sha256"], "browser trace archive.sha256")
    positive(trace_archive["bytes"], "browser trace archive.bytes")
    if Path(string(trace_archive["name"], "browser trace archive.name")).name != trace_archive["name"]:
        raise BrowserReceiptError("browser trace archive name is unsafe")
    if archive is not None and (
        trace_archive["name"] != archive.name
        or trace_archive["bytes"] != archive.stat().st_size
        or trace_archive["sha256"] != sha256(archive)
    ):
        raise BrowserReceiptError("browser trace does not bind the candidate archive")

    try:
        validate_npm_receipt_value(
            evidence["npm_receipt"], archive, trace_revision, release
        )
    except NpmReceiptError as error:
        raise BrowserReceiptError("browser trace npm receipt is invalid") from error

    native = object_(
        evidence["native_reference"], NATIVE_REFERENCE_FIELDS, "browser native reference"
    )
    if (
        native["schema"] != "tritium.browser-native-reference.v1"
        or native["scenarioId"] != "salt-ste-sgd-256-v1"
        or native["backend"] != "cpu"
        or native["backendId"] != "cpu.reference.v1"
        or native["sourceRevision"] != trace_revision
        or native["backendBuild"]
        != f"tritium-train@{release}+source-git:{trace_revision}"
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            string(native["receiptId"], "browser native receiptId"),
        )
        is None
    ):
        raise BrowserReceiptError("browser native reference identity mismatch")
    if not string(
        native["physicalDevice"], "browser native reference physicalDevice"
    ).startswith("cpu:"):
        raise BrowserReceiptError("browser native reference physical device differs")
    string(native["backendBuild"], "browser native reference backendBuild")
    if Path(string(native["artifactName"], "browser native artifactName")).name != native["artifactName"]:
        raise BrowserReceiptError("browser native artifact name is unsafe")
    positive(native["artifactBytes"], "browser native artifactBytes")
    hex_(native["artifactSha256"], "browser native artifactSha256")
    hex_(native["receiptDigest"], "browser native receiptDigest")
    for name, operation in (("export", "lifecycle.export"), ("reload", "lifecycle.reload")):
        expected_fields = set(NATIVE_LIFECYCLE_FIELDS)
        if name == "reload":
            expected_fields.add("reloadedSha256")
        lifecycle = object_(
            native[name], expected_fields, f"browser native {name} receipt"
        )
        if (
            lifecycle["result"] != "pass"
            or lifecycle["operation"] != operation
            or lifecycle["artifactSha256"] != native["artifactSha256"]
            or lifecycle["deviceResident"] is not True
            or lifecycle["hostTransfers"] != 0
            or type(lifecycle["peakResidentBytes"]) is not int
            or lifecycle["peakResidentBytes"] <= 0
            or type(lifecycle["scratchBytes"]) is not int
            or lifecycle["scratchBytes"] < 0
            or re.fullmatch(r"[0-9a-f]{64}", lifecycle["inputDigest"] or "") is None
            or re.fullmatch(r"[0-9a-f]{64}", lifecycle["outputDigest"] or "") is None
            or (
                name == "reload"
                and lifecycle["reloadedSha256"] != native["artifactSha256"]
            )
        ):
            raise BrowserReceiptError(f"browser native {name} receipt differs")
    validate_native_receipt(
        evidence["native_receipt"], native, trace_revision, release
    )

    capabilities = evidence["webdriver_capabilities"]
    if not isinstance(capabilities, dict):
        raise BrowserReceiptError("browser WebDriver capabilities must be an object")
    browser_name = string(capabilities.get("browserName"), "browser WebDriver browserName").lower()
    admitted_names = {
        "chrome": {"chrome", "chromium"},
        "firefox": {"firefox"},
        "safari": {"safari"},
    }
    os_name = lane["os"]["name"].lower()
    platform_name = string(
        capabilities.get("platformName"), "browser WebDriver platformName"
    ).lower()
    platform_matches = (
        (os_name == "linux" and platform_name == "linux")
        or (os_name in {"windows", "win32"} and platform_name == "windows")
        or (
            os_name in {"macos", "darwin"}
            and platform_name in {"mac", "macos", "darwin"}
        )
        or os_name == platform_name
    )
    if (
        browser_name not in admitted_names[lane["engine"]]
        or capabilities.get("browserVersion") != lane["browser_version"]
        or not platform_matches
    ):
        raise BrowserReceiptError("browser WebDriver identity differs from the lane")

    trace = object_(evidence["browser_trace"], BROWSER_TRACE_FIELDS, "browser execution trace")
    unsigned = dict(trace)
    execution_digest = unsigned.pop("executionDigest")
    if execution_digest != sha256_bytes(canonical(unsigned)):
        raise BrowserReceiptError("browser execution trace digest mismatch")
    if (
        trace["schemaId"] != "tritium.physical_browser_training_lane_trace"
        or trace["schemaVersion"] != 1
        or trace["scenarioId"] != "salt-ste-sgd-256-v1"
        or trace["implementation"] != "webgpu"
        or trace["manifestDigest"] != MANIFEST_DIGEST
        or trace["vectorDigest"] != VECTOR_DIGEST
        or re.fullmatch(
            r"wgsl:[0-9a-f]{64}:browser-qualification:salt-ste-sgd-256-v1",
            string(trace["buildId"], "browser execution buildId"),
        )
        is None
        or trace["steadyStateReadbacks"] != 0
        or trace["wasmDispatches"] != 0
        or trace["explicitReadbacks"] != lane["trace"]["explicit_readbacks"]
        or trace["peakBufferBytes"] != lane["trace"]["peak_buffer_bytes"]
    ):
        raise BrowserReceiptError("browser execution trace claims differ from the lane")
    string(trace["physicalDevice"], "browser execution physicalDevice")
    if trace["adapter"] != lane["adapter"]:
        raise BrowserReceiptError("browser execution adapter differs from the lane")
    expected_limits = {
        "maxBufferSize": lane["limits"]["max_buffer_size"],
        "maxStorageBufferBindingSize": lane["limits"]["max_storage_buffer_binding_size"],
        "maxComputeWorkgroupsPerDimension": lane["limits"][
            "max_compute_workgroups_per_dimension"
        ],
        "maxStorageBuffersPerShaderStage": lane["limits"][
            "max_storage_buffers_per_shader_stage"
        ],
    }
    if trace["limits"] != expected_limits:
        raise BrowserReceiptError("browser execution limits differ from the lane")

    vector = object_(trace["vector"], VECTOR_TRACE_FIELDS, "browser vector trace")
    cases = vector["cases"]
    if not isinstance(cases, list) or len(cases) != 117:
        raise BrowserReceiptError("browser vector trace must retain all 117 cases")
    for index, (case, expected) in enumerate(zip(cases, CANONICAL_VECTOR_CASES)):
        if (
            not isinstance(case, dict)
            or set(case)
            != {
                "caseId",
                "implementation",
                "outputDigest",
                "scratchBytes",
                "scratchBytesMax",
            }
            or case["caseId"] != expected["caseId"]
            or case["implementation"] != expected["implementation"]
            or re.fullmatch(r"[0-9a-f]{64}", case["outputDigest"] or "") is None
            or case["scratchBytesMax"] != expected["scratchBytesMax"]
        ):
            raise BrowserReceiptError(
                f"browser vector case {index} differs from canonical metadata"
            )
        if expected["implementation"] == "wasm-validation":
            if case["scratchBytes"] is not None:
                raise BrowserReceiptError(
                    f"browser vector case {expected['caseId']} has invalid scratch evidence"
                )
        elif (
            type(case["scratchBytes"]) is not int
            or case["scratchBytes"] < 0
            or case["scratchBytes"] > expected["scratchBytesMax"]
        ):
            raise BrowserReceiptError(
                f"browser vector case {expected['caseId']} exceeds its scratch bound"
            )
    case_ids = [case.get("caseId") if isinstance(case, dict) else None for case in cases]
    implementations = [
        case.get("implementation") if isinstance(case, dict) else None for case in cases
    ]
    if (
        len(set(case_ids)) != 117
        or implementations.count("webgpu") != 68
        or implementations.count("wasm-codec") != 4
        or implementations.count("wasm-validation") != 45
        or vector["schemaId"] != "tritium.webgpu_vector_conformance_trace"
        or vector["schemaVersion"] != 1
        or vector["implementation"] != "webgpu"
        or vector["manifestDigest"] != MANIFEST_DIGEST
        or vector["vectorDigest"] != VECTOR_DIGEST
        or vector["caseCounts"] != {"valid": 72, "invalid": 45, "skipped": 0}
        or vector["webgpuCaseTransactions"] != 68
        or type(vector["webgpuDispatches"]) is not int
        or vector["webgpuDispatches"] <= 0
        or vector["wasmDispatches"] != 0
        or vector["wasmCodecCalls"] != 4
        or vector["wasmValidationCalls"] != 45
        or vector["executionDigest"] != sha256_bytes(canonical(cases))
    ):
        raise BrowserReceiptError("browser vector execution trace is incomplete")

    lifecycle = object_(
        trace["lifecycle"], LIFECYCLE_TRACE_FIELDS, "browser lifecycle trace"
    )
    for field in (
        "prepare",
        "forward",
        "backward",
        "optimizerStep",
        "checkpointResume",
        "exportReload",
        "nativeArtifactParity",
    ):
        if lifecycle[field] is not True:
            raise BrowserReceiptError("browser lifecycle trace is incomplete")
    if (
        lifecycle["completedSteps"] != 1
        or lifecycle["artifactSha256"] != native["artifactSha256"]
        or lifecycle["nativeArtifactSha256"] != native["artifactSha256"]
        or lifecycle["nativeReferenceDigest"] != native["receiptDigest"]
    ):
        raise BrowserReceiptError("browser native artifact parity is unbound")
    hex_(lifecycle["checkpointSha256"], "browser checkpointSha256")
    receipts = lifecycle["receipts"]
    if not isinstance(receipts, list) or {
        receipt.get("operation") for receipt in receipts if isinstance(receipt, dict)
    } != {
        "session.forward",
        "session.backward",
        "session.step",
        "session.checkpoint",
        "session.resume",
        "session.export",
    }:
        raise BrowserReceiptError("browser lifecycle receipts are incomplete")
    if any(
        not isinstance(receipt, dict)
        or receipt.get("physicalDevice") != trace["physicalDevice"]
        or receipt.get("buildId") != trace["buildId"]
        or type(receipt.get("peakResidentBytes")) is not int
        or receipt["peakResidentBytes"] <= 0
        for receipt in receipts
    ):
        raise BrowserReceiptError("browser lifecycle receipt identity differs")

    faults = object_(trace["faults"], FAULT_TRACE_FIELDS, "browser fault trace")
    if any(
        not isinstance(faults[field], dict)
        or faults[field].get("passed") is not True
        or not isinstance(faults[field].get("errorCode"), str)
        or not faults[field]["errorCode"]
        for field in FAULT_TRACE_FIELDS
    ):
        raise BrowserReceiptError("browser fault trace is incomplete")
    if (
        faults["cancellation"].get("errorCode") != "cancelled"
        or type(faults["cancellation"].get("observedEvents")) is not int
        or faults["cancellation"]["observedEvents"] < 1
        or faults["allocationFailure"].get("errorCode")
        != "injected_allocation_failure"
        or faults["allocationFailure"].get("observedEvents") != 1
    ):
        raise BrowserReceiptError("browser physical fault trace is incomplete")


def validate_lane(
    value: Any,
    ordinal: int,
    root: Path,
    revision: str,
    release: str,
    archive: Path,
) -> dict[str, Any]:
    label = f"receipt.lanes[{ordinal}]"
    lane = object_(value, LANE_FIELDS, label)
    if lane["engine"] != ENGINES[ordinal]:
        raise BrowserReceiptError(
            "browser lanes must be ordered Chrome, Firefox, Safari"
        )
    if (
        re.fullmatch(
            r"[0-9]+(?:\.[0-9]+){1,3}",
            string(lane["browser_version"], f"{label}.browser_version"),
        )
        is None
    ):
        raise BrowserReceiptError(f"{label}.browser_version is invalid")
    os_value = object_(lane["os"], OS_FIELDS, f"{label}.os")
    for field in OS_FIELDS:
        string(os_value[field], f"{label}.os.{field}")
    if lane["engine"] == "safari" and os_value["name"].lower() not in {
        "macos",
        "darwin",
    }:
        raise BrowserReceiptError("Safari evidence must run on physical macOS")
    adapter = object_(lane["adapter"], ADAPTER_FIELDS, f"{label}.adapter")
    for field in ADAPTER_FIELDS - {"software"}:
        string(adapter[field], f"{label}.adapter.{field}")
    description = " ".join(
        str(adapter[field]) for field in ADAPTER_FIELDS - {"software"}
    )
    if adapter["software"] is not False or any(
        marker in description.lower()
        for marker in (
            "swiftshader", "llvmpipe", "software", "emulator", "lavapipe", "warp"
        )
    ):
        raise BrowserReceiptError(f"{label} does not identify a physical adapter")
    limits = object_(lane["limits"], LIMIT_FIELDS, f"{label}.limits")
    for field in LIMIT_FIELDS:
        positive(limits[field], f"{label}.limits.{field}")
    cases = object_(lane["case_counts"], CASE_FIELDS, f"{label}.case_counts")
    if cases != {"valid": 72, "invalid": 45, "skipped": 0}:
        raise BrowserReceiptError(f"{label} must execute all 117 canonical vectors")
    lifecycle = object_(lane["lifecycle"], LIFECYCLE_FIELDS, f"{label}.lifecycle")
    if any(lifecycle[field] is not True for field in LIFECYCLE_FIELDS):
        raise BrowserReceiptError(f"{label} lifecycle is incomplete")
    faults = object_(lane["faults"], FAULT_FIELDS, f"{label}.faults")
    if any(faults[field] is not True for field in FAULT_FIELDS):
        raise BrowserReceiptError(f"{label} fault injection is incomplete")
    trace = object_(lane["trace"], TRACE_FIELDS, f"{label}.trace")
    path = contained_file(root, trace["file"], f"{label}.trace.file", MAX_TRACE_BYTES)
    if trace["bytes"] != path.stat().st_size or hex_(
        trace["sha256"], f"{label}.trace.sha256"
    ) != sha256(path):
        raise BrowserReceiptError(f"{label} trace bytes differ")
    if trace["steady_state_readbacks"] != 0 or trace["wasm_dispatches"] != 0:
        raise BrowserReceiptError(
            f"{label} trace contains forbidden fallback or readback"
        )
    positive(trace["explicit_readbacks"], f"{label}.trace.explicit_readbacks")
    positive(trace["peak_buffer_bytes"], f"{label}.trace.peak_buffer_bytes")
    validate_trace_evidence(path, lane, revision, release, archive)
    return lane


def validate(
    receipt_path: Path, revision: str, release: str, archive: Path
) -> dict[str, Any]:
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.stat().st_size > MAX_RECEIPT_BYTES
    ):
        raise BrowserReceiptError("browser receipt must be a bounded ordinary file")
    if (
        archive.is_symlink()
        or not archive.is_file()
        or archive.stat().st_size <= 0
        or archive.stat().st_size > MAX_ARCHIVE_BYTES
    ):
        raise BrowserReceiptError("browser archive must be an ordinary file")
    try:
        receipt = object_(json.loads(receipt_path.read_bytes()), TOP_FIELDS, "receipt")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BrowserReceiptError("browser receipt must contain UTF-8 JSON") from error
    if receipt["schema"] != SCHEMA or receipt["result"] != "pass":
        raise BrowserReceiptError("browser receipt schema or result mismatch")
    if (
        len(revision) != 40
        or any(character not in HEX for character in revision)
        or not release
    ):
        raise BrowserReceiptError("expected browser source or release is invalid")
    if receipt["source_revision"] != revision or receipt["release"] != release:
        raise BrowserReceiptError("browser receipt source or release is stale")
    string(receipt["run_id"], "receipt.run_id")
    artifact = object_(receipt["artifact"], ARTIFACT_FIELDS, "receipt.artifact")
    if (
        artifact["kind"] != "npm-archive"
        or artifact["name"] != archive.name
        or Path(str(artifact["name"])).name != artifact["name"]
        or not str(artifact["name"]).endswith(".tgz")
        or artifact["bytes"] != archive.stat().st_size
        or hex_(artifact["sha256"], "receipt.artifact.sha256") != sha256(archive)
    ):
        raise BrowserReceiptError("browser receipt does not bind the npm archive")
    if receipt["manifest_digest"] != MANIFEST_DIGEST:
        raise BrowserReceiptError("browser receipt manifest digest mismatch")
    if receipt["vector_digest"] != VECTOR_DIGEST:
        raise BrowserReceiptError("browser receipt vector digest mismatch")
    lanes = receipt["lanes"]
    if not isinstance(lanes, list) or len(lanes) != len(ENGINES):
        raise BrowserReceiptError("browser receipt must contain exactly three lanes")
    root = receipt_path.parent.resolve(strict=True)
    for ordinal, lane in enumerate(lanes):
        validate_lane(lane, ordinal, root, revision, release, archive)
    trace_files = [lane["trace"]["file"] for lane in lanes]
    if len(set(trace_files)) != len(trace_files):
        raise BrowserReceiptError("browser lanes must retain distinct trace files")
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    expected = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if receipt_id != expected:
        raise BrowserReceiptError("browser receipt identity mismatch")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate(
        args.receipt.absolute(),
        args.source_revision,
        args.release,
        args.artifact.absolute(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
