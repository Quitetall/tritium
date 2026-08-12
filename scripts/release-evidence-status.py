#!/usr/bin/env python3
"""Strict ADR 0033 release-evidence registry and local-RC gate report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import runpy
from typing import Any


CUDA_RECEIPT = runpy.run_path(Path(__file__).with_name("verify-cuda-training-receipt.py"))
validate_cuda_receipt = CUDA_RECEIPT["validate"]
CudaReceiptError = CUDA_RECEIPT["ReceiptError"]
WHEEL_RECEIPT = runpy.run_path(Path(__file__).with_name("wheel-functional-smoke.py"))
validate_wheel_receipt = WHEEL_RECEIPT["validate_receipt"]
WheelReceiptError = WHEEL_RECEIPT["SmokeError"]
MATRIX_RECEIPT = runpy.run_path(Path(__file__).with_name("aggregate-wheel-smoke.py"))
validate_matrix_receipt = MATRIX_RECEIPT["validate_receipt"]
MatrixReceiptError = MATRIX_RECEIPT["AggregateError"]
CRATE_RECEIPT = runpy.run_path(Path(__file__).with_name("qualify-crate-archives.py"))
validate_crate_receipt = CRATE_RECEIPT["validate_receipt"]
CrateReceiptError = CRATE_RECEIPT["ArchiveError"]
NPM_RECEIPT = runpy.run_path(Path(__file__).with_name("verify-npm-archive-receipt.py"))
validate_npm_receipt = NPM_RECEIPT["validate_receipt"]
NpmReceiptError = NPM_RECEIPT["NpmReceiptError"]
OCI_RUNTIME = runpy.run_path(Path(__file__).with_name("qualify-oci-runtime.py"))
load_oci_runtime_receipt = OCI_RUNTIME["load_receipt"]
OciRuntimeError = OCI_RUNTIME["QualificationError"]
OCI_SECURITY = runpy.run_path(Path(__file__).with_name("qualify-oci-security.py"))
load_oci_security_receipt = OCI_SECURITY["load_receipt"]
OciSecurityError = OCI_SECURITY["SecurityScanError"]
KUBERNETES_DEPLOYMENT = runpy.run_path(
    Path(__file__).with_name("qualify-kubernetes-deployment.py")
)
load_deployment_receipt = KUBERNETES_DEPLOYMENT["load_receipt"]
DeploymentError = KUBERNETES_DEPLOYMENT["DeploymentError"]
TUTORIAL_RECEIPT = runpy.run_path(
    Path(__file__).resolve().parent.parent
    / "crates/tritium-py/python/tritium/torch/tutorial_receipt.py"
)
validate_tutorial_receipt = TUTORIAL_RECEIPT["validate_receipt"]
validate_hf_lifecycle_receipt = TUTORIAL_RECEIPT["validate_hf_receipt"]
validate_hf_export_receipt = TUTORIAL_RECEIPT["validate_export_receipt"]
OBSERVABILITY_RECEIPT = runpy.run_path(
    Path(__file__).resolve().parent.parent
    / "crates/tritium-py/python/tritium/torch/observability_receipt.py"
)
validate_observability_receipt = OBSERVABILITY_RECEIPT["validate_receipt"]
DISTRIBUTED_RECEIPT = runpy.run_path(
    Path(__file__).with_name("verify-hf-distributed-receipt.py")
)
validate_distributed_receipt = DISTRIBUTED_RECEIPT["validate"]
DistributedReceiptError = DISTRIBUTED_RECEIPT["ReceiptError"]
BROWSER_RECEIPT = runpy.run_path(
    Path(__file__).with_name("verify-browser-training-receipt.py")
)
validate_browser_receipt = BROWSER_RECEIPT["validate"]
BrowserReceiptError = BROWSER_RECEIPT["BrowserReceiptError"]
REPRODUCTION_RECEIPT = runpy.run_path(
    Path(__file__).with_name("verify-release-reproduction.py")
)
validate_second_machine = REPRODUCTION_RECEIPT["validate_second_machine"]
validate_independent_review = REPRODUCTION_RECEIPT["validate_independent_review"]
ReproductionError = REPRODUCTION_RECEIPT["ReproductionError"]
ZOO_COMMUNITY_RECEIPT = runpy.run_path(
    Path(__file__).with_name("verify-zoo-community-receipt.py")
)
validate_model_zoo = ZOO_COMMUNITY_RECEIPT["validate_zoo"]
validate_generated_claims = ZOO_COMMUNITY_RECEIPT["validate_claims"]
validate_governance_docs = ZOO_COMMUNITY_RECEIPT["validate_governance"]
ZooCommunityError = ZOO_COMMUNITY_RECEIPT["ZooCommunityError"]
FLAGSHIP_RECEIPT = runpy.run_path(
    Path(__file__).with_name("verify-flagship-receipt.py")
)
validate_flagship_conversion = FLAGSHIP_RECEIPT["validate"]
FlagshipReceiptError = FLAGSHIP_RECEIPT["FlagshipReceiptError"]
FLAGSHIP_QUALITY_RECEIPT = runpy.run_path(
    Path(__file__).with_name("verify-flagship-quality-receipt.py")
)
validate_flagship_quality = FLAGSHIP_QUALITY_RECEIPT["validate_quality"]
validate_flagship_tasks = FLAGSHIP_QUALITY_RECEIPT["validate_tasks"]
FlagshipQualityError = FLAGSHIP_QUALITY_RECEIPT["FlagshipQualityError"]
FLAGSHIP_RUNTIME_RECEIPT = runpy.run_path(
    Path(__file__).with_name("verify-flagship-runtime-receipt.py")
)
validate_flagship_runtime = FLAGSHIP_RUNTIME_RECEIPT["validate_runtime"]
validate_flagship_physical = FLAGSHIP_RUNTIME_RECEIPT["validate_physical"]
FlagshipRuntimeError = FLAGSHIP_RUNTIME_RECEIPT["FlagshipRuntimeError"]
STAGE7_RECEIPT = runpy.run_path(
    Path(__file__).with_name("verify-stage7-qualification-receipt.py")
)
validate_stage7_freeze = STAGE7_RECEIPT["validate"]
ONNX_RECEIPT = runpy.run_path(
    Path(__file__).with_name("verify-onnx-inference-receipt.py")
)
validate_onnx_inference = ONNX_RECEIPT["validate"]
OnnxReceiptError = ONNX_RECEIPT["OnnxReceiptError"]
TRAINING_BACKEND_RECEIPT = runpy.run_path(
    Path(__file__).with_name("verify-training-backend-receipt.py")
)
validate_training_backends = TRAINING_BACKEND_RECEIPT["validate"]
TrainingBackendReceiptError = TRAINING_BACKEND_RECEIPT["TrainingBackendReceiptError"]
TRAINING_PERFORMANCE_RECEIPT = runpy.run_path(
    Path(__file__).with_name("verify-training-performance-receipt.py")
)
validate_training_performance = TRAINING_PERFORMANCE_RECEIPT["validate"]
TrainingPerformanceError = TRAINING_PERFORMANCE_RECEIPT["TrainingPerformanceError"]
TORCH_DISPATCH_RECEIPT = runpy.run_path(
    Path(__file__).with_name("verify-torch-dispatch-overhead-receipt.py")
)
validate_torch_dispatch_overhead = TORCH_DISPATCH_RECEIPT["validate"]
DispatchOverheadError = TORCH_DISPATCH_RECEIPT["DispatchOverheadError"]
TORCH_DISPATCH_CUDA_RECEIPT = runpy.run_path(
    Path(__file__).with_name("verify-torch-dispatch-cuda-receipt.py")
)
validate_torch_dispatch_cuda = TORCH_DISPATCH_CUDA_RECEIPT["validate"]
torch_dispatch_git_blob_at = TORCH_DISPATCH_CUDA_RECEIPT["git_blob_at"]
DispatchCudaReceiptError = TORCH_DISPATCH_CUDA_RECEIPT["ReceiptError"]
ESTIMATOR_REFINEMENT_RECEIPT = runpy.run_path(
    Path(__file__).with_name("verify-estimator-refinement-receipt.py")
)
validate_estimators = ESTIMATOR_REFINEMENT_RECEIPT["validate_estimators"]
validate_refinement = ESTIMATOR_REFINEMENT_RECEIPT["validate_refinement"]
validate_baseline_ablation = ESTIMATOR_REFINEMENT_RECEIPT["validate_ablation"]
EstimatorRefinementError = ESTIMATOR_REFINEMENT_RECEIPT["EstimatorRefinementError"]

SCHEMA = "tritium.release-evidence-registry.v1"
REPORT_SCHEMA = "tritium.release-gate-report.v1"
TOP_FIELDS = {
    "schema", "release", "source_revision", "candidate_manifest_sha256", "receipts"
}
RECEIPT_FIELDS = {"id", "kind", "path", "sha256", "artifact_id", "parents"}
KNOWN_KINDS = frozenset(
    {
        "cuda-training", "clean-install", "compatibility-matrix",
        "crate-archive", "npm-archive", "oci-runtime-cpu", "oci-runtime-cuda",
        "oci-security-cpu", "oci-security-cuda",
        "serving-deployment-cpu", "serving-deployment-cuda",
        "installed-qat-tutorial",
        "frontend-lifecycle",
        "distributed-training",
        "export-reload",
        "observability",
        "browser-conformance",
        "second-machine",
        "independent-review",
        "model-zoo",
        "generated-claims",
        "governance-docs",
        "conversion-refinement",
        "quality",
        "task-retention",
        "runtime",
        "physical-bytes",
        "stage7-recipe-freeze",
        "onnx-inference",
        "backend-manifest",
        "performance",
        "torch-dispatch-overhead",
        "torch-dispatch-cuda",
        "estimator-validation",
        "refinement",
        "baseline-ablation",
    }
)
HEX = frozenset("0123456789abcdef")
MAX_RECEIPT_BYTES = 32 * 1024 * 1024

# This policy is code, not registry input: a partial or adversarial registry cannot
# remove release gates. New receipt schemas become useful only after a validator lands.
GATES = (
    (
        "flagship-qwen",
        ("conversion-refinement", "quality", "task-retention", "runtime", "physical-bytes"),
    ),
    ("stage7-freeze", ("stage7-recipe-freeze",)),
    (
        "pytorch-hf",
        (
            "installed-qat-tutorial", "frontend-lifecycle",
            "distributed-training", "export-reload",
            "observability", "torch-dispatch-overhead", "torch-dispatch-cuda",
        ),
    ),
    ("native-backends", ("backend-manifest", "cuda-training", "performance")),
    ("estimators-refinement", ("estimator-validation", "refinement", "baseline-ablation")),
    ("browser", ("browser-conformance",)),
    ("onnx", ("onnx-inference",)),
    (
        "packages",
        ("clean-install", "compatibility-matrix", "crate-archive", "npm-archive"),
    ),
    (
        "serving",
        ("oci-runtime-cpu", "oci-runtime-cuda", "oci-security-cpu",
         "oci-security-cuda", "serving-deployment-cpu",
         "serving-deployment-cuda"),
    ),
    ("zoo-community", ("model-zoo", "generated-claims", "governance-docs")),
    ("reproduction-signoff", ("second-machine", "independent-review")),
)


class EvidenceError(ValueError):
    """Registry evidence is malformed, stale, duplicated, or unvalidated."""


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise EvidenceError(f"{label} fields do not match the frozen schema")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label} must be a non-empty string")
    return value


def _hex(value: Any, length: int, label: str) -> str:
    text = _string(value, label)
    if len(text) != length or any(character not in HEX for character in text):
        raise EvidenceError(f"{label} must be {length} lowercase hexadecimal characters")
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _contained_file(root: Path, value: Any, label: str) -> Path:
    logical_text = _string(value, label)
    logical = PurePosixPath(logical_text)
    if logical.is_absolute() or ".." in logical.parts or "\\" in logical_text:
        raise EvidenceError(f"{label} must be a contained POSIX path")
    cursor = root
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise EvidenceError(f"{label} must not traverse a symlink")
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise EvidenceError(f"{label} is not contained below the registry") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise EvidenceError(f"{label} must name an ordinary file")
    return resolved


def _gate_row(
    gate_id: str, required: tuple[str, ...], evidence: dict[str, str]
) -> dict[str, Any]:
    satisfied = sorted(kind for kind in required if evidence.get(kind) == "empirical")
    structural = sorted(kind for kind in required if evidence.get(kind) == "structural")
    missing = sorted(kind for kind in required if kind not in evidence)
    if missing:
        status = "MISSING"
    elif structural:
        status = "STRUCTURAL_ONLY"
    else:
        status = "PASS"
    return {
        "id": gate_id,
        "status": status,
        "required_kinds": list(required),
        "satisfied_kinds": satisfied,
        "structural_kinds": structural,
        "missing_kinds": missing,
    }


def _check_ancestry(entries: dict[str, dict[str, Any]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(receipt_id: str) -> None:
        if receipt_id in visiting:
            raise EvidenceError("receipt ancestry contains a cycle")
        if receipt_id in visited:
            return
        visiting.add(receipt_id)
        for parent in entries[receipt_id]["parents"]:
            if parent not in entries:
                raise EvidenceError(f"receipt {receipt_id!r} has unknown parent {parent!r}")
            visit(parent)
        visiting.remove(receipt_id)
        visited.add(receipt_id)

    for receipt_id in entries:
        visit(receipt_id)


def review_scope_sha256(document: dict[str, Any]) -> str:
    scope = {
        **{field: document[field] for field in TOP_FIELDS - {"receipts"}},
        "receipts": [
            entry for entry in document["receipts"]
            if entry.get("kind") != "independent-review"
        ],
    }
    return hashlib.sha256(
        json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def evaluate(
    registry: Path, candidate: Path, candidate_document: dict[str, Any],
    digest_tool: str = "tritium",
) -> dict[str, Any]:
    if registry.is_symlink() or not registry.is_file():
        raise EvidenceError("registry must be an ordinary file")
    if registry.stat().st_size > MAX_RECEIPT_BYTES:
        raise EvidenceError("registry exceeds the metadata size limit")
    try:
        document = _object(json.loads(registry.read_bytes()), TOP_FIELDS, "registry")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError("registry must contain UTF-8 JSON") from error
    if document["schema"] != SCHEMA:
        raise EvidenceError(f"registry.schema must equal {SCHEMA!r}")
    release = _string(document["release"], "registry.release")
    revision = _hex(document["source_revision"], 40, "registry.source_revision")
    if release != candidate_document.get("release") or revision != candidate_document.get("source_revision"):
        raise EvidenceError("registry release identity does not match candidate")
    expected_candidate = _hex(
        document["candidate_manifest_sha256"], 64, "registry.candidate_manifest_sha256"
    )
    if expected_candidate != _sha256(candidate):
        raise EvidenceError("registry does not bind the exact candidate manifest")

    raw_receipts = document["receipts"]
    if not isinstance(raw_receipts, list):
        raise EvidenceError("registry.receipts must be an array")
    root = registry.parent.resolve(strict=True)
    entries: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    portable_paths: set[str] = set()
    run_ids: set[str] = set()
    artifacts = {
        artifact.get("id"): artifact
        for artifact in candidate_document.get("artifacts", [])
        if isinstance(artifact, dict)
    }
    evidence: dict[str, str] = {}
    validated_receipts: dict[str, dict[str, Any]] = {}
    kinds: dict[str, str] = {}
    artifact_ids: dict[str, str] = {}
    for ordinal, raw in enumerate(raw_receipts):
        label = f"registry.receipts[{ordinal}]"
        entry = _object(raw, RECEIPT_FIELDS, label)
        receipt_id = _string(entry["id"], f"{label}.id")
        kind = _string(entry["kind"], f"{label}.kind")
        if kind not in KNOWN_KINDS:
            raise EvidenceError(f"{label}.kind has no release validator")
        if receipt_id in entries:
            raise EvidenceError(f"duplicate receipt id {receipt_id!r}")
        logical_path = _string(entry["path"], f"{label}.path")
        portable_path = logical_path.casefold()
        if logical_path in paths or portable_path in portable_paths:
            raise EvidenceError(f"duplicate receipt path {logical_path!r}")
        parents = entry["parents"]
        if not isinstance(parents, list) or len(set(parents)) != len(parents) or any(
            not isinstance(parent, str) or not parent for parent in parents
        ):
            raise EvidenceError(f"{label}.parents must be a unique string array")
        receipt_path = _contained_file(root, logical_path, f"{label}.path")
        if receipt_path.stat().st_size > MAX_RECEIPT_BYTES:
            raise EvidenceError(f"{label}.path exceeds the metadata size limit")
        if _sha256(receipt_path) != _hex(entry["sha256"], 64, f"{label}.sha256"):
            raise EvidenceError(f"{label}.sha256 does not match receipt bytes")
        artifact_id = _string(entry["artifact_id"], f"{label}.artifact_id")
        artifact = artifacts.get(artifact_id)
        if artifact is None:
            raise EvidenceError(f"{label}.artifact_id is absent from candidate")
        expected_artifact_kind = (
            "rust-crate" if kind == "crate-archive"
            else "npm-archive" if kind in {"npm-archive", "browser-conformance"}
            else "oci-image" if (
                kind.startswith("oci-") or kind.startswith("serving-deployment-")
            )
            else "model-bundle" if kind in {
                "conversion-refinement", "quality", "task-retention", "runtime",
                "physical-bytes", "stage7-recipe-freeze",
            }
            else "onnx-bundle" if kind == "onnx-inference"
            else "training-receipt-bundle" if kind in {"backend-manifest", "performance"}
            else "model-bundle" if kind in {"refinement", "baseline-ablation"}
            else "python-wheel"
        )
        if artifact.get("kind") != expected_artifact_kind:
            raise EvidenceError(
                f"{kind} evidence must bind candidate {expected_artifact_kind}"
            )
        artifact_path = candidate.parent / _string(artifact.get("path"), "candidate artifact path")
        try:
            if kind == "cuda-training":
                receipt = validate_cuda_receipt(
                    receipt_path, revision, release, artifact_path
                )
            elif kind == "clean-install":
                receipt = validate_wheel_receipt(
                    receipt_path, revision, release, artifact_path
                )
            elif kind == "compatibility-matrix":
                receipt = validate_matrix_receipt(receipt_path, revision, release)
            elif kind == "crate-archive":
                receipt = validate_crate_receipt(
                    receipt_path, candidate.parent,
                    Path(__file__).resolve().parent.parent / "Cargo.lock",
                    revision, release,
                )
            elif kind == "npm-archive":
                receipt = validate_npm_receipt(
                    receipt_path, artifact_path, revision, release
                )
            elif kind == "installed-qat-tutorial":
                receipt = validate_tutorial_receipt(
                    receipt_path,
                    expected_device="cpu",
                    expected_wheel=artifact_path,
                    expected_source_revision=revision,
                    expected_release=release,
                )
            elif kind == "frontend-lifecycle":
                receipt = validate_hf_lifecycle_receipt(
                    receipt_path,
                    expected_wheel=artifact_path,
                    expected_source_revision=revision,
                    expected_release=release,
                )
            elif kind == "distributed-training":
                receipt = validate_distributed_receipt(
                    receipt_path, revision, release, artifact_path
                )
            elif kind == "export-reload":
                receipt = validate_hf_export_receipt(
                    receipt_path,
                    expected_wheel=artifact_path,
                    expected_source_revision=revision,
                    expected_release=release,
                )
            elif kind == "observability":
                receipt = validate_observability_receipt(
                    receipt_path,
                    expected_wheel=artifact_path,
                    expected_source_revision=revision,
                    expected_release=release,
                )
            elif kind == "browser-conformance":
                receipt = validate_browser_receipt(
                    receipt_path, revision, release, artifact_path
                )
            elif kind == "second-machine":
                receipt = validate_second_machine(
                    receipt_path, revision, release, candidate, artifact_path
                )
            elif kind == "independent-review":
                receipt = validate_independent_review(
                    receipt_path, revision, release, candidate, artifact_path
                )
            elif kind == "model-zoo":
                receipt = validate_model_zoo(
                    receipt_path, revision, release, candidate, artifact_path
                )
            elif kind == "generated-claims":
                receipt = validate_generated_claims(
                    receipt_path, revision, release, candidate, artifact_path,
                    Path(__file__).resolve().parent.parent,
                )
            elif kind == "governance-docs":
                receipt = validate_governance_docs(
                    receipt_path, revision, release, candidate, artifact_path,
                    Path(__file__).resolve().parent.parent,
                )
            elif kind == "conversion-refinement":
                receipt = validate_flagship_conversion(
                    receipt_path, revision, release, candidate
                )
            elif kind == "quality":
                receipt = validate_flagship_quality(
                    receipt_path, revision, release, candidate
                )
            elif kind == "task-retention":
                receipt = validate_flagship_tasks(
                    receipt_path, revision, release, candidate
                )
            elif kind == "runtime":
                receipt = validate_flagship_runtime(
                    receipt_path, revision, release, candidate
                )
            elif kind == "physical-bytes":
                receipt = validate_flagship_physical(
                    receipt_path, revision, release, candidate
                )
            elif kind == "stage7-recipe-freeze":
                receipt = validate_stage7_freeze(
                    receipt_path, revision, release, candidate
                )
            elif kind == "onnx-inference":
                receipt = validate_onnx_inference(
                    receipt_path, revision, release, candidate
                )
            elif kind == "backend-manifest":
                receipt = validate_training_backends(
                    receipt_path, revision, release, candidate,
                    Path(__file__).resolve().parent.parent,
                )
            elif kind == "performance":
                receipt = validate_training_performance(
                    receipt_path, revision, release, candidate
                )
            elif kind == "torch-dispatch-overhead":
                receipt = validate_torch_dispatch_overhead(
                    receipt_path,
                    expected_revision=revision,
                    expected_release=release,
                    expected_wheel=artifact_path,
                )
            elif kind == "torch-dispatch-cuda":
                source_blob = torch_dispatch_git_blob_at(
                    Path(__file__).resolve().parent.parent, revision
                )
                receipt = validate_torch_dispatch_cuda(
                    receipt_path, revision, release, artifact_path, source_blob
                )
            elif kind == "estimator-validation":
                receipt = validate_estimators(
                    receipt_path, revision, release, candidate
                )
            elif kind == "refinement":
                receipt = validate_refinement(
                    receipt_path, revision, release, candidate
                )
            elif kind == "baseline-ablation":
                receipt = validate_baseline_ablation(
                    receipt_path, revision, release, candidate
                )
            elif kind in {"oci-runtime-cpu", "oci-runtime-cuda"}:
                receipt = load_oci_runtime_receipt(
                    receipt_path, revision=revision, release=release,
                    artifact_path=artifact_path,
                )
                if receipt["flavor"] != kind.removeprefix("oci-runtime-"):
                    raise OciRuntimeError("runtime receipt flavor differs from evidence kind")
            elif kind in {"oci-security-cpu", "oci-security-cuda"}:
                receipt = load_oci_security_receipt(
                    receipt_path, revision=revision, release=release,
                    artifact_path=artifact_path,
                )
                if receipt["flavor"] != kind.removeprefix("oci-security-"):
                    raise OciSecurityError("security receipt flavor differs from evidence kind")
            elif kind in {"serving-deployment-cpu", "serving-deployment-cuda"}:
                try:
                    raw_deployment = json.loads(receipt_path.read_bytes())
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise DeploymentError(
                        "deployment receipt must contain UTF-8 JSON"
                    ) from error
                if not isinstance(raw_deployment, dict):
                    raise DeploymentError("deployment receipt must be an object")
                chart_record = raw_deployment.get("chart_artifact")
                chart_id = chart_record.get("artifact_id") if isinstance(
                    chart_record, dict
                ) else None
                chart_artifact = artifacts.get(chart_id)
                if not isinstance(chart_artifact, dict) or chart_artifact.get(
                    "kind"
                ) != "helm-chart":
                    raise DeploymentError("deployment chart is absent from candidate")
                chart_path = _contained_file(
                    candidate.parent, chart_artifact.get("path"),
                    "candidate Helm chart path",
                )
                support_paths: dict[str, Path] = {}
                for field in ("bundle_manifest_artifact", "build_receipt_artifact"):
                    record = raw_deployment.get(field)
                    if not isinstance(record, dict) or not isinstance(
                        record.get("name"), str
                    ):
                        raise DeploymentError(f"deployment {field} is malformed")
                    support = _contained_file(root, record["name"], f"deployment {field}")
                    if (support.stat().st_size, _sha256(support)) != (
                        record.get("bytes"), record.get("sha256")
                    ):
                        raise DeploymentError(f"deployment {field} bytes differ")
                    support_paths[field] = support
                receipt = load_deployment_receipt(
                    receipt_path, chart_path=chart_path, image_path=artifact_path,
                    manifest_path=support_paths["bundle_manifest_artifact"],
                    build_receipt=support_paths["build_receipt_artifact"],
                    package_candidate=candidate, digest_tool=digest_tool,
                    revision=revision, release=release,
                )
                if receipt["flavor"] != kind.removeprefix("serving-deployment-"):
                    raise DeploymentError(
                        "deployment receipt flavor differs from evidence kind"
                    )
            else:
                raise EvidenceError(f"{label}.kind has no validator dispatch")
        except (
            OSError, CudaReceiptError, WheelReceiptError, MatrixReceiptError,
            CrateReceiptError, NpmReceiptError,
            OciRuntimeError,
            OciSecurityError,
            DeploymentError,
            DistributedReceiptError,
            BrowserReceiptError,
            ReproductionError,
            ZooCommunityError,
            FlagshipReceiptError,
            FlagshipQualityError,
            FlagshipRuntimeError,
            OnnxReceiptError,
            TrainingBackendReceiptError,
            TrainingPerformanceError,
            DispatchOverheadError,
            DispatchCudaReceiptError,
            EstimatorRefinementError,
            ValueError,
        ) as error:
            raise EvidenceError(f"{label} failed {kind} validation: {error}") from error
        if receipt["receipt_id"] != receipt_id:
            raise EvidenceError(f"{label}.id does not match the receipt identity")
        if kind == "compatibility-matrix":
            candidate_wheels: set[tuple[object, object, object]] = set()
            for item in artifacts.values():
                if item.get("kind") != "python-wheel":
                    continue
                wheel_path = _contained_file(
                    candidate.parent, item.get("path"), "candidate wheel path"
                )
                identity = item.get("identity", {})
                actual = (wheel_path.name, _sha256(wheel_path), wheel_path.stat().st_size)
                declared = (wheel_path.name, identity.get("sha256"), identity.get("bytes"))
                if actual != declared:
                    raise EvidenceError("candidate wheel bytes contradict candidate identity")
                candidate_wheels.add(actual)
            matrix_wheels = {
                (cell["wheel"], cell["wheel_sha256"], cell["wheel_bytes"])
                for cell in receipt["cells"]
            }
            if len(matrix_wheels) != 3 or not matrix_wheels.issubset(candidate_wheels):
                raise EvidenceError("compatibility matrix does not bind three candidate wheels")
            anchor = (
                artifact_path.name,
                artifact.get("identity", {}).get("sha256"),
                artifact.get("identity", {}).get("bytes"),
            )
            if anchor not in matrix_wheels:
                raise EvidenceError("compatibility matrix anchor is not a matrix wheel")
        elif kind == "crate-archive":
            candidate_crates = {
                (
                    item.get("id"),
                    Path(str(item.get("path", ""))).name,
                    item.get("identity", {}).get("sha256"),
                    item.get("identity", {}).get("bytes"),
                )
                for item in artifacts.values()
                if item.get("kind") == "rust-crate"
            }
            receipt_crates = {
                (
                    package["artifact_id"], package["archive"],
                    package["sha256"], package["bytes"],
                )
                for package in receipt["packages"]
            }
            if receipt_crates != candidate_crates:
                raise EvidenceError("crate receipt does not match candidate crate inventory")
            if artifact_id not in {package["artifact_id"] for package in receipt["packages"]}:
                raise EvidenceError("crate receipt anchor is not a qualified crate")
        elif kind in {"npm-archive", "browser-conformance"}:
            identity = artifact.get("identity", {})
            actual = (artifact_path.name, _sha256(artifact_path), artifact_path.stat().st_size)
            declared = (artifact_path.name, identity.get("sha256"), identity.get("bytes"))
            qualified = (
                receipt["artifact"]["name"], receipt["artifact"]["sha256"],
                receipt["artifact"]["bytes"],
            )
            if actual != declared or actual != qualified:
                raise EvidenceError("npm receipt does not bind candidate archive bytes")
        elif kind in {
            "installed-qat-tutorial", "frontend-lifecycle", "export-reload",
            "observability",
        }:
            identity = artifact.get("identity", {})
            actual = (
                artifact_path.name, _sha256(artifact_path), artifact_path.stat().st_size
            )
            declared = (
                artifact_path.name, identity.get("sha256"), identity.get("bytes")
            )
            qualified = (
                receipt["wheel_name"], receipt["wheel_sha256"].removeprefix("sha256:"),
                receipt["wheel_bytes"],
            )
            if actual != declared or actual != qualified:
                raise EvidenceError(f"{kind} receipt does not bind candidate wheel bytes")
        elif kind == "torch-dispatch-overhead":
            identity = artifact.get("identity", {})
            actual = (
                artifact_path.name,
                _sha256(artifact_path),
                artifact_path.stat().st_size,
            )
            declared = (
                artifact_path.name,
                identity.get("sha256"),
                identity.get("bytes"),
            )
            qualified = (
                receipt["wheel"]["name"],
                receipt["wheel"]["sha256"],
                receipt["wheel"]["bytes"],
            )
            if actual != declared or actual != qualified:
                raise EvidenceError(
                    "torch-dispatch-overhead receipt does not bind candidate wheel bytes"
                )
        elif kind == "torch-dispatch-cuda":
            identity = artifact.get("identity", {})
            actual = (
                artifact_path.name,
                _sha256(artifact_path),
                artifact_path.stat().st_size,
            )
            declared = (
                artifact_path.name,
                identity.get("sha256"),
                identity.get("bytes"),
            )
            qualified = (
                receipt["artifact"]["name"],
                receipt["artifact"]["sha256"],
                receipt["artifact"]["bytes"],
            )
            if actual != declared or actual != qualified:
                raise EvidenceError(
                    "torch-dispatch-cuda receipt does not bind candidate wheel bytes"
                )
        elif kind in {
            "second-machine", "independent-review", "model-zoo",
            "generated-claims", "governance-docs",
        }:
            identity = artifact.get("identity", {})
            actual = (
                artifact.get("id"), artifact.get("kind"), artifact_path.name,
                artifact_path.stat().st_size, _sha256(artifact_path),
            )
            anchor = receipt["anchor_artifact"]
            qualified = (
                anchor["id"], anchor["kind"], anchor["name"],
                anchor["bytes"], anchor["sha256"],
            )
            declared = (
                artifact.get("id"), artifact.get("kind"), artifact_path.name,
                identity.get("bytes"), identity.get("sha256"),
            )
            if actual != declared or actual != qualified:
                raise EvidenceError(f"{kind} receipt does not bind candidate anchor bytes")
        elif kind == "conversion-refinement":
            qualified = {
                (
                    track["artifact"]["id"], track["artifact"]["kind"],
                    track["artifact"]["name"], track["artifact"]["bytes"],
                    track["artifact"]["sha256"],
                )
                for track in receipt["tracks"]
            }
            actual = (
                artifact.get("id"), artifact.get("kind"), artifact_path.name,
                artifact_path.stat().st_size, _sha256(artifact_path),
            )
            if actual not in qualified:
                raise EvidenceError(
                    "conversion-refinement anchor is not a qualified model bundle"
                )
        elif kind in {"quality", "task-retention", "runtime", "physical-bytes"}:
            identity = artifact.get("identity", {})
            actual = (
                artifact.get("id"), artifact.get("kind"), artifact_path.name,
                artifact_path.stat().st_size, _sha256(artifact_path),
            )
            qualified_artifact = receipt["artifact"]
            qualified = (
                qualified_artifact["id"], qualified_artifact["kind"],
                qualified_artifact["name"], qualified_artifact["bytes"],
                qualified_artifact["sha256"],
            )
            declared = (
                artifact.get("id"), artifact.get("kind"), artifact_path.name,
                identity.get("bytes"), identity.get("sha256"),
            )
            if actual != declared or actual != qualified:
                raise EvidenceError(f"{kind} receipt does not bind candidate model bytes")
        elif kind == "onnx-inference":
            identity = artifact.get("identity", {})
            actual = (
                artifact.get("id"), artifact.get("kind"), artifact_path.name,
                artifact_path.stat().st_size, _sha256(artifact_path),
            )
            qualified_artifact = receipt["artifact"]
            qualified = (
                qualified_artifact["id"], qualified_artifact["kind"],
                qualified_artifact["name"], qualified_artifact["bytes"],
                qualified_artifact["sha256"],
            )
            declared = (
                artifact.get("id"), artifact.get("kind"), artifact_path.name,
                identity.get("bytes"), identity.get("sha256"),
            )
            if actual != declared or actual != qualified:
                raise EvidenceError("ONNX receipt does not bind candidate archive bytes")
        elif kind == "backend-manifest":
            actual = (
                artifact.get("id"), artifact.get("kind"), artifact_path.name,
                artifact_path.stat().st_size, _sha256(artifact_path),
            )
            qualified = {
                (
                    bundle["artifact"]["id"], bundle["artifact"]["kind"],
                    bundle["artifact"]["name"], bundle["artifact"]["bytes"],
                    bundle["artifact"]["sha256"],
                )
                for bundle in receipt["bundles"]
            }
            if actual not in qualified:
                raise EvidenceError("backend-manifest anchor is not a qualified bundle")
        elif kind == "performance":
            actual = (
                artifact.get("id"), artifact.get("kind"), artifact_path.name,
                artifact_path.stat().st_size, _sha256(artifact_path),
            )
            qualified = {
                (
                    measurement["artifact"]["id"], measurement["artifact"]["kind"],
                    measurement["artifact"]["name"], measurement["artifact"]["bytes"],
                    measurement["artifact"]["sha256"],
                )
                for measurement in receipt["measurements"]
            }
            if actual not in qualified:
                raise EvidenceError("performance anchor is not a measured backend bundle")
        elif kind in {"estimator-validation", "refinement", "baseline-ablation"}:
            identity = artifact.get("identity", {})
            actual = (
                artifact.get("id"), artifact.get("kind"), artifact_path.name,
                artifact_path.stat().st_size, _sha256(artifact_path),
            )
            anchor = receipt["anchor_artifact"]
            qualified = (
                anchor["id"], anchor["kind"], anchor["name"],
                anchor["bytes"], anchor["sha256"],
            )
            declared = (
                artifact.get("id"), artifact.get("kind"), artifact_path.name,
                identity.get("bytes"), identity.get("sha256"),
            )
            if actual != declared or actual != qualified:
                raise EvidenceError(f"{kind} does not bind candidate anchor bytes")
        elif kind.startswith("oci-"):
            identity = artifact.get("identity", {})
            actual = (artifact_path.name, _sha256(artifact_path), artifact_path.stat().st_size)
            declared = (artifact_path.name, identity.get("sha256"), identity.get("bytes"))
            qualified = (
                receipt["artifact"]["name"], receipt["artifact"]["sha256"],
                receipt["artifact"]["bytes"],
            )
            if actual != declared or actual != qualified:
                raise EvidenceError("OCI receipt does not bind candidate image bytes")
        elif kind.startswith("serving-deployment-"):
            identity = artifact.get("identity", {})
            actual = (
                artifact_path.name, _sha256(artifact_path), artifact_path.stat().st_size
            )
            declared = (
                artifact_path.name, identity.get("sha256"), identity.get("bytes")
            )
            qualified = (
                receipt["image_artifact"]["name"],
                receipt["image_artifact"]["sha256"],
                receipt["image_artifact"]["bytes"],
            )
            if actual != declared or actual != qualified:
                raise EvidenceError(
                    "deployment receipt does not bind candidate image bytes"
                )
        elif kind == "stage7-recipe-freeze":
            identity = artifact.get("identity", {})
            actual = (
                artifact.get("id"), artifact.get("kind"), artifact_path.name,
                artifact_path.stat().st_size, _sha256(artifact_path),
            )
            declared = (
                artifact.get("id"), artifact.get("kind"), artifact_path.name,
                identity.get("bytes"), identity.get("sha256"),
            )
            if actual != declared:
                raise EvidenceError(
                    "stage7 recipe-freeze candidate artifact bytes differ"
                )
        elif receipt["artifact"]["kind"] != "python-wheel":
            raise EvidenceError(f"{kind} receipt does not identify a Python wheel")
        run_id = receipt["run_id"]
        if run_id in run_ids:
            raise EvidenceError(f"duplicate run id {run_id!r}")
        if kind in evidence:
            raise EvidenceError(f"duplicate evidence kind {kind!r}")
        run_ids.add(run_id)
        evidence[kind] = "empirical"
        entries[receipt_id] = {**entry, "parents": list(parents)}
        validated_receipts[receipt_id] = receipt
        kinds[receipt_id] = kind
        artifact_ids[receipt_id] = artifact_id
        paths.add(logical_path)
        portable_paths.add(portable_path)
    _check_ancestry(entries)
    backend_ids = [
        receipt_id for receipt_id, kind in kinds.items() if kind == "backend-manifest"
    ]
    performance_ids = [
        receipt_id for receipt_id, kind in kinds.items() if kind == "performance"
    ]
    for receipt_id in performance_ids:
        expected_parents = set(backend_ids)
        performance = validated_receipts[receipt_id]
        if (
            len(expected_parents) != 1
            or set(entries[receipt_id]["parents"]) != expected_parents
            or performance["backend_manifest_receipt_id"] != next(iter(expected_parents))
        ):
            raise EvidenceError("performance must bind the exact backend-manifest parent")
        backend = validated_receipts[next(iter(expected_parents))]
        backend_artifacts = {
            bundle["artifact"]["id"] for bundle in backend["bundles"]
        }
        performance_artifacts = {
            measurement["artifact"]["id"]
            for measurement in performance["measurements"]
        }
        if performance_artifacts != backend_artifacts:
            raise EvidenceError("performance backend inventory differs from conformance")
    conversion_ids = [
        receipt_id for receipt_id, kind in kinds.items()
        if kind == "conversion-refinement"
    ]
    for receipt_id, kind in kinds.items():
        if kind not in {"quality", "task-retention", "runtime", "physical-bytes"}:
            continue
        expected_parents = set(conversion_ids)
        if len(expected_parents) != 1 or set(entries[receipt_id]["parents"]) != expected_parents:
            raise EvidenceError(f"{kind} must have the exact conversion-refinement parent")
        conversion = validated_receipts[next(iter(expected_parents))]
        refined_id = conversion["tracks"][2]["artifact"]["id"]
        if artifact_ids[receipt_id] != refined_id:
            raise EvidenceError(f"{kind} must bind the refined candidate model bundle")
    onnx_ids = [receipt_id for receipt_id, kind in kinds.items() if kind == "onnx-inference"]
    for receipt_id in onnx_ids:
        expected_parents = set(conversion_ids)
        if len(expected_parents) != 1 or set(entries[receipt_id]["parents"]) != expected_parents:
            raise EvidenceError("onnx-inference must have the exact conversion parent")
        conversion = validated_receipts[next(iter(expected_parents))]
        onnx = validated_receipts[receipt_id]
        if onnx["model"]["conversion_mode"] == "refined":
            track = conversion["tracks"][2]
        elif onnx["model"]["profile"] == "compact-v1":
            track = conversion["tracks"][0]
        else:
            track = conversion["tracks"][1]
        if onnx["model_artifact_id"] != track["artifact"]["id"]:
            raise EvidenceError("ONNX source does not match conversion lineage")
    estimator_ids = [
        receipt_id for receipt_id, kind in kinds.items() if kind == "estimator-validation"
    ]
    refinement_ids = [
        receipt_id for receipt_id, kind in kinds.items() if kind == "refinement"
    ]
    ablation_ids = [
        receipt_id for receipt_id, kind in kinds.items() if kind == "baseline-ablation"
    ]
    for receipt_id in refinement_ids:
        expected_parents = set(estimator_ids + conversion_ids)
        if (
            len(estimator_ids) != 1
            or len(conversion_ids) != 1
            or set(entries[receipt_id]["parents"]) != expected_parents
        ):
            raise EvidenceError(
                "refinement must have exact estimator and conversion parents"
            )
        conversion = validated_receipts[conversion_ids[0]]
        if (
            validated_receipts[receipt_id]["parent_artifact_id"]
            != conversion["tracks"][1]["artifact"]["id"]
        ):
            raise EvidenceError("refinement does not descend from NearLossless PTQ")
    quality_ids = [receipt_id for receipt_id, kind in kinds.items() if kind == "quality"]
    for receipt_id in ablation_ids:
        expected_parents = set(conversion_ids + refinement_ids + quality_ids)
        if (
            len(conversion_ids) != 1
            or len(refinement_ids) != 1
            or len(quality_ids) != 1
            or set(entries[receipt_id]["parents"]) != expected_parents
        ):
            raise EvidenceError(
                "baseline-ablation must have exact conversion, refinement, and quality parents"
            )
        refinement = validated_receipts[refinement_ids[0]]
        ablation = validated_receipts[receipt_id]
        if (
            ablation["model_artifact_id"]
            != refinement["children"][-1]["artifact"]["id"]
            or ablation["evaluation_id"]
            != validated_receipts[quality_ids[0]]["evaluation_id"]
        ):
            raise EvidenceError("baseline-ablation model or evaluation lineage differs")
    zoo_ids = [receipt_id for receipt_id, kind in kinds.items() if kind == "model-zoo"]
    claim_ids = [
        receipt_id for receipt_id, kind in kinds.items() if kind == "generated-claims"
    ]
    if zoo_ids:
        zoo_id = zoo_ids[0]
        zoo = validated_receipts[zoo_id]
        model_evidence = {
            evidence_id
            for model in zoo["models"]
            for evidence_id in model["evidence_receipt_ids"]
        }
        model_artifacts = {
            candidate_id
            for model in zoo["models"]
            for candidate_id in model["artifact_ids"]
        }
        if (
            not model_evidence.issubset(entries)
            or set(entries[zoo_id]["parents"]) != model_evidence
            or not model_artifacts.issubset(artifacts)
        ):
            raise EvidenceError(
                "model-zoo must parent admitted evidence and bind candidate artifacts"
            )
    if claim_ids:
        claim_id = claim_ids[0]
        claims = validated_receipts[claim_id]
        required_sources = set(claims["source_receipt_ids"])
        expected_sources = ({zoo_ids[0]} | model_evidence) if zoo_ids else set()
        if (
            required_sources != expected_sources
            or set(entries[claim_id]["parents"]) != expected_sources
        ):
            raise EvidenceError(
                "generated-claims must derive from model-zoo and its evidence"
            )
    review_ids = [receipt_id for receipt_id, kind in kinds.items() if kind == "independent-review"]
    second_ids = [receipt_id for receipt_id, kind in kinds.items() if kind == "second-machine"]
    if review_ids:
        review_id = review_ids[0]
        reviewer_policy = (
            Path(__file__).resolve().parent.parent
            / "release/trusted-reviewers-v1.1.allowed_signers"
        )
        if (
            reviewer_policy.is_symlink() or not reviewer_policy.is_file()
            or validated_receipts[review_id]["signer_policy"]["sha256"]
            != _sha256(reviewer_policy)
        ):
            raise EvidenceError(
                "independent-review signer policy differs from repository policy"
            )
        if validated_receipts[review_id]["review_scope_sha256"] != review_scope_sha256(
            document
        ):
            raise EvidenceError(
                "independent-review must bind the exact pre-review registry scope"
            )
        required_reviewed = set(entries) - {review_id}
        reviewed = set(validated_receipts[review_id]["reviewed_receipt_ids"])
        if reviewed != required_reviewed or set(entries[review_id]["parents"]) != required_reviewed:
            raise EvidenceError(
                "independent-review must review and parent every other registry receipt"
            )
    if review_ids and second_ids:
        review_id = review_ids[0]
        second_id = second_ids[0]
        operator = validated_receipts[second_id]["operator"]
        reviewer = validated_receipts[review_id]["reviewer"]
        if (
            operator["id"] == reviewer["id"]
            or operator["organization"] == reviewer["organization"]
        ):
            raise EvidenceError("reviewer must be independent from reproduction operator")
    for receipt_id, kind in kinds.items():
        if not kind.startswith("serving-deployment-"):
            continue
        flavor = kind.removeprefix("serving-deployment-")
        required_parent_kinds = {f"oci-runtime-{flavor}", f"oci-security-{flavor}"}
        parents = entries[receipt_id]["parents"]
        if {kinds.get(parent) for parent in parents} != required_parent_kinds:
            raise EvidenceError(
                f"{kind} must have exact matching runtime and security parents"
            )
        if any(artifact_ids[parent] != artifact_ids[receipt_id] for parent in parents):
            raise EvidenceError(f"{kind} parents must bind the same candidate image")
        runtime_parent = next(
            validated_receipts[parent]
            for parent in parents
            if kinds[parent] == f"oci-runtime-{flavor}"
        )
        deployment = validated_receipts[receipt_id]
        if runtime_parent.get("startup_receipt") != deployment.get("workload", {}).get(
            "startup_receipt"
        ):
            raise EvidenceError(
                f"{kind} startup receipt differs from runtime parent"
            )

    rows = [_gate_row(gate_id, required, evidence) for gate_id, required in GATES]
    ready = all(row["status"] == "PASS" for row in rows)
    return {
        "schema": REPORT_SCHEMA,
        "release": release,
        "source_revision": revision,
        "candidate_manifest_sha256": expected_candidate,
        "evidence_registry_sha256": _sha256(registry),
        "ready": ready,
        "rows": rows,
        "external_activation": "EXTERNAL_AUTH_REQUIRED",
    }


def render(report: dict[str, Any]) -> str:
    lines = ["STATUS           GATE                  MISSING"]
    for row in report["rows"]:
        missing = ",".join(row["missing_kinds"] + row["structural_kinds"]) or "-"
        lines.append(f"{row['status']:<16} {row['id']:<21} {missing}")
    lines.append("EXTERNAL_AUTH_REQUIRED public-activation     explicit-authorization")
    return "\n".join(lines)
