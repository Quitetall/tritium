"""Portable validator for installed-wheel observability qualification."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import runpy
from typing import Any

if __package__:
    from ._telemetry_binary import tensorboard_values, wandb_values
    from ._wheel_identity import file_sha256, wheel_identity
else:
    _MODULE_ROOT = Path(__file__).resolve().parent
    _TELEMETRY = runpy.run_path(_MODULE_ROOT / "_telemetry_binary.py")
    _WHEEL = runpy.run_path(_MODULE_ROOT / "_wheel_identity.py")
    tensorboard_values = _TELEMETRY["tensorboard_values"]
    wandb_values = _TELEMETRY["wandb_values"]
    file_sha256 = _WHEEL["file_sha256"]
    wheel_identity = _WHEEL["wheel_identity"]


SCHEMA = "tritium.installed-observability.v1"
MAX_RECEIPT_BYTES = 1024 * 1024
FIELDS = {
    "schema",
    "receipt_id",
    "passed",
    "source_revision",
    "release",
    "run_id",
    "wheel_name",
    "wheel_bytes",
    "wheel_sha256",
    "distribution_version",
    "wheel_file_count",
    "wheel_tree_sha256",
    "installed_file_count",
    "installed_tree_sha256",
    "installed_record_sha256",
    "tritium_module",
    "python_version",
    "torch_version",
    "tensorboard_version",
    "wandb_version",
    "opentelemetry_api_version",
    "opentelemetry_sdk_version",
    "environment",
    "snapshot",
    "adapters",
    "telemetry_dir",
    "telemetry_bytes",
    "telemetry_file_count",
    "telemetry_tree_sha256",
}
ENVIRONMENT_FIELDS = {"repository_absent", "compiler_absent", "network_mode"}
SNAPSHOT_FIELDS = {
    "schema_version",
    "step",
    "tensor_count",
    "tensor_path",
    "aliases",
    "trit_counts",
    "physical_bytes",
    "code_scale_bpw",
    "zero_rate",
    "gradient_finite",
    "scalar_metric_count",
    "scalar_metrics",
}
ADAPTER_FIELDS = {"tensorboard", "wandb", "opentelemetry"}
TENSORBOARD_FIELDS = {"event_files", "scalar_tags", "histogram_tags"}
WANDB_FIELDS = {"mode", "log_calls", "scalar_values", "histograms"}
OTEL_FIELDS = {"metric_names", "data_points", "measurements"}
OTEL_METRICS = [
    "tritium.experiment.measurement",
    "tritium.snapshot.code_scale_bpw",
    "tritium.snapshot.code_scale_bytes",
    "tritium.snapshot.tensor_count",
]
RUNTIME_VERSION_FIELDS = {
    "python_version",
    "torch_version",
    "tensorboard_version",
    "wandb_version",
    "opentelemetry_api_version",
    "opentelemetry_sdk_version",
}
HEX = frozenset("0123456789abcdef")
EXPECTED_EXTRA_METRICS = {
    "tritium/memory/resident_bytes": 4096.0,
    "tritium/runtime/decode_ms": 3.5,
    "tritium/teacher_kl": 0.125,
}
EXPECTED_HISTOGRAM_TAG = "tritium/tensor/left.weight/plane_0/trits"
EXPECTED_SCALAR_TAGS = frozenset(
    {
        "tritium/tensors",
        "tritium/physical_bytes",
        "tritium/code_scale_bpw",
        *EXPECTED_EXTRA_METRICS,
        "tritium/tensor/left.weight/planes",
        "tritium/tensor/left.weight/physical_bytes",
        "tritium/tensor/left.weight/code_scale_bpw",
        "tritium/tensor/left.weight/zero_rate",
        "tritium/tensor/left.weight/reconstruction_rmse",
        "tritium/tensor/left.weight/saturation_rate",
        "tritium/tensor/left.weight/gradient_l2",
        "tritium/tensor/left.weight/gradient_finite",
        "tritium/tensor/left.weight/plane_0/negative",
        "tritium/tensor/left.weight/plane_0/zero",
        "tritium/tensor/left.weight/plane_0/positive",
        "tritium/tensor/left.weight/plane_0/scale_min",
        "tritium/tensor/left.weight/plane_0/scale_max",
        "tritium/tensor/left.weight/plane_0/scale_mean",
        "tritium/tensor/left.weight/plane_0/scale_std",
        "tritium/tensor/left.weight/plane_0/saturation_rate",
    }
)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def receipt_id(receipt: dict[str, object]) -> str:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    return "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()


def _ordinary_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an ordinary non-symlink file")
    return path.resolve(strict=True)


def tree_identity(root: Path) -> dict[str, object]:
    """Hash one path-sensitive, symlink-free telemetry tree."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("observability telemetry must be an ordinary directory")
    root = root.resolve(strict=True)
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("observability telemetry must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("observability telemetry contains a non-file entry")
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not entries:
        raise ValueError("observability telemetry must contain files")
    return {
        "bytes": sum(int(entry["bytes"]) for entry in entries),
        "file_count": len(entries),
        "sha256": "sha256:" + hashlib.sha256(canonical(entries)).hexdigest(),
    }


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"observability {label} fields do not match schema")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"observability {label} must be a non-empty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"observability {label} must be a positive integer")
    return value


def _digest(value: Any, label: str) -> str:
    text = _string(value, label)
    if (
        not text.startswith("sha256:")
        or len(text) != 71
        or any(character not in HEX for character in text[7:])
    ):
        raise ValueError(f"observability {label} must be a canonical digest")
    return text


def _json_file(path: Path, label: str) -> Any:
    path = _ordinary_file(path, label)
    if path.stat().st_size > MAX_RECEIPT_BYTES:
        raise ValueError(f"{label} exceeds metadata size limit")
    try:
        return json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must contain UTF-8 JSON") from error


def _validate_retained_telemetry(
    root: Path,
    snapshot: dict[str, Any],
    adapters: dict[str, Any],
) -> None:
    tensorboard_dir = root / "tensorboard"
    event_files = tuple(sorted(tensorboard_dir.glob("events.out.tfevents.*")))
    if len(event_files) != adapters["tensorboard"]["event_files"]:
        raise ValueError("observability TensorBoard event inventory differs")
    tensorboard_scalars, tensorboard_histogram = tensorboard_values(
        event_files,
        expected_scalar_tags=EXPECTED_SCALAR_TAGS,
        expected_step=17,
    )
    for name, expected in snapshot["scalar_metrics"].items():
        observed = tensorboard_scalars[name]
        if not math.isfinite(observed) or not math.isclose(
            observed, float(expected), rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("observability TensorBoard scalar values differ")
    expected_histogram = {
        "tag": EXPECTED_HISTOGRAM_TAG,
        "minimum": -1.0,
        "maximum": 1.0,
        "count": 8.0,
        "sum": 0.0,
        "sum_squares": 4.0,
        "bucket_limits": [-0.5, 0.5, 1.5],
        "bucket_counts": [2.0, 4.0, 2.0],
    }
    if tensorboard_histogram != expected_histogram:
        raise ValueError("observability TensorBoard histogram values differ")
    tensorboard_manifest = _json_file(
        tensorboard_dir / "adapter.json", "TensorBoard adapter manifest"
    )
    if tensorboard_manifest != {
        "histogram_tags": [EXPECTED_HISTOGRAM_TAG],
        "scalar_metrics": snapshot["scalar_metrics"],
    }:
        raise ValueError("observability TensorBoard adapter manifest differs")

    wandb_dir = root / "wandb"
    wandb_run = _ordinary_file(wandb_dir / "offline-run.wandb", "W&B offline run")
    wandb_scalars, wandb_histogram = wandb_values(
        wandb_run,
        expected_scalar_tags=EXPECTED_SCALAR_TAGS,
        expected_histogram_tag=EXPECTED_HISTOGRAM_TAG,
        expected_step=17,
    )
    for name, expected in snapshot["scalar_metrics"].items():
        observed = wandb_scalars[name]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not math.isclose(
                float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ValueError("observability W&B scalar values differ")
    if wandb_histogram != {
        "_type": "histogram",
        "values": [2, 4, 2],
        "bins": [-1.5, -0.5, 0.5, 1.5],
    }:
        raise ValueError("observability W&B histogram values differ")
    if _json_file(wandb_dir / "adapter.json", "W&B adapter manifest") != {
        **adapters["wandb"],
        "scalar_metrics": snapshot["scalar_metrics"],
    }:
        raise ValueError("observability W&B adapter manifest differs")

    if _json_file(
        root / "opentelemetry" / "metrics.json",
        "OpenTelemetry measurement export",
    ) != adapters["opentelemetry"]:
        raise ValueError("observability OpenTelemetry measurement export differs")


def validate_receipt(
    receipt_path: Path,
    *,
    expected_wheel: Path | None = None,
    expected_source_revision: str | None = None,
    expected_release: str | None = None,
    expected_runtime_versions: dict[str, str] | None = None,
) -> dict[str, object]:
    """Validate receipt semantics, candidate wheel, and retained telemetry bytes."""

    receipt_path = _ordinary_file(receipt_path, "observability receipt")
    if receipt_path.stat().st_size > MAX_RECEIPT_BYTES:
        raise ValueError("observability receipt exceeds metadata size limit")
    try:
        receipt = json.loads(receipt_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("observability receipt must contain UTF-8 JSON") from error
    _object(receipt, FIELDS, "receipt")
    if receipt["schema"] != SCHEMA or receipt["passed"] is not True:
        raise ValueError("observability receipt did not pass schema v1")
    revision = _string(receipt["source_revision"], "source revision")
    if len(revision) != 40 or any(character not in HEX for character in revision):
        raise ValueError("observability source revision must be 40 lowercase hex characters")
    for field in (
        "release",
        "run_id",
        "tritium_module",
        "python_version",
        "torch_version",
        "tensorboard_version",
        "wandb_version",
        "opentelemetry_api_version",
        "opentelemetry_sdk_version",
    ):
        _string(receipt[field], field)
    wheel_name = _string(receipt["wheel_name"], "wheel name")
    if Path(wheel_name).name != wheel_name or not wheel_name.endswith(".whl"):
        raise ValueError("observability wheel name is invalid")
    _positive_int(receipt["wheel_bytes"], "wheel bytes")
    _digest(receipt["wheel_sha256"], "wheel sha256")
    wheel_file_count = _positive_int(receipt["wheel_file_count"], "wheel file count")
    installed_file_count = _positive_int(
        receipt["installed_file_count"], "installed file count"
    )
    if installed_file_count < wheel_file_count or installed_file_count > wheel_file_count + 4:
        raise ValueError("observability installed file inventory differs")
    _digest(receipt["wheel_tree_sha256"], "wheel tree sha256")
    _digest(receipt["installed_tree_sha256"], "installed tree sha256")
    _digest(receipt["installed_record_sha256"], "installed RECORD sha256")
    distribution_version = _string(receipt["distribution_version"], "distribution version")
    expected_distribution_version = receipt["release"].replace("-rc.", "rc")
    if distribution_version != expected_distribution_version:
        raise ValueError("observability distribution and release versions differ")
    if not receipt["tritium_module"].replace("\\", "/").endswith(
        "/tritium/__init__.py"
    ):
        raise ValueError("observability imported module path differs")

    environment = _object(receipt["environment"], ENVIRONMENT_FIELDS, "environment")
    if (
        environment["repository_absent"] is not True
        or environment["compiler_absent"] is not True
        or environment["network_mode"] != "offline"
    ):
        raise ValueError("observability qualification was not source/compiler/network-free")

    snapshot = _object(receipt["snapshot"], SNAPSHOT_FIELDS, "snapshot")
    if (
        snapshot["schema_version"] != 1
        or snapshot["step"] != 17
        or snapshot["tensor_count"] != 1
        or snapshot["tensor_path"] != "left.weight"
        or snapshot["aliases"] != ["left.weight", "right.weight"]
        or snapshot["trit_counts"] != [2, 4, 2]
        or snapshot["physical_bytes"] != 6
        or snapshot["gradient_finite"] is not True
    ):
        raise ValueError("observability snapshot fixture differs")
    for field in ("code_scale_bpw", "zero_rate"):
        value = snapshot[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"observability snapshot {field} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"observability snapshot {field} must be finite")
    if float(snapshot["code_scale_bpw"]) != 6.0 or float(snapshot["zero_rate"]) != 0.5:
        raise ValueError("observability snapshot physical metrics differ")
    scalar_count = _positive_int(snapshot["scalar_metric_count"], "scalar metric count")
    scalar_metrics = snapshot["scalar_metrics"]
    if (
        not isinstance(scalar_metrics, dict)
        or set(scalar_metrics) != EXPECTED_SCALAR_TAGS
        or scalar_count != len(EXPECTED_SCALAR_TAGS)
    ):
        raise ValueError("observability scalar metric inventory differs")
    for name, value in scalar_metrics.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"observability scalar metric {name!r} is not finite")
    if any(float(scalar_metrics[name]) != expected for name, expected in EXPECTED_EXTRA_METRICS.items()):
        raise ValueError("observability external metric values differ")
    fixed_metrics = {
        "tritium/tensors": 1.0,
        "tritium/physical_bytes": 6.0,
        "tritium/code_scale_bpw": 6.0,
        "tritium/tensor/left.weight/planes": 1.0,
        "tritium/tensor/left.weight/physical_bytes": 6.0,
        "tritium/tensor/left.weight/code_scale_bpw": 6.0,
        "tritium/tensor/left.weight/zero_rate": 0.5,
        "tritium/tensor/left.weight/gradient_finite": 1.0,
        "tritium/tensor/left.weight/plane_0/negative": 2.0,
        "tritium/tensor/left.weight/plane_0/zero": 4.0,
        "tritium/tensor/left.weight/plane_0/positive": 2.0,
    }
    if any(float(scalar_metrics[name]) != expected for name, expected in fixed_metrics.items()):
        raise ValueError("observability fixed scalar metric values differ")

    adapters = _object(receipt["adapters"], ADAPTER_FIELDS, "adapters")
    tensorboard = _object(adapters["tensorboard"], TENSORBOARD_FIELDS, "TensorBoard")
    if (
        _positive_int(tensorboard["event_files"], "TensorBoard event files") < 1
        or tensorboard["scalar_tags"] != scalar_count
        or tensorboard["histogram_tags"] != 1
    ):
        raise ValueError("observability TensorBoard evidence differs")
    wandb = _object(adapters["wandb"], WANDB_FIELDS, "W&B")
    if (
        wandb["mode"] != "offline"
        or wandb["log_calls"] != 1
        or wandb["scalar_values"] != scalar_count
        or wandb["histograms"] != 1
    ):
        raise ValueError("observability W&B evidence differs")
    otel = _object(adapters["opentelemetry"], OTEL_FIELDS, "OpenTelemetry")
    if (
        otel["metric_names"] != OTEL_METRICS
        or otel["data_points"] != 6
        or not isinstance(otel["measurements"], dict)
        or set(otel["measurements"]) != set(OTEL_METRICS)
    ):
        raise ValueError("observability OpenTelemetry evidence differs")
    expected_otel = {
        "tritium.experiment.measurement": [
            {
                "attributes": {"metric": name.removeprefix("tritium/"), "source": "caller"},
                "value": value,
            }
            for name, value in sorted(EXPECTED_EXTRA_METRICS.items())
        ],
        "tritium.snapshot.code_scale_bpw": [{"attributes": {}, "value": 6.0}],
        "tritium.snapshot.code_scale_bytes": [{"attributes": {}, "value": 6}],
        "tritium.snapshot.tensor_count": [{"attributes": {}, "value": 1}],
    }
    if otel["measurements"] != expected_otel:
        raise ValueError("observability OpenTelemetry measurements differ")

    if receipt["telemetry_dir"] != "telemetry":
        raise ValueError("observability telemetry directory differs")
    declared_tree = {
        "bytes": _positive_int(receipt["telemetry_bytes"], "telemetry bytes"),
        "file_count": _positive_int(
            receipt["telemetry_file_count"], "telemetry file count"
        ),
        "sha256": _digest(receipt["telemetry_tree_sha256"], "telemetry tree sha256"),
    }
    observed_tree = tree_identity(receipt_path.parent / "telemetry")
    if observed_tree != declared_tree:
        raise ValueError("observability telemetry tree identity mismatch")
    _validate_retained_telemetry(receipt_path.parent / "telemetry", snapshot, adapters)
    if receipt["receipt_id"] != receipt_id(receipt):
        raise ValueError("observability receipt identity mismatch")
    if expected_source_revision is not None and revision != expected_source_revision:
        raise ValueError("observability source revision mismatch")
    if expected_release is not None and receipt["release"] != expected_release:
        raise ValueError("observability release mismatch")
    if expected_runtime_versions is not None:
        if set(expected_runtime_versions) != RUNTIME_VERSION_FIELDS or any(
            receipt[field] != expected_runtime_versions[field]
            for field in RUNTIME_VERSION_FIELDS
        ):
            raise ValueError("observability runtime or adapter version mismatch")
    if expected_wheel is not None:
        wheel = _ordinary_file(expected_wheel, "observability candidate wheel")
        wheel_inventory = wheel_identity(wheel)
        if (
            wheel.name != wheel_name
            or wheel.stat().st_size != receipt["wheel_bytes"]
            or file_sha256(wheel) != receipt["wheel_sha256"]
            or wheel_inventory["distribution_version"] != distribution_version
            or wheel_inventory["file_count"] != receipt["wheel_file_count"]
            or wheel_inventory["tree_sha256"] != receipt["wheel_tree_sha256"]
        ):
            raise ValueError("observability receipt does not bind candidate wheel")
    return receipt


__all__ = [
    "SCHEMA",
    "file_sha256",
    "receipt_id",
    "tree_identity",
    "validate_receipt",
    "wheel_identity",
]
