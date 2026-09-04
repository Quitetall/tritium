"""Source-free installed-wheel qualification for telemetry adapters."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any

import torch
from torch import nn

import tritium

from .config import TernaryConfig
from .conversion import prepare_qat
from .observability import (
    OpenTelemetryDiagnostics,
    WandbDiagnostics,
    collect_diagnostics,
    log_tensorboard,
)
from .observability_receipt import (
    EXPECTED_EXTRA_METRICS,
    OTEL_METRICS,
    SCHEMA,
    file_sha256,
    receipt_id,
    tree_identity,
    validate_receipt,
    wheel_identity,
)


_COMPILERS = ("cargo", "rustc", "cc", "c++", "gcc", "g++", "clang", "clang++")


class _TinyTiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 2, bias=False)
        self.right = nn.Linear(4, 2, bias=False)
        self.right.weight = self.left.weight

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.left(values) + self.right(values)


class _RecordingRun:
    def __init__(self, run: Any) -> None:
        self.run = run
        self.log_calls = 0
        self.scalar_values = 0
        self.histograms = 0

    def log(self, payload: dict[str, object], *, step: int) -> None:
        self.log_calls += 1
        scalar_values = sum(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in payload.values()
        )
        self.scalar_values += scalar_values
        self.histograms += len(payload) - scalar_values
        self.run.log(payload, step=step)


def _installed_distribution(
    wheel_inventory: dict[str, object],
) -> tuple[str, Path, dict[str, object]]:
    try:
        distribution = importlib.metadata.distribution("pytritium")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "observability qualification requires installed pytritium"
        ) from error
    module = Path(tritium.__file__).resolve(strict=True)
    if distribution.files is None:
        raise RuntimeError("installed pytritium distribution has no file inventory")
    owned_paths = [str(item).replace("\\", "/") for item in distribution.files]
    if len(owned_paths) != len(set(owned_paths)):
        raise RuntimeError("installed pytritium file inventory contains duplicates")
    owned = {distribution.locate_file(item).resolve() for item in distribution.files}
    if module not in owned:
        raise RuntimeError("imported tritium package is not owned by pytritium")
    entries = wheel_inventory["entries"]
    assert isinstance(entries, list)
    expected = {str(entry["path"]): entry for entry in entries}
    record_path = str(wheel_inventory["record_path"])
    dist_info = record_path.removesuffix("RECORD")
    allowed_installer_files = {
        dist_info + name
        for name in ("INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json")
    }
    actual_paths = set(owned_paths)
    if not set(expected).issubset(actual_paths) or not (
        actual_paths - set(expected)
    ).issubset(allowed_installer_files):
        raise RuntimeError("installed pytritium file inventory differs from wheel")
    installed_entries = []
    for logical in sorted(actual_paths):
        installed = distribution.locate_file(logical)
        if installed.is_symlink() or not installed.is_file():
            raise RuntimeError("installed pytritium RECORD contains a non-file")
        identity = {
            "path": logical,
            "bytes": installed.stat().st_size,
            "sha256": file_sha256(installed),
        }
        installed_entries.append(identity)
        if logical == record_path or logical in allowed_installer_files:
            continue
        entry = expected[logical]
        assert isinstance(entry, dict)
        if (
            identity["bytes"] != entry["bytes"]
            or identity["sha256"] != entry["sha256"]
        ):
            raise RuntimeError("installed pytritium files differ from candidate wheel")
    if distribution.version != wheel_inventory["distribution_version"]:
        raise RuntimeError("installed pytritium version differs from candidate wheel")
    record_entry = next(
        entry for entry in installed_entries if entry["path"] == record_path
    )
    installed_identity = {
        "file_count": len(installed_entries),
        "tree_sha256": "sha256:"
        + hashlib.sha256(
            json.dumps(
                installed_entries, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "record_sha256": record_entry["sha256"],
    }
    return distribution.version, module, installed_identity


def _repository_absent(path: Path) -> bool:
    path = path.resolve(strict=True)
    return all(
        not (parent / ".git").exists() and not (parent / "Cargo.toml").exists()
        for parent in (path, *path.parents)
    )


def _validate_inputs(
    output_dir: Path,
    wheel_artifact: Path,
    source_revision: str,
    release: str,
    run_id: str,
) -> Path:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if wheel_artifact.is_symlink() or not wheel_artifact.is_file():
        raise ValueError("wheel artifact must be an ordinary file")
    wheel_artifact = wheel_artifact.resolve(strict=True)
    if not wheel_artifact.name.endswith(".whl"):
        raise ValueError("wheel artifact must have a .whl filename")
    if (
        len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise ValueError("source revision must be 40 lowercase hexadecimal characters")
    if not release or not run_id:
        raise ValueError("release and run id must be non-empty")
    return wheel_artifact


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"observability qualification requires {name}") from error


def _runtime_versions() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "tensorboard_version": _version("tensorboard"),
        "wandb_version": _version("wandb"),
        "opentelemetry_api_version": _version("opentelemetry-api"),
        "opentelemetry_sdk_version": _version("opentelemetry-sdk"),
    }


def _metric_data(reader: Any) -> tuple[list[str], int, dict[str, list[dict[str, object]]]]:
    data = reader.get_metrics_data()
    names: list[str] = []
    points = 0
    measurements: dict[str, list[dict[str, object]]] = {}
    for resource in data.resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                names.append(metric.name)
                records = []
                for point in metric.data.data_points:
                    records.append(
                        {
                            "attributes": dict(sorted(dict(point.attributes).items())),
                            "value": point.value,
                        }
                    )
                records.sort(key=lambda record: json.dumps(record, sort_keys=True))
                measurements[metric.name] = records
                points += len(records)
    return sorted(names), points, dict(sorted(measurements.items()))


def run_installed_observability(
    output_dir: Path,
    *,
    wheel_artifact: Path,
    source_revision: str,
    release: str,
    run_id: str,
) -> dict[str, object]:
    """Exercise all public telemetry adapters from one installed wheel."""

    wheel_artifact = _validate_inputs(
        output_dir, wheel_artifact, source_revision, release, run_id
    )
    wheel_inventory = wheel_identity(wheel_artifact)
    distribution_version, module_path, installed_identity = _installed_distribution(
        wheel_inventory
    )
    repository_absent = _repository_absent(Path.cwd()) and _repository_absent(module_path)
    compiler_absent = all(shutil.which(command) is None for command in _COMPILERS)
    if not repository_absent or not compiler_absent:
        raise RuntimeError(
            "observability qualification requires a source/compiler-free environment"
        )

    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_DISABLE_GIT"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    output_dir.mkdir(parents=True)
    telemetry_dir = output_dir / "telemetry"
    tensorboard_dir = telemetry_dir / "tensorboard"
    wandb_dir = telemetry_dir / "wandb"
    otel_dir = telemetry_dir / "opentelemetry"
    tensorboard_dir.mkdir(parents=True)
    wandb_dir.mkdir(parents=True)
    otel_dir.mkdir(parents=True)

    model = prepare_qat(_TinyTiedModel(), TernaryConfig.qat())
    with torch.no_grad():
        model.left.weight.copy_(
            torch.tensor([[-2.0, -0.4, 0.3, 2.1], [0.0, 1.0, -1.0, 0.2]])
        )
    model(torch.ones(1, 4)).sum().backward()
    snapshot = collect_diagnostics(
        model,
        step=17,
        extra_metrics={
            "memory/resident_bytes": 4096.0,
            "runtime/decode_ms": 3.5,
            "teacher_kl": 0.125,
        },
    )
    tensor = snapshot.tensors[0]
    metrics = snapshot.scalar_metrics()

    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    writer = SummaryWriter(log_dir=str(tensorboard_dir), max_queue=1, flush_secs=1)
    log_tensorboard(snapshot, writer)
    writer.flush()
    writer.close()
    accumulator = EventAccumulator(str(tensorboard_dir))
    accumulator.Reload()
    tensorboard_tags = accumulator.Tags()
    scalar_tags = sorted(tensorboard_tags.get("scalars", []))
    histogram_tags = sorted(tensorboard_tags.get("histograms", []))
    if set(scalar_tags) != set(metrics) or len(histogram_tags) != 1:
        raise RuntimeError("TensorBoard did not retain complete ternary diagnostics")
    (tensorboard_dir / "adapter.json").write_bytes(
        json.dumps(
            {
                "histogram_tags": histogram_tags,
                "scalar_metrics": metrics,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )

    with tempfile.TemporaryDirectory(prefix="tritium-wandb-") as raw_work:
        wandb_work = Path(raw_work)
        for name in ("cache", "config", "data", "artifacts"):
            (wandb_work / name).mkdir()
        os.environ["WANDB_CACHE_DIR"] = str(wandb_work / "cache")
        os.environ["WANDB_CONFIG_DIR"] = str(wandb_work / "config")
        os.environ["WANDB_DATA_DIR"] = str(wandb_work / "data")
        os.environ["WANDB_ARTIFACT_DIR"] = str(wandb_work / "artifacts")
        import wandb

        run = wandb.init(
            project="tritium-observability-qualification",
            dir=str(wandb_work),
            mode="offline",
            reinit="finish_previous",
            settings=wandb.Settings(console="off", disable_git=True, silent=True),
        )
        if run is None or run.settings.mode != "offline":
            raise RuntimeError("W&B did not enter offline mode")
        recording = _RecordingRun(run)
        WandbDiagnostics(recording).log(snapshot)
        run.finish(exit_code=0, quiet=True)
        run_files = tuple(
            path
            for path in wandb_work.rglob("run-*.wandb")
            if path.is_file() and not path.is_symlink()
        )
        if len(run_files) != 1:
            raise RuntimeError("W&B did not retain exactly one offline run")
        shutil.copyfile(run_files[0], wandb_dir / "offline-run.wandb")
    if recording.log_calls != 1 or recording.scalar_values != len(metrics):
        raise RuntimeError("W&B did not retain complete ternary diagnostics")
    (wandb_dir / "adapter.json").write_bytes(
        json.dumps(
            {
                "mode": "offline",
                "log_calls": recording.log_calls,
                "scalar_values": recording.scalar_values,
                "histograms": recording.histograms,
                "scalar_metrics": metrics,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )

    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    adapter = OpenTelemetryDiagnostics(provider.get_meter("tritium.qualification"))
    adapter.log(snapshot)
    metric_names, data_points, measurements = _metric_data(reader)
    provider.shutdown()
    if (
        metric_names != OTEL_METRICS
        or data_points != 6
        or not set(EXPECTED_EXTRA_METRICS).issubset(metrics)
    ):
        raise RuntimeError("OpenTelemetry did not retain complete aggregate diagnostics")
    otel_evidence = {
        "metric_names": metric_names,
        "data_points": data_points,
        "measurements": measurements,
    }
    (otel_dir / "metrics.json").write_bytes(
        json.dumps(
            otel_evidence,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )

    tree = tree_identity(telemetry_dir)
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "passed": True,
        "source_revision": source_revision,
        "release": release,
        "run_id": run_id,
        "wheel_name": wheel_artifact.name,
        "wheel_bytes": wheel_artifact.stat().st_size,
        "wheel_sha256": file_sha256(wheel_artifact),
        "distribution_version": distribution_version,
        "wheel_file_count": wheel_inventory["file_count"],
        "wheel_tree_sha256": wheel_inventory["tree_sha256"],
        "installed_file_count": installed_identity["file_count"],
        "installed_tree_sha256": installed_identity["tree_sha256"],
        "installed_record_sha256": installed_identity["record_sha256"],
        "tritium_module": str(module_path),
        **_runtime_versions(),
        "environment": {
            "repository_absent": repository_absent,
            "compiler_absent": compiler_absent,
            "network_mode": "offline",
        },
        "snapshot": {
            "schema_version": snapshot.schema_version,
            "step": snapshot.step,
            "tensor_count": len(snapshot.tensors),
            "tensor_path": tensor.path,
            "aliases": list(tensor.aliases),
            "trit_counts": list(tensor.planes[0].trits.as_tuple()),
            "physical_bytes": tensor.physical_bytes,
            "code_scale_bpw": tensor.code_scale_bpw,
            "zero_rate": tensor.zero_rate,
            "gradient_finite": tensor.gradient_finite,
            "scalar_metric_count": len(metrics),
            "scalar_metrics": metrics,
        },
        "adapters": {
            "tensorboard": {
                "event_files": len(tuple(tensorboard_dir.glob("events.out.tfevents.*"))),
                "scalar_tags": len(scalar_tags),
                "histogram_tags": len(histogram_tags),
            },
            "wandb": {
                "mode": "offline",
                "log_calls": recording.log_calls,
                "scalar_values": recording.scalar_values,
                "histograms": recording.histograms,
            },
            "opentelemetry": {
                "metric_names": metric_names,
                "data_points": data_points,
                "measurements": measurements,
            },
        },
        "telemetry_dir": "telemetry",
        "telemetry_bytes": tree["bytes"],
        "telemetry_file_count": tree["file_count"],
        "telemetry_tree_sha256": tree["sha256"],
    }
    receipt["receipt_id"] = receipt_id(receipt)
    return receipt


def _write_receipt(output_dir: Path, receipt: dict[str, object]) -> Path:
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".receipt.", suffix=".json", dir=output_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        destination = output_dir / "receipt.json"
        os.replace(temporary, destination)
        if os.name != "nt":
            directory = os.open(output_dir, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return destination
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--check-receipt", type=Path)
    parser.add_argument("--wheel-artifact", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.check_receipt is not None:
        wheel = args.wheel_artifact.absolute()
        inventory = wheel_identity(wheel)
        version, module, installed = _installed_distribution(inventory)
        receipt = validate_receipt(
            args.check_receipt.absolute(),
            expected_wheel=wheel,
            expected_source_revision=args.source_revision,
            expected_release=args.release,
            expected_runtime_versions=_runtime_versions(),
        )
        if (
            receipt["distribution_version"] != version
            or receipt["tritium_module"] != str(module)
            or receipt["installed_file_count"] != installed["file_count"]
            or receipt["installed_tree_sha256"] != installed["tree_sha256"]
            or receipt["installed_record_sha256"] != installed["record_sha256"]
        ):
            raise ValueError(
                "observability receipt does not bind executed installation inventory"
            )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.run_id is None:
        parser.error("--run-id is required with --output-dir")
    assert args.output_dir is not None
    output = args.output_dir.absolute()
    receipt = run_installed_observability(
        output,
        wheel_artifact=args.wheel_artifact.absolute(),
        source_revision=args.source_revision,
        release=args.release,
        run_id=args.run_id,
    )
    _write_receipt(output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_installed_observability"]
