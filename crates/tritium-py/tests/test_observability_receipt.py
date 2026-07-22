"""Strict installed-observability receipt validation."""

from __future__ import annotations

import copy
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import runpy
import struct
import zipfile
import zlib

import pytest

MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1]
    / "python/tritium/torch/observability_receipt.py"
)
TELEMETRY_BINARY = runpy.run_path(
    Path(__file__).resolve().parents[1]
    / "python/tritium/torch/_telemetry_binary.py"
)
OTEL_METRICS = MODULE["OTEL_METRICS"]
EXPECTED_EXTRA_METRICS = MODULE["EXPECTED_EXTRA_METRICS"]
EXPECTED_HISTOGRAM_TAG = MODULE["EXPECTED_HISTOGRAM_TAG"]
EXPECTED_SCALAR_TAGS = MODULE["EXPECTED_SCALAR_TAGS"]
receipt_id = MODULE["receipt_id"]
tree_identity = MODULE["tree_identity"]
validate_receipt = MODULE["validate_receipt"]
wheel_identity = MODULE["wheel_identity"]
masked_crc32c = TELEMETRY_BINARY["masked_crc32c"]


def _varint(value: int) -> bytes:
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _field_varint(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def _field_bytes(number: int, value: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _field_fixed32(number: int, value: float) -> bytes:
    return _varint((number << 3) | 5) + struct.pack("<f", value)


def _field_fixed64(number: int, value: float) -> bytes:
    return _varint((number << 3) | 1) + struct.pack("<d", value)


def _tfrecord(payload: bytes) -> bytes:
    length = struct.pack("<Q", len(payload))
    return (
        length
        + struct.pack("<I", masked_crc32c(length))
        + payload
        + struct.pack("<I", masked_crc32c(payload))
    )


def _tensorboard_events(metrics: dict[str, float]) -> bytes:
    records = [_tfrecord(_field_bytes(3, b"brain.Event:2"))]
    for name, value in sorted(metrics.items()):
        summary_value = _field_bytes(1, name.encode()) + _field_fixed32(2, value)
        event = _field_varint(2, 17) + _field_bytes(5, _field_bytes(1, summary_value))
        records.append(_tfrecord(event))
    histogram = b"".join(
        (
            _field_fixed64(1, -1.0),
            _field_fixed64(2, 1.0),
            _field_fixed64(3, 8.0),
            _field_fixed64(4, 0.0),
            _field_fixed64(5, 4.0),
            _field_bytes(6, struct.pack("<3d", -0.5, 0.5, 1.5)),
            _field_bytes(7, struct.pack("<3d", 2.0, 4.0, 2.0)),
        )
    )
    summary_value = _field_bytes(1, EXPECTED_HISTOGRAM_TAG.encode()) + _field_bytes(
        5, histogram
    )
    event = _field_varint(2, 17) + _field_bytes(5, _field_bytes(1, summary_value))
    records.append(_tfrecord(event))
    return b"".join(records)


def _wandb_history(metrics: dict[str, float]) -> bytes:
    def item(parts: list[str], value: object) -> bytes:
        payload = b"".join(_field_bytes(2, part.encode()) for part in parts)
        payload += _field_bytes(
            16, json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        )
        return _field_bytes(1, payload)

    items = [item([name], value) for name, value in sorted(metrics.items())]
    items.extend(
        (
            item([EXPECTED_HISTOGRAM_TAG, "_type"], "histogram"),
            item([EXPECTED_HISTOGRAM_TAG, "values"], [2, 4, 2]),
            item([EXPECTED_HISTOGRAM_TAG, "bins"], [-1.5, -0.5, 0.5, 1.5]),
            item(["_step"], 17),
        )
    )
    history = b"".join(items) + _field_bytes(2, _field_varint(1, 17))
    record = _field_bytes(2, history)
    kind = 1
    checksum = zlib.crc32(record, zlib.crc32(bytes([kind]))) & 0xFFFFFFFF
    header = struct.pack("<IHB", checksum, len(record), kind)
    return b":W&B\xe1\xbe\x00" + header + record


def _write_wheel(path: Path) -> dict[str, object]:
    files = {
        "tritium/__init__.py": b"from . import torch\n",
        "tritium/torch/_telemetry_binary.py": b"# binary parsers\n",
        "tritium/torch/_wheel_identity.py": b"# wheel identity\n",
        "tritium/torch/qualify_observability.py": b"# worker\n",
        "tritium/torch/observability_receipt.py": b"# validator\n",
        "tritium_torch-1.1.0rc0.dist-info/METADATA": (
            b"Metadata-Version: 2.4\nName: tritium-torch\nVersion: 1.1.0rc0\n\n"
        ),
    }
    record_name = "tritium_torch-1.1.0rc0.dist-info/RECORD"
    rows = []
    for name, payload in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
        rows.append([name, "sha256=" + digest.decode(), str(len(payload))])
    rows.append([record_name, "", ""])
    stream = io.StringIO()
    csv.writer(stream, lineterminator="\n").writerows(rows)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
        archive.writestr(record_name, stream.getvalue().encode())
    return wheel_identity(path)


def _scalar_metrics() -> dict[str, float]:
    metrics = {name: 0.25 for name in EXPECTED_SCALAR_TAGS}
    metrics.update(EXPECTED_EXTRA_METRICS)
    metrics.update(
        {
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
    )
    return metrics


def _otel() -> dict[str, object]:
    return {
        "metric_names": OTEL_METRICS,
        "data_points": 6,
        "measurements": {
            "tritium.experiment.measurement": [
                {
                    "attributes": {
                        "metric": name.removeprefix("tritium/"),
                        "source": "caller",
                    },
                    "value": value,
                }
                for name, value in sorted(EXPECTED_EXTRA_METRICS.items())
            ],
            "tritium.snapshot.code_scale_bpw": [
                {"attributes": {}, "value": 6.0}
            ],
            "tritium.snapshot.code_scale_bytes": [
                {"attributes": {}, "value": 6}
            ],
            "tritium.snapshot.tensor_count": [{"attributes": {}, "value": 1}],
        },
    }


def _receipt(root: Path) -> tuple[Path, Path, dict[str, object]]:
    wheel = root / "tritium_torch-1.1.0rc0-cp313-abi3-manylinux.whl"
    installed = _write_wheel(wheel)
    telemetry = root / "telemetry"
    (telemetry / "tensorboard").mkdir(parents=True)
    (telemetry / "wandb").mkdir()
    (telemetry / "opentelemetry").mkdir()
    metrics = _scalar_metrics()
    (telemetry / "tensorboard" / "events.out.tfevents.1").write_bytes(
        _tensorboard_events(metrics)
    )
    (telemetry / "tensorboard" / "adapter.json").write_text(
        json.dumps(
            {
                "histogram_tags": [EXPECTED_HISTOGRAM_TAG],
                "scalar_metrics": metrics,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (telemetry / "wandb" / "offline-run.wandb").write_bytes(
        _wandb_history(metrics)
    )
    wandb_evidence = {
        "mode": "offline",
        "log_calls": 1,
        "scalar_values": len(metrics),
        "histograms": 1,
    }
    (telemetry / "wandb" / "adapter.json").write_text(
        json.dumps({**wandb_evidence, "scalar_metrics": metrics}, sort_keys=True),
        encoding="utf-8",
    )
    otel = _otel()
    (telemetry / "opentelemetry" / "metrics.json").write_text(
        json.dumps(otel, sort_keys=True), encoding="utf-8"
    )
    tree = tree_identity(telemetry)
    value: dict[str, object] = {
        "schema": "tritium.installed-observability.v1",
        "passed": True,
        "source_revision": "a" * 40,
        "release": "1.1.0-rc.0",
        "run_id": "observability-run-1",
        "wheel_name": wheel.name,
        "wheel_bytes": wheel.stat().st_size,
        "wheel_sha256": "sha256:"
        + hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "distribution_version": "1.1.0rc0",
        "wheel_file_count": installed["file_count"],
        "wheel_tree_sha256": installed["tree_sha256"],
        "installed_file_count": installed["file_count"],
        "installed_tree_sha256": installed["tree_sha256"],
        "installed_record_sha256": next(
            entry["sha256"]
            for entry in installed["entries"]
            if str(entry["path"]).endswith(".dist-info/RECORD")
        ),
        "tritium_module": "/venv/tritium/__init__.py",
        "python_version": "3.13.5",
        "torch_version": "2.11.0",
        "tensorboard_version": "2.21.0",
        "wandb_version": "0.28.1",
        "opentelemetry_api_version": "1.44.0",
        "opentelemetry_sdk_version": "1.44.0",
        "environment": {
            "repository_absent": True,
            "compiler_absent": True,
            "network_mode": "offline",
        },
        "snapshot": {
            "schema_version": 1,
            "step": 17,
            "tensor_count": 1,
            "tensor_path": "left.weight",
            "aliases": ["left.weight", "right.weight"],
            "trit_counts": [2, 4, 2],
            "physical_bytes": 6,
            "code_scale_bpw": 6.0,
            "zero_rate": 0.5,
            "gradient_finite": True,
            "scalar_metric_count": len(metrics),
            "scalar_metrics": metrics,
        },
        "adapters": {
            "tensorboard": {
                "event_files": 1,
                "scalar_tags": len(metrics),
                "histogram_tags": 1,
            },
            "wandb": wandb_evidence,
            "opentelemetry": otel,
        },
        "telemetry_dir": "telemetry",
        "telemetry_bytes": tree["bytes"],
        "telemetry_file_count": tree["file_count"],
        "telemetry_tree_sha256": tree["sha256"],
    }
    value["receipt_id"] = receipt_id(value)
    path = root / "receipt.json"
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path, wheel, value


def test_observability_receipt_binds_wheel_and_retained_telemetry(tmp_path):
    path, wheel, value = _receipt(tmp_path)

    assert validate_receipt(
        path,
        expected_wheel=wheel,
        expected_source_revision="a" * 40,
        expected_release="1.1.0-rc.0",
    ) == value

    (tmp_path / "telemetry" / "wandb" / "run.wandb").write_bytes(b"changed")
    with pytest.raises(ValueError, match="telemetry tree identity"):
        validate_receipt(path, expected_wheel=wheel)


@pytest.mark.parametrize(
    ("relative", "payload", "message"),
    [
        ("tensorboard/events.out.tfevents.1", b"claimed events", "TFRecord"),
        ("wandb/offline-run.wandb", b"claimed offline run", "W&B offline"),
    ],
)
def test_observability_receipt_rejects_structural_telemetry_substitutes(
    tmp_path, relative, payload, message
):
    path, wheel, value = _receipt(tmp_path)
    (tmp_path / "telemetry" / relative).write_bytes(payload)
    tree = tree_identity(tmp_path / "telemetry")
    value["telemetry_bytes"] = tree["bytes"]
    value["telemetry_file_count"] = tree["file_count"]
    value["telemetry_tree_sha256"] = tree["sha256"]
    value["receipt_id"] = receipt_id(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_receipt(path, expected_wheel=wheel)


def test_observability_receipt_rejects_wheel_record_substitution(tmp_path):
    path, wheel, value = _receipt(tmp_path)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr("tritium/torch/qualify_observability.py", b"substituted")
    value["wheel_bytes"] = wheel.stat().st_size
    value["wheel_sha256"] = "sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest()
    value["receipt_id"] = receipt_id(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate paths|RECORD identity"):
        validate_receipt(path, expected_wheel=wheel)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["environment"].update(compiler_absent=False), "source/compiler"),
        (lambda value: value["snapshot"].update(trit_counts=[3, 3, 2]), "snapshot fixture"),
        (lambda value: value["adapters"]["wandb"].update(mode="online"), "W&B"),
        (
            lambda value: value["adapters"]["opentelemetry"].update(metric_names=[]),
            "OpenTelemetry",
        ),
    ],
)
def test_observability_receipt_rejects_substituted_evidence(tmp_path, mutation, message):
    path, wheel, original = _receipt(tmp_path)
    value = copy.deepcopy(original)
    mutation(value)
    value["receipt_id"] = receipt_id(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_receipt(path, expected_wheel=wheel)


def test_observability_receipt_rejects_substituted_runtime_version(tmp_path):
    path, wheel, original = _receipt(tmp_path)
    value = copy.deepcopy(original)
    value["wandb_version"] = "forged"
    value["receipt_id"] = receipt_id(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    expected = {
        name: str(original[name])
        for name in MODULE["RUNTIME_VERSION_FIELDS"]
    }

    with pytest.raises(ValueError, match="runtime or adapter version"):
        validate_receipt(
            path,
            expected_wheel=wheel,
            expected_runtime_versions=expected,
        )
