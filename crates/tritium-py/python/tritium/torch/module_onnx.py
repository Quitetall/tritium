"""Strict packed PyTorch-module ONNX export and runtime facade."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .. import _tritium
from ..nn import AdditiveTernaryLinear, AdditiveTernaryWeight
from .errors import TritiumError

Pathish = Union[str, os.PathLike[str]]
_MANIFEST = "tritium-module-onnx.json"
_GRAPH = "model.onnx"
_TOP_FIELDS = {
    "schema_version",
    "artifact_kind",
    "artifact_id",
    "checkpoint_digest",
    "opset",
    "input_names",
    "output_names",
    "packed_modules",
    "files",
}


@dataclass(frozen=True)
class ModuleOnnxArtifact:
    artifact_dir: Path
    artifact_id: str
    checkpoint_digest: str
    input_names: Tuple[str, ...]
    output_names: Tuple[str, ...]
    files: Tuple[Tuple[str, str, int], ...]
    schema_version: int = 1


class OnnxModule(nn.Module):
    """CPU PyTorch-shaped facade over one admitted generic ORT session."""

    def __init__(self, session: Any, artifact: ModuleOnnxArtifact) -> None:
        super().__init__()
        self._session = session
        self.artifact = artifact
        self.training = False

    def forward(self, *inputs: Tensor):
        if len(inputs) != len(self.artifact.input_names):
            raise ValueError("ONNX module input count differs from manifest")
        feed = {}
        for name, value in zip(self.artifact.input_names, inputs):
            if not isinstance(value, Tensor) or value.device.type != "cpu":
                raise TypeError("ONNX module inputs must be CPU tensors")
            feed[name] = value.detach().contiguous().numpy()
        values = tuple(
            torch.from_numpy(value)
            for value in self._session.run(list(self.artifact.output_names), feed)
        )
        return values[0] if len(values) == 1 else values


def _pairs_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate module ONNX manifest field {key!r}")
        value[key] = item
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _digest_file(path: Path) -> Tuple[str, int]:
    metadata = path.lstat()
    if path.is_symlink() or not path.is_file() or metadata.st_size <= 0:
        raise ValueError("module ONNX payload must be a nonempty ordinary file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest(), metadata.st_size


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    payload = value.removeprefix("sha256:")
    if len(payload) != 64:
        return False
    try:
        bytes.fromhex(payload)
    except ValueError:
        return False
    return True


def _dependencies():
    try:
        import onnx
        import onnxruntime
        import onnxscript  # noqa: F401 - required by the Dynamo ONNX exporter
    except ImportError as error:
        raise TritiumError(
            "generic ONNX export requires onnx, onnxscript, and onnxruntime",
            code="onnx_dependency_missing",
            stage="module_onnx",
        ) from error
    return onnx, onnxruntime


def _packed_specs(model: nn.Module):
    storage_paths = {}
    for path, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, AdditiveTernaryWeight):
            storage_paths.setdefault(id(module), path)
    specs = []
    for path, module in model.named_modules():
        if not isinstance(module, AdditiveTernaryLinear):
            continue
        storage_path = storage_paths.get(id(module.packed_weight))
        if storage_path is None:
            raise TritiumError(
                "compact module has no registered packed-weight owner",
                code="incomplete_coverage",
                stage="export_module_onnx",
                module=path,
            )
        prefix = f"{storage_path}." if storage_path else ""
        packed = [f"{prefix}packed_trits_{index}" for index in range(module.plane_count)]
        scales = [f"{prefix}scales_{index}" for index in range(module.plane_count)]
        specs.append(
            {
                "path": path,
                "storage_path": storage_path,
                "rows": module.out_features,
                "columns": module.in_features,
                "planes": module.plane_count,
                "packed_initializers": packed,
                "scale_initializers": scales,
            }
        )
    if not specs:
        raise TritiumError(
            "generic ONNX export requires compact additive ternary modules",
            code="incomplete_coverage",
            stage="export_module_onnx",
        )
    return specs


def _reachable_initializers(graph) -> set[str]:
    producers = {
        output: node for node in graph.node for output in node.output if output
    }
    pending = [output.name for output in graph.output]
    visited_values = set()
    reachable = set()
    initializer_names = {value.name for value in graph.initializer}
    while pending:
        value = pending.pop()
        if value in visited_values:
            continue
        visited_values.add(value)
        if value in initializer_names:
            reachable.add(value)
        node = producers.get(value)
        if node is not None:
            pending.extend(name for name in node.input if name)
    return reachable


def _validate_specs(specs):
    if not isinstance(specs, list) or not specs:
        raise ValueError("module ONNX packed coverage is empty")
    fields = {
        "path",
        "storage_path",
        "rows",
        "columns",
        "planes",
        "packed_initializers",
        "scale_initializers",
    }
    paths = set()
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) != fields:
            raise ValueError("module ONNX packed coverage fields differ from schema")
        if (
            not isinstance(spec["path"], str)
            or not isinstance(spec["storage_path"], str)
            or type(spec["rows"]) is not int
            or spec["rows"] <= 0
            or type(spec["columns"]) is not int
            or spec["columns"] <= 0
            or type(spec["planes"]) is not int
            or not 1 <= spec["planes"] <= 3
        ):
            raise ValueError("module ONNX packed coverage geometry is invalid")
        if spec["path"] in paths:
            raise ValueError("module ONNX packed module paths are not unique")
        paths.add(spec["path"])
        for field in ("packed_initializers", "scale_initializers"):
            names = spec[field]
            if (
                not isinstance(names, list)
                or len(names) != spec["planes"]
                or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name for name in names)
            ):
                raise ValueError("module ONNX initializer coverage is invalid")
        if set(spec["packed_initializers"]) & set(spec["scale_initializers"]):
            raise ValueError("module ONNX packed and scale initializers overlap")
    return specs


def _audit_graph(graph, specs, onnx) -> None:
    specs = _validate_specs(specs)
    initializers = {value.name: value for value in graph.initializer}
    reachable = _reachable_initializers(graph)
    required = {
        name
        for spec in specs
        for name in (*spec["packed_initializers"], *spec["scale_initializers"])
    }
    missing = sorted(required - reachable)
    if missing:
        raise TritiumError(
            "ONNX optimization removed packed ternary state",
            code="dense_shadow_detected",
            stage="export_module_onnx",
            details={"missing_initializers": missing},
        )
    for spec in specs:
        packed_elements = (spec["rows"] * spec["columns"] + 4) // 5
        for name in spec["packed_initializers"]:
            value = initializers[name]
            if value.data_type != onnx.TensorProto.UINT8 or tuple(value.dims) != (
                packed_elements,
            ):
                raise ValueError("module ONNX packed initializer geometry is invalid")
        for name in spec["scale_initializers"]:
            value = initializers[name]
            if value.data_type != onnx.TensorProto.FLOAT16 or tuple(value.dims) != (
                spec["rows"],
                1,
            ):
                raise ValueError("module ONNX scale initializer geometry is invalid")
    float_types = {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.BFLOAT16,
    }
    target_shapes = {(spec["rows"], spec["columns"]) for spec in specs}
    dense = sorted(
        name
        for name in reachable
        if name in initializers
        and initializers[name].data_type in float_types
        and tuple(initializers[name].dims) in target_shapes
    )
    if dense:
        raise TritiumError(
            "ONNX graph contains a persistent dense target weight",
            code="dense_shadow_detected",
            stage="export_module_onnx",
            details={"initializers": dense},
        )


def _tensor_outputs(value: Any) -> Tuple[Tensor, ...]:
    values = (value,) if isinstance(value, Tensor) else value
    if not isinstance(values, (tuple, list)) or not values or any(
        not isinstance(item, Tensor) for item in values
    ):
        raise TritiumError(
            "generic ONNX export requires Tensor or tuple-of-Tensor outputs",
            code="unsupported_graph",
            stage="export_module_onnx",
        )
    return tuple(values)


def export_module_onnx(
    model: nn.Module,
    example_inputs: Union[Tensor, Sequence[Tensor]],
    output_dir: Pathish,
    *,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    dynamic_batch: bool = True,
    opset: int = 18,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> ModuleOnnxArtifact:
    """Export, audit, execute, and atomically publish one packed module graph."""

    if not isinstance(model, nn.Module) or model.training:
        raise TritiumError(
            "generic ONNX export requires an eval-mode torch.nn.Module",
            code="invalid_state",
            stage="export_module_onnx",
        )
    inputs = (example_inputs,) if isinstance(example_inputs, Tensor) else tuple(example_inputs)
    if not inputs or any(
        not isinstance(value, Tensor) or value.device.type != "cpu" for value in inputs
    ):
        raise TypeError("example_inputs must contain CPU tensors")
    if type(dynamic_batch) is not bool or type(opset) is not int or opset < 18:
        raise ValueError("generic ONNX export requires bool dynamic_batch and opset >= 18")
    names_in = tuple(input_names or (f"input_{index}" for index in range(len(inputs))))
    if len(names_in) != len(inputs) or len(set(names_in)) != len(names_in):
        raise ValueError("input_names must be unique and match example_inputs")
    with torch.no_grad():
        expected = _tensor_outputs(model(*inputs))
    names_out = tuple(output_names or (f"output_{index}" for index in range(len(expected))))
    if len(names_out) != len(expected) or len(set(names_out)) != len(names_out):
        raise ValueError("output_names must be unique and match model outputs")
    specs = _packed_specs(model)
    checkpoint_digest = getattr(
        getattr(model, "config", None), "tritium_ptq_checkpoint_digest", None
    )
    if checkpoint_digest is None:
        from .ptq import _source_model_digest

        checkpoint_digest = _source_model_digest(model)
    onnx, ort = _dependencies()
    target = Path(output_dir).absolute()
    parent = target.parent.resolve(strict=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"output directory already exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=".tritium-onnx-stage-", dir=parent))
    published = False
    try:
        graph_path = staging / _GRAPH
        dynamic_shapes = None
        if dynamic_batch:
            dynamic_shapes = tuple(
                ({0: "batch"} if value.ndim > 0 else {}) for value in inputs
            )
        torch.onnx.export(
            model,
            inputs,
            graph_path,
            input_names=names_in,
            output_names=names_out,
            opset_version=opset,
            dynamo=True,
            optimize=False,
            do_constant_folding=False,
            external_data=True,
            dynamic_shapes=dynamic_shapes,
        )
        graph = onnx.load(graph_path, load_external_data=False)
        onnx.checker.check_model(graph)
        _audit_graph(graph.graph, specs, onnx)
        external_locations = {
            entry.value
            for initializer in graph.graph.initializer
            for entry in initializer.external_data
            if entry.key == "location"
        }
        for path in staging.iterdir():
            if path.stat().st_size == 0 and path.name not in external_locations:
                path.unlink()
        session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
        observed = session.run(
            list(names_out),
            {name: value.detach().contiguous().numpy() for name, value in zip(names_in, inputs)},
        )
        for actual, wanted in zip(observed, expected):
            torch.testing.assert_close(
                torch.from_numpy(actual), wanted.detach().cpu(), rtol=rtol, atol=atol
            )
        files = []
        for path in sorted(staging.iterdir()):
            if path.name == _MANIFEST:
                continue
            digest, byte_count = _digest_file(path)
            files.append({"file": path.name, "sha256": digest, "bytes": byte_count})
        manifest = {
            "schema_version": 1,
            "artifact_kind": "tritium.packed-module-onnx-v1",
            "checkpoint_digest": checkpoint_digest,
            "opset": opset,
            "input_names": list(names_in),
            "output_names": list(names_out),
            "packed_modules": specs,
            "files": files,
        }
        manifest["artifact_id"] = _digest_bytes(_canonical(manifest))
        (staging / _MANIFEST).write_bytes(_canonical(manifest))
        admitted = load_module_onnx(staging, create_session=False)
        _tritium.publish_directory_noreplace(str(staging), str(target))
        published = True
        reopened = load_module_onnx(target, create_session=False)
        if admitted.artifact_id != reopened.artifact_id:
            raise RuntimeError("published module ONNX identity changed")
        return reopened
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def load_module_onnx(
    artifact_dir: Pathish,
    *,
    create_session: bool = True,
) -> Union[ModuleOnnxArtifact, OnnxModule]:
    """Strictly verify one packed generic ONNX bundle and optionally open ORT."""

    if type(create_session) is not bool:
        raise TypeError("create_session must be a bool")
    requested = Path(artifact_dir)
    if requested.is_symlink():
        raise ValueError("module ONNX directory must not be a symlink")
    directory = requested.resolve(strict=True)
    manifest_path = directory / _MANIFEST
    metadata = manifest_path.lstat()
    if manifest_path.is_symlink() or not manifest_path.is_file() or metadata.st_size > 1024**2:
        raise ValueError("module ONNX manifest must be a bounded ordinary file")
    manifest_bytes = manifest_path.read_bytes()
    value = json.loads(
        manifest_bytes.decode("utf-8"),
        object_pairs_hook=_pairs_without_duplicates,
    )
    if not isinstance(value, dict) or set(value) != _TOP_FIELDS:
        raise ValueError("module ONNX manifest fields differ from schema")
    if manifest_bytes != _canonical(value):
        raise ValueError("module ONNX manifest is not canonical")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["artifact_kind"] != "tritium.packed-module-onnx-v1"
    ):
        raise ValueError("unsupported module ONNX artifact")
    identity = dict(value)
    artifact_id = identity.pop("artifact_id")
    if not _is_sha256(artifact_id) or artifact_id != _digest_bytes(_canonical(identity)):
        raise ValueError("module ONNX artifact identity mismatch")
    if (
        not _is_sha256(value["checkpoint_digest"])
        or type(value["opset"]) is not int
        or value["opset"] < 18
    ):
        raise ValueError("module ONNX recipe identity is invalid")
    files = value["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("module ONNX bundle has no files")
    admitted_files = []
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"file", "sha256", "bytes"}:
            raise ValueError("module ONNX file ledger differs from schema")
        name = entry["file"]
        if (
            not isinstance(name, str)
            or name != Path(name).name
            or name == _MANIFEST
            or not _is_sha256(entry["sha256"])
            or type(entry["bytes"]) is not int
            or entry["bytes"] <= 0
        ):
            raise ValueError("module ONNX filename is not canonical")
        digest, byte_count = _digest_file(directory / name)
        if digest != entry["sha256"] or byte_count != entry["bytes"]:
            raise ValueError("module ONNX file identity mismatch")
        admitted_files.append((name, digest, byte_count))
    admitted_names = [name for name, _, _ in admitted_files]
    if admitted_names != sorted(admitted_names) or len(set(admitted_names)) != len(
        admitted_names
    ):
        raise ValueError("module ONNX file ledger is not canonical")
    expected_names = {_MANIFEST, *(name for name, _, _ in admitted_files)}
    if _GRAPH not in expected_names:
        raise ValueError("module ONNX bundle omitted model.onnx")
    if {path.name for path in directory.iterdir()} != expected_names:
        raise ValueError("module ONNX directory contains unknown files")
    names_in = value["input_names"]
    names_out = value["output_names"]
    specs = value["packed_modules"]
    if (
        not isinstance(names_in, list)
        or not names_in
        or any(not isinstance(name, str) or not name for name in names_in)
        or len(set(names_in)) != len(names_in)
        or not isinstance(names_out, list)
        or not names_out
        or any(not isinstance(name, str) or not name for name in names_out)
        or len(set(names_out)) != len(names_out)
    ):
        raise ValueError("module ONNX interface or coverage is invalid")
    _validate_specs(specs)
    onnx, ort = _dependencies()
    graph_path = directory / _GRAPH
    graph = onnx.load(graph_path, load_external_data=False)
    onnx.checker.check_model(graph)
    external_locations = {
        entry.value
        for initializer in graph.graph.initializer
        for entry in initializer.external_data
        if entry.key == "location"
    }
    if any(
        location != Path(location).name
        or location in {_MANIFEST, _GRAPH}
        or location not in expected_names
        for location in external_locations
    ):
        raise ValueError("module ONNX external-data path is not admitted")
    _audit_graph(graph.graph, specs, onnx)
    if tuple(item.name for item in graph.graph.input) != tuple(names_in):
        raise ValueError("ONNX graph inputs differ from manifest")
    if tuple(item.name for item in graph.graph.output) != tuple(names_out):
        raise ValueError("ONNX graph outputs differ from manifest")
    artifact = ModuleOnnxArtifact(
        artifact_dir=directory,
        artifact_id=artifact_id,
        checkpoint_digest=value["checkpoint_digest"],
        input_names=tuple(names_in),
        output_names=tuple(names_out),
        files=tuple(admitted_files),
    )
    if not create_session:
        return artifact
    session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    if tuple(item.name for item in session.get_inputs()) != artifact.input_names:
        raise ValueError("ORT module inputs differ from manifest")
    if tuple(item.name for item in session.get_outputs()) != artifact.output_names:
        raise ValueError("ORT module outputs differ from manifest")
    return OnnxModule(session, artifact)


__all__ = [
    "ModuleOnnxArtifact",
    "OnnxModule",
    "export_module_onnx",
    "load_module_onnx",
]
