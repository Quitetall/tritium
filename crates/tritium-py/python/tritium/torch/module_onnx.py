"""Strict packed PyTorch-module ONNX export and runtime facade."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .. import _tritium
from ..nn import (
    AdditiveTernaryEmbedding,
    AdditiveTernaryLinear,
    AdditiveTernaryWeight,
    TernaryEmbedding,
    TernaryLinear,
)
from .errors import TritiumError

Pathish = Union[str, os.PathLike[str]]
_MANIFEST = "tritium-module-onnx.json"
_GRAPH = "model.onnx"
_MAX_TERMINAL_PARITY_CAPTURE_BYTES = 64 * 1024 * 1024
_PARITY_DIAGNOSTIC_LAYER_INDEX = 11
_ORT_DEFAULT_INTRA_OP_THREADS = 2
_ORT_INTRA_OP_THREADS_ENV = "TRITIUM_ONNX_INTRA_OP_THREADS"
_ORT_INTER_OP_THREADS = 0
_TOP_FIELDS_V1 = {
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
_TOP_FIELDS_V2 = _TOP_FIELDS_V1 | {"conversion"}
_CONVERSION_FIELDS = {
    "mode",
    "artifact_id",
    "recipe_id",
    "source_model_digest",
    "parent_artifact_id",
    "ancestry",
}


@dataclass(frozen=True)
class ModuleOnnxArtifact:
    artifact_dir: Path
    artifact_id: str
    checkpoint_digest: str
    input_names: Tuple[str, ...]
    output_names: Tuple[str, ...]
    files: Tuple[Tuple[str, str, int], ...]
    lineage: Optional["ModuleOnnxLineage"] = None
    schema_version: int = 1


@dataclass(frozen=True)
class ModuleOnnxLineage:
    """Typed conversion identity embedded into a packed module graph."""

    mode: str
    artifact_id: str
    recipe_id: str
    source_model_digest: str
    parent_artifact_id: Optional[str] = None
    ancestry: Tuple[str, ...] = ()


def _snapshot_lineage(value: Optional[ModuleOnnxLineage]) -> Optional[ModuleOnnxLineage]:
    if value is None:
        return None
    if not isinstance(value, ModuleOnnxLineage):
        raise TypeError("lineage must be a ModuleOnnxLineage")
    lineage = ModuleOnnxLineage(
        mode=value.mode,
        artifact_id=value.artifact_id,
        recipe_id=value.recipe_id,
        source_model_digest=value.source_model_digest,
        parent_artifact_id=value.parent_artifact_id,
        ancestry=tuple(value.ancestry),
    )
    if (
        lineage.mode not in {"qat-hard", "ptq", "scale-only", "hard-pv"}
        or not _is_sha256(lineage.artifact_id)
        or not _is_sha256(lineage.recipe_id)
        or not _is_sha256(lineage.source_model_digest)
        or any(not _is_sha256(item) for item in lineage.ancestry)
        or len(set(lineage.ancestry)) != len(lineage.ancestry)
    ):
        raise ValueError("module ONNX conversion lineage is invalid")
    if lineage.mode in {"scale-only", "hard-pv"}:
        if (
            not _is_sha256(lineage.parent_artifact_id)
            or not lineage.ancestry
            or lineage.ancestry[-1] != lineage.parent_artifact_id
        ):
            raise ValueError("refined module ONNX lineage must bind its immediate parent")
    elif lineage.parent_artifact_id is not None or lineage.ancestry:
        raise ValueError("unrefined module ONNX lineage cannot claim parents")
    return lineage


def _lineage_dict(lineage: ModuleOnnxLineage) -> dict[str, Any]:
    return {
        "mode": lineage.mode,
        "artifact_id": lineage.artifact_id,
        "recipe_id": lineage.recipe_id,
        "source_model_digest": lineage.source_model_digest,
        "parent_artifact_id": lineage.parent_artifact_id,
        "ancestry": list(lineage.ancestry),
    }


def _lineage_from_dict(value: Any) -> ModuleOnnxLineage:
    if not isinstance(value, dict) or set(value) != _CONVERSION_FIELDS:
        raise ValueError("module ONNX conversion fields differ from schema")
    ancestry = value["ancestry"]
    if not isinstance(ancestry, list):
        raise ValueError("module ONNX conversion ancestry must be a list")
    lineage = _snapshot_lineage(
        ModuleOnnxLineage(
            mode=value["mode"],
            artifact_id=value["artifact_id"],
            recipe_id=value["recipe_id"],
            source_model_digest=value["source_model_digest"],
            parent_artifact_id=value["parent_artifact_id"],
            ancestry=tuple(ancestry),
        )
    )
    assert lineage is not None
    return lineage


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


def _runtime_dependencies():
    try:
        import onnx
        import onnxruntime
    except ImportError as error:
        raise TritiumError(
            "generic ONNX bundles require onnx and onnxruntime",
            code="onnx_dependency_missing",
            stage="module_onnx",
        ) from error
    return onnx, onnxruntime


def _ort_intra_op_threads() -> int:
    """Resolve a bounded explicit ORT thread policy (0 restores ORT auto)."""

    configured = os.environ.get(_ORT_INTRA_OP_THREADS_ENV)
    if configured is None:
        return _ORT_DEFAULT_INTRA_OP_THREADS
    try:
        thread_count = int(configured, 10)
    except ValueError as error:
        raise ValueError(
            f"{_ORT_INTRA_OP_THREADS_ENV} must be an integer from 0 to 256"
        ) from error
    if not 0 <= thread_count <= 256:
        raise ValueError(
            f"{_ORT_INTRA_OP_THREADS_ENV} must be an integer from 0 to 256"
        )
    return thread_count


def _session_options(ort):
    """Keep packed decode graphs compact instead of constant-folding weights."""

    options = ort.SessionOptions()
    # ORT's default graph optimizer evaluates the standard-ONNX trit decoder
    # during session creation, materializing every full-precision target
    # matrix. That defeats packed residency and can require tens of GiB.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # ORT auto thread-pool selection changed FP32 accumulation on two exact
    # hosted artifacts. A bounded explicit default stabilizes that reduction;
    # callers can override it with TRITIUM_ONNX_INTRA_OP_THREADS (0..256).
    options.intra_op_num_threads = _ort_intra_op_threads()
    options.inter_op_num_threads = _ORT_INTER_OP_THREADS
    return options


def _cpu_model_name(cpuinfo_path: Path = Path("/proc/cpuinfo")) -> Optional[str]:
    """Read only the first CPU model label, with portable platform fallbacks."""

    try:
        with cpuinfo_path.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                key, separator, value = line.partition(":")
                if separator and key.strip().lower() in {"model name", "hardware"}:
                    model_name = value.strip()
                    if model_name:
                        return model_name[:256]
                if not line.strip():
                    break
    except OSError:
        pass
    fallback = os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor()
    return fallback[:256] if fallback else None


def _parity_runtime_info(onnx, ort, session) -> dict[str, Any]:
    """Return bounded, secret-free runtime context for opt-in failure artifacts."""

    affinity_count = None
    get_affinity = getattr(os, "sched_getaffinity", None)
    if callable(get_affinity):
        try:
            affinity_count = len(get_affinity(0))
        except OSError:
            pass
    get_providers = getattr(session, "get_providers", None)
    providers = list(get_providers()) if callable(get_providers) else []
    thread_environment = {
        name: os.environ[name]
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "OMP_PROC_BIND",
            "KMP_AFFINITY",
            _ORT_INTRA_OP_THREADS_ENV,
        )
        if name in os.environ
    }
    return {
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "onnx": str(getattr(onnx, "__version__", "unknown")),
            "onnxruntime": str(getattr(ort, "__version__", "unknown")),
        },
        "cpu": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "model_name": _cpu_model_name(),
            "logical_count": os.cpu_count(),
            "affinity_count": affinity_count,
        },
        "providers": providers,
        "session": {
            "graph_optimization_level": "ORT_DISABLE_ALL",
            "intra_op_num_threads": _ort_intra_op_threads(),
            "inter_op_num_threads": _ORT_INTER_OP_THREADS,
        },
        "thread_environment": thread_environment,
    }


def _terminal_intermediate_names(graph, output_names: Sequence[str]) -> Tuple[str, ...]:
    """Find bounded concat shards and their shared MatMul/Gemm activation."""

    producers = {
        output: node for node in graph.node for output in node.output if output
    }
    initializers = {value.name for value in graph.initializer}
    graph_inputs = {value.name for value in graph.input}
    for output_name in output_names:
        concat = producers.get(output_name)
        if concat is None or concat.op_type != "Concat":
            continue
        shards = tuple(name for name in concat.input if name)
        if not 2 <= len(shards) <= 16:
            continue
        shard_producers = tuple(producers.get(name) for name in shards)
        if any(
            node is None or node.op_type not in {"MatMul", "Gemm"}
            for node in shard_producers
        ):
            return shards
        common_inputs = None
        for node in shard_producers:
            activation_inputs = {
                name for name in node.input if name and name not in initializers
            }
            common_inputs = (
                activation_inputs
                if common_inputs is None
                else common_inputs.intersection(activation_inputs)
            )
        shared = sorted((common_inputs or set()) - graph_inputs)
        if len(shared) == 1:
            return (*shards, shared[0])
        return shards
    return ()


def _decoder_layer_residual_add_names(
    graph, hidden_size: Optional[int], layer_count: Optional[int]
) -> Tuple[str, ...]:
    """Select ordered Llama-style residual Adds when geometry is explicit."""

    if (
        type(hidden_size) is not int
        or hidden_size <= 0
        or type(layer_count) is not int
        or layer_count <= 0
    ):
        return ()
    available = {
        value.name: value
        for value in (*graph.input, *graph.output, *graph.value_info)
    }
    residual_adds = []
    for node in graph.node:
        if node.op_type != "Add" or len(node.output) != 1:
            continue
        value = available.get(node.output[0])
        if value is None or not value.type.HasField("tensor_type"):
            continue
        tensor_type = value.type.tensor_type
        if (
            tensor_type.elem_type != 1  # TensorProto.FLOAT
            or not tensor_type.HasField("shape")
            or len(tensor_type.shape.dim) != 3
            or tensor_type.shape.dim[-1].dim_value != hidden_size
        ):
            continue
        residual_adds.append(node.output[0])
    # Llama-family decoder blocks have two residual Add outputs apiece. Refuse
    # to guess layer boundaries if export structure differs from that contract.
    if len(residual_adds) != layer_count * 2:
        return ()
    return tuple(residual_adds)


def _decoder_layer_residual_names(
    graph, hidden_size: Optional[int], layer_count: Optional[int]
) -> Tuple[str, ...]:
    """Select Llama-style block outputs (the second residual Add per layer)."""

    residual_adds = _decoder_layer_residual_add_names(
        graph, hidden_size, layer_count
    )
    return tuple(residual_adds[1::2])


def _first_decoder_attention_residual_name(
    graph, hidden_size: Optional[int], layer_count: Optional[int]
) -> Optional[str]:
    """Return the first block's attention residual under the Llama Add contract."""

    residual_adds = _decoder_layer_residual_add_names(
        graph, hidden_size, layer_count
    )
    return residual_adds[0] if residual_adds else None


def _decoder_layer_attention_residual_name(
    graph, hidden_size: Optional[int], layer_count: Optional[int], layer_index: int
) -> Optional[str]:
    """Return one block's attention residual under the Llama Add contract."""

    residual_adds = _decoder_layer_residual_add_names(
        graph, hidden_size, layer_count
    )
    if (
        type(layer_index) is not int
        or layer_index < 0
        or layer_index >= (len(residual_adds) // 2)
    ):
        return None
    return residual_adds[2 * layer_index]


def _first_decoder_block_internal_names(
    graph,
    hidden_size: Optional[int],
    layer_count: Optional[int],
    intermediate_size: Optional[int] = None,
) -> Tuple[str, ...]:
    """Select hidden/MLP-width values inside block zero, excluding its residuals."""

    return _decoder_block_internal_names(
        graph, hidden_size, layer_count, intermediate_size, layer_index=0
    )


def _decoder_block_internal_names(
    graph,
    hidden_size: Optional[int],
    layer_count: Optional[int],
    intermediate_size: Optional[int],
    *,
    layer_index: int,
) -> Tuple[str, ...]:
    """Select hidden/MLP-width values inside one Llama-style block."""

    residual_adds = _decoder_layer_residual_add_names(
        graph, hidden_size, layer_count
    )
    if (
        not residual_adds
        or type(layer_index) is not int
        or layer_index < 0
        or layer_index >= (len(residual_adds) // 2)
    ):
        return ()
    first_add, block_output = residual_adds[2 * layer_index : 2 * layer_index + 2]
    node_indices = {
        output: index
        for index, node in enumerate(graph.node)
        for output in node.output
    }
    first_index = node_indices.get(first_add)
    output_index = node_indices.get(block_output)
    if first_index is None or output_index is None or output_index <= first_index:
        return ()
    available = {
        value.name: value
        for value in (*graph.input, *graph.output, *graph.value_info)
    }
    internal = []
    for node in graph.node[first_index + 1 : output_index]:
        for name in node.output:
            value = available.get(name)
            if value is None or not value.type.HasField("tensor_type"):
                continue
            tensor_type = value.type.tensor_type
            if (
                tensor_type.elem_type == 1
                and tensor_type.HasField("shape")
                and len(tensor_type.shape.dim) == 3
                and tensor_type.shape.dim[-1].dim_value
                in {hidden_size, intermediate_size}
            ):
                internal.append(name)
    return tuple(internal)


def _terminal_capture_size_bytes(
    graph,
    names: Sequence[str],
    input_names: Sequence[str],
    inputs: Sequence[Tensor],
    onnx,
) -> Optional[int]:
    """Return a conservative byte bound, or None when graph geometry is unknown."""

    import numpy as np

    available = {
        value.name: value
        for value in (*graph.input, *graph.output, *graph.value_info)
    }
    symbols = {}
    for input_name, input_value in zip(input_names, inputs, strict=True):
        info = available.get(input_name)
        if info is None or not info.type.HasField("tensor_type"):
            continue
        dimensions = info.type.tensor_type.shape.dim
        if len(dimensions) != input_value.ndim:
            continue
        for dimension, size in zip(dimensions, input_value.shape, strict=True):
            if dimension.dim_param:
                symbols[dimension.dim_param] = int(size)

    total_bytes = 0
    for name in dict.fromkeys(names):
        info = available.get(name)
        if info is None or not info.type.HasField("tensor_type"):
            return None
        tensor_type = info.type.tensor_type
        if not tensor_type.HasField("shape"):
            return None
        elements = 1
        for dimension in tensor_type.shape.dim:
            if dimension.HasField("dim_value"):
                size = dimension.dim_value
            elif dimension.dim_param in symbols:
                size = symbols[dimension.dim_param]
            else:
                return None
            if size < 0:
                return None
            elements *= size
        try:
            item_size = np.dtype(
                onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
            ).itemsize
        except (KeyError, TypeError, ValueError):
            return None
        total_bytes += elements * item_size
        if total_bytes > _MAX_TERMINAL_PARITY_CAPTURE_BYTES:
            return total_bytes
    return total_bytes


def _capture_terminal_intermediates(
    staging: Path,
    input_names: Sequence[str],
    inputs: Sequence[Tensor],
    output_names: Sequence[str],
    onnx,
    ort,
    *,
    hidden_size: Optional[int] = None,
    layer_count: Optional[int] = None,
    intermediate_size: Optional[int] = None,
) -> Tuple[Tuple[str, str, Any], ...]:
    """Replay bounded terminal ONNX values without changing the original graph."""

    graph_path = staging / _GRAPH
    graph = onnx.load(graph_path, load_external_data=False)
    candidates = _terminal_intermediate_names(graph.graph, output_names)
    residual_names = _decoder_layer_residual_names(
        graph.graph, hidden_size, layer_count
    )
    attention_residual_name = _first_decoder_attention_residual_name(
        graph.graph, hidden_size, layer_count
    )
    first_block_internal_names = _first_decoder_block_internal_names(
        graph.graph, hidden_size, layer_count, intermediate_size
    )
    diagnostic_layer_index = (
        min(_PARITY_DIAGNOSTIC_LAYER_INDEX, layer_count - 1)
        if type(layer_count) is int and layer_count > 0
        else 0
    )
    diagnostic_attention_residual_name = _decoder_layer_attention_residual_name(
        graph.graph, hidden_size, layer_count, diagnostic_layer_index
    )
    diagnostic_block_internal_names = _decoder_block_internal_names(
        graph.graph,
        hidden_size,
        layer_count,
        intermediate_size,
        layer_index=diagnostic_layer_index,
    )
    available = {
        value.name: value
        for value in (*graph.graph.input, *graph.graph.output, *graph.graph.value_info)
    }
    captured_names = tuple(
        name
        for name in dict.fromkeys(
            (
                *candidates,
                *residual_names,
                attention_residual_name,
                *first_block_internal_names,
                diagnostic_attention_residual_name,
                *diagnostic_block_internal_names,
            )
        )
        if name is not None
        if name in available
    )
    if not captured_names:
        return ()
    capture_names = (*output_names, *captured_names)
    capture_bytes = _terminal_capture_size_bytes(
        graph.graph,
        capture_names,
        input_names,
        inputs,
        onnx,
    )
    if capture_bytes is None or capture_bytes > _MAX_TERMINAL_PARITY_CAPTURE_BYTES:
        return ()
    for name in captured_names:
        graph.graph.output.add().CopyFrom(available[name])
    diagnostic_graph = staging / ".terminal-diagnostic.onnx"
    try:
        onnx.save(graph, diagnostic_graph)
        session = ort.InferenceSession(
            str(diagnostic_graph),
            sess_options=_session_options(ort),
            providers=["CPUExecutionProvider"],
        )
        feed = {
            name: value.detach().contiguous().numpy()
            for name, value in zip(input_names, inputs, strict=True)
        }
        values = session.run(list(capture_names), feed)
        output_values = values[: len(output_names)]
        intermediate_values = values[len(output_names) :]
        return (
            *(
                ("terminal-output-replay", name, value)
                for name, value in zip(output_names, output_values, strict=True)
            ),
            *(
                (
                    (
                        "terminal-attention-residual"
                        if name == attention_residual_name
                        or name == diagnostic_attention_residual_name
                        else "terminal-layer-residual"
                        if name in residual_names
                        else "terminal-first-block-internal"
                        if name in first_block_internal_names
                        else "terminal-diagnostic-block-internal"
                        if name in diagnostic_block_internal_names
                        else "terminal-intermediate"
                    ),
                    name,
                    value,
                )
                for name, value in zip(
                    captured_names, intermediate_values, strict=True
                )
            ),
        )
    finally:
        diagnostic_graph.unlink(missing_ok=True)


def _capture_reference_terminal_outputs(
    model: nn.Module,
    input_names: Sequence[str],
    inputs: Sequence[Tensor],
) -> Tuple[Tuple[str, str, Tensor], ...]:
    """Best-effort capture of a HF model's final hidden state on the reference path."""

    config = getattr(model, "config", None)
    config = getattr(config, "text_config", config)
    layer_count = getattr(config, "num_hidden_layers", None)
    hidden_size = getattr(config, "hidden_size", None)
    intermediate_size = getattr(config, "intermediate_size", None)
    vocab_size = getattr(config, "vocab_size", None)
    if any(type(value) is not int or value <= 0 for value in (layer_count, hidden_size, vocab_size)):
        return ()
    token_input = next(
        (
            value
            for name, value in zip(input_names, inputs, strict=True)
            if "input_ids" in name
            and value.ndim >= 2
            and value.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}
        ),
        None,
    )
    if token_input is None:
        return ()
    batch = 1
    for size in token_input.shape[:-1]:
        batch *= int(size)
    sequence = int(token_input.shape[-1])
    backbone = getattr(model, "model", None)
    decoder_layers = getattr(backbone, "layers", None)
    first_layer = decoder_layers[0] if decoder_layers else None
    diagnostic_layer_index = min(_PARITY_DIAGNOSTIC_LAYER_INDEX, layer_count - 1)
    diagnostic_layer = (
        decoder_layers[diagnostic_layer_index]
        if decoder_layers and diagnostic_layer_index < len(decoder_layers)
        else None
    )
    post_attention_layernorm = getattr(
        first_layer, "post_attention_layernorm", None
    )
    first_mlp = getattr(first_layer, "mlp", None)
    gate_projection = getattr(first_mlp, "gate_proj", None)
    up_projection = getattr(first_mlp, "up_proj", None)
    mlp_activation = getattr(first_mlp, "act_fn", None)
    can_capture_attention_residual = isinstance(post_attention_layernorm, nn.Module)
    can_capture_mlp_input = can_capture_attention_residual
    can_capture_mlp_output = isinstance(first_mlp, nn.Module)
    can_capture_mlp_projections = (
        type(intermediate_size) is int
        and intermediate_size > 0
        and isinstance(gate_projection, nn.Module)
        and isinstance(up_projection, nn.Module)
        and isinstance(mlp_activation, nn.Module)
    )
    diagnostic_post_attention_layernorm = getattr(
        diagnostic_layer, "post_attention_layernorm", None
    )
    diagnostic_mlp = getattr(diagnostic_layer, "mlp", None)
    diagnostic_gate_projection = getattr(diagnostic_mlp, "gate_proj", None)
    diagnostic_up_projection = getattr(diagnostic_mlp, "up_proj", None)
    diagnostic_mlp_activation = getattr(diagnostic_mlp, "act_fn", None)
    capture_diagnostic_block = diagnostic_layer_index != 0
    can_capture_diagnostic_boundaries = (
        capture_diagnostic_block
        and isinstance(diagnostic_post_attention_layernorm, nn.Module)
        and isinstance(diagnostic_mlp, nn.Module)
    )
    can_capture_diagnostic_projections = (
        can_capture_diagnostic_boundaries
        and type(intermediate_size) is int
        and intermediate_size > 0
        and isinstance(diagnostic_gate_projection, nn.Module)
        and isinstance(diagnostic_up_projection, nn.Module)
        and isinstance(diagnostic_mlp_activation, nn.Module)
    )
    estimated_bytes = (
        (
            layer_count
            + 1
            + int(can_capture_attention_residual)
            + int(can_capture_mlp_input)
            + int(can_capture_mlp_output)
            + int(can_capture_diagnostic_boundaries) * 3
        )
        * batch
        * sequence
        * hidden_size
        * 4
        + 3
        * (
            int(can_capture_mlp_projections)
            + int(can_capture_diagnostic_projections)
        )
        * batch
        * sequence
        * (
            intermediate_size
            if type(intermediate_size) is int and intermediate_size > 0
            else 0
        )
        * 4
        + batch * sequence * vocab_size * 4
    )
    if estimated_bytes > _MAX_TERMINAL_PARITY_CAPTURE_BYTES:
        return ()
    attention_residuals = []
    mlp_inputs = []
    mlp_outputs = []
    gate_projection_outputs = []
    up_projection_outputs = []
    mlp_activation_outputs = []
    diagnostic_attention_residuals = []
    diagnostic_mlp_inputs = []
    diagnostic_mlp_outputs = []
    diagnostic_gate_projection_outputs = []
    diagnostic_up_projection_outputs = []
    diagnostic_mlp_activation_outputs = []

    def capture_attention_residual(target, _module, args):
        if args and isinstance(args[0], Tensor):
            target.append(args[0])

    def capture_module_output(target, _module, _args, output):
        value = output[0] if isinstance(output, (tuple, list)) and output else output
        if isinstance(value, Tensor):
            target.append(value)

    hooks = []
    if can_capture_attention_residual:
        hooks.append(
            post_attention_layernorm.register_forward_pre_hook(
                lambda module, args: capture_attention_residual(
                    attention_residuals, module, args
                )
            )
        )
        hooks.append(
            post_attention_layernorm.register_forward_hook(
                lambda module, args, output: capture_module_output(
                    mlp_inputs, module, args, output
                )
            )
        )
    if can_capture_mlp_output:
        hooks.append(
            first_mlp.register_forward_hook(
                lambda module, args, output: capture_module_output(
                    mlp_outputs, module, args, output
                )
            )
        )
    if can_capture_mlp_projections:
        hooks.extend(
            (
                gate_projection.register_forward_hook(
                    lambda module, args, output: capture_module_output(
                        gate_projection_outputs, module, args, output
                    )
                ),
                up_projection.register_forward_hook(
                    lambda module, args, output: capture_module_output(
                        up_projection_outputs, module, args, output
                    )
                ),
                mlp_activation.register_forward_hook(
                    lambda module, args, output: capture_module_output(
                        mlp_activation_outputs, module, args, output
                    )
                ),
            )
        )
    if can_capture_diagnostic_boundaries:
        hooks.extend(
            (
                diagnostic_post_attention_layernorm.register_forward_pre_hook(
                    lambda module, args: capture_attention_residual(
                        diagnostic_attention_residuals, module, args
                    )
                ),
                diagnostic_post_attention_layernorm.register_forward_hook(
                    lambda module, args, output: capture_module_output(
                        diagnostic_mlp_inputs, module, args, output
                    )
                ),
                diagnostic_mlp.register_forward_hook(
                    lambda module, args, output: capture_module_output(
                        diagnostic_mlp_outputs, module, args, output
                    )
                ),
            )
        )
    if can_capture_diagnostic_projections:
        hooks.extend(
            (
                diagnostic_gate_projection.register_forward_hook(
                    lambda module, args, output: capture_module_output(
                        diagnostic_gate_projection_outputs, module, args, output
                    )
                ),
                diagnostic_up_projection.register_forward_hook(
                    lambda module, args, output: capture_module_output(
                        diagnostic_up_projection_outputs, module, args, output
                    )
                ),
                diagnostic_mlp_activation.register_forward_hook(
                    lambda module, args, output: capture_module_output(
                        diagnostic_mlp_activation_outputs, module, args, output
                    )
                ),
            )
        )
    try:
        with torch.no_grad():
            result = model(*inputs, output_hidden_states=True, use_cache=False)
    except TypeError:
        return ()
    finally:
        for hook in hooks:
            hook.remove()
    if isinstance(result, Mapping):
        hidden_states = result.get("hidden_states")
        logits = result.get("logits")
    else:
        hidden_states = getattr(result, "hidden_states", None)
        logits = getattr(result, "logits", None)
    if (
        not isinstance(hidden_states, (tuple, list))
        or len(hidden_states) not in {layer_count, layer_count + 1}
        or any(not isinstance(value, Tensor) for value in hidden_states)
    ):
        return ()
    capture_tensors = list(hidden_states)
    if isinstance(logits, Tensor):
        capture_tensors.append(logits)
    if attention_residuals:
        capture_tensors.append(attention_residuals[0])
    if mlp_inputs:
        capture_tensors.append(mlp_inputs[0])
    if mlp_outputs:
        capture_tensors.append(mlp_outputs[0])
    if gate_projection_outputs:
        capture_tensors.append(gate_projection_outputs[0])
    if up_projection_outputs:
        capture_tensors.append(up_projection_outputs[0])
    if mlp_activation_outputs:
        capture_tensors.append(mlp_activation_outputs[0])
    for values in (
        diagnostic_attention_residuals,
        diagnostic_mlp_inputs,
        diagnostic_mlp_outputs,
        diagnostic_gate_projection_outputs,
        diagnostic_up_projection_outputs,
        diagnostic_mlp_activation_outputs,
    ):
        if values:
            capture_tensors.append(values[0])
    if sum(value.numel() * value.element_size() for value in capture_tensors) > (
        _MAX_TERMINAL_PARITY_CAPTURE_BYTES
    ):
        return ()
    arrays = []
    if isinstance(logits, Tensor):
        arrays.append(
            ("reference-output-replay", "logits", logits.detach().cpu().contiguous())
        )
    if attention_residuals:
        arrays.append(
            (
                "reference-attention-residual",
                "layers[0].attention_residual",
                attention_residuals[0].detach().cpu().contiguous(),
            )
        )
    if mlp_inputs:
        arrays.append(
            (
                "reference-mlp-input",
                "layers[0].post_attention_layernorm.output",
                mlp_inputs[0].detach().cpu().contiguous(),
            )
        )
    if gate_projection_outputs:
        arrays.append(
            (
                "reference-mlp-gate-projection",
                "layers[0].mlp.gate_proj.output",
                gate_projection_outputs[0].detach().cpu().contiguous(),
            )
        )
    if up_projection_outputs:
        arrays.append(
            (
                "reference-mlp-up-projection",
                "layers[0].mlp.up_proj.output",
                up_projection_outputs[0].detach().cpu().contiguous(),
            )
        )
    if mlp_activation_outputs:
        arrays.append(
            (
                "reference-mlp-activation",
                "layers[0].mlp.act_fn.output",
                mlp_activation_outputs[0].detach().cpu().contiguous(),
            )
        )
    if mlp_outputs:
        arrays.append(
            (
                "reference-mlp-output",
                "layers[0].mlp.output",
                mlp_outputs[0].detach().cpu().contiguous(),
            )
        )
    if diagnostic_attention_residuals:
        arrays.append(
            (
                "reference-attention-residual",
                f"layers[{diagnostic_layer_index}].attention_residual",
                diagnostic_attention_residuals[0].detach().cpu().contiguous(),
            )
        )
    if diagnostic_mlp_inputs:
        arrays.append(
            (
                "reference-mlp-input",
                f"layers[{diagnostic_layer_index}].post_attention_layernorm.output",
                diagnostic_mlp_inputs[0].detach().cpu().contiguous(),
            )
        )
    if diagnostic_gate_projection_outputs:
        arrays.append(
            (
                "reference-mlp-gate-projection",
                f"layers[{diagnostic_layer_index}].mlp.gate_proj.output",
                diagnostic_gate_projection_outputs[0].detach().cpu().contiguous(),
            )
        )
    if diagnostic_up_projection_outputs:
        arrays.append(
            (
                "reference-mlp-up-projection",
                f"layers[{diagnostic_layer_index}].mlp.up_proj.output",
                diagnostic_up_projection_outputs[0].detach().cpu().contiguous(),
            )
        )
    if diagnostic_mlp_activation_outputs:
        arrays.append(
            (
                "reference-mlp-activation",
                f"layers[{diagnostic_layer_index}].mlp.act_fn.output",
                diagnostic_mlp_activation_outputs[0].detach().cpu().contiguous(),
            )
        )
    if diagnostic_mlp_outputs:
        arrays.append(
            (
                "reference-mlp-output",
                f"layers[{diagnostic_layer_index}].mlp.output",
                diagnostic_mlp_outputs[0].detach().cpu().contiguous(),
            )
        )
    arrays.extend(
        (
            "reference-terminal-hidden"
            if index == len(hidden_states) - 1
            else "reference-hidden-state",
            "hidden_states[-1]"
            if index == len(hidden_states) - 1
            else f"hidden_states[{index}]",
            value.detach().cpu().contiguous(),
        )
        for index, value in enumerate(hidden_states)
    )
    return tuple(arrays)


def _retain_parity_failure(
    staging: Path,
    diagnostic_root: Path,
    artifact_name: str,
    checkpoint_digest: str,
    rtol: float,
    atol: float,
    error: AssertionError,
    input_names: Sequence[str],
    inputs: Sequence[Tensor],
    observed: Sequence[Any],
    expected: Sequence[Tensor],
    runtime: Mapping[str, Any],
    additional_arrays: Sequence[Tuple[str, str, Any]] = (),
) -> None:
    """Retain a digest-ledgered graph and replay tensors when opted in."""

    diagnostic_root.mkdir(parents=True, exist_ok=True)
    destination = diagnostic_root / artifact_name
    destination.mkdir(exist_ok=False)
    try:
        files = []
        for source in sorted(staging.iterdir()):
            if source.is_symlink() or not source.is_file():
                raise ValueError("parity diagnostic staging contains a non-file")
            target = destination / source.name
            shutil.copy2(source, target)
            digest, byte_count = _digest_file(target)
            files.append(
                {"file": target.name, "sha256": digest, "bytes": byte_count}
            )
        arrays = []
        diagnostic_values = [
            ("input", name, value)
            for name, value in zip(input_names, inputs)
        ]
        diagnostic_values.extend(
            ("expected-output", str(index), value)
            for index, value in enumerate(expected)
        )
        diagnostic_values.extend(
            ("observed-output", str(index), value)
            for index, value in enumerate(observed)
        )
        diagnostic_values.extend(additional_arrays)
        for index, (role, name, value) in enumerate(diagnostic_values):
            if isinstance(value, Tensor):
                tensor = value.detach().cpu().contiguous()
                payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                dtype = str(tensor.dtype)
                shape = list(tensor.shape)
            else:
                payload = value.tobytes(order="C")
                dtype = str(value.dtype)
                shape = list(value.shape)
            filename = f"replay-{index:03d}.bin"
            array_path = destination / filename
            array_path.write_bytes(payload)
            digest = "sha256:" + hashlib.sha256(payload).hexdigest()
            arrays.append(
                {
                    "role": role,
                    "name": name,
                    "file": filename,
                    "dtype": dtype,
                    "shape": shape,
                    "sha256": digest,
                    "bytes": len(payload),
                }
            )
        diagnostic = {
            "schema_version": 1,
            "artifact_kind": "tritium.onnx-parity-failure-diagnostic.v1",
            "checkpoint_digest": checkpoint_digest,
            "rtol": rtol,
            "atol": atol,
            "failure_type": type(error).__name__,
            "runtime": dict(runtime),
            "files": files,
            "replay_arrays": arrays,
        }
        (destination / "diagnostic.json").write_bytes(_canonical(diagnostic))
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _export_dependencies():
    dependencies = _runtime_dependencies()
    try:
        import onnxscript  # noqa: F401 - required by the Dynamo ONNX exporter
    except ImportError as error:
        raise TritiumError(
            "generic Dynamo ONNX export additionally requires onnxscript",
            code="onnx_dependency_missing",
            stage="module_onnx",
        ) from error
    return dependencies


def _translate_packed_ternary_plane(
    packed,
    scales,
    rows: int,
    columns: int,
    group_size: int,
    dtype_code: int,
):
    """Translate opaque packed decode to a compact standard-ONNX subgraph."""

    from onnxscript import opset18 as op

    packed = op.Cast(packed, to=7)  # int64
    packed = op.Unsqueeze(packed, op.Constant(value_ints=[1]))
    digits = []
    for position in range(5):
        quotient = op.Div(packed, op.Constant(value_int=3**position))
        digits.append(
            op.Mod(quotient, op.Constant(value_int=3), fmod=0)
        )
    decoded = op.Concat(*digits, axis=1)
    decoded = op.Reshape(decoded, op.Constant(value_ints=[-1]))
    decoded = op.Slice(
        decoded,
        op.Constant(value_ints=[0]),
        op.Constant(value_ints=[rows * columns]),
        op.Constant(value_ints=[0]),
    )
    decoded = op.Sub(decoded, op.Constant(value_int=1))
    decoded = op.Reshape(decoded, op.Constant(value_ints=[rows, columns]))
    decoded = op.Cast(decoded, to=dtype_code)
    scales = op.Cast(scales, to=dtype_code)
    columns_index = op.Range(
        op.Constant(value_int=0),
        op.Constant(value_int=columns),
        op.Constant(value_int=1),
    )
    group_index = op.Div(columns_index, op.Constant(value_int=group_size))
    expanded_scales = op.Gather(scales, group_index, axis=1)
    return op.Mul(decoded, expanded_scales)


def _packed_specs(model: nn.Module):
    storage_paths = {}
    for path, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, AdditiveTernaryWeight):
            storage_paths.setdefault(id(module), path)
    specs = []
    for path, module in model.named_modules():
        if not isinstance(module, (AdditiveTernaryLinear, AdditiveTernaryEmbedding)):
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
        packed_weight = module.packed_weight
        packed = [
            f"{prefix}packed_trits_{index}"
            for index in range(packed_weight.plane_count)
        ]
        scales = [
            f"{prefix}scales_{index}" for index in range(packed_weight.plane_count)
        ]
        specs.append(
            {
                "path": path,
                "storage_path": storage_path,
                "rows": packed_weight.out_features,
                "columns": packed_weight.in_features,
                "planes": packed_weight.plane_count,
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
            if (
                value.data_type != onnx.TensorProto.FLOAT16
                or len(value.dims) != 2
                or value.dims[0] != spec["rows"]
                or not 1 <= value.dims[1] <= spec["columns"]
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
    if isinstance(value, Tensor):
        values = (value,)
    elif callable(getattr(value, "to_tuple", None)):
        # Hugging Face ModelOutput exposes only its populated fields this way.
        values = value.to_tuple()
    else:
        values = value
    if not isinstance(values, (tuple, list)) or not values or any(
        not isinstance(item, Tensor) for item in values
    ):
        raise TritiumError(
            "generic ONNX export requires flat Tensor outputs",
            code="unsupported_graph",
            stage="export_module_onnx",
        )
    return tuple(values)


def _promote_float32_matmuls_to_fp64(graph, onnx) -> int:
    """Accumulate every typed float32 MatMul/Gemm in FP64, then restore FP32."""

    value_types = {}
    for value in (*graph.input, *graph.value_info, *graph.output):
        if value.type.WhichOneof("value") == "tensor_type":
            value_types[value.name] = value.type.tensor_type.elem_type
    value_types.update(
        (initializer.name, initializer.data_type)
        for initializer in graph.initializer
    )
    promoted_outputs: set[str] = set()
    for node in graph.node:
        if (
            node.op_type not in {"MatMul", "Gemm"}
            or len(node.output) != 1
            or not node.output[0]
        ):
            continue
        output_type = value_types.get(node.output[0])
        input_types = [value_types.get(name) for name in node.input if name]
        if output_type is None and input_types and all(
            value_type == onnx.TensorProto.FLOAT for value_type in input_types
        ):
            output_type = onnx.TensorProto.FLOAT
        if output_type == onnx.TensorProto.FLOAT and all(
            value_type == onnx.TensorProto.FLOAT for value_type in input_types
        ):
            promoted_outputs.add(node.output[0])
    if not promoted_outputs:
        return 0

    used_names = {
        name
        for node in graph.node
        for name in (*node.input, *node.output, node.name)
        if name
    }
    used_names.update(
        value.name
        for value in (*graph.input, *graph.output, *graph.value_info, *graph.initializer)
    )

    def fresh_name(stem: str) -> str:
        candidate = f"{stem}__tritium_fp64"
        ordinal = 0
        while candidate in used_names:
            ordinal += 1
            candidate = f"{stem}__tritium_fp64_{ordinal}"
        used_names.add(candidate)
        return candidate

    rewritten = []
    promoted = 0
    for node in graph.node:
        if not node.output or node.output[0] not in promoted_outputs:
            rewritten.append(node)
            continue
        original_output = node.output[0]
        double_inputs = []
        casts = []
        for index, input_name in enumerate(node.input):
            if not input_name:
                double_inputs.append(input_name)
                continue
            double_input = fresh_name(f"{original_output}_input_{index}")
            casts.append(
                onnx.helper.make_node(
                    "Cast",
                    [input_name],
                    [double_input],
                    name=fresh_name(f"{original_output}_cast_in_{index}"),
                    to=onnx.TensorProto.DOUBLE,
                )
            )
            double_inputs.append(double_input)
        double_output = fresh_name(original_output)
        double_matmul = onnx.NodeProto()
        double_matmul.CopyFrom(node)
        del double_matmul.input[:]
        double_matmul.input.extend(double_inputs)
        del double_matmul.output[:]
        double_matmul.output.append(double_output)
        rewritten.extend(casts)
        rewritten.append(double_matmul)
        rewritten.append(
            onnx.helper.make_node(
                "Cast",
                [double_output],
                [original_output],
                name=fresh_name(f"{original_output}_cast_out"),
                to=onnx.TensorProto.FLOAT,
            )
        )
        promoted += 1
    del graph.node[:]
    graph.node.extend(rewritten)
    return promoted


def export_module_onnx(
    model: nn.Module,
    example_inputs: Union[Tensor, Sequence[Tensor]],
    output_dir: Pathish,
    *,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    dynamic_batch: bool = True,
    dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
    opset: int = 18,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    lineage: Optional[ModuleOnnxLineage] = None,
) -> ModuleOnnxArtifact:
    """Export, audit, execute, and atomically publish one packed module graph."""

    lineage = _snapshot_lineage(lineage)
    if not isinstance(model, nn.Module):
        raise TritiumError(
            "generic ONNX export requires an eval-mode torch.nn.Module",
            code="invalid_state",
            stage="export_module_onnx",
        )
    if any(
        isinstance(module, (TernaryLinear, TernaryEmbedding))
        for module in model.modules()
    ):
        raise TritiumError(
            "trainable ONNX export requires Tritium v1.3",
            code="trainable_onnx_requires_v1_3",
            stage="export_module_onnx",
        )
    if model.training:
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
    if (
        len(names_in) != len(inputs)
        or len(set(names_in)) != len(names_in)
        or any(not isinstance(name, str) or not name for name in names_in)
    ):
        raise ValueError("input_names must be unique and match example_inputs")
    if dynamic_axes is not None and not isinstance(dynamic_axes, Mapping):
        raise TypeError("dynamic_axes must map input names to axis-name mappings")
    shapes = [
        ({0: "batch"} if dynamic_batch and value.ndim > 0 else {})
        for value in inputs
    ]
    for name, axes in (dynamic_axes or {}).items():
        if name not in names_in or not isinstance(axes, Mapping) or not axes:
            raise ValueError("dynamic_axes contains an unknown input or empty mapping")
        input_index = names_in.index(name)
        for axis, dimension in axes.items():
            if (
                type(axis) is not int
                or not 0 <= axis < inputs[input_index].ndim
                or not isinstance(dimension, str)
                or not dimension.isidentifier()
            ):
                raise ValueError(
                    "dynamic_axes contains an invalid axis or dimension name"
                )
            prior = shapes[input_index].get(axis)
            if prior is not None and prior != dimension:
                raise ValueError("dynamic_axes conflicts with dynamic_batch")
            shapes[input_index][axis] = dimension
    with torch.no_grad():
        expected = _tensor_outputs(model(*inputs))
    names_out = tuple(output_names or (f"output_{index}" for index in range(len(expected))))
    if (
        len(names_out) != len(expected)
        or len(set(names_out)) != len(names_out)
        or any(not isinstance(name, str) or not name for name in names_out)
    ):
        raise ValueError("output_names must be unique and match model outputs")
    specs = _packed_specs(model)
    checkpoint_digest = (
        lineage.source_model_digest
        if lineage is not None
        else getattr(
            getattr(model, "config", None), "tritium_ptq_checkpoint_digest", None
        )
    )
    if checkpoint_digest is None:
        from .ptq import _source_model_digest

        checkpoint_digest = _source_model_digest(model)
    onnx, ort = _export_dependencies()
    target = Path(output_dir).absolute()
    parent = target.parent.resolve(strict=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"output directory already exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=".tritium-onnx-stage-", dir=parent))
    published = False
    try:
        graph_path = staging / _GRAPH
        dynamic_shapes = tuple(shapes) if any(shapes) else None
        torch.onnx.export(
            model,
            inputs,
            graph_path,
            input_names=names_in,
            output_names=names_out,
            opset_version=opset,
            dynamo=True,
            # On PyTorch 2.11, optimize=True removes required packed ternary
            # initializers; the strict graph audit must continue to see them.
            optimize=False,
            do_constant_folding=False,
            external_data=True,
            dynamic_shapes=dynamic_shapes,
            custom_translation_table={
                torch.ops.tritium.decode_packed_ternary_plane.default:
                    _translate_packed_ternary_plane,
            },
        )
        graph = onnx.load(graph_path, load_external_data=False)
        if _promote_float32_matmuls_to_fp64(graph.graph, onnx):
            onnx.save_model(graph, graph_path, save_as_external_data=False)
        # Path-based checking supplies ONNX with the external-data base directory.
        # Checking an in-memory ModelProto makes valid large graphs look missing.
        onnx.checker.check_model(str(graph_path))
        graph = onnx.load(graph_path, load_external_data=False)
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
        session = ort.InferenceSession(
            str(graph_path),
            sess_options=_session_options(ort),
            providers=["CPUExecutionProvider"],
        )
        observed = session.run(
            list(names_out),
            {name: value.detach().contiguous().numpy() for name, value in zip(names_in, inputs)},
        )
        for actual, wanted in zip(observed, expected):
            try:
                torch.testing.assert_close(
                    torch.from_numpy(actual), wanted.detach().cpu(),
                    rtol=rtol, atol=atol,
                )
            except AssertionError as error:
                diagnostic_root = os.environ.get(
                    "TRITIUM_ONNX_PARITY_FAILURE_DIR"
                )
                if diagnostic_root:
                    additional_arrays = ()
                    if os.environ.get("TRITIUM_ONNX_PARITY_CAPTURE_TERMINAL") == "1":
                        try:
                            reference_config = getattr(model, "config", None)
                            reference_config = getattr(
                                reference_config, "text_config", reference_config
                            )
                            additional_arrays = _capture_terminal_intermediates(
                                staging,
                                names_in,
                                inputs,
                                names_out,
                                onnx,
                                ort,
                                hidden_size=getattr(
                                    reference_config, "hidden_size", None
                                ),
                                layer_count=getattr(
                                    reference_config, "num_hidden_layers", None
                                ),
                                intermediate_size=getattr(
                                    reference_config, "intermediate_size", None
                                ),
                            )
                        except Exception as diagnostic_error:
                            print(
                                "could not capture terminal ONNX intermediates: "
                                f"{type(diagnostic_error).__name__}",
                                file=sys.stderr,
                            )
                        try:
                            additional_arrays += _capture_reference_terminal_outputs(
                                model,
                                names_in,
                                inputs,
                            )
                        except Exception as diagnostic_error:
                            print(
                                "could not capture reference terminal outputs: "
                                f"{type(diagnostic_error).__name__}",
                                file=sys.stderr,
                            )
                    try:
                        _retain_parity_failure(
                            staging,
                            Path(diagnostic_root),
                            target.name,
                            checkpoint_digest,
                            rtol,
                            atol,
                            error,
                            names_in,
                            inputs,
                            observed,
                            expected,
                            _parity_runtime_info(onnx, ort, session),
                            additional_arrays,
                        )
                    except Exception as diagnostic_error:
                        print(
                            "could not retain ONNX parity diagnostic: "
                            f"{type(diagnostic_error).__name__}",
                            file=sys.stderr,
                        )
                raise
        files = []
        for path in sorted(staging.iterdir()):
            if path.name == _MANIFEST:
                continue
            digest, byte_count = _digest_file(path)
            files.append({"file": path.name, "sha256": digest, "bytes": byte_count})
        schema_version = 2 if lineage is not None else 1
        manifest = {
            "schema_version": schema_version,
            "artifact_kind": f"tritium.packed-module-onnx-v{schema_version}",
            "checkpoint_digest": checkpoint_digest,
            "opset": opset,
            "input_names": list(names_in),
            "output_names": list(names_out),
            "packed_modules": specs,
            "files": files,
        }
        if lineage is not None:
            manifest["conversion"] = _lineage_dict(lineage)
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
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
        raise ValueError("module ONNX manifest must declare an integer schema version")
    schema_version = value["schema_version"]
    expected_fields = {1: _TOP_FIELDS_V1, 2: _TOP_FIELDS_V2}.get(schema_version)
    if expected_fields is None or set(value) != expected_fields:
        raise ValueError("module ONNX manifest fields differ from schema")
    if manifest_bytes != _canonical(value):
        raise ValueError("module ONNX manifest is not canonical")
    if (
        value["artifact_kind"] != f"tritium.packed-module-onnx-v{schema_version}"
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
    lineage = (
        _lineage_from_dict(value["conversion"])
        if schema_version == 2
        else None
    )
    if lineage is not None and lineage.source_model_digest != value["checkpoint_digest"]:
        raise ValueError("module ONNX source identity differs from conversion lineage")
    onnx, ort = _runtime_dependencies()
    graph_path = directory / _GRAPH
    onnx.checker.check_model(str(graph_path))
    graph = onnx.load(graph_path, load_external_data=False)
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
        lineage=lineage,
        schema_version=schema_version,
    )
    if not create_session:
        return artifact
    session = ort.InferenceSession(
        str(graph_path),
        sess_options=_session_options(ort),
        providers=["CPUExecutionProvider"],
    )
    if tuple(item.name for item in session.get_inputs()) != artifact.input_names:
        raise ValueError("ORT module inputs differ from manifest")
    if tuple(item.name for item in session.get_outputs()) != artifact.output_names:
        raise ValueError("ORT module outputs differ from manifest")
    return OnnxModule(session, artifact)


__all__ = [
    "ModuleOnnxArtifact",
    "ModuleOnnxLineage",
    "OnnxModule",
    "export_module_onnx",
    "load_module_onnx",
]
