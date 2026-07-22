"""Strict durable artifacts for separately typed QAT-hard module state."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn

from .. import _tritium
from ..nn import (
    AdditiveTernaryEmbedding,
    AdditiveTernaryLinear,
    AdditiveTernaryWeight,
    TernaryEmbedding,
    TernaryLinear,
)
from .config import TernaryConfig
from .coverage import CoverageReport
from .errors import TritiumError
from .qat import QatHardResult, QatHardWeight, _canonical, _qat_hard_ids

Pathish = Union[str, os.PathLike[str]]
_MANIFEST = "tritium-qat-hard.json"
_STATE = "model.safetensors"
_TOP_FIELDS = {
    "schema_version",
    "artifact_kind",
    "artifact_id",
    "conversion_artifact_id",
    "source_checkpoint_digest",
    "hard_state_digest",
    "recipe_id",
    "config",
    "source_coverage",
    "weights",
    "state",
}
_WEIGHT_FIELDS = {
    "path",
    "aliases",
    "consumer_kinds",
    "storage_path",
    "shape",
    "algorithm_id",
    "planes",
}
_STATE_FIELDS = {"file", "sha256", "bytes", "tensors"}
_TENSOR_FIELDS = {"name", "dtype", "shape"}
_SAFE_TO_TORCH_DTYPE = {
    "BOOL": "torch.bool",
    "U8": "torch.uint8",
    "I8": "torch.int8",
    "I16": "torch.int16",
    "I32": "torch.int32",
    "I64": "torch.int64",
    "F16": "torch.float16",
    "BF16": "torch.bfloat16",
    "F32": "torch.float32",
    "F64": "torch.float64",
}
_COVERAGE_FIELDS = {
    "path",
    "aliases",
    "disposition",
    "reason",
    "numel",
    "logical_bytes",
}


@dataclass(frozen=True)
class QatHardArtifact:
    """Verified immutable QAT-hard state bundle."""

    artifact_dir: Path
    artifact_id: str
    conversion_artifact_id: str
    source_checkpoint_digest: str
    hard_state_digest: str
    recipe_id: str
    config: TernaryConfig
    source_coverage: CoverageReport
    weights: Tuple[QatHardWeight, ...]
    state_digest: str
    state_bytes: int
    state_tensors: int
    state_ledger: Tuple[Tuple[str, str, Tuple[int, ...]], ...]
    mode: str = "qat-hard"
    schema_version: int = 1


def _dependencies():
    try:
        from safetensors import safe_open
        from safetensors.torch import load_model, save_model
    except ImportError as error:
        raise TritiumError(
            "QAT-hard artifacts require safetensors",
            code="safetensors_dependency_missing",
            stage="qat_hard_artifact",
        ) from error
    return safe_open, load_model, save_model


def _pairs_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate QAT-hard manifest field {key!r}")
        value[key] = item
    return value


def _digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _validate_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError(f"invalid QAT-hard {name}")
    encoded = value.removeprefix("sha256:")
    if len(encoded) != 64:
        raise ValueError(f"invalid QAT-hard {name}")
    try:
        bytes.fromhex(encoded)
    except ValueError as error:
        raise ValueError(f"invalid QAT-hard {name}") from error
    return value


def _digest_file(path: Path, maximum: int) -> Tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("QAT-hard state must be an ordinary file") from error
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > maximum
        ):
            raise ValueError("QAT-hard state exceeds byte ceiling")
        digest = hashlib.sha256()
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest(), metadata.st_size


def _weight_from_dict(value: Any) -> QatHardWeight:
    if not isinstance(value, dict) or set(value) != _WEIGHT_FIELDS:
        raise ValueError("QAT-hard weight fields differ from schema")
    aliases = value["aliases"]
    consumer_kinds = value["consumer_kinds"]
    shape = value["shape"]
    if (
        not isinstance(value["path"], str)
        or not value["path"]
        or not isinstance(aliases, list)
        or not aliases
        or len(set(aliases)) != len(aliases)
        or any(not isinstance(alias, str) or not alias for alias in aliases)
        or not isinstance(consumer_kinds, list)
        or len(consumer_kinds) != len(aliases)
        or any(kind not in {"linear", "embedding"} for kind in consumer_kinds)
        or not isinstance(value["storage_path"], str)
        or not value["storage_path"]
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        or not isinstance(value["algorithm_id"], str)
        or not value["algorithm_id"]
        or type(value["planes"]) is not int
        or not 1 <= value["planes"] <= 3
    ):
        raise ValueError("QAT-hard weight identity is invalid")
    first_module = (
        "" if aliases[0] == "weight" else aliases[0].removesuffix(".weight")
    )
    expected_storage = (
        f"{first_module}._packed_weight" if first_module else "_packed_weight"
    )
    if value["path"] != aliases[0] or value["storage_path"] != expected_storage:
        raise ValueError("QAT-hard weight storage identity is noncanonical")
    return QatHardWeight(
        path=value["path"],
        aliases=tuple(aliases),
        consumer_kinds=tuple(consumer_kinds),
        storage_path=value["storage_path"],
        shape=(shape[0], shape[1]),
        algorithm_id=value["algorithm_id"],
        planes=value["planes"],
    )


def _validate_weight_coverage(
    weights: Tuple[QatHardWeight, ...], coverage: CoverageReport
) -> None:
    converted = {
        entry.path: entry
        for entry in coverage.entries
        if entry.disposition == "converted"
    }
    by_path = {weight.path: weight for weight in weights}
    if (
        not weights
        or len(by_path) != len(weights)
        or converted.keys() != by_path.keys()
        or any(
            converted[path].aliases != weight.aliases
            or converted[path].numel != weight.shape[0] * weight.shape[1]
            for path, weight in by_path.items()
        )
    ):
        raise ValueError("QAT-hard weights differ from source coverage")
    storage_paths = [weight.storage_path for weight in weights]
    if len(set(storage_paths)) != len(storage_paths):
        raise ValueError("QAT-hard storage paths are not unique")


def _config_from_dict(value: Any) -> TernaryConfig:
    fields = {
        "schema_version",
        "mode",
        "estimator",
        "target_modules",
        "planes",
        "profile",
        "target_bpw",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 2
        or value["mode"] != "qat"
        or not isinstance(value["estimator"], str)
        or not value["estimator"]
        or not isinstance(value["target_modules"], list)
        or not value["target_modules"]
        or any(
            not isinstance(name, str) or not name for name in value["target_modules"]
        )
        or type(value["planes"]) is not int
        or not 1 <= value["planes"] <= 3
        or value["profile"] is not None
        or value["target_bpw"] is not None
    ):
        raise ValueError("QAT-hard config differs from QAT schema")
    return TernaryConfig.from_dict(value)


def _coverage_from_dict(value: Any) -> CoverageReport:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "entries"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or not isinstance(value["entries"], list)
        or not value["entries"]
    ):
        raise ValueError("QAT-hard source coverage differs from schema")
    paths = set()
    aliases = set()
    for entry in value["entries"]:
        if not isinstance(entry, dict) or set(entry) != _COVERAGE_FIELDS:
            raise ValueError("QAT-hard coverage entry fields differ from schema")
        entry_aliases = entry["aliases"]
        if (
            not isinstance(entry["path"], str)
            or not entry["path"]
            or entry["path"] in paths
            or not isinstance(entry_aliases, list)
            or not entry_aliases
            or len(set(entry_aliases)) != len(entry_aliases)
            or any(
                not isinstance(alias, str) or not alias or alias in aliases
                for alias in entry_aliases
            )
            or entry["path"] != entry_aliases[0]
            or entry["disposition"] not in {"converted", "preserved"}
            or not isinstance(entry["reason"], str)
            or not entry["reason"]
            or type(entry["numel"]) is not int
            or entry["numel"] <= 0
            or type(entry["logical_bytes"]) is not int
            or entry["logical_bytes"] <= 0
        ):
            raise ValueError("QAT-hard coverage entry identity is invalid")
        paths.add(entry["path"])
        aliases.update(entry_aliases)
    return CoverageReport.from_dict(value)


def _read_manifest(directory: Path) -> Tuple[dict, bytes]:
    path = directory / _MANIFEST
    metadata = path.lstat()
    if path.is_symlink() or not path.is_file() or not 0 < metadata.st_size <= 1024**2:
        raise ValueError("QAT-hard manifest must be a bounded ordinary file")
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=_pairs_without_duplicates)
    if not isinstance(value, dict) or set(value) != _TOP_FIELDS:
        raise ValueError("QAT-hard manifest fields differ from schema")
    if payload != _canonical(value):
        raise ValueError("QAT-hard manifest is not canonical")
    return value, payload


def _inspect_state(
    path: Path,
    weights: Tuple[QatHardWeight, ...],
    coverage: CoverageReport,
    tensor_ledger: Tuple[Tuple[str, str, Tuple[int, ...]], ...],
) -> str:
    safe_open, _, _ = _dependencies()
    with safe_open(path, framework="pt", device="cpu") as state:
        keys = tuple(state.keys())
        expected_keys = tuple(item[0] for item in tensor_ledger)
        if set(keys) != set(expected_keys) or len(keys) != len(expected_keys):
            raise ValueError("QAT-hard state tensor ledger differs")
        for name, dtype, shape in tensor_ledger:
            tensor_slice = state.get_slice(name)
            if (
                _SAFE_TO_TORCH_DTYPE.get(tensor_slice.get_dtype()) != dtype
                or tuple(tensor_slice.get_shape()) != shape
            ):
                raise ValueError("QAT-hard state tensor ledger differs")
        targeted = {alias for weight in weights for alias in weight.aliases}
        if targeted & set(keys):
            raise ValueError("QAT-hard state contains a targeted floating master")
        estimator_state = {
            alias
            for entry in coverage.entries
            if entry.reason == "estimator_state"
            for alias in entry.aliases
        }
        if estimator_state & set(keys):
            raise ValueError("QAT-hard state contains estimator state")
        for weight in weights:
            rows, columns = weight.shape
            packed_elements = (rows * columns + 4) // 5
            for index in range(weight.planes):
                prefix = weight.storage_path + "."
                packed_name = f"{prefix}packed_trits_{index}"
                scale_name = f"{prefix}scales_{index}"
                if packed_name not in keys or scale_name not in keys:
                    raise ValueError("QAT-hard state omitted compact plane tensors")
                packed = state.get_tensor(packed_name)
                scales = state.get_tensor(scale_name)
                if packed.dtype != torch.uint8 or tuple(packed.shape) != (
                    packed_elements,
                ):
                    raise ValueError("QAT-hard packed plane geometry is invalid")
                flat_packed = packed.reshape(-1)
                for start in range(0, flat_packed.numel(), 1024 * 1024):
                    if bool((flat_packed[start : start + 1024 * 1024] > 242).any()):
                        raise ValueError(
                            "QAT-hard packed plane contains invalid B3 bytes"
                        )
                if scales.dtype != torch.float16 or tuple(scales.shape) != (rows, 1):
                    raise ValueError("QAT-hard scale geometry is invalid")
                flat_scales = scales.reshape(-1)
                for start in range(0, flat_scales.numel(), 1024 * 1024):
                    chunk = flat_scales[start : start + 1024 * 1024]
                    if not bool(torch.isfinite(chunk).all()) or bool((chunk < 0).any()):
                        raise ValueError("QAT-hard scales are invalid")
        from .ptq import _hash_tensor

        digest = hashlib.sha256()
        for name, _, _ in tensor_ledger:
            _hash_tensor(
                digest,
                f"state.{name}",
                state.get_tensor(name),
            )
    return "sha256:" + digest.hexdigest()


def _tensor_ledger(
    value: Any,
) -> Tuple[Tuple[str, str, Tuple[int, ...]], ...]:
    if not isinstance(value, list) or not value or len(value) > 1_000_000:
        raise ValueError("QAT-hard tensor ledger is invalid")
    ledger = []
    names = set()
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != _TENSOR_FIELDS
            or not isinstance(item["name"], str)
            or not item["name"]
            or item["name"] in names
            or item["dtype"] not in _SAFE_TO_TORCH_DTYPE.values()
            or not isinstance(item["shape"], list)
            or any(
                type(dimension) is not int or dimension < 0
                for dimension in item["shape"]
            )
        ):
            raise ValueError("QAT-hard tensor ledger is invalid")
        names.add(item["name"])
        ledger.append((item["name"], item["dtype"], tuple(item["shape"])))
    return tuple(ledger)


def load_qat_hard(
    artifact_dir: Pathish,
    model: Optional[nn.Module] = None,
    *,
    inplace: Optional[bool] = None,
    max_state_bytes: int = 128 * 1024 * 1024 * 1024,
) -> Union[QatHardArtifact, nn.Module]:
    """Verify a QAT-hard bundle and optionally bind it to a source model shell."""

    if type(max_state_bytes) is not int or max_state_bytes <= 0:
        raise ValueError("max_state_bytes must be a positive integer")
    requested = Path(artifact_dir)
    if requested.is_symlink():
        raise ValueError("QAT-hard artifact directory must not be a symlink")
    directory = requested.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("QAT-hard artifact must be an ordinary directory")
    value, _ = _read_manifest(directory)
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["artifact_kind"] != "tritium.module-qat-hard-v1"
    ):
        raise ValueError("unsupported QAT-hard artifact")
    for field in (
        "artifact_id",
        "conversion_artifact_id",
        "source_checkpoint_digest",
        "hard_state_digest",
        "recipe_id",
    ):
        _validate_digest(value[field], field)
    identity = dict(value)
    artifact_id = identity.pop("artifact_id")
    if artifact_id != _digest_bytes(_canonical(identity)):
        raise ValueError("QAT-hard artifact identity mismatch")
    config = _config_from_dict(value["config"])
    coverage = _coverage_from_dict(value["source_coverage"])
    raw_weights = value["weights"]
    if not isinstance(raw_weights, list):
        raise ValueError("QAT-hard weights must be a list")
    weights = tuple(_weight_from_dict(item) for item in raw_weights)
    _validate_weight_coverage(weights, coverage)
    if any(weight.planes != config.planes for weight in weights):
        raise ValueError("QAT-hard plane count differs from recipe")
    recipe_id, conversion_artifact_id = _qat_hard_ids(
        source_checkpoint_digest=value["source_checkpoint_digest"],
        hard_state_digest=value["hard_state_digest"],
        config=config,
        source_coverage=coverage,
        weights=weights,
    )
    if (
        value["recipe_id"] != recipe_id
        or value["conversion_artifact_id"] != conversion_artifact_id
    ):
        raise ValueError("QAT-hard conversion ancestry mismatch")
    state = value["state"]
    if not isinstance(state, dict) or set(state) != _STATE_FIELDS:
        raise ValueError("QAT-hard state fields differ from schema")
    if (
        state["file"] != _STATE
        or type(state["bytes"]) is not int
        or state["bytes"] <= 0
    ):
        raise ValueError("QAT-hard state ledger is invalid")
    tensor_ledger = _tensor_ledger(state["tensors"])
    state_digest, state_bytes = _digest_file(directory / _STATE, max_state_bytes)
    if (
        state_digest != _validate_digest(state["sha256"], "state digest")
        or state_bytes != state["bytes"]
    ):
        raise ValueError("QAT-hard state identity mismatch")
    if {child.name for child in directory.iterdir()} != {_MANIFEST, _STATE}:
        raise ValueError("QAT-hard artifact directory contains unknown files")
    hard_state_digest = _inspect_state(
        directory / _STATE, weights, coverage, tensor_ledger
    )
    if hard_state_digest != value["hard_state_digest"]:
        raise ValueError("QAT-hard state differs from hard-state identity")
    artifact = QatHardArtifact(
        artifact_dir=directory,
        artifact_id=artifact_id,
        conversion_artifact_id=value["conversion_artifact_id"],
        source_checkpoint_digest=value["source_checkpoint_digest"],
        hard_state_digest=value["hard_state_digest"],
        recipe_id=value["recipe_id"],
        config=config,
        source_coverage=coverage,
        weights=weights,
        state_digest=state_digest,
        state_bytes=state_bytes,
        state_tensors=len(tensor_ledger),
        state_ledger=tensor_ledger,
    )
    if model is None:
        if inplace is not None:
            raise TypeError("inplace is only valid when loading into a model")
        return artifact
    if not isinstance(model, nn.Module) or type(inplace) is not bool:
        raise TypeError("QAT-hard model load requires nn.Module and explicit inplace")
    _preflight_shell_state(model, artifact)
    target = model if inplace else copy.deepcopy(model)
    target = _prepare_shell(target, artifact)
    _, load_model, _ = _dependencies()
    missing, unexpected = load_model(target, directory / _STATE, strict=True)
    if missing or unexpected:
        raise ValueError("QAT-hard state differs from model shell")
    for module in target.modules():
        if isinstance(module, AdditiveTernaryWeight):
            module.validate_buffers()
    from .ptq import _source_model_digest

    if _source_model_digest(target) != artifact.hard_state_digest:
        raise ValueError("QAT-hard loaded state identity mismatch")
    target.eval()
    target._tritium_qat_hard_artifact_id = artifact.conversion_artifact_id
    target._tritium_qat_checkpoint_digest = artifact.source_checkpoint_digest
    return target


def _preflight_shell_state(model: nn.Module, artifact: QatHardArtifact) -> None:
    """Reject state topology/geometry mismatches before graph mutation."""

    targeted = {alias for weight in artifact.weights for alias in weight.aliases}
    estimator_state = {
        alias
        for entry in artifact.source_coverage.entries
        if entry.reason == "estimator_state"
        for alias in entry.aliases
    }
    compact = {
        name
        for weight in artifact.weights
        for index in range(weight.planes)
        for name in (
            f"{weight.storage_path}.packed_trits_{index}",
            f"{weight.storage_path}.scales_{index}",
        )
    }
    expected = {
        name: (dtype, shape)
        for name, dtype, shape in artifact.state_ledger
        if name not in compact
    }
    observed = {
        name: (str(tensor.dtype), tuple(tensor.shape))
        for name, tensor in model.state_dict().items()
        if name not in targeted and name not in estimator_state
    }
    if observed != expected:
        raise ValueError("QAT-hard preserved model shell state differs")


def _prepare_shell(model: nn.Module, artifact: QatHardArtifact) -> nn.Module:
    modules = dict(model.named_modules(remove_duplicate=False))
    replacements = []
    root = None
    seen_source_weights = set()
    for weight in artifact.weights:
        by_module = {}
        owner = True
        packed = None
        source_weight_id = None
        for alias, consumer_kind in zip(weight.aliases, weight.consumer_kinds):
            module_path = "" if alias == "weight" else alias.removesuffix(".weight")
            if alias != "weight" and not alias.endswith(".weight"):
                raise ValueError("QAT-hard weight alias is not canonical")
            module = modules.get(module_path)
            if type(module) not in {nn.Linear, nn.Embedding, TernaryLinear, TernaryEmbedding}:
                raise ValueError("QAT-hard target is not a supported module shell")
            actual_kind = (
                "linear"
                if isinstance(module, (nn.Linear, TernaryLinear))
                else "embedding"
            )
            if actual_kind != consumer_kind:
                raise ValueError("QAT-hard model shell consumer kind differs")
            if tuple(module.weight.shape) != weight.shape:
                raise ValueError("QAT-hard model shell geometry differs")
            if source_weight_id is None:
                source_weight_id = id(module.weight)
                if source_weight_id in seen_source_weights:
                    raise ValueError("QAT-hard model shell tie topology differs")
                seen_source_weights.add(source_weight_id)
            elif id(module.weight) != source_weight_id:
                raise ValueError("QAT-hard model shell tie topology differs")
            if packed is None:
                packed = AdditiveTernaryWeight.empty(
                    weight.shape[1],
                    weight.shape[0],
                    weight.planes,
                    device=module.weight.device,
                )
            replacement = by_module.get(id(module))
            if replacement is None:
                if isinstance(module, (nn.Linear, TernaryLinear)):
                    replacement = AdditiveTernaryLinear.empty(
                        weight.shape[1],
                        weight.shape[0],
                        weight.planes,
                        bias=module.bias is not None,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                        packed_weight=packed,
                        owner=owner,
                    )
                else:
                    if module.max_norm is not None:
                        raise ValueError("QAT-hard model shell has mutating max_norm")
                    replacement = AdditiveTernaryEmbedding.empty(
                        weight.shape[0],
                        weight.shape[1],
                        weight.planes,
                        padding_idx=module.padding_idx,
                        max_norm=module.max_norm,
                        norm_type=module.norm_type,
                        scale_grad_by_freq=module.scale_grad_by_freq,
                        sparse=module.sparse,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                        packed_weight=packed,
                        owner=owner,
                    )
                owner = False
                by_module[id(module)] = replacement
            if module_path:
                replacements.append((module_path, replacement))
            else:
                root = replacement
    if root is not None and replacements:
        raise ValueError("QAT-hard root target cannot coexist with nested targets")
    if root is not None:
        return root
    for path, replacement in replacements:
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent._modules[part]
        parent._modules[parts[-1]] = replacement
    return model


def export_qat_hard(result: QatHardResult, output_dir: Pathish) -> QatHardArtifact:
    """Verify, serialize, strict-reopen, and atomically publish QAT-hard state."""

    if not isinstance(result, QatHardResult):
        raise TypeError("export_qat_hard requires a QatHardResult")
    from .ptq import _source_model_digest

    if _source_model_digest(result.model) != result.hard_state_digest:
        raise TritiumError(
            "QAT-hard result model changed after conversion",
            code="state_changed",
            stage="export_qat_hard",
        )
    recipe_id, artifact_id = _qat_hard_ids(
        source_checkpoint_digest=result.source_checkpoint_digest,
        hard_state_digest=result.hard_state_digest,
        config=result.config,
        source_coverage=result.source_coverage,
        weights=result.weights,
    )
    if recipe_id != result.recipe_id or artifact_id != result.artifact_id:
        raise TritiumError(
            "QAT-hard result identity is inconsistent",
            code="artifact_mismatch",
            stage="export_qat_hard",
        )
    _, _, save_model = _dependencies()
    target = Path(output_dir).absolute()
    parent = target.parent.resolve(strict=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"output directory already exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=".tritium-qat-hard-stage-", dir=parent))
    published = False
    try:
        state_path = staging / _STATE
        save_model(
            result.model,
            state_path,
            metadata={"format": "pt", "tritium_mode": "qat-hard"},
            force_contiguous=True,
        )
        tensor_ledger = [
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
            for name, tensor in result.model.state_dict().items()
        ]
        with state_path.open("rb") as stream:
            os.fsync(stream.fileno())
        state_digest, state_bytes = _digest_file(state_path, 128 * 1024**3)
        manifest = {
            "schema_version": 1,
            "artifact_kind": "tritium.module-qat-hard-v1",
            "conversion_artifact_id": result.artifact_id,
            "source_checkpoint_digest": result.source_checkpoint_digest,
            "hard_state_digest": result.hard_state_digest,
            "recipe_id": result.recipe_id,
            "config": result.config.to_dict(),
            "source_coverage": result.source_coverage.to_dict(),
            "weights": [weight.to_dict() for weight in result.weights],
            "state": {
                "file": _STATE,
                "sha256": state_digest,
                "bytes": state_bytes,
                "tensors": tensor_ledger,
            },
        }
        manifest["artifact_id"] = _digest_bytes(_canonical(manifest))
        manifest_path = staging / _MANIFEST
        with manifest_path.open("xb") as stream:
            stream.write(_canonical(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        admitted = load_qat_hard(staging)
        _tritium.publish_directory_noreplace(str(staging), str(target))
        published = True
        reopened = load_qat_hard(target)
        if reopened.artifact_id != admitted.artifact_id:
            raise RuntimeError("published QAT-hard artifact identity changed")
        return reopened
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "QatHardArtifact",
    "export_qat_hard",
    "load_qat_hard",
]
