"""Typed, parent-bound post-training refinement workflows."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .. import _tritium
from .config import RefinementConfig
from .errors import TritiumError
from .module_artifacts import (
    FittedWeight,
    ModuleQuantizationResult,
    PackedModuleArtifact,
    WeightCheckpointWriter,
    load_module_conversion,
    load_packed_module,
    module_recipe_id,
    seal_module_conversion,
)
from .projection import TernaryPlane
from .ptq import _hash_value, _invoke_model, _source_model_digest
from .refinement_core import RefinedWeight, refine_weight_diagonal

Pathish = Union[str, os.PathLike[str]]
_MANIFEST = "refinement.json"
_CONVERSION_DIRECTORY = "conversion"
_PACKED_DIRECTORY = "packed"
_TOP_FIELDS = {
    "schema_version",
    "artifact_kind",
    "artifact_id",
    "parent_artifact_id",
    "ancestry",
    "source_model_digest",
    "teacher_digest",
    "training_digest",
    "training_batches",
    "validation_digest",
    "validation_batches",
    "config",
    "child_conversion_artifact_id",
    "packed_artifact_id",
    "packing",
    "validation_loss_before",
    "validation_loss_after",
    "accepted_steps",
}


@dataclass(frozen=True)
class RefinementResult:
    """One immutable refinement child with complete artifact ancestry."""

    artifact_dir: Path
    artifact_id: str
    parent_artifact_id: str
    ancestry: Tuple[str, ...]
    source_model_digest: str
    teacher_digest: str
    training_digest: str
    training_batches: Tuple[str, ...]
    validation_digest: str
    validation_batches: Tuple[str, ...]
    config: RefinementConfig
    conversion: ModuleQuantizationResult
    packed: Optional[PackedModuleArtifact]
    validation_loss_before: float
    validation_loss_after: float
    accepted_steps: int
    schema_version: int = 2

    @property
    def mode(self) -> str:
        return self.config.kind

    @property
    def weight_names(self) -> Tuple[str, ...]:
        return self.conversion.weight_names

    def weight(self, path: str) -> FittedWeight:
        """Load one child weight while restoring its refinement structure."""

        weight = self.conversion.weight(path)
        planes = tuple(
            TernaryPlane(
                trits=plane.trits,
                scales=plane.scales,
                group_size=plane.group_size,
                structure=self.config.structure,
            )
            for plane in weight.planes
        )
        return FittedWeight(
            path=weight.path,
            aliases=weight.aliases,
            planes=planes,
            weighted_mse=weight.weighted_mse,
        )

    def load_model(self, model: nn.Module, *, inplace: bool = False) -> nn.Module:
        """Bind the hard child to an exact dense source shell."""

        from .ptq import load_quantized_module

        return load_quantized_module(model, self.conversion, inplace=inplace)

    def export(self, output_dir: Pathish) -> "RefinementResult":
        return export_refinement(self, output_dir)


@dataclass(frozen=True)
class _Record:
    path: str
    weight_aliases: Tuple[str, ...]
    outputs: int
    features: int


class _ScaleOnlyWeight(nn.Module):
    def __init__(self, planes: Tuple[TernaryPlane, ...]) -> None:
        super().__init__()
        self.plane_count = len(planes)
        self.structures = tuple(plane.structure for plane in planes)
        for index, plane in enumerate(planes):
            self.register_buffer(f"trits_{index}", plane.trits.detach().clone())
            self.register_parameter(
                f"scale_{index}",
                nn.Parameter(plane.scales.detach().to(torch.float32).clone()),
            )

    def decoded(self, dtype: torch.dtype) -> torch.Tensor:
        decoded = None
        for index in range(self.plane_count):
            trits = getattr(self, f"trits_{index}").to(dtype=dtype)
            scale = getattr(self, f"scale_{index}").clamp_min(
                torch.finfo(torch.float16).tiny
            )
            # The forward is exactly the eventual f16 artifact while gradients
            # flow through the full-precision scale parameter.
            stored = scale.to(torch.float16).to(scale.dtype)
            quantized = scale + (stored - scale).detach()
            plane = trits * quantized.to(dtype=dtype)
            decoded = plane if decoded is None else decoded + plane
        assert decoded is not None
        return decoded

    def hard_planes(self) -> Tuple[TernaryPlane, ...]:
        planes = []
        for index in range(self.plane_count):
            trits = getattr(self, f"trits_{index}").detach().cpu().to(torch.int8)
            scales = (
                getattr(self, f"scale_{index}")
                .detach()
                .clamp_min(torch.finfo(torch.float16).tiny)
                .cpu()
                .to(torch.float16)
            )
            planes.append(
                TernaryPlane(
                    trits,
                    scales,
                    trits.shape[1],
                    structure=self.structures[index],
                )
            )
        return tuple(planes)


class _ScaleOnlyLinear(nn.Module):
    def __init__(
        self,
        weight: _ScaleOnlyWeight,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "weight_store", weight)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.compute_dtype = compute_dtype

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.to(dtype=self.compute_dtype)
        return F.linear(
            value,
            self.weight_store.decoded(self.compute_dtype),
            self.bias,
        )


class _ScaleOnlyEmbedding(nn.Module):
    def __init__(self, weight: _ScaleOnlyWeight, source: nn.Embedding) -> None:
        super().__init__()
        object.__setattr__(self, "weight_store", weight)
        self.padding_idx = source.padding_idx
        self.max_norm = source.max_norm
        self.norm_type = source.norm_type
        self.scale_grad_by_freq = source.scale_grad_by_freq
        self.sparse = source.sparse
        self.weight_dtype = source.weight.dtype

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            value,
            self.weight_store.decoded(self.weight_dtype),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _write_manifest_atomic(directory: Path, value: Dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp-refinement-", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / _MANIFEST)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_digest(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
    ):
        raise ValueError(f"invalid {field}")
    try:
        bytes.fromhex(value[7:])
    except ValueError as error:
        raise ValueError(f"invalid {field}") from error
    return value


def _pairs_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate refinement manifest field {key!r}")
        value[key] = item
    return value


def _materialize(
    data: Iterable[Any], role: str
) -> Tuple[Tuple[Any, ...], str, Tuple[str, ...]]:
    try:
        batches = tuple(data)
    except TypeError as error:
        raise TypeError(f"{role} data must be iterable") from error
    if not batches:
        raise ValueError(f"{role} data must contain at least one batch")
    members = []
    for batch in batches:
        digest = hashlib.sha256()
        _hash_value(digest, "batch", batch)
        members.append("sha256:" + digest.hexdigest())
    identity = _digest({"schema_version": 1, "role": role, "batches": members})
    return batches, identity, tuple(members)


def _evidence_id(
    *,
    parent_artifact_id: str,
    teacher_digest: str,
    training_digest: str,
    training_batches: Tuple[str, ...],
    validation_digest: str,
    validation_batches: Tuple[str, ...],
    config: RefinementConfig,
) -> str:
    return _digest(
        {
            "schema_version": 1,
            "parent_artifact_id": parent_artifact_id,
            "teacher_digest": teacher_digest,
            "training_digest": training_digest,
            "training_batches": list(training_batches),
            "validation_digest": validation_digest,
            "validation_batches": list(validation_batches),
            "refinement": config.to_dict(),
        }
    )


def _output_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("logits"), torch.Tensor):
        return value["logits"]
    logits = getattr(value, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if (
        isinstance(value, (tuple, list))
        and value
        and isinstance(value[0], torch.Tensor)
    ):
        return value[0]
    raise TritiumError(
        "refinement model output has no tensor logits",
        code="unsupported_output",
        stage="refine",
    )


def _cast_batch(batch: Any, dtype: Optional[torch.dtype]) -> Any:
    """Cast floating batch leaves for low-precision teacher execution."""

    if dtype is None:
        return batch
    if isinstance(batch, torch.Tensor):
        return batch.to(dtype=dtype) if batch.is_floating_point() else batch
    if not isinstance(batch, Mapping):
        return batch
    return {
        key: value.to(dtype=dtype)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value
        for key, value in batch.items()
    }


def _loss(
    student: torch.Tensor, teacher: torch.Tensor, temperature: float
) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise TritiumError(
            "student and teacher logits have different geometry",
            code="output_geometry",
            stage="refine",
            details={"student": tuple(student.shape), "teacher": tuple(teacher.shape)},
        )
    if student.ndim > 0 and student.shape[-1] > 1:
        return (
            F.kl_div(
                F.log_softmax(student.float() / temperature, dim=-1),
                F.softmax(teacher.float() / temperature, dim=-1),
                reduction="batchmean",
            )
            * temperature**2
        )
    return F.mse_loss(student.float(), teacher.float())


def _evaluate(
    student: nn.Module,
    teacher: nn.Module,
    batches: Tuple[Any, ...],
    temperature: float,
    input_dtype: Optional[torch.dtype] = None,
) -> float:
    total = 0.0
    with torch.no_grad():
        for batch in batches:
            total += float(
                _loss(
                    _output_tensor(
                        _invoke_model(student, _cast_batch(batch, input_dtype))
                    ),
                    _output_tensor(
                        _invoke_model(teacher, _cast_batch(batch, input_dtype))
                    ),
                    temperature,
                )
            )
    value = total / len(batches)
    if not math.isfinite(value):
        raise TritiumError(
            "refinement validation loss is non-finite",
            code="nonfinite_loss",
            stage="refine",
        )
    return value


def _parent_conversion(
    parent: Union[ModuleQuantizationResult, RefinementResult],
) -> Tuple[ModuleQuantizationResult, str, Tuple[str, ...]]:
    if isinstance(parent, RefinementResult):
        reopened = load_refinement(parent.artifact_dir)
        if reopened != parent:
            raise TritiumError(
                "refinement parent changed after admission",
                code="parent_changed",
                stage="refine",
            )
        return (
            parent.conversion,
            parent.artifact_id,
            (*parent.ancestry, parent.artifact_id),
        )
    if isinstance(parent, ModuleQuantizationResult):
        reopened = load_module_conversion(parent.artifact_dir)
        if reopened != parent:
            raise TritiumError(
                "PTQ parent changed after admission",
                code="parent_changed",
                stage="refine",
            )
        return parent, parent.artifact_id, (parent.artifact_id,)
    raise TypeError(
        "refine requires a ModuleQuantizationResult or RefinementResult parent"
    )


def _build_scale_model(
    teacher: nn.Module,
    conversion: ModuleQuantizationResult,
    plane_overrides: Optional[Dict[str, Tuple[TernaryPlane, ...]]] = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Tuple[nn.Module, Dict[str, _ScaleOnlyWeight]]:
    try:
        model = copy.deepcopy(teacher)
    except Exception as error:
        raise TritiumError(
            "teacher could not be copied for refinement",
            code="copy_failed",
            stage="refine",
        ) from error
    if compute_dtype is not torch.float32:
        model.to(dtype=compute_dtype)
    model.requires_grad_(False)
    modules = dict(model.named_modules(remove_duplicate=False))
    stores: Dict[str, _ScaleOnlyWeight] = {}
    replacements: Dict[str, nn.Module] = {}
    replacement_by_module: Dict[int, nn.Module] = {}
    for reference in conversion.weights:
        fitted = conversion.weight(reference.path)
        planes = (
            fitted.planes
            if plane_overrides is None
            else plane_overrides[reference.path]
        )
        first_alias = reference.aliases[0]
        first_path = (
            "" if first_alias == "weight" else first_alias.removesuffix(".weight")
        )
        source = modules.get(first_path)
        if type(source) not in {nn.Linear, nn.Embedding}:
            raise TritiumError(
                "refinement target is not an exact Linear or Embedding",
                code="coverage_mismatch",
                stage="refine",
                module=first_path,
            )
        store = _ScaleOnlyWeight(planes).to(source.weight.device)
        stores[reference.path] = store
        for alias in reference.aliases:
            path = "" if alias == "weight" else alias.removesuffix(".weight")
            module = modules.get(path)
            replacement = replacement_by_module.get(id(module))
            if replacement is None:
                if type(module) is nn.Linear:
                    replacement = _ScaleOnlyLinear(
                        store,
                        module.bias,
                        compute_dtype,
                    )
                elif type(module) is nn.Embedding:
                    if module.max_norm is not None:
                        raise TritiumError(
                            "scale-only refinement cannot preserve mutating Embedding max_norm",
                            code="unsupported_module_option",
                            stage="refine",
                            module=path,
                        )
                    replacement = _ScaleOnlyEmbedding(store, module)
                else:
                    raise TritiumError(
                        "refinement alias does not resolve to a supported module",
                        code="coverage_mismatch",
                        stage="refine",
                        module=path,
                    )
                replacement_by_module[id(module)] = replacement
            replacements[path] = replacement
    root = model
    for path, replacement in replacements.items():
        if path == "":
            root = replacement
            continue
        parts = path.split(".")
        owner = root
        for part in parts[:-1]:
            owner = owner._modules[part]
        owner._modules[parts[-1]] = replacement
    # Register each shared scale store exactly once after graph replacement.
    for index, store in enumerate(stores.values()):
        root.add_module(f"_tritium_scale_store_{index}", store)
    if compute_dtype != torch.float32 and hasattr(
        root, "gradient_checkpointing_enable"
    ):
        # Billion-parameter transformer refinement otherwise retains every
        # activation for scale gradients. Non-FP32 recipes are explicitly the
        # memory-saving path; checkpointing keeps 2048-token Stage-7 batches
        # inside commodity GPU memory.
        try:
            root.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            root.gradient_checkpointing_enable()
    root.eval()
    return root, stores


def _input_metrics(
    teacher: nn.Module,
    conversion: ModuleQuantizationResult,
    batches: Tuple[Any, ...],
    input_dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """Collect diagonal input curvature for each unique target weight."""

    modules = dict(teacher.named_modules(remove_duplicate=False))
    metrics: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    handles = []
    for reference in conversion.weights:
        selected = []
        seen_modules = set()
        for alias in reference.aliases:
            path = "" if alias == "weight" else alias.removesuffix(".weight")
            module = modules.get(path)
            if type(module) is nn.Linear and id(module) not in seen_modules:
                selected.append(module)
                seen_modules.add(id(module))
        metrics[reference.path] = torch.zeros(reference.shape[1], dtype=torch.float64)
        counts[reference.path] = 0
        if not selected:
            # The core metric weights columns. Token frequency weights embedding
            # rows, so it cannot substitute without changing the objective.
            metrics[reference.path].fill_(1)
            counts[reference.path] = 1
            continue

        def collect(module, inputs, path=reference.path, features=reference.shape[1]):
            value = inputs[0]
            if not isinstance(value, torch.Tensor) or value.shape[-1] != features:
                raise TritiumError(
                    "refinement Linear input geometry differs from parent",
                    code="input_geometry",
                    stage="refine",
                    module=path,
                )
            flat = value.detach().reshape(-1, features)
            # RTX-class GPUs have weak FP64 throughput. Accumulate each batch in
            # FP32 on device, then fold into the deterministic FP64 host metric.
            # This keeps receipt precision while avoiding a multi-minute FP64
            # kernel for every hooked target activation.
            partial = flat.float().square().sum(dim=0)
            metrics[path].add_(partial.cpu().to(torch.float64))
            counts[path] += flat.shape[0]

        handles.extend(module.register_forward_pre_hook(collect) for module in selected)
    was_training = teacher.training
    teacher.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                _invoke_model(teacher, _cast_batch(batch, input_dtype))
    finally:
        for handle in handles:
            handle.remove()
        teacher.train(was_training)
    tiny = torch.finfo(torch.float64).tiny
    for path, metric in metrics.items():
        if counts[path] <= 0:
            raise TritiumError(
                "refinement collected no target activations",
                code="empty_activations",
                stage="refine",
                module=path,
            )
        metrics[path] = (metric / counts[path]).clamp_min(tiny)
    return metrics


def _initial_refinement(
    teacher: nn.Module,
    conversion: ModuleQuantizationResult,
    metrics: Dict[str, torch.Tensor],
    config: RefinementConfig,
) -> Tuple[Dict[str, Tuple[TernaryPlane, ...]], Dict[str, RefinedWeight]]:
    modules = dict(teacher.named_modules(remove_duplicate=False))
    planes = {}
    receipts = {}
    for reference in conversion.weights:
        fitted = conversion.weight(reference.path)
        alias = reference.aliases[0]
        path = "" if alias == "weight" else alias.removesuffix(".weight")
        module = modules.get(path)
        if type(module) not in {nn.Linear, nn.Embedding}:
            raise TritiumError(
                "refinement target is not an exact Linear or Embedding",
                code="coverage_mismatch",
                stage="refine",
                module=path,
            )
        receipt = refine_weight_diagonal(
            module.weight.detach().cpu(),
            fitted.planes,
            metrics[reference.path],
            config,
            iterations=max(1, config.pv_iterations),
        )
        planes[reference.path] = receipt.planes
        receipts[reference.path] = receipt
    return planes, receipts


def _snapshot(
    stores: Dict[str, _ScaleOnlyWeight],
) -> Dict[str, Tuple[TernaryPlane, ...]]:
    return {path: store.hard_planes() for path, store in stores.items()}


def _install(
    stores: Dict[str, _ScaleOnlyWeight],
    planes: Dict[str, Tuple[TernaryPlane, ...]],
) -> None:
    for path, store in stores.items():
        for index, plane in enumerate(planes[path]):
            trits = getattr(store, f"trits_{index}")
            scale = getattr(store, f"scale_{index}")
            trits.copy_(plane.trits.to(trits.device))
            scale.data.copy_(plane.scales.to(scale.device))


def _weight_mse(
    master: torch.Tensor,
    planes: Tuple[TernaryPlane, ...],
    metric: torch.Tensor,
) -> float:
    # Source weights and exported scales are at most FP16/FP32. Keep the
    # per-element decode/residual path in FP32; only the final scalar sum is
    # FP64. This preserves deterministic chunk order and receipt precision
    # while avoiding a 3x FP64 bandwidth penalty on billion-parameter models.
    total = 0.0
    denominator = float(metric.sum()) * master.shape[0]
    metric_fp32 = metric.to(torch.float32)
    # Teacher may remain resident on CUDA for the preceding evaluation. Pull
    # each master weight across the bus once; slicing a CUDA tensor inside the
    # loop turns every chunk into a synchronous transfer and dominates sealing.
    master_cpu = master.detach().cpu().to(torch.float32)
    rows_per_chunk = max(1, (64 * 1024 * 1024) // max(1, master.shape[1] * 24))
    for start in range(0, master.shape[0], rows_per_chunk):
        end = min(master.shape[0], start + rows_per_chunk)
        decoded = torch.zeros((end - start, master.shape[1]), dtype=torch.float32)
        for plane in planes:
            decoded.add_(
                plane.trits[start:end].to(torch.float32)
                * plane.scales[start:end].to(torch.float32)
            )
        residual = master_cpu[start:end] - decoded
        total += float(
            (residual.square() * metric_fp32).sum(dtype=torch.float64)
        )
    return total / denominator


def _manifest_value(result: RefinementResult) -> Dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "artifact_kind": f"tritium.module-{result.config.kind}-refinement-v2",
        "artifact_id": result.artifact_id,
        "parent_artifact_id": result.parent_artifact_id,
        "ancestry": list(result.ancestry),
        "source_model_digest": result.source_model_digest,
        "teacher_digest": result.teacher_digest,
        "training_digest": result.training_digest,
        "training_batches": list(result.training_batches),
        "validation_digest": result.validation_digest,
        "validation_batches": list(result.validation_batches),
        "config": result.config.to_dict(),
        "child_conversion_artifact_id": result.conversion.artifact_id,
        "packed_artifact_id": (
            None if result.packed is None else result.packed.artifact_id
        ),
        "packing": None if result.packed is None else result.packed.packing,
        "validation_loss_before": result.validation_loss_before,
        "validation_loss_after": result.validation_loss_after,
        "accepted_steps": result.accepted_steps,
    }


def load_refinement(artifact_dir: Pathish) -> RefinementResult:
    requested = Path(artifact_dir)
    if requested.is_symlink():
        raise ValueError("refinement directory must not be a symlink")
    directory = requested.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("refinement artifact must be an ordinary directory")
    manifest_path = directory / _MANIFEST
    metadata = manifest_path.lstat()
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or metadata.st_size > 1024**2
    ):
        raise ValueError("refinement manifest must be a bounded ordinary file")
    with manifest_path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_pairs_without_duplicates)
    if not isinstance(value, dict) or set(value) != _TOP_FIELDS:
        raise ValueError("refinement manifest fields differ from schema")
    if value["schema_version"] != 2:
        raise ValueError(
            "unsupported refinement schema_version; pre-v2 artifacts must rerun refine"
        )
    config = RefinementConfig.from_dict(value["config"])
    if value["artifact_kind"] != f"tritium.module-{config.kind}-refinement-v2":
        raise ValueError("refinement kind differs from recipe")
    for field in (
        "artifact_id",
        "parent_artifact_id",
        "source_model_digest",
        "teacher_digest",
        "training_digest",
        "validation_digest",
        "child_conversion_artifact_id",
    ):
        _validate_digest(value[field], field)
    if value["packed_artifact_id"] is not None:
        _validate_digest(value["packed_artifact_id"], "packed_artifact_id")
    training_batches = value["training_batches"]
    validation_batches = value["validation_batches"]
    if (
        not isinstance(training_batches, list)
        or not training_batches
        or not isinstance(validation_batches, list)
        or not validation_batches
        or any(
            _validate_digest(item, "batch digest") != item
            for item in (*training_batches, *validation_batches)
        )
        or set(training_batches) & set(validation_batches)
        or value["training_digest"]
        != _digest(
            {
                "schema_version": 1,
                "role": "training",
                "batches": training_batches,
            }
        )
        or value["validation_digest"]
        != _digest(
            {
                "schema_version": 1,
                "role": "validation",
                "batches": validation_batches,
            }
        )
    ):
        raise ValueError("refinement dataset ledger is invalid")
    ancestry = value["ancestry"]
    if (
        not isinstance(ancestry, list)
        or not ancestry
        or ancestry[-1] != value["parent_artifact_id"]
        or any(_validate_digest(item, "ancestry") != item for item in ancestry)
        or len(set(ancestry)) != len(ancestry)
    ):
        raise ValueError("refinement ancestry is invalid")
    before = value["validation_loss_before"]
    after = value["validation_loss_after"]
    accepted = value["accepted_steps"]
    if (
        type(before) not in {int, float}
        or type(after) not in {int, float}
        or not math.isfinite(before)
        or not math.isfinite(after)
        or before < 0
        or after < 0
        or after > before
        or type(accepted) is not int
        or not 0 <= accepted <= config.max_steps
    ):
        raise ValueError("refinement metrics are invalid")
    conversion_path = directory / _CONVERSION_DIRECTORY
    packed_path = directory / _PACKED_DIRECTORY
    if conversion_path.is_symlink() or not conversion_path.is_dir():
        raise ValueError("refinement conversion must be an ordinary directory")
    expected_children = {_MANIFEST, _CONVERSION_DIRECTORY}
    if value["packed_artifact_id"] is not None:
        expected_children.add(_PACKED_DIRECTORY)
        if packed_path.is_symlink() or not packed_path.is_dir():
            raise ValueError("refinement packed package must be an ordinary directory")
    if {child.name for child in directory.iterdir()} != expected_children:
        raise ValueError("refinement directory contains unknown files")
    conversion = load_module_conversion(conversion_path)
    if conversion.artifact_id != value["child_conversion_artifact_id"]:
        raise ValueError("refinement child conversion identity mismatch")
    packable = all(reference.shape[1] % 128 == 0 for reference in conversion.weights)
    packed = None
    if packable:
        if value["packed_artifact_id"] is None:
            raise ValueError("G128 refinement is missing its packed package")
        packed = load_packed_module(packed_path)
        expected_packing = "s34" if config.structure == "s34" else "b3"
        if (
            value["packing"] != expected_packing
            or packed.packing != expected_packing
            or packed.artifact_id != value["packed_artifact_id"]
            or packed.conversion_artifact_id != conversion.artifact_id
            or packed.source_model_digest != conversion.source_model_digest
            or packed.evidence_id != conversion.evidence_id
            or packed.algorithm_id != conversion.algorithm_id
            or packed.recipe_id != conversion.recipe_id
        ):
            raise ValueError("refinement packed package ancestry mismatch")
    elif value["packed_artifact_id"] is not None or value["packing"] is not None:
        raise ValueError("unaligned refinement cannot claim a SALT package")
    expected_evidence_id = _evidence_id(
        parent_artifact_id=value["parent_artifact_id"],
        teacher_digest=value["teacher_digest"],
        training_digest=value["training_digest"],
        training_batches=tuple(training_batches),
        validation_digest=value["validation_digest"],
        validation_batches=tuple(validation_batches),
        config=config,
    )
    expected_algorithm = f"tritium.{config.kind}-{config.structure}-refinement@1"
    expected_recipe = module_recipe_id(
        conversion.source_model_digest,
        expected_evidence_id,
        expected_algorithm,
        conversion.config,
        conversion.coverage,
    )
    if (
        value["teacher_digest"] != value["source_model_digest"]
        or conversion.source_model_digest != value["source_model_digest"]
        or conversion.evidence_id != expected_evidence_id
        or conversion.algorithm_id != expected_algorithm
        or conversion.recipe_id != expected_recipe
    ):
        raise ValueError("refinement child conversion ancestry mismatch")
    if config.structure == "s34":
        for reference in conversion.weights:
            fitted = conversion.weight(reference.path)
            if reference.shape[1] % 4:
                raise ValueError("S34 refinement width is not divisible by four")
            for plane in fitted.planes:
                groups = plane.trits.reshape(reference.shape[0], -1, 4)
                if not bool(torch.all(torch.count_nonzero(groups == 0, dim=2) == 1)):
                    raise ValueError("S34 refinement child violates one-zero groups")
    identity = dict(value)
    artifact_id = identity.pop("artifact_id")
    if artifact_id != _digest(identity):
        raise ValueError("refinement artifact identity mismatch")
    return RefinementResult(
        artifact_dir=directory,
        artifact_id=artifact_id,
        parent_artifact_id=value["parent_artifact_id"],
        ancestry=tuple(ancestry),
        source_model_digest=value["source_model_digest"],
        teacher_digest=value["teacher_digest"],
        training_digest=value["training_digest"],
        training_batches=tuple(training_batches),
        validation_digest=value["validation_digest"],
        validation_batches=tuple(validation_batches),
        config=config,
        conversion=conversion,
        packed=packed,
        validation_loss_before=float(before),
        validation_loss_after=float(after),
        accepted_steps=accepted,
    )


def refine(
    parent: Union[ModuleQuantizationResult, RefinementResult],
    *,
    teacher: nn.Module,
    training: Iterable[Any],
    validation: Iterable[Any],
    config: RefinementConfig,
    work_dir: Pathish,
) -> RefinementResult:
    """Refine a hard PTQ child while preserving explicit ancestry and claims."""

    if not isinstance(teacher, nn.Module):
        raise TypeError("refine teacher must be a torch.nn.Module")
    if not isinstance(config, RefinementConfig):
        raise TypeError("refine config must be a RefinementConfig")
    conversion, parent_id, ancestry = _parent_conversion(parent)
    teacher_digest = _source_model_digest(teacher)
    if teacher_digest != conversion.source_model_digest:
        raise TritiumError(
            "teacher differs from the PTQ source checkpoint",
            code="source_changed",
            stage="refine",
            details={
                "expected": conversion.source_model_digest,
                "observed": teacher_digest,
            },
        )
    training_batches, training_digest, training_members = _materialize(
        training, "training"
    )
    validation_batches, validation_digest, validation_members = _materialize(
        validation, "validation"
    )
    if set(training_members) & set(validation_members):
        raise TritiumError(
            "training and validation data must be content-disjoint",
            code="dataset_overlap",
            stage="refine",
        )
    requested = Path(work_dir)
    if requested.is_symlink():
        raise ValueError("refinement work_dir must not be a symlink")
    directory = requested.resolve()
    if (directory / _MANIFEST).exists():
        result = load_refinement(directory)
        expected = (
            parent_id,
            ancestry,
            teacher_digest,
            training_digest,
            training_members,
            validation_digest,
            validation_members,
            config,
        )
        observed = (
            result.parent_artifact_id,
            result.ancestry,
            result.teacher_digest,
            result.training_digest,
            result.training_batches,
            result.validation_digest,
            result.validation_batches,
            result.config,
        )
        if observed != expected:
            raise ValueError("sealed refinement belongs to another recipe or dataset")
        return result
    directory.mkdir(parents=True, exist_ok=True)
    for child in tuple(directory.iterdir()):
        stale = child.name.startswith((".tmp-refinement-", ".tritium-module-stage-"))
        if stale and not child.is_symlink():
            if child.is_dir():
                shutil.rmtree(child)
            elif child.is_file():
                child.unlink()
    unknown = {
        child.name
        for child in directory.iterdir()
        if child.name not in {_CONVERSION_DIRECTORY, _PACKED_DIRECTORY}
        or child.is_symlink()
        or not child.is_dir()
    }
    if unknown:
        raise ValueError("unsealed refinement work_dir contains unknown state")

    compute_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.compute_dtype]
    original_tensors = {
        name: (tensor, tensor.dtype)
        for name, tensor in (
            *teacher.named_parameters(),
            *teacher.named_buffers(),
        )
        if tensor.is_floating_point() or tensor.is_complex()
    }
    if compute_dtype != torch.float32:
        # Exact source identity was admitted above. The recipe's lower-precision
        # execution mode now applies to teacher and student alike, avoiding a
        # resident FP32 teacher plus FP16 student duplicate on constrained GPUs.
        teacher.to(dtype=compute_dtype)

    teacher_was_training = teacher.training
    teacher.eval()
    try:
        parent_student, _ = _build_scale_model(
            teacher, conversion, compute_dtype=compute_dtype
        )
        before = _evaluate(
            parent_student,
            teacher,
            validation_batches,
            config.temperature,
            input_dtype=compute_dtype,
        )
        del parent_student
        if config.kind == "scale-only":
            # Scale-only refinement freezes trits by contract. Re-solving every
            # plane before optimization is redundant and, for billion-parameter
            # models, needlessly materializes a full FP64 CPU objective. Start
            # from the admitted PTQ planes; hard-PV retains curvature fitting.
            initial_planes = {
                reference.path: conversion.weight(reference.path).planes
                for reference in conversion.weights
            }
            # No activation-weighted solve is performed in this mode. Use a
            # deterministic unit metric for the child diagnostics below.
            metrics = {
                reference.path: torch.ones(
                    reference.shape[1], dtype=torch.float64
                )
                for reference in conversion.weights
            }
        else:
            metrics = _input_metrics(
                teacher,
                conversion,
                training_batches,
                input_dtype=compute_dtype,
            )
            initial_planes, _ = _initial_refinement(
                teacher, conversion, metrics, config
            )
        student, stores = _build_scale_model(
            teacher,
            conversion,
            initial_planes,
            compute_dtype=compute_dtype,
        )
        initial_loss = _evaluate(
            student,
            teacher,
            validation_batches,
            config.temperature,
            input_dtype=compute_dtype,
        )
        best_planes = _snapshot(stores) if initial_loss <= before else None
        best_loss = initial_loss if best_planes is not None else before
        accepted_steps = 0
        parameters = [
            parameter for store in stores.values() for parameter in store.parameters()
        ]
        optimizer = torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=0.0
        )
        checkpointed_training = compute_dtype != torch.float32 and hasattr(
            student, "gradient_checkpointing_enable"
        )
        if checkpointed_training:
            student.train()
        for batch in (
            item for _, item in zip(range(config.max_steps), cycle(training_batches))
        ):
            optimizer.zero_grad(set_to_none=True)
            compute_batch = _cast_batch(batch, compute_dtype)
            with torch.no_grad():
                teacher_output = _output_tensor(
                    _invoke_model(teacher, compute_batch)
                )
            student_output = _output_tensor(_invoke_model(student, compute_batch))
            loss = _loss(student_output, teacher_output, config.temperature)
            if not bool(torch.isfinite(loss)):
                raise TritiumError(
                    "refinement training loss is non-finite",
                    code="nonfinite_loss",
                    stage="refine",
                )
            loss.backward()
            optimizer.step()
            if checkpointed_training:
                student.eval()
            candidate = _evaluate(
                student,
                teacher,
                validation_batches,
                config.temperature,
                input_dtype=compute_dtype,
            )
            if checkpointed_training:
                student.train()
            if candidate < best_loss and candidate <= before:
                best_loss = candidate
                accepted_steps += 1
                best_planes = _snapshot(stores)
        if best_planes is None:
            if config.structure == "s34":
                raise TritiumError(
                    "S34 hard-PV produced no held-out improvement",
                    code="no_admissible_refinement",
                    stage="refine",
                )
            best_planes = {
                reference.path: conversion.weight(reference.path).planes
                for reference in conversion.weights
            }
        _install(stores, best_planes)
        student.eval()
        after = _evaluate(
            student,
            teacher,
            validation_batches,
            config.temperature,
            input_dtype=compute_dtype,
        )
    finally:
        teacher.train(teacher_was_training)
        if compute_dtype != torch.float32:
            for tensor, dtype in original_tensors.values():
                # `teacher.to` mutates tensor storage in place; restore the
                # caller-owned module's exact floating/complex dtypes.
                if tensor.dtype != dtype:
                    tensor.data = tensor.data.to(dtype=dtype)
    if after > before:
        raise TritiumError(
            "held-out predecessor retention failed",
            code="predecessor_retention_failed",
            stage="refine",
            details={"predecessor_loss": before, "retained_loss": after},
        )

    evidence_id = _evidence_id(
        parent_artifact_id=parent_id,
        teacher_digest=teacher_digest,
        training_digest=training_digest,
        training_batches=training_members,
        validation_digest=validation_digest,
        validation_batches=validation_members,
        config=config,
    )
    algorithm_id = f"tritium.{config.kind}-{config.structure}-refinement@1"
    recipe_id = module_recipe_id(
        conversion.source_model_digest,
        evidence_id,
        algorithm_id,
        conversion.config,
        conversion.coverage,
    )
    records = tuple(
        _Record(
            reference.path,
            reference.aliases,
            reference.shape[0],
            reference.shape[1],
        )
        for reference in conversion.weights
    )
    modules = dict(teacher.named_modules(remove_duplicate=False))
    final_planes = _snapshot(stores)
    weight_losses = {}
    for reference in conversion.weights:
        alias = reference.aliases[0]
        path = "" if alias == "weight" else alias.removesuffix(".weight")
        weight_losses[reference.path] = _weight_mse(
            modules[path].weight,
            final_planes[reference.path],
            metrics[reference.path],
        )

    def fit_weight(record: _Record, writer: WeightCheckpointWriter) -> float:
        planes = final_planes[record.path]
        for start in range(0, record.outputs, writer.fit_chunk_rows):
            end = min(record.outputs, start + writer.fit_chunk_rows)
            writer.append(
                tuple(
                    TernaryPlane(
                        trits=plane.trits[start:end],
                        scales=plane.scales[start:end],
                        group_size=plane.group_size,
                        structure=plane.structure,
                    )
                    for plane in planes
                )
            )
        return weight_losses[record.path]

    max_working_bytes = 64 * 1024 * 1024

    def fit_chunk_rows(record: _Record) -> int:
        bytes_per_row = max(1, record.features * 3 + 6)
        return max(1, min(record.outputs, max_working_bytes // bytes_per_row))

    child = seal_module_conversion(
        directory / _CONVERSION_DIRECTORY,
        source_model_digest=conversion.source_model_digest,
        evidence_id=evidence_id,
        algorithm_id=algorithm_id,
        recipe_id=recipe_id,
        config=conversion.config,
        coverage=conversion.coverage,
        records=records,
        fit_weight=fit_weight,
        fit_chunk_rows=fit_chunk_rows,
        max_working_bytes=max_working_bytes,
    )
    packable = all(reference.shape[1] % 128 == 0 for reference in child.weights)
    packing = ("s34" if config.structure == "s34" else "b3") if packable else None
    packed_path = directory / _PACKED_DIRECTORY
    if packing is None:
        if packed_path.exists() or packed_path.is_symlink():
            raise ValueError("unaligned refinement has an unexpected packed package")
        packed = None
    elif packed_path.exists() or packed_path.is_symlink():
        packed = load_packed_module(packed_path)
        if (
            packed.packing != packing
            or packed.conversion_artifact_id != child.artifact_id
            or packed.source_model_digest != child.source_model_digest
            or packed.evidence_id != child.evidence_id
            or packed.algorithm_id != child.algorithm_id
            or packed.recipe_id != child.recipe_id
        ):
            raise ValueError("resumed packed package differs from refinement child")
    else:
        packed = child.pack_native(packed_path, packing=packing)
    identity = {
        "schema_version": 2,
        "artifact_kind": f"tritium.module-{config.kind}-refinement-v2",
        "parent_artifact_id": parent_id,
        "ancestry": list(ancestry),
        "source_model_digest": conversion.source_model_digest,
        "teacher_digest": teacher_digest,
        "training_digest": training_digest,
        "training_batches": list(training_members),
        "validation_digest": validation_digest,
        "validation_batches": list(validation_members),
        "config": config.to_dict(),
        "child_conversion_artifact_id": child.artifact_id,
        "packed_artifact_id": None if packed is None else packed.artifact_id,
        "packing": packing,
        "validation_loss_before": before,
        "validation_loss_after": after,
        "accepted_steps": accepted_steps,
    }
    identity["artifact_id"] = _digest(identity)
    _write_manifest_atomic(directory, identity)
    return load_refinement(directory)


def export_refinement(
    result: RefinementResult, output_dir: Pathish
) -> RefinementResult:
    """Atomically publish and strictly reopen a refinement artifact."""

    if not isinstance(result, RefinementResult):
        raise TypeError("export_refinement requires a RefinementResult")
    current = load_refinement(result.artifact_dir)
    if current != result:
        raise TritiumError(
            "RefinementResult fields differ from strict artifact reload",
            code="artifact_changed",
            stage="export",
        )
    target = Path(output_dir).absolute()
    parent = target.parent.resolve(strict=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"output directory already exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=".tritium-refine-", dir=parent))
    published = False
    try:
        shutil.copytree(
            current.artifact_dir / _CONVERSION_DIRECTORY,
            staging / _CONVERSION_DIRECTORY,
        )
        if current.packed is not None:
            shutil.copytree(
                current.artifact_dir / _PACKED_DIRECTORY,
                staging / _PACKED_DIRECTORY,
            )
        shutil.copyfile(current.artifact_dir / _MANIFEST, staging / _MANIFEST)
        load_refinement(staging)
        for root, directories, files in os.walk(staging, topdown=False):
            for name in files:
                descriptor = os.open(Path(root) / name, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            for name in directories:
                descriptor = os.open(Path(root) / name, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _tritium.publish_directory_noreplace(str(staging), str(target))
        descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return load_refinement(target)


__all__ = ["RefinementResult", "export_refinement", "load_refinement", "refine"]
