"""Hard, inference-only conversion of differentiable QAT module graphs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict, Tuple

import torch
from torch import nn

from ..nn import (
    AdditiveTernaryConv1d,
    AdditiveTernaryConv2d,
    AdditiveTernaryEmbedding,
    AdditiveTernaryLinear,
    AdditiveTernaryWeight,
    TernaryConv1d,
    TernaryConv2d,
    TernaryEmbedding,
    TernaryLinear,
)
from .config import TernaryConfig
from .conversion import PreparedModel
from .coverage import CoverageReport
from .errors import TritiumError
from .projection import ProjectionContext, validate_projection


@dataclass(frozen=True)
class QatHardConsumer:
    """One module consumer bound to a packed QAT-hard weight."""

    alias: str
    kind: str
    contract_id: str
    module_group: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alias": self.alias,
            "kind": self.kind,
            "contract_id": self.contract_id,
            "module_group": self.module_group,
        }


@dataclass(frozen=True)
class QatHardWeight:
    """Identity of one unique latent master frozen into compact planes."""

    path: str
    consumers: Tuple[QatHardConsumer, ...]
    storage_path: str
    shape: Tuple[int, int]
    algorithm_id: str
    planes: int

    @property
    def aliases(self) -> Tuple[str, ...]:
        return tuple(consumer.alias for consumer in self.consumers)

    @property
    def consumer_kinds(self) -> Tuple[str, ...]:
        return tuple(consumer.kind for consumer in self.consumers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "consumers": [consumer.to_dict() for consumer in self.consumers],
            "storage_path": self.storage_path,
            "shape": list(self.shape),
            "algorithm_id": self.algorithm_id,
            "planes": self.planes,
        }


@dataclass(frozen=True, eq=False)
class QatHardResult:
    """Consumed QAT graph with no targeted floating master or estimator state."""

    model: nn.Module
    artifact_id: str
    source_checkpoint_digest: str
    hard_state_digest: str
    recipe_id: str
    config: TernaryConfig
    source_coverage: CoverageReport
    weights: Tuple[QatHardWeight, ...]
    mode: str = "qat-hard"
    schema_version: int = 2

    def export(self, output_dir):
        """Atomically publish this hard result as a strict state bundle."""

        from .qat_artifacts import export_qat_hard

        return export_qat_hard(self, output_dir)


@dataclass(frozen=True)
class _Consumer:
    path: str
    module: nn.Module


@dataclass(frozen=True)
class _WeightGroup:
    path: str
    aliases: Tuple[str, ...]
    consumers: Tuple[_Consumer, ...]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _qat_hard_ids(
    *,
    source_checkpoint_digest: str,
    hard_state_digest: str,
    config: TernaryConfig,
    source_coverage: CoverageReport,
    weights: Tuple[QatHardWeight, ...],
) -> Tuple[str, str]:
    recipe_value = {
        "schema_version": 2,
        "mode": "qat-hard",
        "source_checkpoint_digest": source_checkpoint_digest,
        "config": config.to_dict(),
        "source_coverage": source_coverage.to_dict(),
        "weights": [receipt.to_dict() for receipt in weights],
    }
    recipe_id = _digest(recipe_value)
    artifact_id = _digest(
        {
            **recipe_value,
            "recipe_id": recipe_id,
            "hard_state_digest": hard_state_digest,
        }
    )
    return recipe_id, artifact_id


def _parent_and_child(root: nn.Module, path: str):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent._modules[part]
    return parent, parts[-1]


def _groups(prepared: PreparedModel) -> Tuple[_WeightGroup, ...]:
    model = prepared.model
    coverage = prepared.coverage
    if not isinstance(model, nn.Module) or coverage is None:
        raise TritiumError(
            "QAT hard conversion requires a live prepared module graph",
            code="invalid_phase",
            stage="convert_qat_hard",
        )
    by_weight: Dict[int, list[_Consumer]] = {}
    modules_by_weight: Dict[int, nn.Module] = {}
    for path, module in model.named_modules(remove_duplicate=False):
        if not isinstance(
            module,
            (TernaryLinear, TernaryEmbedding, TernaryConv1d, TernaryConv2d),
        ):
            continue
        if isinstance(module, TernaryEmbedding) and module.max_norm is not None:
            raise TritiumError(
                "QAT hard conversion cannot preserve mutating Embedding max_norm",
                code="unsupported_module_option",
                stage="convert_qat_hard",
                module=path,
            )
        weight_id = id(module.weight)
        by_weight.setdefault(weight_id, []).append(_Consumer(path, module))
        modules_by_weight.setdefault(weight_id, module)
        if modules_by_weight[weight_id].estimator is not module.estimator:
            raise TritiumError(
                "tied QAT masters must share one estimator instance",
                code="coverage_mismatch",
                stage="convert_qat_hard",
                module=path,
            )
    converted = {
        id(parameter): entry
        for entry in coverage.entries
        if entry.disposition == "converted"
        for alias, parameter in model.named_parameters(remove_duplicate=False)
        if alias in entry.aliases
    }
    if not by_weight or set(by_weight) != set(converted):
        raise TritiumError(
            "QAT hard targets differ from prepared coverage",
            code="coverage_mismatch",
            stage="convert_qat_hard",
        )
    groups = []
    for weight_id, consumers in by_weight.items():
        entry = converted[weight_id]
        if set(entry.aliases) != {
            f"{consumer.path}.weight" if consumer.path else "weight"
            for consumer in consumers
        }:
            raise TritiumError(
                "QAT hard aliases differ from prepared coverage",
                code="coverage_mismatch",
                stage="convert_qat_hard",
                module=entry.path,
            )
        consumers_by_alias = {
            f"{consumer.path}.weight" if consumer.path else "weight": consumer
            for consumer in consumers
        }
        groups.append(
            _WeightGroup(
                path=entry.path,
                aliases=entry.aliases,
                consumers=tuple(
                    consumers_by_alias[alias] for alias in entry.aliases
                ),
            )
        )
    return tuple(groups)


def _replacement(
    module: nn.Module,
    packed_weight: AdditiveTernaryWeight,
    *,
    owner: bool,
) -> nn.Module:
    if isinstance(module, TernaryLinear):
        bias = module.bias.detach() if module.bias is not None else None
        return AdditiveTernaryLinear.from_packed_weight(
            packed_weight,
            bias,
            owner=owner,
        )
    if isinstance(module, TernaryEmbedding):
        return AdditiveTernaryEmbedding(
            packed_weight,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
            dtype=module.weight.dtype,
            owner=owner,
        )
    if isinstance(module, TernaryConv1d):
        return AdditiveTernaryConv1d.from_packed_weight(
            module,
            packed_weight,
            owner=owner,
        )
    if isinstance(module, TernaryConv2d):
        return AdditiveTernaryConv2d.from_packed_weight(
            module,
            packed_weight,
            owner=owner,
        )
    raise TypeError("QAT hard replacement requires a supported ternary module")


def _matrix_weight(module: nn.Module) -> torch.Tensor:
    weight = module.weight
    if isinstance(module, (TernaryConv1d, TernaryConv2d)):
        return weight.flatten(start_dim=1)
    return weight


def _consumer_kind(module: nn.Module) -> str:
    if isinstance(module, (nn.Linear, TernaryLinear)):
        return "linear"
    if isinstance(module, (nn.Embedding, TernaryEmbedding)):
        return "embedding"
    if isinstance(module, (nn.Conv1d, TernaryConv1d)):
        return "conv1d"
    if isinstance(module, (nn.Conv2d, TernaryConv2d)):
        return "conv2d"
    raise TypeError("unsupported QAT hard consumer")


def _consumer_contract_id(module: nn.Module) -> str:
    kind = _consumer_kind(module)
    contract: Dict[str, Any] = {
        "schema_version": 1,
        "kind": kind,
        "weight_dtype": str(module.weight.dtype),
        "weight_shape": list(module.weight.shape),
    }
    if kind == "linear":
        contract.update(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
        )
    elif kind == "embedding":
        contract.update(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
        )
    else:
        contract.update(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=list(module.kernel_size),
            stride=list(module.stride),
            padding=(
                module.padding
                if isinstance(module.padding, str)
                else list(module.padding)
            ),
            dilation=list(module.dilation),
            transposed=module.transposed,
            output_padding=list(module.output_padding),
            groups=module.groups,
            padding_mode=module.padding_mode,
            bias=module.bias is not None,
        )
    return _digest(contract)


def _state_tensor_identity(tensor: torch.Tensor):
    storage = tensor.untyped_storage()
    return (
        str(tensor.device),
        storage._cdata,
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
    )


def _state_slot(model: nn.Module, name: str):
    parts = name.split(".")
    module = model
    for part in parts[:-1]:
        child = module._modules.get(part)
        if child is None:
            raise ValueError("QAT-hard state differs from model shell")
        module = child
    leaf = parts[-1]
    parameter = module._parameters.get(leaf)
    buffer = module._buffers.get(leaf)
    if parameter is not None and buffer is None:
        return module, "parameter", leaf, parameter
    if buffer is not None and parameter is None:
        return module, "buffer", leaf, buffer
    raise ValueError("QAT-hard state differs from model shell")


def _tied_state_apply(model: nn.Module, fn, recurse: bool = True):
    base_apply = object.__getattribute__(model, "_tritium_base_apply")
    result = base_apply(model, fn, recurse)
    aliases = object.__getattribute__(model, "_tritium_tied_state_aliases")
    for group in aliases:
        slots = [_state_slot(model, name) for name in group]
        parameter_slots = [slot for slot in slots if slot[1] == "parameter"]
        value = (parameter_slots or slots)[0][3]
        for module, kind, leaf, _ in slots:
            if kind == "parameter":
                module._parameters[leaf] = value
            else:
                module._buffers[leaf] = value
    return result


def _install_tie_aware_apply(model: nn.Module) -> None:
    groups = {}
    for name, tensor in model.state_dict(keep_vars=True).items():
        groups.setdefault(_state_tensor_identity(tensor), []).append(name)
    aliases = tuple(
        tuple(sorted(names))
        for names in groups.values()
        if len(names) > 1
    )
    if not aliases:
        return
    object.__setattr__(model, "_tritium_tied_state_aliases", aliases)
    object.__setattr__(model, "_tritium_base_apply", type(model)._apply)
    object.__setattr__(model, "_apply", MethodType(_tied_state_apply, model))


def _canonical_hard_state_items(
    model: nn.Module,
) -> Tuple[Tuple[str, torch.Tensor], ...]:
    """Select lexicographically canonical names for exact shared tensors."""

    return tuple(
        (name, tensor)
        for name, tensor, _ in _canonical_hard_state_groups(model)
    )


def _canonical_hard_state_groups(model: nn.Module):
    """Select canonical tensors plus every exact state alias."""

    canonical = {}
    for name, tensor in sorted(model.state_dict().items()):
        entry = canonical.setdefault(
            _state_tensor_identity(tensor),
            [name, tensor, []],
        )
        entry[2].append(name)
    return tuple(
        (name, tensor, tuple(aliases))
        for name, tensor, aliases in sorted(canonical.values())
    )


def _hard_state_digest(model: nn.Module) -> str:
    """Hash hard state once per exact shared tensor under its canonical alias."""

    from .ptq import _hash_tensor

    digest = hashlib.sha256()
    for name, tensor, aliases in _canonical_hard_state_groups(model):
        digest.update(_canonical({"aliases": aliases}))
        _hash_tensor(digest, f"state.{name}", tensor)
    return f"sha256:{digest.hexdigest()}"


def convert_qat_hard(prepared: PreparedModel) -> QatHardResult:
    """Consume one prepared QAT graph into a distinct compact hard result."""

    if not isinstance(prepared, PreparedModel):
        raise TypeError("convert_qat_hard requires a PreparedModel")
    if prepared.config.mode != "qat":
        raise TritiumError(
            "QAT hard conversion requires TernaryConfig.qat",
            code="invalid_phase",
            stage="convert_qat_hard",
        )
    groups = _groups(prepared)
    from .ptq import _source_model_digest

    source_digest = _source_model_digest(prepared.model)
    compact_weights = {}
    receipts = []
    with torch.no_grad():
        for group in groups:
            first = group.consumers[0].module
            matrix_weight = _matrix_weight(first)
            projection = first.estimator.project(
                matrix_weight,
                context=ProjectionContext(
                    step=first.estimator.projection_step,
                    training=False,
                    role="weight",
                ),
            )
            validate_projection(
                projection,
                matrix_weight,
                algorithm_id=first.estimator.algorithm_id,
                schema_version=first.estimator.schema_version,
            )
            if not projection.exportable:
                raise TritiumError(
                    "QAT hard conversion requires an exportable projection; "
                    "HESTIA must reach its temperature floor",
                    code="invalid_phase",
                    stage="convert_qat_hard",
                    module=group.path,
                )
            if len(projection.planes) != prepared.config.planes:
                raise TritiumError(
                    "QAT hard projection plane count differs from recipe",
                    code="estimator_contract",
                    stage="convert_qat_hard",
                    module=group.path,
                )
            if any(
                plane.scales.dtype != torch.float16
                or tuple(plane.scales.shape) != (matrix_weight.shape[0], 1)
                or plane.group_size != matrix_weight.shape[1]
                or plane.structure != "dense"
                for plane in projection.planes
            ):
                raise TritiumError(
                    "QAT hard projection is not representable by dense row-scale SALT",
                    code="unsupported_projection_geometry",
                    stage="convert_qat_hard",
                    module=group.path,
                )
            packed = AdditiveTernaryWeight(projection.planes).to(first.weight.device)
            compact_weights[id(first.weight)] = packed
            module_groups: Dict[int, int] = {}
            hard_consumers = []
            for alias, consumer in zip(group.aliases, group.consumers):
                module_group = module_groups.setdefault(
                    id(consumer.module), len(module_groups)
                )
                hard_consumers.append(
                    QatHardConsumer(
                        alias=alias,
                        kind=_consumer_kind(consumer.module),
                        contract_id=_consumer_contract_id(consumer.module),
                        module_group=module_group,
                    )
                )
            receipts.append(
                QatHardWeight(
                    path=group.path,
                    consumers=tuple(hard_consumers),
                    storage_path=(
                        f"{group.consumers[0].path}._packed_weight"
                        if group.consumers[0].path
                        else "_packed_weight"
                    ),
                    shape=(packed.out_features, packed.in_features),
                    algorithm_id=projection.algorithm_id,
                    planes=packed.plane_count,
                )
            )

    replacements = []
    root_replacement = None
    shared_biases: Dict[int, torch.Tensor] = {}
    for group in groups:
        packed = compact_weights[id(group.consumers[0].module.weight)]
        replacement_by_module = {}
        owns_weight = True
        for consumer in group.consumers:
            replacement = replacement_by_module.get(id(consumer.module))
            if replacement is None:
                replacement = _replacement(
                    consumer.module,
                    packed,
                    owner=owns_weight,
                )
                source_bias = getattr(consumer.module, "bias", None)
                if source_bias is not None:
                    bias = shared_biases.setdefault(
                        id(source_bias),
                        source_bias,
                    )
                    replacement._buffers["bias"] = bias
                owns_weight = False
                replacement_by_module[id(consumer.module)] = replacement
            if consumer.path:
                replacements.append((consumer.path, consumer.module, replacement))
            else:
                root_replacement = replacement

    applied = []
    try:
        for path, original, replacement in replacements:
            parent, child = _parent_and_child(prepared.model, path)
            if parent._modules.get(child) is not original:
                raise RuntimeError("QAT module graph changed during hard conversion")
            parent._modules[child] = replacement
            applied.append((parent, child, original))
    except BaseException as error:
        for parent, child, original in reversed(applied):
            parent._modules[child] = original
        raise TritiumError(
            "QAT hard graph conversion failed and was rolled back",
            code="conversion_failed",
            stage="convert_qat_hard",
        ) from error

    model = root_replacement if root_replacement is not None else prepared.model
    model.eval()
    _install_tie_aware_apply(model)
    hard_digest = _hard_state_digest(model)
    recipe_id, artifact_id = _qat_hard_ids(
        source_checkpoint_digest=source_digest,
        hard_state_digest=hard_digest,
        config=prepared.config,
        source_coverage=prepared.coverage,
        weights=tuple(receipts),
    )
    model._tritium_qat_hard_artifact_id = artifact_id
    model._tritium_qat_checkpoint_digest = source_digest
    if hasattr(model, "config"):
        model.config.tritium_qat_hard_artifact_id = artifact_id
        model.config.tritium_qat_checkpoint_digest = source_digest
    return QatHardResult(
        model=model,
        artifact_id=artifact_id,
        source_checkpoint_digest=source_digest,
        hard_state_digest=hard_digest,
        recipe_id=recipe_id,
        config=prepared.config,
        source_coverage=prepared.coverage,
        weights=tuple(receipts),
    )


__all__ = [
    "QatHardConsumer",
    "QatHardResult",
    "QatHardWeight",
    "convert_qat_hard",
]
