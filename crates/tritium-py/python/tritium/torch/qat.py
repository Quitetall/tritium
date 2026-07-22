"""Hard, inference-only conversion of differentiable QAT module graphs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import nn

from ..nn import (
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
class QatHardWeight:
    """Identity of one unique latent master frozen into compact planes."""

    path: str
    aliases: Tuple[str, ...]
    storage_path: str
    shape: Tuple[int, int]
    algorithm_id: str
    planes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "aliases": list(self.aliases),
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
    schema_version: int = 1

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
        "schema_version": 1,
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
    unsupported = [
        path
        for path, module in model.named_modules(remove_duplicate=False)
        if isinstance(module, (TernaryConv1d, TernaryConv2d))
    ]
    if unsupported:
        raise TritiumError(
            "QAT hard conversion does not yet support ternary convolutions",
            code="unsupported_module",
            stage="convert_qat_hard",
            details={"modules": unsupported},
        )
    by_weight: Dict[int, list[_Consumer]] = {}
    modules_by_weight: Dict[int, nn.Module] = {}
    for path, module in model.named_modules(remove_duplicate=False):
        if not isinstance(module, (TernaryLinear, TernaryEmbedding)):
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
        groups.append(
            _WeightGroup(
                path=entry.path,
                aliases=entry.aliases,
                consumers=tuple(consumers),
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
    raise TypeError("QAT hard replacement requires a ternary Linear or Embedding")


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
            projection = first.estimator.project(
                first.weight,
                context=ProjectionContext(training=False, role="weight"),
            )
            validate_projection(
                projection,
                first.weight,
                algorithm_id=first.estimator.algorithm_id,
                schema_version=first.estimator.schema_version,
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
                or tuple(plane.scales.shape) != (first.weight.shape[0], 1)
                or plane.group_size != first.weight.shape[1]
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
            receipts.append(
                QatHardWeight(
                    path=group.path,
                    aliases=group.aliases,
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
    hard_digest = _source_model_digest(model)
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


__all__ = ["QatHardResult", "QatHardWeight", "convert_qat_hard"]
