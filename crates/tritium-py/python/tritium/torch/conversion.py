"""Transactional model inspection and reference QAT conversion."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from torch import nn

from .config import TernaryConfig
from .coverage import CoverageEntry, CoverageReport
from .errors import TritiumError
from .estimators import Estimator, create_estimator


@dataclass(frozen=True)
class _Replacement:
    path: str
    original: nn.Module
    converted: nn.Module


@dataclass(frozen=True)
class PreparedModel:
    """Validated model plus immutable recipe and graph-coverage receipt."""

    model: nn.Module
    config: TernaryConfig
    coverage: CoverageReport
    schema_version: int = 1


def _new_estimator(config: TernaryConfig) -> Estimator:
    if config.planes != 1:
        raise TritiumError(
            "the QAT estimator catalog currently supports one hard ternary plane",
            code="unsupported_recipe",
            stage="inspect",
            details={"planes": config.planes},
        )
    return create_estimator(config.estimator)


def _parameter_coverage(
    model: nn.Module,
    target_weights: set,
    *,
    target_disposition: str = "converted",
) -> CoverageReport:
    by_identity: "OrderedDict[int, Tuple[object, List[str]]]" = OrderedDict()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        key = id(parameter)
        if key not in by_identity:
            by_identity[key] = (parameter, [])
        by_identity[key][1].append(name)

    entries = []
    for parameter, aliases in by_identity.values():
        alias_set = set(aliases)
        target_aliases = alias_set & target_weights
        preserved_aliases = alias_set - target_weights
        if target_aliases and preserved_aliases:
            raise TritiumError(
                "a shared parameter has both targeted and preserved owners",
                code="coverage_conflict",
                stage="inspect",
                module=aliases[0],
                details={"aliases": aliases},
            )
        disposition = target_disposition if target_aliases else "preserved"
        if target_aliases:
            reason = (
                "ptq_target" if target_disposition == "selected" else "target_weight"
            )
        elif any(
            ".estimator." in alias or alias.startswith("estimator.")
            for alias in aliases
        ):
            reason = "estimator_state"
        elif any(alias.endswith(".bias") or alias == "bias" for alias in aliases):
            reason = "bias_preserved"
        else:
            reason = "not_targeted"
        entries.append(
            CoverageEntry(
                path=aliases[0],
                aliases=tuple(aliases),
                disposition=disposition,
                reason=reason,
                numel=parameter.numel(),
                logical_bytes=parameter.numel() * parameter.element_size(),
            )
        )
    return CoverageReport.new(entries)


def _selected_weights(
    model: nn.Module, config: TernaryConfig, *, strict: bool
) -> set:
    supported_targets = {"Conv1d", "Conv2d", "Embedding", "Linear"}
    unknown_targets = set(config.target_modules) - supported_targets
    if unknown_targets:
        raise TritiumError(
            "configuration selects unsupported module types",
            code="unsupported_module",
            stage="inspect",
            details={"targets": sorted(unknown_targets)},
        )

    selected = set()
    module_types = {
        "Conv1d": nn.Conv1d,
        "Conv2d": nn.Conv2d,
        "Embedding": nn.Embedding,
        "Linear": nn.Linear,
    }
    for path, module in model.named_modules(remove_duplicate=False):
        if any(
            target in config.target_modules and isinstance(module, module_type)
            for target, module_type in module_types.items()
        ):
            selected.add(f"{path}.weight" if path else "weight")
    if strict and not selected:
        raise TritiumError(
            "no modules matched the requested targets",
            code="incomplete_coverage",
            stage="inspect",
        )
    _parameter_coverage(model, selected)
    return selected


def _parent_and_child(root: nn.Module, path: str):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent._modules[part]
    return parent, parts[-1]


def _prepare_qat_inplace(
    model: nn.Module, config: TernaryConfig, *, strict: bool = True
) -> nn.Module:
    """Internal validated in-place QAT graph conversion."""

    from ..nn import TernaryConv1d, TernaryConv2d, TernaryEmbedding, TernaryLinear

    if not isinstance(model, nn.Module):
        raise TypeError("prepare_qat requires a torch.nn.Module")
    if config.mode != "qat":
        raise TritiumError(
            "prepare_qat requires TernaryConfig.qat",
            code="invalid_config",
            stage="inspect",
        )
    _selected_weights(model, config, strict=strict)

    replacements: List[_Replacement] = []
    converted_weights = set()
    converted_modules: Dict[int, nn.Module] = {}
    estimators_by_weight: Dict[int, Estimator] = {}
    for path, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, nn.Linear) and "Linear" in config.target_modules:
            converted = converted_modules.get(id(module))
            if converted is None:
                estimator = estimators_by_weight.get(id(module.weight))
                if estimator is None:
                    estimator = _new_estimator(config)
                    estimators_by_weight[id(module.weight)] = estimator
                converted = TernaryLinear.from_float(module, estimator=estimator)
                converted_modules[id(module)] = converted
            replacements.append(_Replacement(path, module, converted))
            converted_weights.add(f"{path}.weight" if path else "weight")
        elif isinstance(module, nn.Embedding) and "Embedding" in config.target_modules:
            converted = converted_modules.get(id(module))
            if converted is None:
                estimator = estimators_by_weight.get(id(module.weight))
                if estimator is None:
                    estimator = _new_estimator(config)
                    estimators_by_weight[id(module.weight)] = estimator
                converted = TernaryEmbedding.from_float(module, estimator=estimator)
                converted_modules[id(module)] = converted
            replacements.append(_Replacement(path, module, converted))
            converted_weights.add(f"{path}.weight" if path else "weight")
        elif isinstance(module, nn.Conv1d) and "Conv1d" in config.target_modules:
            converted = converted_modules.get(id(module))
            if converted is None:
                estimator = estimators_by_weight.get(id(module.weight))
                if estimator is None:
                    estimator = _new_estimator(config)
                    estimators_by_weight[id(module.weight)] = estimator
                converted = TernaryConv1d.from_float(module, estimator=estimator)
                converted_modules[id(module)] = converted
            replacements.append(_Replacement(path, module, converted))
            converted_weights.add(f"{path}.weight" if path else "weight")
        elif isinstance(module, nn.Conv2d) and "Conv2d" in config.target_modules:
            converted = converted_modules.get(id(module))
            if converted is None:
                estimator = estimators_by_weight.get(id(module.weight))
                if estimator is None:
                    estimator = _new_estimator(config)
                    estimators_by_weight[id(module.weight)] = estimator
                converted = TernaryConv2d.from_float(module, estimator=estimator)
                converted_modules[id(module)] = converted
            replacements.append(_Replacement(path, module, converted))
            converted_weights.add(f"{path}.weight" if path else "weight")

    if strict and not replacements:
        raise TritiumError(
            "no modules matched the requested QAT targets",
            code="incomplete_coverage",
            stage="inspect",
        )

    _parameter_coverage(model, converted_weights)

    root_replacement = next((item for item in replacements if not item.path), None)
    if root_replacement is not None:
        result = root_replacement.converted
    else:
        applied: List[Tuple[nn.Module, str, nn.Module]] = []
        try:
            for replacement in replacements:
                parent, child = _parent_and_child(model, replacement.path)
                original = parent._modules[child]
                parent._modules[child] = replacement.converted
                applied.append((parent, child, original))
        except Exception as error:
            for parent, child, original in reversed(applied):
                parent._modules[child] = original
            raise TritiumError(
                "module conversion failed and was rolled back",
                code="conversion_failed",
                stage="convert",
            ) from error
        result = model

    result._tritium_coverage = _parameter_coverage(result, converted_weights)
    if hasattr(result, "config"):
        # Hugging Face serializes this exact dictionary into config.json. The
        # quantizer registration is optional and imported only when
        # transformers is installed.
        from .hf import attach_huggingface_recipe

        attach_huggingface_recipe(result, config)
    return result


def prepare(
    model: nn.Module,
    config: TernaryConfig,
    *,
    strict: bool = True,
    inplace: bool,
) -> PreparedModel:
    """Validate a complete graph, then prepare it with explicit ownership."""

    if not isinstance(model, nn.Module):
        raise TypeError("prepare requires a torch.nn.Module")
    if not isinstance(config, TernaryConfig):
        raise TypeError("prepare requires a TernaryConfig")
    if not isinstance(inplace, bool):
        raise TypeError("prepare inplace must be a bool")

    selected = _selected_weights(model, config, strict=strict)
    target = model if inplace else copy.deepcopy(model)
    if config.mode == "qat":
        target = _prepare_qat_inplace(target, config, strict=strict)
        coverage = inspect(target)
    else:
        if not inplace:
            selected = _selected_weights(target, config, strict=strict)
        coverage = _parameter_coverage(
            target,
            selected,
            target_disposition="selected",
        )
        target._tritium_coverage = coverage
    return PreparedModel(model=target, config=config, coverage=coverage)


def prepare_qat(
    model: nn.Module, config: TernaryConfig, *, strict: bool = True
) -> nn.Module:
    """Compatibility facade over :func:`prepare` using in-place ownership."""

    if isinstance(config, TernaryConfig) and config.mode != "qat":
        raise TritiumError(
            "prepare_qat requires TernaryConfig.qat",
            code="invalid_config",
            stage="inspect",
        )
    return prepare(model, config, strict=strict, inplace=True).model


def inspect(model: Union[nn.Module, PreparedModel]) -> CoverageReport:
    """Return the immutable coverage receipt for a prepared model."""

    if isinstance(model, PreparedModel):
        return model.coverage

    report = getattr(model, "_tritium_coverage", None)
    if not isinstance(report, CoverageReport):
        raise TritiumError(
            "model has no Tritium conversion coverage receipt",
            code="coverage_missing",
            stage="inspect",
        )
    return report
