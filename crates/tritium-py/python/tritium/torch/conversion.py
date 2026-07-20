"""Transactional model inspection and reference QAT conversion."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


def _new_estimator(config: TernaryConfig) -> Estimator:
    if config.planes != 1:
        raise TritiumError(
            "the QAT estimator catalog currently supports one hard ternary plane",
            code="unsupported_recipe",
            stage="inspect",
            details={"planes": config.planes},
        )
    return create_estimator(config.estimator)


def _parameter_coverage(model: nn.Module, converted_weights: set) -> CoverageReport:
    by_identity: "OrderedDict[int, Tuple[object, List[str]]]" = OrderedDict()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        key = id(parameter)
        if key not in by_identity:
            by_identity[key] = (parameter, [])
        by_identity[key][1].append(name)

    entries = []
    for parameter, aliases in by_identity.values():
        alias_set = set(aliases)
        converted_aliases = alias_set & converted_weights
        preserved_aliases = alias_set - converted_weights
        if converted_aliases and preserved_aliases:
            raise TritiumError(
                "a shared parameter has both converted and preserved owners",
                code="coverage_conflict",
                stage="inspect",
                module=aliases[0],
                details={"aliases": aliases},
            )
        disposition = "converted" if converted_aliases else "preserved"
        if converted_aliases:
            reason = "target_weight"
        elif any(".estimator." in alias or alias.startswith("estimator.") for alias in aliases):
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


def _parent_and_child(root: nn.Module, path: str):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent._modules[part]
    return parent, parts[-1]


def prepare_qat(
    model: nn.Module, config: TernaryConfig, *, strict: bool = True
) -> nn.Module:
    """Validate then convert selected modules without cloning master parameters."""

    from ..nn import TernaryEmbedding, TernaryLinear

    if not isinstance(model, nn.Module):
        raise TypeError("prepare_qat requires a torch.nn.Module")
    if config.mode != "qat":
        raise TritiumError(
            "prepare_qat requires TernaryConfig.qat",
            code="invalid_config",
            stage="inspect",
        )
    supported_targets = {"Embedding", "Linear"}
    unknown_targets = set(config.target_modules) - supported_targets
    if unknown_targets:
        raise TritiumError(
            "configuration selects unsupported module types",
            code="unsupported_module",
            stage="inspect",
            details={"targets": sorted(unknown_targets)},
        )

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


def inspect(model: nn.Module) -> CoverageReport:
    """Return the immutable conversion receipt attached by :func:`prepare_qat`."""

    report = getattr(model, "_tritium_coverage", None)
    if not isinstance(report, CoverageReport):
        raise TritiumError(
            "model has no Tritium conversion coverage receipt",
            code="coverage_missing",
            stage="inspect",
        )
    return report
