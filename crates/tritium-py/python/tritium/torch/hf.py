"""External Hugging Face quantizer registration for Tritium QAT checkpoints."""

from __future__ import annotations

from typing import Any

from torch import nn

from .config import TernaryConfig
from .errors import TritiumError

try:
    from transformers.quantizers.auto import (
        AUTO_QUANTIZATION_CONFIG_MAPPING,
        AUTO_QUANTIZER_MAPPING,
    )
    from transformers.quantizers.base import HfQuantizer
    from transformers import Trainer
    from transformers.utils.quantization_config import QuantizationConfigMixin
except ImportError as error:  # pragma: no cover - exercised in torch-only wheels
    raise ImportError("Hugging Face integration requires transformers") from error


class HfTritiumConfig(QuantizationConfigMixin):
    """Hugging Face wrapper around the language-neutral Tritium recipe."""

    quant_method = "tritium"

    def __init__(self, quant_method: str = "tritium", **recipe: Any) -> None:
        if quant_method != self.quant_method:
            raise ValueError("HfTritiumConfig quant_method must be 'tritium'")
        self.recipe = TernaryConfig.from_dict(recipe)

    def to_dict(self):
        return {"quant_method": self.quant_method, **self.recipe.to_dict()}

    def to_diff_dict(self):
        return self.to_dict()


class TritiumHfQuantizer(HfQuantizer):
    """Build Tritium QAT modules before Hugging Face loads their tensors."""

    requires_calibration = False

    @property
    def is_trainable(self):
        return self.quantization_config.recipe.mode == "qat"

    @property
    def is_qat_trainable(self):
        return self.quantization_config.recipe.mode == "qat"

    @property
    def is_compileable(self):
        return True

    def is_serializable(self, safe_serialization=None):
        del safe_serialization
        return True

    def _process_model_before_weight_loading(self, model, **kwargs):
        del kwargs
        recipe = self.quantization_config.recipe
        if recipe.mode == "qat":
            from .conversion import prepare_qat

            return prepare_qat(model, recipe)
        return _prepare_ptq_inference_shell(model, recipe)

    def _process_model_after_weight_loading(self, model, **kwargs):
        del kwargs
        if self.quantization_config.recipe.mode == "ptq":
            from ..nn import AdditiveTernaryLinear

            for module in model.modules():
                if isinstance(module, AdditiveTernaryLinear):
                    module.validate_buffers()
            expected = getattr(model.config, "tritium_ptq_checkpoint_digest", None)
            if not isinstance(expected, str) or not expected.startswith("sha256:"):
                raise TritiumError(
                    "Hugging Face PTQ checkpoint identity is missing",
                    code="state_identity",
                    stage="load",
                )
            from .ptq import _source_model_digest

            observed = _source_model_digest(model)
            if observed != expected:
                raise TritiumError(
                    "Hugging Face PTQ checkpoint identity changed",
                    code="state_identity",
                    stage="load",
                    details={"expected": expected, "observed": observed},
                )
            model.requires_grad_(False)
            model.eval()
        return model


class TritiumTrainer(Trainer):
    """Thin HF Trainer facade that converts before optimizer construction."""

    def __init__(self, *args, tritium_config: TernaryConfig | None = None, **kwargs):
        model = kwargs.get("model", args[0] if args else None)
        if model is None:
            raise ValueError("TritiumTrainer requires a model")
        if tritium_config is not None:
            from .conversion import prepare_qat

            model = prepare_qat(model, tritium_config)
            if args:
                args = (model, *args[1:])
            else:
                kwargs["model"] = model
        else:
            from .conversion import inspect

            inspect(model)
        super().__init__(*args, **kwargs)


def register_huggingface() -> None:
    """Register idempotently with the installed Transformers process."""

    existing_config = AUTO_QUANTIZATION_CONFIG_MAPPING.get("tritium")
    existing_quantizer = AUTO_QUANTIZER_MAPPING.get("tritium")
    if existing_config not in {None, HfTritiumConfig}:
        raise RuntimeError("another package registered Hugging Face quant_method='tritium'")
    if existing_quantizer not in {None, TritiumHfQuantizer}:
        raise RuntimeError("another package registered the Tritium HF quantizer")
    AUTO_QUANTIZATION_CONFIG_MAPPING["tritium"] = HfTritiumConfig
    AUTO_QUANTIZER_MAPPING["tritium"] = TritiumHfQuantizer


def _prepare_ptq_inference_shell(model, recipe: TernaryConfig):
    """Replace exact Linear modules with meta-safe compact state shells."""

    unsupported = set(recipe.target_modules) - {"Linear"}
    if unsupported or "Linear" not in recipe.target_modules:
        raise TritiumError(
            "generic Hugging Face PTQ reload currently supports Linear targets only",
            code="unsupported_recipe",
            stage="load",
            details={"targets": sorted(recipe.target_modules)},
        )
    from ..nn import AdditiveTernaryLinear

    replacements = {}
    pending = []
    for path, module in model.named_modules(remove_duplicate=False):
        if type(module) is not nn.Linear:
            continue
        if not path:
            raise TritiumError(
                "Hugging Face PTQ root must be a PreTrainedModel, not Linear",
                code="unsupported_model",
                stage="load",
            )
        replacement = replacements.get(id(module))
        if replacement is None:
            replacement = AdditiveTernaryLinear.empty(
                module.in_features,
                module.out_features,
                recipe.planes,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            replacements[id(module)] = replacement
        pending.append((path, replacement))
    if not pending:
        raise TritiumError(
            "Hugging Face PTQ recipe selected no exact Linear modules",
            code="incomplete_coverage",
            stage="load",
        )
    for path, replacement in pending:
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent._modules[part]
        parent._modules[parts[-1]] = replacement
    return model


def attach_huggingface_recipe(model, config: TernaryConfig) -> None:
    """Bind a recipe to a converted PreTrainedModel for native HF saving."""

    register_huggingface()
    model.config.quantization_config = HfTritiumConfig(**config.to_dict()).to_dict()
