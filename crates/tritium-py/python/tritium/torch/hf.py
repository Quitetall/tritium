"""External Hugging Face quantizer registration for Tritium QAT checkpoints."""

from __future__ import annotations

from typing import Any

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
        return True

    @property
    def is_qat_trainable(self):
        return True

    @property
    def is_compileable(self):
        return True

    def is_serializable(self, safe_serialization=None):
        del safe_serialization
        return True

    def _process_model_before_weight_loading(self, model, **kwargs):
        del kwargs
        if self.quantization_config.recipe.mode != "qat":
            raise TritiumError(
                "Hugging Face dense checkpoints currently support QAT recipes only",
                code="unsupported_recipe",
                stage="load",
            )
        from .conversion import prepare_qat

        return prepare_qat(model, self.quantization_config.recipe)

    def _process_model_after_weight_loading(self, model, **kwargs):
        del kwargs
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


def attach_huggingface_recipe(model, config: TernaryConfig) -> None:
    """Bind a recipe to a converted PreTrainedModel for native HF saving."""

    register_huggingface()
    model.config.quantization_config = HfTritiumConfig(**config.to_dict()).to_dict()
