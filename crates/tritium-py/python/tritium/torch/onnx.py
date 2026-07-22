"""Whole-Qwen ONNX export and load façade.

The public boundary accepts deployable hard artifacts only. Latent QAT masters,
optimizer state, and training graphs remain a v1.3 feature.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .. import _tritium
from .artifacts import QuantizationResult
from .conversion import PreparedModel
from .errors import TritiumError
from .module_artifacts import ModuleQuantizationResult
from .module_onnx import ModuleOnnxLineage, export_module_onnx, load_module_onnx
from .ptq import load_quantized_module
from .qat import QatHardResult
from .qat_artifacts import QatHardArtifact, load_qat_hard
from .refinement import RefinementResult

_MANIFEST_FILE = "tritium-onnx-manifest.json"
_TOP_FIELDS_V1 = {"schema", "language", "mtp", "weights", "identity", "conversion"}
_TOP_FIELDS_V2 = _TOP_FIELDS_V1 | {"sequence_mode"}
_GRAPH_FIELDS = {"file", "blake3"}
_WEIGHT_FIELDS = {"file", "blake3", "bytes"}
_IDENTITY_FIELDS = {
    "source_model_id",
    "tokenizer_id",
    "recipe_id",
    "package_id",
    "converted_coverage_id",
    "deferred_coverage_id",
}
_CONVERSION_FIELDS = {
    "mode",
    "completion_id",
    "campaign_id",
    "admission_id",
    "selection_id",
}
_TRAINING_CHECKPOINT_FILES = {
    "optimizer.pt",
    "optimizer.bin",
    "pytorch_model.bin",
    "model.safetensors",
    "trainer_state.json",
}


@dataclass(frozen=True)
class OnnxBundleManifest:
    """Strict, content-bound metadata for one language-plus-MTP ONNX bundle."""

    directory: Path
    language_blake3: str
    mtp_blake3: str
    weights_blake3: str
    weights_bytes: int
    sequence_mode: Optional[str]
    source_model_id: str
    tokenizer_id: str
    recipe_id: str
    package_id: str
    converted_coverage_id: str
    deferred_coverage_id: str
    conversion_mode: str
    completion_id: str
    campaign_id: str
    admission_id: str
    selection_id: str


@dataclass(frozen=True, eq=False)
class OnnxCausalLMOutput:
    """PyTorch-shaped output from one authenticated fixed-shape ONNX call."""

    logits: Tensor
    past_key_values: Tuple[Tensor, ...]
    state_names: Tuple[str, ...]


@dataclass(frozen=True, eq=False)
class OnnxMtpOutput:
    """Batch-one output from the authenticated Qwen MTP drafter graph."""

    logits: Tensor
    final_hidden: Tensor
    past_key_values: Tuple[Tensor, ...]
    state_names: Tuple[str, ...]


@dataclass(frozen=True)
class _OnnxConfig:
    model_type: str = "tritium_qwen35_onnx"
    is_encoder_decoder: bool = False
    is_decoder: bool = True
    use_cache: bool = True


class QwenOnnxCausalLM(nn.Module):
    """Inference-only PyTorch facade over Tritium's authenticated ORT runtime."""

    def __init__(self, runtime: Any, manifest: OnnxBundleManifest) -> None:
        super().__init__()
        self._runtime = runtime
        self.manifest = manifest
        self.config = _OnnxConfig()
        self.eval()

    @property
    def device(self) -> torch.device:
        return torch.device(self._runtime.device)

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: Optional[Tuple[Tensor, ...]] = None,
        attention_mask: Optional[Tensor] = None,
        *,
        use_cache: bool = True,
        return_dict: bool = True,
    ) -> Union[OnnxCausalLMOutput, Tuple[Tensor, Tuple[Tensor, ...]]]:
        tokens = _batch_one_tokens(input_ids, "input_ids")
        if attention_mask is not None:
            _validate_attention_mask(attention_mask, tokens.numel())
        states = _flatten_states(past_key_values)
        native = self._runtime.forward_language(
            tokens.to(dtype=torch.int64).contiguous().tolist(), states
        )
        logits = _batch_one_output(native.logits, native.logits_shape, "language logits")
        if len(native.states) != len(native.state_shapes):
            raise RuntimeError("native ONNX state values and shapes differ in count")
        output_states = tuple(
            _tensor_from_flat(values, state_shape, f"language state {index}")
            for index, (values, state_shape) in enumerate(
                zip(native.states, native.state_shapes)
            )
        )
        if not use_cache:
            output_states = ()
        output = OnnxCausalLMOutput(
            logits=logits,
            past_key_values=output_states,
            state_names=tuple(native.state_names),
        )
        return output if return_dict else (output.logits, output.past_key_values)

    def draft(
        self,
        shifted_input_ids: Tensor,
        target_hidden: Tensor,
        past_key_values: Optional[Tuple[Tensor, ...]] = None,
    ) -> OnnxMtpOutput:
        """Execute the bundled ternary MTP drafter for one fixed-shape step."""
        tokens = _batch_one_tokens(shifted_input_ids, "shifted_input_ids")
        if not isinstance(target_hidden, Tensor):
            raise TypeError("target_hidden must be a torch.Tensor")
        if target_hidden.device.type != "cpu" or target_hidden.dtype != torch.float32:
            raise TypeError("target_hidden must be a CPU float32 tensor")
        if target_hidden.ndim == 2:
            hidden = target_hidden
        elif target_hidden.ndim == 3 and target_hidden.shape[0] == 1:
            hidden = target_hidden[0]
        else:
            raise ValueError("target_hidden must have shape [tokens, hidden] or [1, tokens, hidden]")
        if hidden.shape[0] != tokens.numel():
            raise ValueError("target_hidden token dimension must match shifted_input_ids")
        states = _flatten_states(past_key_values)
        native = self._runtime.forward_mtp(
            tokens.to(dtype=torch.int64).contiguous().tolist(),
            hidden.contiguous().view(-1).tolist(),
            states,
        )
        if len(native.states) != len(native.state_shapes):
            raise RuntimeError("native ONNX MTP state values and shapes differ in count")
        return OnnxMtpOutput(
            logits=_batch_one_output(native.logits, native.logits_shape, "MTP logits"),
            final_hidden=_batch_one_output(
                native.final_hidden, native.final_hidden_shape, "MTP final hidden"
            ),
            past_key_values=tuple(
                _tensor_from_flat(values, shape, f"MTP state {index}")
                for index, (values, shape) in enumerate(
                    zip(native.states, native.state_shapes)
                )
            ),
            state_names=tuple(native.state_names),
        )

    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int = 20,
        eos_token_id: Optional[Union[int, Tuple[int, ...], list[int]]] = None,
        attention_mask: Optional[Tensor] = None,
        do_sample: bool = False,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        """Batch-one greedy generation over the authenticated dynamic KV cache."""
        if self.manifest.sequence_mode != "dynamic-cache-v1":
            raise TritiumError(
                "this fixed-shape ONNX bundle cannot grow its KV cache",
                code="dynamic_onnx_generation_unavailable",
                stage="generate_onnx",
            )
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported ONNX generation arguments: {names}")
        if do_sample:
            raise TritiumError(
                "ONNX generation currently supports greedy decoding only",
                code="onnx_sampling_unavailable",
                stage="generate_onnx",
            )
        if use_cache is not True:
            raise ValueError("dynamic ONNX generation requires use_cache=True")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        tokens = _batch_one_tokens(input_ids, "input_ids").to(dtype=torch.int64)
        if tokens.numel() == 0:
            raise ValueError("input_ids must contain at least one token")
        if attention_mask is not None:
            _validate_attention_mask(attention_mask, tokens.numel())
        eos = _eos_ids(eos_token_id)
        generated = tokens.contiguous().tolist()
        states: Optional[Tuple[Tensor, ...]] = None
        step = tokens
        for _ in range(max_new_tokens):
            output = self.forward(step, past_key_values=states)
            next_token = int(torch.argmax(output.logits[0, -1]).item())
            generated.append(next_token)
            states = output.past_key_values
            if next_token in eos:
                break
            step = torch.tensor([next_token], dtype=torch.int64)
        return torch.tensor([generated], dtype=torch.int64)


def export_onnx(
    source: Any,
    output_dir: Union[os.PathLike[str], str],
    *,
    profile: str = "compact-v1",
    tokens: int = 1,
    past_tokens: int = 0,
    max_package_bytes: int = 32 * 1024 * 1024 * 1024,
    max_preserved_bytes: int = 8 * 1024 * 1024 * 1024,
    max_salt_resident_bytes: int = 32 * 1024 * 1024 * 1024,
    max_preserved_fp32_bytes: int = 8 * 1024 * 1024 * 1024,
    example_inputs: Any = None,
    model: Optional[nn.Module] = None,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    dynamic_batch: bool = True,
    dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
    opset: int = 18,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Any:
    """Export one typed hard artifact without changing its conversion lineage.

    Complete Qwen PTQ bundles use Tritium's dynamic language-plus-MTP dialect.
    Generic module PTQ, QAT-hard, and refined results use the audited packed
    module exporter with explicit example inputs. Latent training graphs remain
    a v1.3 feature.
    """
    if isinstance(source, (PreparedModel, nn.Module, torch.optim.Optimizer, Mapping)):
        raise TritiumError(
            "trainable ONNX export requires Tritium v1.3",
            code="trainable_onnx_requires_v1_3",
            stage="export_onnx",
        )
    module_options = {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_batch": dynamic_batch,
        "dynamic_axes": dynamic_axes,
        "opset": opset,
        "rtol": rtol,
        "atol": atol,
    }

    def export_module(hard_model: nn.Module, lineage: ModuleOnnxLineage) -> Any:
        if example_inputs is None:
            raise ValueError(f"{lineage.mode} ONNX export requires example_inputs")
        if (
            profile != "compact-v1"
            or tokens != 1
            or past_tokens != 0
            or max_package_bytes != 32 * 1024 * 1024 * 1024
            or max_preserved_bytes != 8 * 1024 * 1024 * 1024
            or max_salt_resident_bytes != 32 * 1024 * 1024 * 1024
            or max_preserved_fp32_bytes != 8 * 1024 * 1024 * 1024
        ):
            raise ValueError("generic module ONNX export does not accept Qwen bundle options")
        return export_module_onnx(
            hard_model,
            example_inputs,
            output_dir,
            lineage=lineage,
            **module_options,
        )

    if isinstance(source, QatHardResult):
        if model is not None:
            raise ValueError("QAT-hard ONNX export owns its model; model must be omitted")
        return export_module(
            source.model,
            ModuleOnnxLineage(
                mode=source.mode,
                artifact_id=source.artifact_id,
                recipe_id=source.recipe_id,
                source_model_digest=source.source_checkpoint_digest,
            ),
        )
    if isinstance(source, QatHardArtifact):
        if not isinstance(model, nn.Module):
            raise ValueError("QAT-hard artifact ONNX export requires model and example_inputs")
        converted = load_qat_hard(source.artifact_dir, model, inplace=False).eval()
        return export_module(
            converted,
            ModuleOnnxLineage(
                mode=source.mode,
                artifact_id=source.artifact_id,
                recipe_id=source.recipe_id,
                source_model_digest=source.source_checkpoint_digest,
            ),
        )
    if isinstance(source, ModuleQuantizationResult):
        if not isinstance(model, nn.Module):
            raise ValueError("PTQ ONNX export requires model and example_inputs")
        converted = load_quantized_module(model, source, inplace=False).eval()
        return export_module(
            converted,
            ModuleOnnxLineage(
                mode="ptq",
                artifact_id=source.artifact_id,
                recipe_id=source.recipe_id,
                source_model_digest=source.source_model_digest,
            ),
        )
    if isinstance(source, RefinementResult):
        if not isinstance(model, nn.Module):
            raise ValueError("refined ONNX export requires model and example_inputs")
        converted = source.load_model(model, inplace=False).eval()
        return export_module(
            converted,
            ModuleOnnxLineage(
                mode=source.mode,
                artifact_id=source.artifact_id,
                recipe_id=source.conversion.recipe_id,
                source_model_digest=source.source_model_digest,
                parent_artifact_id=source.parent_artifact_id,
                ancestry=source.ancestry,
            ),
        )
    if not isinstance(source, QuantizationResult):
        raise TypeError(
            "export_onnx requires a QuantizationResult, ModuleQuantizationResult, "
            "QatHardResult, QatHardArtifact, or RefinementResult"
        )
    if (
        model is not None
        or example_inputs is not None
        or input_names is not None
        or output_names is not None
        or dynamic_batch is not True
        or dynamic_axes is not None
        or opset != 18
        or rtol != 1e-4
        or atol != 1e-5
    ):
        raise ValueError("whole-Qwen ONNX export does not accept generic module options")
    if source.schema_version != 3 or source.preserved is None or not source.hf_assets:
        raise TritiumError(
            "ONNX export requires a complete schema-v3 language-plus-MTP bundle",
            code="incomplete_artifact",
            stage="export_onnx",
            details={"schema_version": source.schema_version},
        )
    source.artifact(profile)
    if tokens != 1 or past_tokens != 0:
        raise ValueError("dynamic ONNX export requires tokens=1 and past_tokens=0")
    limits = {
        "max_package_bytes": max_package_bytes,
        "max_preserved_bytes": max_preserved_bytes,
        "max_salt_resident_bytes": max_salt_resident_bytes,
        "max_preserved_fp32_bytes": max_preserved_fp32_bytes,
    }
    if any(type(value) is not int or value <= 0 for value in limits.values()):
        raise ValueError("ONNX export byte ceilings must be positive integers")
    native = getattr(_tritium, "export_qwen35_onnx_bundle", None)
    if native is None:
        raise TritiumError(
            "this wheel does not contain the Qwen ONNX exporter",
            code="onnx_export_unavailable",
            stage="export_onnx",
        )
    return native(
        str(source.artifact_dir),
        str(output_dir),
        profile=profile,
        tokens=tokens,
        past_tokens=past_tokens,
        **limits,
    )


def load_onnx(
    artifact_dir: Union[os.PathLike[str], str], *, device: str = "cpu"
) -> Any:
    """Strictly admit a typed Tritium ONNX bundle through one public facade."""
    requested = Path(artifact_dir)
    if requested.is_symlink():
        raise ValueError("ONNX artifact must be an ordinary directory")
    directory = requested.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("ONNX artifact must be an ordinary directory")
    qwen_manifest = directory / _MANIFEST_FILE
    module_manifest = directory / "tritium-module-onnx.json"
    qwen_present = qwen_manifest.exists() or qwen_manifest.is_symlink()
    module_present = module_manifest.exists() or module_manifest.is_symlink()
    if qwen_present and module_present:
        raise ValueError("ONNX artifact contains multiple bundle manifests")
    if module_present:
        if device != "cpu":
            raise TritiumError(
                "generic ONNX bundles currently admit CPU execution only",
                code="onnx_device_unavailable",
                stage="load_onnx",
                details={"device": device},
            )
        return load_module_onnx(requested)
    if not qwen_present and any(
        (directory / name).exists() for name in _TRAINING_CHECKPOINT_FILES
    ):
        raise TritiumError(
            "trainable ONNX import requires Tritium v1.3",
            code="trainable_onnx_requires_v1_3",
            stage="load_onnx",
        )
    manifest = _read_manifest(Path(artifact_dir))
    native = getattr(_tritium, "QwenOnnxModel", None)
    if native is None:
        raise TritiumError(
            "this wheel does not contain the Qwen ONNX Runtime session",
            code="onnx_runtime_unavailable",
            stage="load_onnx",
            details={"artifact_dir": str(manifest.directory)},
        )
    runtime = native.load(str(manifest.directory), device=device)
    return QwenOnnxCausalLM(runtime, manifest)


def _pairs_without_duplicates(pairs):
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate ONNX manifest field {key!r}")
        value[key] = item
    return value


def _batch_one_tokens(value: Tensor, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise TritiumError(
            "this ONNX wheel admits CPU tensors only",
            code="onnx_device_unavailable",
            stage="forward_onnx",
            details={"device": str(value.device)},
        )
    if value.ndim == 1:
        tokens = value
    elif value.ndim == 2 and value.shape[0] == 1:
        tokens = value[0]
    else:
        raise ValueError(f"{label} must have shape [tokens] or [1, tokens]")
    if tokens.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"{label} must have an integer dtype")
    return tokens


def _validate_attention_mask(attention_mask: Tensor, tokens: int) -> None:
    if not isinstance(attention_mask, Tensor):
        raise TypeError("attention_mask must be a torch.Tensor")
    if attention_mask.device.type != "cpu" or tuple(attention_mask.shape) not in (
        (1, tokens),
        (tokens,),
    ):
        raise ValueError("attention_mask must match the batch-one input_ids shape")
    if not bool(torch.all(attention_mask != 0).item()):
        raise TritiumError(
            "padded attention masks are not represented by this ONNX graph",
            code="onnx_padding_unavailable",
            stage="forward_onnx",
        )


def _eos_ids(value: Optional[Union[int, Tuple[int, ...], list[int]]]) -> frozenset[int]:
    if value is None:
        return frozenset()
    values = (value,) if type(value) is int else value
    if not isinstance(values, (tuple, list)) or not values:
        raise TypeError("eos_token_id must be an integer or a non-empty list of integers")
    if any(type(item) is not int or item < 0 for item in values):
        raise ValueError("eos_token_id values must be nonnegative integers")
    return frozenset(values)


def _flatten_states(
    states: Optional[Tuple[Tensor, ...]],
) -> Optional[list[list[float]]]:
    if states is None:
        return None
    if not isinstance(states, tuple):
        raise TypeError("past_key_values must be a tuple of torch.Tensor values")
    flattened = []
    for index, state in enumerate(states):
        if not isinstance(state, Tensor):
            raise TypeError(f"past_key_values[{index}] must be a torch.Tensor")
        if state.device.type != "cpu" or state.dtype != torch.float32:
            raise TypeError(f"past_key_values[{index}] must be a CPU float32 tensor")
        flattened.append(state.contiguous().view(-1).tolist())
    return flattened


def _tensor_from_flat(values: Any, shape: Any, label: str) -> Tensor:
    elements = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            raise RuntimeError(f"native ONNX {label} has an invalid shape")
        elements *= dimension
    if len(values) != elements:
        raise RuntimeError(f"native ONNX {label} values and shape differ")
    return torch.tensor(values, dtype=torch.float32).reshape(tuple(shape))


def _batch_one_output(values: Any, shape: Any, label: str) -> Tensor:
    return _tensor_from_flat(values, shape, label).unsqueeze(0)


def _read_manifest(requested: Path) -> OnnxBundleManifest:
    if requested.is_symlink():
        raise ValueError("ONNX artifact must be an ordinary directory")
    directory = requested.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("ONNX artifact must be an ordinary directory")
    path = directory / _MANIFEST_FILE
    value = _read_json_regular(path, max_bytes=1024 * 1024, label="ONNX manifest")
    if not isinstance(value, dict):
        raise ValueError("ONNX manifest must be a JSON object")
    schema = value.get("schema")
    if schema == "tritium-qwen35-onnx-bundle-v1":
        if set(value) != _TOP_FIELDS_V1:
            raise ValueError("ONNX manifest top-level fields differ from schema v1")
        sequence_mode = None
    elif schema == "tritium-qwen35-onnx-bundle-v2":
        if set(value) != _TOP_FIELDS_V2:
            raise ValueError("ONNX manifest top-level fields differ from schema v2")
        if value["sequence_mode"] != "dynamic-cache-v1":
            raise ValueError("schema-v2 ONNX manifest requires dynamic-cache-v1")
        sequence_mode = value["sequence_mode"]
    else:
        raise ValueError("unsupported ONNX bundle schema")
    language = _graph(value["language"], "language.onnx", "language")
    mtp = _graph(value["mtp"], "mtp.onnx", "MTP")
    weights = value["weights"]
    if not isinstance(weights, dict) or set(weights) != _WEIGHT_FIELDS:
        raise ValueError("ONNX weights manifest fields differ from schema v1")
    if weights["file"] != "weights.bin":
        raise ValueError("ONNX weights filename must be canonical")
    weights_digest = _digest(weights["blake3"], "weights")
    if type(weights["bytes"]) is not int or weights["bytes"] <= 0:
        raise ValueError("ONNX weights bytes must be a positive integer")
    identity = _strings(value["identity"], _IDENTITY_FIELDS, "identity")
    conversion = _strings(value["conversion"], _CONVERSION_FIELDS, "conversion")
    if conversion["mode"] not in {"qat-hard", "ptq", "refined"}:
        raise ValueError("unsupported ONNX conversion mode")
    # This descriptor preflight gives deterministic Python errors. The native
    # runtime is still the security authority: it reopens and authenticates all
    # three files immediately before creating the ORT sessions.
    _inspect_regular(
        directory / "language.onnx", "language.onnx", max_bytes=256 * 1024 * 1024
    )
    _inspect_regular(
        directory / "mtp.onnx", "mtp.onnx", max_bytes=256 * 1024 * 1024
    )
    _inspect_regular(
        directory / "weights.bin",
        "weights.bin",
        max_bytes=64 * 1024 * 1024 * 1024,
        expected_bytes=weights["bytes"],
    )
    return OnnxBundleManifest(
        directory=directory,
        language_blake3=language,
        mtp_blake3=mtp,
        weights_blake3=weights_digest,
        weights_bytes=weights["bytes"],
        sequence_mode=sequence_mode,
        source_model_id=identity["source_model_id"],
        tokenizer_id=identity["tokenizer_id"],
        recipe_id=identity["recipe_id"],
        package_id=identity["package_id"],
        converted_coverage_id=identity["converted_coverage_id"],
        deferred_coverage_id=identity["deferred_coverage_id"],
        conversion_mode=conversion["mode"],
        completion_id=conversion["completion_id"],
        campaign_id=conversion["campaign_id"],
        admission_id=conversion["admission_id"],
        selection_id=conversion["selection_id"],
    )


def _graph(value: Any, filename: str, label: str) -> str:
    if not isinstance(value, dict) or set(value) != _GRAPH_FIELDS:
        raise ValueError(f"ONNX {label} manifest fields differ from schema v1")
    if value["file"] != filename:
        raise ValueError(f"ONNX {label} filename must be canonical")
    return _digest(value["blake3"], label)


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"ONNX {label} BLAKE3 must be 64 lowercase hexadecimal characters")
    return value


def _strings(value: Any, fields: set[str], label: str) -> Dict[str, str]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"ONNX {label} fields differ from schema v1")
    if any(not isinstance(item, str) or not item for item in value.values()):
        raise ValueError(f"ONNX {label} values must be non-empty strings")
    return value


def _read_json_regular(path: Path, *, max_bytes: int, label: str) -> Any:
    descriptor, before, opened = _open_regular(path, label, max_bytes=max_bytes)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            value = json.load(stream, object_pairs_hook=_pairs_without_duplicates)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _require_unchanged(path, before, opened, label)
    return value


def _inspect_regular(
    path: Path,
    label: str,
    *,
    max_bytes: int | None = None,
    expected_bytes: int | None = None,
) -> None:
    descriptor, before, opened = _open_regular(
        path,
        label,
        max_bytes=max_bytes,
        expected_bytes=expected_bytes,
    )
    os.close(descriptor)
    _require_unchanged(path, before, opened, label)


def _open_regular(
    path: Path,
    label: str,
    *,
    max_bytes: int | None = None,
    expected_bytes: int | None = None,
):
    before = path.lstat()
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened as an ordinary file") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or not _same_file(before, opened)
            or opened.st_size <= 0
            or (max_bytes is not None and opened.st_size > max_bytes)
            or (expected_bytes is not None and opened.st_size != expected_bytes)
        ):
            raise ValueError(f"{label} type, identity, or length is invalid")
        return descriptor, before, opened
    except Exception:
        os.close(descriptor)
        raise


def _require_unchanged(
    path: Path, before: os.stat_result, opened: os.stat_result, label: str
) -> None:
    after = path.lstat()
    if (
        not _same_file(before, after)
        or before.st_size != opened.st_size
        or after.st_size != opened.st_size
    ):
        raise ValueError(f"{label} changed while being inspected")


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


__all__ = [
    "OnnxBundleManifest",
    "OnnxCausalLMOutput",
    "OnnxMtpOutput",
    "QwenOnnxCausalLM",
    "export_onnx",
    "load_onnx",
]
