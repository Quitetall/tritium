"""Strict Qwen3.6 language-plus-MTP calibration boundary.

Transformers can expose the language graph while silently dropping the bundled
MTP drafter from a checkpoint.  That is unsafe for plan-0043 evidence: a
language-only calibration must never be presented as language-plus-MTP.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import torch
from torch import nn

from .ptq import ActivationCalibrationReceipt, capture_qwen36_kronecker_evidence

Pathish = Union[str, Path]


class Qwen36ComponentError(ValueError):
    """The supplied graph does not satisfy the Qwen3.6 component contract."""


class Qwen36MtpLoadError(ValueError):
    """The checkpoint cannot be attached to the explicit Qwen3.6 MTP graph."""


def _transformers_mtp_types() -> tuple[type[nn.Module], type[nn.Module], type[nn.Module]]:
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
        )
    except ImportError as error:  # pragma: no cover - dependency-gated path
        raise Qwen36MtpLoadError(
            "Qwen3.6 MTP attachment requires transformers with qwen3_5 support"
        ) from error
    return Qwen3_5DecoderLayer, Qwen3_5RMSNorm, nn.Linear


class Qwen36MtpAdapter(nn.Module):
    """PyTorch MTP graph for the flat ``mtp.*`` Qwen3.6 checkpoint namespace.

    Transformers intentionally ignores ``mtp.*`` when constructing its normal
    conditional-generation graph. This adapter retains that graph explicitly:
    shared embedding -> two pre-FC norms -> 10240-to-5120 projection -> one
    full-attention decoder layer -> final norm. The shared embedding and rotary
    module are unregistered aliases so they do not become duplicate checkpoint
    state or alter Tritium's canonical MTP tensor inventory.
    """

    def __init__(
        self,
        text_config: Any,
        shared_embeddings: nn.Module,
        rotary_embeddings: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if not isinstance(shared_embeddings, nn.Module):
            raise TypeError("shared_embeddings must be a torch module")
        decoder_layer, rms_norm, linear = _transformers_mtp_types()
        mtp_config = copy.deepcopy(text_config)
        mtp_config.layer_types = ["full_attention"]
        self.fc = linear(
            int(text_config.hidden_size) * 2,
            int(text_config.hidden_size),
            bias=False,
        )
        self.layers = nn.ModuleList([decoder_layer(mtp_config, 0)])
        self.norm = rms_norm(
            int(text_config.hidden_size), eps=float(text_config.rms_norm_eps)
        )
        self.pre_fc_norm_embedding = rms_norm(
            int(text_config.hidden_size), eps=float(text_config.rms_norm_eps)
        )
        self.pre_fc_norm_hidden = rms_norm(
            int(text_config.hidden_size), eps=float(text_config.rms_norm_eps)
        )
        # Bypass Module.__setattr__: shared modules must remain aliases, not
        # duplicate registered children or state-dict entries.
        object.__setattr__(self, "_shared_embeddings", shared_embeddings)
        object.__setattr__(self, "_rotary_embeddings", rotary_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        *,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("Qwen3.6 MTP requires target hidden_states")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self._shared_embeddings(input_ids)
        if inputs_embeds.shape[:-1] != hidden_states.shape[:-1]:
            raise ValueError("MTP embeddings and hidden_states must share batch/sequence shape")
        hidden_states = self.fc(
            torch.cat(
                (
                    self.pre_fc_norm_embedding(inputs_embeds),
                    self.pre_fc_norm_hidden(hidden_states),
                ),
                dim=-1,
            )
        )
        if position_embeddings is None:
            if self._rotary_embeddings is None:
                raise ValueError("position_embeddings or rotary_embeddings is required")
            if position_ids is None:
                position_ids = torch.arange(
                    hidden_states.shape[1], device=hidden_states.device
                ).expand(hidden_states.shape[0], -1)
            position_embeddings = self._rotary_embeddings(hidden_states, position_ids)
        hidden_states = self.layers[0](
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        return self.norm(hidden_states)


def _ordinary_path(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise Qwen36MtpLoadError(f"{label} must be an ordinary directory")
    return path.resolve(strict=True)


def attach_qwen36_mtp(
    model: nn.Module,
    model_dir: Pathish,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Qwen36MtpAdapter:
    """Attach and load Qwen3.6's omitted ``mtp.*`` tensors onto ``model``.

    ``AutoModelForImageTextToText.from_pretrained`` currently builds the
    language/vision graph but ignores the flat MTP namespace. This function
    loads exactly those 15 tensors into :class:`Qwen36MtpAdapter`, shares the
    model's language embedding/rotary modules, and publishes it as ``model.mtp``
    so :func:`resolve_qwen36_components` can enforce language-plus-MTP capture.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    root = _ordinary_path(Path(model_dir), "Qwen3.6 model directory")
    index_path = root / "model.safetensors.index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise Qwen36MtpLoadError("Qwen3.6 safetensors index is missing")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen36MtpLoadError("Qwen3.6 safetensors index is unreadable") from error
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise Qwen36MtpLoadError("Qwen3.6 safetensors index has no weight_map")
    mtp_keys = {name for name in weight_map if name.startswith("mtp.")}
    expected_keys = {
        "mtp.fc.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
    if mtp_keys != expected_keys:
        missing = sorted(expected_keys - mtp_keys)
        extra = sorted(mtp_keys - expected_keys)
        raise Qwen36MtpLoadError(
            f"Qwen3.6 MTP tensor inventory differs (missing={missing}, extra={extra})"
        )
    try:
        language_model = model.model.language_model
        shared_embeddings = language_model.embed_tokens
    except AttributeError as error:
        raise Qwen36MtpLoadError(
            "model must expose model.language_model.embed_tokens"
        ) from error
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    if text_config is None:
        raise Qwen36MtpLoadError("model config must expose text_config")
    rotary_embeddings = getattr(language_model, "rotary_emb", None)
    adapter = Qwen36MtpAdapter(text_config, shared_embeddings, rotary_embeddings)
    if dtype is None:
        embedding_dtype = getattr(getattr(shared_embeddings, "weight", None), "dtype", None)
        if embedding_dtype is not None and torch.is_floating_point(
            torch.empty((), dtype=embedding_dtype)
        ):
            dtype = embedding_dtype
    if dtype is not None:
        adapter = adapter.to(dtype=dtype)
    if device is None:
        try:
            inferred = next(model.parameters()).device
        except StopIteration:
            inferred = torch.device("cpu")
        device = torch.device("cpu") if inferred.type == "meta" else inferred
    adapter = adapter.to(device=device)
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - dependency-gated path
        raise Qwen36MtpLoadError("Qwen3.6 MTP attachment requires safetensors") from error
    targets = adapter.state_dict()
    shard_keys: dict[str, list[str]] = {}
    for name in sorted(mtp_keys):
        shard = weight_map[name]
        if not isinstance(shard, str) or not shard:
            raise Qwen36MtpLoadError(f"MTP shard entry for {name} is invalid")
        relative_shard = Path(shard)
        if (
            relative_shard.is_absolute()
            or relative_shard.name != shard
            or any(part in {"", ".", ".."} for part in relative_shard.parts)
        ):
            raise Qwen36MtpLoadError(f"MTP shard path is not a flat name: {shard}")
        shard_keys.setdefault(shard, []).append(name)
    with torch.no_grad():
        for shard, names in shard_keys.items():
            shard_path = root / shard
            if shard_path.is_symlink() or not shard_path.is_file():
                raise Qwen36MtpLoadError(f"MTP shard escapes model directory: {shard}")
            shard_path = shard_path.resolve(strict=True)
            if root not in shard_path.parents:
                raise Qwen36MtpLoadError(f"MTP shard escapes model directory: {shard}")
            try:
                with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                    for source_name in names:
                        target_name = source_name.removeprefix("mtp.")
                        if target_name not in targets:
                            raise Qwen36MtpLoadError(
                                f"MTP adapter lacks target parameter {target_name}"
                            )
                        tensor = handle.get_tensor(source_name)
                        target = targets[target_name]
                        if tuple(tensor.shape) != tuple(target.shape):
                            raise Qwen36MtpLoadError(
                                f"MTP shape mismatch for {source_name}: "
                                f"checkpoint={tuple(tensor.shape)} target={tuple(target.shape)}"
                            )
                        target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            except Qwen36MtpLoadError:
                raise
            except (OSError, KeyError, RuntimeError) as error:
                raise Qwen36MtpLoadError(
                    f"failed loading Qwen3.6 MTP shard {shard}"
                ) from error
    model.mtp = adapter
    return adapter


@dataclass(frozen=True)
class Qwen36Components:
    """Resolved language, MTP, and output-head modules from one graph."""

    root: nn.Module
    language_model: nn.Module
    mtp_model: Optional[nn.Module]
    lm_head: nn.Module
    language_path: str
    mtp_path: Optional[str]
    lm_head_path: str


@dataclass(frozen=True)
class Qwen36LanguageMtpOutput:
    """Output from :class:`Qwen36LanguageMtpOracle`.

    ``base_output`` remains available for model-specific fields.  ``logits``
    and ``loss`` delegate to it so existing Tritium capture objectives can use
    this output without a model-specific branch.
    """

    base_output: Any
    mtp_hidden_states: torch.Tensor
    mtp_logits: torch.Tensor

    @property
    def logits(self) -> Optional[torch.Tensor]:
        return getattr(self.base_output, "logits", None)

    @property
    def loss(self) -> Optional[torch.Tensor]:
        return getattr(self.base_output, "loss", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_output, name)


class Qwen36LanguageMtpOracle(nn.Module):
    """Execute Qwen language and MTP paths without hidden-state offload hooks.

    Transformers' ``output_hidden_states=True`` path can corrupt the final
    hidden state when Accelerate disk offload is active.  This oracle captures
    the already-normalized language output through a local hook while keeping
    the base forward on its ordinary path, then executes the attached MTP
    graph.  It is intended for calibration, parity, and MTP evidence; it does
    not replace the source model's generation implementation.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        components = resolve_qwen36_components(model, require_mtp=True)
        if components.mtp_model is None:
            raise Qwen36ComponentError("Qwen3.6 MTP drafter is required for oracle execution")
        self._model = model
        object.__setattr__(self, "_language_model", components.language_model)
        object.__setattr__(self, "_mtp_model", components.mtp_model)
        object.__setattr__(self, "_lm_head", components.lm_head)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Qwen36LanguageMtpOutput:
        hidden: list[torch.Tensor] = []

        def capture(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise Qwen36ComponentError(
                    "Qwen3.6 language norm must return one hidden-state tensor"
                )
            hidden.append(output)

        handle = self._language_model.norm.register_forward_hook(capture)
        base_kwargs = dict(kwargs)
        # Avoid Transformers output-capturing hooks under disk offload.  The
        # local norm hook above is the authoritative hidden-state capture.
        base_kwargs["output_hidden_states"] = False
        try:
            base_output = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **base_kwargs,
            )
        finally:
            handle.remove()
        if len(hidden) != 1:
            raise Qwen36ComponentError(
                f"Qwen3.6 language norm executed {len(hidden)} times; expected exactly once"
            )
        if input_ids is None and inputs_embeds is None:
            raise Qwen36ComponentError(
                "Qwen3.6 oracle requires input_ids or inputs_embeds for MTP"
            )
        if position_ids is not None and position_ids.ndim != 2:
            raise Qwen36ComponentError(
                "Qwen3.6 oracle requires two-dimensional position_ids for MTP"
            )
        mtp_mask = attention_mask
        if mtp_mask is not None and mtp_mask.dtype != torch.bool and not (
            mtp_mask.is_floating_point() or mtp_mask.is_complex()
        ):
            mtp_mask = mtp_mask.bool()
        mtp_hidden = self._mtp_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            hidden_states=hidden[0],
            attention_mask=mtp_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        if not isinstance(mtp_hidden, torch.Tensor):
            raise Qwen36ComponentError("Qwen3.6 MTP graph must return one hidden-state tensor")
        mtp_logits = self._lm_head(mtp_hidden)
        return Qwen36LanguageMtpOutput(
            base_output=base_output,
            mtp_hidden_states=mtp_hidden,
            mtp_logits=mtp_logits,
        )


def _module(root: nn.Module, path: str, label: str) -> nn.Module:
    current: Any = root
    for part in path.split("."):
        if not hasattr(current, part):
            raise Qwen36ComponentError(f"Qwen3.6 {label} module is missing: {path}")
        current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise Qwen36ComponentError(f"Qwen3.6 {label} is not a torch module: {path}")
    return current


def resolve_qwen36_components(
    model: nn.Module,
    *,
    require_mtp: bool = True,
) -> Qwen36Components:
    """Resolve canonical Qwen3.6 module paths without alias guessing.

    The supported Transformers layout is ``model.language_model`` under the
    top-level ``model`` field, with ``lm_head`` at the root.  The MTP drafter
    must be exposed as ``mtp`` by a loader that actually retained checkpoint
    tensors.  Missing MTP is an error by default, never an implicit downgrade.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    language = _module(model, "model.language_model", "language")
    lm_head = _module(model, "lm_head", "language output head")
    mtp = (
        _module(model, "mtp", "MTP drafter")
        if require_mtp or hasattr(model, "mtp")
        else None
    )
    if id(language) == id(lm_head) or (
        mtp is not None and id(language) == id(mtp)
    ) or (mtp is not None and id(lm_head) == id(mtp)):
        raise Qwen36ComponentError("Qwen3.6 component paths alias unexpectedly")
    return Qwen36Components(
        root=model,
        language_model=language,
        mtp_model=mtp,
        lm_head=lm_head,
        language_path="model.language_model",
        mtp_path="mtp" if mtp is not None else None,
        lm_head_path="lm_head",
    )


def capture_qwen36_components(
    model: nn.Module,
    data_factory: Callable[[Any], Iterable[Any]],
    *,
    model_dir: Pathish,
    declared_revision: str,
    work_dir: Pathish,
    evidence_dir: Pathish,
    curvature: str,
    activation_cache_digest: str,
    token_stream_digest: str,
    damping: float,
    execution_model: Optional[nn.Module] = None,
    guided_loss_reduction: Optional[str] = None,
    max_evidence_bytes: int = 64 * 1024 * 1024,
    max_batch_bytes: int = 256 * 1024 * 1024,
    max_capture_bytes: Optional[int] = None,
    max_objective_bytes: int = 256 * 1024 * 1024,
    max_shared_modules: int = 8,
) -> ActivationCalibrationReceipt:
    """Capture strict language-plus-MTP Qwen3.6 evidence.

    Component resolution happens before native session creation or evidence
    mutation.  The existing model-aware capture loop remains the sole producer
    of canonical records and retains its resumability/provenance guarantees.
    Pass ``execution_model`` when a containing language-plus-MTP oracle must
    exercise modules that are not reached by ``model.language_model`` alone.
    """

    components = resolve_qwen36_components(model, require_mtp=True)
    if components.mtp_model is None:
        raise Qwen36ComponentError("Qwen3.6 MTP drafter is required for capture")
    return capture_qwen36_kronecker_evidence(
        components.language_model,
        data_factory,
        model_dir=model_dir,
        declared_revision=declared_revision,
        work_dir=work_dir,
        evidence_dir=evidence_dir,
        curvature=curvature,
        activation_cache_digest=activation_cache_digest,
        token_stream_digest=token_stream_digest,
        damping=damping,
        mtp_model=components.mtp_model,
        execution_model=execution_model,
        guided_loss_reduction=guided_loss_reduction,
        max_evidence_bytes=max_evidence_bytes,
        max_batch_bytes=max_batch_bytes,
        max_capture_bytes=max_capture_bytes,
        max_objective_bytes=max_objective_bytes,
        max_shared_modules=max_shared_modules,
    )


__all__ = [
    "Qwen36ComponentError",
    "Qwen36Components",
    "Qwen36MtpAdapter",
    "Qwen36MtpLoadError",
    "attach_qwen36_mtp",
    "capture_qwen36_components",
    "resolve_qwen36_components",
]
