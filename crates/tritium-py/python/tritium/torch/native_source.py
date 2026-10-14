"""Opt-in Torch persistent-state adapter to the shared native quantization engine.

This packages complete source state; it does not reconstruct an executable
Torch shell or reinterpret legacy calibration and numerical algorithm identities.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import torch
from torch import nn

from .. import _tritium
from .conversion import PreparedModel, _parameter_coverage
from .ptq import _source_model_digest
from .qat import _state_slot, _state_tensor_identity

_CATALOG = "tritium-torch-state.json"
_OPAQUE = "tritium-torch-preserved.safetensors"
_MAX_CATALOG = 1024 * 1024
_PRECISIONS = {torch.float16: "F16", torch.bfloat16: "Bf16", torch.float32: "F32"}
_STATE_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64,
                 torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool}


def _registered_state(model):
    result = {}
    pending = [("", model, frozenset())]
    while pending:
        prefix, module, ancestors = pending.pop()
        if id(module) in ancestors:
            raise ValueError("native source module graph contains a cycle")
        ancestors = ancestors | {id(module)}
        for kind, slots in (("parameter", module._parameters), ("buffer", module._buffers)):
            for name, tensor in slots.items():
                if tensor is None or (kind == "buffer" and name in module._non_persistent_buffers_set):
                    continue
                path = f"{prefix}.{name}" if prefix else name
                if not name or "." in name or path in result:
                    raise ValueError("ambiguous registered state slot")
                result[path] = tensor
        pending.extend((f"{prefix}.{name}" if prefix else name, child, ancestors)
                       for name, child in module._modules.items() if child is not None)
    return result


def _identity(tensor):
    return _state_tensor_identity(tensor) + (tensor.is_conj(), tensor.is_neg())


def _inventory(prepared):
    if not isinstance(prepared, PreparedModel) or not isinstance(prepared.model, nn.Module):
        raise TypeError("native module conversion requires a live PreparedModel")
    if prepared.config.mode != "ptq" or prepared.coverage is None:
        raise ValueError("native module conversion requires exact PTQ source coverage")
    if prepared.config.target_bpw is not None:
        raise ValueError("native module conversion requires explicit plans, not target_bpw")
    registered = _registered_state(prepared.model)
    selected = {alias for entry in prepared.coverage.entries
                if entry.disposition == "selected" for alias in entry.aliases}
    current = _parameter_coverage(prepared.model, selected, target_disposition="selected")
    if current != prepared.coverage:
        raise ValueError("prepared parameter coverage changed before native conversion")
    parameter_owners = {alias: entry.path for entry in current.entries for alias in entry.aliases}
    state = prepared.model.state_dict(keep_vars=True)
    if set(state) != set(registered):
        raise ValueError("state_dict differs from complete registered persistent state")
    groups = {}
    storages = {}
    for name, tensor in state.items():
        if (not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
                or tensor.layout != torch.strided or tensor.is_quantized
                or tensor.device.type == "meta" or tensor.dtype not in _STATE_DTYPES):
            raise ValueError("native module state must be ordinary materialized tensors")
        slot = _state_slot(prepared.model, name)
        if slot[3] is not tensor:
            raise ValueError("custom state replacement differs from its registered tensor slot")
        identity = _identity(tensor)
        storage = identity[:2]
        if storage in storages and storages[storage] != identity:
            raise ValueError("nonidentical state views share storage")
        storages[storage] = identity
        groups.setdefault(identity, []).append((name, tensor, slot[1]))
    entries = []
    for members in groups.values():
        aliases = sorted(item[0] for item in members)
        tensor = members[0][1]
        kinds = {item[2] for item in members}
        if len(kinds) != 1:
            raise ValueError("state aliases mix parameter and buffer interpretations")
        if len({item[1].requires_grad for item in members}) != 1:
            raise ValueError("state aliases disagree on gradient ownership")
        targets = set(aliases) & selected
        if targets and targets != set(aliases):
            raise ValueError("shared state owner is both selected and preserved")
        owners = {parameter_owners[alias] for alias in aliases if alias in parameter_owners}
        if len(owners) > 1:
            raise ValueError("distinct parameter owners share the same state storage")
        owner = next(iter(owners)) if owners else aliases[0]
        native = tensor.dtype in _PRECISIONS and tensor.ndim > 0 and all(tensor.shape)
        if targets and (not native or tensor.ndim != 2):
            raise ValueError("native selected state must be a positive-shaped float matrix")
        entries.append((owner, aliases, tensor, next(iter(kinds)), bool(targets), native))
    return sorted(entries, key=lambda entry: entry[0])


def _topology(entries):
    return tuple((owner, tuple(aliases), kind, targeted, native, id(tensor),
                  _identity(tensor))
                 for owner, aliases, tensor, kind, targeted, native in entries)


def _source_tensor(tensor):
    return tensor.detach().resolve_conj().resolve_neg().to(device="cpu").contiguous()


def convert_native_module(
    prepared: PreparedModel,
    output_dir,
    run_id: str,
    *,
    actions: Mapping,
    levels,
    limits: Mapping,
    work_dir,
    asset_paths: Mapping,
    architecture: str = "torch-module",
    input_grams: Mapping | None = None,
):
    """Compile persistent state to native files and fit with explicit Rust actions.

    ``actions`` maps every selected coverage owner to the existing Rust
    QuantAction JSON (Fit with DirectFitPlan and explicit metric). Limits and
    level names are the existing QuantPlan fields. Source files are temporary;
    original state assets remain in the published package. The result is the
    readonly native conversion result, not an executable Torch/HF model.
    """
    from safetensors.torch import save_file

    entries = _inventory(prepared)
    topology = _topology(entries)
    selected = {owner for owner, _, _, _, targeted, _ in entries if targeted}
    if not selected or not isinstance(actions, Mapping) or set(actions) != selected:
        raise ValueError("native actions must exactly match selected tensor owners")
    if any(not isinstance(value, Mapping) or set(value) != {"Fit"} for value in actions.values()):
        raise ValueError("selected native actions must explicitly Fit")
    assets = dict(asset_paths)
    if {_CATALOG, _OPAQUE} & assets.keys():
        raise ValueError("caller asset collides with reserved Torch state asset")
    # Interpret all numeric contracts in Rust before any source payload writes.
    tensors = []
    for owner, _, tensor, _, targeted, native in entries:
        if native:
            tensors.append(dict(source=dict(name=owner, shape=list(tensor.shape),
                                            precision=_PRECISIONS[tensor.dtype]),
                                action=copy.deepcopy(actions[owner]) if targeted else "KeepDense"))
    opaque = [entry for entry in entries if not entry[5]]
    names = [*assets, _CATALOG] + ([_OPAQUE] if opaque else [])
    plan = dict(architecture=architecture, tensors=tensors, assets=sorted(names),
                levels=list(levels), limits=dict(limits))
    plan_json = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False)
    _tritium._validate_model_plan(plan_json)
    source_bytes = sum(t.numel() * t.element_size() for _, _, t, _, _, native in entries if native)
    opaque_bytes = sum(t.numel() * t.element_size() for _, _, t, _, _, native in entries if not native)
    if source_bytes > limits["max_source_bytes"] or opaque_bytes > limits["max_asset_bytes"]:
        raise ValueError("persistent state exceeds native source/asset byte limits")
    source_digest = _source_model_digest(prepared.model)
    catalog = dict(adapter="tritium.torch.native-source@1", source_preparation_config=prepared.config.to_dict(),
                   coverage=prepared.coverage.to_dict(), source_model_digest=source_digest,
                   state=[dict(owner=owner, aliases=aliases, dtype=str(tensor.dtype),
                               shape=list(tensor.shape), kind=kind,
                               disposition="fitted" if targeted else "preserved",
                               storage="tensor" if native else _OPAQUE)
                          for owner, aliases, tensor, kind, targeted, native in entries])
    encoded = json.dumps(catalog, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > _MAX_CATALOG or opaque_bytes + len(encoded) > limits["max_asset_bytes"]:
        raise ValueError("Torch state catalog exceeds asset byte limit")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must not be empty")
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"native output already exists: {output}")
    work = Path(work_dir).resolve(strict=True)
    if not work.is_dir():
        raise ValueError("work_dir must be an existing directory")
    assets = {name: os.fspath(Path(path).absolute()) for name, path in assets.items()}
    grams = {name: os.fspath(Path(path).absolute()) for name, path in (input_grams or {}).items()}
    temporary = tempfile.TemporaryDirectory(prefix="tritium-native-source-", dir=work)
    result = None
    try:
        root = Path(temporary.name)
        source = root / "source.safetensors"
        save_file({owner: _source_tensor(tensor)
                   for owner, _, tensor, _, _, native in entries if native}, str(source))
        if opaque:
            path = root / _OPAQUE
            save_file({owner: _source_tensor(tensor)
                       for owner, _, tensor, _, _, _ in opaque}, str(path))
            assets[_OPAQUE] = str(path)
        (root / _CATALOG).write_bytes(encoded)
        assets[_CATALOG] = str(root / _CATALOG)
        # Headers and caller assets count too; the native source repeats this check.
        if sum(Path(path).stat().st_size for path in assets.values()) > limits["max_asset_bytes"]:
            raise ValueError("original Torch state assets exceed native byte limit")
        if _topology(_inventory(prepared)) != topology:
            raise ValueError("model state ownership changed during source capture")
        if _source_model_digest(prepared.model) != source_digest:
            raise ValueError("model state changed during source capture")
        result = _tritium.convert_model_plan(plan_json, [str(source)], str(output), run_id,
                                             asset_paths=assets, input_grams=grams)
    finally:
        try:
            temporary.cleanup()
        except OSError as error:
            if result is None:
                raise
            result = result._with_source_cleanup_warning(str(error))
    return result


__all__ = ["convert_native_module"]
