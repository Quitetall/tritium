"""Invocation-aware ephemeral owner scores, not durable calibrated fitting.

Actual eager CPU calls feed the one Rust numerical collector. This profile
rejects unsupported owner reads instead of assuming module hooks are exhaustive.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from .native_source import _inventory, _graph_contract
from .native_load import _compact


@dataclass(frozen=True)
class NativeOwnerWindowCapture:
    """Rust-owned ephemeral observations; no estimator identity or fit receipt."""
    owners: tuple
    invocations: tuple
    objective_bytes: bytes

    @property
    def source_id(self):
        return self.owners[0].source_id

    def row_score(self, name, row, *, max_workspace_bytes):
        for owner in self.owners:
            if owner.tensor_name == name:
                return torch.from_dlpack(owner.row_score(row, max_workspace_bytes))
        raise ValueError("tensor absent from captured owner inventory")


def _storage(tensor):
    if isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided:
        return (tensor.device, tensor.untyped_storage()._cdata)
    return None


def _same_view(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape
            and left.stride() == right.stride() and left.storage_offset() == right.storage_offset()
            and left.is_conj() == right.is_conj() and left.is_neg() == right.is_neg())


class _OwnerReads(TorchDispatchMode):
    def __init__(self, protected, active, admit):
        self.protected, self.active, self.admit = protected, active, admit

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        flat = tree_flatten((args, kwargs))[0]
        if self.active and (result := self.active[-1].get("operator_output")) is not None:
            frame = self.active[-1]
            touched = [value for value in flat if isinstance(value, torch.Tensor)
                       and _storage(value) == _storage(result)]
            if touched:
                if func in {torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default}:
                    return func(*args, **kwargs)
                module = frame["module"]
                if (type(module) is nn.Linear and module.bias is not None and not frame.get("bias_added")
                        and func is torch.ops.aten.add.Tensor and len(args) == 2
                        and _storage(args[1]) == _storage(module.bias) and _same_view(args[1], module.bias)
                        and kwargs.get("alpha", 1) == 1):
                    result = func(*args, **kwargs)
                    frame["operator_output"], frame["output_version"], frame["bias_added"] = result, result._version, True
                    return result
                raise ValueError("unsupported owner invocation output transformation")
        tensors = [value for value in flat
                   if isinstance(value, torch.Tensor) and _storage(value) in self.protected]
        if not tensors:
            return func(*args, **kwargs)
        if not self.active:
            raise ValueError("unsupported read of selected owner outside its module")
        frame = self.active[-1]
        module = frame["module"]
        weight = module.weight
        if len(tensors) != 1 or _storage(tensors[0]) != _storage(weight):
            raise ValueError("unsupported selected owner view or additional input use")
        if type(module) is nn.Linear and func is torch.ops.aten.t.default:
            if frame.get("transposed") or frame.get("input") is not None or not _same_view(tensors[0], weight):
                raise ValueError("unsupported selected owner transpose")
            frame["transposed"] = True
            return func(*args, **kwargs)
        if frame.get("input") is not None:
            raise ValueError("multiple owner operations inside one declared invocation")
        if type(module) is nn.Linear and func in {torch.ops.aten.mm.default, torch.ops.aten.addmm.default}:
            input_index, weight_index = (0, 1) if func is torch.ops.aten.mm.default else (1, 2)
            view = args[weight_index]
            if (not frame.get("transposed") or _storage(view) != _storage(weight)
                    or view.shape != (weight.shape[1], weight.shape[0])
                    or view.stride() != tuple(reversed(weight.stride()))
                    or view.storage_offset() != weight.storage_offset()
                    or view.is_conj() or view.is_neg()
                    or any(kwargs.get(key, 1) != 1 for key in ("alpha", "beta")) or "out" in kwargs):
                raise ValueError("unsupported selected owner matmul semantics")
            value = args[input_index]
            if value.ndim != 2 or value.shape[1] != module.in_features:
                raise ValueError("owner matmul geometry differs")
            self.admit(frame, value, "matmul")
            frame["bias_added"] = func is torch.ops.aten.addmm.default
        elif type(module) is nn.Embedding and func is torch.ops.aten.embedding.default:
            if (not _same_view(args[0], weight) or len(args) > 2 and
                    tuple(args[2:]) != (-1, False, False)
                    or any(kwargs.get(key, default) != default for key, default in
                           (("padding_idx", -1), ("scale_grad_by_freq", False), ("sparse", False)))):
                raise ValueError("unsupported selected owner embedding semantics")
            self.admit(frame, args[1], "gather")
        else:
            raise ValueError("unsupported selected owner operation")
        result = func(*args, **kwargs)
        frame["operator_output"] = result
        frame["output_version"] = result._version
        return result


def _context(window, maximum):
    if not isinstance(window, Mapping) or "input_ids" not in window:
        raise ValueError("owner capture requires a tensor window with input_ids")
    if {"calibration_weights", "calibration_mask"} & window.keys():
        raise ValueError("token statistical selection is not a whole-window score estimator")
    tokens = window["input_ids"]
    if (not isinstance(tokens, torch.Tensor) or tokens.dtype not in {torch.int32, torch.int64}
            or tokens.numel() == 0):
        raise ValueError("owner window input_ids must be nonempty CPU integer values")
    total = 0
    for name, value in window.items():
        if (not isinstance(name, str) or not isinstance(value, torch.Tensor)
                or value.device.type != "cpu" or value.layout != torch.strided or value.is_quantized):
            raise ValueError("owner window context must contain ordinary CPU tensors")
        total += value.numel() * value.element_size()
    if total > maximum:
        raise ValueError("owner window context exceeds byte limit")
    if any(not torch.isfinite(value).all().item() for value in window.values()):
        raise ValueError("nonfinite owner window context")
    if torch.any(tokens < 0).item() or torch.any(tokens > 2**32 - 1).item():
        raise ValueError("owner window token ids must be u32 values")
    return {name: _compact(value.detach().resolve_conj().resolve_neg().reshape(-1).clone())
            for name, value in window.items()}


def _check_context(window, specs, snapshots):
    if set(window) != set(snapshots):
        raise ValueError("owner window context changed during objective")
    for name, snapshot in snapshots.items():
        value = window[name]
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or (value.dtype, value.shape, value.layout, value.is_conj(), value.is_neg()) != specs[name]):
            raise ValueError("owner window context changed during objective")
        current = _compact(value.detach().resolve_conj().resolve_neg().reshape(-1))
        if not torch.equal(current.view(torch.uint8), snapshot.view(torch.uint8)):
            raise ValueError("owner window context changed during objective")


def _graph_is_supported(loss, maximum):
    seen, pending = set(), [loss.grad_fn]
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        if isinstance(node, torch.autograd.function.BackwardCFunction):
            raise ValueError("custom autograd functions are unsupported by owner capture")
        seen.add(node)
        if len(seen) > maximum:
            raise ValueError("owner objective graph exceeds node limit")
        pending.extend(edge[0] for edge in node.next_functions)


def capture_owner_window(source, window, objective, *, objective_bytes,
                         max_observation_bytes, max_context_bytes, max_invocations,
                         max_samples, max_graph_nodes):
    """Capture one actual scalar objective. Durable cache/fit integration is separate."""
    from .native_calibration import NativeModulePreparation
    if not isinstance(source, NativeModulePreparation) or not callable(objective):
        raise TypeError("owner capture requires native preparation and objective")
    if not isinstance(objective_bytes, bytes) or not objective_bytes:
        raise ValueError("actual declared objective bytes required")
    limits = (max_observation_bytes, max_context_bytes, max_invocations, max_samples, max_graph_nodes)
    if any(type(value) is not int or value <= 0 for value in limits):
        raise ValueError("owner capture limits must be positive integers")
    if len(objective_bytes) > max_context_bytes or torch.is_inference_mode_enabled():
        raise ValueError("owner objective context or inference mode unsupported")
    snapshots = _context(window, max_context_bytes - len(objective_bytes))
    specs = {name: (value.dtype, value.shape, value.layout, value.is_conj(), value.is_neg())
             for name, value in window.items()}
    source.verify()
    model = source.prepared.model
    entries = _inventory(source.prepared)
    graph = _graph_contract(model, entries)
    modules = dict(model.named_modules(remove_duplicate=False))
    consumers, protected = {}, {}
    for owner, aliases, weight, _, selected, _ in entries:
        if not selected:
            continue
        if weight.device.type != "cpu" or weight.is_conj() or weight.is_neg():
            raise ValueError("owner capture requires ordinary CPU selected weights")
        protected[_storage(weight)] = owner
        for alias in aliases:
            path = "" if alias == "weight" else alias.removesuffix(".weight")
            module = modules[path]
            if type(module) not in {nn.Linear, nn.Embedding}:
                raise ValueError("owner consumer must be an exact Linear or Embedding")
            if getattr(module.forward, "__func__", None) is not type(module).forward:
                raise ValueError("overridden owner forwards are unsupported")
            if module._forward_pre_hooks or module._forward_hooks or module._backward_hooks or module._backward_pre_hooks:
                raise ValueError("existing owner hooks are unsupported")
            if type(module) is nn.Embedding and (module.padding_idx is not None or module.max_norm is not None
                    or module.scale_grad_by_freq or module.sparse):
                raise ValueError("unsupported owner embedding backward options")
            consumers.setdefault(id(module), dict(module=module, owner=owner, paths=[]))["paths"].append(path)
    if not consumers or any(tensor.device.type != "cpu" for _, _, tensor, _, _, _ in entries):
        raise ValueError("owner capture requires a complete compatible CPU source")
    training = tuple((module, module.training) for module in modules.values())
    gradients = tuple((tensor, tensor.requires_grad, tensor.grad,
                       None if tensor.grad is None else tensor.grad._version)
                      for _, _, tensor, kind, _, _ in entries if kind == "parameter")
    active, records, hooks = [], [], []
    totals = dict(bytes=0, samples=0, calls=0)
    phase = "forward"

    def pre(module, args):
        if phase != "forward" or not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise ValueError("recomputed or no-grad owner invocation unsupported")
        if getattr(module.forward, "__func__", None) is not type(module).forward:
            raise ValueError("overridden owner forwards are unsupported")
        if type(module) is nn.Embedding and (module.padding_idx is not None or module.max_norm is not None
                or module.scale_grad_by_freq or module.sparse):
            raise ValueError("unsupported owner embedding backward options")
        active.append(dict(consumers[id(module)]))

    def admit(frame, value, role):
        module = frame["module"]
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("owner invocation requires ordinary CPU input")
        samples = value.numel() // module.in_features if role == "matmul" else value.numel()
        size = samples * (4 * (module.in_features + module.out_features) if role == "matmul"
                          else 8 + 4 * module.embedding_dim)
        if (samples == 0 or totals["calls"] + 1 > max_invocations
                or totals["samples"] + samples > max_samples or totals["bytes"] + size > max_observation_bytes):
            raise ValueError("owner invocation exceeds byte/sample/call limit")
        if role == "matmul" and not value.dtype.is_floating_point:
            raise ValueError("owner matmul requires floating input")
        if role == "gather" and value.dtype not in {torch.int32, torch.int64}:
            raise ValueError("owner gather requires integer indices")
        frame["input"] = _compact(value.detach().reshape(samples, -1).to(
            dtype=torch.float32 if role == "matmul" else torch.int64).clone())
        if role == "gather":
            frame["input"] = frame["input"].reshape(-1)
        frame["role"], frame["samples"] = role, samples
        frame["ordinal"] = totals["calls"]
        totals["calls"] += 1; totals["samples"] += samples; totals["bytes"] += size

    def post(module, args, value):
        if not active or active[-1]["module"] is not module:
            raise ValueError("owner invocation nesting differs")
        frame = active.pop()
        width = module.out_features if type(module) is nn.Linear else module.embedding_dim
        result = frame.get("operator_output")
        if (frame.get("input") is None or not isinstance(value, torch.Tensor) or value.ndim == 0
                or value.device.type != "cpu" or not value.dtype.is_floating_point
                or value.numel() != frame["samples"] * width or value.shape[-1] != width
                or not value.is_contiguous() or result is None or result._version != frame["output_version"]
                or _storage(value) != _storage(result) or value.storage_offset() != result.storage_offset()
                or value.dtype != result.dtype or value.is_conj() or value.is_neg()):
            raise ValueError("owner invocation output geometry differs")
        if not value.requires_grad:
            if result.requires_grad:
                raise ValueError("detached owner output is unsupported")
            value.requires_grad_(True)
        frame["output"] = value
        records.append(frame)

    try:
        expected_hooks = {}
        for consumer in consumers.values():
            module = consumer["module"]
            before, after = module.register_forward_pre_hook(pre), module.register_forward_hook(post)
            hooks.extend((before, after))
            expected_hooks[id(module)] = (before.id, after.id)
        with torch.enable_grad():
            with _OwnerReads(protected, active, admit):
                loss = objective(model, window)
            records.sort(key=lambda frame: frame["ordinal"])
            _check_context(window, specs, snapshots)
            if (active or {id(frame["module"]) for frame in records} != set(consumers)):
                raise ValueError("required owner consumers did not execute")
            if (not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.requires_grad
                    or not torch.isfinite(loss).item()):
                raise ValueError("owner objective must be a finite differentiable scalar")
            _graph_is_supported(loss, max_graph_nodes)
            phase = "backward"
            adjoints = torch.autograd.grad(loss, [frame["output"] for frame in records], allow_unused=False)
        _check_context(window, specs, snapshots)
        if (any(module.training != flag for module, flag in training)
                or any(tensor.requires_grad != flag or tensor.grad is not grad
                       or grad is not None and grad._version != version for tensor, flag, grad, version in gradients)
                or _graph_contract(model, _inventory(source.prepared)) != graph):
            raise ValueError("owner source execution state changed")
        for consumer in consumers.values():
            module = consumer["module"]
            before, after = expected_hooks[id(module)]
            if (set(module._forward_pre_hooks) != {before} or set(module._forward_hooks) != {after}
                    or module._backward_hooks or module._backward_pre_hooks):
                raise ValueError("owner hooks changed during objective")
        source.verify()
        uses = {}
        for frame, adjoint in zip(records, adjoints):
            owner = frame["owner"]
            captured = _compact(adjoint.detach().reshape(frame["samples"], -1).to(dtype=torch.float32).clone())
            uses.setdefault(owner, []).append((frame["role"], frame["input"], captured))
        owners = tuple(source.observation._owner_window(name, uses[name], max_observation_bytes)
                       for name in sorted(uses))
        if sum(owner.observation_bytes for owner in owners) != totals["bytes"]:
            raise ValueError("owner observation byte accounting differs")
        invocations = tuple((frame["owner"], tuple(sorted(frame["paths"])), frame["role"], frame["samples"])
                            for frame in records)
        return NativeOwnerWindowCapture(owners, invocations, objective_bytes)
    finally:
        for handle in hooks:
            handle.remove()
        active.clear(); records.clear(); snapshots.clear()
