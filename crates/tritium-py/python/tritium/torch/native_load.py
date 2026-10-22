"""Strict native package application to isolated, compatible CPU model shells."""
from __future__ import annotations
import json
from types import MethodType
import torch
from torch import nn
from .. import _tritium
from .config import TernaryConfig
from .conversion import PreparedModel
from .coverage import CoverageReport
from .module_artifacts import _pairs_without_duplicates
from .native_source import _inventory, _graph_contract, _STATE_DTYPES
from .qat import _state_slot
from .qat_artifacts import _clone_module_shell

_DTYPES = {str(dtype): dtype for dtype in _STATE_DTYPES}
_PRECISIONS = {"F16": torch.float16, "Bf16": torch.bfloat16, "F32": torch.float32}
_FIELDS = {"owner", "aliases", "dtype", "shape", "kind", "disposition", "storage"}


def _compact(value):
    value = value.contiguous()
    stride = 1
    for size, actual in zip(reversed(value.shape), reversed(value.stride())):
        if actual != stride:
            return value.clone(memory_format=torch.contiguous_format)
        stride *= size
    return value


def _check_input_autograd(value):
    if value.device.type != "cpu" or value.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("native input gradients require CPU float16, bfloat16 or float32")
    if torch.is_autocast_enabled("cpu"):
        raise RuntimeError("native input gradients do not support autocast")


def _matmul_output(value, handle, *, admit_backward=False):
    if value.ndim == 0 or value.shape[-1] != handle.columns:
        raise ValueError("native matrix input width differs")
    batch = value.numel() // handle.columns
    if batch * (handle.columns + handle.rows) * 4 > handle.max_output_bytes:
        raise ValueError("native output/scratch exceeds byte limit")
    if admit_backward and (batch + 1) * handle.columns * 4 > handle.max_output_bytes:
        raise ValueError("native input adjoint output/scratch exceeds byte limit")
    source = _compact(value.detach().to(dtype=torch.float32))
    return torch.utils.dlpack.from_dlpack(handle.matmul(source)).to(dtype=value.dtype)


def _input_vjp_output(value, handle):
    _check_input_autograd(value)
    if value.ndim == 0 or value.shape[-1] != handle.rows:
        raise ValueError("native output gradient width differs")
    batch = value.numel() // handle.rows
    if ((batch + 1) * handle.columns * 4 > handle.max_output_bytes
            or batch * (handle.columns + handle.rows) * 4 > handle.max_output_bytes):
        raise ValueError("native input adjoint output/scratch exceeds byte limit")
    source = _compact(value.detach().to(dtype=torch.float32))
    return torch.utils.dlpack.from_dlpack(handle.input_vjp(source)).to(dtype=value.dtype)


class _NativeMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, handle):
        _check_input_autograd(value)
        output = _matmul_output(value, handle, admit_backward=True)
        ctx.handle = handle
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    def backward(ctx, gradient):
        return (None if gradient is None else _NativeInputVjp.apply(gradient, ctx.handle)), None


class _NativeInputVjp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gradient, handle):
        output = _input_vjp_output(gradient, handle)
        ctx.handle = handle
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    def backward(ctx, gradient):
        return (None if gradient is None else _NativeMatmul.apply(gradient, ctx.handle)), None


class NativeTernaryWeight(nn.Module):
    """Immutable native CPU owner, with optional input-only eager autograd."""
    def __init__(self, handle, *, input_gradients=False):
        super().__init__()
        if type(input_gradients) is not bool:
            raise TypeError("input_gradients must be bool")
        self.handle = handle
        self.input_gradients = input_gradients
        self.out_features, self.in_features = handle.rows, handle.columns

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        raise RuntimeError("native packed state must be exported as its native package")

    def _apply(self, fn, recurse=True):
        probe = fn(torch.empty(0, dtype=torch.float32))
        if probe.device.type != "cpu":
            raise ValueError("native packed CPU weights cannot move to another device")
        return self

    def matmul(self, value):
        if value.device.type != "cpu" or not value.dtype.is_floating_point:
            raise ValueError("native packed matmul requires CPU floating activations")
        if torch.is_grad_enabled() and value.requires_grad:
            if not self.input_gradients:
                raise RuntimeError("native packed CPU adapter is inference-only")
            return _NativeMatmul.apply(value, self.handle)
        return _matmul_output(value, self.handle)

    def gather(self, indices, dtype):
        if indices.device.type != "cpu" or indices.dtype not in {torch.int32, torch.int64}:
            raise ValueError("native packed gather requires CPU integer indices")
        if indices.numel() * (self.in_features * 4 + 8) > self.handle.max_output_bytes:
            raise ValueError("native gather exceeds byte limit")
        source = _compact(indices.to(dtype=torch.int64))
        return torch.utils.dlpack.from_dlpack(self.handle.gather(source)).to(dtype=dtype)


class NativeTernaryLinear(nn.Module):
    def __init__(self, weight, bias, *, owner):
        super().__init__()
        self.in_features, self.out_features = weight.in_features, weight.out_features
        if owner: self.add_module("packed_weight", weight)
        else: object.__setattr__(self, "packed_weight", weight)
        # Bias storage is rebound after all consumers exist to preserve its aliases.
        self.register_buffer("bias", bias)

    def forward(self, value):
        output = self.packed_weight.matmul(value)
        return output if self.bias is None else output + self.bias.to(dtype=output.dtype)


class NativeTernaryEmbedding(nn.Module):
    def __init__(self, weight, module, *, owner):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.out_features, weight.in_features
        self.padding_idx = module.padding_idx
        self.output_dtype = module.weight.dtype
        if owner: self.add_module("packed_weight", weight)
        else: object.__setattr__(self, "packed_weight", weight)

    def _apply(self, fn, recurse=True):
        probe = fn(torch.empty(0, dtype=self.output_dtype))
        if probe.device.type != "cpu":
            raise ValueError("native packed embedding cannot move to another device")
        self.output_dtype = probe.dtype
        return super()._apply(fn, recurse)

    def forward(self, indices):
        return self.packed_weight.gather(indices, self.output_dtype)


def _catalog(snapshot):
    data = json.loads(snapshot.asset("tritium-torch-state.json", max_bytes=1024*1024),
                      object_pairs_hook=_pairs_without_duplicates)
    fields = {"adapter", "graph", "source_preparation_config", "coverage", "source_model_digest", "state"}
    if not isinstance(data, dict) or set(data) != fields or data["adapter"] != "tritium.torch.native-source@2":
        raise ValueError("native executable catalog requires state adapter version 2")
    digest = data["source_model_digest"]
    if (not isinstance(digest, str) or len(digest) != 71 or not digest.startswith("sha256:")
            or any(c not in "0123456789abcdef" for c in digest[7:])):
        raise ValueError("native catalog source digest invalid")
    states = data["state"]
    if not isinstance(states, list) or not states:
        raise ValueError("native catalog state inventory empty")
    seen = set()
    owners = []
    for entry in states:
        if not isinstance(entry, dict) or set(entry) != _FIELDS:
            raise ValueError("native catalog state fields invalid")
        aliases = entry["aliases"]
        if (not isinstance(aliases, list) or not aliases
                or any(not isinstance(a, str) or not a or a in seen for a in aliases)
                or aliases != sorted(set(aliases))
                or any(not isinstance(entry[field], str) for field in
                       ("owner", "dtype", "kind", "disposition", "storage"))
                or entry["owner"] not in aliases or entry["dtype"] not in _DTYPES
                or entry["kind"] not in {"parameter", "buffer"}
                or entry["disposition"] not in {"fitted", "preserved"}
                or entry["storage"] not in {"tensor", "tritium-torch-preserved.safetensors"}
                or not isinstance(entry["shape"], list)
                or any(type(n) is not int or n < 0 for n in entry["shape"])):
            raise ValueError("native catalog state interpretation invalid")
        seen.update(aliases)
        owners.append(entry["owner"])
    if owners != sorted(set(owners)):
        raise ValueError("native catalog owners are duplicated or unordered")
    return data


def _guarded_apply(model, fn, recurse=True):
    probe = fn(torch.empty(0, dtype=torch.float32))
    if probe.device.type != "cpu":
        raise ValueError("native packed model is explicitly CPU-only")
    result = model._tritium_native_base_apply(model, fn, recurse)
    # Torch independently replaces buffers in each consumer during dtype moves.
    # Restore the admitted tensor-owner topology without inspecting state_dict,
    # which deliberately cannot represent packed native handles.
    for aliases in model._tritium_native_preserved_aliases:
        _, _, _, value = _state_slot(model, aliases[0])
        for alias in aliases[1:]:
            module, kind, leaf, _ = _state_slot(model, alias)
            (module._parameters if kind == "parameter" else module._buffers)[leaf] = value
    return result


def load_native_module(model, artifact, *, level, load_run_id, inplace,
                       max_package_bytes=8*1024**3, max_payload_bytes=2*1024**3,
                       max_workspace_bytes=64*1024**2, max_output_bytes=64*1024**2,
                       max_evidence_bytes=None, input_gradients=False):
    """Validate topology and privately stage native CPU weights plus original state.

    Artifact may be an actual NativeModelConversion or an engine output directory.
    Persistent shell state may be CPU or meta; nonpersistent buffers must already
    be materialized on CPU. Package access still retains a complete byte image.
    Defaults to inference-only eager CPU execution. Explicit input_gradients=True
    enables f16/bf16/f32 input derivatives through immutable Linear weights, not
    owner/scale/code updates or native refinement. It does not imply torch.compile,
    autocast, ordinary state_dict export, device portability or release qualification.
    """
    if type(input_gradients) is not bool:
        raise TypeError("input_gradients must be bool")
    if not isinstance(model, nn.Module) or type(inplace) is not bool:
        raise TypeError("native model load requires nn.Module and explicit inplace")
    if any(type(limit) is not int or limit <= 0 for limit in
           (max_package_bytes, max_payload_bytes, max_workspace_bytes, max_output_bytes)):
        raise ValueError("native load byte limits must be positive integers")
    if max_evidence_bytes is not None and (type(max_evidence_bytes) is not int or max_evidence_bytes <= 0):
        raise ValueError("native evidence byte limit must be a positive integer")
    evidence_options = {} if max_evidence_bytes is None else dict(max_evidence_bytes=max_evidence_bytes)
    result = artifact if isinstance(artifact, _tritium.NativeModelConversion) else _tritium.NativeModelConversion.open(
        str(artifact), max_package_bytes=max_package_bytes, **evidence_options)
    snapshot = result.open_level(level, load_run_id, max_payload_bytes=max_payload_bytes,
                                 max_workspace_bytes=max_workspace_bytes, max_output_bytes=max_output_bytes,
                                 max_package_bytes=max_package_bytes, **evidence_options)
    data = _catalog(snapshot)
    config = TernaryConfig.from_dict(data["source_preparation_config"])
    coverage = CoverageReport.from_dict(data["coverage"])
    observed = _inventory(PreparedModel(model, config, coverage), materialized=False)
    expected = [dict(owner=owner, aliases=aliases, dtype=str(tensor.dtype), shape=list(tensor.shape),
                     kind=kind, disposition="fitted" if targeted else "preserved",
                     storage="tensor" if native else "tritium-torch-preserved.safetensors")
                for owner, aliases, tensor, kind, targeted, native in observed]
    if expected != data["state"] or coverage.to_dict() != data["coverage"]:
        raise ValueError("native catalog differs from complete model shell state")
    if _graph_contract(model, observed) != data["graph"]:
        raise ValueError("native graph contract differs from model shell")
    if any(tensor.device.type not in {"cpu", "meta"} for _, _, tensor, _, _, _ in observed):
        raise ValueError("native model shell state must be on CPU or meta")
    for module in model.modules():
        for name, value in module._buffers.items():
            if (name in module._non_persistent_buffers_set and value is not None
                    and value.device.type != "cpu"):
                raise ValueError("native model shell nonpersistent buffers must be materialized on CPU")
    descriptors = {item["name"]: item["storage"] for item in json.loads(snapshot.descriptors())}
    native_entries = {entry["owner"]: entry for entry in expected if entry["storage"] == "tensor"}
    if set(descriptors) != set(native_entries):
        raise ValueError("native package tensor inventory differs from catalog")
    for name, entry in native_entries.items():
        storage = descriptors[name]
        if entry["disposition"] == "fitted":
            if set(storage) != {"Additive"}:
                raise ValueError("native fitted state is not additive")
            layout = storage["Additive"]["layout"]
            if [layout["rows"], layout["cols"]] != entry["shape"]:
                raise ValueError("native fitted geometry differs from catalog")
        elif (set(storage) != {"Dense"} or storage["Dense"]["shape"] != entry["shape"]
              or _PRECISIONS[storage["Dense"]["precision"]] != _DTYPES[entry["dtype"]]):
            raise ValueError("native preserved geometry/precision differs from catalog")
    opaque_entries = {entry["owner"]: entry for entry in expected if entry["storage"] != "tensor"}
    opaque = {}
    if opaque_entries:
        from safetensors.torch import load as load_safetensors

        opaque = load_safetensors(snapshot.asset("tritium-torch-preserved.safetensors"))
    if set(opaque) != set(opaque_entries):
        raise ValueError("native preserved asset inventory differs from catalog")
    for name, tensor in opaque.items():
        entry = opaque_entries[name]
        if list(tensor.shape) != entry["shape"] or str(tensor.dtype) != entry["dtype"]:
            raise ValueError("native preserved asset geometry/dtype differs")
    # Clone metadata only; original persistent tensors stay borrowed until replaced.
    target = _clone_module_shell(model)
    modules = dict(target.named_modules(remove_duplicate=False))
    replacements = {}
    for entry in expected:
        if entry["disposition"] != "fitted": continue
        weight = NativeTernaryWeight(snapshot.packed_tensor(entry["owner"]), input_gradients=input_gradients)
        by_module = {}
        owner = True
        for alias in entry["aliases"]:
            path = "" if alias == "weight" else alias.removesuffix(".weight")
            if alias != "weight" and not alias.endswith(".weight"):
                raise ValueError("native fitted alias is not a weight slot")
            module = modules[path]
            replacement = by_module.get(id(module))
            if replacement is None:
                if type(module) is nn.Linear:
                    replacement = NativeTernaryLinear(weight, module.bias, owner=owner)
                elif type(module) is nn.Embedding:
                    replacement = NativeTernaryEmbedding(weight, module, owner=owner)
                else: raise ValueError("native consumer kind unsupported")
                by_module[id(module)] = replacement
                owner = False
            replacements[path] = replacement
    root = target
    for path, replacement in replacements.items():
        if not path: root = replacement
        else:
            parent, _, leaf = path.rpartition(".")
            modules[parent]._modules[leaf] = replacement
    for entry in expected:
        if entry["disposition"] == "fitted": continue
        name = entry["owner"]
        if entry["storage"] == "tensor":
            precision, shape, payload = snapshot.preserved_tensor(name)
            value = torch.frombuffer(bytearray(payload), dtype=_PRECISIONS[precision]).reshape(shape).clone()
        else: value = opaque[name].clone()
        slots = [_state_slot(root, alias) for alias in entry["aliases"]]
        # A preserved bias becomes a buffer in an inference-only Linear.
        if any(kind == "parameter" for _, kind, _, _ in slots):
            value = nn.Parameter(value, requires_grad=False)
        for module, kind, leaf, _ in slots:
            (module._parameters if kind == "parameter" else module._buffers)[leaf] = value
    upload_log = snapshot.finish()
    root.requires_grad_(False)
    root.eval()
    root._tritium_native_artifact = result
    root._tritium_native_level = level
    root._tritium_native_input_gradients = input_gradients
    root._tritium_native_upload_log = upload_log
    root._tritium_native_preserved_aliases = tuple(tuple(entry["aliases"]) for entry in expected
                                                  if entry["disposition"] == "preserved")
    object.__setattr__(root, "_tritium_native_base_apply", type(root)._apply)
    object.__setattr__(root, "_apply", MethodType(_guarded_apply, root))
    if hasattr(root, "all_tied_weights_keys"): root.all_tied_weights_keys = {}
    if hasattr(root, "_tied_weights_keys"): root._tied_weights_keys = {}
    if inplace and root is target:
        object.__setattr__(model, "__dict__", root.__dict__)
        object.__setattr__(model, "_apply", MethodType(_guarded_apply, model))
        return model
    return root


__all__ = ["load_native_module", "NativeTernaryWeight", "NativeTernaryLinear", "NativeTernaryEmbedding"]
