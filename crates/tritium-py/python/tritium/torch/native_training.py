"""Explicit first-order eager CPU scale training; no refined artifact publication."""
from __future__ import annotations
import json
from types import MethodType
import weakref
import tempfile
from pathlib import Path
import torch
from .. import _tritium
from .native_load import (NativeTernaryWeight, NativeTernaryLinear, NativeTernaryEmbedding, _compact, _check_input_autograd,
                          _matmul_output, _input_vjp_output, load_native_module)


class _SavedBudget:
    def __init__(self, maximum):
        self.maximum, self.used = maximum, 0

    def reserve(self, count):
        if self.used + count > self.maximum:
            raise ValueError("native training saved-input budget exceeded")
        self.used += count
        return _SavedLease(self, count)


class _SavedLease:
    def __init__(self, budget, count):
        self.budget, self.count = budget, count

    def release(self):
        if self.budget is not None:
            self.budget.used -= self.count
            self.budget = None

    def __del__(self):
        self.release()


def _guard(ctx):
    trainer = ctx.trainer()
    if trainer is None or not trainer.active:
        raise RuntimeError("native training context is closed")
    if torch.is_grad_enabled():
        raise RuntimeError("native scale training supports first-order backward only")
    if trainer.revision != ctx.revision:
        raise RuntimeError("stale native training backward")
    trainer._validate()
    return trainer


def _backward_guard(fn):
    def guarded(ctx, gradient):
        try:
            return fn(ctx, gradient)
        except BaseException:
            trainer = ctx.trainer()
            if trainer is not None:
                trainer._failed = True
            raise
    return guarded


class _ScaleMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, leaf, handle, trainer, revision, lease):
        ctx.trainer, ctx.revision, ctx.handle, ctx.lease = trainer, revision, handle, lease
        ctx.save_for_backward(value, leaf)
        ctx.set_materialize_grads(False)
        return _matmul_output(value, handle, admit_backward=True)

    @staticmethod
    @_backward_guard
    def backward(ctx, gradient):
        if gradient is None:
            return (None,) * 6
        _guard(ctx)
        value, _ = ctx.saved_tensors  # Torch guards external input/leaf edits.
        _check_input_autograd(gradient)
        x = _compact(value.detach().to(dtype=torch.float32))
        dy = _compact(gradient.detach().to(dtype=torch.float32))
        scales = torch.utils.dlpack.from_dlpack(ctx.handle.scale_vjp(x, dy))
        dx = _input_vjp_output(gradient, ctx.handle) if ctx.needs_input_grad[0] else None
        return dx, scales, None, None, None, None


class _ScaleEmbed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, leaf, handle, trainer, revision, lease, padding, dtype):
        ctx.trainer, ctx.revision, ctx.handle, ctx.lease = trainer, revision, handle, lease
        ctx.padding = padding
        ctx.save_for_backward(indices, leaf)
        ctx.set_materialize_grads(False)
        source = _compact(indices.to(dtype=torch.int64))
        return torch.utils.dlpack.from_dlpack(handle.gather(source)).to(dtype=dtype)

    @staticmethod
    @_backward_guard
    def backward(ctx, gradient):
        if gradient is None:
            return (None,) * 8
        _guard(ctx)
        indices, _ = ctx.saved_tensors
        _check_input_autograd(gradient)
        if ctx.padding is not None:
            gradient = gradient.masked_fill((indices == ctx.padding).unsqueeze(-1), 0)
        ids = _compact(indices.to(dtype=torch.int64))
        dy = _compact(gradient.detach().to(dtype=torch.float32))
        scales = torch.utils.dlpack.from_dlpack(ctx.handle.embedding_scale_vjp(ids, dy))
        return None, scales, None, None, None, None, None, None


def _deny_apply(model, fn, recurse=True):
    raise RuntimeError("native scale training cannot change model dtype/device")


class NativeScaleTrainer:
    """Single-controller first-order CPU scale training over Rust atomic state.

    Use model/loss.backward(), zero_grad(), then step(window_identity). Gradient
    leaves are private f64 transport buffers, not torch.optim Parameters. Rust
    owns deployed scale values and updates. No concurrent backward/step, autocast,
    higher-order/compile/vmap, code training or refined artifact export is admitted.
    """
    def __init__(self, model, backend, plan, weights, max_saved_input_bytes):
        self.model, self._backend = model, backend
        self._plan_json = json.dumps(plan, allow_nan=False)
        self._owners = tuple(plan["owners"])
        self._weights, self._cache = weights, {}
        self._saved = _SavedBudget(max_saved_input_bytes)
        self.active = True
        self._failed = False
        self._recording_work = None
        self._base_apply = model._apply
        self._topology = tuple((name, id(module), type(module)) for name, module in model.named_modules(remove_duplicate=False))
        self._state = tuple((module, kind, name, tensor, tensor._version) for module in model.modules()
                            for kind, entries in (("parameter", module._parameters), ("buffer", module._buffers))
                            for name, tensor in entries.items()
                            if tensor is not None and (kind == "parameter" or name not in module._non_persistent_buffers_set))
        self._consumers = tuple((module, module.packed_weight) for module in model.modules()
                                if type(module) in {NativeTernaryLinear, NativeTernaryEmbedding})
        self._original_handles = {owner: weight.handle for owner, weight in weights.items()}
        self._parent = model._tritium_native_artifact
        model._tritium_native_training_parent_artifact = self._parent
        del model._tritium_native_artifact
        model._tritium_native_unpublished_scales = True
        object.__setattr__(model, "_apply", MethodType(_deny_apply, model))
        for owner, weight in weights.items():
            weight._native_training = (weakref.ref(self), owner)
        model.train()

    @property
    def plan(self):
        """Detached request metadata; edits cannot rebind owner gradient order."""
        return json.loads(self._plan_json)

    @property
    def revision(self):
        return self._backend.completed_updates

    @property
    def saved_input_bytes(self):
        """Conservative graph-lifetime reservation, not process RSS."""
        return self._saved.used

    def _validate(self):
        if not self.active:
            raise RuntimeError("native training context is closed")
        if tuple((n, id(m), type(m)) for n, m in self.model.named_modules(remove_duplicate=False)) != self._topology:
            raise RuntimeError("native training module topology changed")
        for module, kind, name, tensor, version in self._state:
            slots = module._parameters if kind == "parameter" else module._buffers
            if slots.get(name) is not tensor or tensor._version != version:
                raise RuntimeError("native preserved state changed during training")
        if any(module.packed_weight is not weight for module, weight in self._consumers):
            raise RuntimeError("native training consumer owner rebound")
        for owner, weight in self._weights.items():
            binding = getattr(weight, "_native_training", None)
            if binding is None or binding[0]() is not self or binding[1] != owner:
                raise RuntimeError("native training owner binding changed")
            if weight.handle is not self._original_handles[owner]:
                raise RuntimeError("native training owner handle rebound")
            item = self._cache.get(owner)
            if item and item[2] is not None:
                leaf = item[2]
                if (weight._buffers.get("_tritium_scale_leaf") is not leaf
                        or leaf._version != item[3] or not leaf.requires_grad):
                    raise RuntimeError("native gradient leaf changed outside Rust trainer")

    def _frame(self, owner, gradients):
        self._validate()
        if self._failed:
            raise RuntimeError("native backward failed; call zero_grad before reuse")
        revision = self.revision
        cached = self._cache.get(owner)
        if cached is not None and cached[0] == revision and (not gradients or cached[2] is not None):
            return revision, cached[1], cached[2]
        handle, scales = self._backend.frame(owner, revision, gradients)
        leaf = None if scales is None else torch.utils.dlpack.from_dlpack(scales).requires_grad_(True)
        if leaf is not None:
            self._weights[owner].register_buffer("_tritium_scale_leaf", leaf, persistent=False)
        self._cache[owner] = (revision, handle, leaf, None if leaf is None else leaf._version)
        return revision, handle, leaf

    @staticmethod
    def _admit(handle, inputs, gradients, indices=0):
        count = (inputs + gradients) * 4 + handle.columns * 4 + handle.scale_count * 8 + indices * 8
        if count > handle.max_output_bytes:
            raise ValueError("native training backward transfer/output/scratch exceeds byte limit")

    def matmul(self, owner, value):
        _check_input_autograd(value)
        self._validate()
        handle = self._original_handles[owner]
        if value.ndim == 0 or value.shape[-1] != handle.columns:
            raise ValueError("native matrix input width differs")
        if not torch.is_grad_enabled():
            _, handle, _ = self._frame(owner, False)
            return _matmul_output(value, handle)
        self._admit(handle, value.numel(), value.numel() // handle.columns * handle.rows)
        lease = self._saved.reserve(value.numel() * value.element_size() + handle.scale_count * 8)
        try:
            revision, handle, leaf = self._frame(owner, True)
            return _ScaleMatmul.apply(value, leaf, handle, weakref.ref(self), revision, lease)
        except BaseException:
            lease.release()
            raise

    def gather(self, owner, indices, dtype, padding):
        if indices.device.type != "cpu" or indices.dtype not in {torch.int32, torch.int64}:
            raise ValueError("native packed gather requires CPU integer indices")
        if dtype not in {torch.float16, torch.bfloat16, torch.float32} or torch.is_autocast_enabled("cpu"):
            raise ValueError("native scale training requires CPU f16/bf16/f32 without autocast")
        self._validate()
        handle = self._original_handles[owner]
        if indices.numel() * (handle.columns * 4 + 8) > handle.max_output_bytes:
            raise ValueError("native gather exceeds byte limit")
        if not torch.is_grad_enabled():
            _, handle, _ = self._frame(owner, False)
            return torch.utils.dlpack.from_dlpack(handle.gather(_compact(indices.to(dtype=torch.int64)))).to(dtype=dtype)
        self._admit(handle, 0, indices.numel() * handle.columns, indices.numel())
        lease = self._saved.reserve(indices.numel() * indices.element_size() + handle.scale_count * 8)
        try:
            revision, handle, leaf = self._frame(owner, True)
            return _ScaleEmbed.apply(indices, leaf, handle, weakref.ref(self), revision, lease, padding, dtype)
        except BaseException:
            lease.release()
            raise

    def zero_grad(self):
        self._validate()
        for _, _, leaf, _ in self._cache.values():
            if leaf is not None:
                leaf.grad = None
        self._failed = False

    def step(self, window_identity):
        self._validate()
        if self._failed:
            raise RuntimeError("native backward failed; call zero_grad before reuse")
        if (not isinstance(window_identity, (bytes, bytearray, list, tuple)) or len(window_identity) != 32
                or any(type(n) is not int or not 0 <= n <= 255 for n in window_identity)):
            raise ValueError("native training window identity requires 32 bytes")
        revision = self.revision
        gradients = []
        for owner in self._owners:
            item = self._cache.get(owner)
            gradient = None if item is None or item[0] != revision or item[2] is None else item[2].grad
            if gradient is not None:
                if gradient.device.type != "cpu" or gradient.dtype != torch.float64:
                    raise ValueError("native owner gradients require CPU float64")
                gradient = _compact(gradient.detach())
            gradients.append(gradient)
        return self._backend.step(revision, list(window_identity), gradients)

    def checkpoint(self):
        return bytes(self._backend.checkpoint())

    def publish(self, output_dir, run_id, *, machine, validation, limits):
        """Publish an unqualified child through Rust replay and common no-clobber commit.

        Requires recording configured at factory creation. The live shell stays
        unpublished; load the returned NativeModelRefinement for frozen inference.
        Validation values are observations, not authenticated objective execution.
        """
        self._validate()
        if self._failed:
            raise RuntimeError("native backward failed; call zero_grad before reuse")
        if self._recording_work is None:
            raise RuntimeError("native publication requires recorded training")
        fields = {"max_spool_bytes", "max_working_bytes", "max_evidence_bytes", "max_companion_bytes",
                  "max_package_bytes", "max_manifest_bytes", "max_directory_bytes", "max_parent_bytes",
                  "max_replay_working_bytes", "max_journal_bytes"}
        if (not isinstance(limits, dict) or set(limits) != fields
                or any(type(n) is not int or n <= 0 for n in limits.values())):
            raise ValueError("native publication requires exact positive limit fields")
        if not isinstance(run_id, str) or not run_id or not isinstance(machine, str) or not machine:
            raise ValueError("native publication requires explicit run and machine")
        if not isinstance(validation, (list, tuple)):
            raise TypeError("bounded validation observations required")
        return self._backend.publish(str(Path(output_dir).resolve()), run_id, machine,
                                     json.dumps(validation, allow_nan=False), json.dumps(limits, allow_nan=False))

    def close(self):
        if not self.active:
            return self.model
        self._validate()
        revision = self.revision
        staged = [(owner, self._backend.frame(owner, revision, False)[0]) for owner in self._owners]
        for owner, handle in staged:
            weight = self._weights[owner]
            weight.handle = handle
            weight._native_training = None
            weight._buffers.pop("_tritium_scale_leaf", None)
            weight._non_persistent_buffers_set.discard("_tritium_scale_leaf")
        self._cache.clear()
        self.active = False
        object.__setattr__(self.model, "_apply", self._base_apply)
        self._cleanup_recording()
        return self.model

    def _cleanup_recording(self):
        self._backend.close_recording()
        if self._recording_work is not None:
            self._recording_work.cleanup()
            self._recording_work = None

    def __del__(self):
        try:
            self._cleanup_recording()
        except BaseException:
            pass


def native_scale_training(model, artifact, *, level, plan, load_run_id,
                          max_snapshot_bytes, max_parent_bytes, max_saved_input_bytes,
                          checkpoint=None, recording=None, **load_limits):
    """Create a private native eager CPU scale trainer, never a refined artifact.

    plan is the complete Rust NativeRefinementPlan mapping/JSON; obtain its exact
    parent tuple with artifact.native_refinement_parent(level). One declared
    training window per step; pass its content identity to step(). Limits are
    logical, not complete Torch graph/process memory. Source shell is untouched.
    """
    if not isinstance(artifact, _tritium.NativeModelConversion):
        raise TypeError("native scale training requires NativeModelConversion")
    if not isinstance(level, str) or not level or not isinstance(load_run_id, str) or not load_run_id:
        raise ValueError("explicit level and load run required")
    for limit in (max_snapshot_bytes, max_parent_bytes, max_saved_input_bytes):
        if type(limit) is not int or limit <= 0:
            raise ValueError("native training byte limits must be positive integers")
    data = json.loads(plan, object_pairs_hook=_pairs) if isinstance(plan, str) else json.loads(json.dumps(plan))
    text = json.dumps(data, allow_nan=False)
    _tritium._validate_native_scale_plan(text)
    if data["parent"]["level"] != level:
        raise ValueError("native training explicit parent level differs")
    if set(load_limits) - {"max_package_bytes", "max_payload_bytes", "max_workspace_bytes", "max_output_bytes", "max_evidence_bytes"}:
        raise TypeError("unknown native training load limit")
    record_options = None
    if recording is not None:
        fields = {"work_dir", "max_journal_bytes", "max_working_bytes"}
        if (not isinstance(recording, dict) or set(recording) != fields
                or any(type(recording[name]) is not int or recording[name] <= 0 for name in fields - {"work_dir"})):
            raise ValueError("native recording requires work_dir and positive journal/work limits")
        if checkpoint is not None:
            raise ValueError("recorded training requires zero origin; partial journal continuation unavailable")
        work = Path(recording["work_dir"]).resolve(strict=True)
        parent = Path(artifact.output_dir).resolve(strict=True)
        if not work.is_dir() or work == parent or parent in work.parents:
            raise ValueError("recording work must be an existing directory outside parent artifact")
        record_options = (work, recording["max_journal_bytes"], recording["max_working_bytes"])
    private = load_native_module(model, artifact, level=level, load_run_id=load_run_id,
                                 inplace=False, input_gradients=True, **load_limits)
    # Derived buffers belong to this private shell too; do not mutate the caller's
    # nonpersistent state during training forwards.
    copies = {}
    for module in private.modules():
        for name in module._non_persistent_buffers_set:
            value = module._buffers.get(name)
            if value is not None:
                if id(value) not in copies:
                    copies[id(value)] = value.detach().clone()
                module._buffers[name] = copies[id(value)]
    weights = {weight._native_owner: weight for weight in private.modules() if type(weight) is NativeTernaryWeight}
    if any(owner not in weights for owner in data["owners"]):
        raise ValueError("native selected owner differs from loaded graph")
    selected = {name: weights[name] for name in data["owners"]}
    backend = artifact.begin_scale_training(text, [(name, selected[name].handle) for name in data["owners"]],
                                           max_snapshot_bytes=max_snapshot_bytes, max_parent_bytes=max_parent_bytes,
                                           checkpoint=checkpoint)
    trainer = NativeScaleTrainer(private, backend, data, selected, max_saved_input_bytes)
    if record_options is not None:
        work, journal_bytes, working_bytes = record_options
        temporary = tempfile.TemporaryDirectory(prefix="tritium-native-recording-", dir=work)
        trainer._recording_work = temporary
        try:
            backend.start_recording(str(Path(temporary.name) / "gradients.tnsj"), journal_bytes, working_bytes)
        except BaseException:
            trainer._cleanup_recording()
            raise
    return trainer


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate native training plan key")
        result[key] = value
    return result


__all__ = ["NativeScaleTrainer", "native_scale_training"]
