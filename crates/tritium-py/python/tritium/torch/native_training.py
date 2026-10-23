"""Explicit eager CPU scale/PV training, recording and unqualified publication."""
from __future__ import annotations
import json
from types import MethodType
import weakref
import tempfile
from contextlib import nullcontext
from threading import RLock
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
        trainer = ctx.trainer()
        with getattr(trainer, "_capture_lock", None) or nullcontext():
            try:
                return fn(ctx, gradient)
            except BaseException:
                if trainer is not None:
                    trainer._failed = True
                    if getattr(trainer, "_capture_limits", None) is not None:
                        trainer._backend.poison_capture()
                raise
    return guarded


class _ScaleMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, leaf, handle, trainer, revision, lease, owner):
        ctx.trainer, ctx.revision, ctx.handle, ctx.lease = trainer, revision, handle, lease
        ctx.save_for_backward(value, leaf)
        ctx.owner = owner
        ctx.set_materialize_grads(False)
        return _matmul_output(value, handle, admit_backward=True)

    @staticmethod
    @_backward_guard
    def backward(ctx, gradient):
        if gradient is None:
            return (None,) * 7
        trainer = _guard(ctx)
        value, _ = ctx.saved_tensors  # Torch guards external input/leaf edits.
        _check_input_autograd(gradient)
        x = _compact(value.detach().to(dtype=torch.float32))
        dy = _compact(gradient.detach().to(dtype=torch.float32))
        operation = ctx.handle.pv_vjp if trainer._pv else ctx.handle.scale_vjp
        if getattr(trainer, "_capture_limits", None) is not None:
            trainer._backend.admit_capture(ctx.owner, ctx.revision)
        scales = torch.utils.dlpack.from_dlpack(operation(x, dy))
        if getattr(trainer, "_capture_limits", None) is not None:
            trainer._backend.capture_gradient(ctx.owner, ctx.revision, scales)
            scales = None
        dx = _input_vjp_output(gradient, ctx.handle) if ctx.needs_input_grad[0] else None
        return dx, scales, None, None, None, None, None


class _ScaleEmbed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, leaf, handle, trainer, revision, lease, padding, dtype, owner):
        ctx.trainer, ctx.revision, ctx.handle, ctx.lease = trainer, revision, handle, lease
        ctx.padding = padding
        ctx.owner = owner
        ctx.save_for_backward(indices, leaf)
        ctx.set_materialize_grads(False)
        source = _compact(indices.to(dtype=torch.int64))
        return torch.utils.dlpack.from_dlpack(handle.gather(source)).to(dtype=dtype)

    @staticmethod
    @_backward_guard
    def backward(ctx, gradient):
        if gradient is None:
            return (None,) * 9
        trainer = _guard(ctx)
        indices, _ = ctx.saved_tensors
        _check_input_autograd(gradient)
        if ctx.padding is not None:
            gradient = gradient.masked_fill((indices == ctx.padding).unsqueeze(-1), 0)
        ids = _compact(indices.to(dtype=torch.int64))
        dy = _compact(gradient.detach().to(dtype=torch.float32))
        operation = ctx.handle.embedding_pv_vjp if trainer._pv else ctx.handle.embedding_scale_vjp
        if getattr(trainer, "_capture_limits", None) is not None:
            trainer._backend.admit_capture(ctx.owner, ctx.revision)
        scales = torch.utils.dlpack.from_dlpack(operation(ids, dy))
        if getattr(trainer, "_capture_limits", None) is not None:
            trainer._backend.capture_gradient(ctx.owner, ctx.revision, scales)
            scales = None
        return None, scales, None, None, None, None, None, None, None


def _deny_apply(model, fn, recurse=True):
    raise RuntimeError("native scale training cannot change model dtype/device")


class NativeScaleTrainer:
    """Single-controller first-order CPU scale training over Rust atomic state.

    Use model/loss.backward(), zero_grad(), then step(window_identity). Gradient
    leaves are private f64 transport buffers, not torch.optim Parameters. Rust
    owns deployed scale values and updates. No concurrent backward/step, autocast,
    higher-order/compile/vmap or code training is admitted. Recorded publication
    and recovery use the common Rust writers/readers with explicit admission.
    """
    _pv = False
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
        if self._pv:
            model._tritium_native_unpublished_codes = True
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

    def _gradient_count(self, handle):
        return handle.scale_count + (handle.present_weight_count if self._pv else 0)

    def _leaf_count(self, handle):
        return self._gradient_count(handle)

    def _admit(self, handle, inputs, gradients, indices=0):
        count = (inputs + gradients) * 4 + handle.columns * 4 + self._gradient_count(handle) * 8 + indices * 8
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
        lease = self._saved.reserve(value.numel() * value.element_size() + self._leaf_count(handle) * 8)
        try:
            revision, handle, leaf = self._frame(owner, True)
            return _ScaleMatmul.apply(value, leaf, handle, weakref.ref(self), revision, lease, owner)
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
        lease = self._saved.reserve(indices.numel() * indices.element_size() + self._leaf_count(handle) * 8)
        try:
            revision, handle, leaf = self._frame(owner, True)
            return _ScaleEmbed.apply(indices, leaf, handle, weakref.ref(self), revision, lease, padding, dtype, owner)
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

    def save_resume(self, output_dir, *, limits):
        """Save committed state/history outside owned work; pending gradients/RNG are not saved.

        Each explicit save copies the whole prefix. Retain the returned bundle_id
        separately. This is unqualified recovery input, not a deployable child.
        """
        self._validate()
        if self._failed:
            raise RuntimeError("native backward failed; call zero_grad before reuse")
        if self._recording_work is None:
            raise RuntimeError("native resume save requires recorded training")
        fields = {"max_bundle_bytes", "max_manifest_bytes", "max_checkpoint_bytes", "max_working_bytes"}
        if (not isinstance(limits, dict) or set(limits) != fields
                or any(type(n) is not int or n <= 0 for n in limits.values())):
            raise ValueError("native resume requires exact positive bundle limit fields")
        return self._backend.save_resume(str(Path(output_dir).resolve()), json.dumps(limits, allow_nan=False))

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


class NativePvTrainer(NativeScaleTrainer):
    """Explicit eager CPU P/V controller using the common Rust optimizer.

    Private f64 leaves transport scale and present-weight gradients, never latent
    weight values. step() returns (changed scales, changed code units). First-order
    and graph/version restrictions are shared with native scale training.
    """
    _pv = True


class NativePvPagedTrainer(NativePvTrainer):
    """Recorded CPU P/V training with owner-paged optimizer state.

    Checkpoint bytes are intentionally unavailable: use save_resume(). Packed
    models and caller Torch leaves/gradients remain resident. Logical workspace
    reservations are not physical RAM/disk qualification.
    """
    def __init__(self, *args, gradient_capture=None, **kwargs):
        self._capture_limits = gradient_capture
        self._capture_lock = RLock() if gradient_capture is not None else None
        super().__init__(*args, **kwargs)

    def _leaf_count(self, handle):
        return 1 if self._capture_limits is not None else super()._leaf_count(handle)

    def _admit(self, handle, inputs, gradients, indices=0):
        super()._admit(handle, inputs, gradients, indices)
        if (self._capture_limits is not None
                and self._gradient_count(handle)*8+16384 > self._capture_limits.max_owner_working_bytes):
            raise ValueError("capture owner working cap exceeded")

    def zero_grad(self):
        with self._capture_lock or nullcontext():
            if self._capture_limits is not None:
                self._validate()
                self._failed = True
                self._backend.clear_capture()
            super().zero_grad()

    def step(self, window_identity):
        if self._capture_limits is None:
            return super().step(window_identity)
        with self._capture_lock:
            self._validate()
            if self._failed:
                raise RuntimeError("native backward failed; call zero_grad before reuse")
            if (not isinstance(window_identity, (bytes, bytearray, list, tuple)) or len(window_identity) != 32
                    or any(type(n) is not int or not 0 <= n <= 255 for n in window_identity)):
                raise ValueError("native training window identity requires 32 bytes")
            return self._backend.step_captured(self.revision, list(window_identity))
    @property
    def workspace_report(self):
        return json.loads(self._backend.workspace_report)

    def close(self):
        with self._capture_lock or nullcontext():
            return self._close()

    def _close(self):
        try:
            model = super().close()
        except BaseException as failure:
            # Invalid caller state is never admitted, but close must still retire
            # our disk/graph ownership. Preserve the original refusal as primary.
            try:
                self._cleanup_recording()
            except BaseException as cleanup:
                raise failure from cleanup
            finally:
                self.active = False
                self._cache.clear()
                self._original_handles.clear()
            raise
        self._original_handles.clear()
        return model

    def _cleanup_recording(self):
        if getattr(self, "_recording_work", None) is None:
            return
        report = json.loads(self._backend.close_recording())
        self._recording_work = None
        if report["cleanup_warnings"]:
            import warnings
            warnings.warn("; ".join(report["cleanup_warnings"]), RuntimeWarning, stacklevel=2)

    def publish(self, output_dir, run_id, *, machine, validation, limits):
        """Replay and publish an unqualified child without a checkpoint byte buffer.

        Source-owner and replay/journal caps are the frozen controller limits.
        Publication scratch covers one encoded owner plus fixed publisher I/O.
        """
        self._validate()
        if self._failed:
            raise RuntimeError("native backward failed; call zero_grad before reuse")
        fields = {"max_spool_bytes", "max_working_bytes", "max_evidence_bytes", "max_companion_bytes",
                  "max_package_bytes", "max_manifest_bytes", "max_directory_bytes"}
        if (not isinstance(limits, dict) or set(limits) != fields
                or any(type(n) is not int or n <= 0 for n in limits.values())):
            raise ValueError("paged publication requires exact positive limit fields")
        if not isinstance(run_id, str) or not run_id or not isinstance(machine, str) or not machine:
            raise ValueError("paged publication requires explicit run and machine")
        if not isinstance(validation, (list, tuple)):
            raise TypeError("bounded validation observations required")
        return self._backend.publish(str(Path(output_dir).resolve()), run_id, machine,
                                     json.dumps(validation, allow_nan=False), json.dumps(limits, allow_nan=False))


def _native_training(_pv, /, model, artifact, *, level, plan=None, load_run_id,
                          max_snapshot_bytes, max_parent_bytes, max_saved_input_bytes,
                          checkpoint=None, recording=None, resume=None, _paging=None, _load_limits=None):
    """Create a private native eager CPU scale trainer, never a refined artifact.

    plan is the complete Rust NativeRefinementPlan mapping/JSON; obtain its exact
    parent tuple with artifact.native_refinement_parent(level). One declared
    training window per step; pass its content identity to step(). Limits are
    logical, not complete Torch graph/process memory. Source shell is untouched.
    """
    # Public **load_limits must not bind private lifecycle/paging arguments.
    load_limits = {} if _load_limits is None else _load_limits
    if not isinstance(artifact, (_tritium.NativeModelConversion, _tritium.NativeModelRefinement)):
        raise TypeError("native scale training requires NativeModelConversion or NativeModelRefinement")
    if not isinstance(level, str) or not level or not isinstance(load_run_id, str) or not load_run_id:
        raise ValueError("explicit level and load run required")
    caps = (max_snapshot_bytes, max_saved_input_bytes) if _paging is not None else (max_snapshot_bytes, max_parent_bytes, max_saved_input_bytes)
    for limit in caps:
        if type(limit) is not int or limit <= 0:
            raise ValueError("native training byte limits must be positive integers")
    if resume is not None:
        expected_bundle = (_tritium.NativePvPagedResumeBundle if _paging is not None else
                           _tritium.NativePvResumeBundle if _pv else _tritium.NativeScaleResumeBundle)
        if not isinstance(resume, expected_bundle):
            raise TypeError(f"recorded resume requires {expected_bundle.__name__}")
        if checkpoint is not None:
            raise ValueError("checkpoint and recorded resume are mutually exclusive")
        if recording is None and _paging is None:
            raise ValueError("recorded resume requires a fresh recording work directory")
        if plan is None:
            plan = resume.plan_json
    if plan is None:
        raise ValueError("native training requires an explicit plan or recorded resume bundle")
    data = json.loads(plan, object_pairs_hook=_pairs) if isinstance(plan, str) else json.loads(json.dumps(plan))
    text = json.dumps(data, allow_nan=False)
    validator = _tritium._validate_native_pv_plan if _pv else _tritium._validate_native_scale_plan
    validator(text)
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
            raise ValueError("recorded training requires zero origin; use a recorded resume bundle instead of checkpoint")
        work = Path(recording["work_dir"]).resolve(strict=True)
        parent = Path(artifact.output_dir).resolve(strict=True)
        if not work.is_dir() or work == parent or parent in work.parents:
            raise ValueError("recording work must be an existing directory outside parent artifact")
        if resume is not None:
            saved = Path(resume.output_dir).resolve(strict=True)
            if work == saved or saved in work.parents:
                raise ValueError("resume work cannot overlap saved bundle")
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
    temporary, backend = None, None
    try:
        resumed = None
        if record_options is not None:
            work, journal_bytes, working_bytes = record_options
            temporary = tempfile.TemporaryDirectory(prefix="tritium-native-recording-", dir=work)
            journal_path = str(Path(temporary.name) / ("gradients.tnpj" if _pv else "gradients.tnsj"))
            if resume is not None:
                resumed = (resume, json.dumps(dict(path=journal_path, max_journal_bytes=journal_bytes,
                                                   max_working_bytes=working_bytes), allow_nan=False))
        matrices = [(name, selected[name].handle) for name in data["owners"]]
        if _paging is not None:
            workspace, paging_limits, gradient_capture = _paging
            backend = _tritium._begin_paged_pv_training(artifact, text, matrices, workspace,
                         limits=paging_limits, max_snapshot_bytes=max_snapshot_bytes, resume=resume,
                         gradient_capture=gradient_capture)
        else:
            arguments = dict(max_snapshot_bytes=max_snapshot_bytes, max_parent_bytes=max_parent_bytes, checkpoint=checkpoint)
            arguments["recorded_resume"] = resumed
            begin = artifact.begin_pv_training if _pv else artifact.begin_scale_training
            backend = begin(text, matrices, **arguments)
        if _paging is None and temporary is not None and resume is None:
            name = "gradients.tnpj" if _pv else "gradients.tnsj"
            backend.start_recording(str(Path(temporary.name) / name), journal_bytes, working_bytes)
        trainer_type = NativePvPagedTrainer if _paging is not None else NativePvTrainer if _pv else NativeScaleTrainer
        options = {"gradient_capture": gradient_capture} if _paging is not None else {}
        trainer = trainer_type(private, backend, data, selected, max_saved_input_bytes, **options)
        trainer._recording_work = True if _paging is not None else temporary
    except BaseException:
        if backend is not None:
            backend.close_recording()
        if temporary is not None:
            temporary.cleanup()
        raise
    return trainer


def native_scale_training(model, artifact, *, level, plan=None, load_run_id,
                          max_snapshot_bytes, max_parent_bytes, max_saved_input_bytes,
                          checkpoint=None, recording=None, resume=None, **load_limits):
    """Explicit eager CPU ScaleOnly training; see NativeScaleTrainer."""
    return _native_training(False, model, artifact, level=level, plan=plan, load_run_id=load_run_id,
                            max_snapshot_bytes=max_snapshot_bytes, max_parent_bytes=max_parent_bytes,
                            max_saved_input_bytes=max_saved_input_bytes, checkpoint=checkpoint,
                            recording=recording, resume=resume, _load_limits=load_limits)


def native_pv_training(model, artifact, *, level, plan=None, load_run_id,
                       max_snapshot_bytes, max_parent_bytes, max_saved_input_bytes,
                       checkpoint=None, recording=None, resume=None, **load_limits):
    """Explicit first-order CPU AlternatingPv training and unqualified publication.

    Raw checkpoint restart is unrecorded and may require two complete packed
    payload sets during admission. Recorded recovery requires an actual
    NativePvResumeBundle and fresh work. High-level refine's P/V path remains
    unavailable. No torch.optim or latent dense weight master.
    """
    return _native_training(True, model, artifact, level=level, plan=plan, load_run_id=load_run_id,
                            max_snapshot_bytes=max_snapshot_bytes, max_parent_bytes=max_parent_bytes,
                            max_saved_input_bytes=max_saved_input_bytes, checkpoint=checkpoint,
                            recording=recording, resume=resume, _load_limits=load_limits)


def native_pv_paged_training(model, artifact, *, level, plan=None, load_run_id, workspace_dir, limits,
                             max_snapshot_bytes, max_saved_input_bytes, resume=None, gradient_capture=None, **load_limits):
    """Distinct owner-paged, always-recorded CPU P/V controller.

    workspace_dir must be new and outside parent/resume artifacts. Anonymous
    current/next checkpoints, frame, journal and publication/replay spools share
    an aggregate workspace cap and at most six live files. Initial/replayed
    checkpoint reservations use the plan's checkpoint ceiling; later updates
    use exact checkpoint lengths. Actual disk space is a separate requirement.
    Only a NativePvPagedResumeBundle admits recorded restart; no raw bytes or
    silent resident fallback. Other load limits retain their original meanings.
    """
    if not isinstance(limits, _tritium.NativePvPagedTrainingLimits):
        raise TypeError("NativePvPagedTrainingLimits required")
    if gradient_capture is not None and not isinstance(gradient_capture, _tritium.NativePvPagedGradientLimits):
        raise TypeError("NativePvPagedGradientLimits required")
    return _native_training(True, model, artifact, level=level, plan=plan, load_run_id=load_run_id,
               max_snapshot_bytes=max_snapshot_bytes, max_parent_bytes=None,
               max_saved_input_bytes=max_saved_input_bytes, resume=resume,
               _paging=(str(Path(workspace_dir).resolve()), limits, gradient_capture), _load_limits=load_limits)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate native training plan key")
        result[key] = value
    return result


__all__ = ["NativeScaleTrainer", "native_scale_training", "NativePvTrainer", "native_pv_training",
           "NativePvPagedTrainer", "native_pv_paged_training"]
