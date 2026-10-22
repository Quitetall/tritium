"""Public native ScaleOnly orchestration; observed consistency, not qualification."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from collections.abc import Mapping
import torch
import torch.nn.functional as F
from .. import _tritium
from .native_training import native_scale_training, _pairs
from .native_load import load_native_module
from .ptq import _hash_field, _hash_tensor, _source_model_digest, _invoke_model
from .errors import TritiumError

_TRAIN = {"max_snapshot_bytes", "max_parent_bytes", "max_saved_input_bytes", "max_package_bytes",
          "max_payload_bytes", "max_workspace_bytes", "max_output_bytes", "max_evidence_bytes"}
_PUBLISH = {"max_spool_bytes", "max_working_bytes", "max_evidence_bytes", "max_companion_bytes",
            "max_package_bytes", "max_manifest_bytes", "max_directory_bytes", "max_parent_bytes",
            "max_replay_working_bytes", "max_journal_bytes"}
_DATA = {"max_window_bytes", "max_validation_bytes", "max_nodes", "max_depth", "max_loss_bytes"}
_BUNDLE = {"max_bundle_bytes", "max_manifest_bytes", "max_checkpoint_bytes", "max_working_bytes"}
_FLOAT = {torch.float16, torch.bfloat16, torch.float32}
_TYPES = _FLOAT | {torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool}

def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def _limits(value, fields):
    if (not isinstance(value, dict) or set(value) != fields
            or any(type(n) is not int or n <= 0 for n in value.values())):
        raise ValueError("native refinement requires exact positive resource limit fields")
    return dict(value)

def _temperature(value):
    if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
        raise ValueError("native refinement temperature must be finite and positive")
    return float(value)

def _objective(temperature):
    return _json(dict(profile="tritium.native.logit-kd@1", temperature=_temperature(temperature),
                      mask="attention-mask-and-labels@1", roundoff="per-position-clamp-zero")).encode()

@dataclass(frozen=True, init=False)
class NativeRefinementConfig:
    """Frozen explicit native run inputs; not a legacy recipe or quality verdict."""
    _plan_json: str
    _resources_json: str
    level: str
    run_id: str
    machine: str
    temperature: float
    checkpoint_steps: tuple

    def __init__(self, plan, *, level, run_id, machine, training_limits, publication_limits,
                 data_limits, temperature=1.0, checkpoint_steps=(), checkpoint_limits=None):
        data = json.loads(plan, object_pairs_hook=_pairs) if isinstance(plan, str) else json.loads(_json(plan))
        text = _json(data)
        _tritium._validate_native_scale_plan(text)
        if not isinstance(level, str) or data["parent"]["level"] != level:
            raise ValueError("native refinement requires its explicit parent level")
        for label in (run_id, machine):
            if (not isinstance(label, str) or not 1 <= len(label) <= 128
                    or any(not (c.isascii() and (c.isalnum() or c in "-_.:")) for c in label)):
                raise ValueError("native refinement run/machine require bounded identifier labels")
        temperature = _temperature(temperature)
        if data["objective"] != self.objective_id(temperature=temperature):
            raise ValueError("native refinement objective differs from built-in KD profile")
        if data["teacher_source"] != data["parent"]["source"]:
            raise ValueError("native KD teacher_source must explicitly equal original parent source")
        resources = dict(training=_limits(training_limits, _TRAIN), publication=_limits(publication_limits, _PUBLISH),
                         data=_limits(data_limits, _DATA), checkpoint=None)
        if resources["data"]["max_depth"] > 64:
            raise ValueError("native data maximum depth cannot exceed 64")
        if type(checkpoint_steps) not in {tuple, list}:
            raise ValueError("checkpoint steps require an explicit bounded sequence")
        steps = tuple(checkpoint_steps)
        if (any(type(n) is not int or not 0 <= n <= len(data["training"]) for n in steps)
                or tuple(sorted(set(steps))) != steps):
            raise ValueError("checkpoint steps must be ordered unique declared cursors")
        if bool(steps) != (checkpoint_limits is not None):
            raise ValueError("checkpoint steps and limits must be supplied together")
        if steps:
            resources["checkpoint"] = _limits(checkpoint_limits, _BUNDLE)
        for name, value in dict(_plan_json=text, _resources_json=_json(resources), level=level,
                                run_id=run_id, machine=machine, temperature=temperature, checkpoint_steps=steps).items():
            object.__setattr__(self, name, value)

    @property
    def plan(self):
        return json.loads(self._plan_json)

    @property
    def resources(self):
        return json.loads(self._resources_json)

    @staticmethod
    def objective_id(*, temperature=1.0):
        return list(hashlib.sha256(b"tritium.native.refinement.objective.v1\0" + _objective(temperature)).digest())

    @staticmethod
    def window_id(batch, *, limits):
        limits = _limits(limits, _DATA)
        _measure(batch, limits)
        digest = hashlib.sha256(b"tritium.native.refinement.window.v1\0")
        _hash_node(digest, batch)
        return list(digest.digest())

def _measure(batch, limits):
    size, nodes = 0, 0
    def visit(value, depth, ancestors):
        nonlocal size, nodes
        nodes += 1
        if depth > min(64, limits["max_depth"]) or nodes > limits["max_nodes"]:
            raise ValueError("native window node/depth limit exceeded")
        if type(value) is torch.Tensor:
            if value.device.type != "cpu" or value.layout != torch.strided or value.is_quantized or value.dtype not in _TYPES:
                raise ValueError("native window tensors require ordinary supported CPU values")
            size += value.numel() * value.element_size()
        elif isinstance(value, Mapping) or type(value) in {tuple, list}:
            if id(value) in ancestors or nodes + len(value) > limits["max_nodes"]:
                raise ValueError("native window cycle/node limit exceeded")
            ancestors = ancestors | {id(value)}
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if not isinstance(key, str) or len(key) > limits["max_window_bytes"]:
                        raise ValueError("native window keys require bounded strings")
                    size += len(key.encode())
                    visit(item, depth + 1, ancestors)
            else:
                for item in value:
                    visit(item, depth + 1, ancestors)
        elif value is None or type(value) in {bool, int, float, str}:
            if ((type(value) is int and value.bit_length() > limits["max_window_bytes"] * 8)
                    or (type(value) is str and len(value) > limits["max_window_bytes"])):
                raise ValueError("native scalar window limit exceeded")
            size += len(_json(value).encode())
        else:
            raise TypeError("unsupported native refinement window node")
        if size > limits["max_window_bytes"]:
            raise ValueError("native window byte limit exceeded")
    visit(batch, 0, frozenset())
    return size

def _hash_node(digest, value):
    if type(value) is torch.Tensor:
        _hash_field(digest, "node", b"tensor")
        _hash_tensor(digest, "tensor", value)
    elif isinstance(value, Mapping):
        _hash_field(digest, "node", b"mapping")
        _hash_field(digest, "count", struct.pack("<Q", len(value)))
        for key in sorted(value):
            _hash_field(digest, "key", key.encode())
            _hash_node(digest, value[key])
    elif type(value) in {tuple, list}:
        _hash_field(digest, "node", type(value).__name__.encode())
        _hash_field(digest, "count", struct.pack("<Q", len(value)))
        for item in value:
            _hash_node(digest, item)
    else:
        _hash_field(digest, "node", type(value).__name__.encode())
        _hash_field(digest, "scalar", _json(value).encode())

def _snapshot(batch, limits, remaining=None):
    size = _measure(batch, limits)
    if remaining is not None and size > remaining:
        raise ValueError("native validation aggregate byte limit exceeded")
    identity = NativeRefinementConfig.window_id(batch, limits=limits)
    memo = {}
    def copy(value):
        if type(value) is torch.Tensor:
            if id(value) not in memo:
                memo[id(value)] = value.detach().clone(memory_format=torch.contiguous_format)
            return memo[id(value)]
        if isinstance(value, Mapping):
            return {key: copy(item) for key, item in value.items()}
        if type(value) in {list, tuple}:
            return type(value)(copy(item) for item in value)
        return value
    snapshot = copy(batch)
    if (NativeRefinementConfig.window_id(snapshot, limits=limits) != identity
            or NativeRefinementConfig.window_id(batch, limits=limits) != identity):
        raise ValueError("native window changed during snapshot")
    return snapshot, identity, size

def _unchanged(batch, identity, limits):
    if NativeRefinementConfig.window_id(batch, limits=limits) != identity:
        raise ValueError("native window mutated during forward/backward")

class _Teacher:
    def __init__(self, model):
        self.model = model
        self.flags = tuple((module, module.training) for module in model.modules())
        self.topology = tuple((n, id(m), type(m)) for n, m in model.named_modules(remove_duplicate=False))
        self.state = tuple((n, value, value._version) for n, value in model.state_dict(keep_vars=True).items())
        if any(v.device.type != "cpu" or v.layout != torch.strided for _, v, _ in self.state):
            raise ValueError("native KD teacher requires ordinary CPU persistent state")

    def check(self):
        current = self.model.state_dict(keep_vars=True)
        if (tuple((n, id(m), type(m)) for n, m in self.model.named_modules(remove_duplicate=False)) != self.topology
                or tuple(current) != tuple(n for n, _, _ in self.state)
                or any(current[n] is not v or v._version != version for n, v, version in self.state)
                or any(m.training for m, _ in self.flags)):
            raise ValueError("native KD teacher topology/state/mode changed")

    def restore(self):
        for module, training in self.flags:
            module.training = training

def _loss(student, teacher, batch, config):
    from .refinement import _output_tensor
    student, teacher = _output_tensor(student), _output_tensor(teacher)
    if (student.shape != teacher.shape or student.numel() == 0 or student.device.type != "cpu"
            or teacher.device.type != "cpu" or student.dtype not in _FLOAT or teacher.dtype not in _FLOAT):
        raise ValueError("native KD output geometry/dtype/device differs")
    if student.numel() * 32 > config.resources["data"]["max_loss_bytes"]:
        raise ValueError("native KD loss workspace byte limit exceeded")
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("native KD outputs must be finite")
    if student.ndim and student.shape[-1] > 1:
        temperature = config.temperature
        per = F.kl_div(F.log_softmax(student.float() / temperature, dim=-1),
                       F.log_softmax(teacher.float() / temperature, dim=-1),
                       reduction="none", log_target=True).sum(-1).clamp_min(0) * temperature**2
    else:
        per = (student.float() - teacher.float()).square()
        if per.ndim:
            per = per.mean(-1)
    valid = torch.ones_like(per, dtype=torch.bool)
    if isinstance(batch, Mapping):
        for key in ("attention_mask", "labels"):
            value = batch.get(key)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.shape != per.shape:
                raise ValueError("native KD mask geometry differs")
            if key == "attention_mask":
                if not bool(((value == 0) | (value == 1)).all()):
                    raise ValueError("native KD attention mask must be binary")
                valid &= value.bool()
            else:
                if value.dtype not in {torch.int64, torch.int32}:
                    raise ValueError("native KD labels must be integers")
                valid &= value != -100
    if not bool(valid.any()):
        raise ValueError("native KD mask has empty support")
    loss = per[valid].mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("native KD loss must be finite")
    return loss

def _evaluate(model, teacher, guard, windows, config):
    losses = []
    with torch.no_grad():
        for batch, identity in windows:
            guard.check()
            _unchanged(batch, identity, config.resources["data"])
            target = _invoke_model(teacher, batch)
            guard.check()
            _unchanged(batch, identity, config.resources["data"])
            loss = float(_loss(_invoke_model(model, batch), target, batch, config))
            _unchanged(batch, identity, config.resources["data"])
            losses.append(loss)
    return losses

def _train(trainer, teacher, guard, batch, identity, config):
    trainer.zero_grad()
    guard.check()
    with torch.no_grad():
        target = _invoke_model(teacher, batch)
    guard.check()
    _unchanged(batch, identity, config.resources["data"])
    loss = _loss(_invoke_model(trainer.model, batch), target, batch, config)
    _unchanged(batch, identity, config.resources["data"])
    loss.backward()
    guard.check()
    _unchanged(batch, identity, config.resources["data"])
    trainer.step(identity)

def _outside(a, b):
    return not (a == b or a in b.parents or b in a.parents)

def _new_directory(value):
    if value is None:
        raise ValueError("native refinement requires explicit output_dir")
    path = Path(value).absolute()
    path = path.parent.resolve(strict=True) / path.name
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    return path

def refine_native(parent, *, teacher, training, validation, config, work_dir, output_dir,
                  resume=None, checkpoint_dir=None, on_checkpoint=None):
    """Run source-consistent KD and observed retention; return an unqualified child."""
    if not isinstance(config, NativeRefinementConfig) or not isinstance(teacher, torch.nn.Module):
        raise TypeError("native refine requires NativeRefinementConfig and a dense teacher")
    if not isinstance(parent, _tritium.NativeModelConversion):
        raise TypeError("native refined-parent chaining is not admitted")
    if resume is not None and not isinstance(resume, _tritium.NativeScaleResumeBundle):
        raise TypeError("native refinement resume requires NativeScaleResumeBundle")
    if on_checkpoint is not None and (not callable(on_checkpoint) or not config.checkpoint_steps):
        raise ValueError("checkpoint callback requires an explicit checkpoint schedule")
    plan, resources = config.plan, config.resources
    if resume is not None and json.loads(resume.plan_json) != plan:
        raise ValueError("native refinement resume plan differs")
    output = _new_directory(output_dir)
    work = Path(work_dir).resolve(strict=True)
    source = Path(parent.output_dir).resolve(strict=True)
    if not work.is_dir() or not all(_outside(a, b) for a, b in ((work, source), (output, source), (output, work))):
        raise ValueError("native refinement output/work/parent paths overlap")
    checkpoints = None
    if config.checkpoint_steps:
        if checkpoint_dir is None:
            raise ValueError("native checkpoint schedule requires checkpoint_dir")
        checkpoints = Path(checkpoint_dir).resolve(strict=True)
        if not checkpoints.is_dir() or not all(_outside(checkpoints, p) for p in (work, output, source)):
            raise ValueError("native checkpoint root overlaps work/output/parent")
    elif checkpoint_dir is not None:
        raise ValueError("checkpoint_dir requires an explicit schedule")
    parent.verify()
    if json.loads(parent.native_refinement_parent(config.level)) != plan["parent"]:
        raise ValueError("native refinement actual parent tuple differs")
    catalog = json.loads(parent.asset("tritium-torch-state.json"))
    if catalog.get("adapter") != "tritium.torch.native-source@2":
        raise ValueError("native refinement requires executable source catalog v2")
    guard = _Teacher(teacher)
    if _source_model_digest(teacher) != catalog["source_model_digest"]:
        raise ValueError("native refinement teacher differs from original source catalog")
    windows, total = [], 0
    iterator = iter(validation)
    for requested in plan["validation"]:
        try:
            batch = next(iterator)
        except StopIteration as cause:
            raise ValueError("native validation stream is incomplete") from cause
        batch, identity, size = _snapshot(batch, resources["data"], resources["data"]["max_validation_bytes"] - total)
        if identity != requested["content"]:
            raise ValueError("native validation window identity differs")
        total += size
        windows.append((batch, identity))
    if next(iterator, _END) is not _END:
        raise ValueError("native validation stream has trailing windows")
    trainer = None
    try:
        teacher.eval()
        load_limits = {k: v for k, v in resources["training"].items() if k not in {
            "max_snapshot_bytes", "max_parent_bytes", "max_saved_input_bytes"}}
        if resume is not None:
            predecessor = load_native_module(teacher, parent, level=config.level,
                load_run_id=config.run_id + "-parent", inplace=False, **load_limits)
            predecessor.eval()
            before = _evaluate(predecessor, teacher, guard, windows, config)
            del predecessor
        trainer = native_scale_training(teacher, parent, level=config.level, plan=plan,
            load_run_id=config.run_id + "-load", resume=resume,
            recording=dict(work_dir=work, max_journal_bytes=resources["publication"]["max_journal_bytes"],
                           max_working_bytes=resources["publication"]["max_replay_working_bytes"]), **resources["training"])
        trainer.model.eval()
        if resume is None:
            before = _evaluate(trainer.model, teacher, guard, windows, config)
        start = trainer.revision
        def checkpoint():
            if trainer.revision in config.checkpoint_steps and (trainer.revision > start or (start == 0 and resume is None)):
                saved = trainer.save_resume(checkpoints / f"step-{trainer.revision:020d}", limits=resources["checkpoint"])
                if on_checkpoint is not None:
                    on_checkpoint(saved)
                guard.check()
                trainer._validate()
        checkpoint()
        iterator = iter(training)
        for index, requested in enumerate(plan["training"]):
            try:
                batch = next(iterator)
            except StopIteration as cause:
                raise ValueError("native training stream is incomplete") from cause
            batch, identity, _ = _snapshot(batch, resources["data"])
            if identity != requested["content"]:
                raise ValueError("native training window identity differs")
            if index < start:
                continue
            _train(trainer, teacher, guard, batch, identity, config)
            checkpoint()
        if next(iterator, _END) is not _END:
            raise ValueError("native training stream has trailing windows")
        after = _evaluate(trainer.model, teacher, guard, windows, config)
        guard.check()
        if _source_model_digest(teacher) != catalog["source_model_digest"]:
            raise ValueError("native refinement teacher bytes changed before publication")
        weights = [w["weight"] for w in plan["validation"]]
        previous = math.fsum(w * v for w, v in zip(weights, before)) / math.fsum(weights)
        candidate = math.fsum(w * v for w, v in zip(weights, after)) / math.fsum(weights)
        if not math.isfinite(previous) or not math.isfinite(candidate) or candidate > previous:
            raise TritiumError("native held-out predecessor retention failed", code="predecessor_retention_failed",
                               stage="refine", details=dict(predecessor_loss=previous, candidate_loss=candidate))
        observations = [dict(content=w["content"], parent_loss=a, child_loss=b)
                        for w, a, b in zip(plan["validation"], before, after)]
        result = trainer.publish(output, config.run_id, machine=config.machine, validation=observations,
                                 limits=resources["publication"])
    except BaseException:
        try:
            if trainer is not None:
                trainer.close()
        except BaseException:
            # Preserve the original refusal; no recursive/foreign cleanup fallback.
            pass
        finally:
            guard.restore()
        raise
    else:
        try:
            trainer.close()
        except BaseException as cause:
            result = result._with_cleanup_warning(str(cause))
        finally:
            guard.restore()
        return result

_END = object()
