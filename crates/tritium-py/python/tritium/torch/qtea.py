"""Opt-in live CPU Torch/HF capture bound to an existing complete QTEA file source.

No downloads, source checkpoint copies, default embedding metrics, estimator
measurements or qualification. Native metrics for non-Linear/shared consumers
must be supplied explicitly. Original files and live state are caller-owned.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from .. import _tritium, qtea as file_qtea
from .native_source import _registered_state, _identity, _graph_contract, _PRECISIONS
from .native_calibration import _capture_matmul, _encoded


def _ordinary(tensor, maximum=None):
    if (type(tensor) not in {torch.Tensor, nn.Parameter} or tensor.device.type != "cpu"
            or tensor.layout != torch.strided or tensor.is_quantized
            or tensor.dtype not in _PRECISIONS or not tensor.is_contiguous()
            or tensor.is_conj() or tensor.is_neg() or tensor.is_inference()):
        raise ValueError("QTEA live state must be ordinary contiguous CPU F16/BF16/F32 tensors")
    if maximum is not None and tensor.numel()*tensor.element_size()>maximum:
        raise ValueError("QTEA live tensor exceeds byte cap")


def _state(model, plan):
    registered = _registered_state(model)
    state=model.state_dict(keep_vars=True)
    if set(state) != set(registered) or any(state[n] is not t for n,t in registered.items()):
        raise ValueError("QTEA state_dict differs from registered persistent state")
    groups = {}; storage = {}
    for name, tensor in registered.items():
        _ordinary(tensor)
        identity = _identity(tensor)
        if identity[:2] in storage and storage[identity[:2]] != identity:
            raise ValueError("QTEA live state contains nonidentical shared views")
        storage[identity[:2]] = identity
        groups.setdefault(identity, []).append((name, tensor))
    names = {t["source"]["name"] for t in plan["tensors"]}
    result = []
    for members in groups.values():
        aliases = sorted(n for n, _ in members)
        canonical = names.intersection(aliases)
        if len(canonical) != 1:
            raise ValueError("QTEA live state must have exactly one stored canonical owner per alias group")
        owner = next(iter(canonical)); tensor = registered[owner]
        if any(value.requires_grad != tensor.requires_grad for _, value in members):
            raise ValueError("QTEA live aliases disagree on gradient ownership")
        result.append((owner, aliases, tensor))
    if {n for n, _, _ in result} != names:
        raise ValueError("QTEA live state inventory differs from original plan")
    return sorted(result, key=lambda entry: entry[0])


def _contract(model, entries):
    modules = dict(model.named_modules(remove_duplicate=False))
    graph_entries = []
    for owner, aliases, tensor in entries:
        supported = tensor.ndim == 2 and all(
            name.endswith(".weight") and type(modules.get(name.removesuffix(".weight")))
            in {nn.Linear, nn.Embedding} for name in aliases)
        graph_entries.append((owner, aliases, tensor, "parameter", supported, True))
    graph = _encoded(_graph_contract(model, graph_entries))
    topology = tuple((n, tuple(a), id(t), _identity(t), t.requires_grad,
                      None if t.grad is None else (id(t.grad), t.grad._version))
                     for n, a, t in entries)
    module_state = tuple((n, id(m), type(m), getattr(m.forward, "__func__", m.forward), m.training)
                         for n, m in modules.items())
    return graph, topology, module_state


@dataclass(frozen=True)
class QteaPreparedSource:
    """Exact file/live content binding, not an immutable snapshot or approval."""
    model: object
    observation: object
    request_json: str
    files_json: str
    max_json_bytes: int
    max_live_tensor_bytes: int
    _contract: tuple

    def verify(self):
        request = json.loads(self.request_json)
        observed = _tritium._prepare_qtea_file_source(
            self.request_json, self.files_json, max_json_bytes=self.max_json_bytes)
        if observed.source_id != self.observation.source_id:
            raise ValueError("QTEA original source bytes changed")
        entries = _state(self.model, request["source"])
        if _contract(self.model, entries) != self._contract:
            raise ValueError("QTEA live state ownership/graph/options changed")
        for name, _, tensor in entries:
            self.observation.validate_tensor_bytes(name, _encoded(_PRECISIONS[tensor.dtype]),
                list(tensor.shape), tensor.detach().view(torch.uint8).reshape(-1),
                self.max_live_tensor_bytes)
        return observed

    def calibrate(self, windows, objective, *, native_curvature, objective_bytes,
                  cache_limits, max_total_payload_bytes):
        """Capture actual guarded Linear inputs; all other matrix metrics are explicit.

        native_curvature maps owner -> {spec: native-spec, path: canonical-file}.
        This does not qualify those metrics or prove arbitrary producer semantics.
        """
        self.verify()
        request = json.loads(self.request_json); files = json.loads(self.files_json)
        native = {}
        matrix_names = {t["source"]["name"] for t in request["source"]["tensors"]
                        if len(t["source"]["shape"]) == 2}
        if not isinstance(native_curvature, dict) or set(native_curvature) - matrix_names:
            raise ValueError("native curvature must name explicit original matrix owners")
        for name, item in native_curvature.items():
            if not isinstance(item, dict) or set(item) != {"spec", "path"}:
                raise ValueError("native curvature requires explicit spec and path")
            spec = json.loads(_encoded(item["spec"])); path = os.fspath(Path(item["path"]).absolute())
            self.observation.validate_native_curvature(name, _encoded(spec), path,
                request["limits"]["max_curvature_file_bytes"])
            native[name] = dict(spec=spec, path=path)
        modules = dict(self.model.named_modules(remove_duplicate=False)); consumers = {}
        for owner, aliases, _ in _state(self.model, request["source"]):
            if owner not in matrix_names or owner in native:
                continue
            for alias in aliases:
                module = modules.get(alias.removesuffix(".weight")) if alias.endswith(".weight") else None
                if type(module) is not nn.Linear:
                    raise ValueError("embedding/shared/unsupported QTEA consumers require explicit native curvature")
                consumers[id(module)] = (module, owner)
        tokenizer = Path(files["assets"]["tokenizer.json"])
        if not tokenizer.is_file(): raise ValueError("source tokenizer must be ordinary")
        with tokenizer.open("rb") as stream:
            encoded = stream.read(request["source"]["limits"]["max_asset_bytes"] + 1)
        if len(encoded) > request["source"]["limits"]["max_asset_bytes"]:
            raise ValueError("source tokenizer exceeds asset cap")
        caches = _capture_matmul(self.model, self.observation, self.verify, consumers,
            windows, objective, tokenizer=encoded, objective_bytes=objective_bytes,
            cache_limits=cache_limits, max_total_payload_bytes=max_total_payload_bytes,
            output="None", guard_reads=True)
        return QteaModelCalibration(self, caches, _encoded(native), max_total_payload_bytes)


def prepare_qtea_source(model, request, *, source_shards, asset_paths,
                        max_live_tensor_bytes, max_json_bytes=16_777_216):
    """Bind a live CPU Torch/HF model to actual existing complete original files.

    request is tritium.qtea.source-preparation.v1: campaign, source QuantPlan,
    model_profile and limits, with no placeholder cache/run identities.
    """
    if not isinstance(model, nn.Module): raise TypeError("live QTEA source requires a Torch module")
    if type(max_live_tensor_bytes) is not int or max_live_tensor_bytes <= 0:
        raise ValueError("live tensor byte cap must be positive")
    for tensor in _registered_state(model).values():
        _ordinary(tensor,max_live_tensor_bytes)
    encoded = _encoded(request)
    files = _encoded(dict(source_shards=[os.fspath(Path(p).absolute()) for p in source_shards],
        assets={n: os.fspath(Path(p).absolute()) for n, p in asset_paths.items()}, curvature={}))
    observation = _tritium._prepare_qtea_file_source(encoded, files, max_json_bytes=max_json_bytes)
    entries = _state(model, request["source"])
    result = QteaPreparedSource(model, observation, encoded, files, max_json_bytes,
                                max_live_tensor_bytes, _contract(model, entries))
    result.verify()
    return result


@dataclass(frozen=True)
class QteaModelCalibration:
    """Real immutable native caches plus explicit external metrics; unqualified."""
    source: QteaPreparedSource
    caches: tuple
    native_json: str
    max_total_payload_bytes: int

    def _with_inputs(self, run, owner_options, work_dir, max_numeric_bytes,
                     max_batch_samples, consume):
        if any(type(v) is not int or v <= 0 for v in (max_numeric_bytes, max_batch_samples)):
            raise ValueError("positive numerical and replay chunk limits required")
        self.source.verify()
        base = json.loads(self.source.request_json); files = json.loads(self.source.files_json)
        native = json.loads(self.native_json)
        names = [t["source"]["name"] for t in base["source"]["tensors"] if len(t["source"]["shape"]) == 2]
        if not isinstance(owner_options, dict) or set(owner_options) != set(names):
            raise ValueError("QTEA owner recipe/profile/estimate inventory differs")
        options = json.loads(_encoded(owner_options))
        if any(set(value) != {"recipe", "profile", "estimate"} for value in options.values()):
            raise ValueError("QTEA owner options require explicit recipe/profile/estimate")
        if sum(cache.payload_bytes for _, cache in self.caches) > self.max_total_payload_bytes:
            raise ValueError("QTEA aggregate calibration payload exceeds cap")
        directory = Path(work_dir).resolve(strict=True)
        if not directory.is_dir(): raise ValueError("QTEA work_dir must exist")
        temporary = tempfile.TemporaryDirectory(prefix="tritium-qtea-capture-", dir=directory)
        result = None
        try:
            root = Path(temporary.name); curvature = {name: value["spec"] for name, value in native.items()}
            files["curvature"] = {name: value["path"] for name, value in native.items()}
            for index, (name, cache) in enumerate(self.caches):
                self.source.observation.validate_cache(cache)
                binding = json.loads(cache.binding_json)
                if binding["request"]["output"] != "None": raise ValueError("QTEA cache must be input-only")
                path = root / f"cache-{index}.trit"
                path.write_bytes(cache.package(base["limits"]["max_curvature_file_bytes"]))
                files["curvature"][name] = str(path)
                curvature[name] = dict(kind="calibration_cache", id=list(bytes.fromhex(cache.cache_id)),
                    limits=binding["request"]["limits"], max_numeric_bytes=max_numeric_bytes,
                    max_batch_samples=max_batch_samples)
            if set(curvature) != set(names): raise ValueError("QTEA complete curvature inventory differs")
            request = dict(base, schema="tritium.qtea.file-campaign.v1", run=run,
                owners=[dict(name=name, curvature=curvature[name], **options[name]) for name in names])
            result = consume(request, files, root)
        finally:
            try: temporary.cleanup()
            except OSError as error:
                if result is None: raise
                if isinstance(result, dict) and "publication_warnings" in result:
                    result["publication_warnings"].append(f"capture cache cleanup failed: {error}")
                else: raise
        return result

    def inspect(self, run, owner_options, *, work_dir, max_numeric_bytes,
                max_batch_samples, max_report_bytes=16_777_216):
        """Return explicit unqualified selection; no fitting or permanent cache writes."""
        def consume(request, files, root):
            return file_qtea.inspect(request, source_shards=files["source_shards"],
                asset_paths=files["assets"], curvature_paths=files["curvature"],
                max_json_bytes=self.source.max_json_bytes, max_report_bytes=max_report_bytes)
        return self._with_inputs(run, owner_options, work_dir, max_numeric_bytes, max_batch_samples, consume)

    def run(self, run, owner_options, inspection, *, output_dir, work_dir,
            max_numeric_bytes, max_batch_samples, max_report_bytes=16_777_216):
        """Recreate exact cache files, revalidate selection, fit and publish a candidate.

        A post-commit source verification/cleanup/report failure is not rollback.
        """
        def consume(request, files, root):
            result = file_qtea.run(request, inspection, source_shards=files["source_shards"],
                asset_paths=files["assets"], curvature_paths=files["curvature"],
                spool_path=root / "private-spool", output_dir=output_dir,
                max_json_bytes=self.source.max_json_bytes, max_report_bytes=max_report_bytes)
            try: self.source.verify()
            except Exception as error:
                raise ValueError("candidate committed; live source verification failed (not rollback)") from error
            return result
        return self._with_inputs(run, owner_options, work_dir, max_numeric_bytes, max_batch_samples, consume)


__all__ = ["QteaPreparedSource", "QteaModelCalibration", "prepare_qtea_source"]
