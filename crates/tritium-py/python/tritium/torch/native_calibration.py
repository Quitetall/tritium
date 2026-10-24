"""Explicit actual-source, Torch objective capture and shared native calibrated fitting."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import tempfile
from dataclasses import dataclass
from contextlib import nullcontext
import torch
from torch import nn
from .. import _tritium
from .native_source import _run_native_source, _inventory
from .native_load import _compact
from .module_artifacts import _pairs_without_duplicates, _sync_directory


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _observe(plan, shards, output, run, *, asset_paths, input_grams):
    if input_grams: raise ValueError("native calibration cannot silently reinterpret raw Grams")
    return _tritium._observe_model_plan_source(plan, shards, asset_paths)


def _check_window_context(window, specs):
    """Check declared context after caller execution; this is not writer custody."""
    for name, original, snapshot in specs:
        observed = window.get(name)
        if original is None:
            if observed is not None:
                raise ValueError("native window context changed during objective")
            continue
        if (not isinstance(observed, torch.Tensor) or observed.device.type != "cpu"
                or observed.dtype != original[0] or observed.shape != original[1]):
            raise ValueError("native window context changed during objective")
        current = _compact(observed.detach().reshape(-1).to(dtype=snapshot.dtype))
        if not torch.equal(current.view(torch.uint8), snapshot.view(torch.uint8)):
            raise ValueError("native window context changed during objective")


@dataclass(frozen=True)
class NativeModulePreparation:
    """Live source with native byte observation; not an immutable writer snapshot."""
    prepared: object
    observation: object
    options_json: str

    def verify(self):
        observed = _run_native_source(self.prepared, None, "native-source-check",
                                      consume=_observe, **json.loads(self.options_json))
        if observed.source_id != self.observation.source_id:
            raise ValueError("native prepared source content changed")
        return observed

    def capture_owner_window(self, window, objective, **limits):
        """Actual shared-owner window capture; durable calibrated fitting is separate."""
        from .owner_capture import capture_owner_window
        return capture_owner_window(self, window, objective, **limits)


def prepare_native_module(prepared, *, actions, levels, limits, work_dir, asset_paths,
                          architecture="torch-module"):
    """Observe complete actual stored source before calibration, without fitting."""
    options = dict(actions=actions, levels=list(levels), limits=limits, work_dir=os.fspath(Path(work_dir).resolve(strict=True)),
                   asset_paths={name:os.fspath(Path(path).absolute()) for name,path in asset_paths.items()}, architecture=architecture)
    options = json.loads(_encoded(options))
    observed = _run_native_source(prepared, None, "native-source-prepare", consume=_observe, **options)
    return NativeModulePreparation(prepared, observed, _encoded(options))


@dataclass(frozen=True)
class NativeModuleCalibration:
    """Actual immutable caches; structural byte binding is not release qualification."""
    source: NativeModulePreparation
    caches: tuple
    max_total_payload_bytes: int

    def save(self, output_dir, *, max_package_bytes):
        target = Path(output_dir).absolute()
        if target.exists() or target.is_symlink(): raise FileExistsError(target)
        if type(max_package_bytes) is not int or max_package_bytes <= 0:
            raise ValueError("cache package byte limit must be positive")
        if sum(cache.payload_bytes for _,cache in self.caches)>self.max_total_payload_bytes:
            raise ValueError("aggregate calibration payload exceeds byte limit")
        staging = Path(tempfile.mkdtemp(prefix=".tritium-native-cache-", dir=target.parent))
        published = False
        try:
            bindings = {}
            for index, (name,cache) in enumerate(self.caches):
                path = staging / f"cache-{index}.trit"
                with path.open("xb") as stream:
                    stream.write(cache.package(max_package_bytes))
                    stream.flush(); os.fsync(stream.fileno())
                bindings[name] = json.loads(cache.binding_json)
                reopened = _tritium.NativeCalibrationCache.open(str(path), cache.binding_json, max_package_bytes)
                if reopened.cache_id != cache.cache_id: raise ValueError("cache publication identity differs")
            metadata = _encoded(bindings).encode()
            if len(metadata) > 1024*1024: raise ValueError("cache binding metadata exceeds byte limit")
            with (staging / "bindings.json").open("xb") as stream:
                stream.write(metadata); stream.flush(); os.fsync(stream.fileno())
            _sync_directory(staging)
            _tritium.publish_directory_noreplace(str(staging), str(target))
            published = True
            _sync_directory(target.parent)
        finally:
            if not published: shutil.rmtree(staging)
        return target

    @classmethod
    def open(cls, source, directory, *, max_package_bytes, max_total_payload_bytes):
        if any(type(v) is not int or v<=0 for v in (max_package_bytes,max_total_payload_bytes)):
            raise ValueError("cache admission limits must be positive integers")
        source.verify()
        root = Path(directory).absolute()
        if root.is_symlink() or not root.is_dir(): raise ValueError("cache directory must be ordinary")
        path = root / "bindings.json"
        if path.is_symlink() or not path.is_file(): raise ValueError("cache bindings must be ordinary")
        with path.open("rb") as stream: encoded = stream.read(1024*1024+1)
        if len(encoded) > 1024*1024: raise ValueError("cache binding metadata exceeds byte limit")
        bindings = json.loads(encoded, object_pairs_hook=_pairs_without_duplicates)
        names = sorted(owner for owner,_,_,_,targeted,_ in _inventory(source.prepared) if targeted)
        if not isinstance(bindings,dict) or sorted(bindings) != names:
            raise ValueError("cache binding inventory differs from source")
        if {p.name for p in root.iterdir()} != {"bindings.json", *(f"cache-{i}.trit" for i in range(len(names)))}:
            raise ValueError("cache directory inventory differs")
        caches=[]; total=0
        for index,name in enumerate(names):
            path = root / f"cache-{index}.trit"
            if path.is_symlink() or not path.is_file(): raise ValueError("cache package must be ordinary")
            remaining=max_total_payload_bytes-total
            if remaining<=0: raise ValueError("aggregate calibration payload exceeds byte limit")
            cache = _tritium.NativeCalibrationCache.open(str(path), _encoded(bindings[name]), max_package_bytes,
                                                         max_payload_bytes=remaining)
            source.observation.validate_cache(cache)
            total += cache.payload_bytes
            if total > max_total_payload_bytes: raise ValueError("aggregate calibration payload exceeds byte limit")
            caches.append((name,cache))
        return cls(source,tuple(caches),max_total_payload_bytes)


def calibrate_native_module(source, windows, objective, *, objective_bytes, cache_limits,
                            max_total_payload_bytes, output="DenseFisher"):
    """Run actual objectives once and capture selected Linear inputs/loss gradients.

    Windows have CPU integer input_ids, one token per flattened projection sample.
    objective(model, window) returns a differentiable finite scalar for DenseFisher.
    Recipe bytes are declared context, not independent objective execution proof.
    """
    if not isinstance(source,NativeModulePreparation) or not callable(objective):
        raise TypeError("native capture requires preparation and objective callable")
    if output not in {"None","DenseFisher"}: raise ValueError("unsupported native capture output")
    if not isinstance(objective_bytes,bytes) or not objective_bytes:
        raise ValueError("actual declared objective bytes required")
    if type(max_total_payload_bytes) is not int or max_total_payload_bytes <= 0:
        raise ValueError("aggregate calibration byte limit must be positive")
    source.verify()
    entries = _inventory(source.prepared)
    if any(t.device.type != "cpu" for _,_,t,_,_,_ in entries):
        raise ValueError("native Torch capture is explicitly CPU-only")
    model = source.prepared.model
    modules = dict(model.named_modules(remove_duplicate=False))
    consumers = {}
    for owner,aliases,tensor,_,targeted,_ in entries:
        if not targeted: continue
        for alias in aliases:
            path = "" if alias == "weight" else alias.removesuffix(".weight")
            module = modules[path]
            if type(module) is not nn.Linear:
                raise ValueError("native capture requires Linear-only consumers; mixed gather/matmul is unresolved")
            consumers[id(module)] = (module,owner)
        if any(type(cache_limits.get(k)) is not int or cache_limits[k]<=0 for k in
               ("max_payload_bytes","max_samples","max_windows")):
            raise ValueError("native cache limits must be positive integers")
    options=json.loads(source.options_json)
    with Path(options["asset_paths"]["tokenizer.json"]).open("rb") as stream:
        tokenizer=stream.read(options["limits"]["max_asset_bytes"]+1)
    if len(tokenizer)>options["limits"]["max_asset_bytes"]:
        raise ValueError("tokenizer exceeds original asset byte limit")
    caches=_capture_matmul(model,source.observation,source.verify,consumers,windows,objective,
        tokenizer=tokenizer,objective_bytes=objective_bytes,cache_limits=cache_limits,
        max_total_payload_bytes=max_total_payload_bytes,output=output)
    return NativeModuleCalibration(source,caches,max_total_payload_bytes)


def _capture_matmul(model, observation, verify_source, consumers, windows, objective, *,
                    tokenizer, objective_bytes, cache_limits, max_total_payload_bytes,
                    output, guard_reads=False):
    """One actual cache producer. Caller validates source/consumer selection first."""
    if (not callable(objective) or output not in {"None","DenseFisher"}
            or not isinstance(objective_bytes,bytes) or not objective_bytes
            or type(max_total_payload_bytes) is not int or max_total_payload_bytes<=0
            or not isinstance(cache_limits,dict) or set(cache_limits)!={"max_payload_bytes","max_samples","max_windows"}
            or any(type(v) is not int or v<=0 for v in cache_limits.values()) or not consumers):
        raise ValueError("capture requires objective, consumers and positive explicit cache limits")
    if len({owner for _,owner in consumers.values()})*cache_limits["max_payload_bytes"]>max_total_payload_bytes:
        raise ValueError("aggregate declared cache payload exceeds byte limit")
    if guard_reads and torch.is_inference_mode_enabled():
        raise ValueError("guarded capture cannot use inference mode")
    builders={}; totals={}; sample_totals={}; window_totals={}
    for owner in {owner for _,owner in consumers.values()}:
        builders[owner] = observation.cache_builder(owner,tokenizer,objective_bytes,_encoded(output),_encoded(cache_limits))
        totals[owner]=len(tokenizer)+len(objective_bytes); sample_totals[owner]=0; window_totals[owner]=0
    records=[]; hooks=[]; current={}; active=[]
    mode=nullcontext()
    if guard_reads:
        from .owner_capture import _OwnerReads, _storage
        protected={_storage(m.weight):owner for m,owner in consumers.values()}
        def admit(frame,values,role):
            if role!="matmul": raise ValueError("QTEA capture requires Matmul reads")
            frame["input"]=values
        mode=_OwnerReads(protected,active,admit)
        for module,_ in consumers.values():
            if (type(module) is not nn.Linear or getattr(module.forward,"__func__",None) is not nn.Linear.forward
                    or module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or module._backward_pre_hooks):
                raise ValueError("guarded capture requires exact unhooked Linear forwards")
        def pre(module,args):
            active.append(dict(module=module))
    def hook(owner,module,args,value):
        inputs=args[0]
        if guard_reads:
            if not active or active[-1]["module"] is not module: raise ValueError("guarded invocation stack differs")
            frame=active.pop(); observed=frame.get("operator_output")
            if (observed is None or frame.get("input") is None or _storage(value)!=_storage(observed)
                    or value._version!=frame["output_version"]):
                raise ValueError("guarded invocation output differs")
        tokens=current["tokens"]
        if (not isinstance(inputs,torch.Tensor) or not isinstance(value,torch.Tensor)
                or inputs.device.type!="cpu" or value.device.type!="cpu"
                or inputs.ndim==0 or value.shape[:-1]!=inputs.shape[:-1]
                or inputs.shape[-1]!=module.in_features or value.shape[-1]!=module.out_features
                or inputs.numel()//module.in_features != tokens.numel()):
            raise ValueError("native projection/token sample geometry differs")
        n=tokens.numel()
        size=n*(13+4*module.in_features+(4*module.out_features if output=="DenseFisher" else 0))
        if (totals[owner]+size > cache_limits["max_payload_bytes"] or sample_totals[owner]+n>cache_limits["max_samples"]
                or window_totals[owner]+1>cache_limits["max_windows"]):
            raise ValueError("native cache window exceeds logical byte/sample/window limit")
        totals[owner]+=size; sample_totals[owner]+=n; window_totals[owner]+=1
        records.append((owner,id(module),_compact(inputs.detach().to(dtype=torch.float32).reshape(n,module.in_features).clone()),value))
    try:
        for module,owner in consumers.values():
            if guard_reads: hooks.append(module.register_forward_pre_hook(pre))
            hooks.append(module.register_forward_hook(lambda m,a,v,owner=owner: hook(owner,m,a,v)))
        for window in windows:
            tokens=window["input_ids"]
            if (not isinstance(tokens,torch.Tensor) or tokens.device.type!="cpu" or tokens.dtype not in {torch.int32,torch.int64}
                    or tokens.numel()==0 or tokens.numel()>cache_limits["max_samples"]):
                raise ValueError("native token window must be bounded CPU integer values")
            if torch.any(tokens<0).item() or torch.any(tokens>2**32-1).item():
                raise ValueError("native token values must be u32")
            selection = {}
            weights = window.get("calibration_weights")
            if weights is not None:
                if (not isinstance(weights,torch.Tensor) or weights.device.type!="cpu"
                        or not weights.dtype.is_floating_point or weights.shape!=tokens.shape
                        or not torch.isfinite(weights).all().item() or torch.any(weights<0).item()):
                    raise ValueError("calibration weights must be finite nonnegative CPU floating values matching input_ids")
                selection["weights"] = _compact(weights.detach().reshape(-1).to(dtype=torch.float64).clone())
            mask = window.get("calibration_mask")
            if mask is not None:
                if (not isinstance(mask,torch.Tensor) or mask.device.type!="cpu"
                        or mask.dtype!=torch.bool or mask.shape!=tokens.shape):
                    raise ValueError("calibration mask must be CPU bool values matching input_ids")
                selection["mask"] = _compact(mask.detach().reshape(-1).to(dtype=torch.uint8).clone())
            current["tokens"]=_compact(tokens.detach().reshape(-1).to(dtype=torch.int64).clone())
            specs = [("input_ids", (tokens.dtype, tokens.shape), current["tokens"]),
                     ("calibration_weights", None if weights is None else (weights.dtype,weights.shape), selection.get("weights")),
                     ("calibration_mask", None if mask is None else (mask.dtype,mask.shape), selection.get("mask"))]
            records.clear()
            with (torch.enable_grad() if output=="DenseFisher" else torch.no_grad()), mode:
                loss=objective(model,window)
                _check_window_context(window, specs)
                if {module_id for _,module_id,_,_ in records} != set(consumers): raise ValueError("required native consumers did not execute")
                if not isinstance(loss,torch.Tensor) or loss.ndim!=0 or not torch.isfinite(loss).item():
                    raise ValueError("native objective must return a finite scalar")
                if output=="DenseFisher":
                    gradients=torch.autograd.grad(loss, [value for _,_,_,value in records], allow_unused=False)
                else: gradients=[None]*len(records)
            for (owner,_,inputs,_),gradient in zip(records,gradients):
                kwargs={} if gradient is None else {"gradients":_compact(gradient.detach().to(dtype=torch.float32).reshape(inputs.shape[0],-1))}
                builders[owner].capture(current["tokens"],inputs,**selection,**kwargs)
            records.clear()
        verify_source()
        caches=tuple((name,builders[name].finish()) for name in sorted(builders))
        return caches
    finally:
        for handle in hooks: handle.remove()
        records.clear(); current.clear(); active.clear()


def convert_calibrated_native_module(calibration, output_dir, run_id, *, partition,
                                     damping, max_numeric_bytes, max_cache_package_bytes):
    """Fit through the distinct native calibrated lifecycle, then use load_native_module."""
    if not isinstance(calibration,NativeModuleCalibration): raise TypeError("actual native calibration required")
    source=calibration.source
    if any(type(v) is not int or v<=0 for v in (max_numeric_bytes,max_cache_package_bytes)):
        raise ValueError("native fit byte limits must be positive integers")
    for _,cache in calibration.caches:
        source.observation.validate_cache(cache)
        if cache.required_numeric_bytes(_encoded(partition),damping)>max_numeric_bytes:
            raise ValueError("native calibrated numerical workspace exceeds byte limit")
    source.verify()
    options=json.loads(source.options_json)
    def consume(plan,shards,output,run,*,asset_paths,input_grams):
        if input_grams: raise ValueError("native calibration cannot reinterpret raw Grams")
        calibrated=source.observation.calibrated_plan_json(plan,[cache for _,cache in calibration.caches],
            _encoded(partition),damping,max_numeric_bytes,calibration.max_total_payload_bytes,max_cache_package_bytes)
        temporary = tempfile.TemporaryDirectory(prefix="tritium-native-fit-cache-", dir=options["work_dir"])
        result = None
        try:
            paths={}
            for index,(name,cache) in enumerate(calibration.caches):
                path=Path(temporary.name)/f"cache-{index}.trit"
                path.write_bytes(cache.package(max_cache_package_bytes))
                paths[name]=str(path)
            result = _tritium.convert_model_plan(calibrated,shards,output,run,asset_paths=asset_paths,
                                                 calibration_caches=paths,calibrated=True)
        finally:
            try:
                temporary.cleanup()
            except OSError as error:
                if result is None:
                    raise
                result = result._with_source_cleanup_warning(f"CalibrationCacheCleanup: {error}")
        return result
    return _run_native_source(source.prepared,output_dir,run_id,consume=consume,**options)


__all__=["NativeModulePreparation","NativeModuleCalibration","prepare_native_module",
         "calibrate_native_module","convert_calibrated_native_module"]
