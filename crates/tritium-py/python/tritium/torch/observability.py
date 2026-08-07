"""Network-free ternary diagnostics with optional telemetry adapters."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import nn

from .estimators import (
    AbsMeanSTE,
    AdditiveEstimator,
    AnnealedSTE,
    HestiaEstimator,
    LSQEstimator,
    SaltSTE,
    SparseTernaryEstimator,
    TTQEstimator,
    TWNEstimator,
)
from .projection import ProjectionContext, TernaryProjection, validate_projection


@dataclass(frozen=True)
class TritHistogram:
    """Exact counts for the three legal ternary symbols."""

    negative: int
    zero: int
    positive: int

    @property
    def total(self) -> int:
        return self.negative + self.zero + self.positive

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.negative, self.zero, self.positive)


@dataclass(frozen=True)
class ScaleStatistics:
    """Bounded summary of one plane's deployment scales."""

    count: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float


@dataclass(frozen=True)
class PlaneDiagnostics:
    """Code and scale diagnostics for one physical ternary plane."""

    index: int
    trits: TritHistogram
    scales: ScaleStatistics
    group_size: int
    structure: str
    saturation_rate: Optional[float]


@dataclass(frozen=True)
class TensorDiagnostics:
    """Diagnostics for one unique latent master or packed weight owner."""

    path: str
    aliases: Tuple[str, ...]
    shape: Tuple[int, ...]
    mode: str
    estimator_id: Optional[str]
    codec_id: Optional[str]
    schema_version: int
    planes: Tuple[PlaneDiagnostics, ...]
    physical_bytes: int
    zero_rate: float
    reconstruction_rmse: Optional[float]
    saturation_rate: Optional[float]
    gradient_l2: Optional[float]
    gradient_finite: Optional[bool]

    @property
    def code_scale_bpw(self) -> float:
        """Packed code-plus-scale bits per base weight element."""

        elements = math.prod(self.shape)
        return self.physical_bytes * 8 / elements


@dataclass(frozen=True)
class TernaryDiagnostics:
    """Immutable diagnostics snapshot suitable for multiple telemetry sinks."""

    step: int
    tensors: Tuple[TensorDiagnostics, ...]
    extra_metrics: Tuple[Tuple[str, float], ...] = ()
    schema_version: int = 1

    @property
    def physical_bytes(self) -> int:
        """Packed code plus scale bytes, excluding complete-artifact overhead."""

        return sum(tensor.physical_bytes for tensor in self.tensors)

    @property
    def code_scale_bpw(self) -> float:
        """Aggregate code-plus-scale rate over unique base weight elements."""

        elements = sum(math.prod(tensor.shape) for tensor in self.tensors)
        return self.physical_bytes * 8 / elements

    def scalar_metrics(self, prefix: str = "tritium") -> Dict[str, float]:
        """Flatten the snapshot into deterministic scalar metric names."""

        prefix = _canonical_prefix(prefix)
        metrics = {
            f"{prefix}/tensors": float(len(self.tensors)),
            f"{prefix}/physical_bytes": float(self.physical_bytes),
            f"{prefix}/code_scale_bpw": self.code_scale_bpw,
        }
        for name, value in self.extra_metrics:
            metrics[f"{prefix}/{name}"] = value
        for tensor in self.tensors:
            root = f"{prefix}/tensor/{tensor.path}"
            metrics[f"{root}/planes"] = float(len(tensor.planes))
            metrics[f"{root}/physical_bytes"] = float(tensor.physical_bytes)
            metrics[f"{root}/code_scale_bpw"] = tensor.code_scale_bpw
            metrics[f"{root}/zero_rate"] = tensor.zero_rate
            optional = {
                "reconstruction_rmse": tensor.reconstruction_rmse,
                "saturation_rate": tensor.saturation_rate,
                "gradient_l2": tensor.gradient_l2,
            }
            for name, value in optional.items():
                if value is not None:
                    metrics[f"{root}/{name}"] = value
            if tensor.gradient_finite is not None:
                metrics[f"{root}/gradient_finite"] = float(tensor.gradient_finite)
            for plane in tensor.planes:
                plane_root = f"{root}/plane_{plane.index}"
                metrics[f"{plane_root}/negative"] = float(plane.trits.negative)
                metrics[f"{plane_root}/zero"] = float(plane.trits.zero)
                metrics[f"{plane_root}/positive"] = float(plane.trits.positive)
                metrics[f"{plane_root}/scale_min"] = plane.scales.minimum
                metrics[f"{plane_root}/scale_max"] = plane.scales.maximum
                metrics[f"{plane_root}/scale_mean"] = plane.scales.mean
                metrics[f"{plane_root}/scale_std"] = plane.scales.standard_deviation
                if plane.saturation_rate is not None:
                    metrics[f"{plane_root}/saturation_rate"] = plane.saturation_rate
        return metrics


class OpenTelemetryDiagnostics:
    """Reusable, aggregate-by-default OpenTelemetry diagnostics adapter."""

    def __init__(
        self,
        meter,
        *,
        include_tensors: bool = False,
        max_tensor_series: int = 256,
    ) -> None:
        if not isinstance(include_tensors, bool):
            raise TypeError("include_tensors must be bool")
        if (
            not isinstance(max_tensor_series, int)
            or isinstance(max_tensor_series, bool)
            or max_tensor_series <= 0
        ):
            raise ValueError("max_tensor_series must be a positive integer")
        self.meter = meter
        self.include_tensors = include_tensors
        self.max_tensor_series = max_tensor_series
        self._extra_metric_names = None
        self._tensor_identities = None
        self.model_instruments = {
            "tensors": meter.create_gauge("tritium.snapshot.tensor_count"),
            "physical_bytes": meter.create_gauge(
                "tritium.snapshot.code_scale_bytes", unit="By"
            ),
            "code_scale_bpw": meter.create_gauge(
                "tritium.snapshot.code_scale_bpw"
            ),
            "extra": meter.create_gauge("tritium.experiment.measurement"),
        }
        self.instruments = {
            "zero_rate": meter.create_gauge("tritium.tensor.zero_rate"),
            "physical_bytes": meter.create_gauge(
                "tritium.tensor.physical_bytes", unit="By"
            ),
            "code_scale_bpw": meter.create_gauge(
                "tritium.tensor.code_scale_bpw"
            ),
            "reconstruction_rmse": meter.create_gauge(
                "tritium.tensor.reconstruction_rmse"
            ),
            "saturation_rate": meter.create_gauge(
                "tritium.tensor.saturation_rate"
            ),
            "gradient_l2": meter.create_gauge("tritium.tensor.gradient_l2"),
            "gradient_finite": meter.create_gauge(
                "tritium.tensor.gradient_finite"
            ),
            "plane_saturation_rate": meter.create_gauge(
                "tritium.tensor.plane_saturation_rate"
            ),
            "trit_count": meter.create_gauge("tritium.tensor.trit_count"),
        }

    def log(self, snapshot: TernaryDiagnostics) -> None:
        """Record one snapshot without recreating metric instruments."""

        extra_metric_names = tuple(name for name, _value in snapshot.extra_metrics)
        if (
            self._extra_metric_names is not None
            and extra_metric_names != self._extra_metric_names
        ):
            raise ValueError("OpenTelemetry extra metric names changed after first log")
        tensor_identities = tuple(
            (tensor.path, tensor.estimator_id, tensor.codec_id, len(tensor.planes))
            for tensor in snapshot.tensors
        )
        if self.include_tensors:
            if (
                self._tensor_identities is not None
                and tensor_identities != self._tensor_identities
            ):
                raise ValueError("OpenTelemetry tensor identities changed after first log")
            series_count = sum(_otel_tensor_series(tensor) for tensor in snapshot.tensors)
            if series_count > self.max_tensor_series:
                raise ValueError("snapshot exceeds OpenTelemetry tensor-series budget")

        self._extra_metric_names = extra_metric_names
        if self.include_tensors:
            self._tensor_identities = tensor_identities
        self.model_instruments["tensors"].set(len(snapshot.tensors), {})
        self.model_instruments["physical_bytes"].set(snapshot.physical_bytes, {})
        self.model_instruments["code_scale_bpw"].set(snapshot.code_scale_bpw, {})
        for name, value in snapshot.extra_metrics:
            self.model_instruments["extra"].set(
                value, {"metric": name, "source": "caller"}
            )
        if not self.include_tensors:
            return
        for tensor in snapshot.tensors:
            attributes = {
                "tensor": tensor.path,
                "mode": tensor.mode,
            }
            if tensor.estimator_id is not None:
                attributes["estimator"] = tensor.estimator_id
            if tensor.codec_id is not None:
                attributes["codec"] = tensor.codec_id
            self.instruments["zero_rate"].set(tensor.zero_rate, attributes)
            self.instruments["physical_bytes"].set(
                tensor.physical_bytes, attributes
            )
            self.instruments["code_scale_bpw"].set(
                tensor.code_scale_bpw, attributes
            )
            for name in ("reconstruction_rmse", "saturation_rate", "gradient_l2"):
                value = getattr(tensor, name)
                if value is not None:
                    self.instruments[name].set(value, attributes)
            if tensor.gradient_finite is not None:
                self.instruments["gradient_finite"].set(
                    float(tensor.gradient_finite), attributes
                )
            for plane in tensor.planes:
                if plane.saturation_rate is not None:
                    self.instruments["plane_saturation_rate"].set(
                        plane.saturation_rate,
                        {**attributes, "plane": plane.index},
                    )
                for code, count in zip((-1, 0, 1), plane.trits.as_tuple()):
                    self.instruments["trit_count"].set(
                        count,
                        {**attributes, "plane": plane.index, "code": code},
                    )


def _otel_tensor_series(tensor: TensorDiagnostics) -> int:
    series = 3 + 3 * len(tensor.planes)
    series += sum(
        value is not None
        for value in (
            tensor.reconstruction_rmse,
            tensor.saturation_rate,
            tensor.gradient_l2,
            tensor.gradient_finite,
        )
    )
    series += sum(plane.saturation_rate is not None for plane in tensor.planes)
    return series


class WandbDiagnostics:
    """Reusable W&B adapter that rejects decreasing explicit steps."""

    def __init__(self, run, *, prefix: str = "tritium", histogram_factory=None) -> None:
        self.run = run
        self.prefix = _canonical_prefix(prefix)
        self.histogram_factory = histogram_factory
        self.last_step = -1

    def log(self, snapshot: TernaryDiagnostics) -> None:
        """Log one snapshot while enforcing W&B's monotonic-step contract."""

        if snapshot.step < self.last_step:
            raise ValueError("W&B diagnostics step must not decrease")
        _log_wandb_payload(
            snapshot,
            self.run,
            prefix=self.prefix,
            histogram_factory=self.histogram_factory,
        )
        self.last_step = snapshot.step


def _canonical_prefix(prefix: str) -> str:
    if not isinstance(prefix, str) or not prefix or prefix.strip("/") != prefix:
        raise ValueError("metric prefix must be nonempty and must not start or end with '/'")
    return prefix


_EXTRA_METRIC_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,199}\Z")


def _scale_statistics(scales: torch.Tensor) -> ScaleStatistics:
    values = scales.detach().to(device="cpu", dtype=torch.float64).flatten()
    return ScaleStatistics(
        count=values.numel(),
        minimum=float(values.min()),
        maximum=float(values.max()),
        mean=float(values.mean()),
        standard_deviation=float(values.std(unbiased=False)),
    )


def _histogram(trits: torch.Tensor) -> TritHistogram:
    detached = trits.detach()
    return TritHistogram(
        negative=int(torch.count_nonzero(detached == -1)),
        zero=int(torch.count_nonzero(detached == 0)),
        positive=int(torch.count_nonzero(detached == 1)),
    )


def _plane_diagnostics(
    projection: TernaryProjection,
    saturation_rates: Tuple[Optional[float], ...],
) -> Tuple[PlaneDiagnostics, ...]:
    return tuple(
        PlaneDiagnostics(
            index=index,
            trits=_histogram(plane.trits),
            scales=_scale_statistics(plane.scales),
            group_size=plane.group_size,
            structure=plane.structure,
            saturation_rate=saturation_rates[index],
        )
        for index, plane in enumerate(projection.planes)
    )


def _code_bytes(plane: PlaneDiagnostics, trits: torch.Tensor) -> int:
    if plane.structure == "dense":
        return (plane.trits.total + 4) // 5
    flattened = trits.detach().flatten()
    full_elements = flattened.numel() // 4 * 4
    quartets = flattened[:full_elements].reshape(-1, 4)
    if quartets.numel() and not bool(
        torch.all(torch.count_nonzero(quartets == 0, dim=1) == 1)
    ):
        raise ValueError("S34 diagnostics require exactly one zero per quartet")
    tail = flattened[full_elements:]
    if tail.numel() and int(torch.count_nonzero(tail == 0)) > 1:
        raise ValueError("S34 diagnostics tail is not canonically representable")
    stored_groups = (plane.trits.total + 3) // 4
    encoded_bits = stored_groups * 5
    return (encoded_bits + 7) // 8


def _gradient(weight: torch.Tensor) -> Tuple[Optional[float], Optional[bool]]:
    gradient = weight.grad
    if gradient is None:
        return None, None
    values = gradient.coalesce().values() if gradient.is_sparse else gradient
    finite = bool(torch.isfinite(values).all())
    if not finite:
        return None, False
    flattened = values.detach().flatten()
    maximum = float(flattened.abs().max()) if flattened.numel() else 0.0
    if maximum == 0:
        return 0.0, True
    scaled_squares = 0.0
    for chunk in flattened.split(1_000_000):
        scaled = (chunk / maximum).to(torch.float32)
        scaled_squares += float(torch.sum(scaled.square()))
    norm = maximum * math.sqrt(scaled_squares)
    if not math.isfinite(norm):
        return None, False
    return norm, True


_BUILTIN_ESTIMATOR_TYPES = {
    AbsMeanSTE,
    AnnealedSTE,
    HestiaEstimator,
    LSQEstimator,
    SaltSTE,
    SparseTernaryEstimator,
    TTQEstimator,
    TWNEstimator,
}


def _is_builtin_estimator(estimator: nn.Module) -> bool:
    if type(estimator) in _BUILTIN_ESTIMATOR_TYPES:
        return True
    return type(estimator) is AdditiveEstimator and all(
        _is_builtin_estimator(child) for child in estimator.estimators
    )


def _absmean_saturation(master: torch.Tensor) -> float:
    accumulation_dtype = (
        torch.float32
        if master.dtype in {torch.float16, torch.bfloat16}
        else master.dtype
    )
    scales = (
        master.detach()
        .to(accumulation_dtype)
        .abs()
        .mean(dim=1, keepdim=True)
        .to(master.dtype)
    )
    safe = scales.clamp_min(torch.finfo(master.dtype).tiny)
    active = (master.detach().abs() / safe < 1) & (scales > 0)
    return 1.0 - float(torch.mean(active.to(torch.float32)))


def _saturation_rates(
    estimator: nn.Module,
    master: torch.Tensor,
    projection: TernaryProjection,
) -> Tuple[Optional[float], ...]:
    rates = [None] * len(projection.planes)
    if type(estimator) in {AbsMeanSTE, SaltSTE}:
        rates[0] = _absmean_saturation(master)
    elif type(estimator) is AdditiveEstimator:
        residual = master
        decoded = torch.zeros_like(master)
        for index, child in enumerate(estimator.estimators):
            if type(child) in {AbsMeanSTE, SaltSTE}:
                rates[index] = _absmean_saturation(residual)
            plane = projection.planes[index]
            decoded = decoded + plane.trits.to(master.dtype) * plane.scales.to(
                master.dtype
            )
            residual = master - decoded
    return tuple(rates)


def _latent_diagnostics(
    path: str,
    aliases: Tuple[str, ...],
    module: nn.Module,
    *,
    step: int,
    allow_external_estimators: bool,
) -> TensorDiagnostics:
    weight = module.weight
    flat = weight.flatten(start_dim=1)
    if not _is_builtin_estimator(module.estimator) and not allow_external_estimators:
        raise ValueError(
            "external estimator diagnostics require allow_external_estimators=True"
        )
    with torch.no_grad():
        projection = module.estimator.project(
            flat,
            context=ProjectionContext(
                step=step,
                training=module.training,
                role="weight",
            ),
        )
        validate_projection(
            projection,
            flat,
            algorithm_id=module.estimator.algorithm_id,
            schema_version=module.estimator.schema_version,
        )
        saturation_rates = _saturation_rates(module.estimator, flat, projection)
        plane_metrics = _plane_diagnostics(projection, saturation_rates)
        error = projection.dense.detach().to(torch.float32) - flat.detach().to(
            torch.float32
        )
        reconstruction_rmse = float(torch.sqrt(torch.mean(error.square())))
        observed_saturation = [value for value in saturation_rates if value is not None]
        saturation_rate = (
            sum(observed_saturation) / len(observed_saturation)
            if observed_saturation
            else None
        )
    total_trits = sum(plane.trits.total for plane in plane_metrics)
    total_zeros = sum(plane.trits.zero for plane in plane_metrics)
    gradient_l2, gradient_finite = _gradient(weight)
    # Dense B3 stores five trits per byte; S34 stores five bits per quartet.
    # Deployment scales use the dtype emitted by the estimator. Complete
    # artifacts additionally carry manifests and state.
    physical_bytes = sum(
        _code_bytes(plane, projection.planes[index].trits)
        + projection.planes[index].scales.numel()
        * projection.planes[index].scales.element_size()
        for index, plane in enumerate(plane_metrics)
    )
    return TensorDiagnostics(
        path=path,
        aliases=aliases,
        shape=tuple(weight.shape),
        mode="latent",
        estimator_id=module.estimator.algorithm_id,
        codec_id=None,
        schema_version=module.estimator.schema_version,
        planes=plane_metrics,
        physical_bytes=physical_bytes,
        zero_rate=total_zeros / total_trits,
        reconstruction_rmse=reconstruction_rmse,
        saturation_rate=saturation_rate,
        gradient_l2=gradient_l2,
        gradient_finite=gradient_finite,
    )


def _hard_diagnostics(
    path: str,
    aliases: Tuple[str, ...],
    packed: nn.Module,
) -> TensorDiagnostics:
    packed.validate_buffers()
    counts = packed.trit_counts()
    planes = tuple(
        PlaneDiagnostics(
            index=index,
            trits=TritHistogram(*plane_counts),
            scales=_scale_statistics(getattr(packed, f"scales_{index}")),
            group_size=packed.in_features,
            structure="dense",
            saturation_rate=None,
        )
        for index, plane_counts in enumerate(counts)
    )
    total_trits = sum(plane.trits.total for plane in planes)
    return TensorDiagnostics(
        path=path,
        aliases=aliases,
        shape=(packed.out_features, packed.in_features),
        mode="hard",
        estimator_id=None,
        codec_id="tritium.b3-additive",
        schema_version=1,
        planes=planes,
        physical_bytes=packed.physical_bytes,
        zero_rate=sum(plane.trits.zero for plane in planes) / total_trits,
        reconstruction_rmse=None,
        saturation_rate=None,
        gradient_l2=None,
        gradient_finite=None,
    )


def collect_diagnostics(
    model: nn.Module,
    *,
    step: int = 0,
    paths: Optional[Iterable[str]] = None,
    max_latent_elements: Optional[int] = 1_000_000,
    allow_external_estimators: bool = False,
    extra_metrics: Optional[Mapping[str, float]] = None,
) -> TernaryDiagnostics:
    """Collect one bounded latent or hard ternary model snapshot.

    ``extra_metrics`` admits externally measured values such as teacher KL,
    runtime, resident memory, or physical artifact bpw without pretending that
    this in-process tensor walk measured them. Built-in estimators have a pure
    projection contract. Explicitly admitted external estimators may mutate
    their own state and are therefore outside that guarantee.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("diagnostics source must be a torch.nn.Module")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("diagnostics step must be a nonnegative integer")
    if max_latent_elements is not None and (
        not isinstance(max_latent_elements, int)
        or isinstance(max_latent_elements, bool)
        or max_latent_elements <= 0
    ):
        raise ValueError("max_latent_elements must be positive or None")
    if not isinstance(allow_external_estimators, bool):
        raise TypeError("allow_external_estimators must be bool")
    requested_paths = None
    if paths is not None:
        if isinstance(paths, str):
            raise TypeError("paths must be an iterable of weight paths, not a string")
        requested_paths = frozenset(paths)
        if not requested_paths or any(
            not isinstance(path, str) or not path for path in requested_paths
        ):
            raise ValueError("paths must contain canonical nonempty weight paths")
    extras = []
    if len(extra_metrics or {}) > 64:
        raise ValueError("extra_metrics exceeds the 64-series budget")
    for name, value in sorted((extra_metrics or {}).items()):
        if not isinstance(name, str) or not _EXTRA_METRIC_NAME.fullmatch(name):
            raise ValueError("extra metric names must be nonempty and canonical")
        if name in {"tensors", "physical_bytes", "code_scale_bpw"} or name.startswith(
            "tensor/"
        ):
            raise ValueError("extra metric name collides with a built-in metric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("extra metrics must be finite")
        extras.append((name, numeric))

    # Import after the torch facade has initialized. ``tritium.nn`` imports the
    # estimator package, so importing these module classes at file load time
    # would create a package-initialization cycle.
    from ..nn import (
        AdditiveTernaryEmbedding,
        AdditiveTernaryLinear,
        AdditiveTernaryWeight,
        TernaryConv1d,
        TernaryConv2d,
        TernaryEmbedding,
        TernaryLinear,
    )

    latent_types = (TernaryLinear, TernaryEmbedding, TernaryConv1d, TernaryConv2d)
    named_modules = tuple(model.named_modules(remove_duplicate=False))
    latent_groups = {}
    for path, module in named_modules:
        if isinstance(module, latent_types):
            latent_groups.setdefault(id(module.weight), []).append((path, module))
    hard_groups = {}
    for path, module in named_modules:
        if isinstance(module, (AdditiveTernaryLinear, AdditiveTernaryEmbedding)):
            alias = f"{path}.weight" if path else "weight"
            group = hard_groups.setdefault(
                id(module.packed_weight),
                {"packed": module.packed_weight, "aliases": []},
            )
            group["aliases"].append(alias)
    for path, module in named_modules:
        if isinstance(module, AdditiveTernaryWeight) and id(module) not in hard_groups:
            hard_groups[id(module)] = {
                "packed": module,
                "aliases": [path or "weight"],
            }

    available_paths = {
        alias
        for consumers in latent_groups.values()
        for path, _module in consumers
        for alias in (f"{path}.weight" if path else "weight",)
    }
    available_paths.update(
        alias for group in hard_groups.values() for alias in group["aliases"]
    )
    if requested_paths is not None:
        unknown = requested_paths - available_paths
        if unknown:
            raise ValueError(f"unknown diagnostics paths: {sorted(unknown)!r}")

    selected_latent = []
    latent_elements = 0
    for consumers in latent_groups.values():
        aliases = tuple(
            f"{path}.weight" if path else "weight" for path, _module in consumers
        )
        if requested_paths is not None and requested_paths.isdisjoint(aliases):
            continue
        estimator = consumers[0][1].estimator
        if any(module.estimator is not estimator for _path, module in consumers[1:]):
            raise ValueError(
                "tied latent consumers must share one estimator instance"
            )
        training = consumers[0][1].training
        if any(module.training is not training for _path, module in consumers[1:]):
            raise ValueError(
                "tied latent consumers must share one projection training mode"
            )
        module = consumers[0][1]
        latent_elements += module.weight.numel()
        selected_latent.append((aliases, module))
    if max_latent_elements is not None and latent_elements > max_latent_elements:
        raise ValueError(
            "selected latent diagnostics exceed max_latent_elements before projection"
        )

    tensors = [
        _latent_diagnostics(
            aliases[0],
            aliases,
            module,
            step=step,
            allow_external_estimators=allow_external_estimators,
        )
        for aliases, module in selected_latent
    ]
    for group in hard_groups.values():
        aliases = tuple(group["aliases"])
        if requested_paths is not None and requested_paths.isdisjoint(aliases):
            continue
        tensors.append(_hard_diagnostics(aliases[0], aliases, group["packed"]))
    if not tensors:
        raise ValueError("model contains no observable ternary weights")
    tensors.sort(key=lambda tensor: tensor.path)
    return TernaryDiagnostics(step=step, tensors=tuple(tensors), extra_metrics=tuple(extras))


def log_tensorboard(snapshot: TernaryDiagnostics, writer, *, prefix: str = "tritium") -> None:
    """Write diagnostics through an injected TensorBoard ``SummaryWriter``."""

    for tag, value in snapshot.scalar_metrics(prefix).items():
        writer.add_scalar(tag, value, snapshot.step)
    prefix = _canonical_prefix(prefix)
    for tensor in snapshot.tensors:
        for plane in tensor.planes:
            counts = list(plane.trits.as_tuple())
            writer.add_histogram_raw(
                tag=f"{prefix}/tensor/{tensor.path}/plane_{plane.index}/trits",
                min=-1.0,
                max=1.0,
                num=plane.trits.total,
                sum=float(plane.trits.positive - plane.trits.negative),
                sum_squares=float(plane.trits.positive + plane.trits.negative),
                bucket_limits=[-0.5, 0.5, 1.5],
                bucket_counts=counts,
                global_step=snapshot.step,
            )


def _log_wandb_payload(
    snapshot: TernaryDiagnostics,
    run,
    *,
    prefix: str = "tritium",
    histogram_factory=None,
) -> None:

    if histogram_factory is None:
        try:
            from wandb import Histogram as histogram_factory
        except ImportError as error:
            raise ImportError("log_wandb requires wandb or histogram_factory") from error
    prefix = _canonical_prefix(prefix)
    payload = snapshot.scalar_metrics(prefix)
    for tensor in snapshot.tensors:
        for plane in tensor.planes:
            payload[f"{prefix}/tensor/{tensor.path}/plane_{plane.index}/trits"] = (
                histogram_factory(
                    np_histogram=(plane.trits.as_tuple(), (-1.5, -0.5, 0.5, 1.5))
                )
            )
    run.log(payload, step=snapshot.step)


def log_wandb(
    snapshot: TernaryDiagnostics,
    run,
    *,
    prefix: str = "tritium",
    histogram_factory=None,
) -> None:
    """Write one diagnostics snapshot through an injected W&B run."""

    _log_wandb_payload(
        snapshot,
        run,
        prefix=prefix,
        histogram_factory=histogram_factory,
    )


def log_opentelemetry(
    snapshot: TernaryDiagnostics,
    meter,
) -> None:
    """Record one snapshot through an injected OTel ``Meter``.

    For a training loop, construct :class:`OpenTelemetryDiagnostics` once and
    reuse its ``log`` method so the provider sees one instrument set.
    """

    OpenTelemetryDiagnostics(meter).log(snapshot)
