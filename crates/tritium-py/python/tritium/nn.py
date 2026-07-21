"""First-class ternary :mod:`torch.nn` modules."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .torch.errors import TritiumError
from .torch.estimators import AbsMeanSTE, Estimator, SaltSTE
from .torch.ops import ternary_linear
from .torch.projection import ProjectionContext, validate_projection

_B3_MAX_VALID_BYTE = 3**5 - 1


def _estimator_extra_state(estimator: Estimator) -> torch.Tensor:
    """Encode estimator identity as a safetensors-compatible tensor."""

    encoded = f"{estimator.schema_version}\0{estimator.algorithm_id}".encode("utf-8")
    return torch.tensor(tuple(encoded), dtype=torch.uint8)


def _validate_estimator_extra_state(estimator: Estimator, state: Any) -> None:
    # Read legacy development checkpoints emitted before the v1.1 HF gate, but
    # never produce Python-object extra state again.
    if isinstance(state, dict):
        expected = {
            "schema_version": 1,
            "algorithm_id": estimator.algorithm_id,
            "estimator_schema_version": estimator.schema_version,
        }
        observed = state
    elif isinstance(state, torch.Tensor) and state.dtype == torch.uint8 and state.ndim == 1:
        try:
            schema, algorithm_id = bytes(state.detach().cpu().tolist()).decode("utf-8").split(
                "\0", 1
            )
            observed = (int(schema), algorithm_id)
        except (UnicodeDecodeError, ValueError) as error:
            raise TritiumError(
                "state estimator identity is malformed",
                code="state_identity",
                stage="load",
            ) from error
        expected = (estimator.schema_version, estimator.algorithm_id)
    else:
        raise TritiumError(
            "state estimator identity is not a uint8 tensor",
            code="state_identity",
            stage="load",
        )
    if observed != expected:
        raise TritiumError(
            "state estimator identity does not match module estimator",
            code="state_identity",
            stage="load",
            details={"expected": expected, "observed": observed},
        )


def _project(
    estimator: Estimator, weight: torch.Tensor, *, training: bool
) -> torch.Tensor:
    projection = estimator.project(
        weight,
        context=ProjectionContext(training=training, role="weight"),
    )
    validate_projection(
        projection,
        weight,
        algorithm_id=estimator.algorithm_id,
        schema_version=estimator.schema_version,
    )
    return projection.dense


class AdditiveTernaryWeight(nn.Module):
    """Shared inference-only additive ternary matrix storage."""

    def __init__(
        self,
        planes: Sequence[Any],
    ) -> None:
        super().__init__()
        if not isinstance(planes, Sequence) or not 1 <= len(planes) <= 3:
            raise ValueError("AdditiveTernaryWeight requires one to three planes")
        first = planes[0]
        if first.trits.ndim != 2:
            raise ValueError("additive ternary linear trits must be rank 2")
        self.out_features, self.in_features = map(int, first.trits.shape)
        self.weight_elements = self.out_features * self.in_features
        self.plane_count = len(planes)
        for index, plane in enumerate(planes):
            if (
                tuple(plane.trits.shape) != (self.out_features, self.in_features)
                or tuple(plane.scales.shape) != (self.out_features, 1)
                or plane.group_size != self.in_features
            ):
                raise ValueError("additive ternary linear plane geometry differs")
            trits = plane.trits.detach().to(dtype=torch.int8).contiguous()
            scales = plane.scales.detach().to(dtype=torch.float16).contiguous()
            if not bool(torch.all((trits >= -1) & (trits <= 1))):
                raise ValueError("additive ternary linear contains non-ternary values")
            if not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
                raise ValueError("additive ternary linear contains invalid scales")
            digits = (trits.flatten().to(torch.int16) + 1)
            padding = (-digits.numel()) % 5
            if padding:
                digits = F.pad(digits, (0, padding))
            powers = digits.new_tensor((1, 3, 9, 27, 81))
            packed = (digits.reshape(-1, 5) * powers).sum(dim=1).to(torch.uint8)
            self.register_buffer(f"packed_trits_{index}", packed)
            self.register_buffer(f"scales_{index}", scales)
        self.training = False

    @classmethod
    def empty(
        cls,
        in_features: int,
        out_features: int,
        planes: int,
        *,
        device=None,
    ) -> "AdditiveTernaryWeight":
        """Build a metadata-only compact state shell."""

        if in_features <= 0 or out_features <= 0 or not 1 <= planes <= 3:
            raise ValueError("additive ternary linear shell geometry is invalid")
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.in_features = int(in_features)
        module.out_features = int(out_features)
        module.weight_elements = module.in_features * module.out_features
        module.plane_count = int(planes)
        packed_elements = (module.weight_elements + 4) // 5
        for index in range(module.plane_count):
            module.register_buffer(
                f"packed_trits_{index}",
                torch.empty(packed_elements, dtype=torch.uint8, device=device),
            )
            module.register_buffer(
                f"scales_{index}",
                torch.empty(
                    (module.out_features, 1), dtype=torch.float16, device=device
                ),
            )
        module.training = False
        return module

    def validate_buffers(self) -> None:
        """Fail closed after external state loading."""

        for index in range(self.plane_count):
            packed = getattr(self, f"packed_trits_{index}")
            scales = getattr(self, f"scales_{index}")
            if packed.dtype != torch.uint8 or packed.numel() != (
                self.weight_elements + 4
            ) // 5:
                raise TritiumError(
                    "packed ternary state geometry is invalid",
                    code="state_geometry",
                    stage="load",
                )
            if packed.device.type != "meta" and bool(
                (packed > _B3_MAX_VALID_BYTE).any()
            ):
                raise TritiumError(
                    "packed ternary state contains invalid B3 bytes",
                    code="state_domain",
                    stage="load",
                )
            if (
                scales.dtype != torch.float16
                or tuple(scales.shape) != (self.out_features, 1)
            ):
                raise TritiumError(
                    "packed ternary scales have invalid geometry",
                    code="state_geometry",
                    stage="load",
                )
            if scales.device.type != "meta" and (
                not bool(torch.isfinite(scales).all()) or bool((scales < 0).any())
            ):
                raise TritiumError(
                    "packed ternary scales have invalid values",
                    code="state_domain",
                    stage="load",
                )

    @property
    def physical_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.buffers()
            if tensor is not None
        )

    def dense(self, *, dtype: torch.dtype) -> torch.Tensor:
        """Decode the additive matrix for the portable reference path."""

        output = None
        for index in range(self.plane_count):
            packed = getattr(self, f"packed_trits_{index}")
            powers = packed.new_tensor((1, 3, 9, 27, 81), dtype=torch.int16)
            digits = (packed.to(torch.int16).unsqueeze(1) // powers.unsqueeze(0)) % 3
            trits = (digits.flatten()[: self.weight_elements] - 1).reshape(
                self.out_features, self.in_features
            ).to(dtype=dtype)
            scales = getattr(self, f"scales_{index}").to(dtype=dtype)
            plane = trits * scales
            output = plane if output is None else output + plane
        return output


class _AdditiveTernaryConsumer(nn.Module):
    def _bind_packed_weight(
        self, packed_weight: AdditiveTernaryWeight, *, owner: bool
    ) -> None:
        if not isinstance(packed_weight, AdditiveTernaryWeight):
            raise TypeError("packed_weight must be AdditiveTernaryWeight")
        if owner:
            self.add_module("_packed_weight", packed_weight)
        else:
            object.__setattr__(self, "_packed_weight", packed_weight)

    @property
    def packed_weight(self) -> AdditiveTernaryWeight:
        return self._packed_weight

    def validate_buffers(self) -> None:
        self.packed_weight.validate_buffers()

    @property
    def physical_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.buffers()
            if tensor is not None
        )


class AdditiveTernaryLinear(_AdditiveTernaryConsumer):
    """Inference-only linear layer backed by additive trits and row scales."""

    def __init__(
        self,
        planes: Sequence[Any],
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self._initialize(AdditiveTernaryWeight(planes), bias=bias, owner=True)

    def _initialize(
        self,
        packed_weight: AdditiveTernaryWeight,
        *,
        bias: Optional[torch.Tensor],
        owner: bool,
    ) -> None:
        self.in_features = packed_weight.in_features
        self.out_features = packed_weight.out_features
        self.weight_elements = packed_weight.weight_elements
        self.plane_count = packed_weight.plane_count
        self._bind_packed_weight(packed_weight, owner=owner)
        if bias is not None:
            if bias.ndim != 1 or bias.shape[0] != self.out_features:
                raise ValueError("additive ternary linear bias geometry differs")
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.register_buffer("bias", None)
        self.training = False

    @classmethod
    def from_packed_weight(
        cls,
        packed_weight: AdditiveTernaryWeight,
        bias: Optional[torch.Tensor] = None,
        *,
        owner: bool = False,
    ) -> "AdditiveTernaryLinear":
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module._initialize(packed_weight, bias=bias, owner=owner)
        return module

    @classmethod
    def empty(
        cls,
        in_features: int,
        out_features: int,
        planes: int,
        *,
        bias: bool,
        device=None,
        dtype=None,
        packed_weight: AdditiveTernaryWeight | None = None,
        owner: bool = True,
    ) -> "AdditiveTernaryLinear":
        if packed_weight is None:
            packed_weight = AdditiveTernaryWeight.empty(
                in_features, out_features, planes, device=device
            )
        elif (packed_weight.in_features, packed_weight.out_features) != (
            in_features,
            out_features,
        ):
            raise ValueError("additive ternary linear shell geometry differs")
        preserved_bias = (
            torch.empty(out_features, dtype=dtype, device=device) if bias else None
        )
        return cls.from_packed_weight(
            packed_weight, preserved_bias, owner=owner
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.dtype.is_floating_point:
            raise TypeError("additive ternary linear input must be floating point")
        return F.linear(
            input,
            self.packed_weight.dense(dtype=input.dtype),
            self.bias.to(dtype=input.dtype) if self.bias is not None else None,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"planes={self.plane_count}, bias={self.bias is not None}"
        )


class AdditiveTernaryEmbedding(_AdditiveTernaryConsumer):
    """Inference-only embedding backed by shared additive ternary storage."""

    def __init__(
        self,
        packed_weight: AdditiveTernaryWeight,
        *,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        dtype: torch.dtype = torch.float32,
        owner: bool = True,
    ) -> None:
        super().__init__()
        if max_norm is not None:
            raise ValueError("additive ternary embedding does not support max_norm")
        self.num_embeddings = packed_weight.out_features
        self.embedding_dim = packed_weight.in_features
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.output_dtype = dtype
        self._bind_packed_weight(packed_weight, owner=owner)
        self.training = False

    @classmethod
    def empty(
        cls,
        num_embeddings: int,
        embedding_dim: int,
        planes: int,
        *,
        padding_idx: int | None,
        max_norm: float | None,
        norm_type: float,
        scale_grad_by_freq: bool,
        sparse: bool,
        device=None,
        dtype: torch.dtype = torch.float32,
        packed_weight: AdditiveTernaryWeight | None = None,
        owner: bool = True,
    ) -> "AdditiveTernaryEmbedding":
        if packed_weight is None:
            packed_weight = AdditiveTernaryWeight.empty(
                embedding_dim, num_embeddings, planes, device=device
            )
        elif (packed_weight.out_features, packed_weight.in_features) != (
            num_embeddings,
            embedding_dim,
        ):
            raise ValueError("additive ternary embedding shell geometry differs")
        return cls(
            packed_weight,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            dtype=dtype,
            owner=owner,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input,
            self.packed_weight.dense(dtype=self.output_dtype),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"planes={self.packed_weight.plane_count}, padding_idx={self.padding_idx}"
        )


class TernaryLinear(nn.Module):
    """Linear layer with latent floating weights and a hard ternary forward."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        estimator: Optional[Estimator] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.bias = (
            nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        )
        self.estimator = estimator if estimator is not None else AbsMeanSTE()
        self.estimator.to(device=self.weight.device)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_float(
        cls, module: nn.Linear, *, estimator: Optional[Estimator] = None
    ) -> "TernaryLinear":
        """Convert without cloning the source parameters."""

        if not isinstance(module, nn.Linear):
            raise TypeError("TernaryLinear.from_float requires torch.nn.Linear")
        converted = cls.__new__(cls)
        nn.Module.__init__(converted)
        converted.in_features = module.in_features
        converted.out_features = module.out_features
        converted.weight = module.weight
        converted.bias = module.bias
        converted.estimator = estimator if estimator is not None else AbsMeanSTE()
        converted.estimator.to(device=module.weight.device)
        converted.train(module.training)
        return converted

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if type(self.estimator) in {AbsMeanSTE, SaltSTE}:
            return ternary_linear(input, self.weight, self.bias)
        return F.linear(
            input,
            _project(self.estimator, self.weight, training=self.training),
            self.bias,
        )

    def get_extra_state(self):
        return _estimator_extra_state(self.estimator)

    def set_extra_state(self, state) -> None:
        _validate_estimator_extra_state(self.estimator, state)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, estimator={self.estimator.algorithm_id!r}"
        )


class TernaryEmbedding(nn.Module):
    """Embedding table with a latent floating weight and hard ternary lookup."""

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        *,
        estimator: Optional[Estimator] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("num_embeddings and embedding_dim must be positive")
        if padding_idx is not None:
            if padding_idx >= num_embeddings or padding_idx < -num_embeddings:
                raise ValueError("padding_idx must be within num_embeddings")
            if padding_idx < 0:
                padding_idx += num_embeddings
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = float(norm_type)
        self.scale_grad_by_freq = bool(scale_grad_by_freq)
        self.sparse = bool(sparse)
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        self.estimator = estimator if estimator is not None else AbsMeanSTE()
        self.estimator.to(device=self.weight.device)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    @classmethod
    def from_float(
        cls, module: nn.Embedding, *, estimator: Optional[Estimator] = None
    ) -> "TernaryEmbedding":
        """Convert without cloning the source table."""

        if not isinstance(module, nn.Embedding):
            raise TypeError("TernaryEmbedding.from_float requires torch.nn.Embedding")
        converted = cls.__new__(cls)
        nn.Module.__init__(converted)
        converted.num_embeddings = module.num_embeddings
        converted.embedding_dim = module.embedding_dim
        converted.padding_idx = module.padding_idx
        converted.max_norm = module.max_norm
        converted.norm_type = module.norm_type
        converted.scale_grad_by_freq = module.scale_grad_by_freq
        converted.sparse = module.sparse
        converted.weight = module.weight
        converted.estimator = estimator if estimator is not None else AbsMeanSTE()
        converted.estimator.to(device=module.weight.device)
        converted.train(module.training)
        return converted

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        projected = _project(self.estimator, self.weight, training=self.training)
        return F.embedding(
            input,
            projected,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def get_extra_state(self):
        return _estimator_extra_state(self.estimator)

    def set_extra_state(self, state) -> None:
        _validate_estimator_extra_state(self.estimator, state)

    def extra_repr(self) -> str:
        options = [f"{self.num_embeddings}", f"{self.embedding_dim}"]
        if self.padding_idx is not None:
            options.append(f"padding_idx={self.padding_idx}")
        if self.max_norm is not None:
            options.append(f"max_norm={self.max_norm}")
        if self.norm_type != 2:
            options.append(f"norm_type={self.norm_type}")
        if self.scale_grad_by_freq:
            options.append("scale_grad_by_freq=True")
        if self.sparse:
            options.append("sparse=True")
        options.append(f"estimator={self.estimator.algorithm_id!r}")
        return ", ".join(options)


class _TernaryConvNd(nn.Module):
    """Shared parameter/state mechanics for direct ternary convolutions."""

    def _adopt(self, module, estimator: Optional[Estimator]) -> None:
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.kernel_size = module.kernel_size
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.transposed = module.transposed
        self.output_padding = module.output_padding
        self.groups = module.groups
        self.padding_mode = module.padding_mode
        self._reversed_padding_repeated_twice = module._reversed_padding_repeated_twice
        self.weight = module.weight
        self.bias = module.bias
        self.estimator = estimator if estimator is not None else AbsMeanSTE()
        self.estimator.to(device=self.weight.device)
        self.train(module.training)

    def _projected_weight(self) -> torch.Tensor:
        flat = self.weight.flatten(start_dim=1)
        return _project(self.estimator, flat, training=self.training).reshape_as(self.weight)

    def _padded_input(self, input: torch.Tensor):
        if self.padding_mode == "zeros":
            return input, self.padding
        return (
            F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
            0,
        )

    def get_extra_state(self):
        return _estimator_extra_state(self.estimator)

    def set_extra_state(self, state) -> None:
        _validate_estimator_extra_state(self.estimator, state)

    def extra_repr(self) -> str:
        options = [
            f"{self.in_channels}",
            f"{self.out_channels}",
            f"kernel_size={self.kernel_size}",
            f"stride={self.stride}",
            f"padding={self.padding}",
        ]
        if self.dilation != (1,) * len(self.kernel_size):
            options.append(f"dilation={self.dilation}")
        if self.groups != 1:
            options.append(f"groups={self.groups}")
        if self.bias is None:
            options.append("bias=False")
        if self.padding_mode != "zeros":
            options.append(f"padding_mode={self.padding_mode!r}")
        options.append(f"estimator={self.estimator.algorithm_id!r}")
        return ", ".join(options)


class TernaryConv1d(_TernaryConvNd):
    """One-dimensional convolution with hard ternary kernel weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        estimator: Optional[Estimator] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        source = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device=device,
            dtype=dtype,
        )
        self._adopt(source, estimator)

    @classmethod
    def from_float(
        cls, module: nn.Conv1d, *, estimator: Optional[Estimator] = None
    ) -> "TernaryConv1d":
        if not isinstance(module, nn.Conv1d):
            raise TypeError("TernaryConv1d.from_float requires torch.nn.Conv1d")
        converted = cls.__new__(cls)
        nn.Module.__init__(converted)
        converted._adopt(module, estimator)
        return converted

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input, padding = self._padded_input(input)
        return F.conv1d(
            input,
            self._projected_weight(),
            self.bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )


class TernaryConv2d(_TernaryConvNd):
    """Two-dimensional convolution with hard ternary kernel weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        estimator: Optional[Estimator] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        source = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device=device,
            dtype=dtype,
        )
        self._adopt(source, estimator)

    @classmethod
    def from_float(
        cls, module: nn.Conv2d, *, estimator: Optional[Estimator] = None
    ) -> "TernaryConv2d":
        if not isinstance(module, nn.Conv2d):
            raise TypeError("TernaryConv2d.from_float requires torch.nn.Conv2d")
        converted = cls.__new__(cls)
        nn.Module.__init__(converted)
        converted._adopt(module, estimator)
        return converted

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input, padding = self._padded_input(input)
        return F.conv2d(
            input,
            self._projected_weight(),
            self.bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )
