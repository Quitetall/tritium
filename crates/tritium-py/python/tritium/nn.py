"""First-class ternary :mod:`torch.nn` modules."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .torch.errors import TritiumError
from .torch.estimators import AbsMeanSTE, Estimator, SaltSTE
from .torch.ops import ternary_linear
from .torch.projection import ProjectionContext, validate_projection


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
