"""First-class ternary :mod:`torch.nn` modules."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .torch.errors import TritiumError
from .torch.estimators import AbsMeanSTE, Estimator, SaltSTE
from .torch.ops import ternary_linear
from .torch.projection import ProjectionContext, validate_projection


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
        converted.train(module.training)
        return converted

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if type(self.estimator) in {AbsMeanSTE, SaltSTE}:
            return ternary_linear(input, self.weight, self.bias)
        projection = self.estimator.project(
            self.weight,
            context=ProjectionContext(training=self.training, role="weight"),
        )
        validate_projection(
            projection,
            self.weight,
            algorithm_id=self.estimator.algorithm_id,
            schema_version=self.estimator.schema_version,
        )
        return F.linear(input, projection.dense, self.bias)

    def get_extra_state(self):
        return {
            "schema_version": 1,
            "algorithm_id": self.estimator.algorithm_id,
            "estimator_schema_version": self.estimator.schema_version,
        }

    def set_extra_state(self, state) -> None:
        expected = self.get_extra_state()
        if state != expected:
            raise TritiumError(
                "state estimator identity does not match module estimator",
                code="state_identity",
                stage="load",
                details={"expected": expected, "observed": state},
            )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, estimator={self.estimator.algorithm_id!r}"
        )
