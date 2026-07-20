"""Versioned user configuration for PTQ and QAT workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class TernaryConfig:
    """Serializable recipe selection for a Tritium conversion."""

    mode: str
    estimator: str
    target_modules: Tuple[str, ...]
    planes: int
    profile: Optional[str]
    target_bpw: Optional[float]
    refinement: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"qat", "ptq"}:
            raise ValueError("mode must be 'qat' or 'ptq'")
        if self.schema_version != 1:
            raise ValueError("unsupported TernaryConfig schema_version")
        if not 1 <= self.planes <= 3:
            raise ValueError("planes must be between 1 and 3")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        if any(not name for name in self.target_modules):
            raise ValueError("target_modules cannot contain an empty name")
        if self.mode == "ptq" and self.profile not in {"compact-v1", "near-lossless-v1"}:
            raise ValueError("PTQ profile must be 'compact-v1' or 'near-lossless-v1'")
        if self.mode == "qat" and self.profile is not None:
            raise ValueError("QAT configuration does not accept a deployment profile")
        if self.mode == "qat" and self.target_bpw is not None:
            raise ValueError("QAT configuration does not accept target_bpw")
        if self.target_bpw is not None:
            if not math.isfinite(self.target_bpw) or self.target_bpw <= 0:
                raise ValueError("target_bpw must be finite and positive")
            cap = 2.25 if self.profile == "compact-v1" else 4.0
            if self.target_bpw > cap:
                raise ValueError(f"target_bpw exceeds {self.profile} physical ceiling")
        if self.refinement not in {"none", "scale-only", "hard-pv"}:
            raise ValueError("unknown refinement mode")

    @classmethod
    def qat(
        cls,
        *,
        estimator: str = "salt-ste",
        target_modules: Tuple[str, ...] = ("Linear",),
        planes: int = 1,
    ) -> "TernaryConfig":
        return cls(
            mode="qat",
            estimator=estimator,
            target_modules=tuple(target_modules),
            planes=planes,
            profile=None,
            target_bpw=None,
            refinement="none",
        )

    @classmethod
    def ptq(
        cls,
        *,
        profile: str,
        target_modules: Tuple[str, ...] = ("Linear", "Embedding", "Conv1d"),
        target_bpw: Optional[float] = None,
        refinement: str = "none",
    ) -> "TernaryConfig":
        return cls(
            mode="ptq",
            estimator="salt-v2",
            target_modules=tuple(target_modules),
            planes=3,
            profile=profile,
            target_bpw=target_bpw,
            refinement=refinement,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "estimator": self.estimator,
            "target_modules": list(self.target_modules),
            "planes": self.planes,
            "profile": self.profile,
            "target_bpw": self.target_bpw,
            "refinement": self.refinement,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "TernaryConfig":
        expected = {
            "schema_version",
            "mode",
            "estimator",
            "target_modules",
            "planes",
            "profile",
            "target_bpw",
            "refinement",
        }
        if set(value) != expected:
            raise ValueError("TernaryConfig fields do not match schema version 1")
        return cls(
            schema_version=int(value["schema_version"]),
            mode=str(value["mode"]),
            estimator=str(value["estimator"]),
            target_modules=tuple(str(name) for name in value["target_modules"]),
            planes=int(value["planes"]),
            profile=None if value["profile"] is None else str(value["profile"]),
            target_bpw=(
                None if value["target_bpw"] is None else float(value["target_bpw"])
            ),
            refinement=str(value["refinement"]),
        )
