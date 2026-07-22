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
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.mode not in {"qat", "ptq"}:
            raise ValueError("mode must be 'qat' or 'ptq'")
        if self.schema_version != 2:
            raise ValueError("unsupported TernaryConfig schema_version")
        if not 1 <= self.planes <= 3:
            raise ValueError("planes must be between 1 and 3")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        if not self.estimator:
            raise ValueError("estimator must not be empty")
        if any(not name for name in self.target_modules):
            raise ValueError("target_modules cannot contain an empty name")
        if self.mode == "ptq" and self.profile not in {
            "compact-v1",
            "near-lossless-v1",
        }:
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
        )

    @classmethod
    def ptq(
        cls,
        *,
        profile: str,
        target_modules: Tuple[str, ...] = ("Linear", "Embedding", "Conv1d"),
        target_bpw: Optional[float] = None,
    ) -> "TernaryConfig":
        return cls(
            mode="ptq",
            estimator="salt-v2",
            target_modules=tuple(target_modules),
            planes=3,
            profile=profile,
            target_bpw=target_bpw,
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
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "TernaryConfig":
        version = int(value.get("schema_version", -1))
        expected = {
            "schema_version",
            "mode",
            "estimator",
            "target_modules",
            "planes",
            "profile",
            "target_bpw",
        }
        if version == 1:
            legacy_expected = expected | {"refinement"}
            if set(value) != legacy_expected:
                raise ValueError("TernaryConfig fields do not match schema version 1")
            if value["refinement"] != "none":
                raise ValueError(
                    "legacy PTQ/QAT refinement must be represented by RefinementConfig"
                )
            value = {key: item for key, item in value.items() if key != "refinement"}
            value["schema_version"] = 2
        if set(value) != expected:
            raise ValueError("TernaryConfig fields do not match schema version 2")
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
        )


@dataclass(frozen=True)
class RefinementConfig:
    """A separately versioned recipe for refining an existing PTQ result."""

    kind: str
    structure: str
    max_steps: int = 100
    learning_rate: float = 1e-3
    temperature: float = 1.0
    pv_iterations: int = 0
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unsupported RefinementConfig schema_version")
        if self.kind not in {"scale-only", "hard-pv"}:
            raise ValueError("refinement kind must be 'scale-only' or 'hard-pv'")
        if self.structure not in {"dense", "s34"}:
            raise ValueError("refinement structure must be 'dense' or 's34'")
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("refinement max_steps must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("refinement learning_rate must be finite and positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("refinement temperature must be finite and positive")
        if type(self.pv_iterations) is not int or self.pv_iterations < 0:
            raise ValueError("refinement pv_iterations must be a nonnegative integer")
        if self.kind == "scale-only" and self.structure != "dense":
            raise ValueError("scale-only refinement has fixed dense structure")
        if self.kind == "scale-only" and self.pv_iterations != 0:
            raise ValueError("scale-only refinement does not accept pv_iterations")
        if self.kind == "hard-pv" and self.pv_iterations == 0:
            raise ValueError("hard-pv refinement requires pv_iterations")

    @classmethod
    def scale_only(
        cls,
        *,
        max_steps: int = 100,
        learning_rate: float = 1e-3,
        temperature: float = 1.0,
    ) -> "RefinementConfig":
        return cls(
            kind="scale-only",
            structure="dense",
            max_steps=max_steps,
            learning_rate=learning_rate,
            temperature=temperature,
            pv_iterations=0,
        )

    @classmethod
    def hard_pv(
        cls,
        *,
        structure: str = "dense",
        max_steps: int = 100,
        learning_rate: float = 1e-3,
        temperature: float = 1.0,
        pv_iterations: int = 4,
    ) -> "RefinementConfig":
        return cls(
            kind="hard-pv",
            structure=structure,
            max_steps=max_steps,
            learning_rate=learning_rate,
            temperature=temperature,
            pv_iterations=pv_iterations,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "structure": self.structure,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "pv_iterations": self.pv_iterations,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "RefinementConfig":
        if set(value) == {"schema_version", "kind", "structure"}:
            if value["schema_version"] != 1:
                raise ValueError("legacy RefinementConfig must use schema version 1")
            kind = str(value["kind"])
            structure = str(value["structure"])
            if kind == "scale-only" and structure == "dense":
                return cls.scale_only()
            if kind == "hard-pv":
                raise ValueError(
                    "legacy hard-pv RefinementConfig has no explicit iteration schedule; "
                    "create a schema-version-2 recipe"
                )
            raise ValueError("legacy RefinementConfig kind or structure is invalid")
        expected = {
            "schema_version",
            "kind",
            "structure",
            "max_steps",
            "learning_rate",
            "temperature",
            "pv_iterations",
        }
        if set(value) != expected:
            raise ValueError("RefinementConfig fields do not match schema version 2")
        return cls(
            schema_version=int(value["schema_version"]),
            kind=str(value["kind"]),
            structure=str(value["structure"]),
            max_steps=int(value["max_steps"]),
            learning_rate=float(value["learning_rate"]),
            temperature=float(value["temperature"]),
            pv_iterations=int(value["pv_iterations"]),
        )
