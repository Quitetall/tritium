"""Exact parameter-disposition receipts for model conversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Tuple


@dataclass(frozen=True)
class CoverageEntry:
    """One unique parameter and every qualified alias that owns it."""

    path: str
    aliases: Tuple[str, ...]
    disposition: str
    reason: str
    numel: int
    logical_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["aliases"] = list(self.aliases)
        return value

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "CoverageEntry":
        return cls(
            path=str(value["path"]),
            aliases=tuple(str(alias) for alias in value["aliases"]),
            disposition=str(value["disposition"]),
            reason=str(value["reason"]),
            numel=int(value["numel"]),
            logical_bytes=int(value["logical_bytes"]),
        )


@dataclass(frozen=True)
class CoverageReport:
    """Immutable, JSON-compatible accounting for every unique parameter."""

    entries: Tuple[CoverageEntry, ...]
    schema_version: int = 1

    @classmethod
    def new(cls, entries: Iterable[CoverageEntry]) -> "CoverageReport":
        return cls(tuple(entries))

    @property
    def total_parameters(self) -> int:
        return len(self.entries)

    @property
    def converted_parameters(self) -> int:
        return sum(entry.disposition == "converted" for entry in self.entries)

    @property
    def preserved_parameters(self) -> int:
        return sum(entry.disposition == "preserved" for entry in self.entries)

    @property
    def total_numel(self) -> int:
        return sum(entry.numel for entry in self.entries)

    @property
    def logical_bytes(self) -> int:
        return sum(entry.logical_bytes for entry in self.entries)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "CoverageReport":
        if int(value["schema_version"]) != 1:
            raise ValueError("unsupported CoverageReport schema_version")
        return cls.new(CoverageEntry.from_dict(entry) for entry in value["entries"])
