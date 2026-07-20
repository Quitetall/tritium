"""Structured errors for Tritium's PyTorch research surface."""

from __future__ import annotations

from typing import Any, Dict, Optional


class TritiumError(RuntimeError):
    """Stable, structured failure at a Tritium public boundary."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        stage: str,
        module: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.module = module
        self.details = dict(details or {})
