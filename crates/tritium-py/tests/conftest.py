"""Bind source-tree pytest runs to this checkout's Python package.

The repository also builds and installs ``tritium-torch`` wheels.  A global
wheel on ``sys.path`` must never make source tests pass against stale Python
modules or a different native extension.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys

import pytest


SOURCE_ROOT = (Path(__file__).resolve().parents[1] / "python").resolve()
PACKAGE_ROOT = (SOURCE_ROOT / "tritium").resolve()
INSTALLED_WHEEL_ONLY = frozenset({
    "test_hf_lifecycle_receipt.py",
    "test_tutorial_qat.py",
})


def _bind_source_package() -> None:
    source = str(SOURCE_ROOT)
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != SOURCE_ROOT]
    sys.path.insert(0, source)


def pytest_sessionstart(session) -> None:  # type: ignore[no-untyped-def]
    del session
    _bind_source_package()
    import tritium

    origin = Path(tritium.__file__).resolve().parent
    try:
        origin.relative_to(PACKAGE_ROOT)
    except ValueError as error:
        raise RuntimeError(
            "source pytest imported a non-checkout tritium package: "
            f"{origin}; expected below {PACKAGE_ROOT}"
        ) from error


def _installed_distribution_owns_source() -> bool:
    """Return whether pytest is running against an installed wheel.

    Lifecycle and tutorial receipt tests intentionally prove wheel ownership.
    Source-tree pytest must not silently run them against a global/stale wheel,
    but installed-wheel lanes still collect and execute them normally.
    """

    try:
        distribution = importlib.metadata.distribution("tritium-torch")
        import tritium
    except (importlib.metadata.PackageNotFoundError, ImportError):
        return False
    if distribution.files is None:
        return False
    module = Path(tritium.__file__).resolve(strict=True)
    owned = {distribution.locate_file(item).resolve() for item in distribution.files}
    return module in owned


def pytest_collection_modifyitems(config, items) -> None:  # type: ignore[no-untyped-def]
    del config
    if _installed_distribution_owns_source():
        return
    skip = pytest.mark.skip(
        reason="requires installed tritium-torch wheel; run wheel qualification lane"
    )
    for item in items:
        if Path(str(item.fspath)).name in INSTALLED_WHEEL_ONLY:
            item.add_marker(skip)
