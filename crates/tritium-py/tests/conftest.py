"""Bind source-tree pytest runs to this checkout's Python package.

The repository also builds and installs ``tritium-torch`` wheels.  A global
wheel on ``sys.path`` must never make source tests pass against stale Python
modules or a different native extension.
"""

from __future__ import annotations

from pathlib import Path
import sys


SOURCE_ROOT = (Path(__file__).resolve().parents[1] / "python").resolve()
PACKAGE_ROOT = (SOURCE_ROOT / "tritium").resolve()


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
