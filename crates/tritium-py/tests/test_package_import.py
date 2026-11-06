"""Fresh-process import contract for the public Python package."""

import os
from pathlib import Path
import subprocess
import sys


def test_root_import_exposes_torch_and_nn_facades() -> None:
    source_root = Path(__file__).parents[1] / "python"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), environment.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import tritium; "
                "import tritium.torch.artifacts as artifacts; "
                "import tritium.nn; "
                "assert tritium.torch.__name__ == 'tritium.torch'; "
                "assert artifacts.__name__ == 'tritium.torch.artifacts'"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_root_import_does_not_hide_torch_facade_import_errors() -> None:
    source_root = Path(__file__).parents[1] / "python"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), environment.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.abc, importlib.util, sys; "
                "assert importlib.util.find_spec('torch') is not None; "
                "class_source = \"class BrokenFacade(importlib.abc.MetaPathFinder):\\n"
                "    def find_spec(self, fullname, path=None, target=None):\\n"
                "        if fullname == 'tritium.nn': raise ImportError('facade import failed')\\n"
                "\"; "
                "exec(class_source); sys.meta_path.insert(0, BrokenFacade()); "
                "import tritium"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "facade import failed" in result.stderr
