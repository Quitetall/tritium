"""Actual builder invocation with stubbed external tools; not wheel evidence."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ManylinuxBuildResourceTests(unittest.TestCase):
    def test_container_receives_default_and_explicit_cargo_job_limits(self):
        for backend in ("cpu", "cuda"):
            for jobs in (None, "1", "4"):
                with self.subTest(backend=backend, jobs=jobs), tempfile.TemporaryDirectory() as raw:
                    root = Path(raw)
                    binary = root / "bin"
                    binary.mkdir()
                    sysroot = root / "rust"
                    (sysroot / "bin").mkdir(parents=True)
                    for name in ("rustc", "cargo"):
                        executable = sysroot / "bin" / name
                        executable.write_text("#!/bin/sh\nexit 0\n")
                        executable.chmod(0o755)
                    tools = {
                        "git": """#!/bin/sh
case "$*" in
  *--show-toplevel*) printf '%s\\n' "$FIXTURE_ROOT" ;;
  *status*) exit 0 ;;
  *'rev-parse HEAD'*) printf '%040d\\n' 1 ;;
  *) exit 2 ;;
esac
""",
                        "rustup": '#!/bin/sh\nprintf "%s\\n" "$FIXTURE_SYSROOT"\n',
                        "nvcc": '#!/bin/sh\nprintf "Cuda compilation tools, release 13.4\\n"\n',
                        "python": "#!/bin/sh\nexit 0\n",
                        "docker": """#!/usr/bin/python3
import json, os, pathlib, sys
pathlib.Path(os.environ["FIXTURE_DOCKER_ARGS"]).write_text(json.dumps(sys.argv[1:]))
""",
                    }
                    for name, source in tools.items():
                        executable = binary / name
                        executable.write_text(source)
                        executable.chmod(0o755)
                    environment = dict(os.environ)
                    environment.pop("CARGO_BUILD_JOBS", None)
                    environment.update({
                        "PATH": f"{binary}:/usr/bin:/bin",
                        "FIXTURE_ROOT": str(root),
                        "FIXTURE_SYSROOT": str(sysroot),
                        "FIXTURE_DOCKER_ARGS": str(root / "docker.json"),
                        f"TRITIUM_{backend.upper()}_MANYLINUX_CACHE": str(root / "cache"),
                    })
                    if jobs is not None:
                        environment["CARGO_BUILD_JOBS"] = jobs
                    result = subprocess.run(
                        ["bash", str(ROOT / f"scripts/build-{backend}-manylinux-wheel.sh"),
                         str(root / "dist")], env=environment,
                        text=True, capture_output=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    arguments = json.loads((root / "docker.json").read_text())
                    container_environment = [
                        arguments[index + 1] for index, value in enumerate(arguments)
                        if value == "--env"
                    ]
                    expected = f"CARGO_BUILD_JOBS={jobs if jobs is not None else '2'}"
                    self.assertIn(expected, container_environment)


if __name__ == "__main__":
    unittest.main()
