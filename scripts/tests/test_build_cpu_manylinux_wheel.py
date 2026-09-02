import json
import subprocess
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "build-cpu-manylinux-wheel.sh"


class BuildCpuManylinuxWheelTests(unittest.TestCase):
    def test_contract_is_immutable_and_manylinux(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        completed = subprocess.run(
            ["bash", str(SCRIPT), "--print-contract"],
            check=True,
            capture_output=True,
            text=True,
        )
        contract = json.loads(completed.stdout)
        self.assertRegex(contract["image"], r"@sha256:[0-9a-f]{64}$")
        self.assertEqual(contract["rust"], "1.98.0")
        self.assertEqual(contract["maturin"], "1.10.2")
        self.assertEqual(contract["linker"], "/io/scripts/manylinux-static-cxx-linker.sh")
        self.assertRegex(contract["static_cxx"], r"/gcc-toolset-14/root/usr/lib/gcc/.*/14/libstdc\+\+\.a$")
        self.assertEqual(contract["platform"], "manylinux_2_28_x86_64")

    def test_static_linker_contract_is_source_bound(self):
        linker = SCRIPT.parent / "manylinux-static-cxx-linker.sh"
        subprocess.run(["bash", "-n", str(linker)], check=True)
        source = linker.read_text(encoding="utf-8")
        self.assertIn("libstdc++.a", source)
        self.assertIn("libstdc++fs.a", source)
        self.assertIn("--start-group", source)


if __name__ == "__main__":
    unittest.main()
