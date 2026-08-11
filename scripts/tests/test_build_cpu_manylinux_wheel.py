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
        self.assertEqual(contract["rust"], "1.89.0")
        self.assertEqual(contract["maturin"], "1.10.2")
        self.assertRegex(contract["gxx"], r"^gcc-c\+\+-[0-9].*\.x86_64$")
        self.assertEqual(contract["platform"], "manylinux_2_28_x86_64")


if __name__ == "__main__":
    unittest.main()
