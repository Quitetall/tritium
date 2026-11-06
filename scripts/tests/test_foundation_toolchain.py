"""The mandatory bare-metal target belongs to the pinned repository toolchain."""

from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
TARGET = "thumbv7em-none-eabihf"


class FoundationToolchainTests(unittest.TestCase):
    def test_required_foundation_target_is_declared_on_the_repository_pin(self):
        with (ROOT / "rust-toolchain.toml").open("rb") as stream:
            toolchain = tomllib.load(stream)["toolchain"]
        self.assertNotEqual(toolchain["channel"], "stable")
        self.assertIn(TARGET, toolchain["targets"])
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn(
            "cargo check --locked -p tritium-schema -p tritium-core "
            "--no-default-features --lib --target " + TARGET,
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
