"""CLI test sources must compile from their own published crate."""
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "crates/tritium-cli"


class CliFixturePackagingTests(unittest.TestCase):
    def test_package_snapshot_matches_authored_owner(self):
        subprocess.run([sys.executable, str(ROOT / "scripts/sync-cli-test-fixture.py"), "--check"], check=True, timeout=15)

    def test_integration_module_paths_are_package_contained(self):
        for source in (CRATE / "tests").glob("*.rs"):
            for relative in re.findall(r'#\[path = "([^"]+)"\]', source.read_text()):
                with self.subTest(source=source.name, relative=relative):
                    resolved = (source.parent / relative).resolve()
                    self.assertTrue(resolved.is_relative_to(CRATE))
                    self.assertTrue(resolved.is_file())


if __name__ == "__main__":
    unittest.main()
