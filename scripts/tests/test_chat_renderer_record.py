"""Fixed guest provenance must not embed workstation/toolchain install paths."""

import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "chat_renderer_record", Path(__file__).resolve().parents[1] / "generate-chat-renderer-record.py"
)
record = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record)


class HostPathTests(unittest.TestCase):
    def test_inlined_sysroot_and_workspace_paths_refuse(self):
        roots = ["/home/person", "/opt/toolchains/rust", "/work/checkout"]
        for root in roots:
            with self.subTest(root=root), self.assertRaisesRegex(RuntimeError, "unmapped host path"):
                record.validate_host_paths(b"wasm\0" + root.encode() + b"/lib/alloc/src/string.rs", roots)

    def test_canonical_mapped_paths_are_not_host_paths(self):
        record.validate_host_paths(b"/rust-toolchain/lib/alloc/src/string.rs /cargo /tritium",
                                   ["/home/person", "/opt/toolchains/rust", "/work/checkout"])


if __name__ == "__main__":
    unittest.main()
