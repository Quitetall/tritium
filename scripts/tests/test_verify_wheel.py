import base64
import csv
import hashlib
import importlib.util
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "verify-wheel.py"
SPEC = importlib.util.spec_from_file_location("verify_wheel", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(payload, algorithm="sha256"):
    value = hashlib.new(algorithm, payload).digest()
    return algorithm + "=" + base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def build_wheel(
    path,
    *,
    corrupt_record=False,
    hash_algorithm="sha256",
    native_name="tritium/_tritium.abi3.so",
    platform_tag="linux_x86_64",
    extra=None,
):
    files = {
        "tritium/__init__.py": b"from . import _tritium\n",
        native_name: b"native-placeholder",
        "tritium_torch-1.1.0rc0.dist-info/METADATA": (
            b"Metadata-Version: 2.3\nName: tritium-torch\nVersion: 1.1.0rc0\n"
        ),
        "tritium_torch-1.1.0rc0.dist-info/WHEEL": (
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp39-abi3-{platform_tag}\n".encode()
        ),
    }
    if extra:
        files.update(extra)
    record_name = "tritium_torch-1.1.0rc0.dist-info/RECORD"
    rows = []
    for name, payload in files.items():
        encoded = digest(payload, hash_algorithm)
        if corrupt_record and name == "tritium/__init__.py":
            encoded = "sha256=" + "A" * 43
        rows.append((name, encoded, str(len(payload))))
    rows.append((record_name, "", ""))
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    files[record_name] = stream.getvalue().encode("utf-8")
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)


class VerifyWheelTests(unittest.TestCase):
    def test_valid_abi3_wheel_passes(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel = Path(raw) / "tritium_torch-1.1.0rc0-cp39-abi3-linux_x86_64.whl"
            build_wheel(wheel)
            result = MODULE.inspect_wheel(wheel, "1.1.0rc0")
            self.assertEqual(result["platform_tag"], "linux_x86_64")
            self.assertEqual(result["sha256"], hashlib.sha256(wheel.read_bytes()).hexdigest())

    def test_record_corruption_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel = Path(raw) / "tritium_torch-1.1.0rc0-cp39-abi3-linux_x86_64.whl"
            build_wheel(wheel, corrupt_record=True)
            with self.assertRaisesRegex(MODULE.WheelError, "RECORD hash mismatch"):
                MODULE.inspect_wheel(wheel, "1.1.0rc0")

    def test_source_residue_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel = Path(raw) / "tritium_torch-1.1.0rc0-cp39-abi3-linux_x86_64.whl"
            build_wheel(wheel, extra={"src/lib.rs": b"fn main() {}"})
            with self.assertRaisesRegex(MODULE.WheelError, "source/build residue"):
                MODULE.inspect_wheel(wheel, "1.1.0rc0")

    def test_weak_record_hash_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel = Path(raw) / "tritium_torch-1.1.0rc0-cp39-abi3-linux_x86_64.whl"
            build_wheel(wheel, hash_algorithm="md5")
            with self.assertRaisesRegex(MODULE.WheelError, "SHA-256 or stronger"):
                MODULE.inspect_wheel(wheel, "1.1.0rc0")

    def test_split_dist_info_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel = Path(raw) / "tritium_torch-1.1.0rc0-cp39-abi3-linux_x86_64.whl"
            build_wheel(wheel, extra={"other-1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n"})
            with self.assertRaisesRegex(MODULE.WheelError, "only canonical dist-info"):
                MODULE.inspect_wheel(wheel, "1.1.0rc0")

    def test_windows_plain_pyd_is_accepted(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel = Path(raw) / "tritium_torch-1.1.0rc0-cp39-abi3-win_amd64.whl"
            build_wheel(
                wheel,
                native_name="tritium/_tritium.pyd",
                platform_tag="win_amd64",
            )
            result = MODULE.inspect_wheel(wheel, "1.1.0rc0")
            self.assertEqual(result["platform_tag"], "win_amd64")

    def test_filename_version_must_match(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel = Path(raw) / "tritium_torch-1.1.0rc1-cp39-abi3-linux_x86_64.whl"
            build_wheel(wheel)
            with self.assertRaisesRegex(MODULE.WheelError, "filename version"):
                MODULE.inspect_wheel(wheel, "1.1.0rc0")

    def test_directory_requires_exactly_one_wheel(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaisesRegex(MODULE.WheelError, "exactly one wheel"):
                MODULE.resolve_wheel(root)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux target binding")
    def test_target_binding_rejects_wrong_wheel_platform(self):
        with self.assertRaisesRegex(MODULE.WheelError, "does not match"):
            MODULE.qualify_target("linux-x86_64-cpu", "win_amd64")

    def test_target_binding_rejects_unknown_receipt_cell(self):
        with self.assertRaisesRegex(MODULE.WheelError, "unsupported compatibility target"):
            MODULE.qualify_target("linux-aarch64-cpu", "manylinux_2_28_aarch64")


if __name__ == "__main__":
    unittest.main()
