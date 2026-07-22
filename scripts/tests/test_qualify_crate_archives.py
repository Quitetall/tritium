from __future__ import annotations

import io
import json
from pathlib import Path
import runpy
import tarfile
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-crate-archives.py")
ArchiveError = MODULE["ArchiveError"]
qualify = MODULE["qualify"]
validate_receipt = MODULE["validate_receipt"]


REVISION = "a" * 40
RELEASE = "1.1.0-rc.0"


def crate_archive(path: Path) -> None:
    prefix = f"demo-{RELEASE}"
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in (
            (f"{prefix}/Cargo.toml", b"[package]\nname='demo'\nversion='1.1.0-rc.0'\n"),
            (f"{prefix}/src/lib.rs", b"pub fn demo() {}\n"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def package_metadata() -> list[dict]:
    return [
        {
            "name": "demo",
            "version": RELEASE,
            "targets": [{"kind": ["lib"]}],
        }
    ]


class QualifyCrateArchivesTests(unittest.TestCase):
    def _qualify(self, root: Path) -> tuple[Path, dict]:
        (root / "Cargo.lock").write_text("# locked\n", encoding="utf-8")
        archives = root / "archives"
        archives.mkdir()
        archive = archives / f"demo-{RELEASE}.crate"
        crate_archive(archive)
        identity = {"bytes": archive.stat().st_size, "sha256": MODULE["_sha256"](archive)}
        with mock.patch.dict(
            qualify.__globals__,
            {
                "_metadata": lambda _root: package_metadata(),
                "inspect_archive": lambda *_args: identity,
            },
        ), mock.patch.object(MODULE["subprocess"], "run"), mock.patch.object(
            MODULE["subprocess"], "check_output",
            side_effect=[
                '[source.crates-io]\nreplace-with = "vendored-sources"\n',
                "cargo 1.89.0", "rustc 1.89.0",
            ],
        ):
            receipt = qualify(root, archives, REVISION, RELEASE, "run-1")
        return archives, receipt

    def test_exact_archive_set_runs_frozen_offline_consumer(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archives, receipt = self._qualify(root)
            path = root / "receipt.json"
            MODULE["_atomic_write"](path, receipt)
            self.assertEqual(
                validate_receipt(
                    path, archives, root / "Cargo.lock", REVISION, RELEASE
                ),
                receipt,
            )
            self.assertTrue(receipt["offline"])
            self.assertEqual(receipt["compiled_packages"], ["demo"])
            self.assertEqual(receipt["packages"][0]["artifact_id"], "crate-demo")

    def test_receipt_rejects_archive_and_identity_drift(self):
        for mutation in ("archive", "identity", "command"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                archives, receipt = self._qualify(root)
                path = root / "receipt.json"
                if mutation == "archive":
                    (archives / f"demo-{RELEASE}.crate").write_bytes(b"changed")
                elif mutation == "identity":
                    receipt["receipt_id"] = "sha256:" + "0" * 64
                else:
                    receipt["commands"] = [["cargo", "check"]]
                path.write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaises(ArchiveError):
                    validate_receipt(
                        path, archives, root / "Cargo.lock", REVISION, RELEASE
                    )

    def test_inventory_rejects_missing_and_stale_archives(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archives = root / "archives"
            archives.mkdir()
            (archives / "stale-1.0.0.crate").write_bytes(b"stale")
            with mock.patch.dict(
                qualify.__globals__, {"_metadata": lambda _root: package_metadata()}
            ), self.assertRaisesRegex(ArchiveError, "inventory mismatch"):
                qualify(root, archives, REVISION, RELEASE, "run-1")


if __name__ == "__main__":
    unittest.main()
