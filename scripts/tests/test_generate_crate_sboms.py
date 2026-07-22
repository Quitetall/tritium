import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import runpy


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "generate-crate-sboms.py")
CrateSbomError = MODULE["CrateSbomError"]
inspect_archive = MODULE["inspect_archive"]
bind_sbom = MODULE["bind_sbom"]


def crate_archive(
    path: Path,
    *,
    revision: str,
    dirty: bool = False,
    link: bool = False,
    inherited_version: bool = False,
):
    prefix = "demo-1.2.3"
    vcs = {"git": {"sha1": revision}, "path_in_vcs": "crates/demo"}
    if dirty:
        vcs["dirty"] = True
    files = {
        "Cargo.toml.orig": (
            b'[package]\nname = "demo"\nversion.workspace = true\n'
            if inherited_version
            else b'[package]\nname = "demo"\nversion = "1.2.3"\n'
        ),
        "Cargo.toml": b'[package]\nname = "demo"\nversion = "1.2.3"\n',
        ".cargo_vcs_info.json": json.dumps(vcs).encode(),
        "src/lib.rs": b"pub fn demo() {}\n",
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if link:
            info = tarfile.TarInfo(f"{prefix}/escape")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../escape"
            archive.addfile(info)


class GenerateCrateSbomsTests(unittest.TestCase):
    def test_archive_binds_clean_vcs_and_rejects_links_or_dirty_state(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            revision = "a" * 40
            valid = root / "demo-1.2.3.crate"
            crate_archive(valid, revision=revision)
            identity = inspect_archive(valid, "demo", "1.2.3", revision)
            self.assertGreater(identity["bytes"], 0)
            dirty = root / "dirty" / "demo-1.2.3.crate"
            dirty.parent.mkdir()
            crate_archive(dirty, revision=revision, dirty=True)
            with self.assertRaisesRegex(CrateSbomError, "dirty"):
                inspect_archive(dirty, "demo", "1.2.3", revision)
            linked = root / "linked" / "demo-1.2.3.crate"
            linked.parent.mkdir()
            crate_archive(linked, revision=revision, link=True)
            with self.assertRaisesRegex(CrateSbomError, "not a regular file"):
                inspect_archive(linked, "demo", "1.2.3", revision)

            inherited = root / "inherited" / "demo-1.2.3.crate"
            inherited.parent.mkdir()
            crate_archive(inherited, revision=revision, inherited_version=True)
            self.assertGreater(
                inspect_archive(inherited, "demo", "1.2.3", revision)["bytes"], 0
            )

    def test_sbom_rewrites_all_local_paths_and_binds_archive(self):
        old = "path+file:///home/runner/Tritium/crates/demo#1.2.3"
        target = f"{old} bin-target-0"
        value = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "metadata": {
                "component": {
                    "type": "library",
                    "bom-ref": old,
                    "name": "demo",
                    "version": "1.2.3",
                    "purl": "pkg:cargo/demo@1.2.3?download_url=file://.",
                    "components": [
                        {
                            "type": "library",
                            "bom-ref": target,
                            "name": "demo_lib",
                            "version": "1.2.3",
                            "purl": "pkg:cargo/demo@1.2.3?download_url=file://.#src/lib.rs",
                        }
                    ],
                }
            },
            "components": [],
            "dependencies": [{"ref": old, "dependsOn": [target]}],
        }
        bound = bind_sbom(
            value,
            artifact_id="crate-demo",
            name="demo",
            version="1.2.3",
            archive=Path("demo-1.2.3.crate"),
            identity={"bytes": 7, "sha256": "b" * 64},
            revision="a" * 40,
        )
        encoded = json.dumps(bound)
        self.assertNotIn("file://", encoded)
        self.assertNotIn("/home/runner", encoded)
        self.assertEqual(bound["dependencies"][0]["ref"], "crate-demo")
        self.assertTrue(bound["dependencies"][0]["dependsOn"][0].startswith("cargo-local:"))


if __name__ == "__main__":
    unittest.main()
