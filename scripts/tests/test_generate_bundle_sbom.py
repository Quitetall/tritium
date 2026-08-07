import hashlib
import io
import json
import os
import runpy
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "generate-bundle-sbom.py")
BundleSbomError = MODULE["BundleSbomError"]
MODEL_FILES = MODULE["MODEL_FILES"]
ONNX_FILES = MODULE["ONNX_FILES"]
generate = MODULE["generate"]
write_sbom = MODULE["write_sbom"]
PINNED_MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def fake_blake3(payload: bytes) -> str:
    return hashlib.sha256(b"B3" + payload).hexdigest()


def fake_package_id(payload: bytes) -> str:
    return "trp1_" + hashlib.sha256(b"PKG" + payload).hexdigest()


def fake_tool(root: Path) -> Path:
    tool = root / "tritium"
    tool.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib,json,sys\n"
        "assert sys.argv[1:] == ['release', 'digest-stream']\n"
        "b=sys.stdin.buffer.read()\n"
        "print(json.dumps({'schema':'tritium.stream-identity.v1','bytes':len(b),"
        "'sha256':hashlib.sha256(b).hexdigest(),"
        "'blake3':hashlib.sha256(b'B3'+b).hexdigest(),"
        "'package_id':'trp1_'+hashlib.sha256(b'PKG'+b).hexdigest()},"
        "separators=(',',':')))\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    return tool


def onnx_files() -> dict[str, bytes]:
    language = b"language"
    mtp = b"mtp"
    weights = b"weights"
    manifest = {
        "schema": "tritium-qwen35-onnx-bundle-v2",
        "sequence_mode": "dynamic-cache-v1",
        "language": {"file": "language.onnx", "blake3": fake_blake3(language)},
        "mtp": {"file": "mtp.onnx", "blake3": fake_blake3(mtp)},
        "weights": {
            "file": "weights.bin",
            "blake3": fake_blake3(weights),
            "bytes": len(weights),
        },
        "identity": {
            "source_model_id": "source-model",
            "tokenizer_id": "tokenizer",
            "recipe_id": "recipe",
            "package_id": "package",
            "converted_coverage_id": "converted",
            "deferred_coverage_id": "deferred",
        },
        "conversion": {
            "mode": "ptq",
            "completion_id": "completion",
            "campaign_id": "campaign",
            "admission_id": "admission",
            "selection_id": "selection",
        },
    }
    return {
        "language.onnx": language,
        "mtp.onnx": mtp,
        "weights.bin": weights,
        "tritium-onnx-manifest.json": json.dumps(manifest, sort_keys=True).encode(),
    }


def model_files() -> dict[str, bytes]:
    files = {name: name.encode() for name in MODEL_FILES if name != "tritium.json"}
    manifest = {
        "schema_version": 3,
        "artifact_kind": "qwen3.6-language-mtp-salt-v2-hf-bundle",
        "complete_model": False,
        "packing": "d2",
        "completion_id": "completion",
        "campaign_id": "campaign",
        "admission_id": "admission",
        "selection_id": "selection",
        "source_model_id": "source-model",
        "source_identity_status": "authenticated",
        "official_payload_authenticated": True,
        "source_revision": PINNED_MODEL_REVISION,
        "preserved": {
            "file": "preserved.safetensors",
            "package_id": fake_package_id(files["preserved.safetensors"]),
            "tensors": 1,
            "payload_bytes": 1,
            "serialized_bytes": len(files["preserved.safetensors"]),
        },
        "profiles": {
            "compact-v1": {
                "file": "compact.tsalt2",
                "package_id": fake_package_id(files["compact.tsalt2"]),
                "serialized_bytes": len(files["compact.tsalt2"]),
                "resident_bytes": 1,
            },
            "near-lossless-v1": {
                "file": "near-lossless.tsalt2",
                "package_id": fake_package_id(files["near-lossless.tsalt2"]),
                "serialized_bytes": len(files["near-lossless.tsalt2"]),
                "resident_bytes": 1,
            },
        },
        "hf_assets": [
            {
                "file": name,
                "package_id": fake_package_id(files[name]),
                "bytes": len(files[name]),
            }
            for name in sorted(
                MODEL_FILES
                - {
                    "tritium.json",
                    "preserved.safetensors",
                    "compact.tsalt2",
                    "near-lossless.tsalt2",
                }
            )
        ],
    }
    files["tritium.json"] = json.dumps(manifest, sort_keys=True).encode()
    return files


def archive(path: Path, files: dict[str, bytes], *, link: bool = False) -> None:
    with tarfile.open(path, "w") as output:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            output.addfile(info, io.BytesIO(payload))
        if link:
            info = tarfile.TarInfo("escape")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../escape"
            output.addfile(info)


def rewrite_first_header(path: Path, start: int, end: int, value: bytes) -> None:
    payload = bytearray(path.read_bytes())
    if len(value) != end - start:
        raise ValueError("replacement field has wrong width")
    payload[start:end] = value
    payload[148:156] = b"        "
    payload[148:156] = f"{sum(payload[:512]):06o}".encode() + b"\0 "
    path.write_bytes(payload)


class GenerateBundleSbomTests(unittest.TestCase):
    def test_onnx_and_model_archives_have_exact_deterministic_inventory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tool = fake_tool(root)
            for kind, files in (("onnx-bundle", onnx_files()), ("model-bundle", model_files())):
                path = root / f"{kind}.tar"
                archive(path, files)
                first = generate(path, kind, kind, "a" * 40, str(tool))
                second = generate(path, kind, kind, "a" * 40, str(tool))
                self.assertEqual(first, second)
                component = first["metadata"]["component"]
                self.assertEqual(component["bom-ref"], kind)
                self.assertEqual(
                    component["hashes"],
                    [{"alg": "SHA-256", "content": hashlib.sha256(path.read_bytes()).hexdigest()}],
                )
                self.assertEqual(
                    {item["name"] for item in first["components"]}, set(files)
                )
                self.assertEqual(
                    len(first["dependencies"][0]["dependsOn"]), len(files)
                )

    def test_archive_topology_and_manifest_ledgers_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tool = fake_tool(root)
            linked = root / "linked.tar"
            archive(linked, onnx_files(), link=True)
            with self.assertRaisesRegex(BundleSbomError, "not a regular file"):
                generate(linked, "onnx", "onnx-bundle", "a" * 40, str(tool))

            unknown = root / "unknown.tar"
            archive(unknown, {**onnx_files(), "unknown": b"x"})
            with self.assertRaisesRegex(BundleSbomError, "unknown unknown"):
                generate(unknown, "onnx", "onnx-bundle", "a" * 40, str(tool))

            corrupt = onnx_files()
            manifest = json.loads(corrupt["tritium-onnx-manifest.json"])
            manifest["weights"]["bytes"] += 1
            corrupt["tritium-onnx-manifest.json"] = json.dumps(manifest).encode()
            drift = root / "drift.tar"
            archive(drift, corrupt)
            with self.assertRaisesRegex(BundleSbomError, "byte ledger differs"):
                generate(drift, "onnx", "onnx-bundle", "a" * 40, str(tool))

            traversal = root / "traversal.tar"
            files = onnx_files()
            with tarfile.open(traversal, "w") as output:
                for name, payload in files.items():
                    info = tarfile.TarInfo("../language.onnx" if name == "language.onnx" else name)
                    info.size = len(payload)
                    output.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(BundleSbomError, "flat canonical path"):
                generate(traversal, "onnx", "onnx-bundle", "a" * 40, str(tool))

    def test_internal_manifest_identities_are_authenticated(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tool = fake_tool(root)

            onnx = onnx_files()
            onnx_manifest = json.loads(onnx["tritium-onnx-manifest.json"])
            onnx_manifest["language"]["blake3"] = "0" * 64
            onnx["tritium-onnx-manifest.json"] = json.dumps(onnx_manifest).encode()
            onnx_path = root / "bad-onnx.tar"
            archive(onnx_path, onnx)
            with self.assertRaisesRegex(BundleSbomError, "language BLAKE3 differs"):
                generate(onnx_path, "onnx", "onnx-bundle", "a" * 40, str(tool))

            model = model_files()
            model_manifest = json.loads(model["tritium.json"])
            model_manifest["profiles"]["compact-v1"]["package_id"] = "trp1_" + "0" * 64
            model["tritium.json"] = json.dumps(model_manifest).encode()
            model_path = root / "bad-model.tar"
            archive(model_path, model)
            with self.assertRaisesRegex(BundleSbomError, "package identity differs"):
                generate(
                    model_path, "model", "model-bundle", "a" * 40, str(tool)
                )

            unauthenticated = model_files()
            manifest = json.loads(unauthenticated["tritium.json"])
            manifest["official_payload_authenticated"] = False
            unauthenticated["tritium.json"] = json.dumps(manifest).encode()
            unauthenticated_path = root / "unauthenticated.tar"
            archive(unauthenticated_path, unauthenticated)
            with self.assertRaisesRegex(BundleSbomError, "not authenticated"):
                generate(
                    unauthenticated_path,
                    "model",
                    "model-bundle",
                    "a" * 40,
                    str(tool),
                )

            reordered = model_files()
            manifest = json.loads(reordered["tritium.json"])
            manifest["hf_assets"].reverse()
            reordered["tritium.json"] = json.dumps(manifest).encode()
            reordered_path = root / "reordered.tar"
            archive(reordered_path, reordered)
            with self.assertRaisesRegex(BundleSbomError, "filenames differ"):
                generate(
                    reordered_path,
                    "model",
                    "model-bundle",
                    "a" * 40,
                    str(tool),
                )

    def test_leaf_symlink_and_midstream_mutation_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tool = fake_tool(root)
            target = root / "onnx.tar"
            archive(target, onnx_files())
            linked = root / "linked.tar"
            linked.symlink_to(target)
            with self.assertRaisesRegex(BundleSbomError, "symlink traversal"):
                generate(linked, "onnx", "onnx-bundle", "a" * 40, str(tool))

            original_inventory = generate.__globals__["_inventory"]

            def mutate(path, stream, digest_tool):
                result = original_inventory(path, stream, digest_tool)
                with path.open("ab") as output:
                    output.write(b"drift")
                return result

            with mock.patch.dict(generate.__globals__, {"_inventory": mutate}):
                with self.assertRaisesRegex(BundleSbomError, "changed while generating"):
                    generate(target, "onnx", "onnx-bundle", "a" * 40, str(tool))

    def test_parent_symlink_and_parent_replacement_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tool = fake_tool(root)
            source = root / "source"
            source.mkdir()
            archive(source / "onnx.tar", onnx_files())
            linked = root / "linked"
            linked.symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(BundleSbomError, "symlink traversal"):
                generate(
                    linked / "onnx.tar", "onnx", "onnx-bundle", "a" * 40, str(tool)
                )

            current = root / "current"
            replacement = root / "replacement"
            current.mkdir()
            replacement.mkdir()
            archive(current / "onnx.tar", onnx_files())
            archive(replacement / "onnx.tar", onnx_files())
            original_inventory = generate.__globals__["_inventory"]

            def replace_parent(path, stream, digest_tool):
                result = original_inventory(path, stream, digest_tool)
                current.rename(root / "opened")
                replacement.rename(current)
                return result

            with mock.patch.dict(generate.__globals__, {"_inventory": replace_parent}):
                with self.assertRaisesRegex(BundleSbomError, "parent path changed"):
                    generate(
                        current / "onnx.tar",
                        "onnx",
                        "onnx-bundle",
                        "a" * 40,
                        str(tool),
                    )

    def test_mislabeled_compression_and_trailing_payload_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tool = fake_tool(root)
            mislabeled = root / "mislabeled.tar"
            with tarfile.open(mislabeled, "w:gz") as output:
                for name, payload in onnx_files().items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    output.addfile(info, io.BytesIO(payload))
            with self.assertRaises(BundleSbomError):
                generate(mislabeled, "onnx", "onnx-bundle", "a" * 40, str(tool))

            trailing = root / "trailing.tar"
            archive(trailing, onnx_files())
            with trailing.open("ab") as output:
                output.write(b"not-tar-padding")
            with self.assertRaisesRegex(BundleSbomError, "after its end blocks"):
                generate(trailing, "onnx", "onnx-bundle", "a" * 40, str(tool))

    def test_noncanonical_ustar_metadata_fields_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tool = fake_tool(root)
            cases = (
                ("mode", 100, 108, b"ZZZZZZZ\0", "mode.*canonical octal"),
                ("link", 157, 257, b"target" + b"\0" * 94, "link name"),
                ("device", 329, 337, b"0000001\0", "device fields"),
            )
            for label, start, end, replacement, message in cases:
                with self.subTest(label=label):
                    path = root / f"{label}.tar"
                    archive(path, onnx_files())
                    rewrite_first_header(path, start, end, replacement)
                    with self.assertRaisesRegex(BundleSbomError, message):
                        generate(path, "onnx", "onnx-bundle", "a" * 40, str(tool))

    @unittest.skipUnless(shutil.which("zstd"), "zstd command is required")
    def test_tar_zstd_uses_same_inventory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tool = fake_tool(root)
            plain = root / "onnx.tar"
            with tarfile.open(plain, "w") as output:
                for name, payload in onnx_files().items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    output.addfile(info, io.BytesIO(payload))
            compressed = root / "onnx.tar.zst"
            with compressed.open("wb") as stream:
                subprocess.run(
                    ["zstd", "--quiet", "--stdout", "--", str(plain)],
                    stdout=stream,
                    check=True,
                )
            document = generate(
                compressed, "onnx", "onnx-bundle", "a" * 40, str(tool)
            )
            properties = {
                item["name"]: item["value"]
                for item in document["metadata"]["component"]["properties"]
            }
            self.assertEqual(properties["tritium:bundle:archive-format"], "tar-zstd")

    def test_write_is_atomic_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "bundle.cdx.json"
            document = {"bomFormat": "CycloneDX", "specVersion": "1.6"}
            with mock.patch.object(os, "link", side_effect=OSError("publish failed")):
                with self.assertRaisesRegex(OSError, "publish failed"):
                    write_sbom(document, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(raw).iterdir()), [])
            with mock.patch.object(
                os, "fsync", side_effect=[None, OSError("directory sync failed")]
            ):
                with self.assertRaisesRegex(OSError, "directory sync failed"):
                    write_sbom(document, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(raw).iterdir()), [])
            write_sbom(document, output)
            self.assertEqual(json.loads(output.read_text()), document)
            with self.assertRaisesRegex(BundleSbomError, "already exists"):
                write_sbom(document, output)


if __name__ == "__main__":
    unittest.main()
