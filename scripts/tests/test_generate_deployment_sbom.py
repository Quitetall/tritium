import gzip
import hashlib
import io
import json
import os
import runpy
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.tests.helm_fixtures import RELEASE, chart_source


ROOT = Path(__file__).resolve().parents[2]
PACKAGER = runpy.run_path(ROOT / "scripts" / "package-helm-chart.py")
SBOM = runpy.run_path(ROOT / "scripts" / "generate-deployment-sbom.py")
ChartPackageError = PACKAGER["ChartPackageError"]
DeploymentSbomError = SBOM["DeploymentSbomError"]
package_chart = PACKAGER["package"]
generate = SBOM["generate"]
write_sbom = SBOM["write_sbom"]


REVISION = "a" * 40


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


def compressed_tar(entries: list[tuple[tarfile.TarInfo, bytes]]) -> bytes:
    unpacked = io.BytesIO()
    with tarfile.open(fileobj=unpacked, mode="w", format=tarfile.USTAR_FORMAT) as output:
        for info, payload in entries:
            output.addfile(info, io.BytesIO(payload))
    unpacked.seek(0)
    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=compressed, compresslevel=9, mtime=0
    ) as output:
        output.write(unpacked.read())
    return compressed.getvalue()


def tar_entry(name: str, payload: bytes, *, kind: bytes = tarfile.REGTYPE) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.size = len(payload)
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.mtime = 0
    info.uname = ""
    info.gname = ""
    return info


class DeploymentSbomTests(unittest.TestCase):
    def test_packager_and_generator_are_deterministic_complete_and_exact(self):
        archives = []
        documents = []
        for ordinal in range(2):
            with tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                source = chart_source(root)
                for index, path in enumerate(sorted(source.rglob("*"))):
                    os.utime(path, (1000 + ordinal + index, 1000 + ordinal + index))
                artifact = root / f"tritium-{RELEASE}.tgz"
                package_chart(source, artifact, RELEASE)
                document = generate(
                    artifact,
                    "tritium-helm",
                    "helm-chart",
                    RELEASE,
                    REVISION,
                    str(fake_tool(root)),
                )
                archives.append(artifact.read_bytes())
                documents.append(document)
                self.assertEqual(
                    {item["name"] for item in document["components"]},
                    {
                        "tritium/Chart.yaml",
                        "tritium/templates/deployment.yaml",
                        "tritium/values.yaml",
                    },
                )
                self.assertEqual(
                    document["metadata"]["component"]["hashes"],
                    [
                        {
                            "alg": "SHA-256",
                            "content": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        }
                    ],
                )
        self.assertEqual(archives[0], archives[1])
        self.assertEqual(documents[0], documents[1])
        self.assertEqual(archives[0][:10], bytes.fromhex("1f8b08000000000002ff"))

    def test_packager_rejects_version_drift_symlinks_and_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = chart_source(root)
            output = root / f"tritium-{RELEASE}.tgz"
            package_chart(source, output, RELEASE)
            with self.assertRaisesRegex(ChartPackageError, "already exists"):
                package_chart(source, output, RELEASE)
            with self.assertRaisesRegex(ChartPackageError, "outside chart source"):
                package_chart(
                    source, source / f"tritium-{RELEASE}.tgz", RELEASE
                )

            output.unlink()
            (source / "Chart.yaml").write_text(
                (source / "Chart.yaml").read_text().replace(RELEASE, "1.1.0-rc.9", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ChartPackageError, "version"):
                package_chart(source, output, RELEASE)

            (source / "Chart.yaml").unlink()
            (source / "Chart.yaml").symlink_to("values.yaml")
            with self.assertRaisesRegex(ChartPackageError, "symlink"):
                package_chart(source, output, RELEASE)

    def test_generator_rejects_archive_and_chart_contract_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = chart_source(root)
            tool = fake_tool(root)
            artifact = root / f"tritium-{RELEASE}.tgz"
            package_chart(source, artifact, RELEASE)

            with artifact.open("ab") as stream:
                stream.write(b"trailing")
            with self.assertRaisesRegex(DeploymentSbomError, "gzip"):
                generate(
                    artifact,
                    "tritium-helm",
                    "helm-chart",
                    RELEASE,
                    REVISION,
                    str(tool),
                )

            artifact.unlink()
            package_chart(source, artifact, RELEASE)
            payload = bytearray(artifact.read_bytes())
            payload[4] = 1
            artifact.write_bytes(payload)
            with self.assertRaisesRegex(DeploymentSbomError, "canonical gzip"):
                generate(
                    artifact,
                    "tritium-helm",
                    "helm-chart",
                    RELEASE,
                    REVISION,
                    str(tool),
                )

            artifact.unlink()
            package_chart(source, artifact, RELEASE)
            tar_payload = gzip.decompress(artifact.read_bytes())
            alternate = bytearray(gzip.compress(tar_payload, compresslevel=1, mtime=0))
            alternate[:10] = bytes.fromhex("1f8b08000000000002ff")
            artifact.write_bytes(alternate)
            with self.assertRaisesRegex(DeploymentSbomError, "canonical gzip encoding"):
                generate(
                    artifact,
                    "tritium-helm",
                    "helm-chart",
                    RELEASE,
                    REVISION,
                    str(tool),
                )

            chart = (source / "Chart.yaml").read_bytes()
            unsafe = root / f"tritium-{RELEASE}.tgz"
            unsafe.unlink()
            unsafe.write_bytes(
                compressed_tar(
                    [
                        (tar_entry("tritium/../Chart.yaml", chart), chart),
                        (
                            tar_entry("tritium/templates/deployment.yaml", b"template"),
                            b"template",
                        ),
                        (tar_entry("tritium/values.yaml", b"values"), b"values"),
                    ]
                )
            )
            with self.assertRaisesRegex(DeploymentSbomError, "not canonical"):
                generate(
                    unsafe,
                    "tritium-helm",
                    "helm-chart",
                    RELEASE,
                    REVISION,
                    str(tool),
                )

            linked = root / f"tritium-{RELEASE}.tgz"
            linked.unlink()
            link = tar_entry("tritium/Chart.yaml", b"", kind=tarfile.SYMTYPE)
            link.linkname = "../../Chart.yaml"
            linked.write_bytes(compressed_tar([(link, b"")]))
            with self.assertRaisesRegex(DeploymentSbomError, "link or non-file"):
                generate(
                    linked,
                    "tritium-helm",
                    "helm-chart",
                    RELEASE,
                    REVISION,
                    str(tool),
                )

    def test_leaf_symlink_mutation_and_atomic_publish_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = chart_source(root)
            artifact = root / f"tritium-{RELEASE}.tgz"
            package_chart(source, artifact, RELEASE)
            linked = root / "linked.tgz"
            linked.symlink_to(artifact)
            tool = fake_tool(root)
            with self.assertRaisesRegex(DeploymentSbomError, "symlink traversal"):
                generate(
                    linked,
                    "tritium-helm",
                    "helm-chart",
                    RELEASE,
                    REVISION,
                    str(tool),
                )

            original_inventory = generate.__globals__["_inventory"]

            def mutate(payload, digest_tool):
                result = original_inventory(payload, digest_tool)
                with artifact.open("ab") as stream:
                    stream.write(b"drift")
                return result

            with mock.patch.dict(generate.__globals__, {"_inventory": mutate}):
                with self.assertRaisesRegex(DeploymentSbomError, "changed while generating"):
                    generate(
                        artifact,
                        "tritium-helm",
                        "helm-chart",
                        RELEASE,
                        REVISION,
                        str(tool),
                    )
            artifact.write_bytes(artifact.read_bytes()[:-5])

            output = root / "chart.cdx.json"
            with mock.patch.object(os, "link", side_effect=OSError("publish failed")):
                with self.assertRaisesRegex(OSError, "publish failed"):
                    write_sbom({"bomFormat": "CycloneDX"}, output)
            self.assertFalse(output.exists())
            write_sbom({"bomFormat": "CycloneDX"}, output)
            with self.assertRaisesRegex(DeploymentSbomError, "already exists"):
                write_sbom({"bomFormat": "CycloneDX"}, output)
            self.assertEqual(
                {path.name for path in root.iterdir()},
                {
                    "chart",
                    "chart.cdx.json",
                    f"tritium-{RELEASE}.tgz",
                    "linked.tgz",
                    "tritium",
                },
            )


if __name__ == "__main__":
    unittest.main()
