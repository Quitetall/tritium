import os
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class OciContractTests(unittest.TestCase):
    def dockerfile(self, flavor: str) -> str:
        return (ROOT / f"deploy/oci/Dockerfile.{flavor}").read_text(encoding="utf-8")

    def test_root_developer_dockerfile_is_removed(self):
        self.assertFalse((ROOT / "Dockerfile").exists())

    def test_every_from_is_digest_pinned_and_runtime_is_distroless(self):
        for flavor in ("cpu", "cuda"):
            text = self.dockerfile(flavor)
            syntax = text.splitlines()[0]
            self.assertRegex(syntax.rsplit("@", 1)[1], DIGEST)
            refs = re.findall(r"^FROM\s+([^\s]+)", text, flags=re.MULTILINE)
            self.assertGreaterEqual(len(refs), 2)
            for ref in refs:
                self.assertIn("@sha256:", ref)
                self.assertRegex(ref.rsplit("@", 1)[1], DIGEST)
            self.assertIn("gcr.io/distroless/cc-debian12:nonroot@sha256:", refs[-1])

    def test_runtime_contract_has_no_installer_or_mutable_volume(self):
        for flavor in ("cpu", "cuda"):
            runtime = self.dockerfile(flavor).split("\nFROM gcr.io/distroless", 1)[1]
            self.assertNotRegex(runtime, r"(?m)^RUN ")
            self.assertNotRegex(runtime, r"(?m)^VOLUME ")
            self.assertIn("USER 65532:65532", runtime)
            self.assertIn('ENTRYPOINT ["/usr/local/bin/tritium-serve"]', runtime)
            self.assertIn("COPY --chown=65532:65532 LICENSE NOTICE", runtime)
            self.assertIn("COPY --chown=65532:65532 deploy/oci/models /models", runtime)

    def test_build_is_offline_frozen_and_attested(self):
        script = (ROOT / "scripts/build-oci-candidate").read_text(encoding="utf-8")
        for token in (
            "status --porcelain=v1 --untracked-files=all",
            '"$root/scripts/release-status" --candidate',
            "candidate_manifest_sha256",
            "candidate manifest changed during admission",
            "candidate manifest changed during OCI build",
            "source_archive_sha256",
            "git -C \"$root\" archive",
            "cargo vendor --locked --versioned-dirs",
            "--network none",
            "--build-arg CARGO_NET_OFFLINE=true",
            'SOURCE_DATE_EPOCH=$epoch',
            "--attest type=sbom",
            "--attest type=provenance,mode=max",
            "rewrite-timestamp=true",
            'archive_sha256="$(sha256sum',
            '"archive_bytes":%s',
            'scripts/verify-oci-archive.py',
            '--package-candidate "$candidate"',
        ):
            self.assertIn(token, script)
        for flavor in ("cpu", "cuda"):
            self.assertIn("cargo build --frozen --profile dist", self.dockerfile(flavor))

    def test_image_metadata_matches_source_contracts(self):
        cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
        version = re.search(r'(?m)^version = "([^"]+)"', cargo).group(1)
        startup = (ROOT / "crates/tritium-serve/src/startup.rs").read_text(encoding="utf-8")
        loader = (ROOT / "crates/tritium-nn/src/model/qwen35_salt_v2.rs").read_text(
            encoding="utf-8"
        )
        self.assertIn("schema_version: 1", startup)
        self.assertIn("manifest.schema_version != 3", loader)
        for flavor in ("cpu", "cuda"):
            text = self.dockerfile(flavor)
            self.assertIn(f'org.opencontainers.image.version="{version}"', text)
            self.assertIn('io.tritium.artifact.schema="3"', text)
            self.assertIn('io.tritium.startup-receipt.schema="1"', text)

    def test_compose_enforces_runtime_controls(self):
        for flavor in ("cpu", "cuda"):
            text = (ROOT / f"deploy/oci/compose.{flavor}.yaml").read_text(encoding="utf-8")
            for token in (
                "read_only: true",
                "cap_drop: [ALL]",
                "no-new-privileges:true",
                "noexec,nosuid,nodev,size=64m",
                ":/models/bundle:ro",
                "--bundle",
                "--profile",
                "127.0.0.1:",
            ):
                self.assertIn(token, text)

    def test_compose_launcher_rejects_mutable_image_tags(self):
        path = ROOT / "scripts/run-oci-compose"
        script = path.read_text(encoding="utf-8")
        self.assertIn("@sha256:[0-9a-f]{64}", script)
        self.assertIn("TRITIUM_BUNDLE must be an ordinary schema-v3 bundle directory", script)
        self.assertIn("TRITIUM_BUNDLE lacks ordinary required asset", script)
        self.assertIn('export TRITIUM_BUNDLE TRITIUM_PROFILE', script)
        environment = os.environ.copy()
        environment["TRITIUM_IMAGE"] = "tritium:latest"
        result = subprocess.run(
            [path, "cpu", "config"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("exact name@sha256 digest", result.stderr)

    def test_runtime_qualifier_covers_production_and_hardening_gates(self):
        script = (ROOT / "scripts/qualify-oci-runtime.py").read_text(encoding="utf-8")
        for token in (
            "production_artifact_admitted",
            "startup_receipt",
            "manifest_package_id",
            "/v1/chat/completions",
            "data: [DONE]",
            "ReadonlyRootfs",
            "CapDrop",
            "no-new-privileges",
            '"--signal", "TERM"',
            "nvidia-smi",
            "receipt_id",
        ):
            self.assertIn(token, script)


if __name__ == "__main__":
    unittest.main()
