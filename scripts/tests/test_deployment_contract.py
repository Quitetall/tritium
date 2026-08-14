import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "deploy/helm/tritium"
DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


class DeploymentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = (CHART / "values.yaml").read_text(encoding="utf-8")
        cls.schema = json.loads((CHART / "values.schema.json").read_text(encoding="utf-8"))

    def test_defaults_are_explicit_placeholders_and_schema_is_closed(self):
        self.assertEqual(self.schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(self.schema["additionalProperties"])
        self.assertIn("digest: sha256:" + "0" * 64, self.values)
        self.assertIn('expectedManifestSha256: "' + "0" * 64 + '"', self.values)

    def test_cuda_requires_gpu_and_keda_cannot_scale_to_zero(self):
        encoded = json.dumps(self.schema, sort_keys=True)
        self.assertIn('"backend": {"const": "cuda"}', encoded)
        self.assertIn('"enabled": {"const": true}', encoded)
        minimum = self.schema["properties"]["keda"]["properties"]["minReplicaCount"]["minimum"]
        self.assertEqual(minimum, 1)

    def test_chart_binds_digest_artifact_secret_and_security_controls(self):
        deployment = (CHART / "templates/deployment.yaml").read_text(encoding="utf-8")
        for token in (
            'printf "%s@%s" .Values.image.repository .Values.image.digest',
            "sha256sum -c -",
            "bundle contains a symlink",
            "staged bundle contains a symlink",
            "--bundle",
            ".Values.artifact.profile",
            "secretKeyRef:",
            "automountServiceAccountToken: false",
            "readOnlyRootFilesystem",
            "RuntimeDefault",
            "terminationGracePeriodSeconds",
            "maxUnavailable: 0",
            "tritium.ai/image-digest",
            "authenticated-probe",
            "shareProcessNamespace: true",
            "pidof tritium-serve",
            "until wget",
            "kill -KILL",
            "type: Recreate",
            "ephemeral-storage",
            "GPU resource limits require gpu.enabled=true",
            ".Values.probes.startup",
            "--admin-port",
            "preStop:",
            "path: /drain",
            "host: 127.0.0.1",
        ):
            source = (
                (CHART / "values.yaml").read_text(encoding="utf-8")
                if token in {"RuntimeDefault", "readOnlyRootFilesystem"}
                else deployment
            )
            self.assertIn(token, source)

    def test_optional_surfaces_are_bounded(self):
        keda = (CHART / "templates/keda.yaml").read_text(encoding="utf-8")
        self.assertIn("stabilizationWindowSeconds", keda)
        self.assertIn("metricName: tritium_queue_pressure", keda)
        self.assertIn("maxReplicaCount", keda)
        for template in ("pdb.yaml", "networkpolicy.yaml", "servicemonitor.yaml"):
            self.assertTrue((CHART / "templates" / template).is_file())
        service_monitor = (CHART / "templates/servicemonitor.yaml").read_text(encoding="utf-8")
        self.assertIn("with .Values.serviceMonitor.labels", service_monitor)
        self.assertIn("toYaml . | nindent 4", service_monitor)
        network = (CHART / "templates/networkpolicy.yaml").read_text(encoding="utf-8")
        self.assertIn("kubernetes.io/metadata.name: kube-system", network)
        self.assertIn("k8s-app: kube-dns", network)

    def test_standalone_examples_are_bounded_and_honest(self):
        keda = (ROOT / "deploy/keda/scaledobject.yaml").read_text(encoding="utf-8")
        self.assertIn("minReplicaCount: 1", keda)
        self.assertIn("maxReplicaCount: 4", keda)
        knative = (ROOT / "deploy/knative/service.cpu.yaml").read_text(encoding="utf-8")
        image = re.search(r"image: ([^\s]+@sha256:[0-9a-f]{64})", knative).group(1)
        self.assertRegex(image, DIGEST)
        self.assertIn("containerConcurrency: 1", knative)
        self.assertIn('autoscaling.knative.dev/min-scale: "1"', knative)
        self.assertIn("RuntimeDefault", knative)
        self.assertIn("automountServiceAccountToken: false", knative)

    def test_helm_checker_uses_pinned_offline_image(self):
        checker = (ROOT / "scripts/check-deployment-manifests").read_text(encoding="utf-8")
        image = re.search(r'image="([^"]+)"', checker).group(1)
        self.assertRegex(image, DIGEST)
        self.assertIn("--network none", checker)
        self.assertIn("--pull=never", checker)
        self.assertIn("backend=cuda", checker)
        self.assertIn("scientific notation", checker)

    def test_kubernetes_qualifier_is_fail_closed_and_content_addressed(self):
        qualifier = (ROOT / "scripts/qualify-kubernetes-deployment.py").read_text(
            encoding="utf-8"
        )
        for token in (
            "validate_oci_archive", "--atomic", "helm_history", "pod-restart",
            "failed-upgrade", "physical_device_id", "--cuda-probe-image",
            "tritium_tokens_out_total", "receipt_id", "release-cleanup",
            "qualification_lock_uid", "query_range", "servicemonitor/",
            "keda-hpa-", "load_started_unix",
        ):
            self.assertIn(token, qualifier)


if __name__ == "__main__":
    unittest.main()
