from pathlib import Path


RELEASE = "1.1.0-rc.0"


def chart_source(root: Path) -> Path:
    source = root / "chart"
    (source / "templates").mkdir(parents=True)
    (source / "Chart.yaml").write_text(
        "apiVersion: v2\n"
        "name: tritium\n"
        "description: Test Tritium chart\n"
        "type: application\n"
        f"version: {RELEASE}\n"
        f"appVersion: {RELEASE}\n"
        'kubeVersion: ">=1.29.0-0"\n'
        "annotations:\n"
        '  tritium.ai/artifact-schema: "3"\n'
        '  tritium.ai/startup-receipt-schema: "1"\n',
        encoding="utf-8",
    )
    (source / "values.yaml").write_text("replicaCount: 1\n", encoding="utf-8")
    (source / "templates" / "deployment.yaml").write_text(
        "apiVersion: apps/v1\nkind: Deployment\n", encoding="utf-8"
    )
    return source
