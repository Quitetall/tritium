import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "generate-compatibility.py"
SPEC = importlib.util.spec_from_file_location("generate_compatibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def matrix(row):
    return {
        "schema": MODULE.SCHEMA,
        "release": "1.1.0-rc.0",
        "dimensions": {name: [dict(row, id=f"{name}-row")] for name in MODULE.REQUIRED_DIMENSIONS},
    }


def matrix_with_row(dimension, row):
    document = matrix(
        {"target": "not yet qualified", "status": "pending", "blocker": "receipt missing"}
    )
    document["dimensions"][dimension] = [dict(row, id=f"{dimension}-row")]
    return document


class CompatibilityMatrixTests(unittest.TestCase):
    def test_pending_matrix_renders_without_support_claim(self):
        document = matrix({"target": "not yet qualified", "status": "pending", "blocker": "receipt missing"})
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "matrix.json"
            validated = MODULE.validate_matrix(document, path)
        rendered = MODULE.render_markdown(validated, Path("release/matrix.json"))
        self.assertIn("**pending**", rendered)
        self.assertIn("pending` is not support", rendered)

    def test_qualified_row_requires_matching_contained_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            receipt = root / "receipts" / "linux.json"
            receipt.parent.mkdir()
            receipt.write_text(
                json.dumps(
                    {
                        "schema": MODULE.RECEIPT_SCHEMA,
                        "target_id": "artifact-schema-row",
                        "source_revision": "ab" * 20,
                        "passed": True,
                    }
                ),
                encoding="utf-8",
            )
            digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
            row = {
                "target": "Linux x86_64",
                "status": "qualified",
                "receipt": {
                    "path": "receipts/linux.json",
                    "sha256": digest,
                    "source_revision": "ab" * 20,
                },
            }
            document = matrix_with_row("artifact-schema", row)
            MODULE.validate_matrix(document, root / "matrix.json")
            row["receipt"]["sha256"] = "00" * 32
            with self.assertRaisesRegex(MODULE.MatrixError, "does not match"):
                MODULE.validate_matrix(matrix_with_row("artifact-schema", row), root / "matrix.json")

    def test_qualified_row_rejects_traversal(self):
        row = {
            "target": "Linux x86_64",
            "status": "qualified",
            "receipt": {
                "path": "../receipt.json",
                "sha256": "00" * 32,
                "source_revision": "ab" * 20,
            },
        }
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(MODULE.MatrixError, "contained POSIX path"):
                MODULE.validate_matrix(matrix(row), Path(raw) / "matrix.json")

    def test_status_has_exactly_one_evidence_field(self):
        row = {
            "target": "CUDA",
            "status": "pending",
            "blocker": "receipt missing",
            "diagnostic": "TRITIUM_UNSUPPORTED_CUDA",
        }
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(MODULE.MatrixError, "requires exactly"):
                MODULE.validate_matrix(matrix(row), Path(raw) / "matrix.json")

    def test_unsupported_row_requires_stable_diagnostic(self):
        row = {"target": "CUDA", "status": "unsupported", "diagnostic": "no"}
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(MODULE.MatrixError, "TRITIUM_UNSUPPORTED"):
                MODULE.validate_matrix(matrix(row), Path(raw) / "matrix.json")


if __name__ == "__main__":
    unittest.main()
