from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "verify-qwen36-gdn-sensitivity.py"
SPEC = importlib.util.spec_from_file_location("qwen36_gdn_sensitivity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def receipt(delta_terminal: float = 1.0, control_terminal: float = 1.0) -> dict:
    probes = []
    for family, terminal in (
        ("deltanet", delta_terminal),
        ("full_attention", control_terminal),
    ):
        for index, tensor_class in enumerate(MODULE.TENSOR_CLASSES):
            probes.append(
                {
                    "family": family,
                    "tensor_class": tensor_class,
                    "tensor_name": f"{family}.{tensor_class}",
                    "tensor_index": len(probes),
                    "source_weight_sha256": f"{index + 1:064x}",
                    "ternary_artifact_sha256": f"{index + 9:064x}",
                    "bpw": 1.58,
                    "weight_mse": 0.25,
                    "sequence_positions": [128, 512, 2048],
                    "output_divergence": [terminal / 4, terminal / 2, terminal],
                }
            )
    result = {
        "schema": MODULE.SCHEMA,
        "repository": MODULE.REPOSITORY,
        "revision": MODULE.REVISION,
        "source_model_id": "a" * 64,
        "calibration_pack_receipt_id": "sha256:" + "b" * 64,
        "calibration_token_digest": "c" * 64,
        "recipe_id": "d" * 64,
        "matched_bpw": 1.58,
        "runtime_adapter_sha256": "e" * 64,
        "machine": {
            "device": "cuda",
            "device_name": "fixture",
            "runtime_version": "fixture",
            "tritium_revision": MODULE.REVISION,
        },
        "probes": probes,
    }
    result["receipt_id"] = "sha256:" + hashlib.sha256(MODULE.canonical(result)).hexdigest()
    return result


def reseal(value: dict) -> None:
    value["receipt_id"] = "sha256:" + hashlib.sha256(
        MODULE.canonical({key: item for key, item in value.items() if key != "receipt_id"})
    ).hexdigest()


class QwenGdnSensitivityReceiptTests(unittest.TestCase):
    def test_receipt_derives_pass_when_delta_terminal_is_within_threshold(self):
        verified = MODULE.verify(receipt(delta_terminal=2.0, control_terminal=1.0))
        self.assertEqual(verified["gate"], "pass")
        self.assertEqual(verified["route_to_refined_track"], [])
        self.assertEqual(verified["evidence_scope"], "receipt-structure-and-rule-only")

    def test_receipt_routes_delta_family_when_terminal_exceeds_two_times_median(self):
        verified = MODULE.verify(receipt(delta_terminal=2.01, control_terminal=1.0))
        self.assertEqual(verified["gate"], "fail")
        self.assertEqual(verified["route_to_refined_track"], ["down", "gate_up", "output", "qkv"])

    def test_receipt_routes_only_the_failing_delta_tensor_class(self):
        value = receipt()
        value["probes"][0]["output_divergence"][-1] = 2.01
        reseal(value)
        verified = MODULE.verify(value)
        self.assertEqual(verified["route_to_refined_track"], ["qkv"])

    def test_receipt_rejects_changed_content_and_unmatched_bpw(self):
        changed = receipt()
        changed["probes"][0]["weight_mse"] = 0.5
        with self.assertRaisesRegex(MODULE.ReceiptError, "receipt_id"):
            MODULE.verify(changed)

        changed_bpw = receipt()
        changed_bpw["probes"][0]["bpw"] = 2.0
        reseal(changed_bpw)
        with self.assertRaisesRegex(MODULE.ReceiptError, "same matched bpw"):
            MODULE.verify(changed_bpw)

    def test_receipt_requires_four_distinct_classes_per_family(self):
        changed = receipt()
        changed["probes"][1]["tensor_class"] = changed["probes"][0]["tensor_class"]
        with self.assertRaisesRegex(MODULE.ReceiptError, "repeats tensor class"):
            MODULE.verify(changed)

    def test_receipt_rejects_non_monotone_depth_and_duplicate_json_fields(self):
        changed = receipt()
        changed["probes"][0]["sequence_positions"] = [128, 128, 2048]
        with self.assertRaisesRegex(MODULE.ReceiptError, "must increase"):
            MODULE.verify(changed)
        with self.assertRaisesRegex(MODULE.ReceiptError, "duplicate field"):
            MODULE._object([("schema", MODULE.SCHEMA), ("schema", MODULE.SCHEMA)])


if __name__ == "__main__":
    unittest.main()
