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
                    "state_divergence": [terminal / 2, terminal, terminal * 2],
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
        "sequence_count": MODULE.CALIBRATION_SEQUENCES,
        "tokens_per_sequence": MODULE.TOKENS_PER_SEQUENCE,
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


def preflight(value: dict) -> dict:
    probes = []
    for probe in value["probes"]:
        probes.append({
            "family": probe["family"],
            "tensor_class": probe["tensor_class"],
            "tensor_name": probe["tensor_name"],
            "tensor_index": probe["tensor_index"],
            "layer": probe["tensor_index"],
            "shape": [8, 8],
            "source_shard": "model-00001-of-00015.safetensors",
        })
    result = {
        "schema": "tritium.qwen36-gdn-probe-preflight.v1",
        "repository": MODULE.REPOSITORY,
        "revision": MODULE.REVISION,
        "state": "prepared-not-measured",
        "evidence_scope": "local-config-index-and-safetensors-header-only",
        "config_sha256": "a" * 64,
        "weight_index_sha256": "b" * 64,
        "probes": probes,
        "limitations": ["local metadata only"],
    }
    result["preflight_id"] = "sha256:" + hashlib.sha256(MODULE.canonical(result)).hexdigest()
    return result


class QwenGdnSensitivityReceiptTests(unittest.TestCase):
    def test_receipt_derives_pass_when_delta_terminal_is_within_threshold(self):
        verified = MODULE.verify(receipt(delta_terminal=2.0, control_terminal=1.0))
        self.assertEqual(verified["gate"], "pass")
        self.assertEqual(verified["route_to_refined_track"], [])
        self.assertEqual(verified["deltanet_state_divergence_max"], 4.0)
        self.assertEqual(verified["full_attention_state_divergence_median"], 2.0)
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

    def test_receipt_rejects_json_integer_that_overflows_float_conversion(self):
        changed = receipt()
        changed["probes"][0]["weight_mse"] = 10**1000
        reseal(changed)
        with self.assertRaisesRegex(MODULE.ReceiptError, "finite number"):
            MODULE.verify(changed)

    def test_receipt_requires_four_distinct_classes_per_family(self):
        changed = receipt()
        changed["probes"][1]["tensor_class"] = changed["probes"][0]["tensor_class"]
        with self.assertRaisesRegex(MODULE.ReceiptError, "repeats tensor class"):
            MODULE.verify(changed)

    def test_receipt_requires_complete_partition_and_comparable_depth_curves(self):
        changed = receipt()
        changed["sequence_count"] = 128
        with self.assertRaisesRegex(MODULE.ReceiptError, "complete frozen calibration"):
            MODULE.verify(changed)

        changed = receipt()
        changed["probes"][1]["tensor_index"] = changed["probes"][0]["tensor_index"]
        with self.assertRaisesRegex(MODULE.ReceiptError, "indexes must be unique"):
            MODULE.verify(changed)

        changed = receipt()
        changed["probes"][1]["sequence_positions"] = [128, 1024, 2048]
        with self.assertRaisesRegex(MODULE.ReceiptError, "same sequence-depth"):
            MODULE.verify(changed)

    def test_receipt_rejects_non_monotone_depth_and_duplicate_json_fields(self):
        changed = receipt()
        changed["probes"][0]["sequence_positions"] = [128, 128, 2048]
        with self.assertRaisesRegex(MODULE.ReceiptError, "must increase"):
            MODULE.verify(changed)
        with self.assertRaisesRegex(MODULE.ReceiptError, "duplicate field"):
            MODULE._object([("schema", MODULE.SCHEMA), ("schema", MODULE.SCHEMA)])

    def test_receipt_requires_comparable_state_divergence_curves(self):
        changed = receipt()
        del changed["probes"][0]["state_divergence"]
        reseal(changed)
        with self.assertRaisesRegex(MODULE.ReceiptError, "frozen schema"):
            MODULE.verify(changed)

        changed = receipt()
        changed["probes"][0]["state_divergence"] = [0.5, 1.0]
        reseal(changed)
        with self.assertRaisesRegex(MODULE.ReceiptError, "state-divergence curve"):
            MODULE.verify(changed)

        changed = receipt()
        changed["probes"][0]["state_divergence"][1] = float("inf")
        with self.assertRaisesRegex(MODULE.ReceiptError, "outside its allowed range"):
            MODULE.verify(changed)

    def test_measurement_can_be_joined_to_exact_preflight_names_and_ordinals(self):
        measured = receipt()
        prepared = preflight(measured)
        result = MODULE.verify(measured, prepared)
        self.assertEqual(result["preflight_id"], prepared["preflight_id"])
        self.assertEqual(
            result["evidence_scope"],
            "receipt-structure-rule-and-local-preflight-join-only",
        )

    def test_preflight_join_rejects_changed_ordinal_and_invalid_content_id(self):
        measured = receipt()
        prepared = preflight(measured)
        prepared["probes"][0]["tensor_index"] += 1
        prepared["preflight_id"] = "sha256:" + hashlib.sha256(
            MODULE.canonical({key: item for key, item in prepared.items() if key != "preflight_id"})
        ).hexdigest()
        with self.assertRaisesRegex(MODULE.ReceiptError, "names or ordinals"):
            MODULE.verify(measured, prepared)

        prepared = preflight(measured)
        prepared["config_sha256"] = "f" * 64
        with self.assertRaisesRegex(MODULE.ReceiptError, "preflight_id"):
            MODULE.verify(measured, prepared)


if __name__ == "__main__":
    unittest.main()
