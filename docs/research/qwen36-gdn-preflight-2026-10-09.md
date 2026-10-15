# Qwen3.6 Gated DeltaNet probe preflight — 2026-10-09

## Result

The metadata-only preflight for the frozen eight-probe Gated DeltaNet (GDN)
sensitivity study completed against the local Qwen3.6-27B checkpoint inventory.
The probe set contains four DeltaNet matrices and four full-attention controls,
covering qkv, output, gate/up, and down classes in distinct layers. The exact
probe tensor names and shapes are in the immutable preflight receipt.

- Receipt schema: `tritium.qwen36-gdn-probe-preflight.v1`
- State: `prepared-not-measured`
- Preflight identity: `sha256:e17471ab372db83d5ee602977dab13818c20772544d5888a6261c1d95dbda232`
- Pinned model repository/revision: `Qwen/Qwen3.6-27B` at
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- Checkpoint `config.json` SHA-256:
  `69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c`
- Checkpoint weight-index SHA-256:
  `a8ad2c26fb707ff8c245806315b03e3b4b74595528492423af5dae0ce39b4d9b`
- Durable local receipt:
  `/mnt/4tb/tritium-evidence/qwen36-gdn-preflight-8beec6fe/preflight.json`
- Preflight file SHA-256:
  `a78c1a2b58a504dd6e4a1041e8b2310a41139cfcd5e560d6edd22ddba7177add`

The preflight script validated the expected 64-layer Qwen3.6 text config, the
1,199-entry tensor index, 506 language/MTP matrices, tensor-family/class
assignments, and selected tensor headers. `python -m pytest -q
scripts/tests/test_prepare_qwen36_gdn_probes.py` passed (7 tests).

## Evidence boundary and remaining gate

This is only a local structural inventory. The script explicitly does not
authenticate the local checkpoint against Hugging Face, establish calibration
pack provenance, load tensor payloads, or execute the model. It is not a GDN
measurement, a routing result, a quality claim, or a release-gate pass. The
frozen measurement still needs matched-bpw PTQ probes, recurrence state/output
divergence over the frozen calibration sequences, and an independently
reproduced admitted measurement receipt before Stage 8. Plan 0043's thresholds,
coverage, and routing rule are unchanged.

The source checkout was at `8beec6fec130bc46fe440458561afc793c33246d` when the
preflight was produced. No model fitting or model inference was run for this
preflight.
