# tritium-py

PyO3 bindings for the pytritium Python package.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

The expert `tritium.convert_model_plan(plan_json, source_shards, output_dir,
run_id, *, asset_paths=None, input_grams=None, calibration_caches=None,
calibrated=False, max_plan_bytes=16777216)` runs the same Rust model conversion
engine as `tritium convert --plan`. Supply the existing Rust `QuantPlan` JSON,
the complete original safetensors shard inventory and exact named asset paths.
`calibrated=True` selects `CalibratedQuantPlan` and requires its bound `.trit`
cache files. `input_grams` binds named raw little-endian f64 square Gram files;
neither absent nor invalid metrics fall back to Identity. Computation releases
the GIL. The destination must be new and is never overwritten.

The readonly `NativeModelConversion` reports actual source/package/model/evidence
identities and publication warnings. `verify()` reopens `model.trit` and
`evidence.jsonl` and verifies their exact identity and binding.
`preserved_tensor(level, name, max_bytes=67108864)` returns precision, shape and
original dense bytes after verification; it does not decode additive tensors.
Revalidation reads whole files under declared bounds, not a streaming/RSS claim.
This converts the complete declared floating-point inventory, not arbitrary HF
architectures, and is not a model-quality or release-qualification receipt.
The existing torch PTQ lifecycle and its matrix-only packages are unchanged.

`tritium.torch.convert_native_module(prepared, output_dir, run_id, *, actions,
levels, limits, work_dir, asset_paths, architecture="torch-module",
input_grams=None)` is an opt-in persistent-state adapter to that engine. Supply
the existing Rust Fit actions for every selected coverage owner and the original
config/tokenizer assets required by QuantPlan. Its temporary source files live
under an existing caller work directory and are cleaned up after conversion.
Post-publication cleanup faults remain explicit warnings on the successful result.

The state catalog preserves exact aliases, biases, norms and buffers. Admitted
positive-shaped F16/BF16/F32 owners use native Fit/KeepDense. Scalar, empty,
float64, integer and boolean state is preserved as a safetensors asset, not
silently admitted as native executable tensors. The current state dtype profile
is F16/BF16/F32/F64, signed I8/I16/I32/I64, U8 and bool; other dtypes, custom
non-tensor state, meta/sparse/quantized tensors and overlapping views reject.
`result.asset(name, max_bytes=67108864)` reads bounded original asset bytes after
package/log verification. This is source packaging, not fresh-shell Torch/HF
application, calibrated phase migration or runtime/quality qualification.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

Stage-7 campaign executors consume frozen token evidence through
`tritium.torch.Stage7CausalData.open`. It strictly binds the campaign pack and
model-tokenizer identities, reads only the selected sequence window, performs a
terminal same-handle payload check, and yields replayable PyTorch causal-LM
batches. A data receipt is input provenance only, not an execution or quality
result.

`tritium.torch.run_stage7_smoke_model(model, data, output_dir)` consumes that
validated data and transactionally executes or resumes the five concrete smoke
stages: capture, fit, allocate, package, and evaluate. Its result binds source,
tokens, conversion, allocation, package bytes, and measured causal loss. It is
an engineering receipt, not a release-qualified claim.

`tritium.torch.run_stage7_smollm2_smoke(campaign_path, model_dir, output_dir)`
is the frozen campaign wrapper. It admits only the pinned SmolLM2-135M revision,
exact tokenizer/files, source-derived rank-2 inventory, and first 128 C4 members
from the governed evidence pack, then emits the schemas consumed by the Stage-7
qualifier. The wrapper uses local files only and fails closed on unknown resume
artifacts. G64-shaped matrices are stored using explicit SALT V2 package-version
2 scale geometry; G128-only packages retain byte-identical version 1 encoding.
