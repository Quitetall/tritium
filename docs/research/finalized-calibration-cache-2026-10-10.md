# Finalized paired calibration cache candidate

Proposed ADR0060 continuation on `32853e49b8374e4dc57e09fa3300f5b6c82c1986`.
This is an owned in-memory capture/replay candidate, not the persisted/spooled
full-RSS workflow or an independently verified recipe/quality/release receipt.

The builder admits actual retained recipe/tokenizer and scalar/token/mask payload
bytes, samples and context windows before copying. Invalid capture is atomic,
including nonfinite masked inputs/gradients, bad weights/indices or mismatched
encodings. Finalization consumes the builder and hashes actual immutable content;
there are no bootstrap fake hashes or post-reduction source rebinding. Limits
exclude metadata/capacity/allocator/caller buffers and full RSS; reducer admission
uses its own unchanged numerical budget.

One cache/token hash implementation owns the candidate preimage documented in
ADR0060. Original model identity stays declared until actual engine admission.
Exact tokenizer and objective/teacher/recovery recipe bytes are retained and
bound, not reserialized. Actual token IDs, context boundaries, paired f32 values,
f64 weights and masks bind the cache; absent selections canonicalize to defaults.
Changing a physical replay chunk or reduction partition does not rehash the cache.
Changing a declared model-context window does, because forward/objective context
can change. Raw input Gram and dense/indexed Fisher/KFAC are still produced solely
by the existing shared collector; there is no copied numerical reduction loop.

The actual tape test now captures original toy parameters with the canonical
source hasher and actual post-MSE backward buffers through this cache, then checks
same-cache chunked/full replay. It is a software graph test, not HF loader source
admission, production tokenization/recipe provenance or measured model quality.
The explicit real SmolLM tap fixture is unchanged; it does not yet exercise a
production persisted cache. Tests/gates must finish before claims of success.

Remaining: durable shared/spooled/mmap cache and immutable source/data/recipe
verification; registered factor/metric observations, row-aware direct fitting,
complete inventories and package replay/tamper rejection; full source/driver/KV/
metadata/RSS admission; all CLI/Qwen/Python/HF and legacy fixture migrations;
hybrid/MTP coverage; frontier/GPU/out-of-core and physical/quality/reproduction/
serving/packaging/zoo/community/human activation gates. No frozen gate is weakened.

The unchanged-source candidate check completed exit 0: four actual cache/replay
tests and three ordinary actual-tape tests passed; the original real SmolLM test
was explicitly declared ignored in this invocation, not silently passed or newly
cache-qualified. All four derived projections and drift, scoped strict Clippy/docs
and bare-metal alloc/serde schema passed. The source was then integrated over the
source-guard candidate before commit; combined committed regression and canonical
gates/publication remain required. These are software tests, not audited producer
execution, model-quality, full memory, physical or release qualification.
