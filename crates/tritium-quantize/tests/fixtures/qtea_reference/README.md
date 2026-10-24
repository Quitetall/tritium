# QTEA arithmetic oracle (not a model or solver qualification)

`golden.json` was generated on CPU using the actual MIT author implementation
at `e39bd8c0297e3427daeb404427c74cb6e202b683`, Torch2.14.1, one Torch thread.
The generator verifies the SHA256 of both supplied Python source files before
importing them. No source is fetched or code executed during ordinary Rust tests.

To reproduce, obtain `qtea/qtea.py` and `qtea/quantizer.py` from that pinned
revision under an isolated source directory, retaining its MIT license, then run:

```sh
timeout 45s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -I -B crates/tritium-quantize/tests/fixtures/qtea_reference/generate.py \
  /absolute/path/to/source-parent
```

The command emits JSON, not files; compare it with the committed fixture. It does
not download weights/data, fit a model, run LDL/Gram feedback or touch a GPU.
Source licensing/attribution is retained in the crate's `qtea_reference/LICENSE-MIT`
and workspace NOTICE. Source hashes and Torch identity are in the fixture.

Coverage:12 group fits (including a128-column group),108 column refinements,
1010 FP8 cases spanning every positive finite level, each adjacent midpoint and
its neighboring f32s, both signs, four sparse selection/scaling cases and seven
saliency count/order cases. Rust adds refusal, exact logical-budget, tie-policy,
zero-round and overflow cases through the same public arithmetic interface.

Tritium's scalar f32 reductions use a4e-6*(1+abs(expected)) comparison bound for
parameters/decoded values; trits, sparse positions and FP8 codes are exact on these
fixtures. Scale1 rounding fixtures compare decoded f32 bits, including signed
zero. These bounds are test coverage, not a universal numerical-error guarantee.
Upstream equal-score sort ties are not a portable contract: native ranking
deliberately uses ascending original index and tests that policy separately.
Native constant/empty-initial-assignment groups refuse rather than returning
upstream NaN. No quality, throughput, physical-memory or release claim follows.
