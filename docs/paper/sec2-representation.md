# 2. The SALT Representation

## 2.1 Additive ternary planes

SALT represents each weight group $g$ as a sum of at most three ternary
planes with one non-negative scale per plane and scale group:

$$
\hat{W}_{g,i} \;=\; \sum_{p=1}^{P_g} s_{g,p}\, T_{g,p,i},
\qquad T_{g,p,i} \in \{-1, 0, +1\},\quad s_{g,p} \ge 0,\quad P_g \in \{1,2,3\}.
$$
<!-- receipt: docs/adr/0028-salt-v2-additive-ternarization.md §1 "Preserve a zero-point-free additive ternary representation"; crates/tritium-format/src/salt_v2_package.rs SALT_V2_MAX_PLANES = 3 -->

The construction descends from residual binary bases [REF:abcnet] and
additive quantization [REF:aqlm], specialised so that every plane is
ternary. Flat ternary quantization in the BitNet b1.58 style
[REF:bitnet158] is recovered exactly as the $P_g = 1$ special case. The
original SALT fitter — greedy AbsMean residual expansion with
sensitivity-ranked plane allocation, in the spirit of HAWQ and
SqueezeLLM [REF:hawq] [REF:squeezellm] — has since been superseded by
the joint solver of Section 3, but the *representation* above is
unchanged and is the invariant everything else in this paper serves.
<!-- receipt: docs/adr/0001-salt-quantization.md (V1 pipeline); docs/adr/0028 "supersedes the fitting, allocation, and accounting decisions in ADR 0001 while preserving its additive ternary execution invariant" -->

The representation is chosen for what the kernel may do with it. For a
group $g$ and plane $p$ the measured inference path computes

$$
a_{g,p} = \sum_i T_{g,p,i}\, x_{g,i}, \qquad
y = \sum_{g,p} s_{g,p}\, a_{g,p},
$$
<!-- receipt: docs/adr/0028 §6 "Co-design the kernel without weakening the format invariant" -->

i.e. an add, subtract, or skip per coefficient inside each plane, and one
scale application per accumulator. A $P$-plane weight is $P$ passes of
the same multiply-free kernel. The format therefore admits no arbitrary
floating-point codebooks or per-weight centroids, no lattice, trellis, or
Golay decoding to multilevel floating values, no floating residual or
outlier weights outside the planes, no dense affine reconstruction, and
— critically for the benchmark discipline of Section 6 — no dense
dequantization as the measured inference path. Methods that relax any of
these (AQLM, QuIP\#, QTIP, LLVQ) remain comparison targets and donors of
representation-independent optimization machinery, but their storage is
inadmissible here [REF:aqlm] [REF:quip-sharp] [REF:qtip] [REF:llvq].
<!-- receipt: docs/adr/0028 §1 disallowed-artifact list; §Alternatives "Adopt AQLM, QTIP, LLVQ, or UniSVQ storage — Rejected" -->

## 2.2 Why zero-point-free

There is deliberately no per-group zero point or affine bias $b_g$. A
signed scale is canonicalised by negating its trits, so scales are
non-negative by construction. The rationale is executional rather than
statistical: a zero point introduces an additional dense reduction over
each input group and a second scale path, which breaks the pure
add/subtract/skip contract and forfeits the zero-state skip that makes
sparse ternary planes cheap. It also weakens the zero-centred symmetry
that lets one canonical form serve encoder, decoder, and kernel alike.
Binary-plus-bias storage in the BPDQ style [REF:bpdq] was considered and
rejected on these grounds; BPDQ remains a baseline rather than a reason
to add $b_g$. A SALT artifact is consequently fully described by trits,
scales, and geometry.
<!-- receipt: docs/adr/0028 §Alternatives "Add a group bias or zero point as in affine binary decomposition — Rejected" -->

## 2.3 Scale groups and validation

Scales are stored as IEEE half-precision (f16) values, one per group of
128 coefficients; plane presence is allocated at a coarser 256-coefficient
macrotile.
<!-- receipt: crates/tritium-format/src/salt_v2_package.rs SALT_V2_SCALE_GROUP_SIZE = 128, SALT_V2_ALLOCATION_TILE_SIZE = 256 -->
G128 is the binding first geometry; G64 and G256 are preregistered
ablations that may replace it only if their held-out
quality/physical-byte/runtime point is non-dominated, and either
alternative requires an explicit versioned scale-geometry field before it
can be serialized at all.
<!-- receipt: docs/adr/0028 binding profile + Amendment 2026-07-15 "The first package/runtime reference remains G128-only" -->

Validation is unusually strict for a weight format, because the format
doubles as the paper's accounting instrument. Construction rejects: any
coefficient outside $\{-1,0,+1\}$; a plane whose scale count differs from
$\lceil n/128 \rceil$; a non-finite scale; a *negative* scale, including
negative zero, checked at the sign bit; and a zero scale over a group
containing a nonzero trit — a group that pretends to be free but is not.
Planes within a tile form a dense prefix: plane three cannot exist
without plane two, so $P_g$ is always the length of a nested refinement
rather than an arbitrary subset.
<!-- receipt: crates/tritium-format/src/salt_v2_package.rs SaltV2Plane::new (NonCanonicalTrit, WrongScaleCount, NonFiniteScale, NegativeScale, ZeroScaleForNonzeroGroup), SaltV2Tile::new, SaltV2PackageError::NonNestedPlaneMap -->
A tensor's identity is a content digest over its transform parameters,
geometry, ordered tile/plane structure, decoded trits, and exact f16
scale bits — and over nothing physical. Repacking the same tensor under
a different codec leaves the identity unchanged, which is what allows the
codec comparison in Section 2.4 to be a comparison of codecs and not of
accidentally different tensors.
<!-- receipt: crates/tritium-format/src/salt_v2_package.rs SaltV2Tensor::semantic_tensor doc: "Repacking the same tensor as D2, B3, or S34 therefore leaves this identity unchanged" -->

## 2.4 Physical codecs: D2, B3, S34

One semantic tensor admits three physical codecs, and a published
package uses exactly one. **D2** stores four trits per byte in aligned
two-bit fields; it is the mandatory correctness oracle and the fast CUDA
baseline, at $2.125$ physical bpw per plane at G128 (32 payload bytes
plus one 2-byte scale per 128 coefficients).
<!-- receipt: crates/tritium-format/src/salt_v2.rs D2_TRITS_PER_BYTE = 4; rate from ADR 0028 §2 R_direct(P,G) = 2P + 16P/G at P=1, G=128 -->
**B3** packs five radix-3 trits per byte (243 of 256 byte values are
legal; the rest are rejected on decode), giving $1.75$ physical bpw per
plane at G128 — 26 payload bytes plus a 2-byte scale per 128
coefficients. We note explicitly that the radix-3 information rate at
this geometry, $1.7109$ bpw, is a lower bound and not the physical rate;
conflating the two is precisely the habit Section 2.5 legislates against.
<!-- receipt: crates/tritium-format/src/salt_v2.rs B3_TRITS_PER_BYTE = 5, B3_CODE_COUNT = 243; rates from docs/adr/0028 Amendment 2026-07-15 "prefix pricing and physical B3 rate" -->
**S34** encodes a structured one-zero-per-four constraint: each 4-trit
group has exactly one zero, giving 32 legal states in five bits, or
$1.375$ bpw per plane over a full 256-coefficient tile (40 payload bytes
plus four scale bytes). S34 is available only to tensors trained or
recovered under that structural constraint; it is not a lossy
post-processing of an unconstrained tensor.
<!-- receipt: crates/tritium-format/src/salt_v2.rs S34_TRITS_PER_GROUP = 4, S34_BITS_PER_GROUP = 5; rate from ADR 0028 Amendment 2026-07-15 -->
D2 is always produced as the reference; B3 or S34 enters a claimed
frontier only when it is Pareto-better in exact serialized bytes or
exact resident bytes without losing the selected end-to-end wall-time
gate — decode overhead is measured, never assumed away.
<!-- receipt: docs/adr/0028 binding profile (Pareto admission rule); docs/plans/0043 §Frozen campaign structure -->

## 2.5 Physical rates and the exact-byte ceiling

$\log_2 3 \approx 1.585$ is the information content of one trit, not the
storage cost of an artifact. SALT therefore records four distinct rates
and forbids the first from standing alone:

$$
\begin{aligned}
\text{logical\_bpw} &= \textstyle\sum_g |g|\, P_g \log_2 3 \,/\, N_q, \\
\text{matrix\_bpw} &= 8\,B_{\text{matrix}} / N_q, \qquad
\text{artifact\_bpw} = 8\,B_{\text{artifact}} / N, \qquad
\text{resident\_bpw} = 8\,B_{\text{resident}} / N,
\end{aligned}
$$
<!-- receipt: docs/adr/0028 §2 "Make physical bytes the optimization constraint" -->

where the matrix rate charges every encoded plane, scale, presence map,
alignment, and descriptor byte, and the artifact and resident rates
additionally count container metadata, unquantized tensors, and any
runtime shadow. Under this accounting, two-plane ternary storage in the
PTQTP style is $4.25$ bpw at G128 with direct two-bit trits, not
"1.58-bit": two independent trits have a $3.17$-bit information floor
before any scale is stored [REF:ptqtp].
<!-- receipt: docs/adr/0028 §2 ("PTQTP-style two-plane storage is 4.25 bpw at G128") and §Alternatives ("Call two planes 1.58-bit ... Rejected"; 3.1699-bit floor = 2·log2 3) -->
A nominal rate cannot authorize a run or a claim; logical bpw may be
printed only beside the physical figures. We refer to this rule as the
logical-bpw ban, and it is enforced mechanically: the encoder's byte
counters are tested to equal actual file lengths and steady-state device
allocations, so the reported rate is an observable, not an estimate.
<!-- receipt: docs/adr/0028 preregistered correctness gates ("byte counters equal actual file lengths and steady-state device allocations") -->

Rate targets are frozen as integer byte ceilings before any fitting: for
$N_q$ quantized weights the matrix ceiling is
$\lfloor r \cdot N_q / 8 \rfloor$ bytes, and the allocator may not exceed
it by so much as one alignment block. The campaign's primary matrix-rate
points are `R2` at $2.25$ bpw (the `CompactV1` profile), `R3` at $3.50$
bpw (the primary `NearLosslessV1` operating point, below that profile's
hard $4.0$-bpw ceiling), and `R4` at $4.25$ bpw, a D2 dual-plane control
that can publish as neither stable profile.
<!-- receipt: docs/plans/0043-salt-v2-sota-campaign.md §Physical-rate points (R2/R3/R4 table, byte-ceiling formula) -->
`CompactV1` is moreover a *successively refinable prefix* of
`NearLosslessV1`: each group has one deterministic jointly fitted
$P_{\max}$ master solution whose planes are ordered by their measured
prefix-loss curve, both profiles allocate over that same curve, and the
prefix-derivation API clones trits and scale bits verbatim with no
refitting path. The compact artifact is never independently re-fit from
the near-lossless one, so shipping both costs the bytes of one.
<!-- receipt: docs/adr/0028 Amendment 2026-07-15 (exact Compact-prefix invariant); crates/tritium-format/src/salt_v2_package.rs SaltV2Package::derive_prefix ("no fitting or requantization path") -->

## 2.6 Mixed plane counts without dense padding

Sensitivity-directed allocation is only worth its metadata if variable
$P_g$ does not smuggle in padding. Plane presence is recorded as a
two-bit code per 256-coefficient macrotile — $0.0078$ bpw — serialized
once as a package-global stream whose terminal fragment rides in unused
high bits of mandatory count words, so the tail bits occupy no
additional byte yet are still reported explicitly rather than hidden in
header accounting.
<!-- receipt: crates/tritium-format/src/salt_v2_package.rs module doc + SALT_V2_TENSOR_COUNT_BITS = 26, SALT_V2_TILE_COUNT_BITS = 62, SaltV2PackageLedger.allocation_map_embedded_bits; 2/256 = 0.0078125 bpw from ADR 0028 binding profile -->
Only present planes occupy payload or scale bytes; tensor-wide
maximum-$P$ padding is forbidden, and adaptive dispatch or index
metadata is capped at $0.01$ bpw. The indexed runtime layout keeps the
same discipline on the device: it retains codec payloads, scales, the
complete map bytes, and one 4-byte rank prefix per 256 tiles to bound
map scans, and its ledger carries a `dense_shadow_bytes` field that is
structurally zero — the absence of a dense reconstructed weight shadow
is an audited quantity, not an implementation hope.
<!-- receipt: docs/adr/0028 binding profile (no max-P padding; 0.01 bpw cap); crates/tritium-format/src/salt_v2_package.rs SaltV2IndexedRuntimeLedger (SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES = 256, dense_shadow_bytes "structurally zero") -->

## 2.7 Interoperability with the emerging 2-bit standard

Finally, the base plane is deliberately compatible with the format the
wider ecosystem is converging on. llama.cpp's Q2_0 stores 64 weights per
block as an f16 scale followed by sixteen bytes of 2-bit codes — 18
bytes per block, $2.25$ bpw — with levels $\{-1, 0, +1, +2\}$
[REF:llamacpp-q2_0]. Tritium ships an import/export port of the
reference quantize/dequantize routines (llama.cpp PRs #24448, CPU, and
#25707, CUDA), verified against golden byte vectors hand-computed from
the reference layout; because Tritium is ternary, the packer only ever
emits the three ternary codes and the unpacker rejects the $+2$ level as
out of range, so a round-tripped tensor is ternary by construction.
<!-- receipt: crates/tritium-format/src/q2_0.rs (Q2_0_GROUP_SIZE = 64, Q2_0_BLOCK_BYTES = 18, port of PR #24448/#25707, code-3 rejection) -->
Q2_0's ternary subset is, in SALT terms, a single dense plane at a finer
scale geometry; SALT adds what one plane cannot express — nested
refinement planes, sensitivity-directed allocation, and the exact-byte
ledger — while remaining exportable to the mainstream runtime at the
base rate. We regard this as the correct division of labour: the
community standardises the one-plane container, and this paper's
contribution is what can be layered on top of it without breaking the
add/subtract/skip contract.

