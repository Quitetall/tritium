# References ledger — [REF:*] key → target

Canonical key list for the SALT whitepaper. Every `[REF:key]` used in
`sec0`–`sec8-9` appears here exactly once. Dedupe already applied in the
section sources: `kfac` → `kfac2015` (sec3), `bonsai` → `bonsai-family`
(sec1; **not** merged with `bonsai27b` — they are two distinct PrismML
releases, the Apr-2026 1.7B/4B/8B family and the Jul-2026 27B build).

Status legend:
- **OK** — target known with high confidence.
- **VERIFY** — best-known target given; confirm the id/URL before the
  bibliography freezes.
- **NEEDS CITATION** — no reliable target known; must be resolved before
  submission.
- **INTERNAL** — deliberately vendor-anonymous in prose; resolves to the
  companion-artifact verification record, not an external work. Decide at
  bibliography time whether these stay repo-internal citations.

| key | target | status | used in |
|---|---|---|---|
| abcnet | ABC-Net, Lin, Zhao & Pan, NeurIPS 2017 — arXiv:1711.11294 | OK | 2 |
| aqlm | AQLM — arXiv:2401.06118 | OK | 2, 3, 7 |
| bastion2026 | BASTION tree-verify speculative decoding (2026) | NEEDS CITATION | 6 |
| bcjr-qat | BCJR-QAT (proxy-failure finding: per-layer MSE up, ppl down) | NEEDS CITATION | 7 |
| bengio2013ste | Bengio, Léonard & Courville, straight-through estimator — arXiv:1308.3432 | OK | 5 |
| bitnet158 | BitNet b1.58, "The Era of 1-bit LLMs" — arXiv:2402.17764 | OK | 1, 2, 7 |
| bitnet2b4t | BitNet b1.58 2B4T Technical Report — arXiv:2504.12285 | OK | 1, 6, 7 |
| bitnetcpp | Microsoft BitNet kernels — https://github.com/microsoft/BitNet (paper: bitnet.cpp, arXiv:2502.11880) | VERIFY (arXiv id) | 7 |
| blockgtq | Block-GTQ (~2-bit KV via structure-aware allocation, NIAH 70.6→97.4) | NEEDS CITATION | 7 |
| bonsai-family | PrismML Ternary Bonsai 1.7B/4B/8B (Apache-2.0, Apr 2026) — https://huggingface.co/collections/prism-ml/ternary-bonsai | OK | 1 |
| bonsai27b | PrismML Ternary Bonsai 27B (Jul 2026) — https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf ; independent repro: https://github.com/Astezelex/bonsai-27b-16gb-bench | OK | 7 |
| bpdq | BPDQ (binary planes + group bias; coordinate search, Hessian refit) | NEEDS CITATION | 2, 7 |
| catq | CAT-Q (PTQ ternary via softened two-sided relay, 2026) | NEEDS CITATION | 3, 7 |
| dfssm | DF-SSM (recurrent-state quantization for linear-attention layers) | NEEDS CITATION | 7 |
| fairyfuse | FairyFuse (multiply-free CPU kernel; 130× GPU-regression claim refuted). Related line: Fairy2i, arXiv:2512.02901 | NEEDS CITATION | 7 |
| falconedge | Falcon-Edge (TII) — https://huggingface.co/blog/tiiuae/falcon-edge | OK | 1 |
| fisher-kron | Fisher-Kronecker quantization (Kronecker curvature at ~10× lower capture cost) | NEEDS CITATION | 7 |
| flashdecoding | Flash-Decoding, Dao et al. — https://crfm.stanford.edu/2023/10/12/flashdecoding.html | OK | 6 |
| gptq2022 | GPTQ — arXiv:2210.17323 | OK | 4 |
| guidedquant | GuidedQuant (end-loss-guided quantization, ICML 2025) | NEEDS CITATION (arXiv id) | 7 |
| hawq | HAWQ, Dong et al., ICCV 2019 — arXiv:1905.03696 | OK | 2, 7 |
| hestia | HESTIA (softmax relaxation + Hessian-trace-scheduled temperature, 2026) | NEEDS CITATION | 7 |
| hinton2015distillation | Hinton, Vinyals & Dean, knowledge distillation — arXiv:1503.02531 | OK | 5 |
| hutchinson1990 | Hutchinson 1990, Commun. Statist. Simula. 19(2):433–450 — doi:10.1080/03610919008812866 | OK | 4 |
| kfac2015 | Martens & Grosse, K-FAC — arXiv:1503.05671 | OK | 3, 4 |
| kronq | KronQ (Kronecker-factored curvature sketches for PTQ) | NEEDS CITATION | 7 |
| lcqat | LC-QAT — arXiv:2606.10531 (per sec7 UNSOURCED note; not yet checked against the abstract) | VERIFY | 7 |
| leviathan2023spec | Leviathan, Kalman & Matias, speculative decoding — arXiv:2211.17192 | OK | 6 |
| littlebit | LittleBit (0.1 bpw low-rank binarized factorization) | NEEDS CITATION | 7 |
| llamacpp-q2_0 | llama.cpp Q2_0 — https://github.com/ggml-org/llama.cpp ; PR #24448 (CPU), PR #25707 (CUDA, merged 2026-07-30) | OK | 1, 2, 6, 7 |
| llvq | LLVQ (Leech lattice + spherical gain coding; Llama2-7B 6.83→5.48) | NEEDS CITATION | 2, 7 |
| loshchilov2017sgdr | SGDR, cosine annealing with warm restarts — arXiv:1608.03983 | OK | 5 |
| loshchilov2019adamw | Decoupled weight decay (AdamW) — arXiv:1711.05101 | OK | 5 |
| merity2016wikitext | Merity et al., WikiText / Pointer Sentinel — arXiv:1609.07843 | OK | 5 |
| mote | MoTE (all-routed-experts-ternary recipe; cite recipe only — its iso-memory table failed our verification) | NEEDS CITATION | 7 |
| oaem | OA-EM (output-aware E/M initialization; 3B 2-bit 352.39→16.82) | NEEDS CITATION | 7 |
| op2026 | Ordentlich & Polyanskiy quantized-matmul series — opener: arXiv:2410.13780; confirm the 2026 installment the key names | VERIFY | 4, 7 |
| paretoq | ParetoQ — arXiv:2502.02631 | VERIFY | 7 |
| promptlookup | Prompt lookup decoding, A. Saxena — https://github.com/apoorvumang/prompt-lookup-decoding | OK | 6 |
| pt2llm | PT²-LLM (training-free ternarization, ICLR 2026 per docs/research-ternary-sota-mid2026.md §1) | NEEDS CITATION (arXiv id) | 1, 7 |
| ptqtp | PTQTP (two ternary planes, post-training) | NEEDS CITATION | 1, 2, 7 |
| pvtuning | PV-Tuning — arXiv:2405.14852 | OK | 7 |
| qtip | QTIP (trellis coding + incoherence processing) — arXiv:2406.11235 | OK | 2, 7 |
| quip-sharp | QuIP# (E8 lattice) — arXiv:2402.04396 | OK | 2, 7 |
| refuted-kernel-claims | INTERNAL — docs/research-ternary-sota-mid2026.md §1.1 (spbitnet sparse-ternary kernel; all load-bearing perf claims REFUTED 0-3 / 1-2) | INTERNAL | 1 |
| refuted-ptq-comparison | INTERNAL — docs/research-ternary-sota-mid2026.md §8 (PT-BitNet, Neural Networks 2025; "61% vs 51.2%" headline comparison REFUTED 0-3) | INTERNAL | 1 |
| refuted-quality-table | INTERNAL — docs/research-ternary-sota-mid2026.md §2.1 (Ternary Bonsai 8B vendor benchmark-superiority table REFUTED 0-3) | INTERNAL | 1 |
| slidesparse | SlideSparse (6:8 lossless sliding-window sparsity, 1.33× e2e) | NEEDS CITATION | 7 |
| smollm2 | SmolLM2 — arXiv:2502.02737 | OK | 5 |
| squeezellm | SqueezeLLM — arXiv:2306.07629 | OK | 2, 7 |
| tequila | Tequila (STE-deadzone gradient path for ternary QAT, 2026) | NEEDS CITATION | 7 |
| tmac | T-MAC (LUT-based low-bit CPU inference, EuroSys 2025) — arXiv:2407.00088 | OK | 7 |
| unisvq | UniSVQ (quaternary codes + dense affine decoder; 2-bit Qwen3-32B 7.61→9.26) | NEEDS CITATION | 7 |
| vbq | VBQ (learned per-group precision allocation; 1B from-scratch probe) | NEEDS CITATION | 7 |
| veclut | Vec-LUT (parallel LUT inference; 1.60 bpw lossless ternary packing, 2026) | NEEDS CITATION | 7 |
| vptq | VPTQ — arXiv:2409.17066 | OK | 7 |
| yaqa | YAQA, Tseng et al. — arXiv:2505.22988 | VERIFY | 7 |
