# tritium-train

Ternary QAT and distillation: autograd tape, STE and differentiable estimators, optimizers.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

## Native scale recovery (CPU reference)

`pv_tuning::NativeScaleRefinementSession` runs a `ScaleOnly`
`NativeRefinementPlan` over every explicitly selected owner, in plan order.
Supply the admitted parent `AdditiveTensor`s and unweighted gradients of their
**actual stored scales**, not dense logical-weight gradients. The session applies
the declared window weight, the shared AdamW implementation, nonnegative
projection, and rounding to each tensor's declared scale precision. It preserves
ternary codes, compact allocation, law, codec, basis and transport; no f16 bridge
or dense/code master is introduced. Every update commits all owners atomically.

`checkpoint_bytes()` emits the distinct TNSR v1 state format; `resume()` requires
the same frozen plan and exact original parent tensors. Frozen codes and metadata
come from that parent, while the checkpoint retains exact scale bits, moments and
the completed-window cursor. This is not TPV1/TPVB1 and is not an inference file.
Logical size limits exclude caller gradients, graphs/activations, allocator slack
and process RSS. A valid checkpoint is not proof of gradient/objective execution.

This API does not admit package custody, compute scale/owner adjoints, implement
native alternating-PV, emit publication evidence or create a refined child. Public
native refinement stays unavailable until those integrations and replay exist.
