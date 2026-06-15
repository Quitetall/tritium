# ADR 0008 — v0.60 Pretraining + Distributed

- **Status:** Planned
- **Date:** 2026-06-15
- **Relates:** executes the 0.60 milestone of [ADR 0002](./0002-release-roadmap.md); builds on [ADR 0007](./0007-v050-training-core.md); precedes [ADR 0009](./0009-v070-backend-breadth.md)

## Status

Planned — not started. No code, crate, or test for this milestone exists yet.

**Must land first:** v0.50 (Training core, [ADR 0007](./0007-v050-training-core.md))
must be tagged and green — STE autograd, QAT, backward kernels, optimizer, and
single-node LoRA are the substrate this milestone scales out. Distributed correctness
is meaningless until the single-process gradient and optimizer paths are proven.

**Hard blocker:** a **≥2-GPU cluster** (and a multi-node interconnect for the
multi-node path). The load-bearing gates — N-GPU vs 1-GPU loss parity, checkpoint
resharding across J≠K GPUs, scaling efficiency — cannot be validated on a single
device. If no ≥2-GPU CI lane is available, the gate runs as a **documented manual
gate** on rented/borrowed multi-GPU hardware before tagging.

## Scope

Adds from-scratch **pretraining** and **distributed training** on top of the v0.50
single-node trainer: a data pipeline (deterministic sharded shuffle, resumable
mid-epoch), **FSDP/DDP** gradient/parameter sharding, distributed **checkpointing**
with resharding across GPU counts, and **multi-node** orchestration. Touches
`tritium-train` (distributed strategies, checkpoint format, data loader) plus a tiny
from-scratch model target used for the pretrain smoke test.

## Testability (exit gates)

| Gate | Tag | How tested | CI lane |
|------|-----|------------|---------|
| N-GPU (FSDP/DDP) loss curve matches 1-GPU within tolerance for same global batch+seed | C/P | single-vs-multi-GPU | multi-GPU |
| All-reduced gradients equal a single-process summed reference | C | vs-reference | multi-GPU |
| Checkpoint resharding: save on K GPUs, restore on J≠K ⇒ identical forward; resume continues loss curve | C/E | single-vs-multi-GPU | multi-GPU |
| Killing a rank mid-run ⇒ clean error or recovery, never a corrupt checkpoint | F/M | fault-injection / contract test | multi-GPU |
| Data pipeline: deterministic per-seed shuffle; no sample dup/loss across shards; resumable mid-epoch | C/E | golden / coverage test | cpu-only |
| Near-linear throughput scaling to target GPU count: ≥80% scaling efficiency | Pe | bench | multi-GPU |
| From-scratch tiny model reaches target loss in fixed steps (pretrain smoke) | C | vs-reference | multi-GPU |

## Definition of done — tag v0.60.0

- [ ] Multi-GPU (FSDP/DDP) loss curve matches single-GPU within tolerance at the same global batch + seed.
- [ ] All-reduced gradients equal a single-process summed reference.
- [ ] Checkpoint resharding correct: save on K GPUs, restore on J≠K GPUs ⇒ identical forward; resume continues the loss curve.
- [ ] Killing a rank mid-run yields a clean error or recovery — never a corrupt checkpoint.
- [ ] Data pipeline: deterministic per-seed shuffle; no sample duplication or loss across shards; resumable mid-epoch.
- [ ] ≥80% scaling efficiency (per-GPU throughput vs single-GPU) at the target GPU count.
- [ ] From-scratch tiny model reaches the target loss in the fixed step budget (pretrain smoke).
- [ ] U1–U9 green on CPU + multi-GPU; tag `v0.60`.
