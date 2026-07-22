# Generated v1.0 to v1.1 API diff

Report identity: `sha256:afe79ffe88b120831301334ffb73e9f82003b01855043768ff38c3bfe32a3c5c`

This is a structural source report for candidate `1.1.0-rc.0`
against `v1.0.0`. It is not a package-install or runtime receipt.

## Frozen Rust tier

The seven frozen crates require a green cargo-semver-checks run:

```sh
./scripts/check-semver.sh v1.0.0
```

## Python root namespace

Retained v1 names: `Model`, `ternary_matmul`.

Added in v1.1:

- `QwenLoadReceipt`
- `QwenModel`
- `autograd`
- `compiled_backends`
- `conv1d_forward`
- `conv1d_vjp`
- `fsq_forward`
- `fsq_vjp`
- `lsq_forward`
- `lsq_vjp`
- `nn`
- `onnx`
- `portable`
- `salt`
- `ste_absmean_scale`
- `ste_quantize_forward`
- `ste_quantize_vjp`
- `torch`

Removed v1 names: none. Generation fails if this changes.

## Boundaries

- C ABI: separate cargo test -p tritium-ffi gate.
- Evolving Rust: not covered by the stable SemVer promise.
- Runtime evidence: not produced by this structural report.
