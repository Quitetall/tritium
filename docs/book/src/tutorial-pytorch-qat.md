# Installed-wheel PyTorch QAT

This tutorial starts with ordinary floating PyTorch modules, preserves a tied
embedding/output weight, runs one hard-forward STE training step, freezes two
additive ternary planes, checkpoints and resumes model plus AdamW state, exports
a QAT-hard artifact, strict-reloads it, and checks bit-exact output parity. The
runner ships inside `tritium-torch`; no Tritium checkout or compiler is used
after wheel installation.

## Install one exact candidate

Create a fresh environment. Replace `TRITIUM_WHEEL` with one exact wheel built
for the target. Do not point it at a directory containing multiple candidates.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --disable-pip-version-check --only-binary=:all: \
  torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --disable-pip-version-check --only-binary=:all: \
  safetensors==0.8.0
TRITIUM_WHEEL=./dist/tritium_torch-1.1.0rc0-cp39-abi3-PLATFORM.whl
python -m pip install --isolated --disable-pip-version-check \
  --no-index --no-deps --only-binary=:all: "$TRITIUM_WHEEL"
```

`PLATFORM` is a placeholder, not a valid wheel tag. Use the exact generated
filename. CUDA qualification uses the pinned CUDA wheel and matching PyTorch
index instead of the CPU command above.

## Run the complete lifecycle

The output directory must not exist. `-I` disables user-site and environment
path injection, ensuring the module resolves from the active environment.

```sh
python -I -m tritium.torch.tutorial_qat \
  --output-dir ./tutorial-output \
  --device cpu
python -I -m tritium.torch.tutorial_qat \
  --check-receipt ./tutorial-output/receipt.json \
  --device cpu
```

Successful output contains:

- a finite nonzero gradient norm and one deduplicated converted parameter;
- safetensors latent model state, optimizer state, and one resumed step matching
  uninterrupted training exactly, with physical bytes and SHA-256 identities;
- aliases `embed.weight` and `head.weight`, proving tie preservation;
- `tritium.additive-2/tritium.salt-ste@1`, proving two-plane estimator identity;
- content identities for the QAT-hard artifact and hard state;
- a content identity covering every tutorial-result field;
- installed `tritium-torch` version and an owned package path, rejecting a
  source-tree/package shadow;
- `qat-hard/model.safetensors` plus a strict manifest;
- exact parity across latent evaluation, hard conversion, export, and reload.

`receipt.json` uses schema `tritium.installed-qat-tutorial.v2`; v2 adds
checkpoint/resume identities to the original lifecycle result.

For CUDA, use an admitted CUDA wheel and matching PyTorch build, then pass
`--device cuda:0`. Unsupported or unavailable CUDA fails instead of falling
back to CPU.

## What this proves

The CPU and CUDA wheel workflows execute this installed module after installing
the exact candidate without dependency resolution. A separate `python:3.13-slim`
job downloads only the wheel, rejects a checkout or compiler, then executes and
validates the same lifecycle. Their retained output is developer/CI evidence
until bound into the release registry with candidate source, wheel digest, run,
machine, and compatibility-cell identities.

This tiny workflow proves API, autograd, optimizer resume, tie, conversion,
physical serialization, and strict-reload behavior. It does not prove flagship
quality, performance, compression ratio, browser support, or public-release
readiness.
