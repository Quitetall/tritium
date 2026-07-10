# tritium-serve container (P3d packaging).
#
# Two-stage: build with the CUDA devel image (nvcc compiles the PTX at build
# time), run on the slim runtime image. The model is a bind mount — GGUFs are
# GBs and version independently of the server.
#
#   docker build -t tritium-serve .
#   docker run --gpus all -p 127.0.0.1:8080:8080 -e TRITIUM_AUTH_TOKEN=change-me \
#     -v ~/.cache/tritium-models:/models:ro \
#     tritium-serve --model /models/<repo>/<file>.gguf --backend cuda
#
# CPU-only image (no GPU, no nvcc needed at runtime; build still uses the
# devel base for simplicity):
#   docker build --build-arg FEATURES=serve -t tritium-serve:cpu .
#   docker run -p 127.0.0.1:8080:8080 -e TRITIUM_AUTH_TOKEN=change-me \
#     -v ~/.cache/tritium-models:/models:ro \
#     tritium-serve:cpu --model /models/<...>.gguf --backend cpu
#
# Exposure note (threat model): binding 0.0.0.0 inside the container is
# loopback-equivalent only while the port mapping is 127.0.0.1. If you
# publish beyond localhost, set TRITIUM_AUTH_TOKEN — the server refuses
# non-loopback binds without it, and the container entrypoint binds 0.0.0.0.

ARG CUDA_TAG=13.0.1-devel-ubuntu24.04

FROM docker.io/nvidia/cuda:${CUDA_TAG} AS build
ARG FEATURES=cuda
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:$PATH
WORKDIR /src
COPY . .
# --locked: the committed lockfile is the supply-chain pin.
RUN cargo build --release --locked -p tritium-serve --features ${FEATURES} \
        --bin tritium-serve \
    && cargo build --release --locked -p tritium-cli --bin tritium

# Plain ubuntu runtime: the CUDA build dlopens the DRIVER api at runtime
# (cudarc fallback-dynamic-loading — no libcuda link dependency), and
# `--gpus all` injects libcuda via the NVIDIA container toolkit. PTX is
# compiled into the binary at build time, so no toolkit libs are needed.
FROM docker.io/library/ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home tritium
COPY --from=build /src/target/release/tritium-serve /usr/local/bin/
COPY --from=build /src/target/release/tritium /usr/local/bin/
VOLUME /models
USER tritium
EXPOSE 8080
# 0.0.0.0 inside the container: docker's -p mapping decides real exposure.
# TRITIUM_AUTH_TOKEN must be set (the server enforces this for non-loopback).
ENTRYPOINT ["tritium-serve", "--host", "0.0.0.0", "--port", "8080"]
CMD ["--help"]
