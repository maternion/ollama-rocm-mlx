#!/bin/sh
# Build the combined ollama-rocm-mlx Linux release asset:
#   ollama-rocm-mlx-linux-amd64.tar.zst
# containing the Go binary + the ROCm 7.2 llama.cpp runner (rocm_v7_2) +
# the ROCm 10 MLX runner (mlx_rocm_v10) for all supported AMD GPUs.
#
# Usage:
#   scripts/build_rocm_mlx.sh                 # build into ./dist
#   OLLAMA_VERSION=0.0.1 scripts/build_rocm_mlx.sh
#
# This is the asset fetched by scripts/install.sh:
#   _asset="ollama-rocm-mlx-linux-${ARCH}.tar.zst"
#
# It is intended to run in GitHub CI (release.yaml) on a tag push or a
# workflow_dispatch run, not on every PR.

set -eu

PLATFORM="${PLATFORM:-linux/amd64}"
DIST="${DIST:-./dist}"
ARCH="amd64"

rm -rf "$DIST/rocm-mlx-stage"
mkdir -p "$DIST"

# Build the rocm-mlx-archive stage (Go binary + both ROCm runner dirs).
# This stage is defined in Dockerfile and pulls in:
#   - publish-go            (the ollama binary)
#   - llama-server-rocm_v7_2 (GGUF runner, ROCm 7.2, fat arch)
#   - mlx-rocm-v10          (MLX runner, ROCm 10, fat arch)
docker buildx build \
        --output type=local,dest="$DIST/rocm-mlx-stage" \
        --platform="$PLATFORM" \
        --target rocm-mlx-archive \
        -f Dockerfile \
        .

# Pack the combined payload. Layout matches what install.sh expects:
#   bin/ollama
#   lib/ollama/rocm_v7_2/...
#   lib/ollama/mlx_rocm_v10/...
( cd "$DIST/rocm-mlx-stage" && tar c bin lib ) | zstd -9 -T0 \
        > "$DIST/ollama-rocm-mlx-linux-${ARCH}.tar.zst"

rm -rf "$DIST/rocm-mlx-stage"

echo "Wrote $DIST/ollama-rocm-mlx-linux-${ARCH}.tar.zst"
ls -lh "$DIST/ollama-rocm-mlx-linux-${ARCH}.tar.zst"