# ollama-rocm-mlx

Ollama fork: **GGUF + MLX models on AMD ROCm, one `ollama serve`.**

Tested on RX 9060 XT (gfx1200, RDNA4), ROCm 10.

## Features

- **Two engines, one server** — GGUF via llama.cpp ROCm (`rocm_v7_2`),
  MLX models (`*-mlx`) via the MLX ROCm backend (`mlx_rocm_v10`).
- **hipRTC custom kernels** — routes `fast::cuda_kernel` to the ROCm backend,
  so MLX's fused kernels (GatedDeltaNet, depthwise conv, mamba2) run on ROCm.
- **Merged MLX fork** — [ml-explore/mlx#2300](https://github.com/ml-explore/mlx/pull/2300)
  forward-merged to the MLX commit ollama pins (`maternion/mlx`, branch `rocm-current`).
- **Perf** — fp8 GEMV with 16-row weight reuse (~5x prefill at M>1, ~150 GB/s
  decode); `Unembed` dtype fix avoiding per-token lm_head promotion.
- **Turnkey build/install** — bare `cmake -B build` builds both engines;
  `install.sh` takes `OLLAMA_BACKEND=rocm,mlx`.

## Build

```sh
# siblings: ../mlx-rocm (maternion/mlx @ rocm-current), ../mlx-c
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

ROCm auto-detected; GPU arch from `rocm_agent_enumerator`. Override with
`-DOLLAMA_LLAMA_BACKENDS=`, `-DOLLAMA_MLX_BACKENDS=`, `-DAMDGPU_TARGETS=`.

## Install

```sh
curl -fsSL https://github.com/maternion/ollama-rocm-mlx/raw/main/scripts/install.sh | OLLAMA_BACKEND=rocm,mlx sh
```

Installs to `/usr/local/{bin,lib/ollama}` with both backends. Run from a
checkout to use the local `build/` payload; piped (as above) it clones
`main` (plus the `mlx-rocm`/`mlx-c` siblings) and builds. Set
`OLLAMA_PAYLOAD_URL=<tarball>` to install a prebuilt payload instead.

## Results

| Model | Engine | Decode |
|---|---|---|
| qwen3.5:0.8b-mlx (fp8) | MLX | ~104 tok/s |
| qwen3.5:0.8b (GGUF) | llama.cpp | ~105 tok/s |

Both loadable at once; `ollama ps` shows 100% GPU each.
GatedDelta kernels bit-exact vs graph reference (`go test ./mlx/`).

MIT. Upstream: [ollama/ollama](https://github.com/ollama/ollama).
