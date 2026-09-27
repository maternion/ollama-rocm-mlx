# ollama-rocm-mlx

Ollama fork that runs **both** GGUF and MLX models on AMD GPUs (ROCm) from a
single `ollama serve`, with a merged, current MLX ROCm backend.

Developed and validated on **Radeon RX 9060 XT (gfx1200, RDNA4)** with ROCm 10.

## What this fork adds

- **Dual engines, one server.** The same `ollama` binary serves:
  - **GGUF** models through the llama.cpp ROCm runner (`build/lib/ollama/rocm_v7_2`)
  - **MLX** models (`*-mlx`) through the MLX ROCm backend (`build/lib/ollama/mlx_rocm_v10`)
  Both are discovered and scheduled by the normal ollama engine, so `ollama run`,
  `ollama ps`, the HTTP API, and the scheduler all work unchanged.
- **Merged MLX ROCm backend.** The experimental ROCm backend
  ([ml-explore/mlx#2300](https://github.com/ml-explore/mlx/pull/2300), fork
  `NripeshN/mlx`) is merged forward to the exact MLX commit ollama pins in
  `MLX_VERSION`, with local fixes carried on branch `rocm-current` in
  [`maternion/mlx`](https://github.com/maternion/mlx).
- **hipRTC custom kernels on ROCm.** Ollama's MLX runner has Metal/CUDA custom
  kernels only; this fork routes `fast::cuda_kernel` to the ROCm hipRTC
  implementation so the CUDA-style kernel bodies (fused GatedDeltaNet recurrence,
  depthwise conv, mamba2 scan) compile and run on ROCm.
- **Performance work on gfx1200:**
  - fp8 (mxfp8) GEMV kernel with 16-row weight reuse and vectorized activation
    loads — ~5x prefill kernel speedup at M>1, ~150 GB/s effective at decode.
  - `Unembed` casts the f32 hidden state back to the lm_head weight dtype,
    avoiding a per-token bf16→f32 promotion of the whole vocab matrix.
  - MLX `load` staging/allocator fixes for RDNA4 (no-XNACK, 256 MB BAR).
- **Turnkey CMake.** On Linux with `/opt/rocm` and sibling `../mlx-rocm` +
  `../mlx-c` checkouts, a bare `cmake -B build` builds the whole dual-engine
  payload and auto-detects the discrete GPU arch.
- **Installer backend selection.** `scripts/install.sh` accepts
  `OLLAMA_BACKEND=rocm,mlx` to install a locally built payload (binary + both
  backend library sets) into `/usr`.

## Build

```sh
# sibling checkouts required:
#   ../mlx-rocm  (maternion/mlx, branch rocm-current)
#   ../mlx-c      (ml-explore/mlx-c)
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

The defaults detect ROCm and enable `OLLAMA_LLAMA_BACKENDS=rocm_v7_2` and
`OLLAMA_MLX_BACKENDS=rocm_v10` for the detected GPU arch. Override any of
`OLLAMA_LLAMA_BACKENDS`, `OLLAMA_MLX_BACKENDS`, `AMDGPU_TARGETS`, or
`FETCHCONTENT_SOURCE_DIR_MLX{,-C}` on the command line.

## Install to /usr

```sh
OLLAMA_BACKEND=rocm,mlx ./scripts/install.sh
```

Installs the binary to `/usr/local/bin/ollama` and the runtime to
`/usr/local/lib/ollama`, including `rocm_v7_2/` and `mlx_rocm_v10/`.

## Verified

- qwen3.5:0.8b-mlx (MLX, fp8): **~104 tok/s decode** on the 9060 XT; coherent,
  deterministic output; GatedDelta custom kernels bit-exact vs the graph
  reference (`go test ./mlx/`).
- qwen3.5:0.8b (GGUF, llama.cpp ROCm): ~105 tok/s decode on the same GPU.
- Both model types loaded simultaneously by one server, `ollama ps` reports
  100% GPU for each (MLX shows the dynamic 262144 context).

## Upstream

Based on [ollama/ollama](https://github.com/ollama/ollama) with the MLX fork
merged as described above. License: MIT (unchanged).
