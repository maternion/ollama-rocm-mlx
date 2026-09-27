# AGENTS.md

## Building

For a full build from the repository root:

```sh
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

On Linux hosts with a ROCm toolchain (/opt/rocm) and sibling checkouts
`../mlx-rocm` + `../mlx-c`, that bare configure already builds the complete
dual-engine payload: the llama.cpp ROCm runner (`build/lib/ollama/rocm_v7_2`)
plus the MLX ROCm backend (`build/lib/ollama/mlx_rocm_v10`), targeting the
detected discrete GPU arch (rocm_agent_enumerator, iGFX archs are filtered
out when a dGPU is present). The resulting `./ollama serve` loads both GGUF
models (llama.cpp) and `*-mlx` MLX models from one server. Override with
`-DOLLAMA_LLAMA_BACKENDS=... -DOLLAMA_MLX_BACKENDS=... -DAMDGPU_TARGETS=...`
or `-DFETCHCONTENT_SOURCE_DIR_MLX=...` as needed; the old
`OLLAMA_LLM_LIBRARY`/`OLLAMA_LIBRARY_PATH` env vars are only needed when
running a payload from a non-standard directory.

For quick Go-only iteration against an existing native payload:

```sh
go build .
go run . serve
```

See `docs/development.md` for prerequisites, platform notes, GPU backends, and
the full development workflow.
