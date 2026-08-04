# SGLang architecture and workflow

Analysis of [wiliao/sglang](https://github.com/wiliao/sglang), based on the actual
codebase structure.

## Repository note

`wiliao/sglang` is a **fork of `sgl-project/sglang`** (the upstream, canonical
SGLang project maintained by LMSYS). It carries the same README, license, and
commit history as upstream, with no custom commits or divergent branches visible
at the time of this analysis — it tracks `main` directly. Everything below
describes the actual upstream SGLang architecture, since that's what this fork's
code is.

---

## 1. Two front ends, one runtime

SGLang ships two distinct ways to talk to it, both sitting on top of the same
runtime:

| Directory | What it is |
|---|---|
| `python/sglang/lang/` | The original **Structured Generation Language** — SGLang's namesake DSL. Programs are written with constructs like `gen()`, `select()`, and `fork()` to express multi-step, branching generation directly in Python. |
| `python/sglang/srt/` | **SGLang Runtime (SRT)** — the backend engine that actually runs local models. This is what backs the OpenAI-compatible HTTP API most deployments use (including the setup in the companion doc for the R9700). |
| `python/sglang/api.py` | The public Python API surface. |
| `python/sglang/launch_server.py` | Entry point for starting a local server (`python -m sglang.launch_server ...`). |

Most production deployments — including everything in
`How-SGLang-works-with-R9700-and-Qwen3.6-27B-Dense-Model.md` — only touch `srt/`
and the OpenAI-compatible API. The `lang/` DSL is a separate, optional layer for
programs that need explicit control flow inside a single generation session.

---

## 2. Inside `srt/` — the runtime engine

The runtime is organized into a handful of directories, each owning one part of
the request lifecycle:

| Directory / file | Role |
|---|---|
| `srt/managers/tp_worker.py` | The **tensor-parallel worker** — this is the component previously described as the "model runner." It owns the loaded model on a given GPU (or GPU shard) and executes the forward pass for each scheduled batch. |
| `srt/model_loader/loader.py` | Model loading logic — reads checkpoint files, constructs tensors, and applies dtype/device placement (the PyTorch-facing half of the loading pipeline described in the companion doc). |
| `srt/models/` | One file per supported model architecture (e.g. `mixtral.py`). This is where architecture-specific logic lives — for a hybrid model like Qwen3.6-27B, the DeltaNet-vs-full-attention layer pattern is implemented here, not in the generic scheduler. |
| `srt/mem_cache/` | The KV/prefix cache system — RadixAttention's tree-based prefix cache lives here, along with `storage/backend_factory.py`, which supports tiered cache storage (in-memory, disk, remote). |
| `srt/configs/model_config.py` | Parses each model's `config.json` into SGLang's internal representation — including where KV cache is or isn't needed per layer (relevant for encoder/decoder splits and, by extension, hybrid attention patterns), and detecting quantization metadata (e.g. FP8/ModelOpt checkpoints). |
| `srt/server_args.py` | All CLI/server flags, plus the **automatic backend-selection logic** — e.g. choosing FlashInfer vs. Triton vs. FA3 based on the detected GPU generation, and setting KV cache dtype defaults per model family. |
| `srt/utils.py` | Shared utilities used across the runtime. |

---

## 3. Top-level `python/sglang/` utilities

Outside `srt/` and `lang/`, a few files round out the package:

- `bench_offline_throughput.py`, `bench_one_batch.py`, `bench_one_batch_server.py`,
  `bench_serving.py` — the benchmarking scripts used to produce SGLang's published
  throughput numbers (offline batch, single static batch, single batch against a
  live server, and full online serving respectively).
- `check_env.py` — validates the environment and dependencies before a run.
- `global_config.py` — global configuration constants.
- `profiler.py` — sends profiling requests against a running server.
- `test/` — the test suite.

---

## 4. How this maps to the request lifecycle

Using the same five-stage lifecycle from the companion doc, here's where each
stage actually lives in the codebase:

```
Client request        → HTTP layer (OpenAI-compatible API surface, python/sglang/api.py)
        ↓
Tokenizer manager      → srt/managers/ (tokenization + request intake)
        ↓
Scheduler              → srt/managers/ (batching) + srt/mem_cache/ (radix/prefix cache)
        ↓
Model runner            → srt/managers/tp_worker.py, executing srt/models/<arch>.py
                           via kernels loaded per srt/server_args.py's backend selection
        ↓
Detokenizer            → srt/managers/ (detokenization + streaming response)
```

The model-loading path (also from the companion doc) maps to:

```
srt/model_loader/loader.py   → reads weights, builds tensors, sets dtype/device
        ↓
PyTorch (ROCm build)          → executes the actual tensor/device operations
        ↓
ROCm/HIP                      → allocates VRAM, performs the memory copy
```

---

## 5. Where a new model's architecture actually lives

For a model like Qwen3.6-27B — dense, with a hybrid Gated DeltaNet + full-attention
layer pattern — the parts of SGLang that need model-specific logic are narrow and
contained:

- **`srt/models/<model>.py`** — defines the actual layer sequence (e.g. the 3:1
  DeltaNet-to-attention repeating block) and how each layer type consumes/produces
  state.
- **`srt/configs/model_config.py`** — determines, per layer, whether KV cache needs
  to be allocated at all (this is what lets 48 of Qwen3.6-27B's 64 layers skip KV
  cache entirely in favor of fixed-size recurrent state).
- **`srt/server_args.py`** — supplies sensible attention-backend and KV-cache-dtype
  defaults once the model's architecture is known.

Everything else — the scheduler, the tokenizer/detokenizer, the tensor-parallel
worker's execution loop, the radix cache — is architecture-agnostic and doesn't
change per model.

---

## Summary

- `wiliao/sglang` is an unmodified fork of `sgl-project/sglang`; this document
  describes the upstream architecture.
- SGLang has two front ends (`lang/` DSL, `srt/` runtime) sharing one backend —
  nearly all serving deployments only use `srt/`.
- Inside `srt/`, responsibilities split cleanly: `managers/` (scheduling +
  execution), `models/` (architecture-specific layer logic), `mem_cache/`
  (prefix/KV caching), `model_loader/` (weight loading), `configs/` +
  `server_args.py` (model-aware defaults).
- Adding or adapting support for a new architecture (like Qwen3.6-27B's hybrid
  attention) is largely contained to `srt/models/` and `srt/configs/model_config.py`
  — the rest of the runtime doesn't need to change.
