# How SGLang works with the AMD Radeon AI PRO R9700 and Qwen3.6-27B (Dense)

A practical + architectural reference for running Alibaba's Qwen3.6-27B dense model
through SGLang on a single AMD Radeon AI PRO R9700 (32GB GDDR6, gfx1201, RDNA4).

---

## 1. Hardware and software compatibility

- **GPU**: AMD Radeon AI PRO R9700 — 32GB GDDR6, gfx1201, RDNA4, 64 Compute Units,
  128 2nd-gen AI accelerators, FP8/FP16/INT8 support.
- **ROCm**: 7.2 or newer is required — this is when gfx1201 became a fully supported
  target (not a workaround target) for both Instinct and Radeon device families.
- **SGLang**: officially supports gfx1201 via AMD's prebuilt ROCm Docker image.

Check your stack before doing anything else:

```bash
rocminfo | grep gfx
rocm-smi --showdriverversion
```

You want to see `gfx1201` and a ROCm 7.2+ driver.

---

## 2. Installing SGLang for ROCm

### Option A — prebuilt Docker image (recommended)

```bash
docker pull rocm/sgl-dev:v0.5.13.post1-ubuntu24.04-py3.14-rocm7.14

docker run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --network=host --ipc=host \
  --group-add=video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v ~/models:/root/.cache/huggingface \
  rocm/sgl-dev:v0.5.13.post1-ubuntu24.04-py3.14-rocm7.14 \
  bash
```

### Option B — build from source

```bash
git clone -b v0.5.13 https://github.com/sgl-project/sglang.git
cd sglang/sgl-kernel && python setup_rocm.py install
cd .. && rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_hip]"
```

---

## 3. Picking a weight format that fits 32GB

Qwen3.6-27B is a **dense** 27B-parameter model — every parameter is active on every
token (unlike a Mixture-of-Experts model, where only a slice of parameters fires per
token).

| Format | Approx. weight size | Fits on 32GB? | Notes |
|---|---|---|---|
| BF16 | ~54GB | No | Won't fit on a single R9700 |
| FP8 | ~27–28GB | Tight | gfx1201 is missing from AITER's FP8 arch table as of this writing — FP8 kernels silently dequantize to FP32 instead of running native FP8. You lose the memory *and* speed benefit without a hard error. Check for a fix before relying on this. |
| AWQ / INT4 | ~14–17GB | Comfortably | Recommended starting point — leaves headroom for KV cache and the vision encoder |

**Recommendation:** start with an AWQ or INT4 checkpoint rather than FP8, given the
current AITER gap on gfx1201.

---

## 4. Launching the server

```bash
python -m sglang.launch_server \
  --model-path mattbucci/Qwen3.6-27B-AWQ \
  --quantization awq \
  --reasoning-parser qwen3 \
  --attention-backend triton \
  --max-model-len 32768
```

`mattbucci/Qwen3.6-27B-AWQ` is an AWQ 4-bit quantization of Qwen3.6-27B specifically
calibrated for AMD RDNA4 (gfx1201) inference with SGLang — a good default starting
checkpoint for the R9700.

Flag notes:

- `--model-path` — this is what actually determines the weight format. Point it at
  `Qwen/Qwen3.6-27B` directly and you get the full BF16 repo (~54GB, won't fit — see
  Section 3). The checkpoint repo itself carries the quantization, not a flag alone.
- `--quantization awq` — tells SGLang's model runner how to unpack the AWQ-packed
  weights at load time. Must match the checkpoint (use `gptq` instead if you picked a
  GPTQ repo).
- `--attention-backend triton` — Triton is the mature backend on RDNA4/Radeon;
  FlashInfer support for consumer Radeon cards is still catching up.
- `--reasoning-parser qwen3` — Qwen3.6 runs in thinking mode by default.
- `--max-model-len` — keep this well below the native 262K unless you've confirmed
  your VRAM budget (see Section 6 on why context cost isn't uniform across layers).

### Quick sanity check

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mattbucci/Qwen3.6-27B-AWQ", "messages": [{"role": "user", "content": "test"}]}'
```

Watch `rocm-smi` during the first request. Ballooning VRAM or throughput far below
expectations usually means the FP8-fallback issue is biting — switch to AWQ/INT4.

---

## 5. Request lifecycle

A request moving through SGLang passes through five stages. The first three are
generic SGLang behavior — they don't know or care what model is loaded. The
interesting model-specific work happens in the fourth stage.

```
Client request        → Prompt sent to the API
        ↓
Tokenizer manager      → Encodes text to token IDs
        ↓
Scheduler              → Groups requests into batches, checks the prefix cache
        ↓
Model runner (R9700)   → Runs the model's layers via HIP + Triton kernels
        ↓
Detokenizer            → Streams token IDs back to the client as text
```

---

## 6. What's actually happening on the R9700 (why this model is different)

Qwen3.6-27B uses a **hybrid attention architecture** inherited from the Qwen3-Next /
Qwen3.5 line: 64 layers, arranged as 16 repeating macro-blocks of
**3× Gated DeltaNet (linear attention) → 1× Gated Attention (full attention)**, each
followed by a feed-forward network.

This 3:1 ratio changes what SGLang has to manage in VRAM during generation:

| VRAM region | Behavior |
|---|---|
| **Model weights** | Loaded once at startup, fixed for the process lifetime. Nothing at request time touches this. |
| **KV cache** | Only exists for the 16 full-attention layers (1 in 4). This is the only region that grows with context length, and it's what SGLang's prefix/radix cache and continuous batching normally manage. |
| **DeltaNet recurrent state** | The 48 linear-attention layers don't keep a growing cache at all — each holds a fixed-size recurrent state, updated in place token by token, regardless of conversation length. This is why a 262K-context model doesn't need a 262K-token cache. |

**Why this matters practically:** SGLang's scheduler was originally built around a
straight-line assumption — context length in, KV cache size out. With this hybrid
layout, three-quarters of the layer stack stops caring about context length entirely,
and only the attention-layer cache scales. That's the whole reason a 32GB card can
plausibly hold a very long context for this model when it couldn't for an equivalent
dense transformer using full attention everywhere — the memory math is dominated by
the 25% of layers that actually need a growing cache, not by the raw sequence length.

The ROCm/HIP layer underneath is architecture-agnostic: it just executes whatever
kernel graph the model runner builds each step — a DeltaNet chunked-scan kernel for
three layers, a standard attention kernel for the fourth, repeated 16 times, then the
FFNs. All of the adaptation to Qwen3.6's shape happens in SGLang's scheduler and cache
manager — the GPU itself is simply executing whatever it's handed.

---

## 7. Where SGLang ends and PyTorch/ROCm begin

It's easy to blur these three layers together, but each owns a distinct job.

- **SGLang** — the tokenizer manager, scheduler, **model runner**, and detokenizer are
  all SGLang's own code. The model runner in particular is the piece that knows the
  model's structure: for Qwen3.6-27B, it knows to alternate 3 DeltaNet layers with
  1 full-attention layer per macro-block, and to grow the KV cache only for the
  latter.
- **PyTorch (ROCm build)** — the model runner calls into PyTorch to do the actual
  tensor work: reading weight files (safetensors) from disk, constructing tensor
  objects, choosing dtype/device placement, and executing the forward pass kernel
  by kernel. PyTorch's CUDA API is transparently mapped to HIP on ROCm builds, so
  the same `.to('cuda')`-style calls work unchanged.
- **ROCm/HIP** — the driver-level layer underneath PyTorch. It doesn't know what a
  "layer" or a "model" is — it just receives allocation and memcpy requests and
  executes them against the R9700's actual VRAM and compute units.

### How the weights actually land on the R9700

Loading the model at startup follows the same layered handoff as inference does:

```
SGLang model runner   → decides what to load, in what dtype, into what tensor structure
        ↓
PyTorch               → reads weight files, creates tensors, issues device-transfer calls
        ↓
ROCm / HIP            → allocates VRAM and performs the actual memory copy
        ↓
R9700 VRAM            → weights now resident
```

PyTorch never touches VRAM directly, and ROCm never has any concept of "weights" or
"layers" — only memory addresses and byte counts. Neither layer duplicates the
other's job: SGLang understands the model, PyTorch understands tensors and dtypes,
ROCm understands raw memory and hardware execution.

---

## 8. Known caveats on this hardware

- **AITER FP8 fallback**: gfx1201 is missing from AITER's FP8 arch table, so FP8
  kernels silently fall back to FP32 instead of erroring. Confirm this has been fixed
  upstream before choosing FP8 for either memory or speed reasons.
- **FlashInfer maturity**: use `--attention-backend triton` on Radeon consumer cards;
  FlashInfer support is still maturing there relative to Instinct-class GPUs.
- **Prefill gap vs. NVIDIA**: RDNA4 prompt-processing throughput trails NVIDIA's
  current generation more than decode throughput does — relevant if your workload is
  long-prompt agentic coding rather than short-prompt chat.

---

## Summary

- Hardware: R9700 (gfx1201) is officially supported on ROCm 7.2+ and SGLang's ROCm build.
- Model: use AWQ/INT4 quantization of Qwen3.6-27B, not FP8, until the AITER gfx1201 gap closes.
- Architecture: the model's 3:1 DeltaNet-to-attention ratio is why it fits comfortably in 32GB at long context — most layers use fixed-size state instead of a growing KV cache.

---

## Appendix A — SGLang vs. other inference engines for this setup

### Concurrency & caching model

| Engine | Cache strategy | Concurrency model | Where it wins |
|---|---|---|---|
| **SGLang** | RadixAttention — trie-based prefix cache, aggressively shares KV across requests with a common prefix | True parallel batching | Agentic/RAG workloads, multi-turn chat, repeated system prompts — anything where requests share a prefix |
| **vLLM** | PagedAttention — treats KV cache like OS virtual memory, no fragmentation | Continuous batching | Single-turn/unique-prompt throughput at high concurrency; broadest hardware support (NVIDIA, AMD, TPU, Trainium, Ascend) |
| **llama.cpp** | Simple KV cache, no prefix sharing | Single-stream by default (`-np` for limited parallel slots) | Maximum portability — runs on ROCm, Vulkan, CPU, Metal; the widest GGUF quantization ecosystem |
| **Ollama** | Wraps llama.cpp | Effectively sequential/time-sliced under concurrent load | Easiest single-user setup — `ollama run model`, nothing to configure |

One benchmark on unique, non-overlapping prompts actually had vLLM edge out SGLang
(60 vs 52.7 tok/s) — SGLang's advantage is conditional on shared context, not
universal. On shared-prefix workloads SGLang has shown up to 4.6x the throughput of
vLLM in concurrent tests, but that gap is workload-dependent, not a fixed multiplier.

### AMD R9700 / gfx1201 support maturity

- **SGLang**: officially supported via AMD's prebuilt ROCm image (as set up in this
  doc). Straightforward, tracks upstream.
- **vLLM**: gfx1201 supported as an official ROCm plugin target, actively maintained,
  native Qwen3.6-27B support with `--reasoning-parser qwen3` (mirrors SGLang's flag).
- **llama.cpp**: works on gfx1201 today, but field reports are mixed — one user had a
  stable ROCm Docker setup running Qwen3.6-27B at full 262K context, then hit crashes
  and core dumps after a routine update, and had to fall back toward Vulkan (which
  itself warns it's "not a conformant Vulkan implementation, testing use only").
  Expect more hands-on troubleshooting than with SGLang/vLLM.
- **Ollama**: official ROCm v7 support lists gfx1201, but real-world Windows users
  have found the gfx12xx ROCm libraries missing from the shipped installer and had
  to pull community-built replacements (e.g. the `likelovewant/ollama-for-amd`
  project) to get RDNA4 cards recognized. Linux tends to fare better than Windows
  here.

### Bottom line for this setup

- **Multi-user or agentic serving** → stick with SGLang (what this doc covers) or
  vLLM — both have first-class gfx1201 support and native Qwen3.6-27B integration.
- **Single-user, want the simplest possible thing** → Ollama, if on Linux and willing
  to accept the sequential-concurrency ceiling and rougher AMD-driver edges.
- **Maximum quantization flexibility, don't mind some driver wrangling** →
  llama.cpp/GGUF, but budget time for the AMD ROCm/Vulkan flakiness noted above.
