# How to Deploy Local Qwen 3.6 27B on Two Intel Arc Pro B70 GPUs

**OS: Linux (recommended).** This guide targets Ubuntu 24.04 LTS with the SYCL/oneAPI toolchain, which is where Intel's XPU/GPU compute stack is best supported today. A Windows path exists (Intel ships oneAPI for Windows and llama.cpp's SYCL backend builds there too), but driver maturity, container tooling (Intel's LLM-Scaler images), and community documentation are all far more mature on Linux. If you're on Windows, see the [Windows notes](#windows-notes) at the end — the llama.cpp SYCL build steps are broadly similar, but expect more manual troubleshooting.

---

## Overview

| Item | Detail |
|---|---|
| Model | Qwen3.6-27B (dense, hybrid Gated DeltaNet + Gated Attention, 262K native context) |
| Hardware | 2× Intel Arc Pro B70 (32 GB GDDR6, 256-bit bus each — 64 GB combined VRAM) |
| Engine | llama.cpp (SYCL backend) — most reliable engine for this model/hardware combo today |
| Quant | Q8_K_XL (Unsloth dynamic quant, ~35.8 GB) — near-lossless quality with headroom to spare |
| Context | Up to 128K+ tokens (cheap on this architecture — see [Why context is cheap here](#why-context-is-cheap-here)) |

### Why llama.cpp instead of vLLM or SGLang?

Qwen3.6-27B uses **Gated DeltaNet (GDN) hybrid attention**. As of this writing:
- vLLM's Intel XPU backend has been reported to crash on GDN-based models (missing Triton/CUDA kernels), and only serves FP16 (not BF16).
- SGLang has an XPU backend, but it's newer and only validated against the Arc B580, not B70-specific, and still source-build-only.
- **llama.cpp's SYCL backend has run GDN-attention Qwen models successfully** where vLLM XPU failed, and has merged MTP (multi-token prediction) speculative decoding support for Qwen3.6 specifically.

If you want to try vLLM anyway, Intel's `intel/llm-scaler-vllm` Docker image is the officially B70-validated route (added April 2026) — but budget time for possible GDN-related compatibility issues.

---

## Prerequisites (Linux)

- Ubuntu 24.04 LTS (or similar recent kernel — 6.17 HWE recommended for Battlemage support)
- 2× Intel Arc Pro B70 installed, each with its own PCIe 4.0/5.0 x16 slot
- 1000W+ PSU recommended for a dual-B70 system
- At least 64 GB system RAM (model conversion/loading can be RAM-hungry; more is safer)
- Intel GPU driver ≥ 31.0.101

### 1. Install the Intel GPU driver

```bash
# Add Intel's graphics driver repo (adjust for your Ubuntu release per Intel's docs)
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt update
sudo apt install -y intel-opencl-icd intel-level-zero-gpu level-zero \
    intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2

# Verify both GPUs are visible
sudo apt install -y clinfo
clinfo | grep "Device Name"
```

You should see both Arc Pro B70 devices listed.

### 2. Install Intel oneAPI Base Toolkit (for SYCL compiler)

```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update
sudo apt install -y intel-basekit
```

---

## Step 1: Build llama.cpp with the SYCL backend

```bash
source /opt/intel/oneapi/setvars.sh

git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

cmake -B build -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL_F16=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j"$(nproc)"
```

Verify both B70s are detected by the SYCL runtime:

```bash
./build/bin/llama-ls-sycl-device
```

You should see two devices listed (Level Zero / SYCL backend).

---

## Step 2: Download the model

Recommended quant: **Q8_K_XL** (~35.8 GB) — near-lossless quality, fits comfortably across two 32 GB cards with room for long context.

```bash
pip install -U "huggingface_hub[cli]"

huggingface-cli download unsloth/Qwen3.6-27B-MTP-GGUF \
  Qwen3.6-27B-UD-Q8_K_XL.gguf \
  mmproj-F16.gguf \
  --local-dir ./models/qwen3.6-27b
```

| Quant | Size | Notes |
|---|---|---|
| Q4_K_M | ~16.8 GB | fits on a single B70 if needed |
| Q5_K_M | ~19.5 GB | |
| Q6_K | ~22.5 GB | near-lossless |
| **Q8_K_XL** | **~35.8 GB** | **recommended for dual-B70** |
| BF16 (full) | ~55.6 GB | reference only, not recommended to run directly |

---

## Step 3: Launch the server across both B70s

```bash
cd llama.cpp
source /opt/intel/oneapi/setvars.sh

./build/bin/llama-server \
  -m ./models/qwen3.6-27b/Qwen3.6-27B-UD-Q8_K_XL.gguf \
  --mmproj ./models/qwen3.6-27b/mmproj-F16.gguf \
  --split-mode layer \
  --tensor-split 50,50 \
  -ngl 999 \
  -c 131072 \
  -fa on \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --host 0.0.0.0 --port 8080
```

**Flag notes:**
- `--split-mode layer --tensor-split 50,50` — splits transformer layers evenly across both SYCL devices (the two B70s).
- `-ngl 999` — offload all layers to GPU.
- `-c 131072` — 128K context; cheap on this architecture (see below), leaves plenty of VRAM headroom at Q8.
- `-fa on` — flash attention, reduces KV cache memory further.
- `--spec-type draft-mtp --spec-draft-n-max 2` — enables MTP speculative decoding, ~1.4–2x faster generation on this model with no accuracy loss.

Once running, the server exposes an OpenAI-compatible API at `http://localhost:8080/v1`.

---

## Why context is cheap here

Qwen3.6-27B is a hybrid model: 64 layers arranged as 16 blocks, each with 3 Gated DeltaNet (linear attention, O(n)) sublayers followed by 1 Gated Attention (full, quadratic) sublayer. Only 1 in 4 layers actually grows a KV cache with context length — the rest carry a small fixed-size recurrent state. In practice this means jumping from a 4K to 65K context window has been observed to cost under 1 GB of extra VRAM on comparable Qwen hybrid models, nothing like the multi-GB-per-doubling growth of a standard dense transformer. This is why Q8 quality + 128K context fits comfortably in 64 GB combined VRAM.

---

## Known gotchas

- **Ollama does not currently support these GGUFs** — the separate `mmproj` vision file breaks Ollama's loader. Use `llama-server` directly (as above) or Unsloth Studio.
- **Driver/toolkit version pinning matters.** This is a very new model + new hardware combo; if outputs look garbled, check for a driver or oneAPI toolkit update before assuming a config error.
- **Test at a short context first.** Before committing to a long overnight batch job or agent run, validate the server with a small `-c` value and a short prompt to confirm the SYCL build and tensor-split are working correctly.
- **vLLM alternative:** if you want to try Intel's officially validated `intel/llm-scaler-vllm` Docker image instead, be aware it currently serves FP16 only (not BF16) and has had reported issues with GDN-attention models like the Qwen 3.5/3.6 family.

---

## Windows Notes

Windows is supported by both the Intel GPU driver and llama.cpp's SYCL backend, but is less battle-tested for this specific model/hardware pairing:

1. Install the latest Intel Arc graphics driver (Pro or consumer branch) from Intel's website.
2. Install **Intel oneAPI Base Toolkit for Windows**.
3. Build llama.cpp using the SYCL backend from an "Intel oneAPI command prompt" (which sets up the compiler environment), using MSVC + icx, following the same `-DGGML_SYCL=ON` CMake flags as above.
4. Model download and launch steps are otherwise identical — just adjust paths to Windows conventions (e.g. `.\models\qwen3.6-27b\...`, `.\build\bin\llama-server.exe`).

If you hit build issues on Windows, the Linux path above is currently the better-documented and more predictable route for a dual-B70 deployment.

---

## Summary

| Component | Choice |
|---|---|
| OS | Linux (Ubuntu 24.04 LTS) |
| Engine | llama.cpp, SYCL backend |
| Quant | Q8_K_XL (~35.8 GB) |
| Context | 128K tokens |
| Speedup | MTP speculative decoding enabled |
| Split | Layer-split, 50/50 across two B70s |
