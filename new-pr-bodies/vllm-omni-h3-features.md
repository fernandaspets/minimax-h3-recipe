A series that adds SM120/SM121 attention backends and opt-in quantisation paths for MiniMax-H3.
Everything new is **opt-in and off by default**; a stock run is unaffected.

This replaces #8487, which was opened before the end-to-end test was complete and then closed. The
branch has since been read end to end (every added file, not a pattern scan), the lane-local
annotations were removed, and the defects that read turned up were fixed — see below.

## What lands

| area | change | default |
|---|---|---|
| attention backends | B12X (SM120/SM121) registered as an attention backend | not selected |
| | SOL_ATTN registered alongside the others | not selected |
| sparse attention | b12x provider for H3's VSA per-head block lists, behind `is_block_sparse` | off |
| transport | permute folded into the 4D all-to-all path | inert unless requested |
| | quantised Ulysses a2a and TP all-reduce (`vllm_omni/diffusion/h3/`) | off; `bf16` default |
| quantisation | opt-in MXFP8 FFN up-projection, per-role NVFP4/MXFP8 policy | off |
| loading | pre-quantised checkpoint path | off |
| model | MiniMax-H3 model, lora, pipeline and encoder changes | — |
| misc | `sage_attn` correctness fix, video serving request path, lint | — |

## Fixed while reviewing the branch

- `h3/a2a_wire.py` documented an intermediate-buffer pool default that the code forbids — the pool is
  opt-in precisely because the pooled variant deterministically corrupts the payload. The prose now
  matches the code.
- `h3/quant_policy.py`'s `role_for()` accepted a role name and silently resolved it to the
  unquantised default, the exact failure the module exists to prevent.
- The transport control file was opened on every exchange (200 per step); it is now re-read at most
  once a second, matching the quantisation policy.
- `h3/comm/comm_quant.py` carried an unused merged-QKV transport that nothing called; removed.
- `minimax_h3_transformer.py` passed `partition="ref2va"` to the turbo-adapter loader
  unconditionally, so an fl2v adapter served on the FL2VA partition was rejected by the loader's own
  guard and the worker died — t2va and fl2va could not boot. The served partition is now passed
  through, and the reference DiT stays explicitly Ref2VA so a real mismatch still fails.

## Testing

Built from public pins by a separate, shareable recipe on 4x RTX PRO 6000 (SM120), TP2 x USP2, hybrid
quantisation (MLP NVFP4, attention MXFP8), int8 a2a + all-reduce wire, 1344x768, 5 s:

| check | result |
|---|---|
| build gates | import (SM120 falls back to Triton), wire hook present in `comm.py`, runtime gate: 339 base packages checked, 0 changed, 0 removed, 1 allowed |
| unit tests | 49 pass — 42 CPU (policy, wire gating) + 7 CUDA (real Triton transport round trip) |
| ref2va, 4 steps | decodable h264 1344x768 clip, 9,778,890 B |
| t2va, 8 steps | decodable h264 1344x768 clip, 7,316,625 B |
| fl2va, 8 steps | decodable h264 1344x768 clip, 8,206,862 B |

`ruff check`, `ruff format --check` (ruff 0.14.10) and `py_compile` are clean on every changed file.

No performance claims are made in this PR. The paths were validated by serving MiniMax-H3; the timings
live with the deployment and each is tied to a recorded run.

## Scope

No default behaviour changes. The sparse and quantised paths require an explicit opt-in (an
environment variable or a config flag), so a caller that does nothing different gets the same path as
before.

---

## AI disclosure

Authored in a `pi` agentic coding session driven by `deepseek-v4.1-flash`. A human maintainer set
the scope, reviewed the result, and submitted it.

---

## Sources

| source | licence | how it is used |
|---|---|---|
| [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) | Apache-2.0 | the target project; the new backends implement its `AttentionBackend` interface |
| [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) | Apache-2.0 | SM120/SM121 CuTe DSL kernels (`b12x.attention.varlen`) that the B12X backend runs |
| [NVlabs/Sana](https://github.com/NVlabs/Sana) (`sol-engine`) | Apache-2.0 | the SOL_ATTN Triton sparse-attention forward the `sol_attn` backend wraps, and the `h3_runtime/comm_quant.py` packet format the quantised Ulysses exchange follows |
| FlashAttention (CuTe DSL attention) | BSD-3-Clause | vendored under `sol_attn/_vendor/flash_attn/cute/` |
| NVIDIA cuDNN Frontend block-sparse-attention reference (`74785165de2da954a2c879a5e3e6f95411c2292d`) | Apache-2.0 | SM120 warp-MMA/TMA skeleton and online-softmax helpers adapted into the vendored `sol_attn` |
| NVIDIA CUTLASS / CuTe DSL | Apache-2.0 | the kernel DSL beneath b12x |
| [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | Apache-2.0 | the turbo LoRA arms the quantisation paths are exercised with |
| [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | model licence | the model the paths were validated against |

## Dependencies not included

- the `sol_attn` package (NVlabs/Sana `sol-engine`, Apache-2.0) is imported behind a `try/except`:
  without it the `SOL_ATTN` backend is registered but inert.
- the `b12x` package is optional in the same way (SM120/SM121 hosts only).
