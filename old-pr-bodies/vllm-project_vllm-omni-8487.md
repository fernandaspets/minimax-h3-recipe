A series that adds SM120/SM121 attention backends and opt-in quantisation paths for
MiniMax-H3. Everything new is **opt-in and off by default**; a stock run is unaffected.

## Commits

| commit | what | default |
|---|---|---|
| `diffusion attention: add the B12X backend (SM120/SM121)` | registers a B12X attention backend | not selected by default |
| `attention: b12x provider for H3 VSA per-head block lists (opt-in)` | block-list sparse attention for H3's VSA, behind `is_block_sparse` | off |
| `models/minimax_h3: opt-in b12x MXFP8 FFN up-projection` | MXFP8 FFN up-projection path | off |
| `distributed: all-to-all permute in the 4D transport` | permute folded into the 4D all-to-all path | inert unless a caller requests it |
| `attention: add the sol_attn backend` | SOL_ATTN backend registered alongside the others | not selected by default |
| `minimax_h3: model, lora, pipeline and encoder changes` | model/encoder/pipeline support for the above | — |
| `models/minimax_h3: checkpoint pre-quantisation path` | loader support for pre-quantised checkpoints | off |
| `attention: sage_attn fix` | small correctness fix | — |
| `entrypoints: video serving path changes` | request-path plumbing | — |
| `lint: satisfy the repo ruff config` | formatting/lint only, no behaviour change | — |

## Notes for review

- The B12X backend and the VSA provider are deliberately separate commits so the backend
  registration can be reviewed independently of the sparse walk.
- The `sol_attn` backend and the MiniMax-H3 model changes are stacked: the model publishes
  the metadata the backend consumes.
- `lint:` is formatting plus two banned-API replacements
  (`torch.cuda.synchronize`/`current_device` -> `torch.accelerator.*`) and import placement
  in the pre-quantisation loader. It can be squashed on request.

## Testing

`ruff check` and `ruff format --check` are clean at ruff 0.14.10 (the version pinned in
`.pre-commit-config.yaml`). The paths were validated by serving MiniMax-H3 `ref2va` on
4x RTX PRO 6000 Blackwell (SM120) with the backends selected. No performance claims are
made in this PR; the numbers live with the deployment, and each is tied to a recorded run.

## Scope

No default behaviour changes. The sparse and quantised paths require an explicit opt-in
(an environment variable or a config flag), so a caller that does nothing different gets the
same path as before.

---

## AI disclosure

This change was authored in an agentic coding session in **pi** (the pi coding-agent harness)
driving **DeepSeek-V4.1-Flash** (`deepseek-v4.1-flash`, provider `deepseek`) on 2026-10-04.
A human maintainer set the direction, scope and the decision to submit; the agent wrote the
code, ran the gates and wrote these notes.

Skills: **`humanize`** (plan -> implement -> independent review) was consulted and its
planning and acceptance-criteria discipline applied. The independent-review step was **not**
run with a separate agent, so this should be treated as single-author work that had a plan,
not as RLCR-reviewed work. **`kda`** (the kernel-design loop) was **not** used.

Gates actually run before opening this PR: `ruff check` and `ruff format --check` at
**ruff 0.14.10**, the version pinned in `.pre-commit-config.yaml` (not the newer ruff present
on the machine), plus `py_compile` on every changed file. Where something was not measured it
is stated as not measured; no claim here rests on unverified model output.

---

## Sources and inspiration

This integration stands on other people's work, so the list is explicit:

| source | licence | how it is used |
|---|---|---|
| [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) | Apache-2.0 | the target project; backends follow its `AttentionBackend` interface |
| [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) | Apache-2.0 | the SM120/SM121 CuTe DSL kernels (`b12x.attention.varlen`) the B12X backend runs |
| [NVlabs/Sana](https://github.com/NVlabs/Sana) `sol-engine` | Apache-2.0 | the released `sol-attn` Triton backend; its reference integration `Sol-H3/h3_runtime/sparse_attention.py` is cited in the backend docstring |
| FlashAttention (CuTe DSL attention) | BSD-3-Clause | the vendored `sol_attn` package vendors `sol_attn/_vendor/flash_attn/cute/` |
| NVIDIA cuDNN Frontend, block-sparse-attention reference (commit `74785165de2da954a2c879a5e3e6f95411c2292d`) | Apache-2.0 | the SM120 warp-MMA/TMA skeleton and online-softmax helpers adapted into the vendored `sol_attn` |
| NVIDIA CUTLASS / CuTe DSL (`nvidia-cutlass-dsl`) | Apache-2.0 | the kernel DSL beneath b12x |
| NVIDIA PyTorch NGC container (`nvcr.io/nvidia/pytorch@sha256:33ef5fc15e8937602d64022209cdb2777b32dadf742f41332023d946041b3c14`) | NVIDIA container terms | runtime foundation: CUDA 13.4, PyTorch 2.14 |
| Local Inference Lab multi-model CUDA 13.4 runtime | Apache-2.0 | the image build (`jovian_wheel_runtime`, `RUNTIME_SOURCE_COMMIT=6c0e9843bb962f483409b0296b225cca03fe8567`) |
| NCCL 2.31.2 (sm120 build) | NVIDIA | collectives used by the fused, permute-free Ulysses exchange |
| FlashInfer 0.6.18 | Apache-2.0 | present in the runtime |
| [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | see model card | the model whose VSA per-head block lists this integrates with |
| [lightx2v/MiniMax-H3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | see model card | the 4-step distilled LoRA used when serving; not part of this diff |

## Dependencies not included in this PR

- the `sol_attn` package (NVlabs/Sana `sol-engine`, Apache-2.0) is imported **behind a
  `try/except`**: without it the `SOL_ATTN` backend is registered but inert, and nothing else
  breaks. It is vendored separately and is not in this diff.
- the `b12x` package is optional in the same way (SM120/SM121 hosts only).

