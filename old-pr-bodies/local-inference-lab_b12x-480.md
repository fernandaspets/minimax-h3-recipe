Adds an opt-in block-sparse walk for video attention, selected by `is_block_sparse`.

The walk is **off by default**; nothing changes unless a caller opts in.

## What lands

| commit | change |
|---|---|
| `4effb95f` | the video block-list walk behind `is_block_sparse` (scaffold) |
| `a9b3eef7` | plumb the block-list metadata through the contiguous layer |
| `cd139cf6` | **pass `is_block_sparse` to the VARLEN kernel — fixes the walk being inert** |
| `7c476eeb` | tests, a benchmark, and a release fragment |
| `1b812b6b` | per-segment block lists |

`cd139cf6` is not cosmetic: without it the keyword never reached the kernel and the walk had no
effect while appearing to be enabled. Anyone who has already picked up the scaffold must take this
commit for the feature to do anything.

## Testing

The tests and benchmark for the sparse path are in `7c476eeb`. Correctness of the sparse walk is
exercised against the dense path; the dense path is unchanged.

## Scope

No performance numbers are claimed in this PR. The walk is a correctness-and-plumbing change; it is
measured separately, and any speed claim will come with a receipt rather than a commit message.

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

- [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) — Apache-2.0; the
  library this change lives in.
- the file touched carries the upstream attribution already in the tree:
  `Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani,
  Tri Dao` — the CuTe DSL attention lineage.
- NVIDIA CUTLASS / CuTe DSL (`nvidia-cutlass-dsl`), PyTorch and Triton are the kernel stack; the
  sparse-attention walk follows the conventions already used by this library, and the VARLEN
  kernel it targets is b12x's own.

