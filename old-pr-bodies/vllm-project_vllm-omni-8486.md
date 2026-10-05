Cherry-picked from a working H3 deployment. Reported separately because it is an isolated
infrastructure fix that other parallel topologies need.

## Problem

`symmetric-memory` resolves a group's host communicator on the **current device**. In
`group_coordinator.py` the device groups were created with

```python
torch.distributed.new_group(ranks, backend=torch_distributed_backend)
```

so the group's host communicator was not bound to the rank's device. Combined with a
permute-free (fused) Ulysses all-to-all, that fails at runtime with

```
NCCL host communicator for group N not found
```

and the same class of failure occurs if the device is not selected before
`init_process_group`.

## Change

- pass `device_id=torch.device("cuda", self.local_rank)` to every `new_group` call when the
  backend is NCCL (unchanged for other backends);
- select the rank's device (via the platform's `set_device`) **before**
  `init_process_group` in `parallel_state.py`.

No behavioural change when a single parallelism axis is used: the groups are the same, only
bound to the device they were already intended to run on.

## Why it matters

TP and sequence (Ulysses) parallelism can then be used **together**. On a PCIe, no-NVLink
host we measured `TP2 x USP2` as the fastest shape for MiniMax-H3; `TP1 x USP4` and
`TP4 x USP1` are both slower (the Ulysses payload fans out to more peers, or the TP
all-reduce grows), so the combined topology is the one worth having.

## Testing

Validated by running the MiniMax-H3 `ref2va` pipeline at `--tensor-parallel-size 2 --usp 2`
on 4x RTX PRO 6000 Blackwell (SM120): the exchange that previously aborted with the
host-communicator error completes and renders. The project's own test suite was not run
here (it needs the model weights).

## Risk

Low and localised. Only NCCL group creation is touched; the `gloo` groups are untouched.

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

- [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) — Apache-2.0; the group
  coordinator this changes, and the design it follows.
- The behaviour being satisfied is NVIDIA NCCL's `symmetric-memory`: it resolves a group's host
  communicator on the current device, which is why the group must be created with a bound
  `device_id`.
- The topology result (TP x USP together being the fastest shape here) was measured on a Local
  Inference Lab SM120 runtime over the NVIDIA PyTorch NGC container
  (`nvcr.io/nvidia/pytorch@sha256:33ef5fc1…`).
- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) is the model the topology
  was validated with.

