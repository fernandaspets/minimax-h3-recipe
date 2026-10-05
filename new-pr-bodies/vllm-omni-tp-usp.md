An isolated infrastructure fix, reported separately because other parallel topologies need it.

This replaces #8486, closed for the same reason as its companion pull requests (opened before the
end-to-end test was complete). The diff is unchanged.

## Problem

`symmetric-memory` resolves a group's host communicator on the **current device**. In
`group_coordinator.py` the device groups were created with

```python
torch.distributed.new_group(ranks, backend=torch_distributed_backend)
```

so the group's host communicator was not bound to the rank's device. Combined with a permute-free
(fused) Ulysses all-to-all, that fails at runtime with

```
NCCL host communicator for group N not found
```

and the same class of failure occurs if the device is not selected before `init_process_group`.

## Change

- pass `device_id=torch.device("cuda", self.local_rank)` to every `new_group` call when the backend
  is NCCL (unchanged for other backends);
- select the rank's device (via the platform's `set_device`) **before** `init_process_group` in
  `parallel_state.py`.

## Why it matters

TP and sequence (Ulysses) parallelism can then be used together. On this PCIe, no-NVLink host,
TP2 x USP2 is faster than TP1 x USP4 and TP4 x USP1 in our runs (the Ulysses payload fans out to more
peers, or the TP all-reduce grows), so the combined topology is the one worth having.

Single-axis topologies are expected to be unaffected — the groups are the same, only bound to the
device they were already intended to run on — but that is reasoned rather than separately measured
here.

## Testing

Exercised by serving MiniMax-H3 `ref2va` at `--tensor-parallel-size 2 --usp 2` on 4x RTX PRO 6000
(SM120), in a run that rendered a decodable clip; the exchange that previously aborted with the
host-communicator error completes. The project's own test suite was not run here (it needs the model
weights).

## Risk

Low and localised. Only NCCL group creation is touched; the `gloo` groups are untouched.

---

## AI disclosure

Authored in a `pi` agentic coding session driven by `deepseek-v4.1-flash`. A human maintainer set
the scope, reviewed the result, and submitted it.

---

## Sources

- [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) — Apache-2.0; the group
  coordinator this changes.
- NVIDIA NCCL `symmetric-memory` — it resolves a group's host communicator on the current device,
  which is why the group must be created with a bound `device_id`.
- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) — the model the topology was
  validated with.
