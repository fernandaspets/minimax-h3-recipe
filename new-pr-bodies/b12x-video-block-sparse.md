Adds an opt-in block-sparse walk for video attention, selected by `is_block_sparse`. The walk is
**off by default**; nothing changes unless a caller opts in.

This replaces #480, closed for the same reason as its companion pull requests (opened before the
end-to-end test was complete). The diff is unchanged.

## What lands

| commit | change |
|---|---|
| `4effb95f` | the video block-list walk behind `is_block_sparse` (scaffold) |
| `a9b3eef7` | plumb the block-list metadata through the contiguous layer |
| `cd139cf6` | **pass `is_block_sparse` to the VARLEN kernel — fixes the walk being inert** |
| `7c476eeb` | tests, a benchmark, and the release fragment |
| `1b812b6b` | per-segment block lists |

`cd139cf6` is not cosmetic: without it the keyword never reached the kernel and the walk had no effect
while appearing to be enabled. Anyone who picked up the scaffold needs that commit for the feature to
do anything.

## Testing

Correctness of the sparse walk is exercised against the dense path, and the dense path is unchanged
(verified bit-identical). The release fragment records the measurement the walk was developed against:
at H3 geometry (S=18748, H=56, D=128, RTX 5090, 400 W) dense 51.611 ms/call against sparse
5.043 ms/call at 9.6% block density — 10.23x. Gates recorded there: a full list reproduces dense
exactly, a sparse list matches the fp32 oracle under the same mask, an empty list attends nothing
(so the walk is provably live rather than silently dense), partial final blocks are handled, bind
allocates no device memory, and the path is CUDA-graph capturable and replays exactly.

That speedup is a property of the walk at that geometry and density, not a serving claim: the density
is data-dependent, and no end-to-end speed-up is claimed here.

## Scope

No default behaviour change; `block_sparse` defaults to `False`. The new keyword arguments on
`varlen.plan` and `varlen.bind` are optional.

---

## AI disclosure

Authored in a `pi` agentic coding session driven by `deepseek-v4.1-flash`. A human maintainer set
the scope, reviewed the result, and submitted it.

---

## Sources

| source | licence | how it is used |
|---|---|---|
| [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) | Apache-2.0 | the library this change lives in; the VARLEN kernel it targets is b12x's own |
| [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | model licence | the model whose VSA per-head block-list scheme the walk serves |

The file touched carries the upstream attribution already in the tree: `Copyright (c) 2025, Jay Shah,
Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao` — the CuTe DSL attention lineage.
