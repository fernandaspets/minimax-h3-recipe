# Discord post — copy from the line below into Discord

MiniMax-H3 on 4× RTX PRO 6000 (SM120) — three PRs + a runnable image recipe

Working inference for MiniMax-H3 (ref2va / t2va / fl2va) on SM120: int8 collectives, opt-in
MXFP8/NVFP4 DiT quantisation, and a var-length attention backend. 1344×768, ~5 s clips, 4 steps.

• vllm-omni — B12X + SOL_ATTN attention backends, opt-in MXFP8/NVFP4, the H3 comms/quant modules
  <https://github.com/vllm-project/vllm-omni/pull/8487>
• vllm-omni — the communicator binds to the rank device
  <https://github.com/vllm-project/vllm-omni/pull/8486>
• b12x — var-length attention with per-head block lists
  <https://github.com/local-inference-lab/b12x/pull/480>

Image recipe — Dockerfile, build.sh, launcher and request drivers, public sources only:
https://github.com/fernandaspets/minimax-h3-recipe

  ./build.sh h3

No weights are downloaded at build time; the model partition is mounted at run time. README.md has
the arm knobs (H3_QUANT, H3_STEPS, H3_WIRE, H3_TASK_TYPE) and WEIGHTS.md the model layout.

VERIFY LINE — replace once the three-type run is confirmed:
  Verified: 49/49 lane tests, ref2va / t2va / fl2va each rendered end to end, runtime gate
  339 base packages unchanged.

================================================================================
NOTES (not part of the post)

- The three PR links are wrapped in <...> to suppress Discord's preview embeds, as asked. The repo
  link is left plain so Discord shows its card; wrap it too if you want it suppressed.
- Repo: https://github.com/fernandaspets/minimax-h3-recipe (public, main branch, Apache-2.0).
  Rename is free — say the word.
- Everything in this folder is what a stranger clones: Dockerfile, build.sh, e2e_test.sh,
  requirements.lock, runtime-dist.json, README.md, WEIGHTS.md, PROVENANCE.md, LICENSE,
  scripts/, third_party/.
- The image downloads the base runtime, the two source trees, one pinned torchaudio wheel and the
  hashed lock. It never downloads weights.
