# MiniMax-H3 serving recipe

Container build and run recipe for serving MiniMax-H3 on 4× RTX PRO 6000 Blackwell (SM120).

Everything the image needs is either a digest-pinned public image, a SHA-pinned public git
revision, or vendored in this directory with its licence. Model weights are not in the image.

## Component PRs

| component | what it provides |
|---|---|
| [vllm-project/vllm-omni #8490](https://github.com/vllm-project/vllm-omni/pull/8490) | B12X (SM120/SM121) attention backends, SOL_ATTN backend, opt-in MXFP8/NVFP4 paths, the `vllm_omni/diffusion/models/minimax_h3` wire + quantisation modules |
| [vllm-project/vllm-omni #8491](https://github.com/vllm-project/vllm-omni/pull/8491) | communicator binding to the rank device for symmetric memory (not built into this image) |
| [local-inference-lab/b12x #482](https://github.com/local-inference-lab/b12x/pull/482) | var-length attention with per-head block lists (opt-in) |
| this repo | the image recipe: pinned inputs, vendored third-party, hashed lock, launcher and request examples |

## Requirements

- 4× SM120 GPU with ≥ 90 GB each (one partition of the model is 144 GB; it is sharded TP2 × USP2)
- ~300 GB disk for one partition, or ~600 GB for both
- a container runtime with `--gpus`
- the runtime base image is pulled anonymously; no registry credentials are needed

## Build

```bash
./build.sh h3                       # tags local/h3kk:h3
```

The image is built from the online pull requests, never from a local checkout: `build.sh` resolves
each head commit with `git ls-remote` at build time, so a pin can never drift from what was
reviewed.

| source | revision built |
|---|---|
| [vllm-project/vllm-omni #8490](https://github.com/vllm-project/vllm-omni/pull/8490) | `fernandaspets/vllm-omni` `h3/features` |
| [local-inference-lab/b12x #482](https://github.com/local-inference-lab/b12x/pull/482) | `fernandaspets/b12x` `feat/video-block-sparse` |

The image does not take [vllm-project/vllm-omni #8491](https://github.com/vllm-project/vllm-omni/pull/8491)
(the communicator binding): on this runtime the served topology resolves its groups through plain
NCCL, and the renders below were verified without it.

To build from somewhere else — upstream once these merge, or a mirror — override the four knobs.
Nothing else is read from the host:

```bash
VLLM_OMNI_REPO=https://github.com/vllm-project/vllm-omni VLLM_OMNI_REF=refs/heads/main \
B12X_REPO=https://github.com/local-inference-lab/b12x   B12X_REF=refs/heads/main \
  ./build.sh h3
```

`build.sh` enables Docker's BuildKit, which keeps pip's wheel cache in a mount outside the image:
changing a revision or the lockfile does not re-download every wheel, and the image content is
identical either way. The build fetches the base image, the two source trees, one pinned torchaudio
wheel and the hashed lock — it does **not** fetch model weights (see below).

The build fails if: any module it installs cannot be imported; the wire hook is absent from
`vllm_omni/diffusion/distributed/comm.py`; any package the base runtime ships has changed version;
or the SM120 attention fallback is broken (see `Dockerfile`, step 7).

## Weights

| what | repo | notes |
|---|---|---|
| model | [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) | public, ungated. `Ref2VA/` and `FL2VA/` are separate partitions of 144 GB each. |
| 4-step LoRA | [`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` |

The served directory must contain the partition and the shared components:

```
/models/MiniMaxAI/MiniMax-H3/          # shared: text_encoder, tokenizer, processor, vae, audio_vae
/models/MiniMaxAI/MiniMax-H3/Ref2VA/   # partition used by task_type=ref2va
/models/MiniMaxAI/MiniMax-H3/FL2VA/    # partition used by task_type=t2va and fl2va
```

and a `model_index.json` selecting the partition, or point `MODEL` directly at a directory that
already has one.

`H3_WEIGHTS_SOURCE=hf` resolves `MiniMaxAI/MiniMax-H3` into the Hugging Face cache instead of using
a local tree, and builds the partition wrapper.

## Run

```bash
docker run -d --name h3 --network host --ipc host --shm-size 36g \
  --gpus '"device=0,1,2,3"' \
  -v /models:/models \
  -v h3-state:/var/lib/h3 -v h3-cache:/var/cache/h3 \
  -e H3_QUANT=hybrid -e H3_WIRE=int8 -e H3_STEPS=4 \
  -e H3_WEIGHTS_SOURCE=local -e H3_TASK_TYPE=ref2va \
  --entrypoint bash local/h3kk:h3 -lc 'bash /opt/h3/scripts/serve_arwire.sh'
```

The launcher runs as the container's main process, so `docker logs -f h3` streams the lane and the
container exits when the lane does. Startup prints the resolved arm, the quant policy, the LoRA, the
model and the topology; wait for `Application startup complete`, then `GET /health` returns 200.

`H3_PRINT=1` resolves and prints the configuration without starting anything.

### Mounts

| what | how |
|---|---|
| model | `-v <your model dir>:/models`, or `H3_WEIGHTS_SOURCE=hf` to fetch into the HF cache |
| HF cache | `-v <dir or volume>:/root/.cache/huggingface` — `e2e_test.sh` takes this as `H3_HF_CACHE_MOUNT` (default `$HOME/.cache/huggingface:/root/.cache/huggingface`) |
| state / compile cache | `-v h3-state:/var/lib/h3 -v h3-cache:/var/cache/h3` (optional, but they persist the control files and the compile cache) |

The shipped `e2e_test.sh` assumes nothing about the host: model paths come in through `H3_MOUNTS`
as a space-separated list of `host:container` specs, so a plain directory and a named volume both
work.

## Request

One synchronous request; the response body is the mp4.

```bash
# ref2va - audio reference (+ optional reference images)
H3_TASK_TYPE=ref2va \
H3_PROMPT_FILE=/prompts/prompt.txt \
H3_AUDIO_FILE=/audio/ref_32k.wav \
  bash /opt/h3/scripts/request_render.sh out.mp4 4 12.0 3.0 5.0

# t2va - prompt only (FL2VA partition)
H3_TASK_TYPE=t2va H3_PROMPT_FILE=/prompts/prompt.txt \
  bash /opt/h3/scripts/request_render.sh out.mp4 4 12.0 3.0 5.0

# fl2va - keyframes as input_references (FL2VA partition)
H3_TASK_TYPE=fl2va H3_PROMPT_FILE=/prompts/prompt.txt \
H3_INPUT_IMAGES=/refs/first.png,/refs/last.png \
  bash /opt/h3/scripts/request_render.sh out.mp4 4 12.0 3.0 5.0
```

`H3_TASK_TYPE` must match the partition the server was started with: the Ref2VA partition serves
only `ref2va`, the FL2VA partition serves `t2va` and `fl2va`.

Equivalent curl for the model form fields — prompt is a form field, `audio_reference` is a JSON
string field (`{"audio_url": "data:audio/wav;base64,..."}`), and images are file parts named
`input_references`:

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/v1/videos/sync \
  --form-string "prompt=$(cat prompt.txt)" \
  -F aspect_ratio=16:9 -F width=1344 -F height=768 -F fps=24 -F seconds=5 \
  -F flow_shift=12.0 -F num_inference_steps=4 -F seed=0 \
  -F "audio_reference=</tmp/audio_ref.json" \
  -F "input_references=@ref.jpg" \
  -F 'extra_params={"task":"ref2va","duration":5.0,"audio_flow_shift":3.0}' \
  -o out.mp4
```

Output geometry: 1344×768 at 24 fps, frames quantised to `17n+5` (5.0 s → 124 frames).

### Request constraints

These are enforced by the model, so a request that breaks one is rejected with HTTP 400 rather
than silently degraded, and the body names the field that broke.

| field | rule |
|---|---|
| `model` | must be **the model root the server was started with**. Requesting `ref2va` against an FL2VA server, or sending the generic default while the server runs a partition path, fails with `Model mismatch: request specifies 'X' but server is running 'Y'.` |
| `seconds` | output duration in **[4, 15] s** by default (higher only with `long_video`). fps is fixed at **24** and the frame count quantises to `17n+5`, so a 5.0 s clip is 124 frames |
| image parts | `input_references` (**plural**, repeatable). `input_reference` is a separate single-file field, and sending `input_references` together with `input_reference`, `image_reference` or `video_reference` is a 400: `Provide input_references alone, without input_reference, image_reference, or video_reference.` |
| image formats | JPG, JPEG, PNG, WEBP, HEIC, HEIF |
| image file size | ≤ **30 MiB** each |
| image dimensions | both axes in **[256, 5760] px** — `min(width, height) >= 256` and `max(width, height) <= 5760` |
| image aspect ratio | width/height in **[0.4, 2.5]** |
| image grid | each validated axis is then rounded to the **32 px** grid |
| `aspect_ratio` | one of `21:9` `16:9` `4:3` `1:1` `3:4` `9:16` |
| output short edge | fixed at **768** (`1344x768`); it is the output short edge, not a free parameter |
| `ref2va` | requires at least one image or video condition; at most **9 images**, **3 videos**, **3 standalone audio** references and **12 references total** |
| `t2va` | prompt only — image, video and audio conditions are all rejected |
| `fl2va` | keyframes as `input_references`: at least one, **at most the first and last** (2); the keyframe supplies the aspect ratio |
| `audio_reference` | a JSON string field, `{"audio_url": "data:audio/wav;base64,..."}`; only http(s) or data URLs |
| partition | `H3_TASK_TYPE` must match the partition the server was started with: Ref2VA serves `ref2va`, FL2VA serves `t2va` and `fl2va` |

Reference images are **not** resized to the output canvas: each is validated against the ranges
above and its own axes are rounded to the 32 px grid, so its resolution is preserved. Pass a
reference at the size you want the subject encoded at — an image below 256 px on either axis is
rejected outright, not upscaled.

## Reference requests on the t2va weights

The reference pipeline does not need the Ref2VA checkpoint. A `ref2va` request is served by the
Ref2VA transformer only when one is loaded; otherwise it runs on the regular transformer. So the
reference task can run on the **t2va weights** - the practice the community reports as better
quality - and the only thing that blocks it by default is the partition manifest, whose task list for
FL2VA is `["t2va", "fl2va"]`.

Enable it with a wrapper directory that carries the t2va partition and a manifest that also names
`ref2va`. The components are symlinked, so nothing is duplicated and the original partition is left
untouched:

```bash
SRC=/models/MiniMaxAI/MiniMax-H3/FL2VA
W=/models/MiniMaxAI/MiniMax-H3-refpipe-on-t2va
mkdir -p "$W/FL2VA"
cp /models/MiniMaxAI/MiniMax-H3-fl2va/model_index.json "$W/model_index.json"
for e in "$SRC"/*; do b=$(basename "$e"); [ "$b" = model_index.json ] && continue; \
  ln -sfn "../../MiniMax-H3/FL2VA/$b" "$W/FL2VA/$b"; done
python3 - <<'PY'
import json
p = "/models/MiniMaxAI/MiniMax-H3-refpipe-on-t2va/FL2VA/model_index.json"
d = json.load(open("/models/MiniMaxAI/MiniMax-H3/FL2VA/model_index.json"))
d["_minimax_h3"]["tasks"] = ["t2va", "fl2va", "ref2va"]
open(p, "w").write(json.dumps(d, indent=2) + "\n")
PY
```

Symlinks must be **relative**: with a container mount the host's absolute paths do not exist inside
the container, and a component that cannot be resolved fails the boot with only
`Orchestrator initialization failed:` (no cause in the API log).

`scripts/make_ref_on_t2va_wrapper.sh` builds it (idempotent, relative symlinks, refuses a
broken link); the manual form below is only useful to see what it does. To select the route at
serve time set `H3_REF_ON_T2VA=1`: it forces the ref2va task type the adapter requires and points
`MODEL` at the wrapper, so the weights and the task type cannot drift apart.

Serve and request it (the served partition and the request task deliberately differ):

```bash
MODEL=/models/MiniMaxAI/MiniMax-H3-refpipe-on-t2va H3_TASK_TYPE=fl2va H3_STEPS=4 \
  bash /opt/h3/scripts/serve_arwire.sh

H3_TASK_TYPE=fl2va H3_REQUEST_TASK=ref2va \
H3_PROMPT_FILE=/prompts/prompt.txt \
H3_INPUT_IMAGES=/refs/a.png,/refs/b.png \
  bash /opt/h3/scripts/request_render.sh out.mp4 4 12.0 3.0 5.0
```

The turbo adapter is the ref2v family, because the request task is what the adapter binds to.

### Alternative: the `combined` partition

The fork also ships `combined`, which loads the regular FL2VA transformer **and** the Ref2VA
transformer and routes reference requests to the latter. It is a correct partition but a memory
decision: on this 4-GPU box each rank then holds one full transformer's worth of both models and the
boot OOMs at TP2 x USP2 (95 GiB, 107 MiB free); it fits at TP4, which is about 1.7x slower. Use it
only when the Ref2VA transformer's own weights are wanted.

## Latent upscale (`latent_upscale` / `latent_refine`)

The pipeline can decode a larger frame than it sampled: a learned 3D latent upscaler followed by a
short low-denoise second pass, both request-level.

```bash
# serve with the learned upscaler available
H3_LATENT_UPSCALER=/models/MiniMaxAI/h3-latent-upscaler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors \
  bash /opt/h3/scripts/serve_arwire.sh

# request 1344x768 sampling and a 2688x1536 output
H3_LATENT_UPSCALE='{"width":2688,"height":1536}' H3_LATENT_REFINE=0.4 \
  bash /opt/h3/scripts/request_render.sh out.mp4 4 12.0 3.0 5.0
```

`H3_LATENT_REFINE` is the fraction of steps the second pass re-runs (0.3-0.5 is the usable band;
1.0 re-samples rather than refines). The pipeline refuses a refine layout above a A refine pass also needs the quantised linears prepared for its row count: H3_MX_CAPACITY raises
the per-rank prepared capacity (default 40,960 rows, which covers a 1344x768 film scene with
references; a 2688x1536x10 s refine asks for 148,352 and fails with "rows exceeds the prepared
capacity" without it). It costs workspace memory, so it is explicit.

per-rank token
guard - default 65,536, which is a *validated* bound, not a hardware limit - with an error naming the
exact token count. `H3_LATENT_REFINE_MAX_TOKENS` raises that guard explicitly for a deployment that
has already run that attention width; 2688x1536x5 s on TP2 x USP2 needs about 77,568 rows/rank, the
width this lane already runs on 9.5 s film scenes.

## Measured

All numbers are engine time for 3 warm requests after a cold warmup, 1344×768, 4 steps, one writer
(no scheduler queue wait). 480 W power limit.

| arm | engine, 3 warm requests | s/it | clip bytes |
|---|---|---|---|
| bf16 NCCL wire | 18.44 / 18.66 s | 3.52–3.56 | 7,399,626 |
| int8 a2a + all-reduce wire | 14.328 / 14.481 / 14.494 s | 2.69–2.73 | 7,198,351 |
| + per-role quant policy (hybrid) | 13.482 / 13.462 / 13.493 s | 2.47–2.55 | 7,896,537 |
| hybrid, rebuilt image | 13.32 / 13.50 / 13.49 s | — | 9,298,977 |

Each arm is three warm requests after a cold warmup, same seed and geometry, so a rerun on the same
hardware should land in the same band; the int8 arms render byte-identical clips run to run.

The wire change accounts for ~4.2 s per clip, the quant policy a further ~0.9 s.

### Serving under concurrency

Client wall: from the HTTP request being issued to the response body being written to disk,
including connection setup, the multipart upload (prompt and reference images) and the response
body transfer.

Workload: `ref2va`, 1344x768, 24 fps, 5 s, 4 steps, `flow_shift=12`, `audio_flow_shift=3`,
seed 1000, `TP2 x USP2`, `H3_QUANT=mxfp8`. One cold warmup, then C simultaneous requests to
`POST /v1/videos/sync`.

| wire | C | throughput (req/s) | mean (s) | median (s) | p95 (s) | max (s) |
|---|---|---|---|---|---|---|
| bf16 | 1 | 0.057 | 17.40 | 17.40 | 17.40 | 17.40 |
| bf16 | 8 | 0.064 | 71.67 | 71.67 | 120.28 | 125.64 |
| bf16 | 16 | 0.064 | 134.47 | 134.53 | 239.20 | 250.82 |
| bf16 | 32 | 0.064 | 258.93 | 258.93 | 475.97 | 499.93 |
| int8 | 1 | 0.064 | 15.67 | 15.67 | 15.67 | 15.67 |
| int8 | 8 | 0.073 | 62.95 | 63.00 | 105.37 | 110.07 |
| int8 | 16 | 0.073 | 117.91 | 117.80 | 209.91 | 220.01 |
| int8 | 32 | 0.073 | 226.66 | 226.51 | 416.11 | 437.17 |

`wire` is `H3_A2A_WIRE` / `H3_AR_WIRE`. Both arms run `mxfp8`, body `SOL_ATTN`, refiner `B12X`.

### Layout

Base ref2va, 4 steps, 5 s / 1344x768:

| layout | s/step | 4-step render |
|---|---|---|
| TP4 x USP1 | 11.75-12.17 | ~60-66 s |
| TP2 x USP1 | 12.44-13.32 | 68.6 s |
| TP2 x USP2 | 7.83-8.97 | 47.5 s |

`tp2usp2` needs the encoder-group change carried in this branch. On the stock tree
`_build_text_encoder_group` builds its group over `range(text_encoder_tp_size)`, which asserts
whenever the DiT world is TP x SP, so every TP x USP layout fails at startup.

## Knobs

| variable | default | values |
|---|---|---|
| `H3_QUANT` | `mxfp8` | `mxfp8` all linears MXFP8; `hybrid` mlp NVFP4 + attn MXFP8; `nvfp4` W4A4 where supported. Refiner stays bf16. |
| `H3_STEPS` | `4` | `2` `4` `8` — selects the LoRA (the family must match the partition; see WEIGHTS.md); the step count is also sent per request |
| `H3_TASK_TYPE` | `ref2va` | `ref2va` `t2va` `fl2va` — selects the served partition |
| `H3_TOPOLOGY` | `tp2usp2` | `tp2usp2` `usp4` `tp4` `tp2usp1` `tp1usp2` |
| `H3_A2A_PERMUTE` | `0` | `1` enables the permute-free all-to-all; passed to the server as `--ulysses-a2a-permute` (boot-time; requires the LIL NCCL in the image) |
| `H3_WIRE` | `bf16` | `bf16` (stock vLLM path) or `int8` (lossy a2a + all-reduce transport; see Measured) |
| `H3_QUANT_CONFIG` | unset | JSON for `--diffusion-quantization-config`, to set precision per layer role instead of per arm, e.g. `{"transformer.*.mlp": {"method": "nvfp4"}, "transformer.*.attn": "mxfp8"}`. Roles it does not name follow `H3_QUANT`. |
| `H3_A2A_WIRE_BUFCACHE` | `0` | `1` reuses wire buffers; measured to corrupt the output, do not enable |
| `H3_A2A_QKV_BATCH` | `0` | `1` fuses the per-block q/k/v all-to-all; measured slower, left off |
| `SOL_ATTN_TAU` | `1.0` | SOL_ATTN sparsity threshold |
| `SOL_ATTN_GATE` | `0` | `1` enables the correctness gate |
| `H3_VAE_COMPILE` | `0` | `1` compiles the VAE decoder ViT (measured -0.44 to -0.51 s/clip; clip not byte-identical) |
| `H3_STEP_PROFILE` | `0` | `1` per-step timing instead of a timing run |
| `H3_REFMOD_PATHS` | unset | optional identity adapters, `a.safetensors:b.safetensors` |
| `H3_LORA` | per arm | override the turbo adapter; the family must match the partition |

Attention backends are set in `serve_arwire.sh`: body `SOL_ATTN`, refiner `B12X`, default `B12X`.

## Measured and not adopted

Recorded so the same ground is not re-covered; each was measured on this lane and rejected.

| candidate | result |
|---|---|
| `H3_A2A_QKV_BATCH=1` | slower (+3.4 %); the batched collective does not pay for the concatenation |
| `H3_A2A_PERMUTE=1` | rejected; the permute puts bf16 back on the wire (+2.5 s/clip) |
| `H3_A2A_WIRE_BUFCACHE=1` | corrupts the clip (deterministic colour mosaic, still exits 0) |
| 2-step student (`H3_STEPS=2`) | faster (8.85 s MXFP8 / 8.25 s NVFP4) but rejected on output quality |
| `H3_TOPOLOGY=usp4` / `tp4` | both ~1.7× slower; TP2 × USP2 is optimal from both sides |
| `TORCHINDUCTOR_CUDAGRAPH_TREES=1` | no gain, and it breaks the VAE on this lane |
| SOL_ATTN Q_TILE variant | slower; the shipped Triton forward stays |
| SOL_ATTN `SOL_FWD_BV` split | below the run-to-run band |

No performance claim is made for any option above; the observed direction is the whole record.

## Third-party

| component | licence | how |
|---|---|---|
| `third_party/sol_attn` | Apache-2.0 (NVlabs/Sana `sol-engine`) | vendored, `THIRD_PARTY_NOTICES.md` |
| `b12x` | Apache-2.0 | built from the PR revision |
| cuDNN Frontend block-sparse-attention reference | Apache-2.0 | adapted inside the vendored `sol_attn` |
| FlashAttention CuTe DSL | BSD-3-Clause | vendored inside `sol_attn` |
