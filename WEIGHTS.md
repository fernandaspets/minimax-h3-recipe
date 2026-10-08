# WEIGHTS — what to download, from where, and what is optional

The image contains **no model weights**. They are a ~270 GB download and belong in a cache, not an
image layer. Everything below is a public Hugging Face repository except the identity RefMods, which are
optional per-subject assets you supply yourself.

## Required

| what | Hugging Face repo | notes |
|---|---|---|
| **base model** | [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) | public, ungated. Diffusers-format MiniMax-H3. ~269 GB. |
| **4-step turbo LoRA** | [`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | public. Use `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` — this is the standing 4-step recipe. |

### The `ref2va` config

Serving uses `--task-type ref2va`, which points at a directory that is a small `model_index.json`
plus a `Ref2VA` component. The lane's copy is:

```
/models/MiniMaxAI/MiniMax-H3-ref2va/
├── model_index.json        # the ref2va composition
└── Ref2VA -> ../MiniMax-H3/Ref2VA
```

`Ref2VA` already ships inside `MiniMaxAI/MiniMax-H3`, so this is a 4 KB wrapper plus a symlink, not a
separate download. The image's `scripts/ref2va/` holds the wrapper; point it at your
`MiniMax-H3/Ref2VA` and the lane serves.

## Optional — identity RefMods

A RefMod is an identity adapter: one pre-encoded latent in the H3 video-VAE latent space, injected as
reference rows because it cannot be decoded back through the VAE. It pins one subject's identity
across shots. **None are shipped here** — they are per-subject assets you produce or obtain yourself.

**Not required to run.** Without them the lane boots and renders; identity then comes from the prompt
and any reference images, so it drifts between shots. Add them when you want a fixed character.

To use one or more:

1. Put each subject's `.safetensors` where the container can read it. Each file holds a single tensor
   named `latent` — the pre-encoded identity latent (4-D is accepted; the batch axis is added).
2. Mount them and set `H3_REFMOD_PATHS` to the in-container paths, as an `os.pathsep`-separated list:

```bash
docker run ... \
  -v /your/refmods:/refmods:ro \
  -e H3_REFMOD_PATHS=/refmods/alice.safetensors:/refmods/bob.safetensors \
  --entrypoint bash local/h3kk:h3 -lc 'bash /opt/h3/scripts/serve_arwire.sh'
```

They are appended as reference rows in the order given, so that order is a property of the request.

`H3_REFMOD_NORMALIZE` (default on, set by the launcher) applies `(latent - mean) / std` before
patchifying, the same treatment the model's own image path applies. Set it to `0` if your latents are
already normalised.

## Optional - learned latent upscaler (2K delivery)

`minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors` (LBH-123-AI/Minimax_h3_latent_Upscaler) turns
a sampled H3 latent into a larger one, which `latent_refine` then re-samples at low denoise. Point
`H3_LATENT_UPSCALER` at it to enable the `latent_upscale` / `latent_refine` request knobs. It is
trained one normalisation below the pipeline latent, so the loader normalises in and denormalises
out; feeding it a raw pipeline latent inflates the output about 5x and decodes as a magenta grid.

## Reference requests on the t2va weights (the studio's route)

The reference task can run on the t2va partition instead of the Ref2VA checkpoint: the ref2v
turbo adapter binds to the *served task type* (ref2va), while the transformer, VAEs and text
encoder are the t2va partition's. Build the wrapper with
`scripts/make_ref_on_t2va_wrapper.sh` and select it with `H3_REF_ON_T2VA=1`.

## Pointing the lane at the weights

```bash
# default: local paths, as the lane has always run
H3_WEIGHTS_SOURCE=local

# or: resolve from Hugging Face into the cache
H3_WEIGHTS_SOURCE=hf
```

`scripts/h3_fetch_weights.py` handles the `hf` path (repo ids and revisions from the table above).
The acceptance test for the `hf` path is that it renders a clip **byte-identical** to the `local` path
on the same seed; that has not yet been demonstrated, so treat `hf` as convenience, not as verified
equivalence.

## Sizing

| | |
|---|---|
| weights on disk | ~272 GB (model + LoRA) |
| cache volume | mount a large volume at `/root/.cache/huggingface` |
| host RAM | the lane parks in host RAM when asleep (`enable_sleep_mode`), so leave headroom |
| GPUs | 4 × SM120 (RTX PRO 6000 class), TP2 × USP2. The lane does not fit on 32 GB cards. |
