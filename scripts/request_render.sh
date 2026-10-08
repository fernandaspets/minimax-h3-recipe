#!/bin/bash
# request_render.sh - one synchronous render request against a served lane.
#
#   request_render.sh <out.mp4> [steps] [flow_shift] [audio_flow_shift] [seconds]
#
# Env:
#   PORT=8000                     server port
#   H3_TASK_TYPE=ref2va           request task: ref2va | t2va | fl2va
#   H3_REQUEST_TASK=<task>        override just the request task; use it to send a ref2va
#                                 request to a lane served with --task-type combined
#   H3_PROMPT_FILE=<file>         required - the prompt text
#   H3_AUDIO_FILE=<wav>           optional - audio conditioning, sent as a data: URL
#   H3_INPUT_IMAGES=a.jpg,b.jpg   optional - reference images (ref2va) or keyframes (fl2va)
#   SEED=0  WIDTH=1344  HEIGHT=768  FPS=24  ASPECT_RATIO=16:9
#
# The response body IS the mp4; curl writes it straight to <out.mp4>. A non-zero exit means the
# server returned an error, and with --fail-with-body the error body is written to the output file.
set -uo pipefail

OUT=${1:-out.mp4}
STEPS=${2:-4}
FLOW=${3:-12.0}
AFLOW=${4:-3.0}
DUR=${5:-5.0}
SEC=${DUR%%.*}                        # the seconds field must be a positive integer string

PORT=${PORT:-8000}
TASK=${H3_REQUEST_TASK:-${H3_TASK_TYPE:-ref2va}}
SEED=${SEED:-0}
WIDTH=${WIDTH:-1344}
HEIGHT=${HEIGHT:-768}
FPS=${FPS:-24}
ASPECT_RATIO=${ASPECT_RATIO:-16:9}
PROMPT_FILE=${H3_PROMPT_FILE:?set H3_PROMPT_FILE to the prompt file}

ARGS=(
  --fail-with-body -sS -X POST "http://127.0.0.1:${PORT}/v1/videos/sync"
  --form-string "prompt=$(cat "$PROMPT_FILE")"
  -F "aspect_ratio=$ASPECT_RATIO" -F "width=$WIDTH" -F "height=$HEIGHT" -F "fps=$FPS"
  -F "seconds=$SEC" -F "flow_shift=$FLOW" -F "num_inference_steps=$STEPS" -F "seed=$SEED"
)

if [ -n "${H3_AUDIO_FILE:-}" ] && [ -f "${H3_AUDIO_FILE}" ]; then
  AUDIO_JSON=$(mktemp)
  printf '{"audio_url":"data:audio/wav;base64,%s"}' "$(base64 -w0 "$H3_AUDIO_FILE")" > "$AUDIO_JSON"
  ARGS+=(-F "audio_reference=<$AUDIO_JSON")
fi

# reference images (ref2va) or keyframes (fl2va). t2va takes none.
if [ -n "${H3_INPUT_IMAGES:-}" ]; then
  IFS=',' read -r -a _imgs <<< "$H3_INPUT_IMAGES"
  for _img in "${_imgs[@]}"; do
    ARGS+=(-F "input_references=@${_img}")
  done
fi

# Optional latent upscale / refine. Same request, larger output: H3_LATENT_UPSCALE takes the
# upstream sizing spec (2.0, {"scale":2}, {"width":2688,"height":1536}, {"megapixels":4});
# H3_LATENT_REFINE takes the second-pass strength (0.3-0.5), the img2img fraction of steps.
EXTRA="{\"task\":\"$TASK\",\"duration\":$DUR,\"audio_flow_shift\":$AFLOW"
[ -n "${H3_LATENT_UPSCALE:-}" ] && EXTRA="$EXTRA,\"latent_upscale\":$H3_LATENT_UPSCALE"
[ -n "${H3_LATENT_REFINE:-}" ] && EXTRA="$EXTRA,\"latent_refine\":$H3_LATENT_REFINE"
EXTRA="$EXTRA}"
ARGS+=(-F "extra_params=$EXTRA")

echo "[request] task=$TASK steps=$STEPS flow=$FLOW audio_flow=$AFLOW duration=${DUR}s -> $OUT"
time curl "${ARGS[@]}" -o "$OUT"
rc=$?
if [ "$rc" != 0 ]; then
  echo "[request] FAILED rc=$rc - first 300 bytes of the response:"
  head -c 300 "$OUT"; echo
  exit "$rc"
fi
printf '[request] %s bytes  sha256 %s\n' "$(stat -c %s "$OUT")" "$(sha256sum "$OUT" | cut -d' ' -f1)"
