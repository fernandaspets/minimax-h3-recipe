#!/bin/bash
# e2e_test.sh <image-tag> - boot a built image in a NEW container, render one clip for a chosen
# task type, and verify the ARTIFACT (size, sha256, ffprobe, extracted frames) rather than trusting
# an exit code.
#
#   H3_TASK_TYPE=ref2va|t2va|fl2va   which partition/request type to exercise   (default ref2va)
#   H3_QUANT=hybrid|mxfp8|nvfp4      linear arm                                (default mxfp8)
#   H3_WIRE=bf16|int8                int8 a2a + all-reduce transport           (default bf16)
#   H3_STEPS=2|4|8                   sampling steps                            (default 4)
#   GPUS=2,3,4,5                     the four SM120 cards the lane needs
#
# Weights, prompt and (optionally) reference inputs are read from paths that must exist INSIDE the
# container, so bind them in with H3_MOUNTS (see below).
#
#   H3_MODEL_DIR   base dir holding MiniMax-H3/ and the <task> wrappers
#   H3_LORA_DIR    dir holding the per-step turbo adapters
#   H3_PROMPT_FILE the prompt text
#   H3_AUDIO_FILE  optional wav conditioning
#   H3_INPUT_IMAGES optional comma-separated images (ref2va refs / fl2va keyframes); none for t2va
#
# Writes a receipt directory with serve.log, the request log, the clip, its digest and probe output.
set -uo pipefail

TAG=${1:?usage: e2e_test.sh <image-tag>}
TASK=${H3_TASK_TYPE:-ref2va}
NAME=h3e2e-${TAG//\//_}-${TASK}-$$
GPUS=${GPUS:-2,3,4,5}
H3_QUANT=${H3_QUANT:-mxfp8}
H3_WIRE=${H3_WIRE:-bf16}
H3_STEPS=${H3_STEPS:-4}

H3_MODEL_DIR=${H3_MODEL_DIR:-/models/MiniMaxAI/MiniMax-H3}
H3_LORA_DIR=${H3_LORA_DIR:-/models/h3/turbo}
H3_PROMPT_FILE=${H3_PROMPT_FILE:-}
H3_AUDIO_FILE=${H3_AUDIO_FILE:-}
H3_INPUT_IMAGES=${H3_INPUT_IMAGES:-}
H3_REFMOD_PATHS=${H3_REFMOD_PATHS:-}

# Host paths bound into the container, as a space-separated list of docker -v specs. Nothing about
# the host is assumed; the weights, prompt and reference inputs must be reachable via these mounts.
#
#   H3_MOUNTS="/data/weights:/data/weights /data/prompts:/prompts:ro"
H3_MOUNTS=${H3_MOUNTS:-}

RUN=${H3_RUN_DIR:-/tmp}/e2e-$TAG-$TASK-$(date +%Y%m%dT%H%M%S)

# One writer per lane. Two concurrent boots on the same four cards OOM during load, which reads
# exactly like a lane bug, so a second writer fails loudly instead.
LOCK=/tmp/h3-lane.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[e2e] REFUSING: another lane operation holds $LOCK (holder: $(cat "$LOCK" 2>/dev/null || echo unknown))"
  exit 1
fi
echo "$$ $(date -u +%FT%TZ) e2e_test.sh $TAG $TASK" >&9

for g in ${GPUS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 0)
  if [ "${used:-0}" -gt 5000 ]; then
    echo "[e2e] REFUSING: GPU $g already has ${used} MiB in use (another lane is up). Stop it first."
    exit 1
  fi
done
other=$(docker ps --format '{{.Names}}' | grep -E '^(h3|h3kk|h3e2e|h3acc|h3reboot)' | grep -v "^$NAME$" || true)
if [ -n "$other" ]; then
  echo "[e2e] REFUSING: another H3 container is running: $(echo $other | tr '\n' ' ')"
  exit 1
fi

echo "[e2e] image=local/h3kk:$TAG  container=$NAME  gpus=$GPUS  task=$TASK  arm=$H3_QUANT wire=$H3_WIRE steps=$H3_STEPS"
echo "[e2e] receipt dir: $RUN"
mkdir -p "$RUN"
docker rm -f "$NAME" >/dev/null 2>&1 || true

# The HF cache is a mount spec, so a caller with an existing named volume keeps using it
# (H3_HF_CACHE_MOUNT=mine:/root/.cache/huggingface) and a fresh machine gets a host directory.
H3_HF_CACHE_MOUNT=${H3_HF_CACHE_MOUNT:-$HOME/.cache/huggingface:/root/.cache/huggingface}
case "${H3_HF_CACHE_MOUNT%%:*}" in
  /*|~*) mkdir -p "${H3_HF_CACHE_MOUNT%%:*}" 2>/dev/null || true ;;
esac
MOUNTS=(-v "$H3_HF_CACHE_MOUNT")
# The request runs INSIDE the container but writes the clip to the host's receipt directory, so
# that directory has to be visible there at the same path. Mounting it here means a caller only
# sets H3_RUN_DIR and does not have to remember to add the mount.
MOUNTS+=(-v "$RUN:$RUN")
for _spec in ${H3_MOUNTS:-}; do MOUNTS+=(-v "$_spec"); done

# Refuse to serve from an image whose payload cannot import: a build that cannot import is a failed
# build, not a lane to debug.
if ! docker run --rm --gpus "\"device=$GPUS\"" --entrypoint /opt/venv/bin/python "${MOUNTS[@]}" "local/h3kk:$TAG" \
     -c "import vllm_omni; from vllm_omni.diffusion.models.minimax_h3 import a2a_wire, ar_wire, a2a_qkv_batch, nvfp4, vae_sm120" 2>"$RUN/import.err"; then
  echo "[e2e] FAIL: import gate"; tail -5 "$RUN/import.err"; exit 1
fi
echo "[e2e] import gate: ok"

# The launcher IS the container's main process, exactly as the README documents it, so the lane is
# visible with `docker logs -f <container>`.
docker run -d --name "$NAME" \
  --network host --ipc host --shm-size 36g --gpus "\"device=$GPUS\"" \
  "${MOUNTS[@]}" \
  -e H3_QUANT="$H3_QUANT" -e H3_WIRE="$H3_WIRE" -e H3_STEPS="$H3_STEPS" \
  -e H3_WEIGHTS_SOURCE=local -e H3_TASK_TYPE="$TASK" \
  -e H3_MODEL_DIR="$H3_MODEL_DIR" -e H3_LORA_DIR="$H3_LORA_DIR" \
  -e H3_REFMOD_PATHS="$H3_REFMOD_PATHS" \
  --entrypoint bash "local/h3kk:$TAG" -lc 'exec bash /opt/h3/scripts/serve_arwire.sh' \
  >/dev/null || { echo "[e2e] FAIL: container did not start"; exit 1; }
echo "[e2e] container: $(docker inspect -f '{{.State.Status}}' "$NAME")"

cleanup() {
  docker logs "$NAME" > "$RUN/container.log" 2>&1 || true
  echo "[e2e] tearing down $NAME"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[e2e] serving (launcher is PID 1; follow with: docker logs -f $NAME)"

echo "[e2e] waiting for health (warm ~2 min, cold ~10)"
ok=0
for _ in $(seq 1 90); do
  if [ "$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null)" != "running" ]; then
    echo "[e2e] FAIL: container exited while loading"
    docker logs "$NAME" 2>&1 | tail -25
    exit 1
  fi
  code=$(docker exec "$NAME" /opt/venv/bin/python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5).status)" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { ok=1; echo "[e2e] health 200"; break; }
  sleep 10
done
if [ "$ok" != "1" ]; then
  echo "[e2e] FAIL: no health after 15 min"
  docker logs "$NAME" 2>&1 | tail -30
  exit 1
fi

# The lane's own log is the container's log (the launcher is PID 1); keep a copy in the receipt.
docker logs "$NAME" > "$RUN/serve.log" 2>&1 || true

grep -m1 'partition=' "$RUN/serve.log" 2>/dev/null | sed 's/^/  /' || true
if ! grep -qE "mxfp8=[0-9]+ .*nvfp4=[0-9]+ .*bf16_roles=[0-9]+" "$RUN/serve.log"; then
  echo "[e2e] WARN: quant engagement line not found in serve.log (a silent no-op would hide here)"
fi

echo "[e2e] rendering task=$TASK"
OUT="$RUN/clip.mp4"
docker exec \
  -e H3_TASK_TYPE="$TASK" -e H3_PROMPT_FILE="$H3_PROMPT_FILE" \
  -e H3_AUDIO_FILE="$H3_AUDIO_FILE" -e H3_INPUT_IMAGES="$H3_INPUT_IMAGES" -e PORT=8000 \
  "$NAME" bash /opt/h3/scripts/request_render.sh "$OUT" "$H3_STEPS" 12.0 3.0 5.0 2>&1 \
  | tail -8 | tee "$RUN/request.log"
REQ_RC=${PIPESTATUS[0]}
if [ "$REQ_RC" -ne 0 ]; then
  echo "[e2e] FAIL: the request returned rc=$REQ_RC (see $RUN/request.log)"
  exit 1
fi

if [ ! -s "$OUT" ]; then
  echo "[e2e] FAIL: no clip produced (see $RUN/request.log)"
  exit 1
fi

echo "[e2e] --- artifact verification ---"
BYTES=$(stat -c %s "$OUT"); SHA=$(sha256sum "$OUT" | cut -d' ' -f1)
printf "  clip:   %s\n  bytes:  %s\n  sha256: %s\n" "$OUT" "$BYTES" "$SHA"
if [ -n "${H3_EXPECT_BYTES:-}" ]; then
  if [ "$BYTES" = "$H3_EXPECT_BYTES" ] && { [ -z "${H3_EXPECT_SHA:-}" ] || [ "$SHA" = "$H3_EXPECT_SHA" ]; }; then
    echo "  RENDER: byte-identical to the recorded reference -> PASS"
  else
    echo "  RENDER: differs from the recorded reference (expected $H3_EXPECT_BYTES, got $BYTES)"
  fi
fi
# A response body is not an artifact. curl --fail-with-body writes the HTTP error body straight to
# $OUT, so "the file exists and is non-empty" is not enough: require a decodable video stream and a
# real duration before this run may call itself a pass.
if ! ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$OUT" \
     2>"$RUN/ffprobe.err" | grep -q .; then
  echo "[e2e] FAIL: the artifact is not a decodable video"
  echo "  first 200 bytes: $(head -c 200 "$OUT")"
  tail -3 "$RUN/ffprobe.err" | sed 's/^/  /'
  exit 1
fi
VSTREAM=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$OUT" 2>/dev/null | head -1)
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" 2>/dev/null | head -1)
if [ -z "$DUR" ] || [ "${DUR%%.*}" -lt 1 ]; then
  echo "[e2e] FAIL: the artifact reports duration='$DUR'"
  exit 1
fi
echo "  video stream: $VSTREAM, ${DUR}s"
ffprobe -v error -show_entries stream=codec_name,width,height -show_entries format=duration \
  -of default=noprint_wrappers=1 "$OUT" | sed 's/^/  /' | tee "$RUN/ffprobe.txt"

# A mosaic/corrupt clip still decodes as some frames; require first and last to differ.
n=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$OUT" 2>/dev/null | tr -d '\n')
if [ -n "$n" ] && [ "$n" -gt 1 ]; then
  ffmpeg -v error -i "$OUT" -vf "select=eq(n\,0)+eq(n\,$((n-1)))" -vsync 0 -frames:v 2 "$RUN/frame_%02d.png" 2>/dev/null
  if [ -f "$RUN/frame_01.png" ] && [ -f "$RUN/frame_02.png" ]; then
    a=$(sha256sum "$RUN/frame_01.png" | cut -c1-16); b=$(sha256sum "$RUN/frame_02.png" | cut -c1-16)
    [ "$a" != "$b" ] && echo "  frames: differ ($a vs $b) -> motion present" \
                     || echo "  frames: IDENTICAL - possible still/mosaic"
  fi
fi

echo "[e2e] PASS (artifact written). receipt: $RUN"
