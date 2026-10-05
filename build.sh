#!/bin/bash
# Build the lane image from public sources only.
#   build.sh [tag]
# Every input is pinned below; nothing is read from this machine.
set -euo pipefail
cd "$(dirname "$0")"
TAG=${1:-h3-repro}
# The revisions are the heads of the online pull requests, resolved here so the pins can never
# drift from what was reviewed:
#   vllm-project/vllm-omni #8487  B12X/SOL_ATTN backends, opt-in MXFP8/NVFP4, the H3 modules
#   vllm-project/vllm-omni #8486  the communicator binds to the rank device
#   local-inference-lab/b12x #480  var-length attention with per-head block lists
# Point *_REPO / *_REF at upstream once they merge.
VLLM_OMNI_REPO=${VLLM_OMNI_REPO:-https://github.com/fernandaspets/vllm-omni}
VLLM_OMNI_REF=${VLLM_OMNI_REF:-refs/heads/h3/features}
B12X_REPO=${B12X_REPO:-https://github.com/fernandaspets/b12x}
B12X_REF=${B12X_REF:-refs/heads/feat/video-block-sparse}
VLLM_OMNI_SHA=$(git ls-remote "$VLLM_OMNI_REPO" "$VLLM_OMNI_REF" | cut -f1)
B12X_SHA=$(git ls-remote "$B12X_REPO" "$B12X_REF" | cut -f1)
if [ -z "$VLLM_OMNI_SHA" ] || [ -z "$B12X_SHA" ]; then
  echo "[build] FAIL: cannot resolve the online revisions:"
  echo "[build]   $VLLM_OMNI_REPO $VLLM_OMNI_REF"
  echo "[build]   $B12X_REPO $B12X_REF"
  exit 1
fi
echo "[build] vllm-omni $VLLM_OMNI_SHA"
echo "[build] b12x      $B12X_SHA"
export DOCKER_BUILDKIT=1
docker build \
  --build-arg "VLLM_OMNI_REPO=$VLLM_OMNI_REPO" \
  --build-arg "VLLM_OMNI_SHA=$VLLM_OMNI_SHA" \
  --build-arg "B12X_REPO=$B12X_REPO" \
  --build-arg "B12X_SHA=$B12X_SHA" \
  -t "local/h3kk:$TAG" .
docker run --rm --entrypoint /opt/venv/bin/python "local/h3kk:$TAG" -c "
import importlib.metadata as m
for p in ('vllm','b12x','vllm_omni'):
    print('   ', p, m.version(p))
"
