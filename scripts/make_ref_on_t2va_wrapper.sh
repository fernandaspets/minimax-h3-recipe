#!/bin/bash
# make_ref_on_t2va_wrapper.sh - build the "reference pipeline on the t2va weights" wrapper.
#
#   H3_MODEL_DIR=/models/MiniMaxAI/MiniMax-H3   a root holding FL2VA/ and Ref2VA/
#   H3_REF_WRAPPER=<path>                       output (default <H3_MODEL_DIR>-refpipe-on-t2va)
#
# What it builds: a model directory whose Ref2VA/ carries the Ref2VA *manifest* (partition ref2va,
# task ref2va - which is what the ref2v turbo adapter binds to) and, symlinked one level down, the
# FL2VA partition's *components* (transformer, VAEs, text encoder). Serving this directory with
# --task-type ref2va therefore runs the reference pipeline on the t2va weights, which is the route
# this studio uses. Nothing is copied and the original partitions are not modified.
#
# The symlinks are RELATIVE on purpose: a container mount rewrites the path root, so a host-absolute
# link resolves to a path that does not exist inside the container, and the boot then fails with a
# bare "Orchestrator initialization failed:".
#
# Idempotent: re-running rebuilds the wrapper from scratch.
set -euo pipefail

H3_MODEL_DIR=${H3_MODEL_DIR:-/models/MiniMaxAI/MiniMax-H3}
H3_REF_WRAPPER=${H3_REF_WRAPPER:-$H3_MODEL_DIR-refpipe-on-t2va}
SRC=$H3_MODEL_DIR/FL2VA
REF_WRAPPER_SRC=$H3_MODEL_DIR-ref2va

_die() { echo "[ref-on-t2va] ERROR: $*" >&2; exit 2; }

[ -d "$SRC" ] || _die "t2va partition not found: $SRC"
[ -f "$SRC/model_index.json" ] || _die "t2va partition has no model_index.json: $SRC"
[ -f "$H3_MODEL_DIR/Ref2VA/model_index.json" ] || _die "Ref2VA partition not found: $H3_MODEL_DIR/Ref2VA"

# The outer wrapper (the directory that is served) needs a registered pipeline index. Reuse an
# existing single-partition wrapper's when there is one, so the copied index cannot drift from
# what this checkout ships.
if [ -f "$REF_WRAPPER_SRC/model_index.json" ]; then
  OUTER_INDEX=$REF_WRAPPER_SRC/model_index.json
else
  OUTER_INDEX=$H3_MODEL_DIR/Ref2VA/model_index.json
  echo "[ref-on-t2va] WARN: no $REF_WRAPPER_SRC/model_index.json; using the partition's own index"
fi

rm -rf "$H3_REF_WRAPPER"
mkdir -p "$H3_REF_WRAPPER/Ref2VA"
cp "$OUTER_INDEX" "$H3_REF_WRAPPER/model_index.json"

for _comp in transformer video_vae audio_vae text_encoder tokenizer processor; do
  [ -e "$SRC/$_comp" ] || { echo "[ref-on-t2va] skip (absent in the t2va partition): $_comp"; continue; }
  ln -sfn "../../MiniMax-H3/FL2VA/$_comp" "$H3_REF_WRAPPER/Ref2VA/$_comp"
done

# The manifest is the only generated file: partition ref2va so the ref2v adapter binds, and the
# task list is exactly what the reference route needs.
python3 - "$SRC/model_index.json" "$H3_REF_WRAPPER/Ref2VA/model_index.json" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
rel = d.setdefault("_minimax_h3", {})
rel["schema_version"] = rel.get("schema_version", 1)
rel["partition"] = "ref2va"
rel["tasks"] = ["ref2va"]
rel.setdefault("task_aliases", {})
rel.setdefault("sigma_shift_scales", {"video": 12.0, "audio": 3.0})
open(dst, "w").write(json.dumps(d, indent=2) + "\n")
print("[ref-on-t2va] manifest: partition=ref2va tasks=['ref2va'] sigma_shift_scales=%s" % rel["sigma_shift_scales"])
PY

echo "[ref-on-t2va] wrapper: $H3_REF_WRAPPER"
echo "[ref-on-t2va]   transformer -> $(readlink -f "$H3_REF_WRAPPER/Ref2VA/transformer")"
for _link in "$H3_REF_WRAPPER"/Ref2VA/*; do
  [ -L "$_link" ] || continue
  [ -e "$_link" ] || _die "broken link: $_link"
done
echo "[ref-on-t2va] OK - serve it with: MODEL=$H3_REF_WRAPPER H3_TASK_TYPE=ref2va (or H3_REF_ON_T2VA=1)"
