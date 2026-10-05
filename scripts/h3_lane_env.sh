#!/bin/bash
# h3_lane_env.sh - resolve the lane's arm. SOURCE this; do not execute it.
#
#   H3_QUANT=mxfp8|hybrid|nvfp4     default mxfp8
#   H3_STEPS=2|4|8                  default 4
#   H3_WEIGHTS_SOURCE=local|hf      default local
#
#   H3_TASK_TYPE=ref2va|t2va|fl2va  default ref2va   (selects the served partition)
#   H3_WIRE=bf16|int8              default bf16     (int8 a2a + all-reduce transport)
#   H3_STATE_DIR=<dir>              default /var/lib/h3      (policy + control files)
#   H3_MODEL_DIR=<dir>              default /models/MiniMaxAI/MiniMax-H3
#   H3_LORA=<file>                  overrides the per-step default; the adapter family must match
#                                   the partition (ref2v for ref2va, fl2v for t2va/fl2va)
#   H3_LORA_DIR=<dir>               default /models/h3/turbo
#   H3_REFMOD_PATHS=<a.safetensors:b.safetensors>
#                                   unset by default; identity adapters are optional
#   H3_PRINT=1                      resolve and print, then exit without starting anything
#
# An unknown value is an error, never a silent fallback.

set -euo pipefail

H3_QUANT=${H3_QUANT:-mxfp8}
H3_STEPS=${H3_STEPS:-4}
H3_WEIGHTS_SOURCE=${H3_WEIGHTS_SOURCE:-local}
H3_TASK_TYPE=${H3_TASK_TYPE:-ref2va}
H3_STATE_DIR=${H3_STATE_DIR:-/var/lib/h3}
H3_MODEL_DIR=${H3_MODEL_DIR:-/models/MiniMaxAI/MiniMax-H3}
H3_LORA_DIR=${H3_LORA_DIR:-/models/h3/turbo}

_h3_die() { echo "[h3_lane_env] ERROR: $*" >&2; exit 2; }

# ------------------------------------------------------------------ task -> partition
# t2va and fl2va are both served by the FL2VA partition; ref2va by Ref2VA.
case "$H3_TASK_TYPE" in
  ref2va)      H3_PARTITION=ref2va ;;
  t2va|fl2va)  H3_PARTITION=fl2va ;;
  *) _h3_die "unknown H3_TASK_TYPE='$H3_TASK_TYPE' (want ref2va|t2va|fl2va)" ;;
esac
export H3_PARTITION

# ------------------------------------------------------------------ wire transport
# The int8 transports are opt-in and lossy; bf16 is the stock vLLM path. One knob writes both
# control files, so the wire cannot half-apply.
H3_WIRE=${H3_WIRE:-bf16}
case "$H3_WIRE" in
  bf16|int8) ;;
  *) _h3_die "unknown H3_WIRE='$H3_WIRE' (want bf16|int8)" ;;
esac
mkdir -p "$H3_STATE_DIR/control"
printf '%s\n' "$H3_WIRE" > "$H3_STATE_DIR/control/A2A_WIRE_MODE"
printf '%s\n' "$H3_WIRE" > "$H3_STATE_DIR/control/AR_WIRE_MODE"

# ------------------------------------------------------------------ quantisation arm
# Per-role policy, one line per role: mlp / attn / refiner.
case "$H3_QUANT" in
  mxfp8)
    export VLLM_OMNI_DIT_MXFP8=1 VLLM_OMNI_DIT_NVFP4=0
    _h3_policy="mlp mxfp8
attn mxfp8
refiner bf16"
    ;;
  hybrid)
    export VLLM_OMNI_DIT_MXFP8=1 VLLM_OMNI_DIT_NVFP4=1
    _h3_policy="mlp nvfp4
attn mxfp8
refiner bf16"
    ;;
  nvfp4)
    export VLLM_OMNI_DIT_MXFP8=0 VLLM_OMNI_DIT_NVFP4=1
    _h3_policy="mlp nvfp4
attn nvfp4
refiner bf16"
    ;;
  *) _h3_die "unknown H3_QUANT='$H3_QUANT' (want mxfp8|hybrid|nvfp4)" ;;
esac

# The env gates only enable the classes; the policy file is what the model resolves. Both must agree.
_h3_policy_file="$H3_STATE_DIR/arms/$H3_QUANT/QUANT_POLICY"
mkdir -p "$(dirname "$_h3_policy_file")"
printf '%s\n' "$_h3_policy" > "$_h3_policy_file"
export H3_QUANT_POLICY_CONTROL="$_h3_policy_file"

# ------------------------------------------------------------------ step arm
# The turbo adapters declare a task family and bind only to the matching partition: a ref2v adapter
# on the FL2VA partition (or an fl2v adapter on Ref2VA) aborts the boot inside lora.py. Select the
# adapter by partition here, so a mismatch fails at arm resolution instead of ten minutes into a load.
# The FL2VA 8-step v1.0 768p adapter is the one upstream's own Studio runs; see WEIGHTS.md.
case "$H3_STEPS" in
  2|4|8) ;;
  *) _h3_die "unknown H3_STEPS='$H3_STEPS' (want 2|4|8)" ;;
esac
if [ -z "${H3_LORA:-}" ]; then
  case "$H3_PARTITION:$H3_STEPS" in
    ref2va:2) H3_LORA=$H3_LORA_DIR/minimax_h3_ref2v_turbo_2step_v1.0_bf16.safetensors ;;
    ref2va:4) H3_LORA=$H3_LORA_DIR/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors ;;
    ref2va:8) H3_LORA=$H3_LORA_DIR/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors ;;
    fl2va:2)  _h3_die "no 2-step FL2VA turbo adapter is published; use H3_STEPS=4 or 8" ;;
    fl2va:4)  H3_LORA=$H3_LORA_DIR/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors ;;
    fl2va:8)  H3_LORA=$H3_LORA_DIR/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors ;;
  esac
fi
if [ -n "$H3_LORA" ] && [ ! -f "$H3_LORA" ]; then
  _h3_die "LoRA not found: $H3_LORA (download it, or set H3_LORA to one that exists; see WEIGHTS.md)"
fi
export H3_LORA
export H3_REQUEST_STEPS="$H3_STEPS"

# ------------------------------------------------------------------ weights
case "$H3_WEIGHTS_SOURCE" in
  local)
    MODEL=${MODEL:-$H3_MODEL_DIR-$H3_PARTITION}
    ;;
  hf)
    if [ "${H3_HF_DRY:-0}" = "1" ]; then
      MODEL="<hf:${H3_HF_MODEL_REPO:-MiniMaxAI/MiniMax-H3}>"
    else
      MODEL=$(python3 "$(dirname "${BASH_SOURCE[0]}")/h3_fetch_weights.py" --steps "$H3_STEPS")
    fi
    ;;
  *) _h3_die "unknown H3_WEIGHTS_SOURCE='$H3_WEIGHTS_SOURCE' (want local|hf)" ;;
esac
export MODEL
export H3_REFMOD_NORMALIZE=${H3_REFMOD_NORMALIZE:-1}
# H3_REFMOD_PATHS intentionally defaults to empty: identity adapters are a separate, optional input.

# ------------------------------------------------------------------ report
echo "[h3_lane_env] arm: quant=$H3_QUANT steps=$H3_STEPS wire=$H3_WIRE weights=$H3_WEIGHTS_SOURCE task=$H3_TASK_TYPE (partition=$H3_PARTITION)"
echo "[h3_lane_env]   policy      : $(tr '\n' ' ' < "$_h3_policy_file")"
echo "[h3_lane_env]   dit gates   : MXFP8=$VLLM_OMNI_DIT_MXFP8 NVFP4=$VLLM_OMNI_DIT_NVFP4"
echo "[h3_lane_env]   lora        : $H3_LORA"
echo "[h3_lane_env]   model       : $MODEL"
echo "[h3_lane_env]   refmods     : ${H3_REFMOD_PATHS:-<none>}"
echo "[h3_lane_env]   request steps: $H3_REQUEST_STEPS"

if [ "${H3_PRINT:-0}" = "1" ]; then
  echo "[h3_lane_env] H3_PRINT=1 -> caller should stop here"
  H3_PRINT_DONE=1
fi
