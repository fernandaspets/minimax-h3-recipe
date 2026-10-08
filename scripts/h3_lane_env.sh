#!/bin/bash
# h3_lane_env.sh - resolve the lane's arm. SOURCE this; do not execute it.
#
#   H3_QUANT=mxfp8|hybrid|nvfp4     default mxfp8
#   H3_STEPS=2|4|8                  default 4
#   H3_WEIGHTS_SOURCE=local|hf      default local
#
#   H3_REF_ON_T2VA=1               the studio's route: the reference pipeline served on the
#                                   t2va weights. Forces task type ref2va (the ref2v turbo
#                                   adapter binds to it) and MODEL=<root>-refpipe-on-t2va,
#                                   built by scripts/make_ref_on_t2va_wrapper.sh.
#   H3_TASK_TYPE=ref2va|t2va|fl2va|combined  default ref2va   (selects the served partition;
#                                   combined = the regular FL2VA lane that also serves the
#                                   Ref2VA transformer for reference requests)
#   H3_WIRE=bf16|int8              default bf16     (int8 a2a + all-reduce transport)
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
H3_MODEL_DIR=${H3_MODEL_DIR:-/models/MiniMaxAI/MiniMax-H3}
H3_LORA_DIR=${H3_LORA_DIR:-/models/h3/turbo}

_h3_die() { echo "[h3_lane_env] ERROR: $*" >&2; exit 2; }

# ------------------------------------------------------------------ task -> partition
# t2va and fl2va are both served by the FL2VA partition; ref2va by Ref2VA.
# Reference conditioning on the t2va weights: the ref2v adapter is accepted only by a server
# whose task type is ref2va, while every component served is the t2va partition. Choosing the
# route here keeps the wrapper path, the adapter family and the task type from disagreeing.
if [ "${H3_REF_ON_T2VA:-0}" = "1" ]; then
  H3_TASK_TYPE=ref2va
fi
case "$H3_TASK_TYPE" in
  ref2va)          H3_PARTITION=ref2va ;;
  t2va|fl2va)      H3_PARTITION=fl2va ;;
  # combined: the regular (FL2VA) partition plus the Ref2VA transformer, so one boot takes
  # reference requests AND the normal tasks; it resolves Ref2VA/ and FL2VA/ inside one root.
  combined)        H3_PARTITION=combined ;;
  *) _h3_die "unknown H3_TASK_TYPE='$H3_TASK_TYPE' (want ref2va|t2va|fl2va|combined)" ;;
esac
export H3_PARTITION

# ------------------------------------------------------------------ wire transport
# The int8 transports are opt-in and lossy; bf16 is the stock vLLM path. One knob sets both
# transports, so the wire cannot half-apply.
#
# These env vars are what the model reads (a2a_wire._wire / ar_wire._wire). The control-file
# plane that used to carry this choice was removed, so H3_WIRE must export them: writing a file
# to the old state dir would now be silently inert and an int8 arm would quietly run bf16.
H3_WIRE=${H3_WIRE:-bf16}
case "$H3_WIRE" in
  bf16|int8) ;;
  *) _h3_die "unknown H3_WIRE='$H3_WIRE' (want bf16|int8)" ;;
esac
export H3_A2A_WIRE=${H3_A2A_WIRE:-$H3_WIRE}
export H3_AR_WIRE=${H3_AR_WIRE:-$H3_WIRE}

# ------------------------------------------------------------------ quantisation arm
# Per-role policy, one line per role: mlp / attn / refiner.
case "$H3_QUANT" in
  mxfp8)
    export VLLM_OMNI_DIT_MXFP8=1 VLLM_OMNI_DIT_NVFP4=0
    ;;
  hybrid)
    export VLLM_OMNI_DIT_MXFP8=1 VLLM_OMNI_DIT_NVFP4=1
    ;;
  nvfp4)
    export VLLM_OMNI_DIT_MXFP8=0 VLLM_OMNI_DIT_NVFP4=1
    ;;
  *) _h3_die "unknown H3_QUANT='$H3_QUANT' (want mxfp8|hybrid|nvfp4)" ;;
esac

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
    ref2va:2) H3_LORA=$H3_LORA_DIR/minimax_h3_ref2v_turbo_2step_v1.0_bf16.safetensors ;;
    ref2va:4) H3_LORA=$H3_LORA_DIR/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors ;;
    ref2va:8) H3_LORA=$H3_LORA_DIR/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors ;;
    # combined serves the reference task from the Ref2VA transformer, so it carries the ref2v
    # family: a reference request on a combined lane is what this arm exists for.
    combined:2) H3_LORA=$H3_LORA_DIR/minimax_h3_ref2v_turbo_2step_v1.0_bf16.safetensors ;;
    combined:4) H3_LORA=$H3_LORA_DIR/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors ;;
    combined:8) H3_LORA=$H3_LORA_DIR/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors ;;
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
    if [ "${H3_REF_ON_T2VA:-0}" = "1" ]; then
      MODEL=${MODEL:-$H3_MODEL_DIR-refpipe-on-t2va}
      [ -f "$MODEL/Ref2VA/model_index.json" ] || _h3_die \
        "ref-on-t2va wrapper missing or incomplete: $MODEL/Ref2VA/model_index.json (build it with scripts/make_ref_on_t2va_wrapper.sh)"
    else
      # Every partition is read from the sibling <root>-<partition> directory: the wrapper holds
      # the partition as a subdirectory and must NOT carry modular_model_index.json, otherwise the
      # pipeline takes the release's modular path and demands a fastvideo_inference.json the local
      # layout does not have.
      MODEL=${MODEL:-$H3_MODEL_DIR-$H3_PARTITION}
    fi
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
[ "${H3_REF_ON_T2VA:-0}" = "1" ] && echo "[h3_lane_env] route: reference pipeline on the t2va weights"
echo "[h3_lane_env]   dit gates   : MXFP8=$VLLM_OMNI_DIT_MXFP8 NVFP4=$VLLM_OMNI_DIT_NVFP4"
echo "[h3_lane_env]   lora        : $H3_LORA"
echo "[h3_lane_env]   model       : $MODEL"
echo "[h3_lane_env]   refmods     : ${H3_REFMOD_PATHS:-<none>}"
echo "[h3_lane_env]   request steps: $H3_REQUEST_STEPS"

if [ "${H3_PRINT:-0}" = "1" ]; then
  echo "[h3_lane_env] H3_PRINT=1 -> caller should stop here"
  H3_PRINT_DONE=1
fi
