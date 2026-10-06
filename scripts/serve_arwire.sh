#!/bin/bash
# serve_arwire.sh - start the MiniMax-H3 lane.
#
#   MODEL   served weights (resolved by h3_lane_env.sh)
#   H3_QUANT / H3_STEPS / H3_WEIGHTS_SOURCE / H3_TASK_TYPE   see h3_lane_env.sh
#   H3_TOPOLOGY=tp2usp2|usp4|tp4|tp1usp2   default tp2usp2
#   H3_CACHE_DIR=<dir>    compile caches   default /var/cache/h3
#
# Attention backends: body SOL_ATTN, refiner B12X, everything else B12X.
set -euo pipefail

export PATH=/opt/venv/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-14400}

H3_CACHE_DIR=${H3_CACHE_DIR:-/var/cache/h3}

# LIL NCCL 2.31.2 ships with the runtime and provides the sm120 devComm API the fused all-to-all
# permute needs. Without it the lane silently uses the system NCCL and that permute fails. Must be
# set before torch is imported.
export H3_NCCL_IMAGE=${H3_NCCL_IMAGE:-1}
if [ "$H3_NCCL_IMAGE" = "1" ] && [ -f "/opt/venv/lib/python3.12/site-packages/local_inference_nccl/lib/libnccl.so.2.31.2" ]; then
  export LD_PRELOAD="/opt/venv/lib/python3.12/site-packages/local_inference_nccl/lib/libnccl.so.2.31.2${LD_PRELOAD:+:$LD_PRELOAD}"
  export VLLM_NCCL_SO_PATH="/opt/venv/lib/python3.12/site-packages/local_inference_nccl/lib/libnccl.so.2.31.2"
fi

# Boot-time arms: chosen while the model loads, so changing one is a restart.
# 0 = NCCL fallback all-to-all. The library reads this from parallel_config, so the knob
# is translated into the official serve flag rather than an H3 env var.
export H3_A2A_PERMUTE=${H3_A2A_PERMUTE:-0}
export H3_A2A_WIRE_BUFCACHE=${H3_A2A_WIRE_BUFCACHE:-0}

source "$(dirname "${BASH_SOURCE[0]}")/h3_lane_env.sh"
if [ "${H3_PRINT:-0}" = "1" ]; then exit 0; fi

# TP x USP decomposition. tp2usp2 is the standing default.
export H3_TOPOLOGY=${H3_TOPOLOGY:-tp2usp2}
case "$H3_TOPOLOGY" in
  usp4|tp1usp4|usp2x4) H3_TP=1; H3_USP=4; H3_ENC_TP=${H3_ENC_TP:-1} ;;
  tp4|tp4usp1|nousp)   H3_TP=4; H3_USP=1; H3_ENC_TP=${H3_ENC_TP:-4} ;;
  tp2usp1|2gpu)        H3_TP=2; H3_USP=1; H3_ENC_TP=${H3_ENC_TP:-2} ;;
  tp1usp2|tp1ag2)      H3_TP=1; H3_USP=2; H3_ENC_TP=${H3_ENC_TP:-1} ;;
  tp2usp2|*|"")        H3_TP=2; H3_USP=2; H3_ENC_TP=${H3_ENC_TP:-2} ;;
esac
echo "[lane] topology=$H3_TOPOLOGY -> tp=$H3_TP usp=$H3_USP text_encoder_tp=$H3_ENC_TP"

# Optional VAE decoder torch.compile. Measured -0.44 to -0.51 s/clip at 480 W + OC, but the clip is
# not byte-identical (732 B off), so it changes decode numerics and needs a visual check.
export H3_VAE_COMPILE=${H3_VAE_COMPILE:-0}
if [ "$H3_VAE_COMPILE" = "1" ]; then
  export MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE=1
  export MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE=1
  export MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE_MODE=${H3_VAE_COMPILE_MODE:-default}
  export MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE_MODE=${H3_VAE_COMPILE_MODE:-default}
  export MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE_FULLGRAPH=${H3_VAE_COMPILE_FULLGRAPH:-0}
fi

# Wire transports. Each arm is selected only by its own variable, fixed for the whole boot; the
# defaults are the stock bf16 path, so a boot with nothing set runs the unchanged pipeline.
export H3_A2A_QKV_BATCH=${H3_A2A_QKV_BATCH:-0}
export H3_A2A_WIRE=${H3_A2A_WIRE:-bf16}
export H3_AR_WIRE=${H3_AR_WIRE:-bf16}
export VLLM_ENABLE_PCIE_ALLREDUCE=${VLLM_ENABLE_PCIE_ALLREDUCE:-1}

export H3_FUSE_LORA="$H3_LORA"
export H3_STEP_PROFILE=${H3_STEP_PROFILE:-0}

# Persistent compile caches: mount a volume here or every start recompiles from scratch.
export TRITON_CACHE_DIR=$H3_CACHE_DIR/triton
export TORCHINDUCTOR_CACHE_DIR=$H3_CACHE_DIR/inductor
export XDG_CACHE_HOME=$H3_CACHE_DIR/xdg
export VLLM_CACHE_ROOT=$H3_CACHE_DIR/vllm
export CUDA_CACHE_PATH=$H3_CACHE_DIR/cuda
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$CUDA_CACHE_PATH"

export SOL_ATTN_TAU=${SOL_ATTN_TAU:-1.0}
export SOL_ATTN_CORRECTNESS_GATE=${SOL_ATTN_GATE:-0}

ATTN='{"default": {"backend": "B12X"},
       "per_role": {"self": {"backend": "SOL_ATTN"},
                    "minimax_h3.token_refiner": {"backend": "B12X"}}}'

echo "=== MiniMax-H3 lane: body=SOL_ATTN refiner=B12X task=$H3_TASK_TYPE ==="
echo "    torch=$(python -c 'import torch;print(torch.__version__)') cuda=$(python -c 'import torch;print(torch.version.cuda)')"
echo "    a2a wire=$H3_A2A_WIRE ar wire=$H3_AR_WIRE qkv_batch=$H3_A2A_QKV_BATCH linear=${H3_LINEAR_ARM:-mxfp8}"

# The turbo adapter family must match the partition (ref2v for Ref2VA, fl2v for FL2VA); the model
# rejects a mismatch at boot. Empty means no adapter, which the served partition may not support.
LORA_ARGS=()
[ -n "${H3_LORA:-}" ] && LORA_ARGS=(--lora-path "$H3_LORA")
PERMUTE_ARGS=()
[ "$H3_A2A_PERMUTE" = "1" ] && PERMUTE_ARGS=(--ulysses-a2a-permute)

exec vllm serve "$MODEL" \
  --omni --task-type "$H3_TASK_TYPE" \
  "${LORA_ARGS[@]}" \
  --host 0.0.0.0 --port "${PORT:-8000}" --trust-remote-code \
  --enable-sleep-mode \
  --num-gpus $((H3_TP * H3_USP)) --tensor-parallel-size $H3_TP --usp $H3_USP --ring 1 \
  --text-encoder-tp-size $H3_ENC_TP \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
  "${PERMUTE_ARGS[@]}" \
  --diffusion-attention-config "$ATTN"
