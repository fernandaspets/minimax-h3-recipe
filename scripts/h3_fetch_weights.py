#!/usr/bin/env python3
"""Resolve MiniMax-H3 weights from Hugging Face for one task type.

Mirrors the known-good local layout rather than inventing one:

    <cache>/MiniMax-H3/               snapshot of MiniMaxAI/MiniMax-H3
    <cache>/<partition>/              the directory to serve
        model_index.json              partition config (shipped in this bundle)
        <PARTITION_DIR> -> ../MiniMax-H3/<PARTITION_DIR>

t2va and fl2va are both served by the FL2VA partition; ref2va by Ref2VA.

Prints the directory to serve on stdout, so the caller can do MODEL=$(...).

What it deliberately does NOT do: fetch identity RefMods. Those are private working assets
and are never published; the caller leaves H3_REFMOD_PATHS empty and says so.

The full base repo is large (order 250+ GB), so this is not exercised in CI; --dry-run prints
the plan and the exact repositories/patterns without downloading anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MODEL_REPO = os.environ.get("H3_HF_MODEL_REPO", "MiniMaxAI/MiniMax-H3")
LORA_REPO = os.environ.get("H3_HF_LORA_REPO", "lightx2v/Minimax-h3-Turbo")

# Task type -> partition (which weights) and the partition's directory name in the base repo.
TASK_PARTITION = {"ref2va": "ref2va", "t2va": "fl2va", "fl2va": "fl2va"}
PARTITION_DIR = {"ref2va": "Ref2VA", "fl2va": "FL2VA"}

# The turbo adapters live in one HF repo under different filenames. The family must match the
# partition (ref2v for Ref2VA, fl2v for FL2VA); the model rejects a mismatch at boot. The FL2VA
# 8-step v1.0 768p entry is the one upstream's own Studio runs.
LORA_BY_ARM = {
    ("ref2va", "2"): "minimax_h3_ref2v_turbo_2step_v1.0_bf16.safetensors",
    ("ref2va", "4"): "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
    ("ref2va", "8"): "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors",
    ("fl2va", "4"): "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors",
    ("fl2va", "8"): "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors",
}
LORA_OVERRIDE = os.environ.get("H3_HF_LORA", "")

HERE = Path(__file__).resolve().parent


def plan(steps: str, task: str, cache: Path) -> dict:
    partition = TASK_PARTITION[task]
    lora_file = LORA_OVERRIDE or LORA_BY_ARM.get((partition, steps))
    if not lora_file:
        raise SystemExit(f"no turbo adapter is published for partition={partition} steps={steps}")
    return {
        "model_repo": MODEL_REPO,
        "lora_repo": LORA_REPO,
        "lora_file": lora_file,
        "task": task,
        "partition": partition,
        "cache": str(cache),
        "snapshot": str(cache / "MiniMax-H3"),
        "wrapper": str(cache / partition),
        "serve_dir": str(cache / partition),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=os.environ.get("H3_STEPS", "4"), choices=["2", "4", "8"])
    ap.add_argument("--task", default=os.environ.get("H3_TASK_TYPE", "ref2va"),
                    choices=sorted(TASK_PARTITION))
    ap.add_argument("--cache", default=os.environ.get(
        "H3_HF_CACHE", str(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "h3")))
    ap.add_argument("--dry-run", action="store_true", default=os.environ.get("H3_HF_DRY") == "1")
    ap.add_argument("--print-lora", action="store_true", help="print the LoRA path for the arm")
    ap.add_argument("--verify-only", action="store_true", help="check an existing cache, do not download")
    args = ap.parse_args()

    cache = Path(args.cache)
    p = plan(args.steps, args.task, cache)
    partition = p["partition"]
    lora_path = cache / "loras" / p["lora_file"]

    if args.dry_run:
        print(json.dumps({"dry_run": True, **p, "lora_path": str(lora_path),
                          "refmods": "not fetched (private)"}, indent=2), file=sys.stderr)
        print(p["serve_dir"])
        return 0

    if args.verify_only:
        missing = [x for x in (cache / "MiniMax-H3", p["wrapper"], lora_path) if not x.exists()]
        if missing:
            print("missing: " + ", ".join(map(str, missing)), file=sys.stderr)
            return 1
        print(lora_path if args.print_lora else p["serve_dir"])
        return 0

    wrapper_config = HERE / partition / "model_index.json"
    if not wrapper_config.is_file():
        print(f"missing shipped wrapper config: {wrapper_config}", file=sys.stderr)
        return 1

    from huggingface_hub import hf_hub_download, snapshot_download

    # 1. base model
    snapshot_download(repo_id=MODEL_REPO, local_dir=str(cache / "MiniMax-H3"))

    # 2. wrapper, laid out exactly like the local tree
    wrapper = cache / partition
    wrapper.mkdir(parents=True, exist_ok=True)
    (wrapper / "model_index.json").write_text(wrapper_config.read_text())
    link = wrapper / PARTITION_DIR[partition]
    target = Path("..") / "MiniMax-H3" / PARTITION_DIR[partition]
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)

    # 3. the turbo adapter for this arm
    hf_hub_download(repo_id=LORA_REPO, filename=p["lora_file"], local_dir=str(cache / "loras"))

    print(lora_path if args.print_lora else p["serve_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
