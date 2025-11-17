# qk_stats.py
# ---------------------------------------
# Analyze QK cache saved by LlamaQKCollector
# Produces:
#   1. Per-step head stats
#   2. Aggregated (mean over steps) head stats per layer
#   3. Saves both in JSON
# ---------------------------------------

import torch
import json
from tqdm import tqdm
import argparse
import os


# ---------------------------------------------------------
# Try to infer (num_heads, head_dim) robustly
# ---------------------------------------------------------
def infer_head_shapes(hd):
    preferred_head_dims = [256, 128, 64, 32]
    for d in preferred_head_dims:
        if hd % d == 0:
            return hd // d, d

    for d in range(1, hd + 1):
        if hd % d == 0:
            return hd // d, d

    raise RuntimeError(f"Could not infer head_dim/num_heads from hd={hd}")


# ---------------------------------------------------------
# Compute head stats for one (Q, K)
# ---------------------------------------------------------
def compute_head_stats(q, k):
    B, L, hd = q.shape
    _, _, hd_k = k.shape

    num_h_q, head_dim_q = 24,128
    num_h_k, head_dim_k = 8,128
    # num_h_q, head_dim_q = infer_head_shapes(hd)
    # num_h_k, head_dim_k = infer_head_shapes(hd_k)

    qh = q.view(B, L, num_h_q, head_dim_q).squeeze(0).transpose(0, 1)
    kh = k.view(B, L, num_h_k, head_dim_k).squeeze(0).transpose(0, 1)

    if num_h_k != num_h_q:
        repeat_factor = num_h_q // num_h_k
        kh = kh.repeat_interleave(repeat_factor, dim=0)

    attn = torch.matmul(qh, kh.transpose(1, 2))
    attn_mean = attn.abs().mean(dim=(1, 2))  # [H]
    head_order = attn_mean.argsort(descending=True).tolist()

    return {
        "num_heads": num_h_q,
        "head_dim": head_dim_q,
        "attn_mean": attn_mean.tolist(),
        "sorted_heads": head_order,
    }


# ---------------------------------------------------------
# Analyze full QK cache
# ---------------------------------------------------------
def analyze_qk_cache(path):
    print(f"[Loading] {path}")
    qk_cache = torch.load(path)

    # keys are numeric strings
    steps = sorted(qk_cache.keys(), key=lambda x: int(x))
    all_stats = {}

    print(f"[Info] Total steps: {len(steps)}")

    # For aggregation
    layer_accumulator = {}  # layer → list of per-step attn_mean tensors

    for step in tqdm(steps, desc="Analyzing steps"):
        step_dict = qk_cache[step]
        layer_stats = {}

        for layer_idx, layer_data in step_dict.items():
            q = layer_data["q"]
            k = layer_data["k"]

            try:
                stats = compute_head_stats(q, k)
            except Exception as e:
                stats = {"error": str(e)}
                layer_stats[str(layer_idx)] = stats
                continue

            layer_stats[str(layer_idx)] = stats

            # accumulate for aggregation
            if "error" not in stats:
                if layer_idx not in layer_accumulator:
                    layer_accumulator[layer_idx] = []
                layer_accumulator[layer_idx].append(torch.tensor(stats["attn_mean"]))

        all_stats[step] = layer_stats

    # ---------------------------------------------------------
    # Aggregation: compute mean attention magnitude over all steps
    # ---------------------------------------------------------
    aggregated_stats = {}
    for layer_idx, attn_list in layer_accumulator.items():

        attn_tensor = torch.stack(attn_list)  # [num_steps, H]
        mean_attn = attn_tensor.mean(dim=0)   # [H]

        aggregated_stats[str(layer_idx)] = {
            "attn_mean": mean_attn.tolist(),
            "sorted_heads": mean_attn.argsort(descending=True).tolist(),
            "num_steps": len(attn_list),
        }

    return {
        "per_step": all_stats,
        "aggregate": aggregated_stats,
    }


# ---------------------------------------------------------
# Save JSON
# ---------------------------------------------------------
def save_json(data, out_path):
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Saved] {out_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, required=False,
                        default="/mnt/jy/LEval/Predictions/exam_eval/llama3-3b-8k/qk_cache.pt")
    parser.add_argument("--out", type=str, required=False,
                        default="/mnt/jy/LEval/Predictions/exam_eval/llama3-3b-8k/qk_stats.json")

    args = parser.parse_args()

    stats = analyze_qk_cache(args.cache)
    save_json(stats, args.out)