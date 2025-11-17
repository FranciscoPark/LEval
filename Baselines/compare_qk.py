import json
from scipy.stats import spearmanr, kendalltau
import numpy as np
import argparse

# ---------------------------------------------------------
# Bottom-K Spearman for a **single layer**
# ---------------------------------------------------------
def bottom8_spearman_single_layer(a_order, b_order, bottom_k=8):
    """
    Compute Spearman correlation for the bottom-K heads of a single layer.
    a_order, b_order are full sorted head lists (0..H-1).
    """

    bottomA = a_order[-bottom_k:]
    bottomB = b_order[-bottom_k:]

    # normalize rank positions inside each bottom-k slice
    idxA = {h: i for i, h in enumerate(bottomA)}
    idxB = {h: i for i, h in enumerate(bottomB)}

    # Only compare heads that appear in both bottom-K sets
    common = sorted(set(bottomA) & set(bottomB))

    if len(common) < 2:
        return None  # insufficient common heads → undefined

    ranksA = [idxA[h] for h in common]
    ranksB = [idxB[h] for h in common]

    rho, _ = spearmanr(ranksA, ranksB)
    return float(rho)


# ---------------------------------------------------------
# Full-layer comparison (full ranking correlation + top-K + bottom-K)
# ---------------------------------------------------------
def compare_layer(a_order, b_order, topk=5, bottomk=4):
    """
    Compare ranking lists a_order and b_order for one layer.
    """

    # -- Spearman Rank Correlation over full ranking --
    A_rank = np.argsort(a_order)
    B_rank = np.argsort(b_order)
    spearman, _ = spearmanr(A_rank, B_rank)

    # -- Kendall Tau --
    kendall, _ = kendalltau(A_rank, B_rank)

    # -- Top-K Overlap --
    overlap = len(set(a_order[:topk]) & set(b_order[:topk])) / topk

    # -- Bottom-K Spearman --
    bottom_spear = bottom8_spearman_single_layer(a_order, b_order, bottomk)

    return {
        "spearman": float(spearman),
        "kendall": float(kendall),
        "topk_overlap": overlap,
        "bottomk_spearman": bottom_spear,
    }


# ---------------------------------------------------------
# Main: Compare two qk_stats.json files
# ---------------------------------------------------------
def main(path_a, path_b, out):
    A = json.load(open(path_a))
    B = json.load(open(path_b))

    A_agg = A["aggregate"]
    B_agg = B["aggregate"]

    layers = sorted(A_agg.keys(), key=lambda x: int(x))

    results = {}

    for layer in layers:
        a_order = A_agg[layer]["sorted_heads"]
        b_order = B_agg[layer]["sorted_heads"]

        stats = compare_layer(a_order, b_order)
        results[layer] = stats

    # Save results
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Saved comparison] {out}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=False,
                        default="/mnt/jy/LEval/Predictions/exam_eval/llama3-3b-8k/tpo_qk_stats.json",
                        help="qk_stats of model A")
    parser.add_argument("--b", required=False,
                        default="/mnt/jy/LEval/Predictions/exam_eval/llama3-3b-8k/qk_stats.json",
                        help="qk_stats of model B")
    parser.add_argument("--out", default="qk_compare.json")
    args = parser.parse_args()

    main(args.a, args.b, args.out)