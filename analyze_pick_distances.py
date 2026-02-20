#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from filfind_lib import (
    apply_fom_filter,
    compute_nearest_neighbor_distances,
    compute_pairwise,
    describe_distribution,
    load_topaz_coords,
    progress,
    select_candidate_pairs_by_mean_std,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute all pairwise distances between Topaz picks and plot distance histograms."
    )
    parser.add_argument("--autopick", required=True, type=Path, help="Topaz autopick STAR file")
    parser.add_argument("--fom-min", type=float, default=None, help="Minimum FOM filter")
    parser.add_argument("--fom-max", type=float, default=None, help="Maximum FOM filter")
    parser.add_argument("--bins", type=int, default=200, help="Histogram bins")
    parser.add_argument(
        "--candidate-k-std",
        type=float,
        default=None,
        help="Keep candidate pairs with distance <= mean(pairwise) - k*std(pairwise)",
    )
    parser.add_argument(
        "--progress-every-pairs",
        type=int,
        default=500_000,
        help="Report pairwise progress every N pair computations",
    )
    parser.add_argument(
        "--progress-every-picks",
        type=int,
        default=1000,
        help="Report nearest-neighbor progress every N picks",
    )
    parser.add_argument(
        "--hist-max",
        type=float,
        default=300.0,
        help="Max x-axis for histograms (pixels); set <=0 to use full range",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("topaz_distance_analysis"),
        help="Prefix for output files",
    )
    args = parser.parse_args()

    coords, fom = load_topaz_coords(args.autopick)
    coords, fom = apply_fom_filter(coords, fom, args.fom_min, args.fom_max)

    progress(f"[load] loaded picks: {len(coords)}")
    progress(f"[load] FOM filter min={args.fom_min} max={args.fom_max}")
    i_idx, j_idx, pairwise_dist = compute_pairwise(coords, report_every=args.progress_every_pairs)
    nn_dist = compute_nearest_neighbor_distances(coords, report_every=args.progress_every_picks)

    if args.candidate_k_std is not None:
        cand_i, cand_j, cand_d, mean_d, std_d, cutoff = select_candidate_pairs_by_mean_std(
            i_idx, j_idx, pairwise_dist, args.candidate_k_std
        )
    else:
        cand_i, cand_j, cand_d = np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=float)
        mean_d, std_d, cutoff = np.nan, np.nan, np.nan

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        coordinates_xy=coords,
        fom=fom,
        pair_i=i_idx,
        pair_j=j_idx,
        pairwise_distance=pairwise_dist,
        nearest_neighbor_distance=nn_dist,
        candidate_pair_i=cand_i,
        candidate_pair_j=cand_j,
        candidate_pair_distance=cand_d,
        candidate_rule_k_std=np.array([np.nan if args.candidate_k_std is None else args.candidate_k_std]),
        candidate_rule_mean=np.array([mean_d]),
        candidate_rule_std=np.array([std_d]),
        candidate_rule_cutoff=np.array([cutoff]),
    )

    if args.hist_max is not None and args.hist_max > 0:
        pair_for_hist = pairwise_dist[pairwise_dist <= args.hist_max]
        nn_for_hist = nn_dist[nn_dist <= args.hist_max]
        xmax = args.hist_max
    else:
        pair_for_hist = pairwise_dist
        nn_for_hist = nn_dist
        xmax = None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].hist(pair_for_hist, bins=args.bins, color="#4FC3F7", alpha=0.9)
    axes[0].set_title("All Pairwise Distances")
    axes[0].set_xlabel("Distance (pixels)")
    axes[0].set_ylabel("Count")
    if xmax is not None:
        axes[0].set_xlim(0, xmax)

    axes[1].hist(nn_for_hist, bins=args.bins, color="#FF7043", alpha=0.9)
    axes[1].set_title("Nearest-Neighbor Distances")
    axes[1].set_xlabel("Distance (pixels)")
    axes[1].set_ylabel("Count")
    if xmax is not None:
        axes[1].set_xlim(0, xmax)

    fig.suptitle(f"{args.autopick.name}\nFOM filters: min={args.fom_min}, max={args.fom_max}")
    hist_path = args.out_prefix.parent / f"{args.out_prefix.name}_hist.png"
    fig.savefig(hist_path, dpi=220)
    plt.close(fig)

    progress(f"[done] filtered picks: {len(coords)}")
    progress(f"[done] {describe_distribution('Pairwise distances', pairwise_dist)}")
    progress(f"[done] {describe_distribution('Nearest-neighbor distances', nn_dist)}")
    if args.candidate_k_std is not None:
        progress(
            "[done] candidate filter: "
            f"distance <= mean - k*std = {mean_d:.3f} - {args.candidate_k_std:.3f}*{std_d:.3f} = {cutoff:.3f}"
        )
        progress(f"[done] candidate pairs kept: {len(cand_d)} / {len(pairwise_dist)}")
        progress(f"[done] {describe_distribution('Candidate-pair distances', cand_d)}")
    progress(f"[done] saved data: {npz_path.resolve()}")
    progress(f"[done] saved histogram: {hist_path.resolve()}")


if __name__ == "__main__":
    main()
