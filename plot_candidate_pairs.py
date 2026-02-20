#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from filfind_lib import (
    apply_fom_filter,
    load_star_coords,
    load_topaz_coords,
    progress,
    read_mrc_2d,
    select_candidate_pairs_by_mean_std,
)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay candidate Topaz pick-pairs (line connections) on top of an MRC image."
    )
    parser.add_argument("--mrc", required=True, type=Path, help="Input .mrc file")
    parser.add_argument("--autopick", required=True, type=Path, help="Topaz autopick STAR file")
    parser.add_argument("--manual", type=Path, default=None, help="Optional manual endpoint STAR file")
    parser.add_argument("--fom-min", type=float, default=None, help="Minimum FOM filter")
    parser.add_argument("--fom-max", type=float, default=None, help="Maximum FOM filter")
    parser.add_argument(
        "--candidate-k-std",
        type=float,
        default=1.0,
        help="Candidate rule: distance <= mean(pairwise) - k*std(pairwise)",
    )
    parser.add_argument("--max-lines", type=int, default=3000, help="Max candidate lines to draw (<=0 means all)")
    parser.add_argument("--line-alpha", type=float, default=0.25, help="Candidate line alpha")
    parser.add_argument("--line-width", type=float, default=0.6, help="Candidate line width")
    parser.add_argument(
        "--line-color",
        default="#00BCD4",
        help="Fixed line color (used unless --color-by-distance is set)",
    )
    parser.add_argument(
        "--color-by-distance",
        action="store_true",
        help="Color candidate lines by pair distance and show a colorbar",
    )
    parser.add_argument("--line-cmap", default="plasma", help="Colormap for candidate line distances")
    parser.add_argument("--out", default=Path("overlay_candidate_pairs.png"), type=Path, help="Output PNG")
    parser.add_argument("--show", action="store_true", help="Show interactive plot window")
    parser.add_argument("--no-save", action="store_true", help="Do not save PNG output")
    parser.add_argument("--dpi", default=220, type=int, help="Output DPI")
    args = parser.parse_args()

    progress("[load] reading image and STAR files")
    img = read_mrc_2d(args.mrc)
    auto_xy, auto_fom = load_topaz_coords(args.autopick)
    auto_xy, auto_fom = apply_fom_filter(auto_xy, auto_fom, args.fom_min, args.fom_max)
    manual_xy = load_star_coords(args.manual) if args.manual else np.empty((0, 2), dtype=float)

    n = len(auto_xy)
    progress(f"[load] filtered topaz picks: {n}")
    if n < 2:
        raise ValueError("Need at least two filtered picks to form candidate pairs")

    progress("[pairwise] building full pairwise distances")
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = auto_xy[i_idx, 0] - auto_xy[j_idx, 0]
    dy = auto_xy[i_idx, 1] - auto_xy[j_idx, 1]
    pair_dist = np.hypot(dx, dy)

    cand_i, cand_j, cand_d, mean_d, std_d, cutoff = select_candidate_pairs_by_mean_std(
        i_idx, j_idx, pair_dist, args.candidate_k_std
    )
    progress(
        "[pairwise] candidate rule: "
        f"distance <= mean - k*std = {mean_d:.3f} - {args.candidate_k_std:.3f}*{std_d:.3f} = {cutoff:.3f}"
    )
    progress(f"[pairwise] candidate pairs kept: {len(cand_d)} / {len(pair_dist)}")

    if args.max_lines > 0 and len(cand_d) > args.max_lines:
        order = np.argsort(cand_d)
        keep_idx = order[: args.max_lines]
        cand_i = cand_i[keep_idx]
        cand_j = cand_j[keep_idx]
        cand_d = cand_d[keep_idx]
        progress(f"[plot] limiting drawn candidate lines to {len(cand_d)} shortest pairs")

    segments = np.stack((auto_xy[cand_i], auto_xy[cand_j]), axis=1)

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(img, cmap="gray", origin="upper")

    if len(segments):
        if args.color_by_distance:
            lc = LineCollection(
                segments,
                cmap=args.line_cmap,
                array=cand_d,
                linewidths=args.line_width,
                alpha=args.line_alpha,
            )
        else:
            lc = LineCollection(
                segments,
                colors=args.line_color,
                linewidths=args.line_width,
                alpha=args.line_alpha,
            )
        ax.add_collection(lc)
        if args.color_by_distance:
            cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Candidate pair distance (px)")

    ax.scatter(
        auto_xy[:, 0],
        auto_xy[:, 1],
        s=16,
        facecolors="none",
        edgecolors="#4FC3F7",
        linewidths=0.6,
        alpha=0.9,
        label=f"Topaz picks (n={len(auto_xy)})",
    )

    if manual_xy.size:
        ax.scatter(
            manual_xy[:, 0],
            manual_xy[:, 1],
            s=62,
            c="#FF7043",
            marker="x",
            linewidths=1.3,
            alpha=0.95,
            label=f"Manual start/stop (n={len(manual_xy)})",
        )

    ax.set_title(
        f"{args.mrc.name}\nCandidate pair overlay (k_std={args.candidate_k_std}, cutoff={cutoff:.1f}px)"
    )
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.legend(loc="lower right", framealpha=0.75)

    if not args.no_save:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)

    if args.no_save:
        progress("[done] saved overlay: disabled (--no-save)")
    else:
        progress(f"[done] saved overlay: {args.out.resolve()}")


if __name__ == "__main__":
    main()
