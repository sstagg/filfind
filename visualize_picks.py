#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from filfind_lib import apply_fom_filter, load_star_coords, load_topaz_coords, read_mrc_2d


def main():
    parser = argparse.ArgumentParser(description="Overlay STAR picks on an MRC image")
    parser.add_argument("--mrc", required=True, type=Path, help="Input .mrc file")
    parser.add_argument("--autopick", required=True, type=Path, help="Topaz autopick STAR file")
    parser.add_argument("--manual", required=True, type=Path, help="Manual endpoint STAR file")
    parser.add_argument("--fom-min", type=float, default=None, help="Minimum Topaz FOM to keep")
    parser.add_argument("--fom-max", type=float, default=None, help="Maximum Topaz FOM to keep")
    parser.add_argument("--fom-cmap", default="viridis", help="Matplotlib colormap for Topaz FOM")
    parser.add_argument("--out", default=Path("overlay_picks.png"), type=Path, help="Output image path")
    parser.add_argument("--show", action="store_true", help="Show interactive plot window")
    parser.add_argument("--no-save", action="store_true", help="Do not save PNG output")
    parser.add_argument("--dpi", default=220, type=int, help="Output DPI")
    args = parser.parse_args()

    img = read_mrc_2d(args.mrc)
    auto_xy, auto_fom = load_topaz_coords(args.autopick)
    auto_xy, auto_fom = apply_fom_filter(auto_xy, auto_fom, args.fom_min, args.fom_max)
    manual_xy = load_star_coords(args.manual)

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(img, cmap="gray", origin="upper")

    if auto_xy.size:
        valid_fom = np.isfinite(auto_fom)
        if valid_fom.any():
            sc = ax.scatter(
                auto_xy[:, 0],
                auto_xy[:, 1],
                s=18,
                c=auto_fom,
                cmap=args.fom_cmap,
                linewidths=0.2,
                edgecolors="black",
                alpha=0.9,
                label=f"Topaz candidates (n={len(auto_xy)})",
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Topaz FOM")
        else:
            ax.scatter(
                auto_xy[:, 0],
                auto_xy[:, 1],
                s=18,
                facecolors="none",
                edgecolors="#4FC3F7",
                linewidths=0.8,
                alpha=0.85,
                label=f"Topaz candidates (n={len(auto_xy)})",
            )

    if manual_xy.size:
        ax.scatter(
            manual_xy[:, 0],
            manual_xy[:, 1],
            s=60,
            c="#FF7043",
            marker="x",
            linewidths=1.4,
            alpha=0.95,
            label=f"Manual start/stop (n={len(manual_xy)})",
        )

    ax.set_title(f"{args.mrc.name}\nTopaz + manual endpoint picks")
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
        print("Saved overlay: disabled (--no-save)")
    else:
        print(f"Saved overlay: {args.out.resolve()}")
    print(f"Image shape: {img.shape[1]}x{img.shape[0]}")
    print(f"Topaz picks: {len(auto_xy)}")
    if len(auto_fom) and np.isfinite(auto_fom).any():
        valid = auto_fom[np.isfinite(auto_fom)]
        print(f"Topaz FOM range (after filter): {valid.min():.4f} to {valid.max():.4f}")
    print(f"Manual picks: {len(manual_xy)}")


if __name__ == "__main__":
    main()
