#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from filfind_lib import read_mrc_2d


def load_endpoints(assignments_csv: Path):
    by_filament = defaultdict(list)
    with assignments_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["filament_id"])
            order = int(row["order_in_filament"])
            x = float(row["x"])
            y = float(row["y"])
            by_filament[fid].append((order, x, y))

    endpoints = []
    for fid in sorted(by_filament):
        pts = sorted(by_filament[fid], key=lambda t: t[0])
        start = np.array([pts[0][1], pts[0][2]], dtype=float)
        end = np.array([pts[-1][1], pts[-1][2]], dtype=float)
        endpoints.append((fid, start, end, len(pts)))
    return endpoints


def main():
    parser = argparse.ArgumentParser(
        description="Plot only filament start/end points with connecting line for each traced filament"
    )
    parser.add_argument("--mrc", required=True, type=Path, help="MRC image path")
    parser.add_argument("--assignments", required=True, type=Path, help="trace_filaments assignments CSV")
    parser.add_argument("--out", default=Path("filament_endpoints_overlay.png"), type=Path, help="Output PNG")
    parser.add_argument("--show", action="store_true", help="Show interactive plot window")
    parser.add_argument("--no-save", action="store_true", help="Do not save PNG output")
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI")
    args = parser.parse_args()

    endpoints = load_endpoints(args.assignments)
    img = read_mrc_2d(args.mrc)

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(img, cmap="gray", origin="upper")

    cmap = plt.get_cmap("tab20")
    for i, (fid, start, end, npts) in enumerate(endpoints):
        color = cmap(i % 20)
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2.0, alpha=0.9)
        ax.scatter(start[0], start[1], s=56, c=[color], marker="o")
        ax.scatter(end[0], end[1], s=70, c=[color], marker="x", linewidths=1.8)

    ax.set_title(f"{args.mrc.name}\\nFilament endpoints and connecting lines (n={len(endpoints)})")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    if not args.no_save:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)

    if args.no_save:
        print("Saved endpoint overlay: disabled (--no-save)")
    else:
        print(f"Saved endpoint overlay: {args.out.resolve()}")
    print(f"Filaments plotted: {len(endpoints)}")


if __name__ == "__main__":
    main()
