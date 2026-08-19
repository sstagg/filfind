#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from filfind_lib import (
    filament_endpoint_rows_from_star,
    load_star_coords,
    progress,
    read_mrc_2d,
    write_farthest_filament_endpoints_star,
    write_micrograph_endpoint_stars,
)


def default_out_dir(star_path):
    stem = star_path.with_suffix("").name
    return star_path.with_name(f"{stem}_filfind_endpoint_coords")


def strip_suffix(text, suffix):
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text


def render_preview_overlays(endpoint_dir, mrc_dir, max_images=None, preview_max_dim=1024):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/filfind_mplconfig")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from filfind_trace_lib import downsample_for_overlay

    endpoint_dir = Path(endpoint_dir)
    mrc_dir = Path(mrc_dir)
    preview_dir = endpoint_dir
    preview_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for star in sorted(endpoint_dir.glob("*_filfind_endpoints.star")):
        stem = strip_suffix(star.name, "_filfind_endpoints.star")
        mrc = mrc_dir / f"{stem}.mrc"
        if not mrc.exists():
            continue
        coords = load_star_coords(star)
        if len(coords) >= 2:
            candidates.append((len(coords) // 2, stem, star, mrc, coords))

    candidates.sort(key=lambda t: (-t[0], t[1]))
    selected = candidates if max_images is None else candidates[:max(0, max_images)]
    if not selected:
        progress("[preview] no matching endpoint STAR + MRC pairs found")
        return None

    def draw_segments(ax, coords, lw=2.0, s1=34, s2=48):
        cmap = plt.get_cmap("tab20")
        for i in range(0, len(coords) - 1, 2):
            start = coords[i]
            end = coords[i + 1]
            color = cmap((i // 2) % 20)
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=lw, alpha=0.96)
            ax.scatter(start[0], start[1], s=s1, c=[color], marker="o")
            ax.scatter(end[0], end[1], s=s2, c=[color], marker="x", linewidths=1.6)

    summary_rows = []
    render_dpi = 120

    for idx, (segments, stem, _star, mrc, coords) in enumerate(selected, start=1):
        img = read_mrc_2d(mrc)
        img_ds, coords_ds, scale = downsample_for_overlay(img, coords, preview_max_dim)

        fig_w = max(1.0, img_ds.shape[1] / render_dpi)
        fig_h = max(1.0, img_ds.shape[0] / render_dpi)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(img_ds, cmap="gray", origin="upper", interpolation="nearest", resample=False)
        draw_segments(ax, coords_ds)
        ax.set_xlim(0, img_ds.shape[1])
        ax.set_ylim(img_ds.shape[0], 0)
        ax.set_axis_off()
        ax.set_title(f"{stem}\nfinal endpoints: {segments} segments, binned x{scale:.3f}", fontsize=9)
        fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0)
        png = preview_dir / f"{stem}_final_endpoints_overlay.png"
        fig.savefig(png, dpi=render_dpi)
        plt.close(fig)

        summary_rows.append((idx, stem, segments, len(coords), png.name))
        progress(f"[preview] {idx}/{len(selected)} {stem}: {segments} segments")

    summary = preview_dir / "preview_summary.tsv"
    with summary.open("w", encoding="utf-8") as f:
        f.write("rank\tmicrograph_stem\tsegments\tendpoint_points\tpng\n")
        for row in summary_rows:
            f.write("\t".join(map(str, row)) + "\n")

    return {
        "preview_dir": preview_dir,
        "summary": summary,
        "count": len(summary_rows),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recover filament start/end coordinates from a cleaned particle STAR by "
            "grouping picks by filament and keeping the farthest pair in each filament."
        )
    )
    parser.add_argument("--input", "--star", dest="input_star", required=True, type=Path, help="Input particle STAR file")
    parser.add_argument(
        "--output",
        "--out",
        dest="coordinate_files_star",
        type=Path,
        default=None,
        help="Output coordinate-files STAR (default: <out-dir>/filfind_coordinate_files.star)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for per-micrograph endpoint STAR files (default: <input>_filfind_endpoint_coords)",
    )
    parser.add_argument(
        "--single-star",
        type=Path,
        default=None,
        help="Optional extra combined endpoint STAR file, mostly for inspection/debugging",
    )
    parser.add_argument(
        "--rel-base",
        type=Path,
        default=Path.cwd(),
        help="Base directory for relative paths in the coordinate-files STAR (default: current directory)",
    )
    parser.add_argument(
        "--filament-column",
        default=None,
        help="Column identifying the filament (default: auto-detect RELION helical tube columns)",
    )
    parser.add_argument(
        "--micrograph-column",
        default=None,
        help="Optional column identifying the micrograph (default: auto-detect _rlnMicrographName)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=2,
        help="Minimum particles required in a filament group to emit endpoints",
    )
    parser.add_argument(
        "--singleton-parent-endpoint-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of original per-micrograph endpoint STARs. Singleton classified "
            "filaments are matched to the nearest original endpoint segment within "
            "--singleton-max-distance-px and keep those original endpoints."
        ),
    )
    parser.add_argument(
        "--singleton-max-distance-px",
        type=float,
        default=5.0,
        help="Maximum pixel distance for matching a singleton classified particle to an original endpoint segment",
    )
    parser.add_argument(
        "--mrc-dir",
        type=Path,
        default=None,
        help="Directory containing matching MotionCorr .mrc files; enables final endpoint overlay PNG rendering",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of overlay PNGs to render when --mrc-dir is set (default: all matching images)",
    )
    parser.add_argument(
        "--preview-max-dim",
        type=int,
        default=1024,
        help="Maximum preview overlay image dimension in pixels; uses the existing binning helper",
    )
    args = parser.parse_args()

    input_star = args.input_star
    out_dir = args.out_dir or default_out_dir(input_star)
    coordinate_files_star = args.coordinate_files_star or (out_dir / "filfind_coordinate_files.star")

    rows, stats = filament_endpoint_rows_from_star(
        input_star,
        filament_column=args.filament_column,
        micrograph_column=args.micrograph_column,
        min_points=args.min_points,
        singleton_parent_endpoint_dir=args.singleton_parent_endpoint_dir,
        singleton_max_distance_px=args.singleton_max_distance_px,
    )
    if stats["micrograph_column"] is None:
        raise ValueError("Per-micrograph output requires _rlnMicrographName or --micrograph-column")

    write_stats = write_micrograph_endpoint_stars(
        out_dir,
        rows,
        coordinate_files_star=coordinate_files_star,
        rel_base=args.rel_base,
    )
    if args.single_star is not None:
        args.single_star.parent.mkdir(parents=True, exist_ok=True)
        write_farthest_filament_endpoints_star(args.single_star, rows, include_micrograph=True)

    progress(f"[load] input rows: {stats['input_rows']}")
    progress(f"[group] filament column: {stats['filament_column']}")
    progress(f"[group] micrograph column: {stats['micrograph_column'] or 'none'}")
    progress(f"[done] filaments found: {stats['filaments_total']}")
    progress(f"[done] filaments written: {stats['filaments_written']}")
    progress(f"[done] multi-particle filaments written: {stats['multi_particle_filaments']}")
    if args.singleton_parent_endpoint_dir is not None:
        progress(f"[done] singleton groups: {stats['singleton_groups']}")
        progress(f"[done] singleton parent matches: {stats['singleton_matched']}")
        progress(f"[done] singleton unmatched: {stats['singleton_unmatched']}")
        progress(f"[done] singleton duplicate parent matches skipped: {stats['singleton_duplicate_parent_matches']}")
    progress(f"[done] filaments skipped (<{args.min_points} picks): {stats['filaments_skipped']}")
    progress(f"[done] endpoint rows written: {len(rows)}")
    progress(f"[done] per-micrograph coordinate files: {write_stats['micrograph_count']}")
    progress(f"[done] saved coordinate-files STAR: {write_stats['coordinate_files_star'].resolve()}")
    if args.single_star is not None:
        progress(f"[done] saved combined endpoint STAR: {args.single_star.resolve()}")

    if args.mrc_dir is not None:
        preview_stats = render_preview_overlays(out_dir, args.mrc_dir, args.max_images, args.preview_max_dim)
        if preview_stats is not None:
            progress(f"[done] preview overlays: {preview_stats['count']}")
            progress(f"[done] preview dir: {preview_stats['preview_dir'].resolve()}")


if __name__ == "__main__":
    main()
