#!/usr/bin/env python3
import argparse
from pathlib import Path

from filfind_trace_lib import trace_filaments_single


def main():
    parser = argparse.ArgumentParser(description="Trace filaments by growing line-consistent chains of Topaz picks")
    parser.add_argument("--autopick", required=True, type=Path, help="Topaz autopick STAR file")
    parser.add_argument("--mrc", required=True, type=Path, help="MRC for overlay visualization")
    parser.add_argument("--fom-min", type=float, default=0.0, help="Minimum FOM filter")
    parser.add_argument("--fom-max", type=float, default=None, help="Maximum FOM filter")
    parser.add_argument("--candidate-k-std", type=float, default=1.0, help="Pair candidate threshold k in mean-k*std")
    parser.add_argument("--max-neighbors", type=int, default=2, help="Max nearest candidate edges per node")
    parser.add_argument("--max-line-rms", type=float, default=10.0, help="Max allowed RMS deviation from best-fit line")
    parser.add_argument("--save-npz", action="store_true", help="Save trace summary NPZ (disabled by default)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional directory for output files")
    parser.add_argument("--show", action="store_true", help="Show interactive overlay window")
    parser.add_argument("--no-save-overlay", action="store_true", help="Do not save overlay PNG")
    parser.add_argument(
        "--growth-movie-out",
        type=Path,
        default=None,
        help="Optional output animated GIF showing greedy chain growth",
    )
    parser.add_argument("--growth-movie-fps", type=int, default=8, help="FPS for --growth-movie-out")
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI")
    args = parser.parse_args()

    trace_filaments_single(
        autopick_path=args.autopick,
        mrc_path=args.mrc,
        fom_min=args.fom_min,
        fom_max=args.fom_max,
        candidate_k_std=args.candidate_k_std,
        max_neighbors=args.max_neighbors,
        max_line_rms=args.max_line_rms,
        save_npz=args.save_npz,
        show=args.show,
        no_save_overlay=args.no_save_overlay,
        growth_movie_out=args.growth_movie_out,
        growth_movie_fps=args.growth_movie_fps,
        dpi=args.dpi,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
