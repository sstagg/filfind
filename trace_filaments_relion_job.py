#!/usr/bin/env python3
import argparse
from pathlib import Path

from filfind_lib import progress
from filfind_trace_lib import trace_filaments_single


def infer_mrc_name_from_star(star_path):
    name = star_path.name
    if name.endswith("_autopick.star"):
        return name[: -len("_autopick.star")] + ".mrc"
    if name.endswith(".star"):
        return name[:-5] + ".mrc"
    return f"{name}.mrc"


def resolve_mrc_path(star_path, mrc_root):
    mrc_name = infer_mrc_name_from_star(star_path)
    in_place = star_path.with_name(mrc_name)
    if in_place.exists():
        return in_place

    matches = sorted(mrc_root.rglob(mrc_name))
    if not matches:
        return None
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="Run filament tracing on all matching STAR files in a RELION-style job tree")
    parser.add_argument("--job-dir", required=True, type=Path, help="Root directory containing autopick STAR files")
    parser.add_argument(
        "--mrc-root",
        type=Path,
        default=None,
        help="Root directory to search for corresponding MRC files (default: --job-dir)",
    )
    parser.add_argument(
        "--star-pattern",
        default="*_autopick.star",
        help="Glob pattern used recursively under --job-dir to find STAR files",
    )
    parser.add_argument("--fom-min", type=float, default=0.0, help="Minimum FOM filter")
    parser.add_argument("--fom-max", type=float, default=None, help="Maximum FOM filter")
    parser.add_argument("--candidate-k-std", type=float, default=1.0, help="Pair candidate threshold k in mean-k*std")
    parser.add_argument("--max-neighbors", type=int, default=2, help="Max nearest candidate edges per node")
    parser.add_argument("--max-line-rms", type=float, default=10.0, help="Max allowed RMS deviation from best-fit line")
    parser.add_argument("--save-npz", action="store_true", help="Save trace summary NPZ (disabled by default)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional directory for output files")
    parser.add_argument("--no-save-overlay", action="store_true", help="Do not save overlay PNG")
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of STAR files processed")
    args = parser.parse_args()

    job_dir = args.job_dir.resolve()
    mrc_root = (args.mrc_root or args.job_dir).resolve()
    star_files = sorted(job_dir.rglob(args.star_pattern))
    if args.max_files is not None:
        star_files = star_files[: args.max_files]

    progress(f"[batch] job dir: {job_dir}")
    progress(f"[batch] mrc root: {mrc_root}")
    progress(f"[batch] STAR files matched: {len(star_files)}")
    if not star_files:
        return

    ok = 0
    missing_mrc = 0
    failed = 0
    for i, star_path in enumerate(star_files, start=1):
        progress(f"[batch] ({i}/{len(star_files)}) STAR: {star_path}")
        mrc_path = resolve_mrc_path(star_path, mrc_root)
        if mrc_path is None:
            progress(f"[batch] missing MRC for STAR: {star_path.name}")
            missing_mrc += 1
            continue

        try:
            trace_filaments_single(
                autopick_path=star_path,
                mrc_path=mrc_path,
                fom_min=args.fom_min,
                fom_max=args.fom_max,
                candidate_k_std=args.candidate_k_std,
                max_neighbors=args.max_neighbors,
                max_line_rms=args.max_line_rms,
                save_npz=args.save_npz,
                show=False,
                no_save_overlay=args.no_save_overlay,
                growth_movie_out=None,
                growth_movie_fps=8,
                dpi=args.dpi,
                out_dir=args.out_dir,
            )
            ok += 1
        except Exception as exc:
            progress(f"[batch] failed for {star_path.name}: {exc}")
            failed += 1

    progress(f"[batch] done: success={ok}, missing_mrc={missing_mrc}, failed={failed}, total={len(star_files)}")


if __name__ == "__main__":
    main()
