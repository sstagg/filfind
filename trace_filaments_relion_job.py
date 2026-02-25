#!/usr/bin/env python3
import argparse
import shlex
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
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


def quote_cmd(parts):
    return " ".join(shlex.quote(str(p)) for p in parts)


def quiet_progress(_msg):
    # Worker-side no-op logger to avoid noisy interleaved stdout from parallel jobs.
    return


def run_trace_job(job):
    try:
        trace_filaments_single(
            autopick_path=Path(job["autopick_path"]),
            mrc_path=Path(job["mrc_path"]),
            fom_min=job["fom_min"],
            fom_max=job["fom_max"],
            candidate_k_std=job["candidate_k_std"],
            max_neighbors=job["max_neighbors"],
            max_line_rms=job["max_line_rms"],
            save_npz=job["save_npz"],
            show=False,
            no_save_overlay=job["no_save_overlay"],
            growth_movie_out=None,
            growth_movie_fps=8,
            overlay_max_dim=job["overlay_max_dim"],
            out_prefix=Path(job["out_prefix"]) if job["out_prefix"] is not None else None,
            out_dir=Path(job["out_dir"]) if job["out_dir"] is not None else None,
            progress_fn=quiet_progress,
        )
        return {
            "index": job["index"],
            "total": job["total"],
            "star_name": job["star_name"],
            "status": "ok",
            "mrc_path": job["mrc_path"],
            "endpoints_star_path": job["endpoints_star_path"],
        }
    except Exception as exc:
        return {
            "index": job["index"],
            "total": job["total"],
            "star_name": job["star_name"],
            "status": "failed",
            "error": str(exc),
        }


def append_log(path, text):
    with path.open("a", encoding="utf-8") as log:
        log.write(text)


def rel_or_abs(path, base):
    path = Path(path).resolve()
    base = Path(base).resolve()
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def write_coordinate_files_star(path, rows, rel_base):
    # RELION coordinate-file STAR linking each micrograph to its coordinates STAR.
    with path.open("w", encoding="utf-8") as f:
        f.write("\n")
        f.write("# version 50001\n\n")
        f.write("data_coordinate_files\n\n")
        f.write("loop_ \n")
        f.write("_rlnMicrographName #1 \n")
        f.write("_rlnMicrographCoordinates #2 \n")
        for mrc_path, coord_path in rows:
            mrc_txt = rel_or_abs(mrc_path, rel_base)
            coord_txt = rel_or_abs(coord_path, rel_base)
            f.write(f"{mrc_txt} {coord_txt}\n")


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
    parser.add_argument(
        "--out-dir",
        "--output-dir",
        dest="out_dir",
        type=Path,
        default=None,
        help="Optional directory for output files (default: next to each MRC)",
    )
    parser.add_argument("--no-save-overlay", action="store_true", help="Do not save overlay PNG")
    parser.add_argument(
        "--overlay-max-dim",
        type=int,
        default=1024,
        help="Maximum overlay PNG dimension in pixels (largest side, <=0 disables downscale)",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of STAR files processed")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of files to process in parallel (1 = sequential)",
    )
    parser.add_argument(
        "--command-log",
        type=Path,
        default=None,
        help="Optional log file path for human-readable command history",
    )
    args = parser.parse_args()

    workers = max(1, int(args.workers))

    job_dir = args.job_dir.resolve()
    mrc_root = (args.mrc_root or args.job_dir).resolve()
    if args.out_dir is None:
        out_dir = None
    elif args.out_dir.is_absolute():
        out_dir = args.out_dir.resolve()
    else:
        # Interpret relative output dirs with respect to job root for predictable batch behavior.
        out_dir = (job_dir / args.out_dir).resolve()

    if args.command_log is None:
        command_log = (out_dir / "filfind_batch_commands.log") if out_dir is not None else (job_dir / "filfind_batch_commands.log")
    elif args.command_log.is_absolute():
        command_log = args.command_log.resolve()
    else:
        command_log = (job_dir / args.command_log).resolve()
    command_log.parent.mkdir(parents=True, exist_ok=True)

    star_files = sorted(job_dir.rglob(args.star_pattern))
    if args.max_files is not None:
        star_files = star_files[: args.max_files]

    progress(f"[batch] job dir: {job_dir}")
    progress(f"[batch] mrc root: {mrc_root}")
    progress(f"[batch] output dir: {out_dir if out_dir is not None else 'default (near each MRC)'}")
    progress(f"[batch] workers: {workers}")
    progress(f"[batch] command log: {command_log}")
    progress(f"[batch] STAR files matched: {len(star_files)}")
    if not star_files:
        return

    append_log(
        command_log,
        (
            f"\n=== filfind batch run {datetime.now().isoformat(timespec='seconds')} ===\n"
            f"python={sys.executable}\n"
            f"cwd={Path.cwd()}\n"
            f"job_dir={job_dir}\n"
            f"mrc_root={mrc_root}\n"
            f"output_dir={out_dir if out_dir is not None else 'default (near each MRC)'}\n"
            f"workers={workers}\n"
            f"star_pattern={args.star_pattern}\n"
            f"fom_min={args.fom_min} fom_max={args.fom_max}\n"
            f"candidate_k_std={args.candidate_k_std} max_neighbors={args.max_neighbors} max_line_rms={args.max_line_rms}\n"
            f"save_npz={args.save_npz} no_save_overlay={args.no_save_overlay} overlay_max_dim={args.overlay_max_dim}\n"
            f"max_files={args.max_files}\n"
        ),
    )

    missing_mrc = 0
    star_iter = iter(enumerate(star_files, start=1))

    def next_job():
        nonlocal missing_mrc
        for i, star_path in star_iter:
            progress(f"[batch] ({i}/{len(star_files)}) STAR: {star_path}")
            mrc_path = resolve_mrc_path(star_path, mrc_root)
            if mrc_path is None:
                progress(f"[batch] missing MRC for STAR: {star_path.name}")
                append_log(command_log, f"[{i}/{len(star_files)}] status=missing_mrc star={star_path}\n")
                missing_mrc += 1
                continue

            if out_dir is None:
                out_prefix = None
            else:
                stem = mrc_path.with_suffix("").name
                out_prefix = out_dir / f"{stem}_filfind"
                progress(f"[batch] out-prefix: {out_prefix}")
            if out_prefix is None:
                endpoints_star_path = mrc_path.with_suffix("").parent / f"{mrc_path.with_suffix('').name}_filfind_endpoints.star"
            else:
                endpoints_star_path = out_prefix.parent / f"{out_prefix.name}_endpoints.star"

            cmd_equivalent = [
                "python",
                "trace_filaments.py",
                "--autopick",
                star_path,
                "--mrc",
                mrc_path,
                "--fom-min",
                args.fom_min,
                "--candidate-k-std",
                args.candidate_k_std,
                "--max-neighbors",
                args.max_neighbors,
                "--max-line-rms",
                args.max_line_rms,
            ]
            if args.fom_max is not None:
                cmd_equivalent += ["--fom-max", args.fom_max]
            if args.save_npz:
                cmd_equivalent += ["--save-npz"]
            if args.no_save_overlay:
                cmd_equivalent += ["--no-save-overlay"]
            cmd_equivalent += ["--overlay-max-dim", args.overlay_max_dim]
            if out_dir is not None:
                cmd_equivalent += ["--out-dir", out_dir]

            append_log(
                command_log,
                f"[{i}/{len(star_files)}] status=queued\n  command: {quote_cmd(cmd_equivalent)}\n",
            )

            return {
                "index": i,
                "total": len(star_files),
                "star_name": star_path.name,
                "autopick_path": str(star_path),
                "mrc_path": str(mrc_path),
                "fom_min": args.fom_min,
                "fom_max": args.fom_max,
                "candidate_k_std": args.candidate_k_std,
                "max_neighbors": args.max_neighbors,
                "max_line_rms": args.max_line_rms,
                "save_npz": bool(args.save_npz),
                "no_save_overlay": bool(args.no_save_overlay),
                "overlay_max_dim": int(args.overlay_max_dim),
                "out_prefix": str(out_prefix) if out_prefix is not None else None,
                "out_dir": str(out_dir) if out_dir is not None else None,
                "endpoints_star_path": str(endpoints_star_path),
            }
        return None

    ok = 0
    failed = 0
    coordinate_rows = []

    if workers == 1:
        while True:
            item = next_job()
            if item is None:
                break
            result = run_trace_job(item)
            if result["status"] == "ok":
                ok += 1
                progress(f"[batch] done {result['index']}/{result['total']}: {result['star_name']}")
                append_log(command_log, f"[{result['index']}/{result['total']}] status=ok\n")
                coordinate_rows.append((result["mrc_path"], result["endpoints_star_path"]))
            else:
                failed += 1
                progress(f"[batch] failed for {result['star_name']}: {result['error']}")
                append_log(
                    command_log,
                    f"[{result['index']}/{result['total']}] status=failed\n  error: {result['error']}\n",
                )
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                active = {}
                # Initial fill: submit up to <workers> jobs.
                while len(active) < workers:
                    item = next_job()
                    if item is None:
                        break
                    fut = ex.submit(run_trace_job, item)
                    active[fut] = item

                # As each job completes, record status and submit one new job.
                while active:
                    done, _ = wait(active.keys(), return_when=FIRST_COMPLETED)
                    for fut in done:
                        item = active.pop(fut)
                        try:
                            result = fut.result()
                        except Exception as exc:
                            failed += 1
                            progress(f"[batch] failed for {item['star_name']}: {exc}")
                            append_log(
                                command_log,
                                f"[{item['index']}/{item['total']}] status=failed\n  error: {exc}\n",
                            )
                        else:
                            if result["status"] == "ok":
                                ok += 1
                                progress(f"[batch] done {result['index']}/{result['total']}: {result['star_name']}")
                                append_log(command_log, f"[{result['index']}/{result['total']}] status=ok\n")
                                coordinate_rows.append((result["mrc_path"], result["endpoints_star_path"]))
                            else:
                                failed += 1
                                progress(f"[batch] failed for {result['star_name']}: {result['error']}")
                                append_log(
                                    command_log,
                                    f"[{result['index']}/{result['total']}] status=failed\n  error: {result['error']}\n",
                                )

                        new_item = next_job()
                        if new_item is not None:
                            new_fut = ex.submit(run_trace_job, new_item)
                            active[new_fut] = new_item
        except Exception as exc:
            progress(f"[batch] process workers unavailable ({exc}); falling back to sequential")
            append_log(command_log, f"[batch] process workers unavailable ({exc}); falling back to sequential\n")
            while True:
                item = next_job()
                if item is None:
                    break
                result = run_trace_job(item)
                if result["status"] == "ok":
                    ok += 1
                    progress(f"[batch] done {result['index']}/{result['total']}: {result['star_name']}")
                    append_log(command_log, f"[{result['index']}/{result['total']}] status=ok\n")
                    coordinate_rows.append((result["mrc_path"], result["endpoints_star_path"]))
                else:
                    failed += 1
                    progress(f"[batch] failed for {result['star_name']}: {result['error']}")
                    append_log(
                        command_log,
                        f"[{result['index']}/{result['total']}] status=failed\n  error: {result['error']}\n",
                    )

    combined_star_path = (out_dir if out_dir is not None else job_dir) / "filfind_coordinate_files.star"
    if coordinate_rows:
        write_coordinate_files_star(combined_star_path, coordinate_rows, rel_base=job_dir.parent)
        progress(f"[batch] saved combined coordinate STAR: {combined_star_path}")
        append_log(command_log, f"[batch] combined_coordinate_star={combined_star_path}\n")
    else:
        progress("[batch] no successful jobs; combined coordinate STAR not written")
        append_log(command_log, "[batch] no successful jobs; combined coordinate STAR not written\n")

    progress(f"[batch] done: success={ok}, missing_mrc={missing_mrc}, failed={failed}, total={len(star_files)}")


if __name__ == "__main__":
    main()
