#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np


def progress(msg: str):
    print(msg, flush=True)


def read_star_table(star_path: Path):
    import starfile

    table = starfile.read(star_path)
    if isinstance(table, dict):
        if not table:
            raise ValueError(f"{star_path} did not contain any STAR data blocks")
        if "particles" in table:
            table = table["particles"]
        else:
            table = next(iter(table.values()))
    return table


def get_column(table, preferred: str):
    if preferred in table.columns:
        return np.asarray(table[preferred].to_numpy(dtype=float), dtype=float)
    alt = preferred[1:] if preferred.startswith("_") else f"_{preferred}"
    if alt in table.columns:
        return np.asarray(table[alt].to_numpy(dtype=float), dtype=float)
    raise ValueError(f"Missing required column: {preferred}")


def resolve_column(table, preferred: str | None, candidates=(), required=True):
    if preferred:
        names = (preferred,)
    else:
        names = candidates

    for name in names:
        if name in table.columns:
            return name
        alt = name[1:] if name.startswith("_") else f"_{name}"
        if alt in table.columns:
            return alt

    if required:
        if preferred:
            raise ValueError(f"Missing required column: {preferred}")
        joined = ", ".join(candidates)
        raise ValueError(f"Missing required column. Tried: {joined}")
    return None


def load_star_coords(star_path: Path):
    table = read_star_table(star_path)
    x = get_column(table, "_rlnCoordinateX")
    y = get_column(table, "_rlnCoordinateY")
    return np.column_stack((x, y))


def load_topaz_coords(star_path: Path):
    table = read_star_table(star_path)
    x = get_column(table, "_rlnCoordinateX")
    y = get_column(table, "_rlnCoordinateY")
    if "_rlnAutopickFigureOfMerit" in table.columns:
        fom = np.asarray(table["_rlnAutopickFigureOfMerit"].to_numpy(dtype=float), dtype=float)
    elif "rlnAutopickFigureOfMerit" in table.columns:
        fom = np.asarray(table["rlnAutopickFigureOfMerit"].to_numpy(dtype=float), dtype=float)
    else:
        fom = np.full_like(x, np.nan, dtype=float)
    return np.column_stack((x, y)), fom


def farthest_point_pair(points_xy):
    points_xy = np.asarray(points_xy, dtype=float)
    n = len(points_xy)
    if n < 2:
        raise ValueError("At least two points are required")
    diff = points_xy[:, None, :] - points_xy[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    i, j = np.unravel_index(np.argmax(dist2), dist2.shape)
    return int(i), int(j), float(np.sqrt(dist2[i, j]))


def point_to_segment_distance(point_xy, start_xy, end_xy):
    point_xy = np.asarray(point_xy, dtype=float)
    start_xy = np.asarray(start_xy, dtype=float)
    end_xy = np.asarray(end_xy, dtype=float)
    seg = end_xy - start_xy
    seg_len2 = float(seg @ seg)
    if seg_len2 <= 0.0:
        return float(np.hypot(*(point_xy - start_xy))), 0.0
    t = float(((point_xy - start_xy) @ seg) / seg_len2)
    t = min(1.0, max(0.0, t))
    closest = start_xy + t * seg
    return float(np.hypot(*(point_xy - closest))), t


def load_endpoint_segments(endpoint_star_path):
    coords = load_star_coords(endpoint_star_path)
    segments = []
    for i in range(0, len(coords) - 1, 2):
        segments.append((coords[i], coords[i + 1], i // 2))
    return segments


def match_singleton_to_parent_segment(point_xy, parent_segments, max_distance_px):
    best = None
    for start_xy, end_xy, segment_index in parent_segments:
        distance, projection = point_to_segment_distance(point_xy, start_xy, end_xy)
        score = (distance, abs(projection - 0.5))
        if best is None or score < best[0]:
            best = (score, start_xy, end_xy, segment_index, projection)
    if best is None or best[0][0] > max_distance_px:
        return None
    score, start_xy, end_xy, segment_index, projection = best
    return {
        "start": start_xy,
        "end": end_xy,
        "segment_index": segment_index,
        "distance": score[0],
        "projection": projection,
    }


def filament_endpoint_rows_from_star(
    star_path: Path,
    filament_column: str | None = None,
    micrograph_column: str | None = None,
    min_points: int = 2,
    singleton_parent_endpoint_dir: Path | None = None,
    singleton_max_distance_px: float = 5.0,
):
    table = read_star_table(star_path)
    x_col = resolve_column(table, "_rlnCoordinateX")
    y_col = resolve_column(table, "_rlnCoordinateY")
    filament_col = resolve_column(
        table,
        filament_column,
        candidates=(
            "_rlnHelicalTubeID",
            "_rlnFilamentID",
            "_rlnFilamentNumber",
            "_rlnTubeID",
            "_rlnHelicalTubeName",
        ),
    )
    micrograph_col = resolve_column(
        table,
        micrograph_column,
        candidates=("_rlnMicrographName", "_rlnMicrographMovieName"),
        required=False,
    )

    group_cols = [filament_col]
    if micrograph_col is not None:
        group_cols = [micrograph_col, filament_col]
    group_by = group_cols[0] if len(group_cols) == 1 else group_cols

    rows = []
    skipped = 0
    multi_particle_filaments = 0
    singleton_groups = 0
    singleton_matched = 0
    singleton_unmatched = 0
    singleton_duplicate_parent_matches = 0
    used_singleton_parent_segments = set()
    parent_segment_cache = {}
    singleton_parent_endpoint_dir = (
        Path(singleton_parent_endpoint_dir) if singleton_parent_endpoint_dir is not None else None
    )
    grouped = table.groupby(group_by, sort=False)
    for group_key, group in grouped:
        coords = np.column_stack(
            (
                np.asarray(group[x_col].to_numpy(dtype=float), dtype=float),
                np.asarray(group[y_col].to_numpy(dtype=float), dtype=float),
            )
        )

        if micrograph_col is None:
            micrograph_name = None
            filament_id = group_key
        else:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            micrograph_name = group_key[0]
            filament_id = group_key[1]

        if len(group) >= min_points:
            i, j, distance = farthest_point_pair(coords)
            multi_particle_filaments += 1

            for endpoint_index, point_index in enumerate((i, j), start=1):
                rows.append(
                    {
                        "micrograph_name": micrograph_name,
                        "filament_id": filament_id,
                        "endpoint_index": endpoint_index,
                        "x": float(coords[point_index, 0]),
                        "y": float(coords[point_index, 1]),
                        "point_count": int(len(group)),
                        "endpoint_distance": distance,
                        "source": "classified_farthest_pair",
                    }
                )
            continue

        if len(group) == 1 and singleton_parent_endpoint_dir is not None and micrograph_name is not None:
            singleton_groups += 1
            endpoint_star_path = singleton_parent_endpoint_dir / micrograph_endpoint_star_name(micrograph_name)
            if endpoint_star_path not in parent_segment_cache:
                if endpoint_star_path.exists():
                    parent_segment_cache[endpoint_star_path] = load_endpoint_segments(endpoint_star_path)
                else:
                    parent_segment_cache[endpoint_star_path] = []
            parent_segments = parent_segment_cache[endpoint_star_path]
            match = match_singleton_to_parent_segment(coords[0], parent_segments, singleton_max_distance_px)
            if match is None:
                singleton_unmatched += 1
                skipped += 1
                continue

            parent_key = (micrograph_name, int(match["segment_index"]))
            if parent_key in used_singleton_parent_segments:
                singleton_duplicate_parent_matches += 1
                skipped += 1
                continue
            used_singleton_parent_segments.add(parent_key)
            singleton_matched += 1

            for endpoint_index, xy in enumerate((match["start"], match["end"]), start=1):
                rows.append(
                    {
                        "micrograph_name": micrograph_name,
                        "filament_id": filament_id,
                        "endpoint_index": endpoint_index,
                        "x": float(xy[0]),
                        "y": float(xy[1]),
                        "point_count": int(len(group)),
                        "endpoint_distance": float(np.hypot(*(match["end"] - match["start"]))),
                        "source": "singleton_parent_segment",
                        "singleton_parent_distance": float(match["distance"]),
                        "singleton_parent_segment_index": int(match["segment_index"]),
                    }
                )
            continue

        skipped += 1

    return rows, {
        "input_rows": int(len(table)),
        "filaments_total": int(grouped.ngroups),
        "filaments_written": int(len(rows) // 2),
        "filaments_skipped": int(skipped),
        "multi_particle_filaments": int(multi_particle_filaments),
        "singleton_groups": int(singleton_groups),
        "singleton_matched": int(singleton_matched),
        "singleton_unmatched": int(singleton_unmatched),
        "singleton_duplicate_parent_matches": int(singleton_duplicate_parent_matches),
        "filament_column": filament_col,
        "micrograph_column": micrograph_col,
    }


def rel_or_abs(path, base):
    path = Path(path).resolve()
    base = Path(base).resolve()
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def micrograph_endpoint_star_name(micrograph_name):
    stem = Path(str(micrograph_name)).with_suffix("").name
    return f"{stem}_filfind_endpoints.star"


def write_farthest_filament_endpoints_star(path: Path, rows, include_micrograph=None):
    if include_micrograph is None:
        has_micrograph = any(row["micrograph_name"] is not None for row in rows)
    else:
        has_micrograph = bool(include_micrograph)

    with path.open("w", encoding="utf-8") as f:
        f.write("\n")
        f.write("# version 50001\n\n")
        f.write("data_\n\n")
        f.write("loop_ \n")

        col_idx = 1
        if has_micrograph:
            f.write(f"_rlnMicrographName #{col_idx} \n")
            col_idx += 1
        f.write(f"_rlnCoordinateX #{col_idx} \n")
        col_idx += 1
        f.write(f"_rlnCoordinateY #{col_idx} \n")
        col_idx += 1
        f.write(f"_rlnParticleSelectionType #{col_idx} \n")
        col_idx += 1
        f.write(f"_rlnAnglePsi #{col_idx} \n")
        col_idx += 1
        f.write(f"_rlnAutopickFigureOfMerit #{col_idx} \n")
        col_idx += 1
        f.write(f"_rlnHelicalTubeID #{col_idx} \n")

        for row in rows:
            parts = []
            if has_micrograph:
                parts.append(str(row["micrograph_name"] if row["micrograph_name"] is not None else "None"))
            parts.extend(
                (
                    f"{row['x']:11.6f}",
                    f"{row['y']:11.6f}",
                    "2",
                    "-999.00000",
                    "-999.00000",
                    str(row["filament_id"]),
                )
            )
            f.write(" ".join(parts) + " \n")


def write_coordinate_files_star(path, rows, rel_base):
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


def write_micrograph_endpoint_stars(out_dir, rows, coordinate_files_star=None, rel_base=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if coordinate_files_star is None:
        coordinate_files_star = out_dir / "filfind_coordinate_files.star"
    else:
        coordinate_files_star = Path(coordinate_files_star)
    coordinate_files_star.parent.mkdir(parents=True, exist_ok=True)
    rel_base = Path(rel_base) if rel_base is not None else coordinate_files_star.parent

    rows_by_micrograph = {}
    for row in rows:
        micrograph_name = row["micrograph_name"]
        if micrograph_name is None:
            raise ValueError("Per-micrograph output requires a micrograph column in the input STAR")
        rows_by_micrograph.setdefault(micrograph_name, []).append(row)

    coordinate_rows = []
    endpoint_paths = []
    for micrograph_name in sorted(rows_by_micrograph):
        endpoint_path = out_dir / micrograph_endpoint_star_name(micrograph_name)
        write_farthest_filament_endpoints_star(
            endpoint_path,
            rows_by_micrograph[micrograph_name],
            include_micrograph=False,
        )
        coordinate_rows.append((micrograph_name, endpoint_path))
        endpoint_paths.append(endpoint_path)

    write_coordinate_files_star(coordinate_files_star, coordinate_rows, rel_base=rel_base)
    return {
        "coordinate_files_star": coordinate_files_star,
        "endpoint_paths": endpoint_paths,
        "micrograph_count": len(endpoint_paths),
    }


def apply_fom_filter(coords, fom, fom_min=None, fom_max=None):
    keep = np.ones(len(coords), dtype=bool)
    if fom_min is not None:
        keep &= np.isnan(fom) | (fom >= fom_min)
    if fom_max is not None:
        keep &= np.isnan(fom) | (fom <= fom_max)
    return coords[keep], fom[keep]


def read_mrc_2d(mrc_path: Path):
    import mrcfile

    with mrcfile.open(mrc_path, permissive=True) as mrc:
        data = np.asarray(mrc.data)

    if data.ndim == 2:
        img = data
    elif data.ndim == 3:
        img = data[0]
    else:
        raise ValueError(f"Unsupported MRC dimensions: {data.shape}")

    img = np.asarray(img, dtype=np.float32)
    lo = np.percentile(img, 2)
    hi = np.percentile(img, 98)
    if hi <= lo:
        hi = lo + 1.0
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return img


def compute_pairwise(
    coords,
    report_every: int = 500_000,
    progress_fn: Callable[[str], None] | None = progress,
):
    n = len(coords)
    if n < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=float)
    i_idx, j_idx = np.triu_indices(n, k=1)
    total = len(i_idx)
    dist = np.empty(total, dtype=float)
    start = time.time()
    if progress_fn:
        progress_fn(f"[pairwise] total pairs: {total}")
    for start_idx in range(0, total, report_every):
        end_idx = min(start_idx + report_every, total)
        dx = coords[i_idx[start_idx:end_idx], 0] - coords[j_idx[start_idx:end_idx], 0]
        dy = coords[i_idx[start_idx:end_idx], 1] - coords[j_idx[start_idx:end_idx], 1]
        dist[start_idx:end_idx] = np.hypot(dx, dy)
        if progress_fn:
            done = end_idx
            pct = 100.0 * done / total
            elapsed = time.time() - start
            progress_fn(f"[pairwise] {done}/{total} ({pct:.1f}%) elapsed {elapsed:.1f}s")
    return i_idx, j_idx, dist


def compute_nearest_neighbor_distances(
    coords,
    report_every: int = 1000,
    progress_fn: Callable[[str], None] | None = progress,
):
    n = len(coords)
    if n < 2:
        return np.empty(0, dtype=float)
    mins = np.full(n, np.inf, dtype=float)
    start = time.time()
    if progress_fn:
        progress_fn(f"[nearest] total picks: {n}")
    for i in range(n):
        dx = coords[:, 0] - coords[i, 0]
        dy = coords[:, 1] - coords[i, 1]
        d = np.hypot(dx, dy)
        d[i] = np.inf
        mins[i] = d.min()
        if progress_fn and (((i + 1) % report_every == 0) or (i + 1 == n)):
            pct = 100.0 * (i + 1) / n
            elapsed = time.time() - start
            progress_fn(f"[nearest] {i + 1}/{n} ({pct:.1f}%) elapsed {elapsed:.1f}s")
    return mins


def describe_distribution(name, arr):
    if len(arr) == 0:
        return f"{name}: empty"
    p = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    return (
        f"{name}: n={len(arr)} min={arr.min():.3f} max={arr.max():.3f} "
        f"p1={p[0]:.3f} p5={p[1]:.3f} p25={p[2]:.3f} p50={p[3]:.3f} "
        f"p75={p[4]:.3f} p95={p[5]:.3f} p99={p[6]:.3f}"
    )


def select_candidate_pairs_by_mean_std(i_idx, j_idx, dist, k_std):
    if len(dist) == 0:
        return i_idx, j_idx, dist, np.nan, np.nan, np.nan
    mean_d = float(np.mean(dist))
    std_d = float(np.std(dist))
    cutoff = max(0.0, mean_d - k_std * std_d)
    keep = dist <= cutoff
    return i_idx[keep], j_idx[keep], dist[keep], mean_d, std_d, cutoff
