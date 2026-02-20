#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import mrcfile
import numpy as np
import starfile


def progress(msg: str):
    print(msg, flush=True)


def read_star_table(star_path: Path):
    table = starfile.read(star_path)
    if isinstance(table, dict):
        if not table:
            raise ValueError(f"{star_path} did not contain any STAR data blocks")
        table = next(iter(table.values()))
    return table


def get_column(table, preferred: str):
    if preferred in table.columns:
        return np.asarray(table[preferred].to_numpy(dtype=float), dtype=float)
    alt = preferred[1:] if preferred.startswith("_") else f"_{preferred}"
    if alt in table.columns:
        return np.asarray(table[alt].to_numpy(dtype=float), dtype=float)
    raise ValueError(f"Missing required column: {preferred}")


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


def apply_fom_filter(coords, fom, fom_min=None, fom_max=None):
    keep = np.ones(len(coords), dtype=bool)
    if fom_min is not None:
        keep &= np.isnan(fom) | (fom >= fom_min)
    if fom_max is not None:
        keep &= np.isnan(fom) | (fom <= fom_max)
    return coords[keep], fom[keep]


def read_mrc_2d(mrc_path: Path):
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
