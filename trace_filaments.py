#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from filfind_lib import (
    apply_fom_filter,
    compute_pairwise,
    load_topaz_coords,
    progress,
    read_mrc_2d,
    select_candidate_pairs_by_mean_std,
)

MIN_FILAMENT_POINTS = 2


def fit_line_rms(points_xy):
    # Fit an infinite line by PCA/SVD and return RMS orthogonal distance.
    if len(points_xy) < 2:
        return 0.0
    center = points_xy.mean(axis=0)
    centered = points_xy - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    proj = centered @ direction
    closest = np.outer(proj, direction)
    resid = centered - closest
    d = np.hypot(resid[:, 0], resid[:, 1])
    return float(np.sqrt(np.mean(d * d)))


def build_candidate_graph(coords, k_std, max_neighbors):
    # Build an undirected graph from short candidate pairs.
    # For each point, only keep its nearest max_neighbors edges to cap branching.
    i_idx, j_idx, dist = compute_pairwise(coords, report_every=1_000_000)
    cand_i, cand_j, cand_d, mean_d, std_d, cutoff = select_candidate_pairs_by_mean_std(i_idx, j_idx, dist, k_std)

    n = len(coords)
    if len(cand_i) == 0:
        return [[] for _ in range(n)], (mean_d, std_d, cutoff, len(dist), 0)

    selected = np.zeros(len(cand_i), dtype=bool)
    incident = [[] for _ in range(n)]
    for e, (a, b) in enumerate(zip(cand_i, cand_j)):
        incident[a].append(e)
        incident[b].append(e)

    for node in range(n):
        edges = incident[node]
        if not edges:
            continue
        edges = sorted(edges, key=lambda e: cand_d[e])
        for e in edges[:max_neighbors]:
            selected[e] = True

    keep = np.where(selected)[0]
    ki = cand_i[keep]
    kj = cand_j[keep]
    kd = cand_d[keep]

    adj = [[] for _ in range(n)]
    for a, b, d in zip(ki, kj, kd):
        adj[a].append((int(b), float(d)))
        adj[b].append((int(a), float(d)))

    for node in range(n):
        adj[node].sort(key=lambda t: t[1])

    return adj, (mean_d, std_d, cutoff, len(dist), len(keep))


def choose_extension(chain, at_start, adj, coords, used, max_rms):
    # Pick the best next point for one chain end under line-fit constraints.
    endpoint = chain[0] if at_start else chain[-1]

    best = None
    for neigh, edge_d in adj[endpoint]:
        if neigh in used or neigh in chain:
            continue

        test_chain = [neigh] + chain if at_start else chain + [neigh]
        rms = fit_line_rms(coords[test_chain])
        if len(test_chain) >= 3 and rms > max_rms:
            continue

        score = (rms, edge_d)
        if best is None or score < best[0]:
            best = (score, neigh)

    return None if best is None else best[1]


def grow_chain(seed, adj, coords, used, max_rms):
    # Greedily grow both ends until no legal extension remains.
    chain = [seed]
    changed = True
    while changed:
        changed = False
        left = choose_extension(chain, True, adj, coords, used, max_rms)
        if left is not None:
            chain = [left] + chain
            changed = True

        right = choose_extension(chain, False, adj, coords, used, max_rms)
        if right is not None:
            chain = chain + [right]
            changed = True
    return chain


def save_growth_movie_gif(mrc_path, out_path, coords, frames, dpi=150, fps=8):
    # Visualize greedy chain growth as an animated GIF.
    if not frames:
        return
    img = read_mrc_2d(mrc_path)
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    cmap = plt.get_cmap("tab20")

    def draw_chain(chain, color, active=False):
        pts = coords[np.asarray(chain, dtype=int)]
        if len(pts) == 0:
            return
        start = pts[0]
        end = pts[-1]
        ax.scatter(pts[:, 0], pts[:, 1], s=16 if not active else 22, c=[color], alpha=0.9)
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2.0 if not active else 2.6, alpha=0.95)
        ax.scatter(start[0], start[1], s=60, c=[color], marker="o")
        ax.scatter(end[0], end[1], s=74, c=[color], marker="x", linewidths=1.8)

    def update(frame_idx):
        state = frames[frame_idx]
        completed = state["completed"]
        active = state["active"]

        ax.clear()
        ax.imshow(img, cmap="gray", origin="upper")
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=8,
            facecolors="none",
            edgecolors="#4FC3F7",
            linewidths=0.35,
            alpha=0.35,
        )

        for fid, chain in enumerate(completed):
            draw_chain(chain, cmap(fid % 20), active=False)
        if active is not None and len(active) >= 1:
            draw_chain(active, "#FFEB3B", active=True)

        ax.set_title(f"{mrc_path.name}\nGreedy filament growth (frame {frame_idx + 1}/{len(frames)})")
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

    anim = FuncAnimation(fig, update, frames=len(frames), interval=max(1, int(1000 / max(1, fps))), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def write_filament_endpoints_star(path, coords, filaments):
    # Emit only start/end coordinates in RELION-style STAR format (matching manual pick columns).
    rows = []
    for chain in filaments:
        if len(chain) < 1:
            continue
        rows.append(coords[int(chain[0])])
        if len(chain) >= 2:
            rows.append(coords[int(chain[-1])])

    with path.open("w", encoding="utf-8") as f:
        f.write("\n")
        f.write("# version 50001\n\n")
        f.write("data_\n\n")
        f.write("loop_ \n")
        f.write("_rlnCoordinateX #1 \n")
        f.write("_rlnCoordinateY #2 \n")
        f.write("_rlnParticleSelectionType #3 \n")
        f.write("_rlnAnglePsi #4 \n")
        f.write("_rlnAutopickFigureOfMerit #5 \n")
        for xy in rows:
            f.write(f" {xy[0]:11.6f}  {xy[1]:11.6f}            2   -999.00000   -999.00000 \n")


def plot_filaments(mrc_path, out_path, coords, filaments, dpi=220, show=False):
    img = read_mrc_2d(mrc_path)
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(img, cmap="gray", origin="upper")

    cmap = plt.get_cmap("tab20")
    for fid, chain in enumerate(filaments):
        pts = coords[np.asarray(chain, dtype=int)]
        color = cmap(fid % 20)
        start = pts[0]
        end = pts[-1]
        # Keep individual assigned picks visible, but connect only start/end.
        ax.scatter(pts[:, 0], pts[:, 1], s=14, c=[color], alpha=0.9)
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2.0, alpha=0.95)
        ax.scatter(start[0], start[1], s=56, c=[color], marker="o")
        ax.scatter(end[0], end[1], s=70, c=[color], marker="x", linewidths=1.8)

    ax.set_title(f"{mrc_path.name}\\nFilament picks with start/end connecting lines")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


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

    coords, fom = load_topaz_coords(args.autopick)
    coords, fom = apply_fom_filter(coords, fom, args.fom_min, args.fom_max)
    n = len(coords)
    progress(f"[load] filtered picks: {n}")

    adj, graph_stats = build_candidate_graph(coords, args.candidate_k_std, args.max_neighbors)
    mean_d, std_d, cutoff, all_pairs, kept_pairs = graph_stats
    progress(
        "[graph] candidate rule: "
        f"distance <= mean - k*std = {mean_d:.3f} - {args.candidate_k_std:.3f}*{std_d:.3f} = {cutoff:.3f}"
    )
    progress(f"[graph] kept graph edges: {kept_pairs} / {all_pairs}")
    progress(f"[trace] minimum filament length is fixed at {MIN_FILAMENT_POINTS} points")

    start_scores = np.nan_to_num(fom, nan=-np.inf)
    seed_order = np.argsort(-start_scores)

    # One-use policy: once a point is assigned to a filament, it cannot be reused.
    used = set()
    filaments = []
    growth_frames = []
    if args.growth_movie_out is not None:
        growth_frames.append({"completed": [], "active": None})

    for seed in seed_order:
        seed = int(seed)
        if seed in used:
            continue
        if not adj[seed]:
            continue

        chain = grow_chain(seed, adj, coords, used, args.max_line_rms)
        if args.growth_movie_out is not None and len(chain) >= 1:
            completed_snapshot = [list(c) for c in filaments]
            for k in range(1, len(chain) + 1):
                growth_frames.append({"completed": completed_snapshot, "active": list(chain[:k])})
        if len(chain) >= MIN_FILAMENT_POINTS:
            filaments.append(chain)
            used.update(chain)
            progress(
                f"[trace] filament {len(filaments)-1}: points={len(chain)} "
                f"start={chain[0]} end={chain[-1]}"
            )
            if args.growth_movie_out is not None:
                growth_frames.append({"completed": [list(c) for c in filaments], "active": None})
        else:
            # Mark isolated/failed seeds as consumed so they are not recycled later.
            used.update(chain)

    out_prefix = args.mrc.with_suffix("").parent / f"{args.mrc.with_suffix('').name}_filfind"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    overlay_path = None if args.no_save_overlay else (out_prefix.parent / f"{out_prefix.name}_overlay.png")
    plot_filaments(
        args.mrc,
        overlay_path,
        coords,
        filaments,
        dpi=args.dpi,
        show=args.show,
    )

    endpoints_star_path = out_prefix.parent / f"{out_prefix.name}_endpoints.star"
    write_filament_endpoints_star(endpoints_star_path, coords, filaments)

    npz_path = out_prefix.with_suffix(".npz")
    if args.save_npz:
        fil_starts = np.array([c[0] for c in filaments], dtype=int) if filaments else np.empty(0, dtype=int)
        fil_lengths = np.array([len(c) for c in filaments], dtype=int) if filaments else np.empty(0, dtype=int)
        np.savez_compressed(
            npz_path,
            coordinates_xy=coords,
            fom=fom,
            filament_starts=fil_starts,
            filament_lengths=fil_lengths,
            candidate_rule_mean=np.array([mean_d]),
            candidate_rule_std=np.array([std_d]),
            candidate_rule_cutoff=np.array([cutoff]),
            max_line_rms=np.array([args.max_line_rms]),
        )

    used_count = sum(len(c) for c in filaments)
    progress(f"[done] filaments kept: {len(filaments)}")
    progress(f"[done] picks assigned to filaments: {used_count} / {n}")
    if overlay_path is not None:
        progress(f"[done] saved overlay: {overlay_path.resolve()}")
    else:
        progress("[done] overlay PNG disabled (--no-save-overlay)")
    if args.growth_movie_out is not None:
        movie_out = args.growth_movie_out
        if movie_out.suffix.lower() != ".gif":
            movie_out = movie_out.with_suffix(".gif")
        save_growth_movie_gif(args.mrc, movie_out, coords, growth_frames, dpi=args.dpi, fps=args.growth_movie_fps)
        progress(f"[done] saved growth movie: {movie_out.resolve()}")
    progress(f"[done] saved filament endpoints STAR: {endpoints_star_path.resolve()}")
    if args.save_npz:
        progress(f"[done] saved trace summary: {npz_path.resolve()}")
    else:
        progress("[done] trace summary NPZ disabled (use --save-npz to enable)")


if __name__ == "__main__":
    main()
