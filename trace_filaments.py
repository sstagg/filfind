#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def angle_ok(endpoint_xy, inboard_xy, cand_xy, max_turn_deg):
    # Limit sharp turns while extending a chain at either endpoint.
    if max_turn_deg is None:
        return True
    v_ref = endpoint_xy - inboard_xy
    v_new = cand_xy - endpoint_xy
    n_ref = np.linalg.norm(v_ref)
    n_new = np.linalg.norm(v_new)
    if n_ref == 0 or n_new == 0:
        return True
    cosang = float(np.dot(v_ref, v_new) / (n_ref * n_new))
    cosang = max(-1.0, min(1.0, cosang))
    turn_deg = np.degrees(np.arccos(cosang))
    return turn_deg <= max_turn_deg


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


def choose_extension(chain, at_start, adj, coords, used, max_rms, max_turn_deg):
    # Pick the best next point for one chain end under angle + line-fit constraints.
    endpoint = chain[0] if at_start else chain[-1]
    inboard = chain[1] if (at_start and len(chain) > 1) else (chain[-2] if len(chain) > 1 else None)

    best = None
    for neigh, edge_d in adj[endpoint]:
        if neigh in used or neigh in chain:
            continue

        if inboard is not None:
            if not angle_ok(coords[endpoint], coords[inboard], coords[neigh], max_turn_deg):
                continue

        test_chain = [neigh] + chain if at_start else chain + [neigh]
        rms = fit_line_rms(coords[test_chain])
        if len(test_chain) >= 3 and rms > max_rms:
            continue

        score = (rms, edge_d)
        if best is None or score < best[0]:
            best = (score, neigh)

    return None if best is None else best[1]


def grow_chain(seed, adj, coords, used, max_rms, max_turn_deg):
    # Greedily grow both ends until no legal extension remains.
    chain = [seed]
    changed = True
    while changed:
        changed = False
        left = choose_extension(chain, True, adj, coords, used, max_rms, max_turn_deg)
        if left is not None:
            chain = [left] + chain
            changed = True

        right = choose_extension(chain, False, adj, coords, used, max_rms, max_turn_deg)
        if right is not None:
            chain = chain + [right]
            changed = True
    return chain


def endpoint_direction(coords, chain, at_start):
    # Tangent direction at one endpoint for directional merge checks.
    if len(chain) < 2:
        return None
    if at_start:
        p0 = coords[chain[0]]
        p1 = coords[chain[1]]
        vec = p0 - p1
    else:
        p0 = coords[chain[-1]]
        p1 = coords[chain[-2]]
        vec = p0 - p1
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return vec / norm


def direction_angle_ok(v1, v2, max_angle_deg):
    if max_angle_deg is None or v1 is None or v2 is None:
        return True
    cosang = float(np.dot(v1, v2))
    cosang = max(-1.0, min(1.0, cosang))
    ang = np.degrees(np.arccos(cosang))
    return ang <= max_angle_deg


def best_merge_candidate(chains, coords, max_join_dist, max_join_turn_deg, max_merge_rms):
    # Try all endpoint pairings and pick the shortest valid join.
    best = None
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            a = chains[i]
            b = chains[j]
            orientations = [
                (a, b),  # a end -> b start
                (a, b[::-1]),  # a end -> b end
                (a[::-1], b),  # a start -> b start
                (a[::-1], b[::-1]),  # a start -> b end
            ]

            for a_oriented, b_oriented in orientations:
                a_end = coords[a_oriented[-1]]
                b_start = coords[b_oriented[0]]
                join_dist = float(np.linalg.norm(a_end - b_start))
                if join_dist > max_join_dist:
                    continue

                va = endpoint_direction(coords, a_oriented, at_start=False)
                vb = endpoint_direction(coords, b_oriented, at_start=True)
                if not direction_angle_ok(va, vb, max_join_turn_deg):
                    continue

                merged = a_oriented + b_oriented
                if fit_line_rms(coords[merged]) > max_merge_rms:
                    continue

                score = (join_dist, len(merged))
                if best is None or score < best[0]:
                    best = (score, i, j, merged)
    return best


def merge_filaments(chains, coords, max_join_dist, max_join_turn_deg, max_merge_rms):
    # Greedily merge the closest valid chain pair until no merge remains.
    chains = [list(c) for c in chains]
    merges = 0
    while True:
        cand = best_merge_candidate(chains, coords, max_join_dist, max_join_turn_deg, max_merge_rms)
        if cand is None:
            break
        _, i, j, merged = cand
        keep = []
        for idx, c in enumerate(chains):
            if idx not in (i, j):
                keep.append(c)
        keep.append(merged)
        chains = keep
        merges += 1
    return chains, merges


def write_assignments_csv(path, coords, fom, filaments):
    rows = []
    for fid, chain in enumerate(filaments):
        for order, idx in enumerate(chain):
            rows.append((idx, coords[idx, 0], coords[idx, 1], fom[idx], fid, order))

    rows.sort(key=lambda r: r[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["point_index", "x", "y", "fom", "filament_id", "order_in_filament"])
        w.writerows(rows)


def plot_filaments(mrc_path, out_path, coords, filaments, dpi=220, show=False, endpoints_only=False):
    img = read_mrc_2d(mrc_path)
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(img, cmap="gray", origin="upper")

    if endpoints_only and len(coords):
        # Show all picks above FOM cutoff as context, independent of filament assignment.
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=10,
            facecolors="none",
            edgecolors="#4FC3F7",
            linewidths=0.45,
            alpha=0.5,
        )

    cmap = plt.get_cmap("tab20")
    for fid, chain in enumerate(filaments):
        pts = coords[np.asarray(chain, dtype=int)]
        color = cmap(fid % 20)
        if endpoints_only:
            start = pts[0]
            end = pts[-1]
            ax.scatter(pts[:, 0], pts[:, 1], s=12, c=[color], alpha=0.75)
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2.0, alpha=0.95)
            ax.scatter(start[0], start[1], s=56, c=[color], marker="o")
            ax.scatter(end[0], end[1], s=70, c=[color], marker="x", linewidths=1.8)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.0, alpha=0.95)
            ax.scatter(pts[:, 0], pts[:, 1], s=14, c=[color], alpha=0.9)

    if endpoints_only:
        ax.set_title(f"{mrc_path.name}\\nFilament endpoints (start/end with connecting line)")
    else:
        ax.set_title(f"{mrc_path.name}\\nTraced filaments")
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
    parser.add_argument("--max-turn-deg", type=float, default=70.0, help="Max local turning angle at chain ends")
    parser.add_argument(
        "--merge-endpoint-max-dist",
        type=float,
        default=220.0,
        help="Max distance between chain endpoints for post-pass merge",
    )
    parser.add_argument(
        "--merge-max-turn-deg",
        type=float,
        default=35.0,
        help="Max direction mismatch (deg) at merge junction",
    )
    parser.add_argument(
        "--merge-max-rms",
        type=float,
        default=None,
        help="Max RMS line deviation for merged chains (default: --max-line-rms)",
    )
    parser.add_argument("--out-prefix", type=Path, default=Path("filament_trace"), help="Output prefix")
    parser.add_argument("--show", action="store_true", help="Show interactive overlay window")
    parser.add_argument("--no-save-overlay", action="store_true", help="Do not save overlay PNG")
    parser.add_argument(
        "--endpoints-only",
        action="store_true",
        help="Plot only filament start/end points with a line between them",
    )
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
    for seed in seed_order:
        seed = int(seed)
        if seed in used:
            continue
        if not adj[seed]:
            continue

        chain = grow_chain(seed, adj, coords, used, args.max_line_rms, args.max_turn_deg)
        if len(chain) >= MIN_FILAMENT_POINTS:
            filaments.append(chain)
            used.update(chain)
            progress(
                f"[trace] filament {len(filaments)-1}: points={len(chain)} "
                f"start={chain[0]} end={chain[-1]}"
            )
        else:
            # Mark isolated/failed seeds as consumed so they are not recycled later.
            used.update(chain)

    merge_rms = args.max_line_rms if args.merge_max_rms is None else args.merge_max_rms
    pre_merge_count = len(filaments)
    filaments, merge_count = merge_filaments(
        filaments,
        coords,
        max_join_dist=args.merge_endpoint_max_dist,
        max_join_turn_deg=args.merge_max_turn_deg,
        max_merge_rms=merge_rms,
    )
    progress(
        f"[merge] endpoint max dist={args.merge_endpoint_max_dist:.1f}, "
        f"max turn={args.merge_max_turn_deg:.1f}, max rms={merge_rms:.1f}"
    )
    progress(f"[merge] merged {merge_count} times: filaments {pre_merge_count} -> {len(filaments)}")

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    overlay_path = None if args.no_save_overlay else (out_prefix.parent / f"{out_prefix.name}_overlay.png")
    plot_filaments(
        args.mrc,
        overlay_path,
        coords,
        filaments,
        dpi=args.dpi,
        show=args.show,
        endpoints_only=args.endpoints_only,
    )

    csv_path = out_prefix.parent / f"{out_prefix.name}_assignments.csv"
    write_assignments_csv(csv_path, coords, fom, filaments)

    npz_path = out_prefix.with_suffix(".npz")
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
        max_turn_deg=np.array([args.max_turn_deg]),
    )

    used_count = sum(len(c) for c in filaments)
    progress(f"[done] filaments kept: {len(filaments)}")
    progress(f"[done] picks assigned to filaments: {used_count} / {n}")
    if overlay_path is not None:
        progress(f"[done] saved overlay: {overlay_path.resolve()}")
    else:
        progress("[done] overlay PNG disabled (--no-save-overlay)")
    progress(f"[done] saved assignments: {csv_path.resolve()}")
    progress(f"[done] saved trace summary: {npz_path.resolve()}")


if __name__ == "__main__":
    main()
