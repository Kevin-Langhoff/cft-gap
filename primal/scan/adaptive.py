"""Adaptive refinement of the boundary between candidate and excluded regions.

Running the full search on every point of a fine, uniform (c, delta_gap)
grid is wasteful: away from the boundary between the CFT-candidate
region and the excluded region, the loss is either clearly low or
clearly high, and a coarse grid already tells you that. The interesting
compute should go into pinning down *where the boundary is*.

This module does that with a quadtree-style refinement, in the same
spirit as marching squares for contour plotting:

1. Start from a coarse uniform grid of rectangular cells.
2. Run the search once at each of the cheap seed-grid points.
3. For each cell, look at the loss at its 4 corners. If the corners
   straddle the classification threshold (or sit close enough to it
   that single-seed noise could be masking a real straddle), the
   boundary passes through this cell — subdivide it into 4 children and
   evaluate the 5 new points introduced (the cell's edge midpoints and
   center). If the corners are all clearly on one side, the cell is
   done; keep it as-is.
4. Repeat until a maximum depth or an evaluation budget is reached.

At the final refinement round, boundary-adjacent points are evaluated
with several independent random seeds and aggregated (default:
minimum), since a single 300-step run of the search is a chaotic
trajectory and can land in a worse local basin by chance — averaging
multiple attempts gives a more reliable read of whether a consistent
spectrum exists nearby.

This module keeps only the refinement algorithm itself. A full
production driver would add checkpointing, resumability, and
distributed-cluster plumbing around this core loop; those are cut here
since they aren't part of the method itself.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..optimize import run_batched_survey
from ..resources import RUNS_DIR, load_fbcb_sqrt2, load_fine_test_taus, resolve_fbcb_cut
from ..spectrum import build_survey_point

# The classification threshold, in log1p(loss) units (log(1 + loss) with
# loss = 25 is the reference "clearly excluded" level used throughout).
LEVEL_RAW = 25.0
LEVEL_LOG1P = float(np.log1p(LEVEL_RAW))


@dataclass
class Cell:
    c_lo: float; c_hi: float; g_lo: float; g_hi: float
    depth: int

    def corners(self) -> list[tuple[float, float]]:
        return [
            (self.c_lo, self.g_lo),
            (self.c_hi, self.g_lo),
            (self.c_hi, self.g_hi),
            (self.c_lo, self.g_hi),
        ]


@dataclass
class PointRecord:
    c: float; g: float
    L_seeds: list[float] = field(default_factory=list)
    L_seeds_raw: list[float] = field(default_factory=list)
    n_seeds: int = 0
    depth_first_seen: int = 0

    def L(self, agg: str) -> float:
        if not self.L_seeds:
            return float("nan")
        a = np.asarray(self.L_seeds, dtype=np.float64)
        if agg == "min":
            return float(a.min())
        if agg == "median":
            return float(np.median(a))
        if agg == "mean":
            return float(a.mean())
        raise ValueError(f"unknown agg {agg!r}")

    def L_raw(self, agg: str) -> float:
        if not self.L_seeds_raw:
            return float("nan")
        a = np.asarray(self.L_seeds_raw, dtype=np.float64)
        if agg == "min":
            return float(a.min())
        if agg == "median":
            return float(np.median(a))
        return float(a.mean())


def _key(c: float, g: float) -> tuple[float, float]:
    return (round(c, 9), round(g, 9))


def is_boundary_cell(
    corner_Ls: list[float] | np.ndarray,
    *,
    level: float = LEVEL_LOG1P,
    noise_band: float,
) -> bool:
    """True if this cell's corners straddle the classification level, or
    sit within ``noise_band`` of it (a single-seed loss can be noisy
    enough near the boundary that a cell not-quite-straddling could
    still contain it). NaN corners (a corner not yet evaluated) => False."""
    arr = np.asarray(corner_Ls, dtype=np.float64)
    if arr.size == 0 or np.any(np.isnan(arr)):
        return False
    straddles = (arr.min() <= level) and (arr.max() >= level)
    within_band = (
        (arr.min() <= level + noise_band)
        and (arr.max() >= level - noise_band)
    )
    return bool(straddles or within_band)


def should_promote_corner(
    L_single: float,
    *,
    level: float = LEVEL_LOG1P,
    noise_band: float,
) -> bool:
    """True if a single-seed corner sits within +/- noise_band of the
    level. Used in the final refinement round to decide which corners
    get topped up to the full multi-seed count. Far-from-contour corners
    are skipped to save compute."""
    if not np.isfinite(L_single):
        return False
    return abs(L_single - level) <= noise_band


def _evaluate_batch(
    pts: list[tuple[float, float]], dmax: int, n_seeds: int,
    fbcb, fine_taus, n_processes: int | None,
    verbose: bool,
    seed_base: int = 0,
    point_seed_offsets: list[int] | None = None,
    num_iter: int = 300,
    k_factor: int = 8,
):
    """Run k=n_seeds independent search attempts per point; return
    (log1p_per_pt, raw_per_pt).

    Flattens (pts x seeds) into a single point list and calls
    ``run_batched_survey`` once rather than once per seed, so the
    worker-pool spawn cost (dominant at small batch sizes) is paid once
    instead of ``n_seeds`` times.
    """
    if not pts:
        return [], []
    n_pts = len(pts)
    k_sv = int(k_factor) * (dmax - 2)
    n_train = 2 * k_sv

    flat_points = []
    flat_seed_offsets = []
    flat_idx = []  # (pt_idx, seed_idx)
    for s in range(n_seeds):
        for i, (c, g) in enumerate(pts):
            pt = build_survey_point(c, g, max_delta=float(dmax))
            flat_points.append(pt)
            if point_seed_offsets is None:
                flat_seed_offsets.append((seed_base + s) * 100_000 + i)
            else:
                flat_seed_offsets.append(int(point_seed_offsets[i]) + s * 100_000)
            flat_idx.append((i, s))

    results = run_batched_survey(
        flat_points,
        reference_base=fbcb,
        fine_test_taus=fine_taus,
        num_iter=int(num_iter),
        lr=0.1,
        k_sv=k_sv,
        rtol=1e-12,
        n_train=n_train,
        n_test_track=48,
        n_processes=n_processes,
        verbose=verbose,
        seed_offset=flat_seed_offsets,
        tau2_max=float(dmax) + 2,
    )

    out_log1p: list[list[float]] = [[None] * n_seeds for _ in pts]
    out_raw: list[list[float]] = [[None] * n_seeds for _ in pts]
    for r, (i, s) in zip(results, flat_idx):
        L = r.best_loss_refined_log1p
        L_raw = r.best_loss_refined
        out_log1p[i][s] = float(L) if L is not None else float("nan")
        out_raw[i][s] = float(L_raw) if L_raw is not None else float("nan")

    return out_log1p, out_raw


def adaptive_contour_scan(
    dmax: int,
    c_lo: float = 1.001,
    c_hi: float = 1.6,
    g_lo: float = 0.04,
    g_hi: float = 0.6,
    n_c0: int = 32,
    n_g0: int = 16,
    max_depth: int = 4,
    seeds: int = 5,
    seeds_bulk: int = 1,
    k_factor: int = 8,
    aggregator: str = "min",
    noise_band: float = 0.5,
    budget: int = 5500,
    fbcb_cut: int | None = None,
    processes: int | None = None,
    verbose: bool = False,
) -> dict[tuple[float, float], PointRecord]:
    """Run the adaptive quadtree refinement and return the per-point cache.

    Parameters mirror the historical CLI: ``seeds`` is the number of
    independent search attempts run at boundary-adjacent points in the
    final refinement round (``seeds_bulk`` elsewhere); ``max_depth`` caps
    the number of subdivision rounds; ``budget`` is a soft cap on the
    total number of unique (c, delta_gap) points evaluated.
    """
    resolved_cut = resolve_fbcb_cut(dmax, fbcb_cut)
    fbcb = load_fbcb_sqrt2(cut=resolved_cut)
    fine_taus = load_fine_test_taus()

    cache: dict[tuple[float, float], PointRecord] = {}

    def get_or_create(c: float, g: float, depth: int) -> PointRecord:
        k = _key(c, g)
        if k not in cache:
            cache[k] = PointRecord(c=c, g=g, depth_first_seen=depth)
        return cache[k]

    # Phase A: coarse single-seed seed grid.
    cs0 = np.linspace(c_lo, c_hi, n_c0 + 1)
    gs0 = np.linspace(g_lo, g_hi, n_g0 + 1)
    seed_pts = [(float(c), float(g)) for c in cs0 for g in gs0]
    if verbose:
        print(f"Phase A: coarse single-seed grid — {len(seed_pts)} points")
    seed_Ls_log1p, seed_Ls_raw = _evaluate_batch(
        seed_pts, dmax, seeds_bulk, fbcb, fine_taus, processes, verbose,
        k_factor=k_factor,
    )
    for (c, g), Lvec, Lvec_raw in zip(seed_pts, seed_Ls_log1p, seed_Ls_raw):
        rec = get_or_create(c, g, depth=0)
        rec.L_seeds.extend(Lvec)
        rec.L_seeds_raw.extend(Lvec_raw)
        rec.n_seeds += len(Lvec)

    # Build initial cell list.
    cells: deque[Cell] = deque()
    for i in range(n_c0):
        for j in range(n_g0):
            cells.append(Cell(
                c_lo=float(cs0[i]), c_hi=float(cs0[i + 1]),
                g_lo=float(gs0[j]), g_hi=float(gs0[j + 1]),
                depth=0,
            ))

    final_cells: list[Cell] = []

    def _eval_and_record(pts_list, n_seeds, seed_base, depth):
        offsets = [seed_base * 100_000 + j for j in range(len(pts_list))]
        Ls, Ls_raw = _evaluate_batch(
            pts_list, dmax, n_seeds, fbcb, fine_taus, processes, verbose,
            point_seed_offsets=offsets, k_factor=k_factor,
        )
        for (c, g), Lvec, Lvec_raw in zip(pts_list, Ls, Ls_raw):
            rec = get_or_create(c, g, depth)
            rec.L_seeds.extend(Lvec)
            rec.L_seeds_raw.extend(Lvec_raw)
            rec.n_seeds += len(Lvec)

    # Phase B+: refine quadtree. Strategy: single-seed everywhere up to
    # and including max_depth - 1, then multi-seed at the final
    # refinement round only. Boundary detection at intermediate depths
    # uses single-seed L (noisy) but the *final* round before subdivision
    # has multi-seed L, giving a clean stable contour where it matters.
    round_num = 0
    while cells and len(cache) < budget:
        round_num += 1
        next_cells: deque[Cell] = deque()
        boundary_new_pts: list[tuple[float, float]] = []
        boundary_new_pts_set: set[tuple[float, float]] = set()
        round_cells = list(cells); cells.clear()
        boundary_cells: list[Cell] = []
        for cell in round_cells:
            corner_Ls = []
            ok = True
            for c, g in cell.corners():
                rec = cache.get(_key(c, g))
                if rec is None or not rec.L_seeds:
                    ok = False; break
                corner_Ls.append(rec.L(aggregator))
            if not ok:
                final_cells.append(cell); continue
            do_refine = (
                is_boundary_cell(corner_Ls, noise_band=noise_band)
                and cell.depth < max_depth
            )
            if not do_refine:
                final_cells.append(cell); continue
            boundary_cells.append(cell)
        if not boundary_cells:
            break

        next_depth = boundary_cells[0].depth + 1
        is_final_round = (next_depth >= max_depth)
        seeds_this_round = seeds if is_final_round else seeds_bulk

        for cell in boundary_cells:
            c_mid = 0.5 * (cell.c_lo + cell.c_hi)
            g_mid = 0.5 * (cell.g_lo + cell.g_hi)
            new_pts = [
                (cell.c_lo, g_mid), (c_mid, cell.g_lo),
                (c_mid, g_mid),
                (cell.c_hi, g_mid), (c_mid, cell.g_hi),
            ]
            for c, g in new_pts:
                if _key(c, g) not in cache and (c, g) not in boundary_new_pts_set:
                    boundary_new_pts.append((c, g))
                    boundary_new_pts_set.add((c, g))

        if verbose:
            print(f"Round {round_num}: {len(boundary_cells)} boundary cells, "
                  f"{len(boundary_new_pts)} new midpoints, k={seeds_this_round}")

        if boundary_new_pts:
            _eval_and_record(
                boundary_new_pts, seeds_this_round,
                seed_base=10 * round_num, depth=next_depth,
            )

        # If this is the final round, also top up boundary-cell corners
        # to k=seeds (so the contour at the highest resolution uses
        # multi-seed L on both new midpoints AND the surrounding corners).
        if is_final_round:
            corners_to_promote = []
            for cell in boundary_cells:
                for c, g in cell.corners():
                    rec = cache.get(_key(c, g))
                    if rec is not None and rec.n_seeds < seeds:
                        if (c, g) not in [tuple(p) for p in corners_to_promote]:
                            corners_to_promote.append((c, g))
            corners_to_promote = [
                (c, g) for (c, g) in corners_to_promote
                if should_promote_corner(
                    cache[_key(c, g)].L(aggregator), noise_band=noise_band,
                )
            ]
            if corners_to_promote and len(cache) < budget:
                extras_needed = max(
                    seeds - cache[_key(c, g)].n_seeds
                    for (c, g) in corners_to_promote
                )
                _eval_and_record(
                    corners_to_promote, extras_needed,
                    seed_base=10 * round_num + 1, depth=next_depth,
                )

        # Subdivide each boundary cell into 4 children for next round.
        for cell in boundary_cells:
            c_mid = 0.5 * (cell.c_lo + cell.c_hi)
            g_mid = 0.5 * (cell.g_lo + cell.g_hi)
            for cl, gl, ch, gh in [
                (cell.c_lo, cell.g_lo, c_mid, g_mid),
                (c_mid, cell.g_lo, cell.c_hi, g_mid),
                (cell.c_lo, g_mid, c_mid, cell.g_hi),
                (c_mid, g_mid, cell.c_hi, cell.g_hi),
            ]:
                next_cells.append(Cell(
                    c_lo=cl, c_hi=ch, g_lo=gl, g_hi=gh,
                    depth=cell.depth + 1,
                ))

        cells = next_cells
        if len(cache) >= budget:
            if verbose:
                print(f"budget {budget} reached at cache size {len(cache)}; stopping")
            for c in cells: final_cells.append(c)
            break

    if verbose:
        print(f"Final cache: {len(cache)} unique points, "
              f"{len(final_cells)} final cells")
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmax", type=int, required=True)
    ap.add_argument("--c-lo", type=float, default=1.001)
    ap.add_argument("--c-hi", type=float, default=1.6)
    ap.add_argument("--g-lo", type=float, default=0.04)
    ap.add_argument("--g-hi", type=float, default=0.6)
    ap.add_argument("--n-c0", type=int, default=32)
    ap.add_argument("--n-g0", type=int, default=16)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5,
                    help="k seeds per boundary point in the final round.")
    ap.add_argument("--seeds-bulk", type=int, default=1,
                    help="Seeds per non-boundary (bulk) point — keep at 1.")
    ap.add_argument("--k-factor", type=int, default=8,
                    help="Search rank K_SV = k_factor*(dmax-2) (n_train=2*K_SV).")
    ap.add_argument("--aggregator", choices=["min", "median", "mean"],
                    default="min", help="Multi-seed aggregator.")
    ap.add_argument("--noise-band", type=float, default=0.5,
                    help="Conservative single-seed noise band in log1p(loss) units.")
    ap.add_argument("--budget", type=int, default=5500,
                    help="Soft cap on unique (c, delta_gap) evaluations.")
    ap.add_argument("--fbcb-cut", type=int, default=None, choices=[6, 8, 12, 16])
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    cache = adaptive_contour_scan(
        dmax=args.dmax, c_lo=args.c_lo, c_hi=args.c_hi,
        g_lo=args.g_lo, g_hi=args.g_hi,
        n_c0=args.n_c0, n_g0=args.n_g0, max_depth=args.max_depth,
        seeds=args.seeds, seeds_bulk=args.seeds_bulk, k_factor=args.k_factor,
        aggregator=args.aggregator, noise_band=args.noise_band,
        budget=args.budget, fbcb_cut=args.fbcb_cut, processes=args.processes,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - t0

    out_records = []
    for rec in cache.values():
        out_records.append({
            "c": rec.c, "delta_gap": rec.g,
            "best_loss_refined_log1p": rec.L(args.aggregator),
            "best_loss_refined": rec.L_raw(args.aggregator),
            "n_seeds": rec.n_seeds,
            "depth_first_seen": rec.depth_first_seen,
            "L_seeds": rec.L_seeds,
            "L_seeds_raw": rec.L_seeds_raw,
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_records, indent=2))
    print(f"Wrote {args.out} ({len(out_records)} pts, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
