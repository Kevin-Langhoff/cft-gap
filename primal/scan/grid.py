"""Run the search over a grid of (c, delta_gap) points — the main scan.

This is the driver that produces the search landscape: for every point
on a grid of central charge and spectral gap, build a candidate
spectrum (``spectrum.build_survey_point``) and run it through the
batched local search (``optimize.run_batched_survey``). The result for
each point is a loss value; points whose loss drops below a fixed
threshold look like they admit a consistent CFT ("candidates"), and
points whose loss stays high are excluded.

The default grid (11 central-charge values x 15 gap values = 165
points, at truncation depth 4) reproduces the reference figure. Denser
grids or deeper truncations can be requested via the CLI flags below;
reference data (the comparison spectrum, the fixed evaluation grid) is
loaded from cached JSON under ``primal/data/`` — nothing here calls out
to an external computer-algebra system at run time.

Hyperparameters: the optimizer's rank ``K_SV = k_factor*(delta_max-2)``
with ``k_factor=8`` by default, and the training-batch size is always
``2*K_SV``. At ``delta_max=4`` this gives ``K_SV=16``, matching the
reference figure exactly.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..optimize import run_batched_survey
from ..resources import RUNS_DIR, load_fbcb_sqrt2, load_fine_test_taus, resolve_fbcb_cut
from ..spectrum import build_survey_point


# Fixed hyperparameters.
NUM_ITER = 300
LR = 0.1
RTOL = 1e-12
N_TEST_TRACK = 48


def _k_sv(dmax: int, factor: int) -> int:
    """K_SV = factor * (delta_max - 2) for delta_max >= 3."""
    if dmax < 3:
        raise ValueError(f"delta_max must be >= 3, got {dmax}")
    return factor * (dmax - 2)


def _build_grid(c_range: tuple[float, float, float],
                gap_range: tuple[float, float, float],
                c_values=None, gap_values=None) -> list[tuple[float, float]]:
    """Grid points, from ``(lo, hi, step)`` ranges or explicit value lists.

    Explicit lists exist for the near-c=1 regime, where the interesting
    structure is logarithmic in ``c - 1`` and a linear step either misses it or
    wastes hundreds of points reaching it.
    """
    if c_values is not None:
        cs = np.round(np.asarray(c_values, dtype=float), 9)
    else:
        c_lo, c_hi, c_step = c_range
        cs = np.round(np.arange(c_lo, c_hi + 1e-9, c_step), 6)
    if gap_values is not None:
        gs = np.round(np.asarray(gap_values, dtype=float), 9)
    else:
        g_lo, g_hi, g_step = gap_range
        gs = np.round(np.arange(g_lo, g_hi + 1e-9, g_step), 6)
    return [(float(c), float(g)) for c in cs for g in gs]


def _summarize(results: list[dict]) -> None:
    import collections
    buckets = collections.Counter()
    for r in results:
        L = r["best_loss_refined"]
        for thresh, lbl in (
            (4, "<4"), (9, "<9"), (16, "<16"),
            (1e3, "<1e3"), (1e7, "<1e7"), (float("inf"), ">=1e7"),
        ):
            if L < thresh:
                buckets[lbl] += 1
                break
    print("Loss buckets:", dict(buckets))


def run_landscape_scan(
    dmax: int = 4,
    c_range: tuple[float, float, float] = (1.00, 1.20, 0.02),
    gap_range: tuple[float, float, float] = (0.04, 0.60, 0.04),
    k_factor: int = 8,
    subset: int | None = None,
    processes: int | None = None,
    fbcb_cut: int | None = None,
    verbose: bool = False,
    record_history: bool = False,
    c_values=None,
    gap_values=None,
) -> list[dict]:
    """Run the (c, delta_gap) search grid and return per-point result dicts.

    Reproduces the reference figure with default arguments (``dmax=4``,
    the default grid, ``k_factor=8``): 165 points, K_SV=16, N_sample=32.

    ``record_history`` forwards to :func:`optimize.run_batched_survey` and adds
    the per-iteration loss and parameter trajectories to each result dict. Off
    by default: it multiplies the payload size.

    ``c_values`` / ``gap_values`` override the corresponding ``(lo, hi, step)``
    range with an explicit list, for log-spaced near-c=1 scans.
    """
    resolved_cut = resolve_fbcb_cut(dmax, fbcb_cut)
    k_sv = _k_sv(dmax, k_factor)
    n_train = 2 * k_sv

    fbcb = load_fbcb_sqrt2(cut=resolved_cut)
    fine_taus = load_fine_test_taus()

    grid = _build_grid(c_range, gap_range, c_values, gap_values)
    if subset is not None:
        grid = grid[:subset]

    points = [build_survey_point(c, dg, max_delta=float(dmax)) for c, dg in grid]

    results = run_batched_survey(
        points,
        reference_base=fbcb,
        fine_test_taus=fine_taus,
        num_iter=NUM_ITER,
        lr=LR,
        k_sv=k_sv,
        rtol=RTOL,
        n_train=n_train,
        n_test_track=N_TEST_TRACK,
        n_processes=processes,
        verbose=verbose,
        record_history=record_history,
    )
    return [r.to_dict() for r in results]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dmax", type=int, default=4,
                        help="Dimension truncation cutoff (default 4 = reference).")
    parser.add_argument("--c-range", type=float, nargs=3, metavar=("LO", "HI", "STEP"),
                        default=(1.00, 1.20, 0.02),
                        help="c grid: (lo hi step). Default reproduces the reference figure.")
    parser.add_argument("--gap-range", type=float, nargs=3, metavar=("LO", "HI", "STEP"),
                        default=(0.04, 0.60, 0.04),
                        help="delta_gap grid: (lo hi step). Default reproduces the reference figure.")
    parser.add_argument("--k-factor", type=int, default=8, choices=[8, 12],
                        help="K_SV = k_factor*(delta_max-2); N_sample = 2*K_SV. "
                             "Default 8 reproduces the reference (dmax=4 -> K_SV=16, N=32). "
                             "Use 12 to check convergence.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON filename (written under data/runs/). "
                             "Default: landscape_dmax{N}_k{factor}.json")
    parser.add_argument("--subset", type=int, default=None,
                        help="Smoke-test: run only the first N grid points.")
    parser.add_argument("--processes", type=int, default=None,
                        help="Worker processes for sigma-precompute + refined eval.")
    parser.add_argument("--fbcb-cut", type=int, default=None, choices=[6, 8, 12, 16],
                        help="Reference-spectrum cache cut. Default is 6 for "
                             "dmax < 6 and 12 for dmax >= 6.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.fbcb_cut = resolve_fbcb_cut(args.dmax, args.fbcb_cut)
    k_sv = _k_sv(args.dmax, args.k_factor)
    n_train = 2 * k_sv
    print(f"K_SV={k_sv}  N_sample={n_train}  "
          f"(factor={args.k_factor}, delta_max={args.dmax})")

    if args.dmax >= 6 and args.fbcb_cut == 6:
        print("[warn] delta_max >= 6 with cut=6: the reference-spectrum error "
              "estimate can vanish identically at c=1.00 -> huge apparent loss "
              "there. Use --fbcb-cut 12.")

    t0 = time.perf_counter()
    grid = _build_grid(tuple(args.c_range), tuple(args.gap_range))
    if args.subset is not None:
        grid = grid[: args.subset]
    print(f"Grid: {len(grid)} points ({len({c for c, _ in grid})} c x "
          f"{len({g for _, g in grid})} delta_gap)   delta_max={args.dmax}")

    result_dicts = run_landscape_scan(
        dmax=args.dmax,
        c_range=tuple(args.c_range),
        gap_range=tuple(args.gap_range),
        k_factor=args.k_factor,
        subset=args.subset,
        processes=args.processes,
        fbcb_cut=args.fbcb_cut,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - t0
    print(f"Total wall time: {elapsed:.1f}s ({len(grid)} points x {NUM_ITER} iters)")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.output or f"landscape_dmax{args.dmax}_k{args.k_factor}.json"
    out_path = RUNS_DIR / out_name
    out_path.write_text(json.dumps(result_dicts, indent=2))
    print(f"Wrote {out_path}")

    _summarize(result_dicts)


if __name__ == "__main__":
    main()
