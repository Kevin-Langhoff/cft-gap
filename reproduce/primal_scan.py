"""Reproduce the primal search landscape: grid -> scrub -> (refine -> scrub) x rounds.

A thin driver over :mod:`primal.scan.continuation`, which holds the method
(one cell per optimizer batch, parameter-carrying continuation, the
free-boson restart, basin-propagating sweep order) and documents it. This file
only chains the stages into one store and prints the boundary.

A cell is a CANDIDATE iff its refined loss on the 952 fixed torus points
satisfies L = exp(mean log(1 + r^2)) - 1 < 25, with the paper loss
(per-tau fp64 rounding error in quadrature, kappa = 4; no constant floor).
This is a search: a candidate is evidence that a consistent truncated
spectrum exists, an exclusion only means that the restart ladder found none.

Settings
--------
--small         Delta_max = 3, a 4 x 4 grid of the linear sector, short
                ladder, 2 workers: a smoke test of the whole pipeline that
                runs on a laptop in about 40 s.
(default)       the paper's landscape setting for ONE sector: Delta_max = 6,
                20 x 20 base grid over Delta_gap in [0.04, 0.60], ladder
                m_cold = m_warm = 6, scrub with 4 sources per target over
                sigma_warm in {0.05, 0.15, 0.35, 0.7}, then refinement
                rounds. The three sectors (log: c - 1 in [1e-4, 1e-2], lin:
                [1e-2, 0.2], lin2: [0.2, 0.5]) are independent stores; run
                one job per sector.
--optimizer     "default": k_SV = 8 (Delta_max - 2), n_train = 2 k_SV, 300
                steps. "long": k_SV = 64, n_train = 128, 1000 steps, about 30x
                the cost per solve. Near c = 1 single cells are chaotic under
                either setting (a cell one optimizer admits the other can
                miss), so neither dominates there.

Cost (measured at Delta_max = 6 on Xeon Gold 6330 cores): a solve takes 2-5 s
with the default optimizer on an otherwise idle core and about 9 s per worker
with 20 concurrent workers on a shared node (the search is memory-bandwidth
bound); the long optimizer takes 80-110 s per solve. A cell that becomes a
candidate costs a median of 5 solves; an excluded cell climbs the whole ladder
(23 solves) and dominates. ``--workers`` runs independent
cells concurrently with results independent of the worker count (the grid
runs its anti-diagonals as waves, so at most n_c cells are in flight).
Every stage appends its measured wall/CPU time and solve count to the
store under "timing".

Usage::

    python reproduce/primal_scan.py --small --out small.json
    python reproduce/primal_scan.py --sector lin --workers 20 --out d6_lin.json
    python reproduce/primal_scan.py --resume --out d6_lin.json --rounds 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OPTIMIZERS = {
    "default": dict(k_sv=None, n_train=None, num_iter=300),
    "long": dict(k_sv=64, n_train=128, num_iter=1000),
}
SMALL = dict(dmax=3, sector="lin", n_c=4, n_gap=4, m_cold=2, m_warm=2, workers=2,
             rounds=1, scrub_max_iter=2, scrub_sources=2, warm_sweep=[0.15, 0.35])
PAPER = dict(dmax=6, n_c=20, n_gap=20, m_cold=6, m_warm=6, workers=1,
             rounds=4, scrub_max_iter=20, scrub_sources=4, warm_sweep=[0.05, 0.15, 0.35, 0.7])


def boundary(store: dict) -> list[tuple[float, float | None]]:
    """``[(c, highest candidate gap)]`` per c column, c ascending (None: no candidate)."""
    cols: dict[float, list[dict]] = {}
    for r in store["rows"].values():
        cols.setdefault(round(r["c"], 12), []).append(r)
    out = []
    for c in sorted(cols):
        ok = [r["delta_gap"] for r in cols[c] if r["allowed"]]
        out.append((c, max(ok) if ok else None))
    return out


def main(argv=None) -> None:
    from primal.scan import continuation as C

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="the store (JSON)")
    ap.add_argument("--small", action="store_true", help="laptop smoke test (see above)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing store: skip the grid, run --rounds more rounds")
    ap.add_argument("--dmax", type=int, default=None)
    ap.add_argument("--sector", choices=tuple(C.SECTORS), default=None)
    ap.add_argument("--n-c", type=int, default=None)
    ap.add_argument("--n-gap", type=int, default=None)
    ap.add_argument("--gap-range", type=float, nargs=2, default=(0.04, 0.60), metavar=("LO", "HI"))
    ap.add_argument("--optimizer", choices=tuple(OPTIMIZERS), default="default")
    ap.add_argument("--rounds", type=int, default=None, help="refine -> scrub rounds after the grid")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    s = dict(SMALL if args.small else PAPER)
    for k in ("dmax", "sector", "n_c", "n_gap", "rounds", "workers"):
        if getattr(args, k) is not None:
            s[k] = getattr(args, k)
    s.setdefault("sector", "lin")

    def scrub(store):
        cfg = store["config"]
        cfg.update(scrub_gap_max=None, scrub_sources=s["scrub_sources"], warm_sweep=s["warm_sweep"])
        C.stage_scrub(store, args.out, verbose=args.verbose, max_iter=s["scrub_max_iter"])

    if args.resume:
        store = C.load_store(args.out)
        store["config"]["workers"] = s["workers"]
    else:
        if args.out.exists():
            raise SystemExit(f"{args.out} exists; pass --resume to continue it")
        cfg = C.make_config(dmax=s["dmax"], sector=s["sector"], n_c=s["n_c"], n_gap=s["n_gap"],
                            gap_lo=args.gap_range[0], gap_hi=args.gap_range[1],
                            m_cold=s["m_cold"], m_warm=s["m_warm"], workers=s["workers"],
                            **OPTIMIZERS[args.optimizer])
        store = C.stage_grid(cfg, args.out, verbose=args.verbose)
        scrub(store)
    for _ in range(s["rounds"]):
        C.stage_refine(store, args.out, verbose=args.verbose)
        scrub(store)

    rows = store["rows"].values()
    print(f"\n{args.out}: {sum(r['allowed'] for r in rows)}/{len(rows)} candidates, "
          f"{store['total_solves']} solves, {store['total_cpu_s'] / 3600:.2f} CPU-h")
    print("highest candidate gap per column (c, Delta_gap):")
    for c, g in boundary(store):
        print(f"  c = {c:.6f}   " + ("none" if g is None else f"{g:.4f}"))


if __name__ == "__main__":
    main()
