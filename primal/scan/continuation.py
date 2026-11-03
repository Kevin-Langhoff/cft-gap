"""Continuation scan: the landscape driver that warm-starts each cell from
its converged neighbours.

Three composable stages, each reading and writing ONE store file so runs
chain:

  grid    build an n_c x n_gap lattice in one sector and evaluate it
  scrub   re-evaluate EXCLUDED points that touch a CANDIDATE neighbour,
          warm-started from that neighbour's converged parameter vector;
          keep the new result only if it strictly lowers the loss
  refine  subdivide every quadtree cell whose corners straddle the
          classification threshold and evaluate the new corners

Run scrub BEFORE refine: recovering a false exclusion moves the boundary,
and there is no point resolving a boundary that scrubbing will shift.

Why not a plain grid of cold optimizations
------------------------------------------
The loss is a network of needle-like minima: displacing a converged
parameter vector by ~1e-3 raises the log1p loss by 8-15. A single cold
start therefore reports "no candidate" wherever the good basin is narrow,
which is exactly the low-c, low-gap corner: a large-batch single-start
scan calls about 6% of that strip a candidate where this driver calls
nearly all of it one. Four mechanisms make the difference:

1. ONE CELL PER BATCH. Batch size changes the float64 reduction order,
   and the chaotic 300-step trajectory amplifies it: the same cell
   (c, gap) = (1.001, 0.04) scored log1p 0.84 alone and 24.31 inside a
   320-cell batch. Every optimizer batch here is one cell's restarts, all
   sharing (c, delta_gap, num_spin), and the torus-point seeds are frozen
   (``seed_offset=0``), so all spread comes from the initialization.
2. WARM STARTS CARRY PARAMETERS, NOT SPECTRA. Resolving a converged point
   to ``(delta, spin, degen)`` and re-encoding it falls off the needle.
   Continuation copies the converged pre-sigmoid parameters by variable
   key (:func:`primal.warm_start.lift_init`).
3. THE FREE-BOSON LANE. Near c = 1 the low-gap strip is the compactified
   boson at ``R^2 = 1/(2 delta_gap)``, which has a primary exactly at the
   gap. That is one specific point in a 100-250 dimensional space; random
   restarts do not find it, so it is constructed
   (:func:`primal.warm_start.free_boson_point`).
4. BASIN-PROPAGATING SWEEP ORDER. Cells are visited from large c - 1
   toward c -> 1 and, within a column, from low gap upward, so both
   continuation sources of a cell have converged before it runs.

Each cell climbs a ladder of restart tiers, cheapest first, and EXITS
EARLY once it lands below the threshold: the question is whether a
candidate exists, not what the infimum is. Cells that stay above the
threshold pay for the whole ladder, which is the right allocation, since
those are the cells reported as excluded.

Loss and verdict
----------------
Every solve uses the paper loss: ``run_batched_survey`` with the per-tau
fp64 rounding error added in quadrature (``sigma_round_kappa = 4``), no
constant floor (``sigma_epsilon = 0``) and ``tau2_max = Delta_max + 2``.
A cell is a CANDIDATE iff its refined score on the 952 fixed torus points
satisfies ``S = mean log(1 + r^2) < log 26``, i.e. ``L = e^S - 1 < 25``.
This is a search, not a proof: a candidate is evidence that a consistent
truncated spectrum exists, and an exclusion means only that the ladder
found none.

Parallelism
-----------
A cell depends only on its sources, so ``--workers W`` evaluates
independent cells in separate processes with identical results: the grid
stage runs its anti-diagonals as waves (both sources of a cell lie on the
previous diagonal), scrub targets within one pass are independent, and
refine draws every new point's sources from the rows that existed when the
stage began. Each worker runs its optimizer batches in-process
(``processes = 1``); with ``W = 1`` the batch-internal pool uses
``--processes`` instead.

Store schema
------------
{
  "config": {...},                    sector geometry + optimizer settings
  "rows": {"<u>:<v>": {               one evaluated cell, keyed on (u, v)
      "c", "delta_gap", "u", "v",
      "best_loss_refined", "best_loss_refined_log1p", "allowed",
      "flex_delta": [...],            converged parameters (continuation source)
      "spectrum": [[delta, spin, degen], ...],   resolved numeric spectrum
      "lane": "legacy"|"linear_delta"|"fb"|"continuation"|"cold",
      "n_solves": int,                restarts spent on this cell
      "lane_losses": {lane: best log1p seen in that lane},
      "provenance": "grid"|"scrub"|"refine",
      "level": int,                   refinement ROUND that introduced the point
      "depth": int,                   quadtree depth of the cell that spawned it
      "stage_seq": int,               stage counter when this value was written
      "scrub_gen": int|None, "scrub_old_log1p": float|None,
      "scrub_source": "u:v,..."|None,
      "scrub_tried": ["u:v", ...],    sources already tried, so passes don't repeat
      "wall_time": float}},
  "cells": [{"u0", "v0", "u1", "v1", "depth"}],
  "timing": [{stage, stage_seq, n_points, n_solves, wall_s, eval_s, cpu_s, ...}]
}

``level`` is the refinement round (0 = base grid), so filtering rows to
``level <= k`` reconstructs what was known after round k; ``depth``
measures local mesh fineness instead; ``stage_seq`` replays the run stage
by stage, including scrub stages, which change values without adding
points.

Normalized coordinates: ``u`` in [0, 1] spans the sector's c - 1 range,
logarithmically in a log sector and linearly in a linear one, and ``v``
in [0, 1] spans the gap range. Quadtree cells stay square in (u, v) in
every sector, so one refinement rule serves all of them. The three
standard sectors tile c - 1 in [1e-4, 0.5] end to end and share their
endpoints exactly.

Usage::

    python -m primal.scan.continuation --stage grid   --store d6_lin.json \\
        --dmax 6 --sector lin --workers 8
    python -m primal.scan.continuation --stage scrub  --store d6_lin.json --workers 8
    python -m primal.scan.continuation --stage refine --store d6_lin.json --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as _mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from ..optimize import run_batched_survey
from ..resources import load_fbcb_sqrt2, load_fine_test_taus, resolve_fbcb_cut
from ..spectrum import SurveyPoint, _default_num_spin, build_survey_point
from ..warm_start import free_boson_point, lift_init, resolve_flexdelta_to_numeric
from .adaptive import LEVEL_LOG1P, is_boundary_cell

# A cell is a CANDIDATE ("allowed") iff log1p-mean loss < log 26 (L < 25).
LEVEL = float(LEVEL_LOG1P)

SECTORS = {
    "log":  dict(cm1_lo=1e-4, cm1_hi=1e-2, spacing="log"),
    "lin":  dict(cm1_lo=1e-2, cm1_hi=0.2, spacing="linear"),
    "lin2": dict(cm1_lo=0.2, cm1_hi=0.5, spacing="linear"),   # -> c = 1.5
}


# ============================================================================
# Coordinates.
# ============================================================================


def c_of_u(u: float, cfg: dict) -> float:
    lo, hi = cfg["cm1_lo"], cfg["cm1_hi"]
    if cfg["spacing"] == "log":
        cm1 = 10.0 ** (math.log10(lo) + u * (math.log10(hi) - math.log10(lo)))
    else:
        cm1 = lo + u * (hi - lo)
    return 1.0 + cm1


def gap_of_v(v: float, cfg: dict) -> float:
    return cfg["gap_lo"] + v * (cfg["gap_hi"] - cfg["gap_lo"])


def key_of(u: float, v: float) -> str:
    return f"{u:.12f}:{v:.12f}"


# ============================================================================
# Ansatz size: the c-dependent slot rule.
# ============================================================================

# N_j = k_eff(c, d) * (d - j), with k_eff fit to the smallest slot count that
# reaches the loss floor, over d = 3..7 and twelve central charges:
#   k_eff = max(k_0, k_0 * exp(alpha(d) * max(0, sqrt(c) - sqrt(c_th(d))))),
#   c_th(d) = a + b/d,  alpha(d) = alpha_0 / d^beta,  k_0 = 6 (d <= 5), 8 (d >= 6),
# quantized to even k and capped at 32. Below c_th(d) (c < 1.23 at d = 6) it is
# the default rule of ``spectrum.build_survey_point``; at d <= 4, c <= 1.25 it
# returns that rule exactly. Past d = 7 (no fit data) it uses k_eff(c, 7).
_V2_PARAMS = dict(a=1.226609, b=0.024657, alpha_0=5.184627, beta=-0.414845)


def num_spin_v2(max_delta: float, c: float, *, k_step: int = 2, k_max: int = 32) -> dict[int, int]:
    """Flex slots per spin under the c-dependent rule (see the comment above)."""
    d = int(max_delta)
    if float(max_delta) != d or d < 3:
        raise ValueError(f"delta_max must be an integer >= 3, got {max_delta!r}")
    if d <= 4 and float(c) <= 1.25:
        return _default_num_spin(float(max_delta))
    d_eff = min(d, 7)
    k_0 = 6 if d <= 5 else 8
    k_floor = max(4, k_0 - 2)
    p = _V2_PARAMS
    c_th = p["a"] + p["b"] / d_eff
    alpha_d = p["alpha_0"] / (d_eff ** p["beta"])
    k0_eff = 6.0 if d_eff <= 5 else 8.0
    k_raw = k0_eff * math.exp(alpha_d * max(0.0, math.sqrt(c) - math.sqrt(max(c_th, 0.01))))
    k_quant = int(k_step) * int(round(k_raw / float(k_step)))
    k_eff = min(int(k_max), max(int(k_floor), int(k_quant)))
    return {j: k_eff * (d - j) for j in range(d)}


def num_spin_for(rule: str, dmax: int, c: float) -> dict[int, int] | None:
    """``"v2"``: :func:`num_spin_v2`; ``"default"``: the fixed rule of
    ``build_survey_point`` (returned as ``None``)."""
    if rule == "v2":
        return num_spin_v2(dmax, c)
    if rule == "default":
        return None
    raise ValueError(f"unknown num_spin_rule {rule!r}")


# ============================================================================
# Lanes: the restart ladder for one cell.
# ============================================================================

_REF: dict[int, tuple] = {}


def reference_for(dmax: int) -> tuple:
    """``(reference spectrum, fixed torus points)`` for this ``Delta_max``,
    cached per process."""
    if dmax not in _REF:
        _REF[dmax] = (load_fbcb_sqrt2(cut=resolve_fbcb_cut(dmax, None)),
                      load_fine_test_taus())
    return _REF[dmax]


def _linear_delta_init(point: SurveyPoint) -> np.ndarray:
    """Parameters that space each spin's flex dimensions evenly in Delta
    (``x = log(r/(1-r))``, ``r = i/(n+1)``), where the default init spaces
    them evenly in sigmoid argument and crowds the window's middle."""
    counts: dict[int, int] = {}
    for var in point.cft.variables:
        counts[int(var.args[0])] = counts.get(int(var.args[0]), 0) + 1
    out = []
    for s in sorted(counts):
        n = counts[s]
        out.extend(math.log((i / (n + 1)) / (1.0 - i / (n + 1))) for i in range(1, n + 1))
    return np.asarray(out, dtype=np.float64)


def _base_point(c: float, gap: float, cfg: dict, *, strategy: str = "legacy",
                sigma: float = 0.0, seed: int = 0) -> SurveyPoint:
    """The standard ansatz at (c, gap). ``strategy``: ``"legacy"`` (the
    default init), ``"linear_delta"``, or ``"perturbed"`` (default init plus
    Gaussian noise of width ``sigma`` from ``default_rng(seed)``)."""
    pt = build_survey_point(c, gap, max_delta=float(cfg["dmax"]),
                            num_spin=num_spin_for(cfg["num_spin_rule"], cfg["dmax"], c))
    if strategy == "legacy":
        return pt
    if strategy == "linear_delta":
        return replace(pt, init_values=_linear_delta_init(pt))
    if strategy == "perturbed":
        rng = np.random.default_rng(int(seed))
        return replace(pt, init_values=pt.init_values
                       + rng.normal(0.0, float(sigma), size=pt.init_values.shape))
    raise ValueError(f"unknown init strategy {strategy!r}")


def _carry(base: SurveyPoint, source: dict, cfg: dict) -> np.ndarray | None:
    """Lift a stored row's ``flex_delta`` into ``base``'s ansatz by variable key.

    The row stores only numbers, so its variable list is rebuilt from its
    (c, gap); building a survey point runs no optimization. Under the
    c-dependent slot rule source and target can differ in length, which
    ``lift_init`` handles by matching on variable identity.
    """
    flex = source.get("flex_delta")
    if not flex:
        return None
    src_pt = _base_point(float(source["c"]), float(source["delta_gap"]), cfg)
    iv, _info = lift_init(
        source_variables=src_pt.cft.variables,
        source_flex=np.asarray(flex, dtype=float),
        target_variables=base.cft.variables,
        target_init_fallback=base.init_values,
    )
    return np.asarray(iv, dtype=float)


def _fb_points(c: float, gap: float, cfg: dict) -> list[SurveyPoint]:
    try:
        return [free_boson_point(c, gap, cfg["dmax"],
                                 num_spin_for(cfg["num_spin_rule"], cfg["dmax"], c))]
    except Exception:  # degenerate gaps can fail to encode; skip the lane
        return []


def build_lanes(c: float, gap: float, cfg: dict, sources: list[dict]) -> list[tuple]:
    """The restart ladder for one cell: ``[(tier_name, [(lane, point), ...])]``.

    Tier 1 pools every DETERMINISTIC start (default init, linear-Delta init,
    the free boson, one carry per continuation source): a handful of solves
    that settle most cells. Tier 2 perturbs each carry ``m_warm`` times at
    width ``sigma_warm``, staying on the source basin's shoulder. Tier 3 is
    ``m_cold`` perturbed default inits at width ``sigma_cold``: different
    basins, last because it is the least targeted.
    """
    base = _base_point(c, gap, cfg)

    det: list[tuple[str, SurveyPoint]] = [("legacy", base)]
    det.append(("linear_delta", _base_point(c, gap, cfg, strategy="linear_delta")))
    if cfg.get("include_fb", True):
        for p in _fb_points(c, gap, cfg):
            det.append(("fb", p))
    carries = []
    for src in sources:
        iv = _carry(base, src, cfg)
        if iv is not None:
            carries.append(iv)
            det.append(("continuation", replace(base, init_values=iv.copy())))

    tiers: list[tuple] = [("deterministic", det)]

    # Per-cell noise stream; hash() of a float tuple is deterministic in CPython.
    rng = np.random.default_rng(int(cfg.get("seed_base", 0))
                                + int(abs(hash((round(c, 9), round(gap, 9)))) % 10**6))
    warm = []
    for iv in carries:
        for _ in range(int(cfg["m_warm"])):
            warm.append(("continuation", replace(base, init_values=iv + rng.normal(
                0.0, float(cfg["sigma_warm"]), size=iv.shape))))
    if warm:
        tiers.append(("warm_perturbed", warm))

    cold = [("cold", _base_point(c, gap, cfg, strategy="perturbed",
                                 sigma=float(cfg["sigma_cold"]), seed=k))
            for k in range(int(cfg["m_cold"]))]
    if cold:
        tiers.append(("cold_ensemble", cold))
    return tiers


def _run_batch(points: list[SurveyPoint], cfg: dict, processes: int) -> tuple[list, list]:
    """One uniform batch through the paper loss; returns (log1p losses, flexes)."""
    fbcb, fine_taus = reference_for(cfg["dmax"])
    res = run_batched_survey(
        points, reference_base=fbcb, fine_test_taus=fine_taus,
        num_iter=int(cfg["num_iter"]), lr=0.1, k_sv=int(cfg["k_sv"]), rtol=1e-12,
        n_train=int(cfg["n_train"]), n_test_track=48, n_processes=int(processes),
        seed_offset=0, tau2_max=float(cfg["dmax"]) + 2.0,
        sigma_epsilon=float(cfg["sigma_epsilon"]),
        sigma_round_kappa=float(cfg["sigma_round_kappa"]),
        damping_lambda=float(cfg.get("damping_lambda", 0.0)),
    )
    losses = [float("nan") if r.best_loss_refined_log1p is None
              else float(r.best_loss_refined_log1p) for r in res]
    return losses, [r.flex_delta for r in res]


def eval_cell(c: float, gap: float, cfg: dict, sources: list[dict], *,
              u: float, v: float, processes: int | None = None) -> dict:
    """Climb the lane ladder at one cell; return its best result as a row."""
    t0 = time.perf_counter()
    procs = int(cfg.get("processes", 1) if processes is None else processes)
    best_loss, best_flex, best_lane, best_pt = float("inf"), None, None, None
    lane_losses: dict[str, float] = {}
    n_solves = 0

    for _tier_name, entries in build_lanes(c, gap, cfg, sources):
        # A tier shares (c, gap) but not always the operator list: the
        # free-boson point carries extra fixed primaries. One optimizer batch
        # needs one structure, so split by signature; each group is still
        # uniform in (c, gap, num_spin), which is what batch-size stability needs.
        groups: dict[tuple, list[int]] = {}
        for i, (_nm, pt) in enumerate(entries):
            groups.setdefault((len(pt.cft.primaries), len(pt.cft.variables)), []).append(i)
        for idx in groups.values():
            names = [entries[i][0] for i in idx]
            pts = [entries[i][1] for i in idx]
            losses, flexes = _run_batch(pts, cfg, procs)
            n_solves += len(pts)
            for nm, L, fx, pt in zip(names, losses, flexes, pts):
                if not np.isfinite(L):
                    continue
                if nm not in lane_losses or L < lane_losses[nm]:
                    lane_losses[nm] = float(L)
                if L < best_loss:
                    best_loss, best_flex, best_lane, best_pt = float(L), fx, nm, pt
        if best_loss < LEVEL:
            break  # a candidate exists; stop buying restarts

    spectrum = None
    if best_flex is not None and best_pt is not None:
        try:
            resolved = resolve_flexdelta_to_numeric(best_pt.cft, np.asarray(best_flex, dtype=float))
            spectrum = [[float(p.delta), int(p.spin), int(p.degen)] for p in resolved.primaries]
        except Exception:  # the spectrum is a convenience, never fatal
            spectrum = None

    lg = best_loss if np.isfinite(best_loss) else float("inf")
    return dict(
        u=float(u), v=float(v), c=float(c), delta_gap=float(gap),
        best_loss_refined=float(math.expm1(lg)) if np.isfinite(lg) else float("inf"),
        best_loss_refined_log1p=float(lg),
        allowed=bool(lg < LEVEL),
        flex_delta=([float(x) for x in best_flex] if best_flex is not None else None),
        spectrum=spectrum, lane=best_lane, n_solves=int(n_solves),
        lane_losses={k: round(x, 6) for k, x in lane_losses.items()},
        wall_time=round(time.perf_counter() - t0, 3),
    )


def _eval_job(job: tuple) -> dict:
    c, gap, cfg, sources, u, v = job
    return eval_cell(c, gap, cfg, sources, u=u, v=v, processes=1)


class _Evaluator:
    """Evaluate independent cells serially or on a pool of worker processes."""

    def __init__(self, cfg: dict, workers: int):
        self.cfg = cfg
        self.workers = int(workers)
        self._ex = None

    def __enter__(self):
        if self.workers > 1:
            self._ex = ProcessPoolExecutor(max_workers=self.workers,
                                           mp_context=_mp.get_context("spawn"))
        return self

    def __exit__(self, *exc):
        if self._ex is not None:
            self._ex.shutdown()

    def run(self, jobs: list[tuple]) -> list[dict]:
        """``jobs``: ``[(c, gap, sources, u, v)]`` -> rows, in order."""
        full = [(c, g, self.cfg, s, u, v) for c, g, s, u, v in jobs]
        if self._ex is None:
            return [eval_cell(c, g, self.cfg, s, u=u, v=v) for c, g, _, s, u, v in full]
        return list(self._ex.map(_eval_job, full))


# ============================================================================
# Store.
# ============================================================================


def load_store(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save_store(store: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(store))
    tmp.replace(path)


def _next_stage_seq(store: dict) -> int:
    return int(store["config"].get("stage_seq", 0)) + 1


def record_timing(store: dict, stage: str, *, n_points: int, n_solves: int,
                  wall_s: float, eval_s: float, level: int | None = None,
                  stage_seq: int | None = None, scrub_gen: int | None = None) -> None:
    """Append one stage's cost and refresh the store totals.

    ``wall_s`` includes the invocation's import and reference load;
    ``eval_s`` is the optimizer alone and is what scales with cell count.
    ``cpu_s = wall_s * max(workers, processes)`` is an upper bound on the
    CPU time charged. ``n_solves >= n_points`` because a failing cell climbs
    the whole ladder.
    """
    cfg = store["config"]
    ncpu = max(int(cfg.get("workers", 1)), int(cfg.get("processes", 1)))
    entry = dict(stage=stage, n_points=int(n_points), n_solves=int(n_solves),
                 wall_s=round(float(wall_s), 3), eval_s=round(float(eval_s), 3),
                 cpu_s=round(float(wall_s) * ncpu, 3), cpus=ncpu,
                 utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if level is not None:
        entry["level"] = int(level)
    if stage_seq is not None:
        entry["stage_seq"] = int(stage_seq)
    if scrub_gen is not None:
        entry["scrub_gen"] = int(scrub_gen)
    log = store.setdefault("timing", [])
    log.append(entry)
    store["total_wall_s"] = round(sum(e["wall_s"] for e in log), 3)
    store["total_eval_s"] = round(sum(e["eval_s"] for e in log), 3)
    store["total_cpu_s"] = round(sum(e["cpu_s"] for e in log), 3)
    store["total_points_evaluated"] = int(sum(e["n_points"] for e in log))
    store["total_solves"] = int(sum(e["n_solves"] for e in log))
    print(f"[time] {stage}: {wall_s:.1f} s wall ({eval_s:.1f} s eval, {n_points} cells / "
          f"{n_solves} solves) | store total {store['total_wall_s']:.1f} s wall, "
          f"{store['total_cpu_s']:.1f} CPU-s, {store['total_solves']} solves", flush=True)


def _stamp(r: dict, *, provenance: str, level: int, depth: int, stage_seq: int,
           scrub_gen=None, scrub_old=None, scrub_source=None, scrub_tried=None) -> dict:
    r.update(provenance=provenance, level=int(level), depth=int(depth),
             stage_seq=int(stage_seq), scrub_gen=scrub_gen, scrub_old_log1p=scrub_old,
             scrub_source=scrub_source, scrub_tried=list(scrub_tried or []))
    return r


def _report(r: dict, n: int, total: int, verbose: bool) -> None:
    if verbose or n % 25 == 0 or n == total:
        print(f"  [{n}/{total}] c-1={r['c'] - 1:.4e} gap={r['delta_gap']:.4f} "
              f"log1p={r['best_loss_refined_log1p']:8.3f} "
              f"{'CANDIDATE' if r['allowed'] else 'excluded '} "
              f"lane={r['lane']} solves={r['n_solves']}", flush=True)


# ============================================================================
# Stages.
# ============================================================================


def stage_grid(cfg: dict, out: Path, *, verbose: bool = False) -> dict:
    """Evaluate the base lattice in basin-propagating order.

    Cell ``(iu, iv)`` takes its sources from ``(iu+1, iv)`` (the next larger
    c - 1 at the same gap) and ``(iu, iv-1)`` (the next lower gap in its
    column). Both lie on the anti-diagonal before it, so the diagonals run
    as waves and every cell sees exactly the sources the serial sweep would
    give it.
    """
    t_stage = time.time()
    n_c, n_g = int(cfg["n_c"]), int(cfg["n_gap"])
    us = np.linspace(0.0, 1.0, n_c)
    vs = np.linspace(0.0, 1.0, n_g)
    print(f"[grid] {n_c}x{n_g} cells, dmax={cfg['dmax']}, sector={cfg['sector']}, "
          f"m_cold={cfg['m_cold']} m_warm={cfg['m_warm']} fb={cfg['include_fb']} "
          f"num_spin={cfg['num_spin_rule']} workers={cfg['workers']}", flush=True)

    rows: dict[str, dict] = {}
    by_idx: dict[tuple[int, int], dict] = {}
    t_eval = time.time()
    done = 0
    with _Evaluator(cfg, cfg["workers"]) as ev:
        for k in range(n_c + n_g - 1):
            wave = [(n_c - 1 - a, k - a) for a in range(n_c) if 0 <= k - a < n_g]
            jobs = []
            for iu, iv in wave:
                sources = [by_idx[s] for s in ((iu + 1, iv), (iu, iv - 1)) if s in by_idx]
                jobs.append((c_of_u(float(us[iu]), cfg), gap_of_v(float(vs[iv]), cfg),
                             sources, float(us[iu]), float(vs[iv])))
            for (iu, iv), r in zip(wave, ev.run(jobs)):
                _stamp(r, provenance="grid", level=0, depth=0, stage_seq=1)
                rows[key_of(r["u"], r["v"])] = r
                by_idx[(iu, iv)] = r
                done += 1
                _report(r, done, n_c * n_g, verbose)
    eval_s = time.time() - t_eval

    cells = [dict(u0=float(us[i]), v0=float(vs[j]), u1=float(us[i + 1]),
                  v1=float(vs[j + 1]), depth=0)
             for i in range(n_c - 1) for j in range(n_g - 1)]
    cfg["stage_seq"] = 1
    store = dict(config=cfg, rows=rows, cells=cells)
    record_timing(store, "grid", n_points=len(rows),
                  n_solves=sum(r["n_solves"] for r in rows.values()),
                  wall_s=time.time() - t_stage, eval_s=eval_s, level=0, stage_seq=1)
    save_store(store, out)
    n_ok = sum(1 for r in rows.values() if r["allowed"])
    print(f"[grid] candidates {n_ok}/{len(rows)}  -> {out}", flush=True)
    return store


def _corner_keys(cell: dict) -> list[str]:
    return [key_of(cell["u0"], cell["v0"]), key_of(cell["u1"], cell["v0"]),
            key_of(cell["u0"], cell["v1"]), key_of(cell["u1"], cell["v1"])]


def _neighbours(store: dict) -> dict[str, list[str]]:
    """Each row key -> the keys sharing a quadtree cell with it."""
    rows = store["rows"]
    adj: dict[str, set] = {k: set() for k in rows}
    for cell in store["cells"]:
        present = [k for k in _corner_keys(cell) if k in rows]
        for a in present:
            adj[a].update(b for b in present if b != a)
    return {k: sorted(v) for k, v in adj.items()}


def stage_scrub(store: dict, out: Path, *, verbose: bool = False, max_iter: int = 20) -> dict:
    """Scrub repeatedly until no cell improves, or ``max_iter`` passes.

    Iterating is not redundant: a recovered cell becomes a legal source for
    its own excluded neighbours on the next pass, so recoveries cascade
    outward from the boundary one cell per pass (more passes are needed at
    larger Delta_max). ``scrub_tried`` stops a pass from re-running an
    identical (target, source) pair, so a converged scrub costs one cheap
    no-op pass to prove it.
    """
    for it in range(1, max_iter + 1):
        if _scrub_once(store, out, verbose=verbose, iteration=it) == 0:
            print(f"[scrub] converged after {it} pass(es)", flush=True)
            break
    else:
        print(f"[scrub] hit max_iter={max_iter} without converging", flush=True)
    return store


def _scrub_job(job: tuple) -> dict:
    """Retry one target over the warm-width sweep; keep the best."""
    old, picked, cfg, processes = job
    new = None
    for sw in cfg.get("warm_sweep") or [cfg.get("sigma_warm", 0.15)]:
        trial = dict(cfg, sigma_warm=float(sw))
        cand = eval_cell(old["c"], old["delta_gap"], trial, picked,
                         u=old["u"], v=old["v"], processes=processes)
        if new is None or cand["best_loss_refined_log1p"] < new["best_loss_refined_log1p"]:
            cand["n_solves"] += 0 if new is None else new["n_solves"]
            new = cand
        else:
            new["n_solves"] += cand["n_solves"]
        if new["allowed"]:
            break  # a candidate exists; no need to price the other widths
    return new


def _scrub_once(store: dict, out: Path, *, verbose: bool, iteration: int) -> int:
    """One scrub pass; returns how many cells improved.

    Targets are excluded cells with at least one candidate neighbour. Each
    takes its best UNTRIED candidate neighbours (``scrub_sources`` of them),
    so every pass contributes genuinely new starts without widening the
    neighbourhood beyond cells that share a quadtree cell. Boundary quality
    is limited by warm sampling, not cold: one step in ``sigma_warm`` has
    moved a stubborn cell by 2.0-2.5 in log1p where sweeping ``sigma_cold``
    moved it by 0.2, hence the optional ``warm_sweep`` over several widths.
    """
    t_stage = time.time()
    cfg = store["config"]
    rows = store["rows"]
    adj = _neighbours(store)
    seq = _next_stage_seq(store)
    gen = int(cfg.get("scrub_gen", 0)) + 1
    gap_max = cfg.get("scrub_gap_max")
    n_src = max(int(cfg.get("scrub_sources", 1)), 1)

    todo = []
    n_exhausted = n_nosrc = n_outside = 0
    for k, r in rows.items():
        if r["allowed"]:
            continue
        if gap_max is not None and r["delta_gap"] > gap_max:
            n_outside += 1
            continue
        cands = [rows[n] for n in adj.get(k, []) if rows[n]["allowed"] and rows[n].get("flex_delta")]
        if not cands:
            n_nosrc += 1
            continue
        tried = set(r.get("scrub_tried") or [])
        fresh = sorted((z for z in cands if key_of(z["u"], z["v"]) not in tried),
                       key=lambda z: z["best_loss_refined_log1p"])
        if not fresh:
            n_exhausted += 1
            continue
        picked = fresh[:n_src]
        todo.append((k, picked, [key_of(z["u"], z["v"]) for z in picked]))

    n_excl = sum(1 for r in rows.values() if not r["allowed"])
    print(f"[scrub pass {iteration}] {len(todo)} cells to retry (of {n_excl} excluded; "
          f"{n_exhausted} exhausted all neighbours; {n_nosrc} with no candidate neighbour"
          + (f"; {n_outside} outside gap<={gap_max}" if gap_max is not None else "")
          + ")", flush=True)
    if not todo:
        record_timing(store, "scrub", n_points=0, n_solves=0, wall_s=time.time() - t_stage,
                      eval_s=0.0, stage_seq=seq, scrub_gen=gen)
        cfg["stage_seq"], cfg["scrub_gen"] = seq, gen
        save_store(store, out)
        return 0

    t_eval = time.time()
    workers = int(cfg.get("workers", 1))
    procs = 1 if workers > 1 else int(cfg.get("processes", 1))
    jobs = [(rows[k], picked, cfg, procs) for k, picked, _ in todo]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=_mp.get_context("spawn")) as ex:
            results = list(ex.map(_scrub_job, jobs))
    else:
        results = [_scrub_job(j) for j in jobs]
    eval_s = time.time() - t_eval

    n_better = n_recovered = n_solves = 0
    for (k, _picked, src_keys), new in zip(todo, results):
        old = rows[k]
        n_solves += new["n_solves"]
        tried = list(old.get("scrub_tried") or []) + src_keys
        if new["best_loss_refined_log1p"] < old["best_loss_refined_log1p"]:
            n_better += 1
            n_recovered += int(new["allowed"] and not old["allowed"])
            _stamp(new, provenance="scrub", level=old.get("level", 0), depth=old["depth"],
                   stage_seq=seq, scrub_gen=gen, scrub_old=old["best_loss_refined_log1p"],
                   scrub_source=",".join(src_keys), scrub_tried=tried)
            rows[k] = new
            _report(new, n_better, len(todo), verbose)
        else:
            # Record failures too, so the next pass moves to the next-best source.
            old["scrub_tried"] = tried

    print(f"[scrub pass {iteration}] improved {n_better}/{len(todo)}; RECOVERED "
          f"{n_recovered} cells from excluded to candidate", flush=True)
    cfg["stage_seq"], cfg["scrub_gen"] = seq, gen
    record_timing(store, "scrub", n_points=len(todo), n_solves=n_solves,
                  wall_s=time.time() - t_stage, eval_s=eval_s, stage_seq=seq, scrub_gen=gen)
    save_store(store, out)
    return n_better


def _nearest_sources(rows: dict, u: float, v: float, n: int = 2) -> list[dict]:
    """The ``n`` closest candidate rows that carry parameters."""
    cand = [r for r in rows.values() if r["allowed"] and r.get("flex_delta")]
    cand.sort(key=lambda r: (r["u"] - u) ** 2 + (r["v"] - v) ** 2)
    return cand[:n]


def stage_refine(store: dict, out: Path, *, verbose: bool = False) -> dict:
    """Subdivide every quadtree cell whose corners straddle the threshold.

    The straddle test is :func:`primal.scan.adaptive.is_boundary_cell` with
    no noise band; the scrub stage, not a band, is what guards against
    noisy corners here. Each straddling cell splits into four and its five
    new points (edge midpoints and centre) are evaluated, warm-started from
    the two nearest candidates among the rows present when the stage began.
    """
    t_stage = time.time()
    cfg = store["config"]
    rows = store["rows"]
    seq = _next_stage_seq(store)

    straddling = []
    for idx, cell in enumerate(store["cells"]):
        ks = _corner_keys(cell)
        if all(k in rows for k in ks) and is_boundary_cell(
                [rows[k]["best_loss_refined_log1p"] for k in ks], level=LEVEL, noise_band=0.0):
            straddling.append(idx)
    print(f"[refine] {len(straddling)} straddling cells of {len(store['cells'])}", flush=True)
    if not straddling:
        record_timing(store, "refine", n_points=0, n_solves=0, wall_s=time.time() - t_stage,
                      eval_s=0.0, stage_seq=seq)
        cfg["stage_seq"] = seq
        save_store(store, out)
        return store

    level = int(cfg.get("round", 0)) + 1
    cfg["round"] = level
    new_cells, want = [], {}
    for idx in straddling:
        cell = store["cells"][idx]
        um, vm = 0.5 * (cell["u0"] + cell["u1"]), 0.5 * (cell["v0"] + cell["v1"])
        d = cell["depth"] + 1
        for a, b in ((um, cell["v0"]), (um, cell["v1"]), (cell["u0"], vm),
                     (cell["u1"], vm), (um, vm)):
            k = key_of(a, b)
            # A point on a shared edge can have two parents; keep the finer depth.
            if k not in rows and (k not in want or d > want[k][2]):
                want[k] = (a, b, d)
        for x0, x1 in ((cell["u0"], um), (um, cell["u1"])):
            for y0, y1 in ((cell["v0"], vm), (vm, cell["v1"])):
                new_cells.append(dict(u0=x0, v0=y0, u1=x1, v1=y1, depth=d))

    items = list(want.values())
    print(f"[refine] {len(items)} new points at level {level}", flush=True)
    jobs = [(c_of_u(a, cfg), gap_of_v(b, cfg), _nearest_sources(rows, a, b), a, b)
            for a, b, _d in items]
    t_eval = time.time()
    with _Evaluator(cfg, cfg.get("workers", 1)) as ev:
        results = ev.run(jobs)
    eval_s = time.time() - t_eval
    for n, ((a, b, d), r) in enumerate(zip(items, results), 1):
        _stamp(r, provenance="refine", level=level, depth=d, stage_seq=seq)
        rows[key_of(a, b)] = r
        _report(r, n, len(items), verbose)

    gone = set(straddling)
    store["cells"] = [c for i, c in enumerate(store["cells"]) if i not in gone] + new_cells
    cfg["stage_seq"] = seq
    record_timing(store, "refine", n_points=len(items),
                  n_solves=sum(r["n_solves"] for r in results),
                  wall_s=time.time() - t_stage, eval_s=eval_s, level=level, stage_seq=seq)
    save_store(store, out)
    n_ok = sum(1 for r in rows.values() if r["allowed"])
    print(f"[refine] store now {len(rows)} points ({n_ok} candidates), "
          f"{len(store['cells'])} quadtree cells -> {out}", flush=True)
    return store


# ============================================================================
# CLI.
# ============================================================================


def make_config(*, dmax: int, sector: str = "lin", n_c: int = 20, n_gap: int = 20,
                gap_lo: float = 0.04, gap_hi: float = 0.60,
                cm1_lo: float | None = None, cm1_hi: float | None = None,
                spacing: str | None = None, num_iter: int = 300,
                k_sv: int | None = None, n_train: int | None = None,
                sigma_round_kappa: float = 4.0, sigma_epsilon: float = 0.0,
                num_spin_rule: str = "v2", m_cold: int = 6, m_warm: int = 6,
                sigma_cold: float = 0.5, sigma_warm: float = 0.15,
                include_fb: bool = True, damping_lambda: float = 0.0,
                seed_base: int = 0, processes: int = 1, workers: int = 1) -> dict:
    """A grid-stage config. Defaults are the paper's landscape settings,
    except ``workers``/``processes``, which only change wall time.
    ``k_sv`` defaults to ``8 (Delta_max - 2)`` and ``n_train`` to ``2 k_sv``."""
    sec = dict(SECTORS[sector])
    if cm1_lo is not None:
        sec["cm1_lo"] = float(cm1_lo)
    if cm1_hi is not None:
        sec["cm1_hi"] = float(cm1_hi)
    if spacing is not None:
        sec["spacing"] = spacing
    k = int(k_sv) if k_sv is not None else 8 * (int(dmax) - 2)
    return dict(
        driver="primal.scan.continuation", sector=sector, dmax=int(dmax),
        n_c=int(n_c), n_gap=int(n_gap), gap_lo=float(gap_lo), gap_hi=float(gap_hi),
        num_iter=int(num_iter), k_sv=k, n_train=int(n_train) if n_train is not None else 2 * k,
        sigma_round_kappa=float(sigma_round_kappa), sigma_epsilon=float(sigma_epsilon),
        num_spin_rule=num_spin_rule, m_cold=int(m_cold), m_warm=int(m_warm),
        sigma_cold=float(sigma_cold), sigma_warm=float(sigma_warm),
        include_fb=bool(include_fb), damping_lambda=float(damping_lambda),
        seed_base=int(seed_base), processes=int(processes), workers=int(workers),
        level=LEVEL, round=0, stage_seq=0, scrub_gen=0, **sec,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("grid", "scrub", "refine"))
    ap.add_argument("--store", required=True, type=Path)
    g = ap.add_argument_group("grid stage (stored in the config; later stages reuse it)")
    g.add_argument("--dmax", type=int, default=6)
    g.add_argument("--sector", choices=tuple(SECTORS), default="lin")
    g.add_argument("--cm1-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="override the sector's c-1 range")
    g.add_argument("--spacing", choices=("log", "linear"), default=None,
                   help="override the sector's c-1 spacing")
    g.add_argument("--n-c", type=int, default=20)
    g.add_argument("--n-gap", type=int, default=20)
    g.add_argument("--gap-lo", type=float, default=0.04)
    g.add_argument("--gap-hi", type=float, default=0.60)
    g.add_argument("--num-iter", type=int, default=300)
    g.add_argument("--k-sv", type=int, default=None, help="default 8*(dmax-2)")
    g.add_argument("--n-train", type=int, default=None, help="default 2*k_sv")
    g.add_argument("--sigma-round-kappa", type=float, default=4.0)
    g.add_argument("--num-spin-rule", choices=("v2", "default"), default="v2")
    g.add_argument("--m-cold", type=int, default=6)
    g.add_argument("--m-warm", type=int, default=6)
    g.add_argument("--sigma-cold", type=float, default=0.5)
    g.add_argument("--sigma-warm", type=float, default=0.15)
    g.add_argument("--no-fb", action="store_true",
                   help="disable the free-boson lane (only to measure what it buys)")
    g.add_argument("--seed-base", type=int, default=0)
    r = ap.add_argument_group("every stage")
    r.add_argument("--workers", type=int, default=None,
                   help="cells evaluated concurrently (default 1); results do not depend on it")
    r.add_argument("--processes", type=int, default=None,
                   help="pool size inside one optimizer batch when workers=1 (default 1)")
    r.add_argument("--verbose", action="store_true")
    s = ap.add_argument_group("scrub stage (per invocation, not stored)")
    s.add_argument("--scrub-max-iter", type=int, default=20)
    s.add_argument("--scrub-sources", type=int, default=4,
                   help="best untried candidate neighbours carried per pass")
    s.add_argument("--warm-sweep", type=float, nargs="+", default=[0.05, 0.15, 0.35, 0.7],
                   help="sigma_warm values tried per target, best kept")
    s.add_argument("--scrub-gap-max", type=float, default=None,
                   help="scrub only cells with delta_gap <= this")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.stage == "grid":
        if args.store.exists():
            raise SystemExit(f"{args.store} exists; the grid stage never overwrites a store")
        cfg = make_config(
            dmax=args.dmax, sector=args.sector, n_c=args.n_c, n_gap=args.n_gap,
            gap_lo=args.gap_lo, gap_hi=args.gap_hi,
            cm1_lo=None if args.cm1_range is None else args.cm1_range[0],
            cm1_hi=None if args.cm1_range is None else args.cm1_range[1],
            spacing=args.spacing, num_iter=args.num_iter, k_sv=args.k_sv,
            n_train=args.n_train, sigma_round_kappa=args.sigma_round_kappa,
            num_spin_rule=args.num_spin_rule, m_cold=args.m_cold, m_warm=args.m_warm,
            sigma_cold=args.sigma_cold, sigma_warm=args.sigma_warm,
            include_fb=not args.no_fb, seed_base=args.seed_base,
            processes=args.processes or 1, workers=args.workers or 1,
        )
        stage_grid(cfg, args.store, verbose=args.verbose)
        return

    store = load_store(args.store)
    cfg = store["config"]
    if args.workers is not None:
        cfg["workers"] = args.workers
    if args.processes is not None:
        cfg["processes"] = args.processes
    if args.stage == "scrub":
        # Per invocation, not sticky: a later plain scrub must not inherit a band filter.
        cfg["scrub_gap_max"] = args.scrub_gap_max
        cfg["scrub_sources"] = args.scrub_sources
        cfg["warm_sweep"] = args.warm_sweep
        stage_scrub(store, args.store, verbose=args.verbose, max_iter=args.scrub_max_iter)
    else:
        stage_refine(store, args.store, verbose=args.verbose)


if __name__ == "__main__":
    main()
