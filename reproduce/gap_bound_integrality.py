"""Upper bound on the scalar gap from window-count integrality, optimised over the window width.

For a central charge ``c`` and a counted spin (0 or 1), every window width
``w`` gives a family of certified count brackets
(:func:`dual.scan.count_bracket.certified_count_bracket`) indexed by the
assumed scalar gap ``g``. The bracket at ``g`` bounds the count of every
unitary modular-invariant spectrum with a twist gap (no conserved currents)
and scalar gap ``>= g``, positivity checked through spin ``l_check`` (beyond
the twist gap it assumes only that no non-vacuum scalar lies below ``g``; the spin-0 window ``[g, g + w]``
is anchored at ``g``, the spin-1 window is ``[1, 1 + w]``). So a single
certified bracket at ``g`` that contains no admissible count (spin 0: no
integer; spin 1: no even integer) proves

    Delta_gap < g

for spectra with a twist gap (no conserved currents), parity-invariant when
the spin-1 ``2Z`` lattice is used (``--lattice Z`` drops parity), with spins
``<= l_check`` checked (a finite extension: spins above ``l_check`` are not
covered by this code).

Per width the driver searches downward in ``g`` for the smallest such
exclusion, ``g_exc(w)``, and reports

    Delta* = min over w of g_exc(w)

with the winning width and every probe. Only CERTIFIED exclusions count:
both sides pass the whole-window interval certificate, the in-SDP spin
certificate and the all-spin post-check, and the integrality test runs on the
rigorous bracket ``[A/(|a0_lo|+delta_lo), B/(|a0_up|-delta_up)]`` (a raw
bracket can miss the lattice and its widened one hit it). Uncertified
brackets, solver walls and errors are recorded and never counted.

The search strategy decides only how deep a width reaches, never the
validity of what it reports: each counted exclusion holds by itself, under
the assumptions above.
Both strategies assume one transition in ``g`` (excluded above, admitted
below); an admitted band can hide a deeper excluded one, which a finer
``--g-step`` or a lower ``--g-top`` finds.

- ``--search walk`` (default): probe ``g_top, g_top - step, ...`` and stop at
  the first probe that is not a certified exclusion. Probes above the strip
  that return no bracket (a ``wall``, where the lower solve is infeasible, or
  a solver error: solver statuses, not certificates) are walked through. With
  ``--backend sdpa_gmp`` a wall can also be an artefact of its float64 input
  (measured: spin 1 at c = 1.14, Lambda = 12 (order 11), where SDPB finds a
  certified bracket); SDPB is the reference.
- ``--search bisect``: probe ``g_top`` (must exclude) and ``--g-floor`` (must
  not), then bisect to ``--res``. Fewer probes; SDPB has no warm start, so
  every probe is a cold solve either way.
- Pruning (off with ``--no-prune``): once some width has ``g_exc``, a later
  width starts one step below it; if it does not exclude there it cannot win
  and costs one probe.

Settings
--------
--small         c = 1.14, spin 1, widths {0.2, 0.3}, g_top = 0.525, step
                0.0005, at most 6 probes per width, with the count bracket's
                --small recipe (Lambda = 8 (order 7), lmax = 16, NN = 20).
                Measured with ``--backend sdpb`` (one core, 8 min): walls at
                0.525 and 0.5245, certified exclusions at 0.524 and 0.5235
                (w = 0.2: [7.009, 7.910] holds no even integer), admitted at
                0.523, so Delta_gap < 0.5235; w = 0.3 is pruned. With the
                default SDPA-GMP backend (under a minute) SDPA-GMP produces
                NO certified bound: the probes at 0.524 miss the lattice but
                their functionals fail the in-SDP gate, so they are
                diagnostics only, and the run says that SDPB is needed.
                SDPA-GMP reads the constraint matrix in float64 and is not
                reliable at the edge of the allowed region. No other small
                setting tried did better with it (gaps 0.5232-0.524, w = 0.15
                -0.25, 1024 bits, Lambda = 10 (order 9), c = 1.1): each probe
                was a wall, admitted or failed the in-SDP gate, and it reports
                a wall at 0.5235 (w = 0.2) where SDPB certifies an exclusion.
(default)       the count bracket's paper recipe (Lambda = 20 (order 19),
                lmax = 40, SDPB 1024-bit, all-spin check to 90).
--nn / --sample-step   window samples NN = max(nn, ceil(w / sample_step) + 1);
                defaults nn = 40, sample_step = 0.077, the rule behind the
                paper's Lambda = 19 brackets (NN = 40 for w <= 2.9).

Paper settings (what produced the quoted gap bounds; the widths listed are
the winners of each width scan; lmax = 40 and 1024 bits throughout):
  * spin 0, Lambda = 20 (order 19): windows [g, g + w]; at c - 1 <= 5e-3 the
    scan over w in {0.7, 1.0, 1.4} is won by w = 1.0, at c ~ 1.02-1.03 by
    w ~ 0.75-0.83. Near c = 1 the window was sampled at step 0.005
    (``--sample-step 0.005 --nn 1``: NN = w/0.005 + 1) on a gap grid of 0.01.
  * spin 1, Lambda = 20 (order 19): windows [1, 1 + w]; the winning w grows
    toward c = 1, from 0.15 at c = 1.14 through 0.3 at c = 1.1 and 1.5 at
    c = 1.04 to 4.0 at c - 1 <= 1e-3. NN = 40 (the default rule) for c >= 1.03,
    sample step 0.005 near c = 1.
  * spin 1, Lambda = 50 (order 49), c - 1 in {1e-4, 3e-4, 1e-3, 3e-3}: wide
    windows w = 11-12.3 at sample step 0.005; certified exclusions
    Delta_gap < 0.098, 0.075, 0.070, 0.0483. Near c = 1 the brackets pin the spin-1
    count to 6 (its value on the c = 1 circle and orbifold lines), so an
    exclusion there needs A_r > 6 and B_r < 8. These solves take hours per side.
  The quoted high-Lambda cells (the Lambda = 50 (order 49) row) were also
  certified with an exact closure of the spin tail beyond spin 90, which
  this repository does not include: here the all-spin check stops at
  ``l_check``.

Usage::

    python reproduce/gap_bound_integrality.py --small --out small_gap.json
    python reproduce/gap_bound_integrality.py --c 1.001 --spin 1 --widths 3.6 4.0 4.4 \\
        --g-top 0.2 --g-step 0.005 --sample-step 0.005 --nn 1 --out s1_c1.001.json

(the second line uses the near-c = 1 sampling of the paper settings above,
NN = ceil(w / 0.005) + 1; its cost at these widths was not measured with
this code, so size it from a one-bracket smoke first).

``--out`` is rewritten after every probe; ``--resume`` reuses the probes
already in it (same recipe) so an interrupted search restarts where it
stopped. For a parallel width scan run one width per job (with
``--no-prune``) and take the minimum of the per-job ``delta_star``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from count_bracket import add_recipe_args, check_lattice, lattice_note, number, recipe_from  # noqa: E402,E501

SMALL = dict(c="1.14", spin=1, widths=["0.2", "0.3"], g_top="0.525", g_step="0.0005",
             max_probes=6)


def nn_for(w, nn, sample_step):
    """Window samples: ``max(nn, ceil(w / sample_step) + 1)`` (just ``nn`` without a step)."""
    if not sample_step:
        return int(nn)
    return max(int(nn), int(math.ceil(float(w) / float(sample_step))) + 1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--c")
    ap.add_argument("--spin", type=int, choices=[0, 1])
    ap.add_argument("--widths", nargs="+", help="window widths w to optimise over")
    ap.add_argument("--g-top", help="first (largest) gap probed; should be excluded")
    ap.add_argument("--g-step", help="walk step in g (default 0.004)")
    ap.add_argument("--g-floor", default="0", help="lowest gap probed (bisect: must be admitted)")
    ap.add_argument("--search", choices=["walk", "bisect"], default="walk")
    ap.add_argument("--res", default="0.001", help="bisection resolution in g")
    ap.add_argument("--max-probes", type=int, help="probes per width (default 20)")
    ap.add_argument("--no-prune", action="store_true", help="search every width from g_top")
    ap.add_argument("--sample-step", type=float, default=0.077,
                    help="NN = max(nn, ceil(w/sample_step)+1); 0 = use nn as given")
    add_recipe_args(ap)
    ap.add_argument("--out", help="JSON record, rewritten after every probe")
    ap.add_argument("--resume", action="store_true", help="reuse the probes already in --out")
    args = ap.parse_args(argv)

    pre = SMALL if args.small else {}
    pick = lambda k, v, d=None: v if v is not None else pre.get(k, d)
    c_s, spin = pick("c", args.c), pick("spin", args.spin)
    widths_s, g_top_s = pick("widths", args.widths), pick("g_top", args.g_top)
    if c_s is None or spin is None or not widths_s or g_top_s is None:
        ap.error("--c, --spin, --widths and --g-top are required (or --small)")
    max_probes = pick("max_probes", args.max_probes, 20)

    from mpmath import mp
    from dual.scan.count_bracket import certified_count_bracket, default_lattice, paper_order

    check_lattice(ap, args.lattice, spin)
    lattice = args.lattice or ("Z" if default_lattice(spin) == 1 else "2Z")
    c, g_top = number(c_s), number(g_top_s)
    g_step, g_floor, res = (number(pick("g_step", args.g_step, "0.004")),
                            number(args.g_floor), number(args.res))
    recipe = recipe_from(args)
    nn0 = recipe.pop("NN")
    if args.search == "bisect" and g_floor <= 0:
        ap.error("--search bisect needs a positive --g-floor that is admitted")

    run = dict(c=float(c), c_str=c_s, spin=spin, widths=[float(number(w)) for w in widths_s],
               g_top=float(g_top), g_step=float(g_step), g_floor=float(g_floor),
               search=args.search, res=float(res), max_probes=max_probes,
               prune=not args.no_prune, nn=nn0, sample_step=args.sample_step,
               Lambda_order=paper_order(recipe["Lambda"]),
               preset="small" if args.small else "paper", **recipe)
    out = dict(run=run, widths={}, probes=[], delta_star=None, w_star=None,
               lattice=lattice, parity_assumed=lattice == "2Z")

    cache = {}
    key = lambda w, g: (mp.nstr(w, 12), mp.nstr(g, 12))
    if args.resume and args.out and Path(args.out).exists():
        old = json.loads(Path(args.out).read_text())
        same = {k: v for k, v in old["run"].items() if k not in ("max_probes", "prune", "search")}
        if same != {k: v for k, v in run.items() if k not in ("max_probes", "prune", "search")}:
            ap.error(f"--resume: {args.out} was written with a different recipe")
        n_err = 0
        for p in old["probes"]:
            if p["verdict"] == "solver-error":   # transient (container, timeout): re-run
                n_err += 1
                continue
            cache[(p["w_key"], p["g_key"])] = p
        print(f"resume: {len(cache)} probes from {args.out}"
              + (f" ({n_err} solver-error probe(s) will be re-run)" if n_err else ""),
              flush=True)

    def save():
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=1))

    def probe(w, g):
        """Certified bracket at (w, g); returns its verdict."""
        k = key(w, g)
        rec = cache.get(k)
        if rec is None:
            rec = certified_count_bracket(c, g, spin, w, NN=nn_for(w, nn0, args.sample_step),
                                          **recipe)
            rec.update(w_key=k[0], g_key=k[1])
        out["probes"].append(rec)
        f = lambda x: "-" if x is None else f"{x:.5f}"
        B_r = "inf" if rec.get("B_r_str") == "inf" else f(rec["B_r"])
        print(f"  w={k[0]} g={k[1]}: [A_r, B_r] = [{f(rec['A_r'])}, {B_r}] "
              f"(raw [{f(rec['A'])}, {f(rec['B'])}]) certified={rec['certified']} -> "
              f"{rec['verdict']}  ({rec['wall']:.0f} s)", flush=True)
        save()
        return rec["verdict"]

    def walk(w, start):
        g, deepest, n = start, None, 0
        while n < max_probes and g >= g_floor:
            n += 1
            v = probe(w, g)
            if v == "excluded":
                deepest = g
            elif not (v in ("wall", "solver-error") and deepest is None):
                break                    # start not excluded, or below the excluded strip
            g = g - g_step               # (no bracket above the strip: walk through)
        return deepest, n

    def bisect(w, start):
        if probe(w, start) != "excluded":
            return None, 1
        if probe(w, g_floor) == "excluded":
            return g_floor, 2
        lo, hi, n = g_floor, start, 2
        while hi - lo > res and n < max_probes:
            mid = (lo + hi) / 2
            n += 1
            if probe(w, mid) == "excluded":
                hi = mid
            else:
                lo = mid
        return hi, n

    t0 = time.time()
    best = None
    for ws in widths_s:
        w = number(ws)
        step = g_step if args.search == "walk" else res
        start = g_top if (best is None or args.no_prune) else min(g_top, best[1] - step)
        print(f"w = {ws} (NN {nn_for(w, nn0, args.sample_step)}), start g = "
              f"{mp.nstr(start, 8)}", flush=True)
        g_exc, n = (walk if args.search == "walk" else bisect)(w, start)
        state = ("excluded" if g_exc is not None
                 else "pruned" if start < g_top else "start not excluded")
        out["widths"][mp.nstr(w, 12)] = dict(
            w=float(w), NN=nn_for(w, nn0, args.sample_step), start=float(start),
            g_exc=None if g_exc is None else float(g_exc),
            g_exc_str=None if g_exc is None else mp.nstr(g_exc, 15),
            n_probes=n, state=state)
        if g_exc is not None and (best is None or g_exc < best[1]):
            best = (w, g_exc)
        if best is not None:
            out.update(w_star=float(best[0]), delta_star=float(best[1]),
                       delta_star_str=mp.nstr(best[1], 15))
        save()

    out["wall"] = time.time() - t0
    save()
    print(f"\nc = {c_s}, spin {spin}, Lambda = {recipe['Lambda']} (order "
          f"{paper_order(recipe['Lambda'])}), {len(out['probes'])} probes, "
          f"{out['wall']:.0f} s")
    for k, v in out["widths"].items():
        print(f"  w = {k:<8} g_exc = {v['g_exc_str'] or '-':<12} ({v['state']}, "
              f"{v['n_probes']} probes)")
    if best is None:
        print("no certified exclusion: raise --g-top or change the widths")
        n_gate = sum(1 for p in out["probes"] if p["verdict"] == "uncertified")
        if n_gate:
            print(f"  {n_gate} probe(s) miss the lattice but fail a certificate gate")
        if recipe["backend"] == "sdpa_gmp":
            print("  sdpa_gmp produced no certified bound: its brackets here are diagnostics "
                  "only (float64 constraint input). A certified bound needs SDPB "
                  "(--backend sdpb).")
    else:
        print(f"Delta_gap < {out['delta_star_str']}  (certified; winning width w = "
              f"{mp.nstr(best[0], 8)}; {lattice_note(lattice, recipe['l_check'])})")
    if args.out:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
