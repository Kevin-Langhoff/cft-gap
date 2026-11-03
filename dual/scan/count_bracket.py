"""``certified_count_bracket`` — one two-sided window count bracket, certified end to end.

The bracket ``[A, B]`` bounds the number of primaries of one spin in a window
``[a, b]`` for every unitary modular-invariant spectrum with a twist gap (no
conserved currents) whose non-vacuum scalars all have ``Delta >= gap``, with
positivity checked through spin ``l_check``. Both sides come from
:func:`dual.scan.density_bound.bound_sab` (the resampling window solve) built on
ONE shared :func:`dual.assembly._prep_window_bound`.

Window rule (anchored at the sector gap, so no second-gap assumption enters):
- spin 0: ``[gap, gap + w]``;
- spin s >= 1: ``[s, s + w]`` (unitarity floor; spin 1 -> ``[1, 1 + w]``).

The solver imposes the window constraint only at sample points. Per side the
recovered functional is then certified on the WHOLE window continuum by
:func:`dual.postcheck.certify_window_constraint` (adaptive Arb ball
subdivision), run as a ladder ``(rel_tol, max_depth)`` = ``(1e-9, 60)``,
``(1e-7, 60)``, ``(1e-6, 80)`` that stops at the first pass; a pass at 1e-9 is
replayed DOWN at 1e-11 and 1e-13 (depth 100), keeping the last passing run.
The certificate proves ``f = a0 + (window term) >= -delta`` for the functional
as recovered, ``a0 = alpha[0]`` its count bookkeeping slot and ``delta`` an
ABSOLUTE slack set by the passing ``rel_tol`` (it is the certificate's
resolution, so tightening shrinks it). The rigorous bracket is

    [A / (|a0_lo| + delta_lo), B / (|a0_up| - delta_up)]   (B side unbounded if delta_up >= |a0_up|),

with ``A`` / ``B`` the bounds of the same unnormalised functionals, the
denominators formed exactly and the quotients rounded outward (``A_r`` down,
``B_r`` up) at the working precision. SDPB's normalization fold returns
``a0 = +-1`` exactly; SDPA-GMP meets the normalization row only to its float64
input, so its ``|a0|`` differs from 1 in the last float64 digits.

Each functional is then checked on every spin continuum, on two gates:

- in-SDP blocks (``inblock``): the continua the SDP imposed, spins
  ``0 .. lmax`` with their gaps, by the exact half-line certificate
  :func:`dual.verify.certify_functional` on the SDP's continuum blocks (the
  window samples are left to the interval certificate). The solver satisfies
  these only to its own tolerance; SDPA-GMP, which ingests the constraint
  matrix in float64, can return an "optimal" functional that is negative on
  them (measured at the --small recipe, c = 1.14: spin-0 functionals negative
  at the spin 6-9 thresholds, with bounds up to 0.13 off SDPB's certified ones).
- all-spin (``allspin``): the spins the SDP left out, ``lmax < l <= l_check``,
  by :func:`dual.postcheck.postcheck_spin_truncation` (a finite extension;
  spins above ``l_check`` are not covered).

Both half-line gates run at zero tolerance (``rel_tol = 0``: ``p >= 0``
exactly). The banded default of :func:`dual.verify.certify_nonneg_on_halfline`
(``p >= -1e-9 max|coeff|``) is too lenient here: it accepted an SDPA-GMP lower
functional that is negative at the spin-6 threshold by 7e-11 of its
coefficient scale, whose bound sat 0.13 above the certified SDPB bound
(c = 1.14, gap 0.524, w = 0.05, --small recipe). Every SDPB functional tested
passes at zero tolerance (12 sides: three paper-recipe brackets, one order-11
bracket and two --small brackets).

``certified`` = both interval certificates pass AND both sides are clean on
both half-line gates. It is computed only after every ingredient is in the
record. A bracket that fails any gate is a DIAGNOSTIC: its numbers are
recorded, it never excludes anything. Solver statuses are not certificates either: an
``infeasible`` lower side is reported as a ``wall`` and an ``infeasible``
upper side as ``admitted-unbounded`` (the count is unbounded above).

Integrality: admissible counts lie on ``Z`` for spin 0 and on ``2Z`` for spin
s >= 1. The ``2Z`` lattice ASSUMES parity invariance (a parity-invariant
spectrum carries spin +s and -s in pairs, and the window counts both). ``Z``
for spin s >= 1 is allowed (conservative); a lattice coarser than
:func:`default_lattice` (``2Z`` for spin 0) is rejected with ``ValueError``.
The test runs on the rigorous bracket through the exact
:func:`dual.bounds.integer_in` (mpmath floor/ceil, no float64 cast). Counts are
non-negative, so the lattice is cut at 0.

Scope of an exclusion: a certified bracket with no admissible count excludes
the assumed gap for spectra with a twist gap (no conserved currents:
``Delta > s`` for every non-vacuum primary of spin ``s >= 1``), parity-invariant
when the spin-1 ``2Z`` lattice is used, with spins ``<= l_check`` checked.
Spins above ``l_check`` are not covered by this code.

Precision tiers: only the ``sdpb`` and ``sdpa_gmp`` backends are accepted here;
both are arbitrary precision, and their output passes through the certificates
above. Lambda is the code's derivative cutoff; the real derivative order (the
paper's Lambda) is ``Lambda`` if odd, else ``Lambda - 1``.
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

from mpmath import mp, mpf

from .density_bound import bound_sab
from ..assembly import _prep_window_bound
from ..bounds import _SdpView, integer_in
from ..postcheck import certify_window_constraint, postcheck_spin_truncation
from ..verify import certify_functional

# Tolerance of the two half-line gates (in-SDP blocks, all-spin): exact.
HALFLINE_REL_TOL = 0

# (rel_tol, max_depth) escalation for the whole-window interval certificate,
# then the replay DOWN in rel_tol for a pass at the first rung.
CERT_LADDER = ((1e-9, 60), (1e-7, 60), (1e-6, 80))
CERT_TIGHTEN = ((1e-11, 100), (1e-13, 100))

BACKENDS = ("sdpb", "sdpa_gmp")
# SDPA-GMP ingests the constraint matrix in float64 (dual.backends.sdpa_gmp), so
# its stopping tolerance cannot go far below 1e-12: at 1e-15 the --small spin-0
# upper side stalls (pdINF at a 1e-6 relative gap) and bound_sab rejects it; at
# 1e-12 it reaches pdOPT. The bound is recomputed in mpmath from the functional,
# which the certificates below check independently.
SDPA_EPSILON = mpf(10) ** -12


def paper_order(Lambda):
    """Real derivative order of code cutoff ``Lambda`` (the paper's Lambda)."""
    return int(Lambda) if int(Lambda) % 2 else int(Lambda) - 1


def count_window(spin, gap, w):
    """The counted window ``(a, b)``: spin 0 -> ``[gap, gap+w]``, spin s -> ``[s, s+w]``."""
    a = mpf(gap) if spin == 0 else mpf(spin)
    return a, a + mpf(w)


def default_lattice(spin):
    """Spacing of admissible counts: 1 (Z) for spin 0, 2 (2Z) for spin >= 1."""
    return 1 if spin == 0 else 2


def admissible_counts(lo, hi, step, limit=12):
    """Non-negative multiples of ``step`` in the closed ``[lo, hi]`` (``hi`` may be inf)."""
    lo = max(mpf(lo), mpf(0))
    n = int(mp.ceil(lo / step))
    out = []
    while len(out) < limit and (hi == mp.inf or n * step <= hi):
        out.append(n * step)
        n += 1
    return out


def contains_admissible(lo, hi, step):
    """Exact test: does ``[lo, hi]`` contain a non-negative multiple of ``step``?"""
    if hi == mp.inf:
        return True
    return integer_in(max(mpf(lo), mpf(0)) / step, mpf(hi) / step)


def _short_status(status):
    """Solver status on the short scale (``optimal`` / ``feasible`` / ``infeasible`` /
    ``unbounded`` / ``error``); ``bound_sab`` keeps SDPB's raw terminate reason."""
    if status in ("optimal", "feasible", "infeasible", "unbounded", "error", "skipped"):
        return status
    from ..backends.sdpb import _status_from_reason
    return _status_from_reason(status or "")


def _solve_fn(backend):
    if backend == "sdpb":
        return None          # bound_sab's default: dual.backends.sdpb
    if backend == "sdpa_gmp":
        from ..backends import sdpa_gmp

        def solve(sdp, np_ranks=None, **kw):
            return sdpa_gmp.solve_sdp(sdp, epsilon=SDPA_EPSILON, **kw)
        return solve
    raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")


def interval_certificate(alpha, c, a, b, spin, Lambda):
    """Whole-window certificate of one functional as a ladder; returns a record.

    ``ok`` / ``rel_tol`` / ``max_depth`` / ``delta`` / ``reason`` / ``witness_Delta``
    belong to the run that decides: the last passing run, or the last failing
    rung if none passed. ``delta`` is meaningful only when ``ok``. ``ladder``
    keeps every run in order.
    """
    al = [mpf(x) for x in alpha]

    def run(tol, depth):
        ok, info = certify_window_constraint(al[0], al, c, a, b, spin, Lambda,
                                             rel_tol=mpf(tol), max_depth=depth)
        return dict(ok=bool(ok), rel_tol=tol, max_depth=depth,
                    delta=info.get("delta"), reason=info.get("reason"),
                    witness_Delta=info.get("witness_Delta"))

    ladder = []
    best = None
    for tol, depth in CERT_LADDER:
        best = run(tol, depth)
        ladder.append(best)
        if best["ok"]:
            break
    if best["ok"] and best["rel_tol"] == CERT_LADDER[0][0]:
        for tol, depth in CERT_TIGHTEN:
            r = run(tol, depth)
            ladder.append(r)
            if not r["ok"]:
                break
            best = r
    return dict(best, ladder=[{k: q[k] for k in ("rel_tol", "max_depth", "ok", "reason",
                                                  "delta")} for q in ladder])


def inblock_certificate(alpha, prep):
    """Exact half-line certificate of ``alpha`` on the SDP's continuum blocks (spins 0..lmax).

    The view is built from ``prep``'s continuum blocks alone: they do not
    depend on the sign, the samples or the resampling points.
    """
    view = _SdpView(SimpleNamespace(pvms=prep[3]))
    ok, viol = certify_functional([mpf(x) for x in alpha], view, rel_tol=HALFLINE_REL_TOL)
    return dict(ok=bool(ok), n_viol=len(viol), n_blocks=len(view.pols2),
                witness=[{k: (mp.nstr(v, 12) if isinstance(v, mpf) else v)
                          for k, v in q.items() if not isinstance(v, list)}
                         for q in viol[:3]])


def _side(c, gap, spin, a, b, sense, recipe, prep, solve_fn):
    """One sign: solve, then the interval certificate and the two half-line gates.

    Returns ``(record, bound, a0)``: the JSON-ready record, the ``mpf`` bound of
    the unnormalised functional and its bookkeeping slot ``a0 = alpha[0]``.
    """
    t0 = time.time()
    res = bound_sab(c, gap, spin, a, b, sense, recipe["Lambda"], recipe["lmax"],
                    recipe["NN"], recipe["ngrid"], recipe["max_outer"],
                    precision_bits=recipe["precision_bits"], np_ranks=recipe["np_ranks"],
                    arb_certify=True, allow_conditional=False, solve_fn=solve_fn,
                    prep=prep)
    t1 = time.time()
    out = dict(sense=sense, status=_short_status(res.status), reason=res.status,
               rounds=res.rounds,
               bound=None if res.bound is None else float(res.bound),
               bound_str=None if res.bound is None else mp.nstr(res.bound, 40),
               converged=bool(res.converged), inline_cert=res.arb_certified,
               n_extra_points=len(res.extra_points),
               alpha=None if res.alpha is None else [str(x) for x in res.alpha],
               interval_cert=None, inblock=None, allspin=None, wall_solve=t1 - t0)
    if res.alpha is None:
        out["wall"] = time.time() - t0
        return out, None, None
    out["interval_cert"] = interval_certificate(res.alpha, c, a, b, spin, recipe["Lambda"])
    out["inblock"] = inblock_certificate(res.alpha, prep)
    t2 = time.time()
    ok, rows = postcheck_spin_truncation(res.alpha, c, recipe["Lambda"], recipe["lmax"],
                                         recipe["l_check"], rel_tol=HALFLINE_REL_TOL)
    out["allspin"] = dict(l_check=recipe["l_check"], ok=bool(ok),
                          n_viol=int(sum(0 if q["ok"] else 1 for q in rows)),
                          witness=[q for q in rows if not q["ok"]][:3])
    t3 = time.time()
    out.update(wall_cert=t2 - t1, wall_allspin=t3 - t2, wall=t3 - t0)
    return out, res.bound, mpf(res.alpha[0])


def _delta_up(x):
    """A recorded float delta, rounded one ulp up (it is an upper bound on the slack)."""
    return math.nextafter(float(x), math.inf)


def certified_count_bracket(c, gap, spin, w, *, Lambda=20, lmax=40, NN=40, ngrid=None,
                            max_outer=4, precision_bits=1024, backend="sdpb",
                            np_ranks=1, l_check=90, lattice=None):
    """Certified two-sided bracket on the spin-``spin`` count in the window of width ``w``.

    Defaults are the paper recipe: ``Lambda=20`` (derivative order 19),
    ``lmax=40`` spin blocks, ``NN=40`` samples (also the resampling grid unless
    ``ngrid`` is given), ``max_outer=4`` resampling rounds, 1024-bit SDPB,
    all-spin post-check to ``l_check=90``. ``lattice`` overrides the spacing of
    admissible counts (default :func:`default_lattice`); a spacing coarser than
    the default raises ``ValueError``.

    Returns a JSON-ready dict: the recipe, per side ``lo`` / ``up`` (bound,
    status, rounds, functional ``alpha`` as strings, ``interval_cert`` with its
    ``delta``, ``inblock``, ``allspin``, walls), the raw ``A`` / ``B``, the rigorous
    ``A_r`` / ``B_r`` (None where a side has no certificate; an unbounded
    ``B_r`` is written as None with ``B_r_str = "inf"``), ``certified``, the
    lattice and whether it assumes parity, the admissible-count test and a
    ``verdict``: ``excluded`` (certified, no admissible count: no spectrum with
    this gap exists among those with a twist gap, parity-invariant if the
    lattice is ``2Z``, with spins ``<= l_check`` checked), ``admitted``,
    ``admitted-unbounded``, ``uncertified`` (no admissible count but a
    certificate failed: a diagnostic, not an exclusion), ``wall`` or
    ``solver-error``.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    spin = int(spin)
    c, gap, w = mpf(c), mpf(gap), mpf(w)
    a, b = count_window(spin, gap, w)
    step = int(lattice or default_lattice(spin))
    if not 1 <= step <= default_lattice(spin):
        raise ValueError(f"lattice {step}Z is coarser than the admissible counts of spin "
                         f"{spin}; only Z" + (" or 2Z" if default_lattice(spin) == 2 else "")
                         + " is allowed")
    recipe = dict(Lambda=int(Lambda), lmax=int(lmax), NN=int(NN),
                  ngrid=int(ngrid or NN), max_outer=int(max_outer),
                  precision_bits=int(precision_bits), np_ranks=int(np_ranks or 1),
                  l_check=int(l_check), backend=backend)
    solve_fn = _solve_fn(backend)

    t0 = time.time()
    prep = _prep_window_bound(c, gap, spin, a, b, recipe["Lambda"], recipe["lmax"],
                              allow_conditional=False)
    wall_prep = time.time() - t0
    lo, A, a0_lo = _side(c, gap, spin, a, b, "min", recipe, prep, solve_fn)
    if A is None:
        # no lower bound -> no bracket to complete
        up, B, a0_up = dict(sense="max", status="skipped", bound=None, interval_cert=None,
                            inblock=None, allspin=None), None, None
    else:
        up, B, a0_up = _side(c, gap, spin, a, b, "max", recipe, prep, solve_fn)

    rec = dict(c=float(c), gap=float(gap), spin=spin, w=float(w), a=float(a), b=float(b),
               c_str=mp.nstr(c, 30), gap_str=mp.nstr(gap, 30), w_str=mp.nstr(w, 30),
               lattice="Z" if step == 1 else f"{step}Z", parity_assumed=step == 2,
               Lambda_order=paper_order(Lambda), **recipe, lo=lo, up=up,
               A=lo["bound"], B=up["bound"], wall_prep=wall_prep)

    # rigorous ends: only from a passing interval certificate. The certificate
    # bounds a0 + (window term) >= -delta for the unnormalised functional, so
    # the ends divide by |a0| +- delta (a0 > 0 on the lower side, < 0 on the
    # upper); denominators exact, quotients rounded outward.
    cl, cu = lo["interval_cert"], up["interval_cert"]
    dlo = _delta_up(cl["delta"]) if cl and cl["ok"] else None
    dup = _delta_up(cu["delta"]) if cu and cu["ok"] else None
    if A is None or dlo is None or not a0_lo > 0:
        A_r = None
    else:
        A_r = mp.fdiv(A, mp.fadd(a0_lo, dlo, exact=True), rounding="f")
    if B is None or dup is None or not a0_up < 0:
        B_r = None
    else:
        den = mp.fsub(-a0_up, dup, exact=True)
        B_r = mp.inf if den <= 0 else mp.fdiv(B, den, rounding="c")
    rec.update(delta_lo=dlo, delta_up=dup,
               a0_lo=None if a0_lo is None else mp.nstr(a0_lo, 30),
               a0_up=None if a0_up is None else mp.nstr(a0_up, 30),
               A_r=None if A_r is None else float(A_r),
               B_r=None if (B_r is None or B_r == mp.inf) else float(B_r),
               A_r_str=None if A_r is None else mp.nstr(A_r, 30),
               B_r_str=None if B_r is None else ("inf" if B_r == mp.inf
                                                 else mp.nstr(B_r, 30)))

    # every ingredient is in the record: now the certificate verdict
    clean = lambda g: all(s[g] is not None and s[g]["ok"] and s[g]["n_viol"] == 0
                          for s in (lo, up))
    inblock_ok, allspin_ok = clean("inblock"), clean("allspin")
    certified = bool(A_r is not None and B_r is not None and inblock_ok and allspin_ok)
    rec.update(inblock_clean=inblock_ok, allspin_clean=allspin_ok, certified=certified)

    if A is None:
        verdict, has = ("wall" if lo["status"] == "infeasible" else "solver-error"), None
    elif B is None:
        verdict = "admitted-unbounded" if up["status"] == "infeasible" else "solver-error"
        has = True if verdict == "admitted-unbounded" else None
    else:
        # the widest bracket in hand: rigorous ends where certified, raw otherwise
        lo_end = A_r if A_r is not None else A
        hi_end = B_r if B_r is not None else B
        has = contains_admissible(lo_end, hi_end, step)
        if hi_end == mp.inf:
            verdict = "admitted-unbounded"
        elif has:
            verdict = "admitted"
        else:
            verdict = "excluded" if certified else "uncertified"
        rec["admissible"] = [int(n) for n in admissible_counts(lo_end, hi_end, step)]
    rec.update(contains_admissible=has, verdict=verdict, excluded=verdict == "excluded",
               wall=time.time() - t0)
    return rec
