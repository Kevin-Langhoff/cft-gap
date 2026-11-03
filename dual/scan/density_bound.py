"""``bound_sab`` — single-window two-sided density bound with resampling.

Port of ``boundSAB`` (the discrete-sampling toolkit): solve the window
density-bound SDP over discrete samples, evaluate the recovered functional
``f(Delta) = a0 + alpha' . Comwithprefactor(Delta, spin)`` on a dense
(ngrid+1)-point grid across ``[a, b]``, add ``{p - step/2, p, p + step/2}`` at
every strictly-negative grid point and re-solve, up to ``max_outer + 1`` solves
(``While[nouter <= maxOuter]``). Only an "optimal" solver termination is accepted.

``converged`` semantics (a rigor upgrade over the reference):
- grid clean within the round budget -> True (the reference criterion). The
  high-precision whole-window certificate
  (:func:`dual.postcheck.certify_window_constraint`, ball subdivision over
  ``[a, b]`` — grid-positive is not interval-positive) is RECORDED in
  ``arb_certified`` but does NOT gate this verdict, so grid-clean stays
  comparable to the reference.
- resample STAGNATION (every negative grid point's ``{p-step/2, p,
  p+step/2}`` triple is already a sample — a touching zero of the extremal
  functional, where the reference would spin out all ``maxOuter`` rounds): we
  break early and the high-precision certificate ADJUDICATES — ``converged`` is
  True iff ``f >= -delta`` over the whole window (a touching zero passes; a
  genuine dip refutes with a witness). This unblocks the reference gold upper
  bound (hi=6.2074), which sits on such a touching zero.

``arb_info`` carries the ``stagnated`` flag and a ``min_grid_val`` diagnostic.

Needs the semidefinite solver container when run against the real backend;
``solve_fn`` injection keeps the loop unit-testable without one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mpmath import mpf

from .resample import (
    insert_triples,
    negative_points,
    window_functional_values,
)
from ..assembly import _prep_window_bound, build_window_bound_sdp


@dataclass
class WindowBoundResult:
    """Mirror of boundSAB's ``{converged, bound, rounds, samplePts}`` + extras."""

    converged: bool          # grid-clean within max_outer (NOT the certificate verdict)
    bound: object            # mpf Zmin; None on solver failure
    status: str              # final solver status / raw reason
    rounds: int              # number of solves performed
    extra_points: list       # final resampling points (morex)
    alpha: object            # full functional [a0] + deriv (z.txt layout); None on failure
    arb_certified: object    # True/False from the window certificate; None if skipped
    arb_info: dict = field(default_factory=dict)


def bound_sab(c, gap, spin, a, b, sign, Lmax, lmax, NNsample, ngrid, max_outer, *,
              precision_bits=1024, np_ranks=None, arb_certify=True,
              allow_conditional=False, solve_fn=None, verbose=False, prep=None):
    """Two-sided window density bound (``sign`` = ``"min"`` lower / ``"max"`` upper).

    ``prep``: optional precomputed ``(pairs, Base0, gaplist, pols2)`` from
    :func:`_prep_window_bound` for the SAME ``(c, gap, spin, a, b, Lmax, lmax)``.
    Since prep is SIGN-independent, a caller issuing both the ``"min"`` and
    ``"max"`` solve for one window (``RobustEval.single_bracket``) builds it ONCE
    and hands it to both, skipping the dominant ``_build_pols2`` rebuild on the
    second side. Bit-identical to ``prep=None`` (which builds it internally).
    """
    if sign not in ("min", "max"):
        raise ValueError(f"sign must be 'min' or 'max', got {sign!r}")
    c = mpf(c)
    gap = mpf(gap)
    a = mpf(a)
    b = mpf(b)
    step = (b - a) / ngrid

    if solve_fn is None:
        from ..backends import sdpb

        def solve_fn(sdp, **kw):
            return sdpb.solve_sdp(sdp, **kw)

    def _certify(result):
        """Whole-window certificate for the recovered functional."""
        if not arb_certify:
            return None, {}
        from ..postcheck import certify_window_constraint
        return certify_window_constraint(result.alpha[0], result.alpha[1:],
                                         c, a, b, spin, Lmax)

    def _annotate(info, grid_vals, *, stagnated):
        info = dict(info or {})
        info["stagnated"] = stagnated
        if grid_vals:
            info["min_grid_val"] = float(min(v for (_, v) in grid_vals))
        return info

    # Round-independent blocks (pairs/Base0/gaplist/pols2) depend only on the
    # loop-invariant args, so build them once and reuse across resampling
    # rounds instead of rebuilding pols2 every round (bit-identical). The caller
    # may also supply prep (sign-independent) to share it across the min/max pair.
    if prep is None:
        prep = _prep_window_bound(c, gap, spin, a, b, Lmax, lmax,
                                  allow_conditional=allow_conditional)
    morex = []
    res = None
    vals = None
    nouter = 0
    rounds = 0
    stagnated = False
    while nouter <= max_outer:
        sdp = build_window_bound_sdp(c, gap, spin, a, b, NNsample, Lmax, lmax,
                                     sign=(+1 if sign == "min" else -1),
                                     extra_points=morex,
                                     allow_conditional=allow_conditional,
                                     prep=prep)
        res = solve_fn(sdp, sense=sign, precision_bits=precision_bits,
                       max_iteration=2000, np_ranks=np_ranks)
        rounds += 1
        if res.status != "optimal":
            # Anything but "found primal-dual optimal solution" -> "No bound".
            return WindowBoundResult(converged=False, bound=None,
                                     status=(getattr(res, "reason", None) or res.status),
                                     rounds=rounds, extra_points=morex,
                                     alpha=None, arb_certified=None)

        vals = window_functional_values(res.alpha[0], res.alpha[1:], c, spin,
                                        a, b, Lmax, ngrid)
        negs = negative_points(vals)
        if verbose:
            print(f"> round {nouter}, extra={len(morex)}, neg={len(negs)}, "
                  f"bound={res.bound}")
        if not negs:
            # Grid-clean: the reference's exit. The certificate is RECORDED but
            # does NOT gate this verdict (grid-clean is the acceptance criterion).
            arb_ok, arb_info = _certify(res)
            return WindowBoundResult(converged=True, bound=res.bound,
                                     status=res.status, rounds=rounds,
                                     extra_points=morex, alpha=res.alpha,
                                     arb_certified=arb_ok,
                                     arb_info=_annotate(arb_info, vals,
                                                        stagnated=False))
        new_morex = insert_triples(morex, negs, step)
        if len(new_morex) == len(morex):
            # Every negative grid point's {p-step/2, p, p+step/2} triple is
            # already a sample: resampling is a no-op (a touching zero of the
            # extremal functional). The reference spins out all maxOuter rounds
            # here; we break early and let the certificate adjudicate below.
            stagnated = True
            break
        morex = new_morex
        nouter += 1

    # Non-clean exit (resample stagnation, or max_outer exhausted): the grid
    # still shows a negative point. Unlike the reference (which returns the bound
    # uncertified), run the rigorous window certificate -- a touching zero
    # certifies (f >= -delta), a genuine dip refutes with a witness. converged
    # is True ONLY if the certificate passes.
    arb_ok, arb_info = _certify(res)
    return WindowBoundResult(converged=bool(arb_ok), bound=res.bound,
                             status=res.status, rounds=rounds,
                             extra_points=morex, alpha=res.alpha,
                             arb_certified=arb_ok,
                             arb_info=_annotate(arb_info, vals,
                                                stagnated=stagnated))
