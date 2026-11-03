"""Post-hoc validity checks for recovered dual functionals (hardening layer).

``certify_functional`` certifies a functional only on the spin sectors PRESENT
in the SDP (l <= lmax). Spins ABOVE the truncation are absent from both the
SDP and the certificate, and for the dual-functional method that errs on the
INVALID side: fewer constraints enlarge the feasible set, so a recovered
functional may be negative on some spin-(l > lmax) operator -- making an
"exclusion" unsound. The risk grows with c (the crossing kernel's natural spin
scale grows), exactly where the gap curve extends to large c.

``postcheck_spin_truncation`` closes the seam up to a finite ``l_check``: for
each l in (lmax, l_check] it forms the exact action polynomial of the SAME
functional on the spin-l continuum (starting at that sector's gap assumption)
and runs the exact half-line certificate. All-pass means the exclusion is
valid with the spin truncation extended to ``l_check``; any failure is a real
refutation witness (the exclusion needs lmax raised, not a numerics flag).

This is a FINITE extension (l <= l_check), not an asymptotic l -> inf proof;
the asymptotic tail argument is a separate analytic step. Margins/trends over
l are reported so a clearly-degrading functional is visible before it fails.

Pure mpmath/flint arithmetic -- no SDP solves; cheap enough to run inline on
every excluded cell.
"""
from __future__ import annotations

import math
from fractions import Fraction

from flint import arb, ctx as _flint_ctx, fmpq
from mpmath import mp, mpf

from .characters import tb
from .verify import (_mpf_to_fraction, build_pols2_action,
                     certify_nonneg_on_halfline)


def postcheck_spin_truncation(alpha, c, Lmax, lmax, l_check, *,
                              gap_for_spin=None, rel_tol=mpf(10) ** -9):
    """Certify a functional's positivity on spins lmax+1 .. l_check.

    - ``alpha``: the FULL recovered functional INCLUDING the bookkeeping
      slot 0 (``BoundResult.alpha`` / z.txt layout, length len(tb)+1). Slot 0
      multiplies only the counted-window indicator, which is absent from every
      continuum block, so it is dropped exactly.
    - ``gap_for_spin``: callable l -> continuum start for that sector. Default
      is the spin-method convention (unitarity, ``g_l = l``); pass
      ``lambda l: max(gap, l)`` for the interval-method convention.

    Returns ``(ok, rows)``: ok iff every sector certifies; rows carry per-spin
    verdicts and refutation witnesses (x_star above the sector gap).
    """
    pairs = tb(Lmax)
    a = [mpf(x) for x in alpha]
    if len(a) == len(pairs) + 1:
        a = a[1:]
    elif len(a) != len(pairs):
        raise ValueError(f"alpha length {len(a)} matches neither len(tb)={len(pairs)}"
                         f" nor len(tb)+1")
    if gap_for_spin is None:
        gap_for_spin = lambda l: mpf(l)

    rows = []
    ok_all = True
    for l in range(lmax + 1, l_check + 1):
        g_l = mpf(gap_for_spin(l))
        coeffs = build_pols2_action(a, pairs, c, g_l, l)
        ok, info = certify_nonneg_on_halfline(coeffs, rel_tol=rel_tol)
        row = {"spin": l, "gap": float(g_l), "ok": bool(ok)}
        if not ok:
            ok_all = False
            row["reason"] = info.get("reason")
            row["witnesses_x"] = [mp.nstr(w, 15)
                                  for w in info.get("sign_change_roots", [])]
        rows.append(row)
    return ok_all, rows


def _arb_exact(x):
    """mpf/int/Fraction -> arb, exactly (via the dyadic/rational value)."""
    if isinstance(x, Fraction):
        return arb(fmpq(x.numerator, x.denominator))
    f = _mpf_to_fraction(mpf(x))
    return arb(fmpq(f.numerator, f.denominator))


def certify_window_constraint(a0, alpha, c, win_lo, win_hi, s, Lmax, *,
                              rel_tol=mpf(10) ** -9, max_depth=42, prec=512):
    """Certify the counted-window constraint on the FULL window continuum.

    The window builders impose ``a0 + 2 e^{-2pi(Delta-(c-1)/12)} P(Delta) >= 0``
    only at the NN+1 grid points of the counted window (P = the functional's
    Com action at spin ``s``; spin windows: [LL, LL+dela] with s=LL; scalar
    intervals: [gap, gap+w] with s=0). An off-grid operator inside the
    window is otherwise unconstrained -- the discretization validity seam this
    closes: certify ``f(x) = a0 + 2 e^{-2pi(x + win_lo - (c-1)/12)} P(x) >=
    -delta`` for ALL ``x in [0, win_hi-win_lo]`` by adaptive Arb ball
    subdivision (every enclosure is rigorous; a refutation returns a witness
    Delta).

    ``a0`` is the bookkeeping-slot value of the recovered functional (+1 for
    the lower solve, -1 for the upper); ``alpha`` full (slot included) or
    36-component. Returns ``(ok, info)``; ``info["witness_Delta"]`` on refute,
    ``info["reason"] = "depth"`` if inconclusive at ``max_depth`` (treat as
    not certified).
    """
    pairs = tb(Lmax)
    a = [mpf(x) for x in alpha]
    if len(a) == len(pairs) + 1:
        a = a[1:]
    elif len(a) != len(pairs):
        raise ValueError("alpha length matches neither len(tb) nor len(tb)+1")

    coeffs = build_pols2_action(a, pairs, c, mpf(win_lo), s)  # P in x = D-win_lo
    W = mpf(win_hi) - mpf(win_lo)
    if W <= 0:
        raise ValueError("win_hi must exceed win_lo")
    scale = max([abs(mpf(a0))] + [abs(cf) for cf in coeffs])
    delta = mpf(rel_tol) * scale

    old_prec = _flint_ctx.prec
    _flint_ctx.prec = prec
    try:
        cs = [_arb_exact(cf) for cf in coeffs]
        a0_b = _arb_exact(a0)
        two = arb(2)
        twopi = 2 * arb.pi()
        # exponential shift = win_lo - (c-1)/12, exact from dyadic mpf values
        shift = _arb_exact(mpf(win_lo) - (mpf(c) - 1) / 12)
        neg_delta = -_arb_exact(delta)

        def f_ball(x):
            P = arb(0)
            for cf in reversed(cs):
                P = P * x + cf
            return a0_b + two * (-(twopi) * (x + shift)).exp() * P

        Wf = _mpf_to_fraction(W)
        stack = [(Fraction(0), Wf, 0)]
        deepest = 0
        while stack:
            lo, hi, d = stack.pop()
            deepest = max(deepest, d)
            mid = (lo + hi) / 2
            rad = (hi - lo) / 2
            # round the ball radius UP (float() rounds to nearest, which can
            # under-cover the subinterval; one ulp up restores rigor)
            rad_up = math.nextafter(float(rad), math.inf)
            x = _arb_exact(mid) + arb(0, rad_up)
            e = f_ball(x)
            if e > neg_delta:
                continue
            if e < neg_delta:
                return False, {"reason": "violation",
                               "witness_Delta": float(mpf(win_lo)
                                                      + mpf(mid.numerator)
                                                      / mpf(mid.denominator)),
                               "depth": d, "delta": float(delta)}
            if d >= max_depth:
                return False, {"reason": "depth",
                               "witness_Delta": float(mpf(win_lo)
                                                      + mpf(mid.numerator)
                                                      / mpf(mid.denominator)),
                               "depth": d, "delta": float(delta)}
            stack.append((lo, mid, d + 1))
            stack.append((mid, hi, d + 1))
        return True, {"reason": None, "max_depth_used": deepest,
                      "delta": float(delta)}
    finally:
        _flint_ctx.prec = old_prec
