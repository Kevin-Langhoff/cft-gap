"""Verify-and-resample core for the discrete-sampling toolkit.

After a solve over discrete window samples, the recovered functional is
evaluated on a dense grid across the window; wherever it dips negative the
offending points (each with two half-step neighbours) become new sample points
and the SDP is re-solved. Grid-clean is the loop's exit criterion; it is NOT a
rigorous interval certificate (a dip narrower than the grid step can hide) — the
drivers therefore record a post-hoc high-precision certificate via
:func:`dual.postcheck.certify_window_constraint` alongside.

Pure mpmath, no solver — fully unit-testable in the fast suite.
"""
from __future__ import annotations

from mpmath import mp, mpf

from ..characters import comwithprefactor_at, tb


def window_functional_values(a0, alpha_deriv, c, spin, a, b, Lmax, ngrid, *,
                             eval_dps=30):
    """Evaluate ``f(Delta) = a0 + alpha_deriv . Comwithprefactor(Delta, spin)``
    on the (ngrid+1)-point uniform grid over ``[a, b]``.

    Returns ``[(Delta_k, f(Delta_k))]``. ``eval_dps`` mirrors the reference
    ``N[..., 30]`` sign check; values within ~1e-{eval_dps} of zero may be
    misjudged, which only ever ADDS sample points (conservative) — final rigor
    comes from the high-precision certificate, not this scan.
    """
    a = mpf(a)
    b = mpf(b)
    pairs = tb(Lmax)
    if len(alpha_deriv) != len(pairs):
        raise ValueError(f"alpha_deriv length {len(alpha_deriv)} != len(tb) "
                         f"{len(pairs)}")
    step = (b - a) / ngrid
    grid = [a + k * step for k in range(ngrid + 1)]
    vals = []
    with mp.workdps(eval_dps):
        for d in grid:
            f = mpf(a0) + mp.fsum(
                mpf(alpha_deriv[i]) * comwithprefactor_at(a1, a2, c, d, spin)
                for i, (a1, a2) in enumerate(pairs))
            vals.append((d, f))
    return vals


def negative_points(vals):
    """Grid points where the functional is strictly negative."""
    return [d for (d, v) in vals if v < 0]


def insert_triples(existing, negs, step):
    """Merge ``{p - step/2, p, p + step/2}`` per negative point into the extra
    sample set — deduped, UNclamped (out-of-window points are harmless extra
    constraints), sorted.
    """
    merged = [mpf(p) for p in existing]
    half = mpf(step) / 2
    for p in negs:
        p = mpf(p)
        merged.extend([p - half, p, p + half])
    tol = mpf(10) ** -50
    out = []
    for d in sorted(merged):
        if not out or abs(d - out[-1]) > tol:
            out.append(d)
    return out


def slice_alpha(alpha, j, K, Lmax):
    """Split a K-slot functional: ``(a0 = alpha[j], deriv = alpha[K:])``
    (``funcl = Vec1[[j]] + Drop[Vec1, lengthint] . baseexact``)."""
    npairs = len(tb(Lmax))
    if len(alpha) != K + npairs:
        raise ValueError(f"alpha length {len(alpha)} != K + len(tb) = {K + npairs}")
    return mpf(alpha[j]), [mpf(v) for v in alpha[K:]]
