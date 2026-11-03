"""fp64 cutting-plane FIND + exact Newton polish -- no arbitrary-precision solver.

Solves the same dual SDP as :mod:`dual.backends.sdpa_gmp` (maximize
``V = objective . alpha`` subject to the normalization equality, the ``pols1``
scalar positivity rows, and per-spin continuum positivity ``P_s(x) >= 0`` on
``x >= 0``) WITHOUT any arbitrary-precision interior-point iteration. Three
stages:

1. **fp64 FIND (cutting planes).** The continuum positivity is a semi-infinite
   LP; a finite sample grid alone leaves it unbounded (the tangency constraints
   that pin the optimum live between any fixed samples), so the LP is solved by
   Kelley cutting planes: solve on the current cut set with HiGHS, locate every
   negative minimum of each spin polynomial by fp64 root-finding on ``P_s'``,
   add those ``x`` as new cuts, repeat. Rows and columns are max-|entry|
   equilibrated (the same conditioning treatment as the SDPA-GMP reduction --
   without it HiGHS fails with numerical difficulties at Lmax >~ 15).
2. **Active-set extraction.** At the LP vertex the binding structure IS the
   extremal-functional zero pattern: tight ``pols1`` rows, boundary zeros
   ``P_s(0) = 0``, and interior double zeros ``P_s(x*) = P_s'(x*) = 0``.
3. **Exact Newton polish.** The KKT system of the true (semi-infinite) SDP on
   that active set -- stationarity ``obj = mu*norm + sum c_j r_j + sum w_i
   grad P``, the normalization, the tight rows, and the zero/tangency
   conditions -- is square for ANY active set. Newton at the ambient
   ``mp.dps`` (200) converges quadratically from the fp64 seed: the digits of
   the bound come from a handful of small exact linear solves, not from the LP.

The recovered ``alpha`` is a *candidate* like every FIND result: rigor comes
from :func:`dual.verify.certify_functional` (invoked by
:func:`dual.bounds.solve_bound_fp64`, which also feeds refutation witnesses
back into the active set -- a wrong or incomplete zero pattern is repaired, a
spurious zero is dropped when its multiplier converges negative).

float64 is allowed here ONLY as a search heuristic; every number that reaches
the returned bound is recomputed in ``mpmath`` from the exact assembled SDP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from mpmath import mp, mpf
from scipy.optimize import linprog

from ..assembly import safe_coeff_list
from ..characters import DampedRational


@dataclass
class SDPResult:
    """Mirrors :class:`dual.backends.sdpa_gmp.SDPResult`."""

    status: str
    alpha: object
    bound: object
    feasible: bool
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# problem views
# ---------------------------------------------------------------------------

def _exact_view(sdp):
    """Split the assembled SDP into exact mpmath pieces.

    Returns ``(obj, norm, rows, sectors)``: objective / normalization vectors,
    the ``pols1`` scalar rows, and per-``pols2``-sector component coefficient
    lists (ascending, exact ``mpf``) in pvm order.
    """
    obj = [mpf(o) for o in sdp.objective]
    norm = [mpf(v) for v in sdp.normalization]
    rows, sectors = [], []
    for pvm in sdp.pvms:
        vec = pvm.matrix[0][0]
        if isinstance(pvm.prefactor, DampedRational):
            sectors.append([safe_coeff_list(comp) for comp in vec])
        else:
            rows.append([mpf(v) for v in vec])
    return obj, norm, rows, sectors


def _fp64_view(obj, norm, rows, sectors):
    """fp64 casts: objective, normalization, pols1 row matrix, per-sector
    ``(n, deg+1)`` coefficient matrices."""
    obj_f = np.array([float(o) for o in obj])
    norm_f = np.array([float(v) for v in norm])
    rows_f = (np.array([[float(v) for v in r] for r in rows])
              if rows else np.zeros((0, len(obj))))
    mats = []
    for comps in sectors:
        deg = max(len(c) for c in comps)
        m = np.zeros((len(obj), deg))
        for k, comp in enumerate(comps):
            for i, ci in enumerate(comp):
                m[k, i] = float(ci)
        mats.append(m)
    return obj_f, norm_f, rows_f, mats


# ---------------------------------------------------------------------------
# stage 1: fp64 cutting-plane LP
# ---------------------------------------------------------------------------

_BASE_GRID = np.concatenate([np.linspace(0.0, 3.0, 61),
                             [4., 5., 6., 8., 12., 20., 50., 200.]])


def _critical_points(c_asc):
    """Positive real roots of ``P'`` for ascending fp64 coeffs of ``P``."""
    dc = c_asc[1:] * np.arange(1, len(c_asc))
    dc = np.trim_zeros(dc, "b")
    if len(dc) <= 1:
        return []
    r = np.roots(dc[::-1])
    return [float(z.real) for z in r if abs(z.imag) < 1e-7 and z.real > 0]


def _rel_min_value(c_asc, x):
    """``P(x)`` and a positive magnitude reference ``sum |c_i| max(x,1)^i``."""
    val = np.polyval(c_asc[::-1], x)
    mag = np.polyval(np.abs(c_asc)[::-1], max(x, 1.0))
    return val, (mag if mag > 0 else 1.0)


def _lp_once(obj_f, norm_f, rows_f, mats, cuts, box=1e12, margin=0.0):
    """One equilibrated HiGHS solve on the current cut set.

    Returns ``(alpha, status_str)``; ``alpha`` is ``None`` on failure.

    ``margin``: continuum CUT rows enforce ``P(x) >= margin * max|row entry|``
    (``b_ub = -margin`` after row equilibration normalizes each cut row to
    max-|entry| 1) instead of ``P >= 0``. The ``pols1`` rows and the
    normalization equality are unchanged. A strictly positive margin makes the
    returned vertex STRICTLY feasible with headroom over both the HiGHS
    feasibility tolerance (~1e-9 on equilibrated rows) and the exact
    certificate's ``1e-9`` band -- the price is a bound weakened by
    ``O(margin)`` (a certified suboptimal functional is still a VALID bound).
    """
    n = len(obj_f)
    A = [-r for r in rows_f]
    for si, mat in enumerate(mats):
        deg = mat.shape[1]
        for x in cuts[si]:
            A.append(-(mat @ (x ** np.arange(deg))))
    A = np.asarray(A)
    # Row equilibration: each >= 0 row is scale-invariant.
    rn = np.abs(A).max(axis=1)
    rn[rn == 0] = 1.0
    A = A / rn[:, None]
    # Column equilibration: solve in beta = col * alpha.
    col = np.abs(A).max(axis=0)
    col[col == 0] = 1.0
    A = A / col[None, :]
    b_ub = np.zeros(len(A))
    if margin:
        b_ub[len(rows_f):] = -margin
    obj = -(obj_f / col)
    A_eq = (norm_f / col)[None, :]
    bnds = [(-box, box)] * n
    # HiGHS status-4 ("numerical difficulties") fallback ladder. The default
    # auto-HiGHS (dual simplex + presolve) chokes on the ill-conditioned
    # cut-loaded LP at Lmax >~ 23 near c=1: once dozens of near-coincident
    # tangent cuts accumulate across refine rounds the constraint rows go
    # near-linearly-dependent (cond ~1e14) and the presolved basis pivots fail.
    # The interior-point method never forms that basis, so it solves the
    # IDENTICAL system where simplex cannot (measured on the c=1.03125 gap
    # 0.49721 Lmax=27 point: ipm rescues 7/8 of the failing LPs, presolve-off
    # 5/8). Certificate-safe and non-regressive: the LP is only a search
    # heuristic, and the FIRST attempt is the unchanged default, so every point
    # the default already solves returns a byte-identical vertex -- the
    # gold-test reproduction contract is untouched; the ladder fires only where
    # the old code returned no vertex at all.
    res = None
    for method, opts in (("highs", None),
                         ("highs-ipm", {"presolve": False}),
                         ("highs", {"presolve": False}),
                         ("highs-ds", {"presolve": False})):
        res = linprog(obj, A_ub=A, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                      bounds=bnds, method=method, options=opts)
        if res.success and res.x is not None:
            return res.x / col, "optimal"
    return None, f"lp_{res.status}"


def _cutting_plane(obj_f, norm_f, rows_f, mats, *, max_rounds=60,
                   extra_cuts=None, margin=0.0, verbose=False):
    """Kelley loop: LP -> separation oracle on P' roots -> new cuts.

    Returns ``(alpha, status, cuts)`` with ``cuts`` the final per-sector cut
    sets (returned so an exact-oracle repair or a continuation warm start can
    re-seed the next solve instead of re-discovering the tangency structure).
    On an LP conditioning failure the cut set is pruned to the near-binding
    cuts of the last good vertex and re-tried once per round; if HiGHS still
    fails, the last good ``alpha`` is returned with status ``"lp_stalled"``
    (the polish stage may still recover full digits from a coarse-but-correct
    pattern; the certificate decides). ``margin`` is forwarded to
    :func:`_lp_once` (cut rows enforce ``P >= margin`` relative).
    """
    cuts = [set(_BASE_GRID.tolist()) for _ in mats]
    if extra_cuts:
        for si, x in extra_cuts:
            cuts[si].add(float(x))
    alpha_good, V_prev, stall = None, None, 0
    for it in range(max_rounds):
        alpha, status = _lp_once(obj_f, norm_f, rows_f, mats, cuts,
                                 margin=margin)
        if alpha is None:
            if alpha_good is None:
                return None, status, cuts
            # prune to near-binding cuts of the last good vertex + coarse grid
            for si, mat in enumerate(mats):
                keep = set(np.linspace(0.0, 3.0, 16).tolist())
                for x in cuts[si]:
                    val, mag = _rel_min_value(alpha_good @ mat, x)
                    if val < 1e-4 * mag:
                        keep.add(x)
                cuts[si] = keep
            alpha, status = _lp_once(obj_f, norm_f, rows_f, mats, cuts,
                                     margin=margin)
            if alpha is None:
                return alpha_good, "lp_stalled", cuts
        alpha_good = alpha
        V = float(obj_f @ alpha)
        new = 0
        for si, mat in enumerate(mats):
            c = alpha @ mat
            for x in _critical_points(c):
                val, mag = _rel_min_value(c, x)
                if val < -1e-14 * mag and round(x, 14) not in cuts[si]:
                    cuts[si].add(round(x, 14))
                    new += 1
        if verbose:
            print(f"  [cut] round {it}: V={V:.15g} new={new}")
        if new == 0 and V_prev is not None and abs(V - V_prev) <= 1e-12 * abs(V):
            break
        # Stagnation break: near the vertex the oracle keeps emitting ever-finer
        # cuts around the tangency points without moving V. The LP is only the
        # polish seed, so a plateau at fp64 accuracy is convergence.
        if V_prev is not None and abs(V - V_prev) <= 1e-9 * abs(V):
            stall += 1
            if stall >= 5:
                break
        else:
            stall = 0
        V_prev = V
    return alpha_good, "optimal", cuts


# ---------------------------------------------------------------------------
# stage 1b: exact separation oracle (the pass fp64 is blind to)
# ---------------------------------------------------------------------------

def _exact_action(alpha, comps):
    """Exact ascending coeff list of ``P(x) = sum_k alpha[k] comps[k](x)``.

    ``alpha`` entries are promoted with ``mpf`` (exact for the fp64 vertex);
    each coefficient is an ``mp.fsum`` at the ambient ``mp.dps``.
    """
    deg = max(len(c) for c in comps)
    return [mp.fsum(mpf(alpha[k]) * comps[k][i]
                    for k in range(len(alpha)) if i < len(comps[k]))
            for i in range(deg)]


def _exact_oracle(alpha, sectors, *, threshold_rel=None):
    """Exact-arithmetic separation pass over every sector.

    The measured fp64 wall: the EXACT action of an fp64
    vertex has negative dips at ~1e-8 of the action-coefficient scale that
    fp64 evaluation cannot see (the ``alpha . B`` sum cancels ~1e-12 of its
    term magnitudes), so the fp64 oracle in :func:`_cutting_plane` terminates
    blind to them and the certificate refutes. This pass closes the gap in
    exact arithmetic: per sector, form the exact action polynomial (200-digit
    ``mpf`` -> exact ``fmpq``), isolate ALL positive critical points with
    certified Arb enclosures (the same flint machinery the certificate uses),
    evaluate the exact action there, and report every dip below
    ``-threshold_rel * max|action coeff|`` as a cut location. Together with
    ``P(0)`` (checked here) and a positive leading coefficient (the LP
    objective direction), clearing every critical point clears the whole
    half-line -- the pass is complete, not a heuristic.

    ``threshold_rel`` defaults to ``1e-10`` -- one decade inside the exact
    certificate's ``1e-9`` band, so a vertex that clears this oracle passes
    the certificate with headroom. Returns ``[(sector_index, x_float)]``.
    """
    from ..verify import _positive_real_roots, _to_fmpq_poly

    if threshold_rel is None:
        threshold_rel = mpf(10) ** -10
    hits = []
    for si, comps in enumerate(sectors):
        action = _exact_action(alpha, comps)
        scale = max(abs(ci) for ci in action)
        if scale == 0:
            continue
        floor = -threshold_rel * scale
        if action[0] < floor:
            hits.append((si, 0.0))
        p = _to_fmpq_poly(action)
        for x in _positive_real_roots(p.derivative()):
            if _poly_eval(action, mpf(x)) < floor:
                hits.append((si, float(x)))
    return hits


def _oracle_repair(alpha_f, sectors, cuts, obj_f, norm_f, rows_f, mats, *,
                   margin, max_passes=4, max_rounds=60, verbose=False):
    """Exact-oracle repair loop for the LP-tier path.

    Alternates :func:`_exact_oracle` on the current vertex with a
    cut-seeded re-run of :func:`_cutting_plane` until the exact oracle is
    clean (no dip below its threshold) or ``max_passes`` is hit. Each re-run
    is seeded with the FULL previous cut set, so it converges in a few
    rounds rather than re-discovering the tangency structure.

    Requires ``margin > 0`` to converge: a repeat offender at an existing cut
    location means HiGHS is enforcing that cut only to its own feasibility
    tolerance, which no amount of re-cutting fixes -- the margin is what
    lifts the enforced value clear of the certificate band. With
    ``margin == 0`` a repeat offender terminates the loop (recorded via the
    returned pass count; the certificate still gates downstream).

    Returns ``(alpha_f, cuts, n_new_cuts, clean)``.
    """
    n_new = 0
    clean = False
    for _pass in range(max_passes):
        hits = _exact_oracle(alpha_f, sectors)
        if not hits:
            clean = True
            break
        new = [(si, x) for si, x in hits if round(x, 14) not in cuts[si]]
        if verbose:
            print(f"  [exact-oracle] pass {_pass}: {len(hits)} dip(s), "
                  f"{len(new)} new cut(s)")
        if not new:
            break
        n_new += len(new)
        flat = [(si, x) for si, cs in enumerate(cuts) for x in cs]
        alpha_2, _status, cuts = _cutting_plane(
            obj_f, norm_f, rows_f, mats, max_rounds=max_rounds,
            extra_cuts=flat + new, margin=margin, verbose=verbose)
        if alpha_2 is None:
            break
        alpha_f = alpha_2
    return alpha_f, cuts, n_new, clean


# ---------------------------------------------------------------------------
# stage 2: active-set extraction
# ---------------------------------------------------------------------------

def _extract_pattern(alpha, rows_f, mats, *, tol_row=1e-6, tol_zero=3e-5):
    """Read the extremal zero pattern off the fp64 vertex.

    Returns ``(tight_rows, zeros)`` with ``zeros`` a list of
    ``(sector_index, x_float, interior_bool)``; boundary zeros carry ``x = 0``.

    Deliberately CONSERVATIVE (absolute ``tol_zero`` on the relative depth):
    at LP resolution the exponentially suppressed tail sectors show genuine
    zeros and pure noise minima at indistinguishable depths, so no fp64
    threshold can complete the pattern reliably. Missing zeros are added by
    the certificate-witness loop (exact refutation locations via
    :func:`witness_zero_hints`); extra zeros are dropped by the negative-
    multiplier repair in :func:`_polish`. Both repairs use exact information.
    """
    tight = []
    for j, r in enumerate(rows_f):
        if (r @ alpha) < tol_row * np.abs(r).max():
            tight.append(j)
    zeros = []
    for si, mat in enumerate(mats):
        c = alpha @ mat
        val0, mag0 = _rel_min_value(c, 0.0)
        depth0 = abs(val0) / mag0
        sector_zeros = []                    # (depth, x, interior)
        if depth0 < tol_zero:
            sector_zeros.append((depth0, 0.0, False))
        for x in _critical_points(c):
            if x < 1e-8:
                continue
            val, mag = _rel_min_value(c, x)
            if abs(val) < tol_zero * mag:
                if all(abs(x - zx) > 1e-3 * max(1.0, x)
                       for _d, zx, _i in sector_zeros):
                    sector_zeros.append((abs(val) / mag, x, True))
        # A boundary zero and a critical point hugging x = 0 are the same
        # zero seen twice at LP resolution -- a near-dependent pair that makes
        # the Newton system oscillate. Keep ONE: the boundary reading when it
        # is within a factor 100 of the deepest (a boundary zero costs no x
        # unknown and no tangency row), else the deepest reading. A wrong
        # choice is repaired by the certificate-witness loop.
        near = [zc for zc in sector_zeros if zc[1] < 3e-2]
        if len(near) > 1:
            deepest = min(near)
            bnd = [zc for zc in near if not zc[2]]
            keep = (bnd[0] if bnd and bnd[0][0] <= 100.0 * deepest[0]
                    else deepest)
            sector_zeros = [zc for zc in sector_zeros
                            if zc[1] >= 3e-2 or zc == keep]
        zeros.extend((si, x, inter) for _, x, inter in sector_zeros)
    return tight, zeros


# ---------------------------------------------------------------------------
# stage 3: exact Newton polish of the KKT system
# ---------------------------------------------------------------------------

def _poly_eval(coeffs, x):
    acc = mpf(0)
    for ci in reversed(coeffs):
        acc = acc * x + ci
    return acc


def _poly_der(coeffs):
    return [i * coeffs[i] for i in range(1, len(coeffs))]


def _polish(obj, norm, rows, sectors, tight, zeros, alpha_seed, *,
            max_iter=40, max_repairs=4, accept_res=None, verbose=False):
    """Newton-solve the KKT system on the active set at ambient ``mp.dps``.

    Unknowns ``[alpha (n), mu, c_j (tight), w_i (zeros), x_i (interior)]``;
    equations: stationarity (n), normalization, tight rows, ``P(x_i) = 0``
    per zero, ``P'(x_i) = 0`` per interior zero -- square for any active set.
    A zero whose multiplier ``w_i`` converges negative is spurious (dropped,
    re-solved); merging/degenerate zeros raise and are dropped likewise.
    A tight ROW whose multiplier ``c_j`` converges negative is dropped the
    same way: Newton with a wrongly-tight row converges to a stationary
    point of the WRONG face -- a feasible (hence certifiable!) functional
    that is not the optimum, which the caller cannot distinguish from the
    real thing (both KKT sufficiency and the certificate need ALL
    multipliers ``>= 0``). Matters for any solve whose seed pattern is
    displaced from the true active set: a margin-shifted LP vertex, or a
    continuation warm start whose active set drifted along the path.

    ``accept_res`` (default ``1e-10``): a Newton STALL at scaled KKT residual
    below this is accepted as converged. Rationale: the extremal active set
    extends over every spin sector with exponentially suppressed weights
    (``exp(-2 pi l)``), so a pattern truncated at the resolvable sectors can
    never reach an exact root -- it stalls at the first omitted sector's
    weight. A 1e-10 stall leaves positivity dips well inside the exact
    certificate's 1e-9 tolerance band and bound errors far below the gold
    bars; the certificate remains the sole rigor gate.

    Returns ``(alpha, tight, zeros, ok)`` (the possibly reduced active set).
    """
    if accept_res is None:
        accept_res = mpf(10) ** -10
    n = len(obj)
    d1 = {si: [_poly_der(c) for c in comps] for si, comps in enumerate(sectors)}
    d2 = {si: [_poly_der(c) for c in d1[si]] for si in d1}

    zeros = list(zeros)
    for _repair in range(max_repairs + 1):
        t, Z = len(tight), len(zeros)
        interior = [i for i, (_, _, inter) in enumerate(zeros) if inter]
        M = len(interior)
        x_slot = {zi: n + 1 + t + Z + m for m, zi in enumerate(interior)}
        nz = n + 1 + t + Z + M

        # seed
        z = [mpf(a) for a in alpha_seed] + [mpf(0)] * (1 + t + Z + M)
        for m, zi in enumerate(interior):
            z[n + 1 + t + Z + m] = mpf(zeros[zi][1])
        # (mu, c, w) seed from fp64 least squares of the stationarity
        # obj = mu*norm - sum_j c_j r_j - sum_i w_i B(x_i)   (c, w >= 0):
        # maximizing obj.alpha over {g >= 0} puts obj in the cone spanned by
        # MINUS the active gradients plus the normalization's normal space.
        cols = [[float(v) for v in norm]]
        cols += [[-float(v) for v in rows[j]] for j in tight]
        for (si, x0, _) in zeros:
            cols.append([-float(_poly_eval(c, mpf(x0))) for c in sectors[si]])
        # Column-normalized least squares: the raw columns span many orders of
        # magnitude, and an unscaled fp64 lstsq returns near-zero multipliers
        # (a conditioning artifact) -- which empties the w-coupling terms of
        # the first Newton Jacobian and can strand the iteration outside the
        # basin on zero-rich patterns (the interval problems).
        A_seed = np.array(cols).T
        cs = np.linalg.norm(A_seed, axis=0)
        cs[cs == 0] = 1.0
        y, *_ = np.linalg.lstsq(A_seed / cs,
                                np.array([float(o) for o in obj]), rcond=None)
        y = y / cs
        for i, yi in enumerate(y):
            z[n + i] = mpf(float(yi))

        def residual_jacobian(z, need_j=True):
            """Residual ``F``, Jacobian ``J``, and per-row term-magnitude
            scales ``S`` (the natural round-off scale of each equation).

            ``need_j=False`` skips every Jacobian entry (returns ``J=None``):
            the backtracking line search only compares scaled residuals, and
            building the dense ``nz x nz`` mpmath ``J`` there costs more than
            the residual itself.
            """
            F = [mpf(0)] * nz
            S = [mpf(1)] * nz
            J = mp.zeros(nz, nz) if need_j else None
            alpha = z[:n]
            mu = z[n]
            # E1 stationarity: obj - mu*norm + sum c_j r_j + sum w_i B(x_i) = 0
            # (i.e. obj = mu*norm - sum c r - sum w B with c, w >= 0 -- the
            # maximization KKT cone; see the seed comment above).
            for k in range(n):
                F[k] = obj[k] - mu * norm[k]
                S[k] += abs(obj[k]) + abs(mu * norm[k])
                if need_j:
                    J[k, n] = -norm[k]
            for j_idx, j in enumerate(tight):
                cj = z[n + 1 + j_idx]
                for k in range(n):
                    F[k] += cj * rows[j][k]
                    S[k] += abs(cj * rows[j][k])
                    if need_j:
                        J[k, n + 1 + j_idx] = rows[j][k]
            for zi, (si, x0, inter) in enumerate(zeros):
                xi = z[x_slot[zi]] if inter else mpf(x0)
                wi = z[n + 1 + t + zi]
                bv = [_poly_eval(c, xi) for c in sectors[si]]
                for k in range(n):
                    F[k] += wi * bv[k]
                    S[k] += abs(wi * bv[k])
                    if need_j:
                        J[k, n + 1 + t + zi] = bv[k]
                if inter and need_j:
                    dbv = [_poly_eval(c, xi) for c in d1[si]]
                    for k in range(n):
                        J[k, x_slot[zi]] = wi * dbv[k]
            # E2 normalization
            r_norm = n
            F[r_norm] = mp.fsum(norm[k] * alpha[k] for k in range(n)) - 1
            if need_j:
                for k in range(n):
                    J[r_norm, k] = norm[k]
            # E3 tight rows
            for j_idx, j in enumerate(tight):
                r = n + 1 + j_idx
                F[r] = mp.fsum(rows[j][k] * alpha[k] for k in range(n))
                S[r] += mp.fsum(abs(rows[j][k] * alpha[k]) for k in range(n))
                if need_j:
                    for k in range(n):
                        J[r, k] = rows[j][k]
            # E4 zeros / E5 tangency
            for zi, (si, x0, inter) in enumerate(zeros):
                xi = z[x_slot[zi]] if inter else mpf(x0)
                bv = [_poly_eval(c, xi) for c in sectors[si]]
                r = n + 1 + t + zi
                F[r] = mp.fsum(alpha[k] * bv[k] for k in range(n))
                S[r] += mp.fsum(abs(alpha[k] * bv[k]) for k in range(n))
                if need_j:
                    for k in range(n):
                        J[r, k] = bv[k]
                if inter:
                    dbv = [_poly_eval(c, xi) for c in d1[si]]
                    dP = mp.fsum(alpha[k] * dbv[k] for k in range(n))
                    r2 = x_slot[zi]
                    F[r2] = dP
                    S[r2] += mp.fsum(abs(alpha[k] * dbv[k]) for k in range(n))
                    if need_j:
                        J[r, x_slot[zi]] = dP
                        for k in range(n):
                            J[r2, k] = dbv[k]
                        ddbv = [_poly_eval(c, xi) for c in d2[si]]
                        J[r2, x_slot[zi]] = mp.fsum(
                            alpha[k] * ddbv[k] for k in range(n))
            return F, J, S

        def res_norm(F, S):
            return max(abs(F[i]) / S[i] for i in range(nz))

        ok = False
        stalled = False
        r0 = None
        try:
            for it in range(max_iter):
                F, J, S = residual_jacobian(z)
                r0 = res_norm(F, S)
                if r0 < mpf(10) ** (-(mp.dps - 30)):
                    ok = True
                    break
                step = mp.lu_solve(J, mp.matrix([-f for f in F]))
                # Backtracking: accept the largest halved step that reduces
                # the scaled residual (an undamped Newton oscillates when the
                # fp64 pattern seeds two nearly-dependent zeros).
                lam = mpf(1)
                accepted = False
                for _bt in range(30):
                    z_try = [z[i] + lam * step[i] for i in range(nz)]
                    F_try, _, S_try = residual_jacobian(z_try, need_j=False)
                    if res_norm(F_try, S_try) < r0:
                        z = z_try
                        accepted = True
                        break
                    lam /= 2
                if verbose:
                    print(f"  [newton] it {it}: res {mp.nstr(r0, 3)} "
                          f"lam {mp.nstr(lam, 3)}")
                if not accepted:
                    stalled = True
                    break
            else:
                stalled = True
        except (ZeroDivisionError, ValueError, TypeError):
            # ZeroDivisionError: mpmath LU on a singular system; TypeError: the
            # same condition via mpmath's unset pivot (p[j] is None).
            ok = False
            stalled = False

        def _solved_zeros():
            """The pattern with each interior x replaced by its SOLVED value
            (a continuation warm start then seeds from the exact tangency
            locations rather than the fp64 extraction readings)."""
            return [(si, float(z[x_slot[zi]]) if inter else x0, inter)
                    for zi, (si, x0, inter) in enumerate(zeros)]

        if stalled and not ok and r0 is not None and r0 < accept_res:
            # Converged-enough stall (see docstring). Skip the w-sign repair:
            # at stall level a tail zero's true multiplier can be smaller than
            # the residual and wobble negative -- dropping it would churn.
            if verbose:
                print(f"  [newton] accepted stall at res {mp.nstr(r0, 3)}")
            return z[:n], tight, _solved_zeros(), True

        if not ok:
            # A stall above accept_res still returns its alpha: the pattern is
            # incomplete, and the exact certificate refutes this candidate AT
            # the missing zeros -- the witness feedback that completes the
            # active set (see solve_bound_fp64). No alpha on a hard failure.
            return (z[:n] if stalled and r0 is not None else None), tight, zeros, False

        # multiplier sign check: a negative w_i marks a spurious zero, a
        # negative c_j a wrongly-tight row (see docstring); drop both kinds
        # and re-solve on the reduced active set.
        neg = -mpf(10) ** (-mp.dps // 2)
        w = [z[n + 1 + t + zi] for zi in range(Z)]
        bad = [zi for zi in range(Z) if w[zi] < neg]
        cvals = [z[n + 1 + j_idx] for j_idx in range(t)]
        bad_rows = [j_idx for j_idx in range(t) if cvals[j_idx] < neg]
        if not bad and not bad_rows:
            return z[:n], tight, _solved_zeros(), True
        zeros = [zc for zi, zc in enumerate(zeros) if zi not in set(bad)]
        tight = [j for j_idx, j in enumerate(tight)
                 if j_idx not in set(bad_rows)]
        alpha_seed = z[:n]
        if verbose:
            print(f"  [newton] dropped {len(bad)} spurious zero(s), "
                  f"{len(bad_rows)} wrongly-tight row(s), re-solving")
    return None, tight, zeros, False


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def solve_sdp(sdp, *, sense="max", extra_cuts=None, extra_zeros=None,
              max_rounds=60, margin=0.0, exact_oracle=True,
              max_exact_passes=12, warm_start=None, verbose=False):
    """fp64 cutting-plane FIND + exact Newton polish for an assembled SDP.

    ``sense`` follows :func:`dual.backends.sdpa_gmp.solve_sdp`: the search
    always MAXIMIZES ``V = objective . alpha`` (the bounded direction);
    ``"max"`` returns ``bound = -V`` (upper-bound drivers), ``"min"`` returns
    ``bound = +V``. ``extra_cuts`` is an optional ``[(sector_index, x)]`` of
    positivity cuts added to the LP (raw certificate witnesses; inequality
    cuts are always safe). ``extra_zeros`` is an optional ``[(sector_index,
    x)]`` of KKT active-set zero locations added to the polish pattern --
    these must come from exact-arithmetic analysis of a SOUND functional
    (:func:`witness_zero_hints` on an LP or accepted-polish alpha), never
    from a warped Newton stall: hard-pinning ``P = P' = 0`` at a wrong
    location damages the functional, unlike a cut.

    ``margin`` (cut rows enforce ``P >= margin`` relative, see
    :func:`_lp_once`) and the exact-oracle repair (``exact_oracle``,
    ``max_exact_passes``; see :func:`_exact_oracle` / :func:`_oracle_repair`)
    only matter on the LP tier: the polished tier's digits come from the
    exact KKT solve, whose optimum does not depend on the LP's margin, and a
    polished functional has no fp64-invisible dips by construction (its zeros
    are exact double zeros), so the oracle pass is skipped there to keep the
    fast path fast.

    Two accuracy tiers share one status ``"optimal"`` path:

    - **polished**: the KKT Newton converged (or stalled below its
      ``accept_res``) -- the bound carries working-precision digits
      (``info["tier"] = "polished"``).
    - **lp**: the polish could not converge (zero-rich extremal patterns
      whose exponentially suppressed tail sectors defeat fp64 pattern
      extraction -- the interval problems). The LP vertex itself is
      returned, after the exact-oracle repair loop above: with a positive
      ``margin`` its exact action clears the certificate band everywhere
      the oracle checked, so the certificate passes and the LP tier becomes
      CERTIFIABLE -- at a bound weakened by ``O(margin)``
      (``info["tier"] = "lp"``).

    Rigor in both tiers comes from the caller's exact certificate, never from
    the solver. Returns an :class:`SDPResult` with ``alpha`` as ``mpf`` and
    ``bound`` recomputed with ``mp.fsum`` against the exact objective.
    ``info`` carries the final per-sector cut sets (``"cuts"``) and the
    active-set pattern (``"pattern"``) for continuation warm starts.

    ``warm_start`` (continuation across NEARBY problems): a dict with keys
    ``"alpha"``/``"tight"``/``"zeros"``/``"cuts"``/``"tier"`` -- the
    ``info``/``alpha`` of a previous solve of a nearby SDP (same shape:
    zeros move continuously, the active set is usually stable along the
    path). A ``"polished"`` warm start goes STRAIGHT to the Newton polish
    seeded with the previous functional and pattern, skipping the LP
    entirely; the multiplier sign repair in :func:`_polish` guards against
    active-set drift (a wrongly-tight row or stale zero converges to a
    negative multiplier and is dropped -- without that check the polish can
    land on a feasible-but-suboptimal face, which would CERTIFY at the wrong
    bound). If the warm polish fails, or the warm start is LP-tier, the LP
    runs seeded with the previous cut set, which skips most of the Kelley
    cut discovery.
    """
    obj, norm, rows, sectors = _exact_view(sdp)
    obj_f, norm_f, rows_f, mats = _fp64_view(obj, norm, rows, sectors)

    if warm_start:
        w_alpha = warm_start.get("alpha")
        w_tight = warm_start.get("tight") or []
        w_zeros = warm_start.get("zeros") or []
        w_cuts = warm_start.get("cuts")
        if (warm_start.get("tier") == "polished" and w_alpha is not None
                and len(w_alpha) == len(obj)
                and all(j < len(rows) for j in w_tight)
                and all(si < len(sectors) for si, _x, _i in w_zeros)):
            alpha, tight_w, zeros_w, ok = _polish(
                obj, norm, rows, sectors, list(w_tight),
                [tuple(zc) for zc in w_zeros], [mpf(a) for a in w_alpha],
                verbose=verbose)
            if ok:
                obj_dot = mp.fsum(obj[k] * alpha[k] for k in range(len(obj)))
                return SDPResult(
                    status="optimal", alpha=list(alpha),
                    bound=-obj_dot if sense == "max" else obj_dot,
                    feasible=True,
                    info={"lp_status": "warm", "tier": "polished",
                          "n_tight": len(tight_w), "n_zeros": len(zeros_w),
                          "n_oracle_cuts": 0, "cuts": w_cuts,
                          "pattern": {"tight": list(tight_w),
                                      "zeros": list(zeros_w)}})
            if verbose:
                print("  [warm] polish-first failed; falling back to LP")
        # LP fallback: the previous cut set replaces most of cut discovery.
        if w_cuts and len(w_cuts) == len(sectors):
            extra_cuts = (list(extra_cuts or [])
                          + [(si, x) for si, xs in enumerate(w_cuts)
                             for x in xs])

    alpha_f, lp_status, cuts = _cutting_plane(
        obj_f, norm_f, rows_f, mats, max_rounds=max_rounds,
        extra_cuts=extra_cuts, margin=margin, verbose=verbose)
    if alpha_f is None:
        return SDPResult(status=lp_status, alpha=None, bound=None,
                         feasible=False, info={"stage": "lp"})

    tight, zeros = _extract_pattern(alpha_f, rows_f, mats)
    if extra_zeros:
        for si, x in extra_zeros:
            # Certificate-derived zero locations (exact-arithmetic pairing in
            # witness_zero_hints). Dedupe at the extractor's resolution --
            # a hint within ~1e-3 of a pattern zero IS that zero -- and
            # collapse boundary-hugging hints onto the boundary.
            if x < 3e-2:
                if not any(si == zsi and zx == 0.0 for zsi, zx, _ in zeros):
                    zeros.append((si, 0.0, False))
                continue
            if not any(si == zsi and abs(x - zx) <= 1e-3 * max(1.0, x)
                       for zsi, zx, _ in zeros):
                zeros.append((si, float(x), True))

    if warm_start and warm_start.get("tier") == "lp":
        # The SAME problem a small deformation away just failed to polish
        # (zero-rich pattern, tier "lp"); the attempt costs ~seconds of
        # doomed mpmath Newton per point, so the continuation skips it and
        # goes straight to the certified-LP flow. A sequence that walks INTO
        # polishable territory pays one extra LP-tier point before the next
        # cold/polished warm state resets this.
        ok = False
    else:
        alpha, tight, zeros, ok = _polish(obj, norm, rows, sectors, tight,
                                          zeros, alpha_f, verbose=verbose)
    n_oracle = 0
    if not ok:
        # LP fallback tier (see docstring): exact-oracle repair, then return
        # the (repaired) vertex itself. The repair needs a POSITIVE margin to
        # be actionable -- without one, HiGHS enforces the new cuts only to
        # its own feasibility tolerance and every oracle pass re-finds the
        # same band-depth dips (measured: margin=0 repair burns ~4 LP re-runs
        # per witness round for an unchanged certificate outcome).
        if exact_oracle and margin:
            alpha_f, cuts, n_oracle, _clean = _oracle_repair(
                alpha_f, sectors, cuts, obj_f, norm_f, rows_f, mats,
                margin=margin, max_passes=max_exact_passes,
                max_rounds=max_rounds, verbose=verbose)
        alpha = [mpf(a) for a in alpha_f]
        tier = "lp"
    else:
        tier = "polished"

    obj_dot = mp.fsum(obj[k] * alpha[k] for k in range(len(obj)))
    bound = -obj_dot if sense == "max" else obj_dot
    return SDPResult(status="optimal", alpha=list(alpha), bound=bound,
                     feasible=True,
                     info={"lp_status": lp_status, "tier": tier,
                           "n_tight": len(tight), "n_zeros": len(zeros),
                           "n_oracle_cuts": n_oracle,
                           "cuts": [sorted(cs) for cs in cuts],
                           "pattern": {"tight": list(tight),
                                       "zeros": list(zeros)}})


def witness_zero_hints(sdp, alpha, violations):
    """Pair certificate witnesses into missing-zero locations ``[(sector, x)]``.

    A missing or mis-pinned double zero shows up in a refuted functional as a
    negative dip whose odd-multiplicity roots come back as a witness PAIR
    (entering and leaving the negative region). Group each sector's
    witnesses, evaluate the EXACT action at consecutive midpoints -- fp64
    evaluation is blind here: the action cancels by ~1e-12 of its term
    magnitudes on these problems -- and emit one location per genuinely
    negative interval. Dips shallower than 10x the certificate band are
    skipped (band-level noise, not structure); a dip hugging ``x = 0`` is
    collapsed to a boundary-zero hint ``(sector, 0.0)``.

    ``alpha`` must be a sound functional (LP vertex or accepted polish); the
    dips of a warped Newton stall pair into wrong locations.
    """
    _obj, _norm, _rows, sectors = _exact_view(sdp)
    a = [mpf(v) for v in alpha]
    by_sector = {}
    for v in violations:
        if v.get("kind") == "pols2":
            by_sector.setdefault(int(v["spin"]), []).append(mpf(v["x_star"]))
    hints = []
    for si, ws in by_sector.items():
        comps = sectors[si]
        deg = max(len(c) for c in comps)
        action = [mp.fsum(a[k] * comps[k][i] for k in range(len(a))
                          if i < len(comps[k]))
                  for i in range(deg)]
        scale = max(abs(ci) for ci in action)
        if scale == 0:
            continue
        edges = sorted(set([mpf(0)] + [w for w in ws if w > 0]))
        for lo, hi in zip(edges, edges[1:]):
            mid = (lo + hi) / 2
            if _poly_eval(action, mid) < -mpf(10) ** -8 * scale:
                hints.append((si, 0.0 if mid < mpf(3) / 100 else float(mid)))
    return hints
