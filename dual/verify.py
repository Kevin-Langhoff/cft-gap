"""High-precision certificate that a recovered dual functional is non-negative.

The rigor layer for the scipy-LP FIND path. The LP solves the dual SDP in
float64 and recovers a candidate linear functional ``alpha``; that float64
output is *not* trusted for the final scientific claim. This module
**certifies (or refutes) continuum positivity at high precision, independent
of the interior-point solver**.

For each spin sector the functional action is a polynomial in ``x = Delta - gap``
times a *strictly positive* damped-rational prefactor
``DampedRational[2, {}, exp(-2 pi), x + gap]`` (the ``pols2`` prefactor). Since
the prefactor is ``2 * exp(-2 pi (x+gap)) > 0`` for all real ``x`` it cannot change
the sign of the action on ``x >= 0`` -- so we certify the **numerator polynomial
only** (documented in ``certify_functional``). The physical constraint the SDP
imposes is ``alpha . Fvec_{Delta,l} >= 0`` for every operator above the gap, i.e.
the numerator polynomial ``P_l(x) >= 0`` for all ``x >= 0``.

Method (exact, certified):

- Promote the (possibly float64-recovered) functional to ``mpmath`` and form the
  action polynomial exactly, then rationalise its coefficients to exact ``fmpq``
  (an mpf has an exact dyadic value, so no rounding is introduced beyond the LP's
  own; float64 is forbidden for the emitted certificate).
- ``factor_squarefree`` over Q splits the polynomial into squarefree factors with
  exact multiplicities. ``p(x) >= 0`` on ``[0, inf)`` iff:
    1. the leading coefficient is ``> 0`` (else ``-> -inf``),
    2. ``p(0) >= 0``,
    3. NO squarefree factor of ODD multiplicity has a real root in the OPEN
       interval ``(0, inf)`` (an odd-multiplicity root is a genuine sign change;
       even-multiplicity roots only touch zero).
  Each candidate factor's real roots are isolated as certified ``arb`` enclosures
  via python-flint ``complex_roots`` (Arb's rigorous ball arithmetic): a root is
  real iff its imaginary enclosure is exactly zero, and lies in ``(0, inf)`` iff
  its real enclosure is provably ``> 0``.

This module guards against a failure mode where the interior-point LP can
report a "feasible" functional whose action is in fact slightly NEGATIVE at
an unconstrained sub-gap ``x* > 0`` (a spin/Delta the sample grid missed).
The witness root list pins that ``x*`` so the refutation is explicit, not a
numeric near-miss.
"""

from __future__ import annotations

from fractions import Fraction

import flint
from flint import ctx, fmpq, fmpq_poly
import mpmath
from mpmath import mp, mpf

from .characters import DPS, com_poly_in_x


# Arb ball precision for root isolation. The certificate is exact (factor over Q
# is exact; multiplicities are exact); ``complex_roots`` only refines the *root
# enclosures*, which we use solely for the real/positivity sign tests. Set it
# well above DPS-equivalent bits so a true sign-changing root is resolved away
# from 0 and the real/imag enclosures collapse cleanly.
_ROOT_PREC_BITS = max(4 * DPS, 1024)

# Canonical default numerical-zero band. Frozen once here (at the working
# precision set by importing ``characters``) so the producer's default and
# any caller/test compare bit-identically: ``mpf`` is a global-precision-stateful
# singleton, and re-deriving ``mpf(10) ** -9`` at a *different* ambient precision
# yields a different mantissa (~1 ulp), breaking exact ``==``. rel_tol is only
# ever used as a multiplicative threshold (``delta = rel_tol * scale``), so its
# exact low bits are immaterial; sharing one value just keeps equality stable.
DEFAULT_REL_TOL = mpf(10) ** -9


def _mpf_to_fraction(x) -> Fraction:
    """Exact ``Fraction`` of an ``mpf`` (its dyadic value ``(-1)^s man 2^exp``).

    No rounding: an ``mpf`` is a binary float with an exact rational value, so the
    emitted certificate carries the LP-recovered numbers losslessly (no float64
    leaks into the rationalised polynomial). Plain Python ints/Fractions pass
    through.
    """
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    xm = mpf(x)
    sign, man, exp, _bc = xm._mpf_
    if man == 0:
        return Fraction(0)
    val = Fraction(int(man)) * (Fraction(2) ** int(exp))
    return -val if sign else val


def _to_fmpq_poly(coeffs) -> fmpq_poly:
    """Ascending mpmath/rational coeff list -> exact ``fmpq_poly``."""
    return fmpq_poly([fmpq(f.numerator, f.denominator)
                      for f in (_mpf_to_fraction(c) for c in coeffs)])


def certify_nonneg_on_halfline(poly_coeffs_x, *, dps=200, rel_tol=DEFAULT_REL_TOL):
    """Certify ``p(x) >= 0`` for ALL ``x >= 0`` (exact, via Arb real-root isolation).

    ``poly_coeffs_x`` is the ASCENDING coefficient list of ``p`` (``mpmath``/int/
    ``Fraction``); index 0 is the constant term. The strictly-positive damped
    prefactor of ``pols2`` does NOT affect the sign on ``x >= 0`` and must NOT be
    included here -- pass the numerator polynomial only.

    Returns ``(is_nonneg: bool, info: dict)``. ``info`` carries:

    - ``sign_change_roots``: list of ``mpf`` locations of odd-multiplicity real
      roots in the OPEN interval ``(0, inf)`` (the refutation witnesses); empty iff
      certified non-negative there.
    - ``n_sign_changes``: ``len(sign_change_roots)``.
    - ``p0``: ``p(0)`` as ``mpf`` (must be ``>= 0``).
    - ``leading``: leading coefficient as ``mpf`` (must be ``> 0`` for a non-trivial
      degree; a non-negative constant is also accepted).
    - ``reason``: short string when refuted (``'leading_coeff'`` / ``'p(0)'`` /
      ``'sign_change'``), else ``None``.

    Decision rule: ``p >= 0`` on ``[0, inf)`` iff the leading
    coefficient is ``> 0`` (or ``p`` is a non-negative constant), ``p(0) >= 0``,
    and ``p`` has no odd-multiplicity real root in ``(0, inf)``.

    Numerical-zero band (``rel_tol``). The recovered functional is solver-limited
    (the SDPA-GMP backend ingests float64 matrix data), so an EXTREMAL
    functional -- which saturates a few constraints to *exactly zero* at the
    optimum -- has its boundary zeros displaced by ~1e-16 relative, roughly half
    landing slightly negative. A zero-tolerance test would then refute a correct
    bound. We therefore certify the *tolerance-banded* inequality
    ``p(x) >= -delta`` with ``delta = rel_tol * max_k|coeff_k|`` -- an EXACT
    certificate of the relaxed statement: the decision uses the ``delta``-shifted
    polynomial ``p + delta``, while the reported witnesses are the exact roots of
    the ORIGINAL ``p``. This stays rigorous -- a genuine relative-``>= rel_tol``
    sign change (an O(1) relative dip) is STILL refuted; only sub-``rel_tol``
    boundary displacements are tolerated. ``rel_tol=0`` recovers the exact
    zero-tolerance certificate. ``info`` additionally carries ``rel_tol``,
    ``scale``, ``delta``.
    """
    rel_tol = mpf(rel_tol)
    with mp.workdps(dps):
        p = _to_fmpq_poly(poly_coeffs_x)
        deg = p.degree()

        # Zero / empty polynomial: identically 0 -> non-negative everywhere.
        if deg < 0:
            return True, {"sign_change_roots": [], "n_sign_changes": 0,
                          "p0": mpf(0), "leading": mpf(0), "reason": None,
                          "rel_tol": rel_tol, "scale": mpf(0), "delta": mpf(0)}

        cs = p.coeffs()  # ascending fmpq
        p0 = mpf(int(cs[0].p)) / mpf(int(cs[0].q))
        lead = cs[-1]
        lead_mpf = mpf(int(lead.p)) / mpf(int(lead.q))
        # Numerical-zero band: delta = rel_tol * (max |coeff|). The decision
        # certifies p(x) + delta >= 0 (i.e. p(x) >= -delta); witnesses below.
        scale = max((abs(mpf(int(c.p)) / mpf(int(c.q))) for c in cs),
                    default=mpf(0))
        delta = rel_tol * scale

        info = {"sign_change_roots": [], "n_sign_changes": 0,
                "p0": p0, "leading": lead_mpf, "reason": None,
                "rel_tol": rel_tol, "scale": scale, "delta": delta}

        # Degree-0: a constant. Non-negative (up to band) iff constant >= -delta.
        if deg == 0:
            ok = p0 >= -delta
            if not ok:
                info["reason"] = "p(0)"
            return ok, info

        # Refutation witnesses: odd-multiplicity positive roots of the ORIGINAL p
        # (the exact physical sign-change / sub-gap locations). Reported on
        # any refutation so the caller always sees x*; the DECISION below uses the
        # delta-shifted polynomial so a sub-delta boundary displacement is tolerated.
        witnesses = _odd_mult_positive_roots(p)

        # Decision (banded): leading > 0, p(0) >= -delta, and p + delta has no
        # odd-multiplicity positive root (no sign change deeper than the band).
        if lead_mpf <= 0:
            info["reason"] = "leading_coeff"
            info["sign_change_roots"] = witnesses
            info["n_sign_changes"] = len(witnesses)
            return False, info
        if p0 < -delta:
            info["reason"] = "p(0)"
            info["sign_change_roots"] = witnesses
            info["n_sign_changes"] = len(witnesses)
            return False, info

        if delta > 0:
            df = _mpf_to_fraction(delta)
            p_dec = p + fmpq_poly([fmpq(df.numerator, df.denominator)])
        else:
            p_dec = p
        if _odd_mult_positive_roots(p_dec):
            info["reason"] = "sign_change"
            info["sign_change_roots"] = witnesses
            info["n_sign_changes"] = len(witnesses)
            return False, info

        return True, info


def _odd_mult_positive_roots(p: fmpq_poly):
    """Real roots of ODD multiplicity in the open ``(0, inf)``, as ``mpf`` (sorted).

    Factors ``p`` over Q into squarefree factors with exact multiplicities; only
    odd-multiplicity factors can produce a sign change. For each such factor the
    real roots are isolated as certified ``arb`` enclosures (``complex_roots`` at
    ``_ROOT_PREC_BITS``): the imaginary enclosure is *exactly* zero for a true real
    root, and the real enclosure is provably ``> 0`` for a root in ``(0, inf)``.
    Roots at exactly ``x = 0`` (boundary) and negative roots are excluded.
    """
    _const, factors = p.factor_squarefree()
    roots = []
    saved = ctx.prec
    ctx.prec = _ROOT_PREC_BITS
    try:
        for fac, mult in factors:
            if mult % 2 == 0:
                continue  # even multiplicity: touches zero, no sign change
            for r, _rmult in fac.complex_roots():
                if not r.imag.is_zero():
                    continue  # genuine complex root (conjugate pair)
                re = r.real
                if re.is_zero():
                    continue  # root at the boundary x = 0 (handled by p(0) test)
                # Strictly positive iff the certified real enclosure is > 0.
                if re > 0:
                    roots.append(_arb_to_mpf(re))
    finally:
        ctx.prec = saved
    roots.sort()
    return roots


def _arb_to_mpf(a) -> mpf:
    """Midpoint of a certified ``arb`` enclosure as an ``mpf`` (witness location).

    Uses the exact dyadic ``(man, exp)`` of the arb midpoint (``value = man*2^exp``)
    so no decimal round-trip is involved.
    """
    man, exp = a.mid().man_exp()
    return mpf(int(man)) * mpf(2) ** int(exp)


def certify_functional(alpha, sdp, *, rel_tol=DEFAULT_REL_TOL):
    """Certify a recovered functional ``alpha`` is non-negative on the spectrum.

    ``alpha`` is the length-``len(Base)`` functional (``Vec1``),
    float64/``mpmath``; it is promoted to ``mpmath`` here. ``sdp`` is the assembled
    program; this routine reads its positivity constraints:

    - ``sdp.pols2``: one continuum constraint per spin ``ii`` with effective gap
      ``g``, whose matrix entry is the polynomial vector ``Base(Delta = x + g,
      s = ii)`` in ``x >= 0``. The action polynomial is
      ``P_ii(x) = sum_k alpha[k] * Base_k(x; ii, g)``; it must be ``>= 0`` for all
      ``x >= 0``. The strictly-positive damped prefactor is omitted (it cannot
      change the sign), so only the numerator polynomial is certified.
    - ``sdp.pols1``: scalar (single-point) constraints; each action
      ``sum_k alpha[k] * v_k`` must be ``>= 0``.

    Returns ``(ok: bool, violations)``. ``violations`` is a list of dicts; for a
    continuum failure ``{'kind': 'pols2', 'spin': ii, 'x_star': mpf, ...}`` (the
    positive-action-at-an-unconstrained-sub-gap witness), for a scalar
    failure ``{'kind': 'pols1', 'index': j, 'value': mpf}``.

    ``sdp`` is duck-typed: it must expose ``pols2`` as an iterable of entries each
    providing ``(spin, gap, base_polys)`` and ``pols1`` as an iterable of scalar
    coefficient vectors. Two shapes are accepted per field (see ``_iter_pols2`` /
    ``_iter_pols1``) so this layer does not bind to a specific assembly class.
    """
    a = [mpf(ai) for ai in alpha]
    violations = []

    # --- pols2: continuum positivity, one polynomial per spin sector ---
    for spin, gap, base_polys in _iter_pols2(sdp):
        action = _combine_action(a, base_polys)
        ok, info = certify_nonneg_on_halfline(action, rel_tol=rel_tol)
        if not ok:
            for x_star in info["sign_change_roots"]:
                violations.append({"kind": "pols2", "spin": spin, "gap": mpf(gap),
                                   "x_star": x_star, "reason": info["reason"]})
            if not info["sign_change_roots"]:
                # leading-coeff / p(0) refutation (no interior witness root).
                violations.append({"kind": "pols2", "spin": spin, "gap": mpf(gap),
                                   "x_star": mpf(0), "reason": info["reason"]})

    # --- pols1: scalar positivity at sampled Delta points ---
    for j, vec in enumerate(_iter_pols1(sdp)):
        terms = [a[k] * mpf(vec[k]) for k in range(len(vec))]
        val = sum(terms, mpf(0))
        scale1 = max((abs(t) for t in terms), default=mpf(0))
        if val < -mpf(rel_tol) * scale1:
            violations.append({"kind": "pols1", "index": j, "value": val})

    return (len(violations) == 0), violations


def _combine_action(alpha, base_polys):
    """``sum_k alpha[k] * base_polys[k]`` as one ascending mpmath coeff list."""
    out = []
    for k, coeffs in enumerate(base_polys):
        ak = alpha[k]
        for i, ci in enumerate(coeffs):
            if i >= len(out):
                out.extend([mpf(0)] * (i + 1 - len(out)))
            out[i] += ak * mpf(ci)
    return out if out else [mpf(0)]


def _iter_pols2(sdp):
    """Yield ``(spin, gap, base_polys)`` for each continuum constraint.

    Accepts either ``sdp.pols2`` entries that are 3-tuples
    ``(spin, gap, base_polys)``, or objects exposing ``.spin``/``.gap``/``.base``
    (or ``.base_polys``) attributes. ``base_polys`` is a list (over the Base
    components) of ascending-coefficient polynomials in ``x``.
    """
    for entry in getattr(sdp, "pols2"):
        if isinstance(entry, (tuple, list)) and len(entry) == 3:
            yield entry[0], entry[1], entry[2]
        else:
            spin = getattr(entry, "spin")
            gap = getattr(entry, "gap")
            base = getattr(entry, "base", None)
            if base is None:
                base = getattr(entry, "base_polys")
            yield spin, gap, base


def _iter_pols1(sdp):
    """Yield each scalar coefficient vector for the ``pols1`` constraints."""
    pols1 = getattr(sdp, "pols1", None)
    if pols1 is None:
        return
    for entry in pols1:
        if isinstance(entry, (tuple, list)):
            yield entry
        else:
            yield getattr(entry, "vec", entry)


def build_pols2_action(alpha, k_kb_pairs, c, gap, s):
    """Convenience: build the pols2 action polynomial from ``(k, kb)`` pairs.

    ``P(x) = sum_j alpha[j] * Com[k_j, kb_j, c]/.{Delta -> x + gap, s -> s}``.
    Returns the ascending mpmath coeff list ready for
    ``certify_nonneg_on_halfline``. ``alpha`` is the per-component functional over
    the SAME ``(k, kb)`` ordering. Strictly positive prefactor omitted by design.
    """
    a = [mpf(ai) for ai in alpha]
    base_polys = [com_poly_in_x(k, kb, c, gap, s) for (k, kb) in k_kb_pairs]
    return _combine_action(a, base_polys)


# ---------------------------------------------------------------------------
# Certified extremal-spectrum read-off
# ---------------------------------------------------------------------------
#
# Complementary slackness: at the convex-SDP optimum the dual functional action
# P_l(x) >= 0 touches zero (EVEN-multiplicity / "double" zeros) exactly on the
# extremal spectrum. ``certify_nonneg_on_halfline`` certifies P_l >= 0 by isolating
# the ODD-multiplicity roots (sign-change violations) and SKIPS the even ones.
# The even ones are precisely the operators -- so this section adds
# the mirror primitive plus a displacement-bounded read-off. Additive: nothing here
# changes ``certify_functional`` / ``certify_nonneg_on_halfline``.


def _even_mult_positive_roots(p: fmpq_poly):
    """Real roots of EVEN multiplicity in the open ``(0, inf)`` as ``[(mpf, mult), ...]``.

    Mirror of :func:`_odd_mult_positive_roots` with the parity test INVERTED (keep
    even, skip odd). Even-multiplicity squarefree factors are the touch-and-stay
    zeros (complementary-slackness extremal operators), not sign changes. The
    multiplicity is returned so a true double (``mult == 2``) is distinguishable from
    a higher even touch (the ``(x-2)^4`` case: ``mult == 4`` is kept -- the test is
    ``mult % 2``, not ``mult == 2``). Roots at exactly ``x = 0`` (the gap boundary)
    and negative roots are excluded, exactly as in the odd-mult sibling.
    """
    _const, factors = p.factor_squarefree()
    roots = []
    saved = ctx.prec
    ctx.prec = _ROOT_PREC_BITS
    try:
        for fac, mult in factors:
            if mult % 2 == 1:
                continue  # odd multiplicity: genuine sign change, not a touch
            for r, _rmult in fac.complex_roots():
                if not r.imag.is_zero():
                    continue  # genuine complex root (conjugate pair)
                re = r.real
                if re.is_zero():
                    continue  # root at the boundary x = 0
                if re > 0:
                    roots.append((_arb_to_mpf(re), int(mult)))
    finally:
        ctx.prec = saved
    roots.sort()
    return roots


def _iv_enclose(coeffs_asc, x_lo, x_hi):
    """Rigorous ``mpmath.iv`` interval enclosure of ``sum coeffs_asc[i] x^i`` over the
    real interval ``[x_lo, x_hi]`` (outward-rounded; ``coeffs_asc`` are exact ``fmpq``).

    Horner in ``mpmath.iv`` arithmetic, which carries certified outward rounding -- so
    the returned ``iv.mpf`` provably contains every value of the polynomial on the
    interval. Used for the displacement bound's ``m+`` (value upper bound) and
    ``H_lo`` (curvature lower bound).
    """
    iv = mpmath.iv
    X = iv.mpf([x_lo, x_hi])
    acc = iv.mpf(0)
    for c in reversed(coeffs_asc):              # descending (Horner) from ascending list
        cv = iv.mpf(int(c.p)) / iv.mpf(int(c.q))
        acc = acc * X + cv
    return acc


def _poly_eval_point(coeffs_asc, x):
    """Evaluate ``sum coeffs_asc[i] x^i`` (exact ``fmpq`` coeffs) at the point ``x``
    in ``mpmath`` (Horner). Used for ``m = P(x0)`` in the displacement bound -- a point
    value, NOT an interval (the interval enclosure over a ball bloats by the Horner
    dependency problem and would inflate ``r``)."""
    x = mpf(x)
    acc = mpf(0)
    for c in reversed(coeffs_asc):
        acc = acc * x + mpf(int(c.p)) / mpf(int(c.q))
    return acc


def _curv_radius(p, x0, delta, anchors, dps):
    """Displacement radius ``r`` enclosing the touch-zero near ``x0`` + its ingredients.

    Returns ``(rho, H_lo, r, ok, m_plus)``. Picks a neighbourhood half-width ``rho``
    (half the distance to the nearest other anchor, capped), SHRINKS it until the
    curvature lower bound ``H_lo`` (a rigorous ``mpmath.iv`` enclosure of ``P''`` over
    ``[x0 +/- rho]``) is provably ``> 0``, then ``r = sqrt(2(m+ + delta)/H_lo)`` with
    ``m+`` the rigorous upper bound of ``P`` on the ball (``= 0`` up to outward rounding
    at an exact even root). ``ok`` iff ``H_lo > 0`` AND ``r <= rho`` (the local quadratic
    model is self-consistent on its own neighbourhood).
    """
    with mp.workdps(dps):
        x0 = mpf(x0)
        ppp = p.derivative().derivative()                  # exact P''
        rho = _default_rho(float(x0), anchors)
        H_lo = mpf(0)
        for _ in range(50):
            H_iv = (_iv_enclose(ppp.coeffs(), x0 - rho, x0 + rho)
                    if ppp.degree() >= 0 else mpmath.iv.mpf(0))
            H_lo = mpf(H_iv.a)                              # certified lower bound on P''
            if H_lo > 0:
                break
            rho = rho / 2
            if rho < mpf("1e-12"):
                break
        m_plus = max(mpf(0), _poly_eval_point(p.coeffs(), x0))  # P(x0) at the point
        if H_lo <= 0:
            return rho, H_lo, None, False, m_plus
        r = mp.sqrt(2 * (m_plus + mpf(delta)) / H_lo)
        return rho, H_lo, r, (r <= rho), m_plus


def certified_extremal_operators(P_coeffs, gap, *, rel_tol=DEFAULT_REL_TOL,
                                 dps=200, touch_rel=mpf(10) ** -3):
    """Certified-tier extremal-spectrum read-off from a recovered functional numerator.

    ``P_coeffs`` is the ascending-coefficient continuum-action numerator (the SAME
    object :func:`certify_nonneg_on_halfline` certifies; strictly-positive prefactor
    omitted by design). ``gap`` maps ``x -> Delta = x + gap``. Returns
    ``(operators, info)``.

    Each operator dict carries ``x``, ``Delta``, ``kind``, ``interval`` (the
    ``(Delta_lo, Delta_hi)`` enclosure), ``mult``, ``m``, ``H_lo``, ``r``. ``kind``:

    - ``'double-zero'`` -- ``x0`` is an EXACT even-multiplicity root of the recovered
      ``P`` and ``P'' > 0`` is provable; the operator is enclosed to ``+/- r`` by the
      displacement bound (``m = 0`` exactly at a true root, so ``r = sqrt(2 delta/H_lo)``).
    - ``'bracketed'`` -- ``P`` dips below the band (``reason == 'sign_change'``): the
      operator is bracketed between two straddling odd-mult roots.
    - ``'min-estimate'`` -- a positive minimiser of ``P`` near the band floor whose
      curvature lower bound is not provably positive (or ``r > rho``): a NON-rigorous
      location estimate (the perturbation-split double zero of a real solver output).

    Note: only exactly-rational input (``Fraction``/``fmpq``, or an ``mpf`` with an
    exact dyadic value -- the intended LP/SDPB path emits exact ``fmpq``) can produce a
    ``'double-zero'``. Float-built coefficients near a split double zero almost never
    factor as an exact even root and correctly fall to ``'min-estimate'``.

    RIGOR CAVEAT: every record locates where the
    RECOVERED functional saturates. The recovered != optimal gap (finite duality gap)
    is a SEPARATE error NOT closed here -- do not present ``Delta`` as a certified CFT
    operator dimension; this certifies structure + the recovered touch locations.
    """
    gap = mpf(gap)
    with mp.workdps(dps):
        p = _to_fmpq_poly(P_coeffs)
        is_nonneg, cinfo = certify_nonneg_on_halfline(P_coeffs, dps=dps, rel_tol=rel_tol)
        delta = cinfo["delta"]
        scale = cinfo["scale"]
        touch = mpf(touch_rel) * scale + delta
        even = _even_mult_positive_roots(p)
        even_locs = [x for (x, _m) in even]
        operators = []

        # --- band violated: operators are bracketed between odd-mult roots ---
        if (not is_nonneg) and cinfo["reason"] == "sign_change":
            odd = cinfo["sign_change_roots"]
            for i in range(0, len(odd) - 1, 2):
                lo, hi = odd[i], odd[i + 1]
                operators.append({
                    "x": lo, "Delta": lo + gap, "kind": "bracketed",
                    "interval": (lo + gap, hi + gap), "mult": None,
                    "m": None, "H_lo": None, "r": (hi - lo) / 2})

        crit = _positive_real_roots(p.derivative())
        anchors = sorted(set(float(z) for z in crit) | set(float(z) for z in even_locs))

        # --- exact even-mult roots -> certified double-zeros (m = 0 at a true root) ---
        for x0, mult in even:
            rho, H_lo, r, ok, m_plus = _curv_radius(p, x0, delta, anchors, dps)
            rr = r if r is not None else rho
            operators.append({
                "x": mpf(x0), "Delta": mpf(x0) + gap,
                "kind": "double-zero" if ok else "min-estimate",
                "interval": (mpf(x0) + gap - rr, mpf(x0) + gap + rr),
                "mult": mult, "m": mpf(0), "H_lo": H_lo, "r": r})

        # --- P'-minima near the floor that are NOT exact even roots -> min-estimate
        # (a real solver's perturbation-split double zero leaves a near-zero minimum
        # with no exact root). Skip maxima (H_lo<=0) and points well above the floor.
        for x0 in crit:
            if any(abs(mpf(x0) - xe) < 1e-6 for xe in even_locs):
                continue
            rho, H_lo, r, ok, m_plus = _curv_radius(p, x0, delta, anchors, dps)
            if H_lo <= 0 or m_plus > touch:
                continue
            rr = r if r is not None else rho
            operators.append({
                "x": mpf(x0), "Delta": mpf(x0) + gap, "kind": "min-estimate",
                "interval": (mpf(x0) + gap - rr, mpf(x0) + gap + rr),
                "mult": None, "m": m_plus, "H_lo": H_lo, "r": r})

        operators.sort(key=lambda o: o["Delta"])
        info = {"rel_tol": cinfo["rel_tol"], "scale": scale, "delta": delta,
                "is_nonneg": is_nonneg, "reason": cinfo["reason"],
                "even_mult_roots": even,
                "odd_mult_roots": cinfo["sign_change_roots"]}
        return operators, info


def _positive_real_roots(p: fmpq_poly):
    """All real roots of ``p`` in the open ``(0, inf)`` as ``mpf`` (any multiplicity)."""
    if p.degree() < 0:
        return []
    roots = []
    saved = ctx.prec
    ctx.prec = _ROOT_PREC_BITS
    try:
        for r, _mult in p.complex_roots():
            if not r.imag.is_zero():
                continue
            re = r.real
            if re.is_zero() or not (re > 0):
                continue
            roots.append(_arb_to_mpf(re))
    finally:
        ctx.prec = saved
    roots.sort()
    return roots


def _default_rho(x0, anchors):
    """Half the distance to the nearest distinct anchor (other root), capped to
    ``[1e-6, x0/2]`` so the curvature ball stays inside ``(0, inf)`` and away from
    neighbouring zeros."""
    others = [a for a in anchors if abs(a - x0) > 1e-12]
    if others:
        d = min(abs(a - x0) for a in others) / 2
    else:
        d = abs(x0) / 2 if x0 != 0 else mpf("0.5")
    return mpf(max(1e-6, min(d, abs(x0) / 2 if x0 != 0 else 0.5)))
