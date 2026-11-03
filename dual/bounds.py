"""Bound drivers + integrality scan -- top of the dual-bootstrap stack.

Port of the driver layer to the assembled-SDP + verified-LP pipeline. Each
driver builds an :class:`~dual.assembly.SDP` (via the shared scaffolding in
:mod:`dual.assembly`), solves it with the fp64 FIND backend
:func:`dual.backends.scipy_lp.find_functional`, certifies the recovered
functional at high precision with
:func:`dual.verify.certify_functional`, and returns the certified ``Zmin``
scalar bound recomputed in ``mpmath`` from ``alpha``.

The verified-LP loop :func:`solve_bound` is the rigor wrapper: the LP only
*samples* continuum positivity, so a recovered ``alpha`` may be negative at an
unconstrained sub-gap ``x*`` the grid missed. Each certificate refutation
hands back the witness ``x*`` values, which are appended to the LP sample
grid (``extra_x``) and the solve is repeated, up to ``max_refine`` rounds.

Numerics: SDP construction, the recomputed bound, and the certificate are all
``mpmath`` at ``mp.dps >= 200``; float64 lives only inside the LP.

Legacy/conditional note: ``integer_lowerbound_spin`` / ``integer_upperbound_spin``
below are the single-operator second-gap ("spin chain") drivers built on
:func:`dual.assembly.build_bound_sdp_spin` -- the legacy conditional method
that assumes a second gap. They are kept for reference and are not used by the
certified path; prefer the unconditional interval drivers
(``integer_{lower,upper}bound_interval``) for new work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mpmath import mp, mpf

from .backends.scipy_lp import find_functional
from .characters import DampedRational
from .assembly import (
    build_bound_sdp,
    build_bound_sdp_spin,
    build_fes_sdp,
    build_interval_sdp,
)
from . import verify


# ---------------------------------------------------------------------------
# SDP -> certify_functional adapter
# ---------------------------------------------------------------------------

def _gap_from_prefactor(pref: DampedRational):
    """Recover the shift ``g`` from a ``DampedRational[2, {}, B, x+g]`` prefactor.

    ``damped_shift`` set ``const = 2 * B**g``, so ``g = log(const/2) / log(B)``.
    Used to label each ``pols2`` constraint with its effective gap when splitting
    the flat ``pvms`` list back into the two channels for the certificate.
    """
    return mp.log(pref.const / 2) / mp.log(pref.base)


class _SdpView:
    """Duck-typed carrier exposing ``pols2`` / ``pols1`` for ``certify_functional``.

    :func:`dual.verify.certify_functional` reads ``sdp.pols2`` as
    ``(spin, gap, base_polys)`` triples and ``sdp.pols1`` as scalar coefficient
    vectors. The assembled :class:`~dual.assembly.SDP` carries a
    flat ``pvms`` list of ``PositiveMatrixWithPrefactor``; this view splits it back
    into the two channels (``DampedRational`` prefactor -> ``pols2`` with its spin
    index and recovered gap; scalar prefactor -> ``pols1`` scalar vector).
    """

    def __init__(self, sdp):
        pols2 = []
        pols1 = []
        spin = 0
        for pvm in sdp.pvms:
            vec = pvm.matrix[0][0]
            if isinstance(pvm.prefactor, DampedRational):
                gap = _gap_from_prefactor(pvm.prefactor)
                base_polys = [list(comp) for comp in vec]
                pols2.append((spin, gap, base_polys))
                spin += 1
            else:
                pols1.append([mpf(v) for v in vec])
        self.pols2 = pols2
        self.pols1 = pols1


# ---------------------------------------------------------------------------
# Verified-LP solver loop
# ---------------------------------------------------------------------------

@dataclass
class BoundResult:
    """Result of a verified-LP bound solve (:func:`solve_bound`).

    - ``certified``: ``True`` iff the recovered functional passed the
      high-precision certificate (no continuum / scalar positivity violation).
    - ``bound``: the driver's return ``Zmin`` recomputed in ``mpmath`` from the
      certified ``alpha`` (``mpf``); ``None`` when not certified or the LP found no
      functional.
    - ``status``: the FIND-backend status string of the final solve.
    - ``n_refine``: number of refine rounds used (0 = certified on first solve).
    - ``alpha``: the recovered functional (list of ``float``), or ``None``.
    - ``residual``: diagnostic dict -- ``n_violations`` of the final certificate and
      the witness ``x_star`` values that were (or would be) fed back.
    - ``reason``: the SDPB backend's raw ``terminateReason`` string (empty for the
      LP/SDPA paths). Lets a ``status='infeasible'`` cell record WHY -- a
      ``maxComplementarity exceeded`` precision-stall vs a genuine
      ``primal-dual infeasible`` feasibility wall must NOT be conflated (they
      can produce the same status string with opposite physical meaning).
    - ``warm``: fp64-backend continuation state (``alpha``/``tight``/``zeros``/
      ``cuts``/``tier``), consumable as the ``warm_start`` of the next solve of
      a NEARBY problem (:func:`solve_bound_fp64_sequence`). ``None`` on the
      other backends. Excluded from ``repr`` (it carries the full functional).
    - ``zmin_ball``: rigorous flint-``arb`` enclosure of the SAME ``bound`` dot
      product, as a decimal ``(lower_str, upper_str)`` pair with
      ``lower_str <= bound <= upper_str`` (see :func:`_dot_ball`). Additive: it
      never changes ``bound``. Populated only on the fp64 LP-FIND
      (:func:`solve_bound`) and fp64 cutting-plane (:func:`solve_bound_fp64`)
      paths, whose ``bound`` is the recomputed ``mp.fsum`` point value; ``None``
      elsewhere. Conservative endpoint per weak duality: LOWER for a
      lower-bound driver, UPPER for an upper-bound driver.
    """

    certified: bool
    bound: object
    status: str
    n_refine: int
    alpha: object = None
    residual: dict = field(default_factory=dict)
    reason: str = ""
    warm: object = field(default=None, repr=False)
    zmin_ball: object = None


def solve_bound(build_sdp_callable, *, sign, return_zmin, max_refine=10,
                samples_per_spin=800):
    """Verified-LP solve: FIND -> certify -> refine on witness ``x*``.

    ``build_sdp_callable()`` returns the assembled
    :class:`~dual.assembly.SDP`. The loop:

    1. ``find_functional(sdp, sign=sign)`` -- fp64 LP FIND. If the LP is not
       feasible, return that status immediately (no functional).
    2. ``certify_functional(alpha, view)`` -- high-precision certificate. If it
       certifies, recompute the driver's ``Zmin`` via ``return_zmin(alpha, sdp)``
       in ``mpmath`` and return it.
    3. Otherwise append the violation witness ``x*`` values to the LP sample grid
       (``extra_x``) and re-find. Repeat up to ``max_refine`` rounds.

    ``sign`` is the LP objective sign passed to ``find_functional`` (``+1`` for the
    maximise-then-negate upper-bound style). ``return_zmin`` is
    the per-driver map ``(alpha, sdp) -> Zmin`` (e.g. ``-Vec1.obj`` for an upper
    bound). Returns a :class:`BoundResult`.
    """
    sdp = build_sdp_callable()
    extra_x = []
    status = "error"
    alpha = None
    last_info = {}

    for n_refine in range(max_refine + 1):
        result = find_functional(sdp, sign=sign, samples_per_spin=samples_per_spin,
                                 extra_x=extra_x if extra_x else None)
        status = result.status
        if not result.feasible:
            return BoundResult(certified=False, bound=None, status=status,
                               n_refine=n_refine, alpha=None,
                               residual={"n_violations": None})
        alpha = result.alpha

        view = _SdpView(sdp)
        certified, violations = verify.certify_functional(alpha, view)
        if certified:
            zmin = return_zmin(alpha, sdp)
            sense = "max" if return_zmin is _zmin_upper else "min"
            return BoundResult(certified=True, bound=zmin, status=status,
                               n_refine=n_refine, alpha=alpha,
                               residual={"n_violations": 0},
                               zmin_ball=_dot_ball(sdp.objective, alpha, sense))

        # Refine: feed the sub-gap witnesses x* back into the LP sample grid.
        witnesses = [v["x_star"] for v in violations if "x_star" in v]
        last_info = {"n_violations": len(violations),
                     "x_star": [mp.nstr(w, 15) for w in witnesses]}
        new = [float(w) for w in witnesses if float(w) > 0]
        if not new:
            # No actionable interior witness (scalar pols1 failure, or x*=0):
            # refining the grid cannot help, so stop.
            break
        extra_x = extra_x + new

    return BoundResult(certified=False, bound=None, status=status,
                       n_refine=max_refine, alpha=alpha, residual=last_info)


# ---------------------------------------------------------------------------
# High-precision SDPA-GMP solve path
# ---------------------------------------------------------------------------

def solve_bound_sdpa(build_sdp_callable, *, sense, certify=True,
                     precision_bits=1024, epsilon=mpf(10) ** -15,
                     max_iteration=500, cert_rel_tol=mpf(10) ** -9):
    """High-precision bound solve via the SDPA-GMP backend (method of record).

    ``build_sdp_callable()`` returns the assembled
    :class:`dual.assembly.SDP`. Unlike :func:`solve_bound` (the
    fp64 LP FIND pre-screen, a relaxation that cannot reproduce the SDP numbers),
    this routes the exact SOS-on-the-half-line standard SDP to SDPA-GMP
    (:func:`dual.backends.sdpa_gmp.solve_sdp`) and returns its
    arbitrary-precision bound. ``sense`` is ``"max"`` (upper-bound drivers) or
    ``"min"`` (lower-bound drivers).

    When ``certify`` the recovered functional is independently re-certified at
    high precision with :func:`dual.verify.certify_functional`;
    ``BoundResult.certified`` reflects it (the bound is returned regardless, so a
    precision-limited boundary dip does not discard a usable number).
    """
    from .backends import sdpa_gmp  # local: optional heavy dep

    sdp = build_sdp_callable()
    res = sdpa_gmp.solve_sdp(sdp, precision_bits=precision_bits, sense=sense,
                             epsilon=epsilon, max_iteration=max_iteration)
    if not res.feasible:
        return BoundResult(certified=False, bound=None, status=res.status,
                           n_refine=0, alpha=None,
                           residual={"n_violations": None})

    certified = True
    n_viol = 0
    if certify:
        certified, violations = verify.certify_functional(
            res.alpha, _SdpView(sdp), rel_tol=cert_rel_tol)
        n_viol = len(violations)

    return BoundResult(certified=certified, bound=res.bound, status=res.status,
                       n_refine=0, alpha=res.alpha,
                       residual={"n_violations": n_viol,
                                 "cert_rel_tol": float(cert_rel_tol),
                                 "solver": "sdpa"})


# ---------------------------------------------------------------------------
# fp64 cutting-plane + exact-polish solve path (no arbitrary-precision IPM)
# ---------------------------------------------------------------------------

def _round_alpha(alpha, dps):
    """Round a functional to ``dps`` significant digits (exact binary round).

    ``mp.workdps`` + unary plus re-rounds each entry to the reduced mantissa,
    so a dps=200 polish output (~665-bit mantissas) shrinks to ~133 bits. The
    functional moves by ~1e-``dps`` RELATIVE -- its positivity dips move by
    the same amount, dozens of decades inside the certificate's 1e-9 band --
    while the exact-rational coefficients the certificate must factor shrink
    enormously (flint's root isolation cost scales with coefficient bit
    size). The certificate emitted downstream is still EXACT, for THIS
    (still feasible) rounded functional; the caller must therefore return
    the rounded alpha and a bound recomputed from it, never the unrounded
    pair.
    """
    with mp.workdps(dps):
        return [+mpf(a) for a in alpha]


def _bound_from_alpha(alpha, sdp, sense):
    """``bound = -+ objective . alpha`` recomputed at ambient precision."""
    obj_dot = mp.fsum(mpf(o) * mpf(a) for o, a in zip(sdp.objective, alpha))
    return -obj_dot if sense == "max" else obj_dot


# ---------------------------------------------------------------------------
# Rigorous enclosure of the bound dot product (flint-arb ball arithmetic)
# ---------------------------------------------------------------------------

def _mpf_to_arb(x):
    """Exact ``arb`` (zero-radius ball) of an ``mpf``.

    An ``mpf`` has an exact dyadic value ``man * 2**exp``, so it maps to an
    ``arb`` with NO rounding (the ball has zero radius). Reuses
    :func:`dual.verify._mpf_to_fraction` for the exact dyadic split.
    """
    import flint
    fr = verify._mpf_to_fraction(x)
    return flint.arb(flint.fmpq(fr.numerator, fr.denominator))


def _arb_endpoint_str(endpoint, places, *, up):
    """Directed-decimal string of an EXACT ``arb`` endpoint (``man * 2**exp``).

    ``up=False`` truncates toward ``-inf`` (a safe LOWER string), ``up=True``
    toward ``+inf`` (a safe UPPER string), so the binary->decimal conversion can
    only WIDEN the reported enclosure -- never narrow it below the true value.
    """
    from fractions import Fraction
    man, exp = endpoint.man_exp()
    fr = Fraction(int(man)) * (Fraction(2) ** int(exp))
    scaled = fr * (10 ** places)
    p, q = scaled.numerator, scaled.denominator
    n = (-((-p) // q)) if up else (p // q)   # ceil / floor toward the safe side
    neg = n < 0
    digits = str(abs(n)).rjust(places + 1, "0")
    cut = len(digits) - places
    intpart = digits[:cut] if places else digits
    fracpart = digits[cut:] if places else ""
    return ("-" if neg else "") + intpart + ("." + fracpart if places else "")


def _dot_ball(objective, alpha, sense):
    """Rigorous flint-``arb`` enclosure of the sign-adjusted bound dot product.

    Recomputes ``Zmin = +/- sum_k objective[k] * alpha[k]`` in Arb ball
    arithmetic (each ``mpf`` converts to an ``arb`` EXACTLY -- its value is
    dyadic -- so the ONLY inflation is the certified outward rounding of the ball
    products/sums). Returns the enclosure as a decimal ``(lower_str, upper_str)``
    pair with ``lower_str <= Zmin <= upper_str`` rigorously.

    Conservative side (weak duality direction). A certified LOWER-bound
    (``sense='min'``) functional value is itself a rigorous lower bound on the
    CFT quantity, so its safe surrogate is the ball's LOWER endpoint (reporting a
    smaller number keeps it a valid lower bound); a certified UPPER-bound
    (``sense='max'``) functional over-estimates, so its safe surrogate is the
    UPPER endpoint. The pair carries both; the driver reads the conservative side
    per this convention (documented on :class:`BoundResult`). Additive: leaves
    ``bound`` and every other field untouched.
    """
    import flint
    saved = flint.ctx.prec
    flint.ctx.prec = max(4 * mp.prec, 1024)
    try:
        acc = flint.arb(0)
        for o, a in zip(objective, alpha):
            acc = acc + _mpf_to_arb(o) * _mpf_to_arb(a)
        if sense == "max":
            acc = -acc
        rad = acc.rad()
        if rad.is_zero():
            places = 50
        else:
            rman, rexp = rad.man_exp()
            rmid = mpf(int(rman)) * mpf(2) ** int(rexp)
            places = min(max(int(mp.floor(-mp.log10(rmid))) + 2, 6), 100)
        lo = _arb_endpoint_str(acc.lower(), places, up=False)
        hi = _arb_endpoint_str(acc.upper(), places, up=True)
    finally:
        flint.ctx.prec = saved
    return (lo, hi)


def _warm_state(res, margin=None):
    """Continuation state of a backend result (see ``BoundResult.warm``).

    ``margin`` records the (possibly escalated) LP margin the solve ENDED at:
    a warm-started successor resumes from it instead of re-learning the same
    escalation ladder two refine rounds at a time.
    """
    info = res.info or {}
    pattern = info.get("pattern") or {}
    return {"alpha": res.alpha, "tight": pattern.get("tight"),
            "zeros": pattern.get("zeros"), "cuts": info.get("cuts"),
            "tier": info.get("tier"), "margin": margin}


def solve_bound_fp64(build_sdp_callable, *, sense, certify=True, max_refine=10,
                     cert_rel_tol=mpf(10) ** -9, margin=1e-8, round_dps=40,
                     warm_start=None, verbose=False):
    """fp64-search bound solve: cutting-plane LP FIND + exact Newton polish.

    The search tier is float64 only (HiGHS LP + numpy root-finding; no
    arbitrary-precision interior-point iterations anywhere); the digits come
    from the mpmath Newton polish of the KKT active set inside
    :func:`dual.backends.cutting_plane.solve_sdp`, and rigor from the same
    independent certificate as the SDPA/SDPB tiers. A certificate refutation
    hands its witness ``x*`` back to the backend as an active-set hint (a zero
    the fp64 stage missed) and the solve repeats, up to ``max_refine`` rounds
    -- the exact-precision analogue of :func:`solve_bound`'s refine loop.

    ``margin`` is the LP cut-row margin (see
    :func:`dual.backends.cutting_plane._lp_once`): on the LP tier it is what
    makes the returned vertex strictly feasible -- and hence CERTIFIABLE --
    at the price of a bound weakened by ``O(margin)``; a certified suboptimal
    functional is still a valid bound. On the polished tier it only shifts
    the Newton seed, not the returned digits. ``margin=0`` recovers the
    original diagnostic-only LP tier. The default sits in a MEASURED window
    (c=1.01 interval anchor): ``1e-6`` certifies but weakens the
    bound by ~3e-6 relative (loses the published-anchor bar), ``1e-8``
    certifies at ~1.4e-8 relative, ``1e-10`` fails to certify (below the
    HiGHS enforcement floor, so the exact-oracle cuts stop being actionable).
    When the certificate keeps refuting on the LP tier with both feedback
    channels exhausted, the margin ESCALATES one decade per refine round (cap
    ``1e-5``) -- off-anchor points have per-problem cancellation ratios no
    single margin fits, and a certified bound weakened by O(margin) beats an
    uncertified sharp one. Escalation consumes ``max_refine`` rounds.

    ``round_dps``: the candidate functional is rounded to this many
    significant digits BEFORE certification (see :func:`_round_alpha`), and
    on success the returned ``alpha``/``bound`` are the rounded pair -- the
    certificate and the number it guarantees always refer to the same
    object. ``round_dps=None`` disables (certify the raw polish output).

    ``warm_start``: continuation state from a previous solve of a NEARBY
    problem (a ``BoundResult.warm``); forwarded to the backend on the FIRST
    round only (a refutation means the warm pattern was wrong -- refine
    rounds go back through the full LP path). Every return carries the new
    continuation state in ``BoundResult.warm``.

    ``sense`` is ``"max"`` (upper-bound drivers) / ``"min"`` (lower-bound
    drivers), as in :func:`solve_bound_sdpa`. Returns a :class:`BoundResult`
    tagged ``solver="fp64"``.
    """
    from .backends import cutting_plane

    sdp = build_sdp_callable()
    extra_cuts = []
    extra_zeros = []
    violations = []
    res = None
    prev_lp_bound = None
    cur_margin = margin
    margin_max = 1e-5
    lp_rounds = 0
    if warm_start and warm_start.get("margin") and margin:
        # Resume from the predecessor's (possibly escalated) margin level
        # instead of re-climbing the same ladder two refine rounds per
        # decade. Never below the caller's requested margin; never above the
        # cap. A user margin of 0 (diagnostic tier) is respected as-is.
        cur_margin = min(max(margin, warm_start["margin"]), margin_max)

    def _escalate():
        """Refutation persists on the LP tier with the feedback channels
        exhausted: raise the margin one decade (up to ``margin_max``) and try
        again -- a certified bound weakened by O(margin) beats an uncertified
        one. Off-anchor points need this: the cancellation ratio between the
        LP row scales and the certificate's action-coefficient scale varies
        by problem, so no single margin fits every point of a scan."""
        nonlocal cur_margin, prev_lp_bound
        if not cur_margin or cur_margin >= margin_max:
            return False
        cur_margin = min(10 * cur_margin, margin_max)
        prev_lp_bound = None
        if verbose:
            print(f"  [refine {n_refine}] escalating margin to {cur_margin:g}")
        return True

    for n_refine in range(max_refine + 1):
        res = cutting_plane.solve_sdp(sdp, sense=sense,
                                      extra_cuts=extra_cuts or None,
                                      extra_zeros=extra_zeros or None,
                                      margin=cur_margin,
                                      warm_start=(warm_start if n_refine == 0
                                                  else None),
                                      verbose=verbose)
        if not res.feasible:
            return BoundResult(certified=False, bound=None, status=res.status,
                               n_refine=n_refine, alpha=None,
                               residual={"n_violations": None,
                                         "solver": "fp64"})
        if not certify:
            return BoundResult(certified=True, bound=res.bound,
                               status=res.status, n_refine=n_refine,
                               alpha=res.alpha,
                               residual={"n_violations": 0, "solver": "fp64"},
                               warm=_warm_state(res, cur_margin))
        alpha_c = (_round_alpha(res.alpha, round_dps) if round_dps
                   else res.alpha)
        certified, violations = verify.certify_functional(
            alpha_c, _SdpView(sdp), rel_tol=cert_rel_tol)
        if certified:
            # Margin ANNEAL: the escalation ladder above overshoots for
            # robustness; step the margin back down one decade at a time --
            # keeping every cut the climb discovered (res.info["cuts"]) --
            # while the certificate keeps passing. The returned level is the
            # sharpest CERTIFIABLE one for this cut set, i.e. a property of
            # the problem rather than of the path taken to it: cold and
            # warm-started solves land on the same level (the between-path
            # identity the benchmark asserts), and the anchor keeps its
            # published 1e-8-sharp bracket instead of the climb's 1e-7.
            while cur_margin and margin and cur_margin > 1.0001 * margin:
                trial = max(cur_margin / 10, margin)
                res2 = cutting_plane.solve_sdp(
                    sdp, sense=sense, extra_cuts=extra_cuts or None,
                    extra_zeros=extra_zeros or None, margin=trial,
                    warm_start={"tier": "lp", "cuts": res.info.get("cuts")},
                    verbose=verbose)
                if not res2.feasible:
                    break
                alpha2 = (_round_alpha(res2.alpha, round_dps) if round_dps
                          else res2.alpha)
                ok2, _viol2 = verify.certify_functional(
                    alpha2, _SdpView(sdp), rel_tol=cert_rel_tol)
                if verbose:
                    print(f"  [anneal] margin {trial:g}: "
                          f"{'certified' if ok2 else 'refuted'}")
                if not ok2:
                    break
                res, alpha_c, cur_margin = res2, alpha2, trial
            return BoundResult(certified=True,
                               bound=_bound_from_alpha(alpha_c, sdp, sense),
                               status=res.status, n_refine=n_refine,
                               alpha=alpha_c,
                               residual={"n_violations": 0,
                                         "cert_rel_tol": float(cert_rel_tol),
                                         "solver": "fp64",
                                         "tier": res.info.get("tier"),
                                         "margin": cur_margin},
                               zmin_ball=_dot_ball(sdp.objective, alpha_c, sense),
                               warm=_warm_state(res, cur_margin))
        # Feed the refutation back on both channels: raw witnesses as LP
        # positivity CUTS (safe, monotone), and exact-paired dip locations as
        # KKT pattern ZEROS (what the fp64 extraction is blind to -- the
        # action cancels by ~1e-12 of its term magnitudes on the zero-rich
        # interval problems). res.alpha is an LP vertex or an accepted
        # polish, never a warped stall, so the pairing is sound.
        new_cuts = [(int(v["spin"]), float(v["x_star"])) for v in violations
                    if v.get("kind") == "pols2"
                    and float(v.get("x_star", 0)) > 0]
        new_cuts = [(si, x) for si, x in new_cuts
                    if all(si != s0 or abs(x - x0) > 1e-12
                           for s0, x0 in extra_cuts)]
        new_zeros = cutting_plane.witness_zero_hints(sdp, alpha_c,
                                                     violations)
        new_zeros = [(si, x) for si, x in new_zeros
                     if all(si != s0 or abs(x - x0) > 1e-6
                            for s0, x0 in extra_zeros)]
        if verbose and (new_cuts or new_zeros):
            print(f"  [refine {n_refine}] adding {len(new_cuts)} cut(s), "
                  f"{len(new_zeros)} pattern zero(s)")
        extra_cuts = extra_cuts + new_cuts
        extra_zeros = extra_zeros + new_zeros
        if res.info.get("tier") == "lp":
            # LP-tier escalation policy: witness feedback gets SIX rounds at
            # the caller's BASE margin -- the sharpest level, where iterated
            # witness cuts genuinely converge (the c=1.01 anchor's lower side
            # certifies at 1e-8 only around round 5; a 4-round budget was
            # measured to escalate first and lose the published-anchor bar,
            # and no one-shot anneal-down recovers it: a single re-solve at
            # 1e-8 with the climb's cut set is refuted, only the iterated
            # witness rounds get there) -- and TWO rounds per escalated
            # level. A refutation that survives its budget -- or a stalled
            # bound, or empty feedback channels -- means the remaining dips
            # sit below what this margin can lift, and re-cutting is
            # whack-a-mole (measured on the gap-down walk: 14 fresh pols2
            # dips per round at a fixed 1e-8).
            lp_rounds += 1
            stable = (prev_lp_bound is not None
                      and abs(mpf(res.bound) - prev_lp_bound)
                      <= mpf(10) ** -12 * abs(mpf(res.bound)))
            prev_lp_bound = mpf(res.bound)
            level_budget = 6 if cur_margin == margin else 2
            if ((not new_cuts and not new_zeros) or stable
                    or lp_rounds >= level_budget):
                if _escalate():
                    lp_rounds = 0
                    continue
                if (not new_cuts and not new_zeros) or stable:
                    break
        elif not new_cuts and not new_zeros:
            break

    return BoundResult(certified=False, bound=res.bound, status=res.status,
                       n_refine=n_refine, alpha=res.alpha,
                       residual={"n_violations": len(violations),
                                 "solver": "fp64",
                                 "tier": res.info.get("tier")},
                       warm=_warm_state(res, cur_margin))


# ---------------------------------------------------------------------------
# fp64 continuation drivers (homotopy across nearby problems)
# ---------------------------------------------------------------------------

def solve_bound_fp64_sequence(builders, *, sense, warm_start=None, **kwargs):
    """Continuation solve of a SEQUENCE of nearby problems (fp64 backend).

    ``builders`` is an ordered iterable of zero-argument SDP builders whose
    problems deform CONTINUOUSLY (e.g. a bisection walk in the gap at fixed
    ``c``, all other parameters identical). Each point is solved with
    :func:`solve_bound_fp64` warm-started from the previous point's
    continuation state: a polished predecessor seeds the Newton polish
    directly (the LP is skipped entirely), an LP-tier predecessor seeds the
    cut set (the Kelley discovery is skipped). The first point is a cold
    solve -- if its polish stalls, bootstrap it with
    :func:`solve_bound_fp64_ladder` and pass the result's ``warm`` in as
    ``warm_start``.

    Correctness does not depend on the warm start: every point still passes
    the SAME independent certificate, and the multiplier sign repair in the
    polish rejects a drifted active set (see
    :func:`dual.backends.cutting_plane.solve_sdp`). ``kwargs`` are forwarded
    to :func:`solve_bound_fp64`. Returns the list of :class:`BoundResult`.
    """
    results = []
    warm = warm_start
    for build in builders:
        res = solve_bound_fp64(build, sense=sense, warm_start=warm, **kwargs)
        results.append(res)
        warm = res.warm or warm
    return results


def interval_bracket_fp64_sequence(points, NN, Lmax, lmax, **kwargs):
    """Continuation brackets along a walk of interval problems.

    ``points`` is an ordered iterable of ``(c, gap, Da, Db)`` tuples deforming
    continuously. Per point the ±sign SDP pair is built ONCE
    (:func:`dual.assembly.build_interval_sdp_pair`; the two programs differ
    only in the normalization head) and each side is solved with
    :func:`solve_bound_fp64`, warm-started along its own chain (the lower and
    upper extremal functionals deform independently, so the chains never mix).
    Returns a list of ``(lower, upper)`` :class:`BoundResult` pairs, the
    :func:`two_sided_bracket` convention. ``kwargs`` forward to
    :func:`solve_bound_fp64`.
    """
    from .assembly import build_interval_sdp_pair

    out = []
    warm_lo = warm_up = None
    for (c, gap, Da, Db) in points:
        plus, minus = build_interval_sdp_pair(c, gap, Da, Db, NN, Lmax, lmax)
        lo = solve_bound_fp64(lambda: plus, sense="min", warm_start=warm_lo,
                              **kwargs)
        up = solve_bound_fp64(lambda: minus, sense="max", warm_start=warm_up,
                              **kwargs)
        warm_lo = lo.warm or warm_lo
        warm_up = up.warm or warm_up
        out.append((lo, up))
    return out


def _embed_alpha(alpha, Lmax_from, Lmax_to):
    """Embed a functional across derivative-index tables ``tb(Lmax)``.

    The Base layout is ``[count slot] + [one component per tb(Lmax) pair]``,
    and ``tb`` interleaves new pairs among old ones when ``Lmax`` grows (it
    sorts by first index), so the embedding maps components BY PAIR, zero-
    filling the pairs the smaller table lacks. Exact: entries are copied,
    never recomputed.
    """
    from .characters import tb

    pairs_from = tb(Lmax_from)
    pairs_to = tb(Lmax_to)
    index_from = {p: i for i, p in enumerate(pairs_from)}
    out = [mpf(0)] * (1 + len(pairs_to))
    out[0] = mpf(alpha[0])
    for i, p in enumerate(pairs_to):
        j = index_from.get(p)
        if j is not None:
            out[1 + i] = mpf(alpha[1 + j])
    return out


def solve_bound_fp64_ladder(build_factory, lmaxes, *, sense, **kwargs):
    """Lmax-stepping bootstrap: climb ``lmaxes`` reusing each solution.

    ``build_factory(Lmax)`` returns the zero-argument-equivalent SDP for that
    truncation order (all other parameters fixed, in particular ``lmax`` --
    the spin-sector count must not change or the zero pattern cannot carry).
    Each rung is solved warm-started from the previous one with the
    functional embedded across the derivative tables (:func:`_embed_alpha`)
    and the zero pattern reused as-is (zeros move continuously with
    ``Lmax``). Use when a cold solve at the target ``Lmax`` stalls: the
    low-``Lmax`` rungs are cheap and pin the active set. Returns the final
    rung's :class:`BoundResult` (its ``warm`` continues a sequence).
    """
    warm = None
    res = None
    for i, L in enumerate(lmaxes):
        res = solve_bound_fp64(lambda L=L: build_factory(L), sense=sense,
                               warm_start=warm, **kwargs)
        warm = res.warm or warm
        if warm and warm.get("alpha") is not None and i + 1 < len(lmaxes):
            warm = dict(warm)
            warm["alpha"] = _embed_alpha(warm["alpha"], L, lmaxes[i + 1])
    return res


# ---------------------------------------------------------------------------
# fp64 parallel bracket pair (two processes -- NEVER threads)
# ---------------------------------------------------------------------------

def _interval_worker(conn, sense, sign, params, kwargs):
    """Child-process body of :func:`interval_bracket_fp64_parallel`.

    Runs one side of the bracket and ships the :class:`BoundResult` (or the
    raised exception) back over the pipe. Module-level so the ``spawn``
    context can pickle it; the child re-imports :mod:`dual`, which re-runs
    the ``characters`` import side effect (``mp.dps = 200``), so the child's
    ambient precision matches a fresh interpreter's.
    """
    try:
        res = solve_bound_fp64(
            lambda: build_interval_sdp(*params, sign=sign),
            sense=sense, **kwargs)
        conn.send(res)
    except Exception as exc:  # ship the failure; the parent re-raises
        conn.send(exc)
    finally:
        conn.close()


def interval_bracket_fp64_parallel(c, gap, Da, Db, NN, Lmax, lmax, **kwargs):
    """Two-sided interval bracket with the lower/upper solves in PARALLEL.

    Spawns one process per side (``sense="min"``/``sign=+1`` and
    ``sense="max"``/``sign=-1``, the :func:`integer_{lower,upper}bound_interval`
    conventions) and returns ``(lower, upper)`` :class:`BoundResult`. The two
    sides are independent solves of comparable cost, so the pair completes in
    ~1x single-side wall instead of 2x.

    PROCESSES, never threads: mpmath precision is process-global state (the
    ``characters`` import sets ``mp.dps = 200`` process-wide), so two solver
    threads would race on it. The ``spawn`` context is forced -- ``fork``
    would duplicate live mpmath/flint state into the children. ``mpf`` values
    pickle exactly (dyadic payload), so the shipped bounds/functionals are
    bit-identical to a sequential run's. ``kwargs`` are forwarded to
    :func:`solve_bound_fp64` in both children.
    """
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    params = (mpf(c), mpf(gap), mpf(Da), mpf(Db), int(NN), int(Lmax),
              int(lmax))
    procs, parents = [], []
    for sense, sign in (("min", +1), ("max", -1)):
        parent, child = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_interval_worker,
                        args=(child, sense, sign, params, kwargs))
        p.start()
        child.close()
        procs.append(p)
        parents.append(parent)
    out = []
    for parent, p in zip(parents, procs):
        obj = parent.recv()
        p.join()
        if isinstance(obj, Exception):
            for q in procs:
                if q.is_alive():
                    q.terminate()
            raise obj
        out.append(obj)
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Rigorous SDPB solve path + two-tier escalation
# ---------------------------------------------------------------------------

def solve_bound_sdpb(build_sdp_callable, *, sense, certify=True,
                     precision_bits=1024, duality_gap_threshold=mpf(10) ** -30,
                     max_iteration=500, cert_rel_tol=mpf(10) ** -9,
                     np_ranks=None, checkpoint_dir=None,
                     initial_checkpoint_dir=None):
    """Rigorous bound solve via the SDPB backend (the fp64-wall-free tier).

    Sign-for-sign the SDPB twin of :func:`solve_bound_sdpa`. SDPB reads the SDP
    from arbitrary-precision text (no float64 ingest cast), so it returns an
    ``"optimal"`` bracket on the high-``c`` / low-gap / large-``Lmax`` cells where
    SDPA-GMP spuriously stalls. ``sense`` is ``"max"`` (upper) / ``"min"`` (lower).
    Runs only where ``apptainer`` + the SDPB image exist.

    ``checkpoint_dir`` / ``initial_checkpoint_dir`` are a transparent conduit
    forwarded verbatim to :func:`sdpb.solve_sdp`; both ``None`` (the default)
    forwards ``None``/``None``, leaving the live call byte-identical.
    """
    from .backends import sdpb  # local: heavy / environment-specific dep

    sdp = build_sdp_callable()
    res = sdpb.solve_sdp(sdp, precision_bits=precision_bits, sense=sense,
                         duality_gap_threshold=duality_gap_threshold,
                         max_iteration=max_iteration, np_ranks=np_ranks,
                         checkpoint_dir=checkpoint_dir,
                         initial_checkpoint_dir=initial_checkpoint_dir)
    if not res.feasible:
        return BoundResult(certified=False, bound=None, status=res.status,
                           n_refine=0, alpha=None,
                           residual={"n_violations": None, "solver": "sdpb"},
                           reason=res.reason)

    certified = True
    n_viol = 0
    if certify:
        certified, violations = verify.certify_functional(
            res.alpha, _SdpView(sdp), rel_tol=cert_rel_tol)
        n_viol = len(violations)

    return BoundResult(certified=certified, bound=res.bound, status=res.status,
                       n_refine=0, alpha=res.alpha,
                       residual={"n_violations": n_viol,
                                 "cert_rel_tol": float(cert_rel_tol),
                                 "solver": "sdpb"},
                       reason=res.reason)


def solve_bound_escalate(build_sdp_callable, *, sense, certify=True,
                         precision_bits=1024, epsilon=mpf(10) ** -15,
                         max_iteration=500, cert_rel_tol=mpf(10) ** -9,
                         duality_gap_threshold=mpf(10) ** -30, np_ranks=None):
    """Two-tier solve: SDPA-GMP first, escalate to SDPB only when it falls short.

    Runs the fast :func:`solve_bound_sdpa`; if that yields a certified
    ``"optimal"`` bound it is returned (tagged ``solver="sdpa"``). Otherwise -- the
    fp64-input wall (no bound) or a precision-limited, uncertified dip -- the same
    SDP is re-solved with the rigorous :func:`solve_bound_sdpb` and that result is
    returned, tagged ``solver="sdpb"`` and carrying ``sdpa_status`` for provenance.
    This is the "try the simple method, elevate to the rigorous one on failure"
    policy: SDPA carries the cheap interior, SDPB rescues the walls and extremes.
    """
    sdpa_res = solve_bound_sdpa(build_sdp_callable, sense=sense, certify=certify,
                                precision_bits=precision_bits, epsilon=epsilon,
                                max_iteration=max_iteration,
                                cert_rel_tol=cert_rel_tol)
    if sdpa_res.status == "optimal" and sdpa_res.bound is not None \
            and sdpa_res.certified:
        return sdpa_res

    sdpb_res = solve_bound_sdpb(build_sdp_callable, sense=sense, certify=certify,
                                precision_bits=precision_bits,
                                duality_gap_threshold=duality_gap_threshold,
                                max_iteration=max_iteration,
                                cert_rel_tol=cert_rel_tol, np_ranks=np_ranks)
    sdpb_res.residual["sdpa_status"] = sdpa_res.status
    return sdpb_res


# ---------------------------------------------------------------------------
# Zmin return maps
# ---------------------------------------------------------------------------

def _vec_dot_obj(alpha, sdp):
    """``Vec1 . obj`` recomputed in ``mpmath`` from the recovered ``alpha``.

    The objective is ``Base0``; ``alpha`` aligns with it slot-for-
    slot (the FIND backend works in un-reshuffled Base coordinates).
    """
    return mp.fsum(mpf(o) * mpf(a) for o, a in zip(sdp.objective, alpha))


def _zmin_lower(alpha, sdp):
    """``Zmin = +Vec1.obj`` (lower-bound drivers)."""
    return _vec_dot_obj(alpha, sdp)


def _zmin_upper(alpha, sdp):
    """``Zmin = -Vec1.obj`` (upper-bound drivers; the minus = maximisation)."""
    return -_vec_dot_obj(alpha, sdp)


# ---------------------------------------------------------------------------
# Plain drivers
# ---------------------------------------------------------------------------

def integer_lowerbound(c, gap, Da, NN, Lmax, lmax, *, max_refine=10,
                       samples_per_spin=800):
    """``integerlowerbound(c, gap, Da, NN, Lmax, lmax)``.

    ``norm = +e0``, ``obj = Base0``, return ``Zmin = +Vec1.obj`` -- the certified
    LOWER bound on the vacuum coefficient. Returns a :class:`BoundResult`.
    """
    return solve_bound(lambda: build_bound_sdp(c, gap, Da, NN, Lmax, lmax, sign=+1),
                       sign=+1, return_zmin=_zmin_lower,
                       max_refine=max_refine, samples_per_spin=samples_per_spin)


def integer_upperbound(c, gap, Da, NN, Lmax, lmax, *, max_refine=10,
                       samples_per_spin=800):
    """``integerupperbound(c, gap, Da, NN, Lmax, lmax)``.

    ``norm = -e0``, ``obj = Base0``, return ``Zmin = -Vec1.obj`` -- the certified
    UPPER bound on the vacuum coefficient. Returns a :class:`BoundResult`.
    """
    return solve_bound(lambda: build_bound_sdp(c, gap, Da, NN, Lmax, lmax, sign=-1),
                       sign=+1, return_zmin=_zmin_upper,
                       max_refine=max_refine, samples_per_spin=samples_per_spin)


def integer_fes(Ndel, c, gap, Da, NN, Lmax, lmax, *, max_refine=10,
                samples_per_spin=800, certified=True):
    """``integerfes(Ndel, c, gap, Da, NN, Lmax, lmax)``.

    Pure feasibility (``VAC0_SLOT = -Ndel``, ``norm = Base0``, ``obj = 0*norm``);
    returns a status STRING (no ``Zmin``). A dual-feasible solve excludes the
    assumption ("does a vacuum coeff ``>= Ndel`` exist?"); a primal-feasible solve
    admits it.

    Returns the FIND-backend status string. With the fp64 LP a feasible solve maps
    to ``"optimal"`` (a functional was found, i.e. the dual is feasible -> excluded)
    and an infeasible LP maps to ``"infeasible"`` (no separating functional ->
    the assumption is admitted).

    ``certified`` (default ``True``): gate exclusion on a CERTIFIED separating
    functional. The fp64 LP only *samples* continuum positivity, so an
    ``"optimal"`` LP vertex can still be negative at a sub-gap ``x*`` the grid
    missed -- a FALSE exclusion. This runs the FIND -> certify -> refine-on-witness
    loop (the :func:`solve_bound` pattern): each certificate refutation appends
    the witness ``x*`` to the LP grid (``extra_x``) and re-solves, up to
    ``max_refine`` rounds. Exclusion (``"optimal"``) is returned ONLY when
    :func:`dual.verify.certify_functional` certifies the recovered functional at
    high precision. If certification keeps refuting until refinement is exhausted
    (or no actionable interior witness remains), the LP-feasible functional is NOT
    a valid certificate, so the point is treated as NOT excluded (allowed-side):
    the returned status is ``"uncertified"``, which :func:`_fes_excluded` maps to
    admitted -- never a silent exclusion. ``certified=False`` restores the legacy
    single-solve behaviour (raw LP status, no certificate).
    """
    sdp = build_fes_sdp(Ndel, c, gap, Da, NN, Lmax, lmax)
    if not certified:
        result = find_functional(sdp, sign=+1, samples_per_spin=samples_per_spin)
        return result.status

    extra_x = []
    for _n_refine in range(max_refine + 1):
        result = find_functional(sdp, sign=+1, samples_per_spin=samples_per_spin,
                                 extra_x=extra_x if extra_x else None)
        if not result.feasible:
            # No separating functional at this grid -> assumption ADMITTED.
            return result.status
        ok, violations = verify.certify_functional(result.alpha, _SdpView(sdp))
        if ok:
            # A CERTIFIED separating functional excludes the assumption.
            return result.status
        # Refine: feed the sub-gap witnesses x* back into the LP sample grid.
        new = [float(v["x_star"]) for v in violations
               if "x_star" in v and float(v["x_star"]) > 0]
        if not new:
            # No actionable interior witness (scalar failure / x*=0): refining the
            # grid cannot help, so stop -- uncertified, hence NOT excluded.
            break
        extra_x = extra_x + new

    # Certification never succeeded within max_refine: NOT excluded (allowed-side).
    return "uncertified"


def integer_upperbound_interval(c, gap, Da, Db, NN, Lmax, lmax, *, max_refine=10,
                                samples_per_spin=800, backend="sdpa",
                                precision_bits=1024, epsilon=mpf(10) ** -15):
    """``integerupperboundinterval(c, gap, Da, Db, NN, Lmax, lmax)``.

    Interval method (HeavisideTheta indicator + dual ``Union`` grid): UPPER
    bound ``B`` on the operator coefficient INTEGRATED over ``[Da, Db]``, no
    second-gap assumption. ``norm = -e0``, return ``Zmin = -Vec1.obj``. Returns a
    :class:`BoundResult`.

    ``backend="sdpa"`` (default) uses the arbitrary-precision SDPA-GMP solver
    (method of record; reproduces the paper bracket); ``backend="lp"`` uses the
    fp64 LP FIND pre-screen (a relaxation that cannot reproduce the SDP numbers
    and returns ``infeasible`` on the narrow interval anchor).
    """
    build = lambda: build_interval_sdp(c, gap, Da, Db, NN, Lmax, lmax, sign=-1)
    if backend == "sdpa":
        return solve_bound_sdpa(build, sense="max", precision_bits=precision_bits,
                                epsilon=epsilon)
    if backend == "sdpb":
        return solve_bound_sdpb(build, sense="max", precision_bits=precision_bits)
    if backend == "fp64":
        return solve_bound_fp64(build, sense="max")
    if backend == "escalate":
        return solve_bound_escalate(build, sense="max",
                                    precision_bits=precision_bits, epsilon=epsilon)
    return solve_bound(build, sign=+1, return_zmin=_zmin_upper,
                       max_refine=max_refine, samples_per_spin=samples_per_spin)


def integer_lowerbound_interval(c, gap, Da, Db, NN, Lmax, lmax, *, max_refine=10,
                                samples_per_spin=800, backend="sdpa",
                                precision_bits=1024, epsilon=mpf(10) ** -15):
    """``integerlowerboundinterval(c, gap, Da, Db, NN, Lmax, lmax)``.

    Sign twin of :func:`integer_upperbound_interval`: ``norm = +e0``, return
    ``Zmin = +Vec1.obj`` -- LOWER bound ``A`` on the integrated coefficient over
    ``[Da, Db]``. Returns a :class:`BoundResult`.

    Requires the counted-spin continuum fix in :func:`dual.assembly.build_interval_sdp`
    (spin-0 SOS started at ``Db``); without it ``A`` collapses to ~0. With it,
    ``A`` reproduces the published value (e.g. ``A = 4.0722684`` at the c=1.01
    anchor). ``backend`` as in :func:`integer_upperbound_interval`.
    """
    build = lambda: build_interval_sdp(c, gap, Da, Db, NN, Lmax, lmax, sign=+1)
    if backend == "sdpa":
        return solve_bound_sdpa(build, sense="min", precision_bits=precision_bits,
                                epsilon=epsilon)
    if backend == "sdpb":
        return solve_bound_sdpb(build, sense="min", precision_bits=precision_bits)
    if backend == "fp64":
        return solve_bound_fp64(build, sense="min")
    if backend == "escalate":
        return solve_bound_escalate(build, sense="min",
                                    precision_bits=precision_bits, epsilon=epsilon)
    return solve_bound(build, sign=+1, return_zmin=_zmin_lower,
                       max_refine=max_refine, samples_per_spin=samples_per_spin)


# ---------------------------------------------------------------------------
# Spin drivers -- legacy conditional / second-gap method (see module docstring)
# ---------------------------------------------------------------------------

def integer_lowerbound_spin(c, gap, Da, LL, NN, Lmax, lmax, *, max_refine=10,
                            samples_per_spin=800, backend="sdpa",
                            precision_bits=1024, epsilon=mpf(10) ** -15):
    """``integerlowerboundspin(c, gap, Da, LL, NN, Lmax, lmax)``.

    Legacy conditional driver: single-operator second-gap method on spin
    ``LL`` (spin substitution + second-gap ``gaplist``, see
    :func:`dual.assembly.build_bound_sdp_spin`). ``norm = +e0``, return
    ``Zmin = +Vec1.obj`` -- LOWER bound on the spin-``LL`` operator
    coefficient. Returns a :class:`BoundResult`.

    ``backend="sdpa"`` (default) uses the arbitrary-precision SDPA-GMP solver
    (method of record); ``backend="lp"`` uses the fp64 LP FIND pre-screen.
    """
    build = lambda: build_bound_sdp_spin(c, gap, Da, LL, NN, Lmax, lmax, sign=+1)
    if backend == "sdpa":
        return solve_bound_sdpa(build, sense="min", precision_bits=precision_bits,
                                epsilon=epsilon)
    if backend == "sdpb":
        return solve_bound_sdpb(build, sense="min", precision_bits=precision_bits)
    if backend == "fp64":
        return solve_bound_fp64(build, sense="min")
    if backend == "escalate":
        return solve_bound_escalate(build, sense="min",
                                    precision_bits=precision_bits, epsilon=epsilon)
    return solve_bound(build, sign=+1, return_zmin=_zmin_lower,
                       max_refine=max_refine, samples_per_spin=samples_per_spin)


def integer_upperbound_spin(c, gap, Da, LL, NN, Lmax, lmax, *, max_refine=10,
                            samples_per_spin=800, backend="sdpa",
                            precision_bits=1024, epsilon=mpf(10) ** -15):
    """``integerupperboundspin(c, gap, Da, LL, NN, Lmax, lmax)``.

    Legacy conditional driver (sign twin of :func:`integer_lowerbound_spin`):
    ``norm = -e0``, return ``Zmin = -Vec1.obj`` -- UPPER bound on the
    spin-``LL`` operator coefficient. A validation gold value at ``Lmax=11``
    is ``[[1]] = 1.0801972581458...``. Returns a :class:`BoundResult`.

    ``backend="sdpa"`` (default) uses the arbitrary-precision SDPA-GMP solver
    (method of record); ``backend="lp"`` uses the fp64 LP FIND pre-screen.
    """
    build = lambda: build_bound_sdp_spin(c, gap, Da, LL, NN, Lmax, lmax, sign=-1)
    if backend == "sdpa":
        return solve_bound_sdpa(build, sense="max", precision_bits=precision_bits,
                                epsilon=epsilon)
    if backend == "sdpb":
        return solve_bound_sdpb(build, sense="max", precision_bits=precision_bits)
    if backend == "fp64":
        return solve_bound_fp64(build, sense="max")
    if backend == "escalate":
        return solve_bound_escalate(build, sense="max",
                                    precision_bits=precision_bits, epsilon=epsilon)
    return solve_bound(build, sign=+1, return_zmin=_zmin_upper,
                       max_refine=max_refine, samples_per_spin=samples_per_spin)


# ---------------------------------------------------------------------------
# Integrality filter
# ---------------------------------------------------------------------------

def integer_in(lo, hi):
    """Is there an integer in the closed bracket ``[lo, hi]``?

    The integrality test: ``[A, B]`` contains no integer -> the assumption is
    excluded. ``floor(hi) >= ceil(lo)`` iff a (closed-interval) integer exists.
    ``lo``/``hi`` are ``mpf``/numeric; the bracket is taken inclusive of both ends.
    Handles ``lo > hi`` (degenerate / inverted bracket) as no integer.
    """
    lo = mpf(lo)
    hi = mpf(hi)
    if lo > hi:
        return False
    return int(mp.floor(hi)) >= int(mp.ceil(lo))


def two_sided_bracket(make_lower, make_upper):
    """Run the lower + upper drivers and return ``[A, B]`` for the integrality filter.

    ``make_lower`` / ``make_upper`` are zero-argument callables returning a
    :class:`BoundResult` (e.g. ``lambda: integer_lowerbound_spin(...)`` and its
    upper twin with the SAME ``(c, gap, Da, LL, ...)`` arguments). Returns
    ``([A, B], lower_result, upper_result)`` where ``A`` is the certified lower
    bound and ``B`` the certified upper bound (either may be ``None`` if that side
    did not certify). The caller passes ``[A, B]`` to :func:`integer_in`.
    """
    lower = make_lower()
    upper = make_upper()
    A = lower.bound
    B = upper.bound
    return [A, B], lower, upper


# ---------------------------------------------------------------------------
# Gap bisection
# ---------------------------------------------------------------------------

def gap_bisection(make_fes, gap_lo, gap_hi, tol):
    """Bisect the critical gap via feasibility.

    ``make_fes(gap)`` runs a feasibility driver (e.g. :func:`integer_fes`) at the
    trial ``gap`` and returns its status STRING. A dual-feasible status
    (``"optimal"`` from the fp64 FIND LP -- a separating functional exists) means
    the gap is EXCLUDED; a primal-feasible / infeasible-LP status means it is
    still admitted.

    The bound is monotone in the gap (increasing the gap can only make
    the assumption harder to satisfy), so we bisect: ``gap_lo`` admitted,
    ``gap_hi`` excluded, narrow to width ``tol`` and return the critical gap
    (largest admitted / smallest excluded boundary, the midpoint of the final
    bracket).
    """
    lo = mpf(gap_lo)
    hi = mpf(gap_hi)
    tol = mpf(tol)
    while (hi - lo) > tol:
        mid = (lo + hi) / 2
        status = make_fes(mid)
        excluded = _fes_excluded(status)
        if excluded:
            hi = mid  # mid is excluded -> critical gap is below mid
        else:
            lo = mid  # mid is admitted -> critical gap is above mid
    return (lo + hi) / 2


def _fes_excluded(status):
    """Map a feasibility status STRING to "is the assumption excluded?" (boolean).

    A dual-feasible solve (a separating functional was found) excludes the
    assumption. With the fp64 FIND LP that is the ``"optimal"`` / ``"feasible"``
    status; an ``"infeasible"`` LP means no separating functional -> admitted.
    """
    return status in ("optimal", "feasible")
