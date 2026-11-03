"""Crossing derivatives -> semidefinite program -> solver input.

Turns the crossing-derivative composites of :mod:`dual.characters`
(``Com`` / ``Com0`` / ``Comwithprefactor``) into the semidefinite program the
solver backends consume, and emits it as the solver's
**polynomial-vector-matrix** input.

What it owns:

- ``reshuffle_with_normalization`` — folds the normalization constraint
  ``norm . a == 1`` into the first slot; ``safe_coeff_list`` / ``poly_degree``
  polynomial-in-``x`` helpers (ascending ``mpmath`` coeff lists, same convention
  as :mod:`dual.characters`).
- ``PositiveMatrixWithPrefactor`` and ``SDP`` dataclasses, plus the shared driver
  scaffolding ``build_bound_sdp`` (``base/baseexact/base0`` -> ``Base/Baseexact/
  Base0``, ``scalartb``, ``gaplist``, ``pols1``, ``pols2``).
- ``build_interval_sdp`` (the two-sided window bracket, including the split
  continuum for a scalar gap strictly below the counting window),
  ``build_window_bound_sdp`` / ``build_multi_window_fes_sdp`` (the explicit-edge
  window builders), ``build_window_bound_sdp_continuous`` (the same window
  bound with the window constraint on the continuum, via the rigorous
  polynomial bounds of ``exp_poly_bounds``), and the legacy conditional spin
  builder ``build_bound_sdp_spin`` (assumes a second gap).
- ``write_bootstrap_sdp`` emitting the polynomial-vector-matrix XML tree; every
  number is rendered with :func:`dual.characters.nf` (NO ``e``/``E``).

All construction numerics are ``mpmath`` at ``mp.dps >= 200``; float64 is
forbidden on this path.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath
from mpmath import mp, mpf

from .characters import (
    DAMP_BASE,
    PREC_DIGITS,
    DampedRational,
    Com0,
    _poly_compose_affine,
    com_poly_in_x,
    comwithprefactor_at,
    damped_shift,
    max_index_by,
    nf,
    orthogonal_polynomials,
    rescaled_laguerre_sample_points,
    tb,
)


# ---------------------------------------------------------------------------
# polynomial-in-x helpers
# ---------------------------------------------------------------------------

def safe_coeff_list(poly_coeffs):
    """``safeCoefficientList``: ascending coeff list, ``[0]`` for empty.

    A polynomial in ``x`` is carried as an ascending ``mpmath`` coefficient list
    (index 0 = constant term), reusing the character module's convention. The
    empty list maps to ``[mpf(0)]`` so a zero polynomial still emits one
    ``<coeff>`` (``CoefficientList`` returns ``{0}`` for ``0``).
    """
    if len(poly_coeffs) == 0:
        return [mpf(0)]
    return [mpf(ci) for ci in poly_coeffs]


def poly_degree(poly_coeffs):
    """Degree of an ascending coeff list: index of the highest nonzero term.

    Trailing exact zeros are ignored. An all-zero or empty list has degree 0
    (it is the constant ``0``), matching ``Exponent[0, x] -> 0`` after the
    ``safeCoefficientList`` normalization.
    """
    c = safe_coeff_list(poly_coeffs)
    deg = 0
    for i, ci in enumerate(c):
        if ci != 0:
            deg = i
    return deg


# ---------------------------------------------------------------------------
# reshuffleWithNormalization
# ---------------------------------------------------------------------------

def reshuffle_with_normalization(normalization, v):
    """``reshuffleWithNormalization[normalization, v]``.

    Folds the normalization constraint ``normalization . a == 1`` into the first
    slot so ``a . v == First[v'] + Rest[a] . Rest[v']``:

        j     = argmax_i |normalization[i]|        # pivot (first-max)
        const = v[j] / normalization[j]
        w     = v - const * normalization          # same length
        w     = delete element j from w            # length len(v)-1
        return [const] + w                         # length len(v)

    Applied identically to the objective and every polynomial vector. The pivot
    ``j`` (first-max via :func:`max_index_by`) must match
    ``restoreCoefficients`` (which lives with the solver). The components of
    ``v`` may be scalars (``pols1``) or polynomials-as-coeff-lists (``pols2`` /
    objective entries are scalars); arithmetic is delegated to ``_sub`` /
    ``_scale`` so both work.
    """
    j = max_index_by(normalization, key=lambda nval: abs(nval))
    const = _scale(v[j], mpf(1) / normalization[j])  # v[j] / normalization[j]
    w = [_sub(v[i], _scale(normalization[i], const)) for i in range(len(v))]
    del w[j]
    return [const] + w


def _is_poly(z):
    """A coeff-list component is a Python list/tuple; a scalar is anything else."""
    return isinstance(z, (list, tuple))


def _scale(z, alpha):
    """``z * alpha`` where either factor may be a scalar or coeff list.

    A scalar times a coeff list scales every coefficient; a coeff list times a
    coeff list never occurs here (``normalization`` is scalar and ``const`` is
    the component type of ``v``, so at most one factor is a polynomial).
    """
    if _is_poly(z) and _is_poly(alpha):
        raise TypeError("poly * poly not supported in reshuffle scaling")
    if _is_poly(z):
        return [ci * alpha for ci in z]
    if _is_poly(alpha):
        return [z * ci for ci in alpha]
    return z * alpha


def _sub(a, b):
    """``a - b`` where each is a scalar or an ascending coeff list (auto-padded)."""
    if not _is_poly(a) and not _is_poly(b):
        return a - b
    ac = list(a) if _is_poly(a) else [a]
    bc = list(b) if _is_poly(b) else [b]
    n = max(len(ac), len(bc))
    ac += [mpf(0)] * (n - len(ac))
    bc += [mpf(0)] * (n - len(bc))
    return [ac[i] - bc[i] for i in range(n)]


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PositiveMatrixWithPrefactor:
    """``PositiveMatrixWithPrefactor[prefactor, matrix]``.

    ``prefactor`` is a :class:`DampedRational` (``pols2``) or the scalar ``1``
    (``pols1``). ``matrix`` is a ``1x1`` list-of-lists whose single entry
    ``matrix[0][0]`` is a polynomial **vector**:

    - ``pols2``: a list of per-component polynomials in ``x``, each an ascending
      ``mpmath`` coeff list (``Base`` evaluated at ``Delta = x + gap``, ``s = ii``).
    - ``pols1``: a list of scalar values (``Baseexact`` evaluated at a fixed
      ``Delta = scalartb[k]``, ``s = 0``); the damping already lives inside.
    """

    prefactor: object
    matrix: list


@dataclass
class SDP:
    """``SDP[objective, normalization, pvms]``.

    ``objective`` and ``normalization`` are length-``len(Base0)`` ``mpmath``
    vectors; ``pvms`` is the list of :class:`PositiveMatrixWithPrefactor`
    (``pols1 + pols2``). This is everything ``write_bootstrap_sdp`` needs and
    enough for a solver to sample positivity and read the functional back.
    """

    objective: list
    normalization: list
    pvms: list


# ---------------------------------------------------------------------------
# shared bound-driver scaffolding
# ---------------------------------------------------------------------------

def _build_pols2(pairs, c, gaplist, lmax, n_count_slots=1):
    """Per-spin ``pols2`` positivity constraints, shared by every bound driver.

    For each spin ``ii in 0..lmax`` the prefactor is
    ``DampedRational[2, {}, B, x + gaplist[ii]]`` (``damped_shift`` of
    ``DampedRational(2, (), B)``: const ``2 B**gaplist[ii]``, empty poles) and the
    matrix entry is the polynomial vector ``[0]*n_count_slots + [Com(a1,a2,c)
    /. {Delta -> x + gaplist[ii], s -> ii}]`` in ``x`` (one zero bookkeeping
    slot per counted window; every driver has exactly one).

    The plain / spin / interval drivers differ ONLY in how ``gaplist`` is built
    (the second-gap convention), so this loop is factored here while each driver
    keeps its own ``gaplist`` inline.
    """
    pols2 = []
    for ii in range(0, lmax + 1):
        g = gaplist[ii]
        prefactor = damped_shift(DampedRational(mpf(2), (), DAMP_BASE), g)
        vec = ([[mpf(0)] for _ in range(n_count_slots)]
               + [com_poly_in_x(a1, a2, c, g, ii) for (a1, a2) in pairs])
        pols2.append(PositiveMatrixWithPrefactor(prefactor, [[vec]]))
    return pols2


def build_bound_sdp(c, gap, Da, NN, Lmax, lmax, *, sign):
    """Build the shared bound SDP (the common scaffolding).

    Reproduces the ``base/baseexact/base0 -> Base/Baseexact/Base0``, ``scalartb``,
    ``gaplist``, ``pols1``, ``pols2`` scaffolding shared by every bound driver,
    and wires the objective/normalization:

    - ``objective = Base0`` (``[0] + [Com0(a1,a2,c) ...]``).
    - ``normalization`` pins the vacuum slot: ``sign > 0`` -> ``(+1, 0, ..., 0)``
      (``integerlowerbound``); ``sign < 0`` -> ``(-1, 0, ..., 0)``
      (``integerupperbound``).

    ``scalartb`` is the ``NN+1``-point inclusive ``Delta`` grid on ``[gap, Da]``
    (``Table[ii, {ii, gap, Da, (Da-gap)/NN}]``). ``gaplist`` is
    ``[Da] + [max(gap, l) for l in 1..lmax]`` (0-indexed here: ``gaplist[0] = Da``
    is the spin-0 effective gap; ``gaplist[l] = max(gap, l)`` for spin ``l``).
    For each spin ``ii in 0..lmax`` ``pols2`` uses ``gaplist[ii]`` (the 1-indexed
    ``gaplist[[ii+1]]``).

    The integrality / feasibility / interval / spin variants (``bounds.py``) build
    on this scaffolding.

    Numerics are ``mpmath``; all rationals (``c``, ``gap``, ``Da``) should be
    passed as exact ``mpf`` ratios (e.g. ``mpf(10001)/10000``) to match the
    exact ``Rationalize`` inputs.
    """
    c = mpf(c)
    gap = mpf(gap)
    Da = mpf(Da)

    pairs = tb(Lmax)

    # base/baseexact/base0 and their [0]/[1]-prepended Base/Baseexact/Base0.
    # Base   = [0] + [Com(a1,a2,c) polynomials in Delta]            (pols2, s per spin)
    # Baseexact = ([1] + [Comwithprefactor...]) /. s->0             (pols1, numeric at a point)
    # Base0  = [0] + [Com0(a1,a2,c) numbers]                        (objective)
    base0 = [Com0(a1, a2, c) for (a1, a2) in pairs]
    Base0 = [mpf(0)] + base0

    # scalartb = NN+1 Delta-samples on [gap, Da] (inclusive endpoints).
    step = (Da - gap) / NN
    scalartb = [gap + k * step for k in range(NN + 1)]

    # gaplist[0] = Da (spin-0 effective gap); gaplist[l] = max(gap, l).
    gaplist = [Da] + [max(gap, mpf(l)) for l in range(1, lmax + 1)]

    # pols1: one 1x1 scalar positivity per Delta sample; prefactor = 1 (damping
    # already folded into Baseexact = Comwithprefactor). The single matrix entry
    # is the length-len(Base0) scalar vector  [1] + [Comwithprefactor at Delta].
    pols1 = []
    for delta in scalartb:
        vec = [mpf(1)] + [comwithprefactor_at(a1, a2, c, delta, 0)
                          for (a1, a2) in pairs]
        pols1.append(PositiveMatrixWithPrefactor(1, [[vec]]))

    # pols2: one positivity constraint per spin (shared _build_pols2). The
    # gaplist above (spin-0 -> Da; spin-l -> max(gap, l)) is the only per-driver
    # difference.
    pols2 = _build_pols2(pairs, c, gaplist, lmax)

    objective = list(Base0)
    e0 = mpf(1) if sign > 0 else mpf(-1)
    normalization = [e0] + [mpf(0)] * (len(Base0) - 1)

    return SDP(objective, normalization, pols1 + pols2)


def build_bound_sdp_spin(c, gap, Da, LL, NN, Lmax, lmax, *, sign):
    """Legacy CONDITIONAL spin bound SDP — ``integer{lower,upper}boundspin``.

    This is the retired second-gap (conditional) builder: it ISOLATES the target
    spin-``LL`` operator below an ASSUMED second gap ``Da`` in its own spin
    sector. That assumption makes it a conditional method, distinct from the
    robust / unconditional interval and window methods (``build_interval_sdp`` /
    ``build_window_bound_sdp``) that are the method of record. It is kept for
    reference; nothing in this repository's certified path uses it.

    Differences from :func:`build_bound_sdp` (the plain path):

    - **spin substitution** ``s -> LL`` in the vacuum-functional (``pols1``)
      sector: ``Baseexact = ([1] + baseexact) /. s -> LL`` (the plain path hard-
      codes ``s -> 0``). The per-spin ``pols2`` entries still use their own
      ``s = ii`` (the spin index), exactly as ``Table[..., s->ii]``.
    - **second gap** ``gaplist = [gap] + [If[l == LL, Da, l] for l in 1..lmax]``:
      the spin-0 effective gap is the *scalar* gap ``gap`` (NOT ``Da`` — that is
      the plain driver); the target spin ``LL`` gets the assumed second gap
      ``Da``; every other spin ``l`` gets its twist/unitarity gap ``l``.
    - **spintb** ``= Table[ii, {ii, LL, Da, (Da-LL)/NN}]`` (the ``NN+1``-point
      ``Delta`` grid from ``LL`` to ``Da``) replaces ``scalartb``; ``pols1``
      evaluates ``Baseexact /. Delta -> spintb[k]``.

    ``sign > 0`` -> ``integerlowerboundspin`` (``norm = +e0``, return
    ``Zmin = Vec1.obj``); ``sign < 0`` -> ``integerupperboundspin`` (``norm =
    -e0``, ``Zmin = -Vec1.obj``). The objective is ``Base0 = [0] + base0``.
    """
    c = mpf(c)
    gap = mpf(gap)
    Da = mpf(Da)

    pairs = tb(Lmax)

    base0 = [Com0(a1, a2, c) for (a1, a2) in pairs]
    Base0 = [mpf(0)] + base0

    # spintb = NN+1 Delta-samples on [LL, Da] (inclusive endpoints).
    step = (Da - mpf(LL)) / NN
    spintb = [mpf(LL) + k * step for k in range(NN + 1)]

    # Second-gap gaplist: spin-0 -> scalar gap; spin-LL -> Da; spin-l -> l.
    gaplist = [gap] + [(Da if l == LL else mpf(l)) for l in range(1, lmax + 1)]

    # pols1: spin substitution s -> LL inside Comwithprefactor.
    pols1 = []
    for delta in spintb:
        vec = [mpf(1)] + [comwithprefactor_at(a1, a2, c, delta, LL)
                          for (a1, a2) in pairs]
        pols1.append(PositiveMatrixWithPrefactor(1, [[vec]]))

    # pols2: per-spin positivity; each sector uses its own s = ii (shared
    # _build_pols2). The second-gap gaplist above is the per-driver difference.
    pols2 = _build_pols2(pairs, c, gaplist, lmax)

    objective = list(Base0)
    e0 = mpf(1) if sign > 0 else mpf(-1)
    normalization = [e0] + [mpf(0)] * (len(Base0) - 1)

    return SDP(objective, normalization, pols1 + pols2)


def build_bound_sdp_spin_pair(c, gap, Da, LL, NN, Lmax, lmax):
    """Both signs of :func:`build_bound_sdp_spin` from ONE assembly pass.

    ``build_bound_sdp_spin(..., sign=+1)`` and ``sign=-1`` differ ONLY in the
    normalization head (``e0 = +1`` vs ``-1``, zeros elsewhere) -- the objective
    and ``pvms`` (the expensive ``pols1``/``pols2`` assembly) are sign-independent.
    So build the ``sign=+1`` SDP once and derive the ``sign=-1`` twin by swapping
    in the ``-1`` normalization head, instead of re-running the whole assembly.

    The two returned :class:`SDP`\\ s SHARE the same ``pvms`` list object (and
    each gets its own shallow ``objective``/``normalization`` list copies);
    every consumer treats ``pvms`` as read-only, and its leaf values are
    immutable ``mpf``/``DampedRational`` instances, so the aliasing is safe.
    The minus normalization is built with the exact same
    ``[mpf(-1)] + [mpf(0)] * (n-1)`` expression the ``sign < 0`` branch above
    uses, so this pair is bit-identical to two independent
    ``build_bound_sdp_spin(..., sign=+1)`` / ``build_bound_sdp_spin(..., sign=-1)``
    calls.
    """
    plus = build_bound_sdp_spin(c, gap, Da, LL, NN, Lmax, lmax, sign=+1)
    minus = SDP(list(plus.objective),
                [mpf(-1)] + [mpf(0)] * (len(plus.objective) - 1),
                plus.pvms)
    return plus, minus


def _binomial_row(m):
    """Ascending coeff list of ``(1+u)^m``, exact integers as ``mpf``.

    Shared helper for :func:`projective_halfline_coeffs` (Pascal's-triangle row
    built by repeated ``(1+u)`` multiplication).
    """
    row = [mpf(1)]
    for _ in range(m):
        nxt = [mpf(1)] + [row[i] + row[i + 1] for i in range(len(row) - 1)] + [mpf(1)]
        row = nxt
    return row


def _poly_add(a, b):
    """Ascending coeff-list addition, auto-padded to the longer length."""
    n = max(len(a), len(b))
    a = list(a) + [mpf(0)] * (n - len(a))
    b = list(b) + [mpf(0)] * (n - len(b))
    return [a[i] + b[i] for i in range(n)]


def projective_halfline_coeffs(coeffs, width, block_degree):
    """``(1+u)^D * p(W*u/(1+u))`` as an ascending coeff list in ``u`` (exact).

    Maps a polynomial ``p(t)`` on the bounded slab variable ``t in [0, W)`` to
    an equivalent half-line polynomial ``q(u)`` on ``u in [0, inf)`` via the
    projective (Moebius) substitution ``t = W*u/(1+u)`` cleared of denominators
    by ``(1+u)^D``:

        q(u) = (1+u)^D * p(W*u/(1+u)) = sum_j c_j W^j u^j (1+u)^(D-j)

    Since ``(1+u)^D > 0`` everywhere, ``q(u) >= 0`` for all ``u >= 0`` iff
    ``p(t) >= 0`` for all ``t in [0, W)``, with the closed endpoint ``t = W``
    recovered in the ``u -> inf`` limit (``q``'s leading coefficient equals
    ``p(W)``). ``coeffs`` is the ascending coeff list of ``p(t)``; ``width =
    W``; ``block_degree = D`` must dominate every component's degree (all
    components of one PVM block share the same ``D`` so the constraint stays
    linear in ``alpha``). See :func:`build_interval_sdp`'s ``gap < Da`` slab
    block.
    """
    coeffs = safe_coeff_list(coeffs)
    if poly_degree(coeffs) > block_degree:
        raise ValueError("block_degree must dominate every component degree")
    out = [mpf(0)]
    wj = mpf(1)  # W^j
    for j, cj in enumerate(coeffs):
        if j > block_degree:
            break
        if cj != 0:
            # c_j * W^j * u^j * (1+u)^(D-j)
            term = [mpf(0)] * j + [cj * wj * b for b in _binomial_row(block_degree - j)]
            out = _poly_add(out, term)
        wj *= width
    return out


def build_interval_sdp(c, gap, Da, Db, NN, Lmax, lmax, *, sign):
    """Interval bound SDP — ``integer{lower,upper}boundinterval``.

    The robust interval method: a genuine two-sided bracket on the operator
    coefficient **integrated** over ``[Da, Db]`` with NO second-gap assumption.
    Differences from :func:`build_bound_sdp`:

    - **Indicator vacuum slot** replaces the constant ``1`` vacuum slot of
      ``Baseexact``: ``VAC_SLOT = HeavisideTheta[Delta-Da] HeavisideTheta[Db-Delta]``
      with ``HeavisideTheta[0] -> 1`` forced (the indicator of the closed window
      ``[Da, Db]``; ``/. s -> 0``).
    - ``scalartb = Union[grid(gap, Da, NN), grid(Da, Db, NN)]`` samples BOTH the
      pre-window ``[gap, Da]`` and the indicator window ``[Da, Db]``.
    - ``gaplist = [Db] + [max(gap, l) for l in 1..lmax]`` (no second gap). The
      counted spin-0 continuum starts at ``Db`` (ABOVE the window), not at the
      scalar ``gap``: inside ``[Da, Db]`` the spin-0 operators are counted by the
      pols1 indicator (the +/-1 slack), and the whole ``[gap, Db]`` is already
      sampled discretely by ``scalartb``, so re-imposing a plain spin-0 sum of
      squares over the window would kill the lower bound. Starting spin-0 at
      ``Db`` is the correction that reproduces the published lower bound ``A``.
    - **``gap < Da`` split continuum.** When the scalar gap sits strictly below
      the indicator window, the slab ``[gap, Da)`` carries no continuum
      positivity of its own (only the discrete ``scalartb`` samples); one extra
      spin-0 PVM block enforces genuine continuum positivity there, mapped to
      the half-line by :func:`projective_halfline_coeffs` (the projective
      substitution ``t = (Da-gap)*u/(1+u)``, exact, zero backend changes — see
      that function's docstring). ``gap == Da`` (every anchor) emits the SDP
      unchanged (no slab block appended); ``gap > Da`` is degenerate
      (``grid1`` step < 0) and raises ``ValueError``. The c=1 marginal bracket
      (``gap=1/2``, ``Da=2``, ``Db=2.05``) certifies ``B = 3.0000`` against the
      generic-Virasoro-primary target 3.

    ``sign > 0`` -> ``integerlowerboundinterval`` (``norm = +e0``,
    ``Zmin = Vec1.obj``); ``sign < 0`` -> ``integerupperboundinterval``
    (``norm = -e0``, ``Zmin = -Vec1.obj``).
    """
    c = mpf(c)
    gap = mpf(gap)
    Da = mpf(Da)
    Db = mpf(Db)

    # gap > Da is degenerate (grid1 step < 0); fail loudly rather than build a
    # nonsensical descending grid.
    if gap > Da:
        raise ValueError(f"build_interval_sdp requires gap <= Da (got gap={gap}, "
                         f"Da={Da})")

    pairs = tb(Lmax)

    base0 = [Com0(a1, a2, c) for (a1, a2) in pairs]
    Base0 = [mpf(0)] + base0

    # scalartb = Union of the two grids (sorted, de-duplicated like Union).
    step1 = (Da - gap) / NN
    grid1 = [gap + k * step1 for k in range(NN + 1)]
    step2 = (Db - Da) / NN
    grid2 = [Da + k * step2 for k in range(NN + 1)]
    merged = grid1 + grid2
    scalartb = []
    for d in merged:
        if not any(abs(d - e) <= mpf(10) ** -(PREC_DIGITS // 2) for e in scalartb):
            scalartb.append(d)
    scalartb.sort()

    # gaplist: the COUNTED spin (s=0) continuum positivity must start ABOVE the
    # window [Da, Db]. Inside the window the spin-0 operators are counted via the
    # pols1 indicator, whose constraint is beta_0*1 + alpha.Fvec >= 0: the +/-1
    # indicator slack is what makes the bracket two-sided. A plain spin-0 sum of
    # squares started at `gap` would re-impose alpha.Fvec_{Delta,0} >= 0 across
    # [Da, Db], destroying the -1 slack the LOWER bound (beta_0 = +1) needs -> A
    # collapses to ~0 (the upper, beta_0 = -1, is unaffected: its indicator
    # forces alpha.Fvec >= +1, so the continuum is redundant). The whole [gap, Db]
    # is already covered DISCRETELY by the pols1 grid (scalartb = grid1 U grid2),
    # so the spin-0 continuum only needs (Db, inf): start it at Db. Spin-l >= 1
    # have indicator delta_{s,l} = 0 everywhere, so keep max(gap, l).
    gaplist = [Db] + [max(gap, mpf(l)) for l in range(1, lmax + 1)]

    # pols1: indicator VAC_SLOT (Theta[0]->1) replaces the constant 1.
    pols1 = []
    for delta in scalartb:
        indicator = mpf(1) if (delta >= Da and delta <= Db) else mpf(0)
        vec = [indicator] + [comwithprefactor_at(a1, a2, c, delta, 0)
                             for (a1, a2) in pairs]
        pols1.append(PositiveMatrixWithPrefactor(1, [[vec]]))

    # pols2: per-spin positivity (shared _build_pols2). The interval gaplist
    # above (spin-0 -> scalar gap; no second gap) is the per-driver difference.
    pols2 = _build_pols2(pairs, c, gaplist, lmax)

    # gap < Da: append ONE extra spin-0 block enforcing genuine continuum
    # positivity on the bounded slab [gap, Da), which the discrete scalartb
    # samples alone do not provide. Component polynomials are Com at
    # (Delta = t + gap, s = 0) — i.e. a half-line block anchored at `gap` —
    # then mapped u -> t projectively onto the slab width W = Da - gap via
    # projective_halfline_coeffs. gap == Da (W == 0, every anchor)
    # skips this branch entirely, so that path is untouched by construction.
    if gap < Da:
        width = Da - gap
        slab_polys = [com_poly_in_x(a1, a2, c, gap, 0) for (a1, a2) in pairs]
        block_degree = max(poly_degree(p) for p in slab_polys)
        vec = [[mpf(0)]] + [projective_halfline_coeffs(p, width, block_degree)
                            for p in slab_polys]
        slab_prefactor = damped_shift(DampedRational(mpf(2), (), DAMP_BASE), gap)
        pols2 = pols2 + [PositiveMatrixWithPrefactor(slab_prefactor, [[vec]])]

    objective = list(Base0)
    e0 = mpf(1) if sign > 0 else mpf(-1)
    normalization = [e0] + [mpf(0)] * (len(Base0) - 1)

    return SDP(objective, normalization, pols1 + pols2)


def build_interval_sdp_pair(c, gap, Da, Db, NN, Lmax, lmax):
    """Both signs of :func:`build_interval_sdp` from ONE assembly pass.

    ``build_interval_sdp(..., sign=+1)`` and ``sign=-1`` differ ONLY in the
    normalization head (``e0 = +1`` vs ``-1``, zeros elsewhere) -- the objective
    and ``pvms`` (the expensive ``pols1``/``pols2``/split-continuum-slab
    assembly) are sign-independent. So build the ``sign=+1`` SDP once and derive
    the ``sign=-1`` twin by swapping in the ``-1`` normalization head, instead of
    re-running the whole assembly.

    The two returned :class:`SDP`\\ s SHARE the same ``pvms`` list object (and
    each gets its own shallow ``objective``/``normalization`` list copies);
    every consumer treats ``pvms`` as read-only, and its leaf values are
    immutable ``mpf``/``DampedRational`` instances, so the aliasing is safe.
    The minus normalization is built with the exact same
    ``[mpf(-1)] + [mpf(0)] * (n-1)`` expression the ``sign < 0`` branch above
    uses, so this pair is bit-identical to two independent
    ``build_interval_sdp(..., sign=+1)`` / ``build_interval_sdp(..., sign=-1)``
    calls.
    """
    plus = build_interval_sdp(c, gap, Da, Db, NN, Lmax, lmax, sign=+1)
    minus = SDP(list(plus.objective),
                [mpf(-1)] + [mpf(0)] * (len(plus.objective) - 1),
                plus.pvms)
    return plus, minus


def build_fes_sdp(Ndel, c, gap, Da, NN, Lmax, lmax):
    """Feasibility SDP — ``integerfes``.

    Built on the plain :func:`build_bound_sdp` scaffolding but rewired for a pure
    feasibility test of "does a vacuum coefficient ``>= Ndel`` exist?":

    - ``VAC0_SLOT = -Ndel`` (the objective/normalization vacuum slot, replacing 0).
    - ``normalization = Base0`` (with ``Base0[0] = -Ndel``).
    - ``objective = 0 * normalization`` (pure feasibility; no bound to maximise).

    The ``pols1`` / ``pols2`` positivity (``scalartb``, ``gaplist`` of the plain
    driver) are unchanged. A dual-feasible solve excludes the assumption; a
    primal-feasible solve admits a vacuum coefficient ``>= Ndel``.
    """
    sdp = build_bound_sdp(c, gap, Da, NN, Lmax, lmax, sign=+1)
    # VAC0_SLOT = -Ndel in Base0; norm = Base0; obj = 0*norm.
    base0_full = list(sdp.objective)
    base0_full[0] = -mpf(Ndel)
    normalization = list(base0_full)
    objective = [mpf(0)] * len(base0_full)
    return SDP(objective, normalization, sdp.pvms)


# ---------------------------------------------------------------------------
# Explicit-edge window builders (discrete-sampling toolkit)
# ---------------------------------------------------------------------------

def window_sample_points(a, b, NN, extra=()):
    """``intervalSamplePoints[a, b, NN]`` (+ extras) — the explicit-edge grid.

    **NN points** with step ``(b-a)/(NN-1)``, endpoints included — NOT the
    NN+1-point / step ``(b-a)/NN`` convention of the plain builders
    (``scalartb``/``spintb``). The two grids differ everywhere except the
    endpoints (the known "NN-grid seam"); the toolkit ports use this one.

    ``extra`` (the resampling ``morex``) is merged in; the merged list is
    sorted and de-duplicated (the source Joins without dedup — duplicate
    degree-0 constraints are harmless, dropping them is the only deviation).
    """
    a = mpf(a)
    b = mpf(b)
    if NN < 2:
        raise ValueError(f"window_sample_points needs NN >= 2 (got {NN})")
    step = (b - a) / (NN - 1)
    merged = [a + k * step for k in range(NN)] + [mpf(p) for p in extra]
    tol = mpf(10) ** -(PREC_DIGITS // 2)
    pts = []
    for d in sorted(merged):
        if not pts or abs(d - pts[-1]) > tol:
            pts.append(d)
    return pts


def _window_sector_gap(spin, gap):
    """The sector gap a window's left edge must sit at."""
    return mpf(gap) if spin == 0 else mpf(spin)


def _check_window_anchored(spin, a, gap, allow_conditional):
    """``a`` must be the sector gap: below-window operators are assumed absent.

    A window floating above the true sector gap leaves ``[sector_gap, a)``
    unconstrained — a hidden second-gap assumption that biases toward exclusion.
    Fail loudly unless the caller opts in (``allow_conditional=True`` documents
    the conditionality).
    """
    sector = _window_sector_gap(spin, gap)
    if mpf(a) != sector and not allow_conditional:
        raise ValueError(
            f"window (spin={spin}, a={mpf(a)}) is not anchored at the sector gap "
            f"{sector}; [gap, a) would be silently assumed empty (a second-gap "
            "assumption). Pass allow_conditional=True to accept it explicitly.")


def _prep_window_bound(c, gap, spin, a, b, Lmax, lmax, *, allow_conditional=False):
    """Round- and sign-independent blocks of :func:`build_window_bound_sdp`.

    Returns ``(pairs, Base0, gaplist, pols2)`` — the parts that depend only on
    ``(c, gap, spin, a, b, Lmax, lmax)`` and NOT on the resampling
    ``extra_points`` nor on ``sign``. A resampling loop (``bound_sab``) builds
    these once and passes them back via ``build_window_bound_sdp(..., prep=...)``
    so the dominant per-round ``_build_pols2`` rebuild happens once rather than
    once per round. Same code path and same loop-invariant inputs as the inline
    build, so reusing the result is bit-identical.
    """
    c = mpf(c)
    gap = mpf(gap)
    a = mpf(a)
    b = mpf(b)
    _check_window_anchored(spin, a, gap, allow_conditional)
    if gap > 1:
        import warnings
        warnings.warn("gap > 1: the unitarity gaplist (spin-l continuum at l) "
                      "differs from the interval builder's max(gap, l) here")
    pairs = tb(Lmax)
    base0 = [Com0(a1, a2, c) for (a1, a2) in pairs]
    Base0 = [mpf(0)] + base0
    # counted spin -> b; scalar (no window) -> gap; else unitarity.
    gaplist = [(b if ii == spin else (gap if ii == 0 else mpf(ii)))
               for ii in range(lmax + 1)]
    pols2 = _build_pols2(pairs, c, gaplist, lmax)
    return pairs, Base0, gaplist, pols2


def build_window_bound_sdp(c, gap, spin, a, b, NNsample, Lmax, lmax, *, sign,
                           extra_points=(), allow_conditional=False, prep=None):
    """Window density-bound SDP — ``boundSAB`` inner problem.

    Two-sided bound on the operator count of spin sector ``spin`` integrated
    over the window ``[a, b]`` (explicit edges), discrete-sampling form:

    - ``pols1``: one degree-0 constraint per sample point of
      :func:`window_sample_points` (grid + resampling ``extra_points``); the
      vector is ``[1] + [Comwithprefactor /. {s -> spin, Delta -> samp}]``
      (slot 0 = the count bookkeeping slot).
    - ``pols2`` gaplist: the counted sector's continuum starts at the
      window TOP ``b`` (the same continuum-above-the-window convention as
      ``build_interval_sdp``'s Db correction); spin-0 without window -> ``gap``;
      other spins -> bare unitarity ``l`` (NOTE: the toolkit convention, NOT the
      interval builder's ``max(gap, l)``; identical whenever ``gap <= 1``, which
      holds everywhere in scope).
    - ``objective = Base0``; ``normalization = +e0`` (``sign > 0``, lower
      bound, ``Zmin = +alpha.obj``) / ``-e0`` (``sign < 0``, upper bound,
      ``Zmin = -alpha.obj``) — the existing solver ``sense`` mapping.

    ``a`` must equal the sector gap (``gap`` for spin 0, ``spin`` for spin >= 1)
    unless ``allow_conditional=True`` (see :func:`_check_window_anchored`).

    ``prep``: optional precomputed ``(pairs, Base0, gaplist, pols2)`` from
    :func:`_prep_window_bound` for the SAME ``(c, gap, spin, a, b, Lmax,
    lmax)``. When supplied, the round-independent blocks are reused instead of
    rebuilt (``bound_sab``'s resampling loop passes it every round); the result
    is bit-identical to ``prep=None``, which rebuilds them as before.
    """
    c = mpf(c)
    gap = mpf(gap)
    a = mpf(a)
    b = mpf(b)
    if prep is None:
        prep = _prep_window_bound(c, gap, spin, a, b, Lmax, lmax,
                                  allow_conditional=allow_conditional)
    pairs, Base0, _gaplist, pols2 = prep

    pols1 = []
    for delta in window_sample_points(a, b, NNsample, extra_points):
        vec = [mpf(1)] + [comwithprefactor_at(a1, a2, c, delta, spin)
                          for (a1, a2) in pairs]
        pols1.append(PositiveMatrixWithPrefactor(1, [[vec]]))

    objective = list(Base0)
    e0 = mpf(1) if sign > 0 else mpf(-1)
    normalization = [e0] + [mpf(0)] * (len(Base0) - 1)
    return SDP(objective, normalization, pols1 + pols2)


# ---------------------------------------------------------------------------
# Continuous-window builder (rigorous polynomial bounds on the window exponential)
# ---------------------------------------------------------------------------

def exp_poly_bounds(width, rel_tol=mpf(10) ** -12, max_degree=400):
    """Rigorous polynomial bounds ``L(t) <= exp(2 pi t) <= U(t)`` on ``t in [0, width]``.

    ``p_n`` interpolates ``f(t) = exp(2 pi t)`` at the ``n+1`` Chebyshev nodes of
    ``[0, w]``. The interpolation remainder is at most
    ``max|f^(n+1)| (w/2)^(n+1) / (2^n (n+1)!)`` with
    ``max|f^(n+1)| = (2 pi)^(n+1) e^{2 pi w}``, i.e.
    ``R = 2 e^{2 pi w} (pi w / 2)^(n+1) / (n+1)!``, so
    ``p_n - R <= f <= p_n + R`` on the interval. ``n`` is the smallest degree
    with ``R <= rel_tol`` (``f >= 1`` there, so ``R`` is also the relative gap);
    ``ValueError`` if that exceeds ``max_degree``. A margin
    ``eps = 10^-(dps/2) (e^{2 pi w} + sum_i |p_i| w^i)``, far above the
    ``dps``-digit rounding of the coefficients, is added on both sides.

    Returns ``(L, U, n, R)``: ``L = p_n - R - eps`` and ``U = p_n + R + eps`` as
    ascending coeff lists in ``t``, the degree ``n`` and the remainder ``R``.
    """
    w = mpf(width)
    ekw = mp.e ** (2 * mp.pi * w)
    x = mp.pi * w / 2
    n = 0
    R = 2 * ekw * x
    while R > rel_tol:
        n += 1
        if n > max_degree:
            raise ValueError(f"exp_poly_bounds: degree > {max_degree} for width {w}")
        R = R * x / (n + 1)  # 2 e^{2 pi w} x^{n+1} / (n+1)!
    N = n + 1
    # Chebyshev coefficients from the node values (t = w (s+1)/2).
    theta = [mp.pi * (i + mpf(1) / 2) / N for i in range(N)]
    vals = [mp.e ** (mp.pi * w * (mp.cos(th) + 1)) for th in theta]
    cheb = [2 * mp.fsum(v * mp.cos(j * th) for v, th in zip(vals, theta)) / N
            for j in range(N)]
    cheb[0] /= 2
    # sum_j cheb_j T_j(s) as ascending coeffs in s, then s = (2/w) t - 1.
    Ts = [[mpf(1)], [mpf(0), mpf(1)]]
    while len(Ts) < N:
        nxt = [mpf(0)] + [2 * ci for ci in Ts[-1]]
        for i, ci in enumerate(Ts[-2]):
            nxt[i] -= ci
        Ts.append(nxt)
    p_s = [mpf(0)] * N
    for j in range(N):
        for i, ci in enumerate(Ts[j]):
            p_s[i] += cheb[j] * ci
    p = _poly_compose_affine(p_s, 2 / w, mpf(-1))
    eps = mpf(10) ** -(mp.dps // 2) * (ekw + mp.fsum(abs(ci) * w ** i
                                                    for i, ci in enumerate(p)))
    L = [p[0] - R - eps] + p[1:]
    U = [p[0] + R + eps] + p[1:]
    return L, U, n, R


def build_window_bound_sdp_continuous(c, gap, spin, a, b, Lmax, lmax, *, sign,
                                      max_block_degree=20,
                                      max_sub_width=mpf("0.5"),
                                      exp_rel_tol=mpf(10) ** -12,
                                      allow_conditional=False, prep=None):
    """Window density-bound SDP with the window constraint on the CONTINUUM.

    Same problem as :func:`build_window_bound_sdp` (same objective,
    normalization and ``pols2``), but the counted-window constraint

        a0 + 2 e^{-2pi(Delta-(c-1)/12)} P(Delta) >= 0   for ALL Delta in [a, b]

    is imposed on the whole window instead of at sample points (``a0`` is fixed
    to ``+1`` for ``sign > 0`` and ``-1`` for ``sign < 0`` by the
    normalization; ``P`` is the functional's ``Com`` action at ``spin``).
    Multiplying by ``e^{2pi(Delta-(c-1)/12)}/2 > 0`` gives ``a0 E + P >= 0`` with
    ``E(Delta) = e^{2pi(Delta-(c-1)/12)}/2``. ``E`` is replaced by a polynomial on
    the safe side, ``L <= E`` for ``a0 = +1`` and ``U >= E`` for ``a0 = -1``
    (:func:`exp_poly_bounds`), so the polynomial constraint implies the true
    one exactly; the price is a bracket at most a relative ``exp_rel_tol``
    weaker than the exact-exponential problem.

    The window is split into the fewest equal sub-windows ``K >= 2`` of width at
    most ``max_sub_width`` whose bound degree is at most ``max_block_degree``
    (few blocks keep the solver's matching-row count, and with it its cost,
    near the sampled builder's). Both caps are measured limits of the SDPA-GMP
    backend, whose matrix data is float64: at Lmax=10 degrees 12/16/20 agree to
    1e-11, 24 drifts and 32 fails; and a sub-window of width 1 (``E`` spanning
    a factor 535 inside one block) stalls at iteration 5 on the c=1.001, [1,5]
    window, which converges at width 1/3.
    Each sub-window is one positivity block on a bounded slab, mapped to the
    half-line by
    :func:`projective_halfline_coeffs` (as in the ``gap < Da`` slab of
    :func:`build_interval_sdp`), so the backends and
    :func:`dual.verify.certify_functional` handle it unchanged. The map sends
    ``u = 0`` to one end of the slab and ``u -> inf`` to the other, where the
    certificate tests the leading coefficient with no tolerance band. An
    extremal functional can touch zero at a window edge, so the last block is
    mapped in reverse: both window edges then sit at ``u = 0`` and every
    ``u -> inf`` end is an interior sub-window boundary (hence ``K >= 2``). The
    window blocks come AFTER ``pols2`` so the certificate's sequential spin
    labels ``0..lmax`` still name the spin sectors.
    """
    c = mpf(c)
    a = mpf(a)
    b = mpf(b)
    if prep is None:
        prep = _prep_window_bound(c, gap, spin, a, b, Lmax, lmax,
                                  allow_conditional=allow_conditional)
    pairs, Base0, _gaplist, pols2 = prep

    K = max(2, int(mp.ceil((b - a) / mpf(max_sub_width))))
    while True:
        try:
            L, U, _n, _R = exp_poly_bounds((b - a) / K, exp_rel_tol,
                                           max_degree=max_block_degree)
            break
        except ValueError:
            K += 1
    w = (b - a) / K
    surrogate = L if sign > 0 else U

    window = []
    for k in range(K):
        ak = a + k * w
        scale = mp.e ** (2 * mp.pi * (ak - (c - 1) / 12)) / 2
        polys = ([[scale * ci for ci in surrogate]]
                 + [com_poly_in_x(a1, a2, c, ak, spin) for (a1, a2) in pairs])
        edge = ak
        if k == K - 1:
            # reversed slab: t = w - s, so s = 0 is the window top b
            polys = [_poly_compose_affine(p, mpf(-1), w) for p in polys]
            edge = b
        block_degree = max(poly_degree(p) for p in polys)
        vec = [projective_halfline_coeffs(p, w, block_degree) for p in polys]
        prefactor = damped_shift(DampedRational(mpf(2), (), DAMP_BASE), edge)
        window.append(PositiveMatrixWithPrefactor(prefactor, [[vec]]))

    objective = list(Base0)
    e0 = mpf(1) if sign > 0 else mpf(-1)
    normalization = [e0] + [mpf(0)] * (len(Base0) - 1)
    return SDP(objective, normalization, pols2 + window)


def _prep_multi_window_fes(c, gap, windows, Lmax, lmax, *, allow_conditional=False):
    """Assignment- and round-independent blocks of
    :func:`build_multi_window_fes_sdp`: ``(pairs, base0, gaplist, pols2)``.

    These depend only on ``(c, gap, windows, Lmax, lmax)`` — NOT on the integer
    assignment ``counts`` (whose only footprint is the ``normalization`` leading
    slots) nor on the resampling ``extra_points``. So the joint assignment
    loop and ``feas_multi_sab``'s resample loop can build them once and reuse
    them via ``build_multi_window_fes_sdp(..., prep=...)``, avoiding the
    dominant ``_build_pols2`` rebuild. Also validates the counts-independent
    window invariants (one window per spin, anchoring). Bit-identical to letting
    ``build_multi_window_fes_sdp`` rebuild them.
    """
    c = mpf(c)
    gap = mpf(gap)
    K = len(windows)
    spins = [w[0] for w in windows]
    if len(set(spins)) != K:
        raise ValueError(f"duplicate window spins {spins}: at most one window per "
                         "spin sector (the spinB Association silently drops "
                         "duplicates; a split continuum is not implemented)")
    for (spin, a, b) in windows:
        _check_window_anchored(spin, a, gap, allow_conditional)
    pairs = tb(Lmax)
    base0 = [Com0(a1, a2, c) for (a1, a2) in pairs]
    # window spins -> window top b; scalar without window -> gap;
    # other spins -> unitarity.
    b_of_spin = {spin: mpf(b) for (spin, a, b) in windows}
    gaplist = [b_of_spin.get(ii, gap if ii == 0 else mpf(ii))
               for ii in range(lmax + 1)]
    pols2 = _build_pols2(pairs, c, gaplist, lmax, n_count_slots=K)
    return pairs, base0, gaplist, pols2


def build_multi_window_fes_sdp(counts, c, gap, windows, NNsample, Lmax, lmax, *,
                               extra_points=None, allow_conditional=False, prep=None):
    """Multi-window joint feasibility SDP — ``feasMultiSAB`` inner problem.

    Tests whether a modular-invariant spectrum exists with EXACTLY ``counts[j]``
    operators in each window ``windows[j] = (spin_j, a_j, b_j)``:

    - one count bookkeeping slot per window: the window-``j`` sample vector is
      ``insertZeros`` — ``[0]*K`` one-hot at ``j``, then the ``Comwithprefactor``
      derivative components at ``s = spin_j``;
    - ``pols2`` carry ``K`` leading zero slots; gaplist: window spins
      -> their window top ``b_j``; spin-0 without window -> ``gap``; else
      unitarity;
    - ``normalization = [-n_j ...] + base0``; ``objective = 0`` (pure
      feasibility — the K-slot generalization of :func:`build_fes_sdp`).

    At most ONE window per spin sector: the ``spinB`` Association silently
    keeps the last duplicate, which either leaves an inter-window
    hole unconstrained or re-imposes a plain continuum across the upper window
    (the count-collapse mode) — rejected here until a genuine split continuum
    exists. Each window must be anchored at its sector gap (see
    :func:`_check_window_anchored`).
    """
    c = mpf(c)
    gap = mpf(gap)
    K = len(windows)
    if len(counts) != K:
        raise ValueError(f"counts has length {len(counts)}, windows {K}")
    if prep is None:
        prep = _prep_multi_window_fes(c, gap, windows, Lmax, lmax,
                                      allow_conditional=allow_conditional)
    pairs, base0, _gaplist, pols2 = prep
    if extra_points is None:
        extra_points = [() for _ in range(K)]

    pols1 = []
    for j, (spin, a, b) in enumerate(windows):
        for delta in window_sample_points(a, b, NNsample, extra_points[j]):
            slots = [mpf(0)] * K
            slots[j] = mpf(1)
            vec = slots + [comwithprefactor_at(a1, a2, c, delta, spin)
                           for (a1, a2) in pairs]
            pols1.append(PositiveMatrixWithPrefactor(1, [[vec]]))

    normalization = [-mpf(n) for n in counts] + base0
    objective = [mpf(0)] * len(normalization)
    return SDP(objective, normalization, pols1 + pols2)


def restore_coefficients(normalization, y):
    """``restoreCoefficients`` — exact inverse of :func:`reshuffle_with_normalization`.

    Given the solver's reduced dual vector ``y`` (length ``N-1``; the pivot slot
    ``j = argmax_i |normalization[i]|`` was eliminated by the normalization fold),
    recover the full functional ``alpha`` (length ``N``, ``alpha . normalization
    == 1``): ``alpha_i = y_(i)`` for ``i != j`` and ``alpha_j = (1 - y .
    shortNorm) / normalization_j``. Used only as the ``y.txt`` fallback when a
    jump-terminated solve writes no ``z.txt``.
    """
    norm = [mpf(v) for v in normalization]
    if len(y) != len(norm) - 1:
        raise ValueError(f"y has length {len(y)}, expected {len(norm) - 1}")
    j = max_index_by(norm, key=lambda nval: abs(nval))
    short_norm = [norm[i] for i in range(len(norm)) if i != j]
    yj = (1 - mpmath.fsum(mpf(yi) * ni for yi, ni in zip(y, short_norm))) / norm[j]
    alpha = [mpf(yi) for yi in y]
    alpha.insert(j, yj)
    return alpha


# ---------------------------------------------------------------------------
# WriteBootstrapSDP (polynomial-vector-matrix XML)
# ---------------------------------------------------------------------------

def _elt_vector(tag, values):
    """``<tag><elt>nf(v)</elt>...</tag>`` for a flat numeric vector."""
    inner = "".join("<elt>{}</elt>".format(nf(v)) for v in values)
    return "<{0}>{1}</{0}>".format(tag, inner)


def _polynomial_xml(coeffs):
    """``<polynomial><coeff>..</coeff>...</polynomial>``, ascending powers."""
    c = safe_coeff_list(coeffs)
    inner = "".join("<coeff>{}</coeff>".format(nf(ci)) for ci in c)
    return "<polynomial>{}</polynomial>".format(inner)


def _polynomial_vector_xml(components):
    """``<polynomialVector>`` of one ``<polynomial>`` per (reshuffled) component.

    Each component is either a scalar (``pols1``) or an ascending coeff list
    (``pols2``); a scalar is emitted as a degree-0 polynomial ``{coeff}``.
    """
    polys = []
    for comp in components:
        polys.append(_polynomial_xml(comp if _is_poly(comp) else [comp]))
    return "<polynomialVector>{}</polynomialVector>".format("".join(polys))


def _pvm_degree(pvm):
    """``degree = Max[Exponent[m, x]]`` over the matrix entries.

    ``pols1`` entries are scalars -> degree 0; ``pols2`` entries are coeff
    lists -> their polynomial degree.
    """
    deg = 0
    for row in pvm.matrix:
        for pv in row:
            for comp in pv:
                if _is_poly(comp):
                    deg = max(deg, poly_degree(comp))
    return deg


def _polynomial_vector_matrix_xml(pvm, normalization):
    """One ``<polynomialVectorMatrix>``.

    ``degree = Max Exponent``; ``samplePoints = rescaledLaguerreSamplePoints(
    degree+1)``; ``sampleScalings = [prefactor(a)]`` (``1`` when ``prefactor==1``);
    ``bilinearBasis = orthogonalPolynomials(prefactor, floor(degree/2))``.
    Each ``<polynomialVector>`` is the reshuffled matrix entry.
    """
    m = pvm.matrix
    rows = len(m)
    cols = len(m[0])
    degree = _pvm_degree(pvm)

    sample_points = rescaled_laguerre_sample_points(degree + 1)

    prefactor = pvm.prefactor
    if isinstance(prefactor, DampedRational):
        sample_scalings = [prefactor(a) for a in sample_points]
        bilinear = orthogonal_polynomials(prefactor, degree // 2)
    else:
        # prefactor == 1 (pols1 scalar constraints): every scaling is 1, and the
        # matrix entry is always degree 0, so the bilinear basis is the constant
        # {1} (orthogonalPolynomials[1, 0]).
        sample_scalings = [mpf(1) for _ in sample_points]
        bilinear = [[mpf(1)]]

    elements = []
    for row in m:
        for pv in row:
            reshuffled = reshuffle_with_normalization(normalization, pv)
            elements.append(_polynomial_vector_xml(reshuffled))

    parts = [
        "<rows>{}</rows>".format(rows),
        "<cols>{}</cols>".format(cols),
        "<elements>{}</elements>".format("".join(elements)),
        _elt_vector("samplePoints", sample_points),
        _elt_vector("sampleScalings", sample_scalings),
        "<bilinearBasis>{}</bilinearBasis>".format(
            "".join(_polynomial_xml(p) for p in bilinear)),
    ]
    return "<polynomialVectorMatrix>{}</polynomialVectorMatrix>".format("".join(parts))


def write_bootstrap_sdp(sdp: SDP) -> str:
    """``WriteBootstrapSDP[SDP[objective, normalization, pvms]]`` -> XML.

    Emits the polynomial-vector-matrix XML tree. ``<objective>`` is the reshuffled
    objective vector; one ``<polynomialVectorMatrix>`` per ``pvm`` carries
    ``<rows>/<cols>/<elements>`` (each ``<polynomialVector>`` = the reshuffled
    matrix entry, ascending ``<coeff>`` per ``<polynomial>``),
    ``<samplePoints> = rescaledLaguerreSamplePoints(degree+1)``,
    ``<sampleScalings> = prefactor at each sample point``, and
    ``<bilinearBasis> = orthogonalPolynomials(prefactor, floor(degree/2))``.

    Every number is rendered with :func:`nf` (full-precision plain decimal, no
    ``e``/``E`` — the solver's arbitrary-precision parser rejects scientific
    notation).
    """
    obj = reshuffle_with_normalization(sdp.normalization, sdp.objective)
    objective_xml = _elt_vector("objective", obj)
    pvm_xml = "".join(
        _polynomial_vector_matrix_xml(pvm, sdp.normalization) for pvm in sdp.pvms
    )
    return ("<sdp>{0}<polynomialVectorMatrices>{1}"
            "</polynomialVectorMatrices></sdp>").format(objective_xml, pvm_xml)


# ---------------------------------------------------------------------------
# Split-continuum interval SDP -- thin wrappers (graduated to build_interval_sdp
# above; kept so existing callers of the old names keep working unchanged).
# ---------------------------------------------------------------------------

def build_interval_split_sdp(c, gap, Da, Db, NN, Lmax, lmax, *, sign):
    """Interval bound SDP allowing ``gap < Da`` (split spin-0 continuum).

    Thin delegation to :func:`build_interval_sdp`, which implements this
    directly. Kept for backward compatibility with existing callers/tests.
    """
    return build_interval_sdp(c, gap, Da, Db, NN, Lmax, lmax, sign=sign)


def interval_split_bracket(c, gap, Da, Db, NN, Lmax, lmax, *,
                           precision_bits=1024, epsilon=mpf(10) ** -15,
                           certify=True):
    """Two-sided bracket ``(A, B)`` on the ``[Da, Db]`` count with ``gap < Da`` OK.

    SDPA-GMP backend (fp64-input arbitrary-precision iterate -- DIAGNOSTIC
    grade unless re-solved through SDPB). Mirrors the core
    ``integer_{lower,upper}bound_interval`` sign conventions.
    """
    from .bounds import solve_bound_sdpa

    lo = solve_bound_sdpa(
        lambda: build_interval_split_sdp(c, gap, Da, Db, NN, Lmax, lmax, sign=+1),
        sense="min", precision_bits=precision_bits, epsilon=epsilon,
        certify=certify)
    hi = solve_bound_sdpa(
        lambda: build_interval_split_sdp(c, gap, Da, Db, NN, Lmax, lmax, sign=-1),
        sense="max", precision_bits=precision_bits, epsilon=epsilon,
        certify=certify)
    return lo, hi
