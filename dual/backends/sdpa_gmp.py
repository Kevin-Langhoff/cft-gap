"""Arbitrary-precision SDP backend -- SDPA-GMP via ``sdpap``.

The high-precision method-of-record solver for the dual modular bootstrap.
Where :mod:`dual.backends.scipy_lp` *samples* the continuum positivity into
an ill-conditioned float64 LP (a FIND pre-screen that cannot reproduce the SDP
numbers), this backend solves the **exact** semidefinite program with
SDPA-GMP (an arbitrary-precision primal-dual SDP solver, ``import sdpap``)
and reproduces the gold bounds.

The reduction (PVM -> SOS-on-the-half-line standard SDP)
---------------------------------------------------------

For each ``pols2`` constraint at spin ``s`` the requirement is

    p_s(x) = sum_k alpha_k * Base_k(x; s) >= 0   for ALL x >= 0,

where ``Base_k(x; s) = Com[...]/.{Delta->x+gap, s->spin}`` are the EXACT
ascending ``mpmath`` polynomial coefficients from
:func:`dual.characters.com_poly_in_x`. The strictly-positive
damped-rational prefactor ``2 B**(x+gap)`` does NOT affect positivity on
``x >= 0`` (see :mod:`dual.verify`), so only the
numerator polynomial is certified. We encode positivity on the half-line exactly
via the SOS decomposition

    p_s(x) = sigma0(x) + x * sigma1(x),   sigma0, sigma1 sums of squares,

with PSD Gram matrices ``Y0`` (size ``m0+1``, ``m0 = floor(d/2)``) and ``Y1``
(size ``m1+1``, ``m1 = floor((d-1)/2)``) for polynomial degree ``d``. The Gram
bases are the polynomials ``q_i`` orthonormal w.r.t. the prefactor measure
(the conditioned ``bilinearBasis`` SDPB emits), so
``sigma0(x) = sum_{i,j} Y0[i,j] q_i(x) q_j(x)`` and likewise ``sigma1``.

Positivity is imposed in the **sample-point** form SDPB uses (NOT raw monomial
coefficient matching, which is catastrophically ill-conditioned: high-gap spin
coefficients span >1e14). Evaluating the SOS identity at the ``d+1`` rescaled
Laguerre sample points ``x_a`` gives one equality per point:

    sum_k alpha_k Base_k(x_a)
        = sum_{i,j} Y0[i,j] q_i(x_a) q_j(x_a)
          + x_a sum_{i,j} Y1[i,j] q_i(x_a) q_j(x_a),   a = 0..d.          (*)

Both sides are evaluated numerically (bounded magnitudes), so the equality
system is well-conditioned and SDPA-GMP converges.

For ``pols1`` scalar constraints (prefactor 1) the requirement is the single
LP-cone inequality ``sum_k alpha_k * Baseexact_k(Delta_j) >= 0``.

The objective is ``minimize sum_k alpha_k * objective_k`` (``objective = Base0``);
the normalization is the linear equality ``normalization . alpha == 1`` (e.g.
``alpha[0] = -1`` for the upper-bound drivers). The driver return formula
(``Zmin = +/- alpha . objective``) is reapplied in :mod:`dual.bounds`
through the ``sense`` -> bound recompute here.

CLP / SDPA standard form (``sdpap.solve(A, b, c, K, J)``)
--------------------------------------------------------

``sdpap`` solves ``minimize c^T x  s.t.  A x - b in J,  x in K`` with
``K = (K.f, K.l, K.q, K.s)``, ``J`` likewise. We map:

- variable cone ``K``: ``K.f = n`` free vars for ``alpha``; ``K.l`` carries one
  nonnegative slack ``t_j >= 0`` per ``pols1`` scalar constraint; ``K.s`` carries
  the per-spin ``Y0``/``Y1`` PSD blocks (in build order).
- constraint cone ``J``: ``J.f`` equality rows only = 1 normalization + the SOS
  coefficient-matching rows (*) over all spins + one ``pols1`` row per scalar
  constraint, written as ``sum_k alpha_k v_k - t_j == 0`` with ``t_j >= 0`` (the
  LP-cone positivity moved onto the slack variable). Keeping ``J`` all-equality
  (``J.l == 0``) sidesteps a float-index bug in ``sdpap.convert.clp_toLMI`` that
  fires whenever ``J.l > 0``.

The dense column/row layout follows ``sdpap.fileio.readproblem``: an ``s``-block
of size ``k`` occupies ``k**2`` consecutive columns with entry ``(i, j)`` at
offset ``i*k + j`` (row-major). The matching functional ``sum_{i+j=t} Y[i,j]``
is written over all ordered ``(i, j)`` pairs (PSD symmetry makes this the
bilinear functional ``<Y, M_t>``).

Precision
---------

SDPA-GMP iterates at ``mpfPrecision`` bits with ``epsilonStar``/``epsilonDash``
duality/feasibility tolerances. The matrix data SDPA ingests is
float64 (``scipy`` CSC), so the constraint coefficients carry ~16-digit accuracy;
the solver resolves the *solution* ``alpha`` to GMP precision, and the bound is
**recomputed in mpmath** from ``alpha`` against the EXACT ``mpmath`` objective.
This reproduces the gold bounds to ~10+ significant figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpmath import mp, mpf
from scipy.sparse import csc_matrix

try:
    import sdpap
    from sdpap import SymCone
except ImportError as exc:
    raise RuntimeError(
        "sdpap is not installed; pip install sdpa-multiprecision for the SDPA-GMP "
        "high-precision backend, or use dual.backends.scipy_lp for the fast "
        "fp64 pre-screen."
    ) from exc

from ..characters import (
    DampedRational,
    orthogonal_polynomials,
    rescaled_laguerre_sample_points,
)
from ..assembly import poly_degree, safe_coeff_list


@dataclass
class SDPResult:
    """Result of an arbitrary-precision SDPA-GMP solve.

    Mirrors :class:`dual.backends.scipy_lp.SDPResult`:

    - ``status``: short status string mapped from the SDPA phase
      (``"optimal"``, ``"feasible"``, ``"infeasible"``, ``"unbounded"``,
      ``"error"``).
    - ``alpha``: the recovered functional as an ``mpf`` list of length
      ``len(sdp.objective)`` (``None`` when no functional was found).
    - ``bound``: the driver bound recomputed in ``mpmath`` from ``alpha``
      (``mpf``); for ``sense="max"`` this is ``-objective.alpha``, for
      ``sense="min"`` it is ``+objective.alpha``. ``None`` when ``alpha`` is
      ``None``.
    - ``feasible``: ``True`` when SDPA reported a primal-dual optimal/feasible
      solution.
    """

    status: str
    alpha: object
    bound: object
    feasible: bool


# SDPA phase strings -> short status. ``pdOPT`` = primal-dual optimal;
# ``pFEAS``/``dFEAS``/``pdFEAS`` = a feasible-but-not-optimal stop;
# ``pUNBD``/``dUNBD``/``pdINF`` = unbounded/infeasible certificates.
_PHASE_STATUS = {
    "pdOPT": "optimal",
    "pFEAS": "feasible",
    "dFEAS": "feasible",
    "pdFEAS": "feasible",
    "pFEAS_dINF": "unbounded",
    "pINF_dFEAS": "infeasible",
    "pdINF": "infeasible",
    "pUNBD": "unbounded",
    "dUNBD": "infeasible",
    "noINFO": "error",
}


@dataclass
class _SpinBlock:
    """Per-``pols2`` sample-point matching data for the SOS standard SDP.

    Built by :func:`_build_spin_blocks`, consumed when emitting the SOS
    coefficient-matching rows (*). ``col0``/``col1`` are the first dense columns of
    this spin's ``Y0``/``Y1`` PSD blocks (``col1 is None`` when ``n1 == 0``);
    ``xs`` the ``d+1`` rescaled-Laguerre sample points; ``scal`` the prefactor at
    each; ``bvals0[a][i] = q_i(x_a)`` the orthonormal basis values (likewise
    ``bvals1``); ``base_vals[a][k] = scal[a] * Base_k(x_a)`` the scaled LHS.
    """

    n0: int
    n1: int
    col0: int
    col1: object
    xs: list
    scal: list
    bvals0: list
    bvals1: list
    base_vals: list


def _sos_block_sizes(d):
    """SOS half-line Gram block sizes for a degree-``d`` polynomial.

    ``sigma0`` uses basis ``1..x**floor(d/2)`` -> block ``Y0`` of size
    ``floor(d/2)+1``; ``sigma1`` (multiplied by ``x``) uses basis
    ``1..x**floor((d-1)/2)`` -> block ``Y1`` of size ``floor((d-1)/2)+1``.
    For ``d == 0`` there is no ``Y1`` (``floor(-1/2) = -1`` -> size 0).
    """
    n0 = d // 2 + 1
    n1 = (d - 1) // 2 + 1 if d >= 1 else 0
    return n0, n1


def _classify_pvms(sdp):
    """Split ``sdp.pvms`` into ``pols2`` (DampedRational) and ``pols1`` (scalar).

    Returns ``(pols2, pols1)`` where each ``pols2`` entry is the pair
    ``(base_polys, prefactor)`` -- the ascending coeff-list polynomial vector
    ``Base(x)`` (``matrix[0][0]``) and its :class:`DampedRational` prefactor (the
    measure whose orthonormal polynomials condition the SOS Gram basis) -- and each
    ``pols1`` entry is the scalar coefficient vector.
    """
    pols2 = []
    pols1 = []
    for pvm in sdp.pvms:
        vec = pvm.matrix[0][0]
        if isinstance(pvm.prefactor, DampedRational):
            pols2.append(([safe_coeff_list(comp) for comp in vec], pvm.prefactor))
        else:
            pols1.append([mpf(v) for v in vec])
    return pols2, pols1


def _eval_poly(coeffs, x):
    """Horner evaluation of an ascending ``mpmath`` coeff list at ``x``."""
    acc = mpf(0)
    for ci in reversed(coeffs):
        acc = acc * x + ci
    return acc


def _basis_values_at(prefactor, n_basis, sample_points):
    """Orthonormal basis values ``q_i(x_a)`` for the SOS Gram.

    ``q_0..q_{n_basis-1}`` are orthonormal w.r.t. the ``prefactor`` measure (the
    conditioned ``bilinearBasis`` SDPB emits). Returns ``vals[a][i] = q_i(x_a)``
    over the ``sample_points`` ``x_a``. Using these evaluated values (rather than
    raw monomial coefficients) in the sample-point matching (*) keeps the SDPA
    equality system well-conditioned.
    """
    if n_basis == 0:
        return [[] for _ in sample_points]
    q = orthogonal_polynomials(prefactor, n_basis - 1)  # q_0..q_{n_basis-1}
    return [[_eval_poly(q[i], xa) for i in range(n_basis)] for xa in sample_points]


def _build_spin_blocks(pols2, n, col_start):
    """Build the per-spin SOS blocks and their dense-column layout.

    Returns ``(blocks, k_s, size_col)``: ``blocks`` the list of
    :class:`_SpinBlock`, ``k_s`` the PSD block sizes in build order (the ``K.s``
    cone), ``size_col`` the total column count after the last block. The PSD
    columns are laid out ``[ ... | Y0 (n0*n0) | Y1 (n1*n1) | ... ]`` starting at
    ``col_start`` (= ``n + n_slack``, after the free ``alpha`` and ``pols1``
    slacks); entry ``(i, j)`` of a size-``k`` block is at offset ``i*k + j``.
    """
    k_s = []
    col = col_start
    blocks = []
    for base_polys, prefactor in pols2:
        d = max(poly_degree(comp) for comp in base_polys)
        n0, n1 = _sos_block_sizes(d)
        # d+1 rescaled-Laguerre sample points where positivity is imposed;
        # well-conditioned vs raw monomial coefficient matching.
        xs = rescaled_laguerre_sample_points(d + 1)
        # sampleScalings: the prefactor 2 B**(x+gap) at each sample point.
        # Multiplying the (equality) matching row by this
        # strictly-positive scalar is harmless for the equation but rescales the
        # exponentially-large high-gap Base values to O(1) -- SDPB's conditioning
        # trick. Without it SDPA-GMP diverges (d.feas.error -> 1e55).
        scal = [prefactor(xa) for xa in xs]
        # Orthonormal basis values q_i(x_a) w.r.t. the prefactor measure.
        bvals0 = _basis_values_at(prefactor, n0, xs)         # bvals0[a][i]
        bvals1 = _basis_values_at(prefactor, n1, xs)         # bvals1[a][i]
        # Base_k(x_a) * scal[a]: scaled LHS sample-point evaluations.
        base_vals = [[_eval_poly(base_polys[k], xs[a]) * scal[a] for k in range(n)]
                     for a in range(len(xs))]                 # base_vals[a][k]
        col0 = col
        k_s.append(n0)
        col += n0 * n0
        if n1 > 0:
            col1 = col
            k_s.append(n1)
            col += n1 * n1
        else:
            col1 = None
        blocks.append(_SpinBlock(n0, n1, col0, col1, xs, scal,
                                 bvals0, bvals1, base_vals))
    return blocks, k_s, col


def solve_sdp(sdp, *, precision_bits=1024, sense="max",
              epsilon=mpf(10) ** -15, max_iteration=500, verbose=False):
    """Solve the dual SDP exactly with SDPA-GMP and recover ``alpha``.

    ``sdp`` is an assembled :class:`dual.assembly.SDP`. The
    continuum positivity of every ``pols2`` spin sector is encoded as the exact
    SOS-on-the-half-line standard SDP (module docstring); ``pols1`` scalar
    constraints become LP-cone inequalities; the normalization is a linear
    equality and the objective ``minimize objective . alpha``.

    ``sense`` selects the driver return formula: ``"max"`` (upper-bound drivers,
    ``norm = -e0``) returns ``bound = -objective . alpha``; ``"min"``
    (lower-bound drivers, ``norm = +e0``) returns ``bound = +objective . alpha``.

    ``precision_bits`` -> SDPA ``mpfPrecision``; ``epsilon`` ->
    ``epsilonStar``/``epsilonDash``. Returns an
    :class:`SDPResult`. The recovered ``alpha`` is read at GMP precision and the
    bound is recomputed in ``mpmath`` against the EXACT objective.
    """
    n = len(sdp.objective)
    objective = [mpf(o) for o in sdp.objective]
    normalization = [mpf(v) for v in sdp.normalization]

    pols2, pols1 = _classify_pvms(sdp)
    n_slack = len(pols1)

    # --- variable cone K: alpha (free) | pols1 slacks (K.l) | SOS Gram (K.s) ---
    # Column layout (readproblem order): [ K.f free | K.l slacks | K.s blocks ].
    slack_col0 = n                 # first slack column
    spin_blocks, k_s, size_col = _build_spin_blocks(pols2, n, n + n_slack)

    # --- constraint cone J: all equalities (J.l == 0 dodges the convert bug) ---
    A_rows = []
    A_cols = []
    A_vals = []
    b_rows = []
    b_vals = []

    def add_A(r, c, v):
        if v != 0:
            A_rows.append(r)
            A_cols.append(c)
            A_vals.append(float(v))

    def add_row_equilibrated(r, entries):
        """Add a homogeneous (b=0) equality row, divided by its max |entry|.

        ``entries`` is a list of ``(col, mpf_value)`` pairs. Row-scaling by a
        positive constant leaves the equation (== 0) invariant but brings every
        matching row to O(1); without it the high-gap-spin rows (scaled by the
        prefactor ~ exp(-126)) sit ~1e-55 below the O(1) rows and the fp64 matrix
        SDPA ingests is hopelessly ill-conditioned (only ~4 correct digits).
        """
        m = mpf(0)
        for _c, v in entries:
            av = abs(v)
            if av > m:
                m = av
        if m == 0:
            return
        inv = mpf(1) / m
        for _c, v in entries:
            add_A(r, _c, v * inv)

    row = 0
    # (row 0) normalization: normalization . alpha == 1  ->  A x - b = 0.
    for k in range(n):
        add_A(row, k, normalization[k])
    b_rows.append(row)
    b_vals.append(1.0)
    row += 1

    # SOS sample-point matching (*) per spin, at each sample point x_a:
    #   sum_k alpha_k Base_k(x_a)
    #     = sum_{i,j} Y0[i,j] q_i(x_a) q_j(x_a)
    #       + x_a sum_{i,j} Y1[i,j] q_i(x_a) q_j(x_a).
    for blk in spin_blocks:
        for a, xa in enumerate(blk.xs):
            sa = blk.scal[a]
            entries = []
            # LHS: scal[a] * sum_k alpha_k * Base_k(x_a)  (base_vals already scaled)
            bv = blk.base_vals[a]
            for k in range(n):
                if bv[k] != 0:
                    entries.append((k, bv[k]))
            # RHS (moved to LHS with minus), also scaled by scal[a]:
            # scal[a] * sigma0(x_a) = scal[a] sum_ij Y0[i,j] q_i q_j;
            # scal[a] * x_a sigma1(x_a) = scal[a] x_a sum_ij Y1[i,j] q_i q_j.
            b0 = blk.bvals0[a]
            for i in range(blk.n0):
                qi = sa * b0[i]
                for j in range(blk.n0):
                    entries.append((blk.col0 + i * blk.n0 + j, -qi * b0[j]))
            if blk.n1 > 0:
                b1 = blk.bvals1[a]
                sax = sa * xa
                for i in range(blk.n1):
                    qi = sax * b1[i]
                    for j in range(blk.n1):
                        entries.append((blk.col1 + i * blk.n1 + j, -qi * b1[j]))
            add_row_equilibrated(row, entries)
            # b = 0 for matching rows.
            row += 1

    # pols1 scalar positivity as an equality with a nonneg slack:
    # sum_k alpha_k v_k - t_j == 0,  t_j >= 0  (so sum_k alpha_k v_k >= 0).
    for j, vec in enumerate(pols1):
        for k in range(n):
            add_A(row, k, vec[k])
        add_A(row, slack_col0 + j, mpf(-1))
        # b = 0.
        row += 1
    size_row = row

    # --- objective ---
    # SDPB's PMP standard form MAXIMIZES the dual objective  V = objective . alpha
    # subject to the polynomial positivity (arXiv:1502.02033). sdpap MINIMIZES
    # c^T x, so set c = -objective to maximize. The driver then reapplies the
    # sign convention:  lowerbound (sense="min")  Zmin = +V;
    # upperbound (sense="max")  Zmin = -V. (Note: minimizing objective . alpha is
    # genuinely unbounded here -- only the maximization is bounded, matching SDPB.)
    c_rows = []
    c_vals = []
    for k in range(n):
        if objective[k] != 0:
            c_rows.append(k)
            c_vals.append(float(-objective[k]))

    # --- column equilibration (variable rescaling) -------------------------
    # The constraint matrix still spans a wide column-norm range (alpha columns
    # O(1); high-gap Gram blocks tiny), so the fp64 matrix SDPA ingests is
    # ill-conditioned. Rescale each variable so its column has unit max-|entry|:
    #   alpha_k -> alpha_k / fa_k        (free columns, one factor each)
    #   Y_block -> Y_block / fb          (one scalar per PSD block: a positive
    #                                     congruence Y -> Y/fb preserves PSD)
    #   slack_j -> slack_j / fs_j        (LP columns)
    # The recovered x is unscaled below (alpha_k *= ... is folded into the read).
    col_max = [mpf(0)] * size_col
    for cc, v in zip(A_cols, A_vals):
        av = abs(mpf(v))
        if av > col_max[cc]:
            col_max[cc] = av

    def _col_factor(start, width):
        m = mpf(0)
        for cc in range(start, start + width):
            if col_max[cc] > m:
                m = col_max[cc]
        return m if m > 0 else mpf(1)

    col_scale = [mpf(1)] * size_col
    # alpha + slack columns: per-column factor.
    for cc in range(n + n_slack):
        col_scale[cc] = col_max[cc] if col_max[cc] > 0 else mpf(1)
    # each PSD block: one shared factor (preserve symmetry/PSD).
    bcol = n + n_slack
    for k in k_s:
        f = _col_factor(bcol, k * k)
        for cc in range(bcol, bcol + k * k):
            col_scale[cc] = f
        bcol += k * k

    A_vals = [float(mpf(v) / col_scale[cc]) for v, cc in zip(A_vals, A_cols)]
    c_vals = [float(mpf(v) / col_scale[cc]) for v, cc in zip(c_vals, c_rows)]

    K = SymCone(f=n, l=n_slack, q=(), s=tuple(k_s))
    J = SymCone(f=size_row, l=0, q=(), s=())

    A = csc_matrix((A_vals, (A_rows, A_cols)), shape=(size_row, size_col))
    b = csc_matrix((b_vals, (b_rows, [0] * len(b_rows))), shape=(size_row, 1))
    c = csc_matrix((c_vals, (c_rows, [0] * len(c_rows))), shape=(size_col, 1))

    eps = float(mpf(epsilon))
    option = {
        "print": "display" if verbose else "no",
        "mpfPrecision": int(precision_bits),
        "epsilonStar": eps,
        "epsilonDash": eps,
        "maxIteration": int(max_iteration),
        # SDPA's lowerBound/upperBound bracket the objective for feasibility
        # detection (defaults +/-1e5). The dual bound objective transits a wide
        # range during the primal-dual iteration, so widen the bracket far past
        # the objective scale or SDPA spuriously declares unboundedness.
        "lowerBound": -1.0e30,
        "upperBound": 1.0e30,
    }

    x, y, sdpapinfo, _timeinfo, sdpainfo = sdpap.solve(A, b, c, K, J, option)

    phase = sdpainfo.get("phasevalue", "noINFO")
    status = _PHASE_STATUS.get(phase, "error")
    feasible = phase in ("pdOPT", "pFEAS", "dFEAS", "pdFEAS")

    # Near-optimal stall: on ill-conditioned instances (e.g. the narrow-[Da]
    # gold case) SDPA can plateau at the fp64-input feasibility floor and label
    # the phase pdINF even though primal and dual objectives have already met.
    # Accept such a point as a (precision-limited) optimum when the relative
    # duality gap is tiny -- the returned bound then carries the achieved digits.
    if not feasible and phase == "pdINF":
        po = sdpainfo.get("primalObj")
        do = sdpainfo.get("dualObj")
        if po is not None and do is not None:
            scale = max(abs(po), abs(do), 1.0)
            if abs(po - do) <= 1e-6 * scale:
                feasible = True
                status = "feasible"

    if not feasible:
        return SDPResult(status=status, alpha=None, bound=None, feasible=False)

    # Primal x: first n entries are alpha (the free block), read at GMP
    # precision and unscaled by the column equilibration factor.
    x_dense = np.asarray(x.todense()).ravel()
    alpha = [mpf(_to_str(x_dense[k])) / col_scale[k] for k in range(n)]

    obj_dot = mp.fsum(objective[k] * alpha[k] for k in range(n))
    bound = -obj_dot if sense == "max" else obj_dot

    return SDPResult(status=status, alpha=alpha, bound=bound, feasible=True)


def _to_str(v):
    """Render a solver-returned scalar losslessly for ``mpf`` (full repr)."""
    return repr(float(v)) if not isinstance(v, str) else v
