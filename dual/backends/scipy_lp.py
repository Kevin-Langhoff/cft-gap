"""fp64 FIND backend -- locate a dual functional by sampling positivity into an LP.

The *fast* FIND adapter for the dual modular bootstrap (this specific LP
design is the pure-Python pre-screen / FIND companion to the SDPB-XML
backend). It consumes an assembled :class:`~dual.assembly.SDP`
(``objective`` + ``normalization`` + ``pvms`` of
:class:`~dual.assembly.PositiveMatrixWithPrefactor`) and finds a
candidate linear functional ``alpha`` by discretising the continuum positivity
constraints into a finite linear program.

Design (float64 is allowed ONLY inside the LP):

- Variables: the functional ``alpha``, length ``n = len(sdp.objective)``.
- Positivity rows (fp64): each ``pols2`` matrix entry is a polynomial **vector**
  in ``x = Delta - gap``; sampling ``x`` over ``[0, x_max]`` (dense grid + a few
  large-``x`` tail probes + any ``extra_x``) gives one row per sample whose
  ``k``-th entry is the ``k``-th component polynomial evaluated at ``x``. Each
  ``pols1`` entry is already a scalar vector, contributing one row directly.
  Every row constrains ``r . alpha >= 0``.
- Normalization: ``sdp.normalization . alpha == 1`` (equality).
- Objective: minimise ``sign * (sdp.objective . alpha)`` (``sign = +1`` is the
  upperbound-style maximise-then-negate convention; exposed so
  ``bounds.py`` picks). ``alpha`` is box-bounded to ``[-alpha_bound, alpha_bound]``
  for LP boundedness.
- Solve via :func:`scipy.optimize.linprog` (``method="highs"``), map the status,
  return ``alpha`` as a list, and recompute ``bound = sdp.objective . alpha`` in
  ``mpmath`` from the fp64 ``alpha`` (a clean high-ish value).

The LP is float64 and its feasibility is only sampled, so the result is a
*candidate*. Rigor comes from :func:`dual.verify.certify_functional`
certifying the returned ``alpha`` at high precision (the between-sample positivity
the LP cannot see), NOT from the LP itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpmath import mp, mpf
from scipy.optimize import linprog

from ..characters import DampedRational


@dataclass
class SDPResult:
    """Result of an fp64 FIND solve.

    - ``status``: short status string mapped from ``linprog`` (``"optimal"``,
      ``"feasible"``, ``"infeasible"``, ``"unbounded"``, or ``"error"``).
    - ``alpha``: the recovered functional as a Python ``float`` list of length
      ``len(sdp.objective)`` (``None`` when no functional was found).
    - ``bound``: ``sdp.objective . alpha`` recomputed in ``mpmath`` from the fp64
      ``alpha`` (``mpf``); ``None`` when ``alpha`` is ``None``.
    - ``feasible``: ``True`` when the LP reported an optimal/feasible solution.
    """

    status: str
    alpha: object
    bound: object
    feasible: bool


# Map scipy HiGHS exit codes to short status strings (a small ordered status
# vocabulary; the LP only distinguishes these four cases).
_STATUS = {
    0: "optimal",
    2: "infeasible",
    3: "unbounded",
    4: "error",  # numerical difficulties
}


def _eval_poly(coeffs, x):
    """Horner evaluation of an ascending coeff list at float ``x`` -> float."""
    acc = 0.0
    for ci in reversed(coeffs):
        acc = acc * x + float(ci)
    return acc


def _pvm_rows(pvm, sample_x):
    """Positivity rows (fp64) for one PVM constraint.

    The single matrix entry ``pvm.matrix[0][0]`` is the length-``n`` component
    vector. For a ``pols2`` (``DampedRational`` prefactor) constraint each
    component is a polynomial in ``x``; we emit one row per sample ``x`` whose
    ``k``-th entry is the ``k``-th component evaluated there. For a ``pols1``
    (scalar prefactor) constraint the components are already numeric, so a single
    row is the component vector itself (``sample_x`` is ignored).

    Returns a list of fp64 rows (each a list of ``float``), one per binding point.
    """
    vec = pvm.matrix[0][0]
    if isinstance(pvm.prefactor, DampedRational):
        # pols2: each component is an ascending coeff list in x. The strictly
        # positive damped prefactor cannot change the sign on x >= 0, so it is
        # omitted from the row (matches verify.certify_nonneg_on_halfline).
        rows = []
        for x in sample_x:
            rows.append([_eval_poly(comp, x) for comp in vec])
        return rows
    # pols1: components are scalars (damping already folded into Baseexact).
    return [[float(comp) for comp in vec]]


def find_functional(sdp, *, samples_per_spin=400, extra_x=None, sign=+1,
                    alpha_bound=1e6):
    """Locate a dual functional ``alpha`` via a sampled float64 LP (FIND).

    ``sdp`` is an assembled :class:`~dual.assembly.SDP`. The
    continuum positivity of each ``pols2`` spin sector is discretised by sampling
    ``x = Delta - gap`` over ``[0, x_max]`` with ``samples_per_spin`` dense points
    plus a few large-``x`` tail probes (so the LP also sees the asymptotic sign),
    plus any ``extra_x`` carried over from a previous refine round (e.g. a witness
    ``x*`` that :mod:`dual.verify` flagged). ``x_max`` is taken from the
    scalar (``pols1``) sample window -- the widest ``Delta`` probed -- which is the
    natural support of the constraint grid.

    LP:  minimise ``sign * (objective . alpha)``  s.t.
    ``normalization . alpha == 1``, every sampled positivity row ``r . alpha >= 0``,
    and ``-alpha_bound <= alpha_i <= alpha_bound``. ``sign = +1`` is the
    upperbound-style maximise-then-negate convention.

    Returns an :class:`SDPResult`. ``bound`` is recomputed in ``mpmath`` from the
    fp64 ``alpha`` (a clean high-ish value); the LP's float64 feasibility is only
    sampled, so the returned ``alpha`` is a *candidate* -- certify it with
    :func:`dual.verify.certify_functional`.
    """
    n = len(sdp.objective)

    # x_max from the pols1 (scalar) Delta window: those constraints sample Delta
    # in [gap, Da], i.e. x = Delta - gap up to Da - gap. We do not have the raw
    # grid here, so derive x_max from each pols2 prefactor's gap and the spread.
    # Simplest robust choice: sample x over [0, x_max] with a fixed span plus a
    # tail. We use x_max = 1 (the Da - gap span of the standard scalartb window is
    # O(1)); the large-x tail probes catch the asymptotic sign beyond that.
    x_max = 1.0
    base_grid = list(np.linspace(0.0, x_max, samples_per_spin))
    # A few geometric tail probes so the LP sees the asymptotic (leading-coeff)
    # sign of each spin polynomial, not just the near-gap behaviour.
    tail = [2.0, 5.0, 10.0, 50.0, 200.0]
    sample_x = base_grid + tail
    if extra_x is not None:
        sample_x = sample_x + [float(x) for x in extra_x]

    # Inequality rows A_ub @ alpha <= b_ub. Our constraint is r . alpha >= 0, i.e.
    # (-r) . alpha <= 0.
    A_ub = []
    for pvm in sdp.pvms:
        for row in _pvm_rows(pvm, sample_x):
            A_ub.append([-v for v in row])
    A_ub = np.asarray(A_ub, dtype=float) if A_ub else None
    b_ub = np.zeros(A_ub.shape[0]) if A_ub is not None else None

    # Equality: normalization . alpha == 1.
    A_eq = np.asarray([[float(v) for v in sdp.normalization]], dtype=float)
    b_eq = np.asarray([1.0])

    # Objective: minimise sign * (objective . alpha).
    cobj = np.asarray([float(sign) * float(o) for o in sdp.objective], dtype=float)

    bounds = [(-alpha_bound, alpha_bound)] * n

    res = linprog(cobj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    status = _STATUS.get(res.status, "error")
    feasible = res.success and res.x is not None

    if not feasible:
        return SDPResult(status=status, alpha=None, bound=None, feasible=False)

    alpha = [float(a) for a in res.x]

    # Recompute the bound in mpmath from the fp64 alpha (clean high-ish value).
    bound = mp.fsum(mpf(o) * mpf(a) for o, a in zip(sdp.objective, alpha))

    return SDPResult(status=status, alpha=alpha, bound=bound, feasible=True)
