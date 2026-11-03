"""Build a candidate spectrum at one point of the (c, gap) search grid.

Each point of the search is labeled by two numbers: a central charge
``c`` and a spectral gap ``delta_gap`` (the dimension of the lightest
non-identity scalar operator we require the theory to have). From that
pair we build a starting candidate spectrum for the optimizer to refine:

* 2 fixed operators: the identity, and a "gap" scalar pinned at
  dimension ``delta_gap`` so the search never drifts into a spectrum
  with something lighter than the gap we're testing.
* A block of free ("flex") operators at each integer spin
  ``j = 0, 1, ..., delta_max - 1``, with ``N_j`` copies at spin ``j``.
  Each flex operator's dimension is a sigmoid-squashed function of one
  free real parameter, so it can move continuously but is guaranteed to
  stay inside a fixed window ``[max(j, delta_gap), 2*delta_max - lo]``
  as the optimizer updates that parameter.

If the optimizer (see ``optimize.py``) can push the loss on this
candidate below the "consistent" threshold, the point survives, i.e. it
looks like a viable CFT; if every attempt in the search fails, the point
counts as excluded. Running this construction over every (c, delta_gap)
on a grid and testing each candidate is what produces the search
landscape.

The number of flex operators per spin is fixed by a piecewise rule
verified against 3 independently known exact spectra (a parafermion
theory, a super-Virasoro minimal model, and Ising x tricritical-Ising):
all three match ``N_j = 6*(delta_max - j)`` for ``delta_max <= 5``. At
``delta_max >= 6`` the per-spin count is widened to ``N_j = 8*(delta_max
- j)``: with the paper's k=6 rule the highest-spin slot came up 2 copies
short of a fifth known exact spectrum (a minimal model at delta_max=6),
silently dropping those operators and giving a false "no consistent
spectrum" reading that was originally mis-diagnosed as physical. The
wider k=8 rule clears that spectrum's loss threshold and gives headroom
against similar shortages at spins/central charges without a cached
exact comparison.

Initial values for the free parameters spread the flex operators evenly
across their allowed window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .data_types import CFTData, DSD, WLExpr

# Deprecated module-level default. Prefer the ``max_delta`` argument to
# :func:`build_survey_point`. Kept for backward-compat with any caller
# that imports it directly; the function's default matches this value.
MAX_DELTA = 4
NUM_SPIN = {0: 24, 1: 18, 2: 12, 3: 6}


def _default_num_spin(max_delta: float) -> dict[int, int]:
    """Piecewise counts: ``N_j = k*(delta_max - j)`` with ``k = 6`` at
    ``delta_max <= 5`` (matches the 3 independently known exact spectra)
    and ``k = 8`` at ``delta_max >= 6`` (widened to clear a 2-copy
    shortage at the top spin found against a 4th exact spectrum).

    At ``delta_max = 4`` returns ``{0:24, 1:18, 2:12, 3:6}``.
    """
    d = int(max_delta)
    if float(max_delta) != d or d < 3:
        raise ValueError(f"delta_max must be an integer >= 3, got {max_delta!r}")
    k = 6 if d <= 5 else 8
    return {j: k * (d - j) for j in range(d)}


def _sigmoid_esque_expr(var: WLExpr, min_val: float, max_val: float) -> WLExpr:
    """``min + (max-min)/(1+exp(-x))`` as a symbolic expression tree, so
    ``partition.py``/``optimize.py`` evaluate it through the same
    Plus/Times/Power/Exp machinery as any other primary."""
    span = float(max_val) - float(min_val)
    # min + span * 1/(1+exp(-x))  == min + span * (1 + exp(-x))^(-1)
    inner = WLExpr("Plus", (
        1.0,
        WLExpr("Exp", (WLExpr("Times", (-1.0, var)),)),
    ))
    return WLExpr("Plus", (
        float(min_val),
        WLExpr("Times", (span, WLExpr("Power", (inner, -1.0)))),
    ))


@dataclass(frozen=True)
class SurveyPoint:
    """One grid point: central charge x spectral gap x the resulting
    primaries + initial variable values."""
    c: float
    delta_gap: float
    cft: CFTData
    init_values: np.ndarray

    @property
    def name(self) -> str:
        return f"c{int(round(self.c * 100)):03d}_dg{int(round(self.delta_gap * 100)):02d}"


def _initial_value(i: int, n: int) -> float:
    """``log(r / (2 - r))`` with ``r = i/(n+1)`` — spreads operators
    evenly across the sigmoid's range."""
    r = i / (n + 1)
    return float(math.log(r / (2.0 - r)))


def build_survey_point(
    c: float,
    delta_gap: float,
    max_delta: float = MAX_DELTA,
    num_spin: dict[int, int] | None = None,
    pseudogap_floor: float | None = None,
) -> SurveyPoint:
    """Build a flex-operator-parameterized candidate CFT at (c, delta_gap).

    Parameters
    ----------
    c
        Central charge.
    delta_gap
        Spectral-gap anchor (first non-vacuum scalar's dimension).
    max_delta
        Dimension cutoff. Sets both ``cft.max_delta`` and the sigmoid
        upper bound ``hi = 2*max_delta - lo``. Default ``4``; larger
        values let the search probe wider regions.
    num_spin
        ``{spin: count}`` flex-operator counts. ``None`` -> the
        piecewise default (see :func:`_default_num_spin`). At
        ``max_delta=4`` the default is exactly ``{0:24, 1:18, 2:12,
        3:6}``.
    pseudogap_floor
        Opt-in: when set to a small positive float, replaces the fixed
        gap scalar ``DSD(delta_gap, 0, 1)`` with ``DSD(pseudogap_floor,
        0, 1)`` and uses ``max(spin, pseudogap_floor)`` as the flex
        sigmoid lower bound instead of ``max(spin, delta_gap)``. Only
        intended for probing a central charge without pinning a gap
        directly. Default ``None`` preserves the standard behavior
        bit-for-bit.
    """
    if num_spin is None:
        num_spin = _default_num_spin(max_delta)

    use_floor = pseudogap_floor is not None
    if use_floor and float(pseudogap_floor) <= 0.0:
        raise ValueError(
            f"pseudogap_floor must be > 0 if set, got {pseudogap_floor!r}"
        )
    gap_anchor = float(pseudogap_floor) if use_floor else float(delta_gap)

    primaries: list[DSD] = [
        DSD(delta=0.0, spin=0, degen=1),
        DSD(delta=gap_anchor, spin=0, degen=1),
    ]
    variables: list[WLExpr] = []
    init_values: list[float] = []
    sig_mins_list: list[float] = []
    sig_scales_list: list[float] = []

    for s in sorted(num_spin):
        n = num_spin[s]
        lo = float(max(s, gap_anchor))
        hi = float(2 * max_delta - lo)
        for i in range(1, n + 1):
            var = WLExpr("FlexDelta", (float(s), float(i)))
            variables.append(var)
            init_values.append(_initial_value(i, n))
            sig_mins_list.append(lo)
            sig_scales_list.append(hi - lo)
            primaries.append(DSD(
                delta=_sigmoid_esque_expr(var, lo, hi),
                spin=int(s),
                degen=1,
            ))

    init_array = np.asarray(init_values, dtype=np.float64)

    cft = CFTData(
        name=f"survey_c{c:.2f}_dg{delta_gap:.2f}",
        central_charge=float(c),
        primaries=tuple(primaries),
        max_delta=float(max_delta),
        variables=tuple(variables),
    )
    return SurveyPoint(
        c=float(c),
        delta_gap=float(delta_gap),
        cft=cft,
        init_values=init_array,
    )


def survey_grid() -> list[tuple[float, float]]:
    """The canonical (c, delta_gap) grid reproducing the paper figure.

    * c in {1.00, 1.02, ..., 1.20} (11 values, 0.02 spacing)
    * delta_gap in {0.04, 0.08, ..., 0.60} (15 values, 0.04 spacing)
    """
    cs = np.round(np.arange(1.00, 1.20 + 1e-9, 0.02), 6)
    dgs = np.round(np.arange(0.04, 0.60 + 1e-9, 0.04), 6)
    return [(float(c), float(dg)) for c in cs for dg in dgs]
