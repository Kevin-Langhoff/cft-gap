"""Warm starts: hand the optimizer a starting point that is already good.

The search loss is a landscape of narrow, needle-like minima: displacing a
converged parameter vector by ~1e-3 in a random direction raises the
log1p loss by 8-15. A start drawn from the generic initialization
therefore rarely lands in the basin of a known solution, even when that
solution sits at the very (c, delta_gap) point being tested. Two ways of
building a better start live here:

* **Carry parameters** (:func:`lift_init`). Copy a converged pre-sigmoid
  parameter vector from a neighbouring point into the target point's
  ``init_values``, matching slots by variable identity. This is the warm
  start the continuation scan uses, because it preserves the basin.
* **Encode a spectrum** (:func:`build_warm_started_survey_point`). Invert
  the sigmoid map to place the flex operators on the dimensions of a known
  numeric spectrum. Rounding through ``(delta, spin, degen)`` falls off the
  needle, so this is NOT a substitute for carrying parameters between
  neighbouring cells; its use is to seed a point from an exactly known
  theory, the compactified free boson (:func:`free_boson_point`).

:func:`resolve_flexdelta_to_numeric` goes the other way: it evaluates
every sigmoid-parameterized dimension at a parameter vector, giving the
numeric spectrum a converged point stands for.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Sequence, Union

import numpy as np
import torch

from ._expr_eval import eval_wl
from .data_types import CFTData, DSD, WLExpr, resolved_character_kind
from .spectrum import SurveyPoint, build_survey_point

SpectrumEntry = Union[DSD, tuple[float, int, int], tuple[float, int, int, str]]


# ============================================================================
# Carry converged parameters between ansatz instances.
# ============================================================================


def lift_init(
    source_variables: Sequence[WLExpr],
    source_flex: np.ndarray | Sequence[float],
    target_variables: Sequence[WLExpr],
    target_init_fallback: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, dict]:
    """Copy source flex parameters into a target init by variable key.

    Slots are matched on the ``(head, *args)`` key of each ``WLExpr``
    variable, i.e. on ``FlexDelta[spin, index]``, so the carry survives a
    change in ansatz size between source and target (a c-dependent slot
    rule gives neighbouring cells different slot counts). Target slots with
    no source match keep ``target_init_fallback``; pass the target's own
    ``init_values`` for "warm where possible, generic elsewhere".

    Returns ``(init_array, info)`` where ``info`` counts ``source_slots``,
    ``target_slots``, ``copied_slots`` and ``fresh_slots``.
    """
    source = np.asarray(source_flex, dtype=np.float64)
    fallback = np.asarray(target_init_fallback, dtype=np.float64)
    if len(source) != len(source_variables):
        raise ValueError(
            f"source flex length {len(source)} does not match "
            f"source_variables length {len(source_variables)}"
        )
    if len(fallback) != len(target_variables):
        raise ValueError(
            f"target_init_fallback length {len(fallback)} does not match "
            f"target_variables length {len(target_variables)}"
        )

    source_map: dict[tuple, int] = {
        (v.head, *v.args): i for i, v in enumerate(source_variables)
    }
    target_init = fallback.copy()
    copied = 0
    for i, var in enumerate(target_variables):
        j = source_map.get((var.head, *var.args))
        if j is not None:
            target_init[i] = source[j]
            copied += 1
    info = {
        "source_slots": int(len(source_variables)),
        "target_slots": int(len(target_variables)),
        "copied_slots": int(copied),
        "fresh_slots": int(len(target_variables) - copied),
    }
    return target_init, info


# ============================================================================
# Encode a numeric spectrum as flex parameters.
# ============================================================================


def _as_dsd(entry: SpectrumEntry) -> DSD:
    if isinstance(entry, DSD):
        return DSD(float(entry.delta), int(entry.spin), int(entry.degen),
                   entry.character_kind)
    if len(entry) == 4:
        delta, spin, degen, character_kind = entry
        return DSD(float(delta), int(spin), int(degen), character_kind)
    delta, spin, degen = entry
    return DSD(float(delta), int(spin), int(degen))


def _inverse_sigmoid_esque(delta: float, lo: float, hi: float) -> float:
    """Invert ``delta = lo + (hi - lo) / (1 + exp(-x))``."""
    if not (lo < delta < hi):
        raise ValueError(f"Delta={delta} outside sigmoid range ({lo}, {hi})")
    return float(math.log((delta - lo) / (hi - delta)))


def _is_conserved_current(primary: DSD) -> bool:
    return resolved_character_kind(primary) == "conserved"


def _is_vacuum(primary: DSD) -> bool:
    return resolved_character_kind(primary) == "vacuum"


def _fixed_boundary_generic_primaries(
    spectrum: tuple[DSD, ...],
    delta_gap: float,
    max_delta: int | float,
    source_delta_gap: float,
) -> tuple[DSD, ...]:
    """Generic primaries sitting exactly at their sigmoid lower bound.

    Flex slots parameterize the open interval ``lo < Delta < hi``. A
    generic spin-j primary at ``Delta == max(j, delta_gap)`` is physical
    but needs an infinite parameter value, so it is appended to the
    candidate as a fixed numeric primary instead.
    """
    out: list[DSD] = []
    dmax = float(max_delta)
    for p in spectrum:
        d = float(p.delta)
        j = int(p.spin)
        if d > dmax + 1e-9:
            continue
        if resolved_character_kind(p) != "generic":
            continue
        if j == 0 and abs(d - source_delta_gap) < 1e-9:
            continue
        lo = float(max(j, delta_gap))
        if abs(d - lo) < 1e-9:
            out.append(p)
    return tuple(out)


def encode_spectrum_init_values(
    spectrum: Iterable[SpectrumEntry],
    num_spin: dict[int, int],
    delta_gap: float,
    max_delta: int | float,
    *,
    source_delta_gap: float | None = None,
    dead_slot_x: float = 30.0,
) -> tuple[np.ndarray, dict]:
    """Encode a numeric spectrum as flex initial values.

    Primaries are expanded by degeneracy, bucketed by spin, sorted by
    dimension, and assigned to the lowest flex slots at that spin. The
    vacuum, the fixed gap scalar and conserved currents are skipped (the
    ansatz holds them outside the flex slots). Unused slots get
    ``dead_slot_x``, which parks their dimension near the top of the
    sigmoid window, above the truncation, where they drop out.

    ``source_delta_gap`` is the gap scalar of the source spectrum (default:
    the target ``delta_gap``). When the source is a known solution at a
    different gap, pass it so that its own gap scalar is encoded as an
    ordinary flex scalar rather than being dropped.
    """
    dmax = float(max_delta)
    source_gap = float(delta_gap if source_delta_gap is None else source_delta_gap)
    source_gap_copies_consumed = 0
    copies_by_spin: dict[int, list[float]] = defaultdict(list)
    for primary in (_as_dsd(p) for p in spectrum):
        d = float(primary.delta)
        j = int(primary.spin)
        if _is_vacuum(primary):
            continue
        if _is_conserved_current(primary):
            continue
        if j == 0 and abs(d - source_gap) < 1e-9 and abs(source_gap - delta_gap) < 1e-9:
            source_gap_copies_consumed += int(primary.degen)
            continue
        for _ in range(int(primary.degen)):
            if j == 0 and abs(d - source_gap) < 1e-9 and source_gap_copies_consumed == 0:
                source_gap_copies_consumed += 1
                continue
            copies_by_spin[j].append(d)
    for copies in copies_by_spin.values():
        copies.sort()

    init_values: list[float] = []
    stats = {
        "per_spin": {},
        "dead_slots": 0,
        "dropped_copies": 0,
        "out_of_range_copies": 0,
        "source_gap_copies_consumed": int(source_gap_copies_consumed),
    }
    for spin in sorted(num_spin):
        n_slots = int(num_spin[spin])
        copies = copies_by_spin.get(int(spin), [])
        lo = float(max(int(spin), delta_gap))
        hi = float(2.0 * dmax - lo)

        in_range = [delta for delta in copies if lo < delta < hi]
        out_of_range = len(copies) - len(in_range)
        matched = in_range[:n_slots]
        dropped = len(in_range) - len(matched)
        dead = n_slots - len(matched)

        for delta in matched:
            init_values.append(_inverse_sigmoid_esque(delta, lo, hi))
        init_values.extend([float(dead_slot_x)] * dead)

        stats["per_spin"][int(spin)] = {
            "n_slots": n_slots,
            "n_spectrum_copies": len(copies),
            "n_out_of_range": out_of_range,
            "n_matched": len(matched),
            "n_dead": dead,
            "n_dropped": dropped,
            "lo": lo,
            "hi": hi,
        }
        stats["dead_slots"] += dead
        stats["dropped_copies"] += dropped
        stats["out_of_range_copies"] += out_of_range

    return np.asarray(init_values, dtype=np.float64), stats


def build_warm_started_survey_point(
    c: float,
    delta_gap: float,
    max_delta: int | float,
    spectrum: Iterable[SpectrumEntry],
    *,
    num_spin: dict[int, int] | None = None,
    source_delta_gap: float | None = None,
    dead_slot_x: float = 30.0,
    add_conserved_currents: bool = True,
) -> tuple[SurveyPoint, dict]:
    """A survey point whose flex slots start on a known numeric spectrum.

    Builds the standard ansatz with :func:`spectrum.build_survey_point`,
    encodes ``spectrum`` into its ``init_values``, and appends as fixed
    primaries whatever the flex slots cannot represent: extra copies of the
    gap scalar, generic primaries at their sigmoid lower bound, and (with
    ``add_conserved_currents``) the conserved currents. The appended
    primaries change the candidate's operator count, so such a point cannot
    share an optimizer batch with a plain survey point.

    Returns ``(survey_point, stats)`` with the slot-mapping report of
    :func:`encode_spectrum_init_values`.
    """
    spectrum_dsds = tuple(_as_dsd(p) for p in spectrum)
    resolved_source_gap = float(delta_gap if source_delta_gap is None else source_delta_gap)
    fixed_source_gap_extra_copies = 0
    if abs(resolved_source_gap - float(delta_gap)) < 1e-9:
        source_gap_scalar_copies = sum(
            int(p.degen)
            for p in spectrum_dsds
            if int(p.spin) == 0 and abs(float(p.delta) - resolved_source_gap) < 1e-9
        )
        fixed_source_gap_extra_copies = max(0, source_gap_scalar_copies - 1)
    point = build_survey_point(c, delta_gap, max_delta=float(max_delta), num_spin=num_spin)
    resolved_num_spin: dict[int, int] = {}
    for var in point.cft.variables:
        spin = int(var.args[0])
        resolved_num_spin[spin] = resolved_num_spin.get(spin, 0) + 1

    init_values, stats = encode_spectrum_init_values(
        spectrum_dsds,
        resolved_num_spin,
        delta_gap,
        max_delta,
        source_delta_gap=source_delta_gap,
        dead_slot_x=dead_slot_x,
    )
    stats["fixed_source_gap_extra_copies"] = int(fixed_source_gap_extra_copies)
    fixed_boundary_generic = _fixed_boundary_generic_primaries(
        spectrum_dsds, delta_gap, max_delta, resolved_source_gap,
    )
    stats["fixed_boundary_generic_copies"] = int(
        sum(int(p.degen) for p in fixed_boundary_generic)
    )

    fixed_extra_gap = (
        (DSD(float(delta_gap), 0, int(fixed_source_gap_extra_copies)),)
        if fixed_source_gap_extra_copies
        else ()
    )
    extra = fixed_extra_gap + fixed_boundary_generic
    if add_conserved_currents:
        extra = extra + tuple(p for p in spectrum_dsds if _is_conserved_current(p))
    if extra:
        cft = replace(point.cft, primaries=tuple(point.cft.primaries) + extra)
        point = replace(point, cft=cft)

    return replace(point, init_values=init_values), stats


def resolve_flexdelta_to_numeric(cft: CFTData, params_vec: np.ndarray) -> CFTData:
    """Evaluate every sigmoid-parameterized dimension of ``cft`` at
    ``params_vec``; returns a CFTData with numeric primaries only (no
    truncation applied)."""
    if len(params_vec) != len(cft.variables):
        raise ValueError(
            f"params length {len(params_vec)} != n_variables {len(cft.variables)}"
        )
    var_values = {
        (v.head, *v.args): torch.tensor(float(params_vec[i]), dtype=torch.float64)
        for i, v in enumerate(cft.variables)
    }

    primaries: list[DSD] = []
    for primary in cft.primaries:
        if isinstance(primary.delta, (int, float)):
            delta = float(primary.delta)
        else:
            if not isinstance(primary.delta, WLExpr):
                raise TypeError(f"Unsupported symbolic delta: {primary.delta!r}")
            value = eval_wl(primary.delta, var_values)
            delta = float(value) if isinstance(value, torch.Tensor) else float(value)
        primaries.append(
            DSD(delta, int(primary.spin), int(primary.degen), primary.character_kind)
        )

    return CFTData(
        name=f"{cft.name}_resolved",
        central_charge=float(cft.central_charge),
        primaries=tuple(primaries),
        max_delta=float(cft.max_delta),
        variables=(),
    )


# ============================================================================
# The compactified free boson: a known solution at every gap below 1/2.
# ============================================================================


def free_boson_primaries(
    R_squared: float,
    max_delta: float,
    *,
    include_chiral: bool = True,
    chiral_block_for_R2_2: bool = False,
) -> tuple[DSD, ...]:
    """Virasoro primaries of the c = 1 compactified boson at radius R, up to
    ``Delta <= max_delta``.

    Vertex operators ``V_{p,w}``, ``p, w`` integers::

        h_L = (p/R + w R)^2 / 4,   h_R = (p/R - w R)^2 / 4,
        Delta = (p^2/R^2 + w^2 R^2) / 2,   spin = h_L - h_R = p w.

    ``spin`` is stored as ``|spin|``: only non-negative signed spins are
    emitted, the mirror partners being implicit in the loss. Primaries with
    equal ``(Delta, spin)`` are merged by summing degeneracies. In this
    convention ``R^2 = 2`` reproduces the cached reference spectrum
    (``primal/data/reference/fbcb_sqrt2_*.json``) with
    ``chiral_block_for_R2_2=True``.

    ``include_chiral`` adds the radius-independent chiral-algebra primaries:
    the U(1) current ``J`` at ``(1, 1)`` and ``J Jbar`` at ``(2, 0)``. Away
    from ``R^2 = 1`` the current is given the ``auto`` character kind, so a
    vertex state landing on ``Delta = spin`` keeps a generic character.
    Purely chiral vertex operators (``h_L = 0`` or ``h_R = 0``) are marked
    generic for the same reason. ``chiral_block_for_R2_2`` adds the extra
    primaries specific to the rational radius ``R^2 = 2``; leave it off
    elsewhere.

    The lightest scalar vertex operator is ``(p, w) = (1, 0)`` at
    ``Delta = 1/(2 R^2)``, so ``R^2 = 1/(2 delta_gap)`` places a primary
    exactly at the gap.
    """
    r2_is_exact_one = abs(float(R_squared) - 1.0) < 1e-12
    spec: dict[tuple[float, int, str], int] = defaultdict(int)

    def add(delta: float, spin: int, degen: int = 1, character_kind: str = "auto") -> None:
        spec[(round(float(delta), 10), int(spin), character_kind)] += int(degen)

    add(0.0, 0)  # vacuum

    if include_chiral:
        current_kind = "conserved" if r2_is_exact_one else "auto"
        add(1.0, 1, character_kind=current_kind)  # J; Jbar implicit
        add(2.0, 0)                                # J Jbar
        if chiral_block_for_R2_2:
            add(3.0, 1, degen=2)
            if max_delta >= 9.0 - 1e-9:
                add(9.0, 7, degen=2)

    delta_buf = float(max_delta) + 1e-9
    R = float(np.sqrt(R_squared))
    p_max = int(math.ceil(R * math.sqrt(2 * max_delta + 0.5))) + 2
    w_max = int(math.ceil(math.sqrt(2 * max_delta + 0.5) / R)) + 2

    for p in range(-p_max, p_max + 1):
        for w in range(-w_max, w_max + 1):
            if p == 0 and w == 0:
                continue  # vacuum already counted
            h_L = (p / R + w * R) ** 2 / 4.0
            h_R = (p / R - w * R) ** 2 / 4.0
            delta = h_L + h_R
            if delta > delta_buf:
                continue
            signed_spin = h_L - h_R
            if signed_spin < -1e-12:
                continue  # mirror partner; implicit
            spin_int = int(round(abs(signed_spin)))
            pure_chiral_vertex = min(abs(h_L), abs(h_R)) < 1e-12
            keep_r2_2_kind = chiral_block_for_R2_2 and abs(float(R_squared) - 2.0) < 1e-12
            character_kind = (
                "generic" if pure_chiral_vertex and not keep_r2_2_kind else "auto"
            )
            add(round(delta, 10), spin_int, character_kind=character_kind)

    kind_order = {"vacuum": 0, "conserved": 1, "generic": 2, "auto": 3}
    items = sorted(
        spec.items(),
        key=lambda kv: (kv[0][0], kv[0][1], kind_order.get(kv[0][2], 9)),
    )
    return tuple(
        DSD(delta=float(d), spin=int(s), degen=int(deg), character_kind=kind)
        for (d, s, kind), deg in items
    )


def free_boson_point(
    c: float,
    delta_gap: float,
    max_delta: int | float,
    num_spin: dict[int, int] | None = None,
) -> SurveyPoint:
    """A survey point at ``(c, delta_gap)`` seeded on the free boson with a
    primary at the gap: ``R^2 = 1/(2 g)``, ``g = min(delta_gap, 1/2)``.

    Near c = 1 the low-gap strip is this family of solutions, and a
    random restart cannot find it in a space of a hundred or more
    parameters, so it is constructed and handed to the optimizer. Above
    ``delta_gap = 1/2`` the boson has no primary at the gap; the seed then
    stays at the self-dual radius ``R^2 = 1``.
    """
    source_gap = min(float(delta_gap), 0.5)
    spectrum = free_boson_primaries(1.0 / (2.0 * source_gap), max_delta=float(max_delta))
    point, _stats = build_warm_started_survey_point(
        c=float(c),
        delta_gap=float(delta_gap),
        max_delta=float(max_delta),
        spectrum=spectrum,
        num_spin=num_spin,
        source_delta_gap=source_gap,
    )
    return point
