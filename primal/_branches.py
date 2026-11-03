"""Split a candidate's operator list into the four branches the
batched optimizer's vectorized partition-function code treats
differently.

The partition-function contribution of an operator takes a different
closed form depending on its type:

* the identity (vacuum) operator,
* a conserved current (integer dimension equal to its spin),
* a generic scalar (spin 0, non-integer or non-conserved dimension),
* a generic tensor (spin > 0, same).

``optimize.py`` needs each of these grouped separately so it can batch
the corresponding tensor ops across every operator of a given type at
once, across every grid point simultaneously. Keeping the split in one
place (rather than duplicating the classification logic at each call
site) prevents the two from drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data_types import CFTData, resolved_character_kind, uses_short_character


@dataclass
class PrimaryPartition:
    """Pure-Python four-way partition. Callers wrap into torch tensors /
    nn.Module buffers as they see fit."""

    vac_degen_sum: float
    curr_spins: list[float]
    curr_degens: list[float]
    scal_idx: list[int]
    scal_degens: list[float]
    tens_idx: list[int]
    tens_spins: list[float]
    tens_degens: list[float]
    integer_mask: list[bool]


def partition_primaries(cft: CFTData) -> PrimaryPartition:
    spins = [int(p.spin) for p in cft.primaries]
    degens = [int(p.degen) for p in cft.primaries]
    integer_mask = [uses_short_character(p) for p in cft.primaries]

    vac_degen_sum = 0.0
    curr_spins: list[float] = []
    curr_degens: list[float] = []
    scal_idx: list[int] = []
    scal_degens: list[float] = []
    tens_idx: list[int] = []
    tens_spins: list[float] = []
    tens_degens: list[float] = []

    for i, (p, s, deg) in enumerate(zip(cft.primaries, spins, degens)):
        kind = resolved_character_kind(p)
        if kind == "vacuum":
            vac_degen_sum += float(deg)
        elif kind == "conserved":
            curr_spins.append(float(s))
            curr_degens.append(float(deg))
        elif s == 0:
            scal_idx.append(i)
            scal_degens.append(float(deg))
        else:
            tens_idx.append(i)
            tens_spins.append(float(s))
            tens_degens.append(float(deg))

    return PrimaryPartition(
        vac_degen_sum=vac_degen_sum,
        curr_spins=curr_spins,
        curr_degens=curr_degens,
        scal_idx=scal_idx,
        scal_degens=scal_degens,
        tens_idx=tens_idx,
        tens_spins=tens_spins,
        tens_degens=tens_degens,
        integer_mask=integer_mask,
    )
