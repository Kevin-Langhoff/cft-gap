"""Plain data structures describing a candidate 2D CFT spectrum.

A conformal field theory in 2D is specified, for our purposes, by its
central charge and its list of primary operators. Each primary carries a
scaling dimension (energy), a spin, and a degeneracy (how many operators
share that dimension and spin). ``CFTData`` below is just that: a
central charge plus a tuple of primaries.

During the search (see ``optimize.py``), most of a candidate's primaries
have a *fixed numeric* dimension, but a handful are left free so the
optimizer can move them — these carry a small symbolic expression tree
(``WLExpr``) instead of a plain number, built from a sigmoid so the
dimension stays inside a chosen window as the underlying parameter
ranges over all reals. ``WLExpr`` supports just enough arithmetic head
types (``Plus``, ``Times``, ``Power``, ``Exp``, ``Log``) to express that
sigmoid; ``partition.py`` and ``optimize.py`` evaluate it once numeric
parameter values are substituted in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

import numpy as np


@dataclass(frozen=True)
class WLExpr:
    """A generic symbolic expression node: ``head[*args]``.

    Atomic leaves (numbers, strings, symbolic names with no arguments) are
    represented as plain Python values and never wrapped in ``WLExpr``.
    """

    head: str
    args: tuple[Any, ...]

    def __repr__(self) -> str:
        return f"{self.head}[{', '.join(repr(a) for a in self.args)}]"


Expr = Union[WLExpr, float, int, str]
CharacterKind = Literal["auto", "generic", "vacuum", "conserved"]
_VALID_CHARACTER_KINDS = frozenset(("auto", "generic", "vacuum", "conserved"))


@dataclass(frozen=True)
class DSD:
    """Dimension / spin / degeneracy of a single Virasoro primary operator."""

    delta: Expr
    spin: int
    degen: int
    character_kind: CharacterKind = "auto"

    def __post_init__(self) -> None:
        if self.character_kind not in _VALID_CHARACTER_KINDS:
            raise ValueError(
                "character_kind must be one of "
                f"{sorted(_VALID_CHARACTER_KINDS)}, got {self.character_kind!r}"
            )


def _auto_character_kind(delta: Expr, spin: int) -> CharacterKind:
    if isinstance(delta, (int, float)):
        delta_f = float(delta)
        spin_f = float(spin)
        if delta_f == spin_f and delta_f.is_integer():
            return "vacuum" if int(spin) == 0 else "conserved"
    return "generic"


def resolved_character_kind(primary: DSD) -> CharacterKind:
    """Resolve the explicit or legacy-inferred Virasoro character type.

    ``auto`` preserves the historical rule: a numeric integer dimension
    equal to the spin uses the shortened vacuum/conserved-current
    character (whose partition-function contribution has a closed form
    independent of the general formula); every other case is generic.
    """
    if primary.character_kind == "auto":
        return _auto_character_kind(primary.delta, int(primary.spin))
    return primary.character_kind


def uses_short_character(primary: DSD) -> bool:
    return resolved_character_kind(primary) in ("vacuum", "conserved")


@dataclass(frozen=True)
class CFTData:
    name: str
    central_charge: float
    primaries: tuple[DSD, ...]
    max_delta: float
    variables: tuple[WLExpr, ...]

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    @property
    def n_primaries(self) -> int:
        return len(self.primaries)


@dataclass(frozen=True)
class EtaDeformedCFTData:
    """A reference CFT rescaled to a nearby central charge.

    Used to build the "reference" model that the optimizer compares a
    candidate against: the denominator of the loss function (see
    ``optimize.py``) measures how much a candidate's partition function
    could still change if its missing high-dimension operators looked
    like this reference's spectrum instead. ``cut`` is the dimension
    truncation used when building that reference's series expansion.
    """

    base: CFTData
    new_central_charge: float
    cut: float
