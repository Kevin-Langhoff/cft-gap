"""Evaluate the small symbolic-expression trees used by free operators.

Free (flex) operator dimensions are stored as :class:`WLExpr` trees (see
``data_types.py``) built from a sigmoid — e.g. ``min + span/(1+exp(-x))``
for some free parameter ``x``. This module walks such a tree and
substitutes in concrete numeric parameter values, returning a torch
tensor so the substitution composes with the rest of the optimizer's
tensor pipeline. Supports just the handful of arithmetic heads that
appear in practice: ``Plus``, ``Times``, ``Power``, ``Exp``, ``Log``,
plus the symbol ``E`` and lookups into a map of free-parameter values.
Any other head raises ``ValueError`` — the survey construction never
produces one.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch

from .data_types import WLExpr

VarMap = Mapping[tuple, torch.Tensor]


def eval_wl(expr, var_values: VarMap) -> torch.Tensor:
    """Recursively evaluate ``expr`` to a scalar ``float64`` tensor."""
    if isinstance(expr, (int, float)):
        return torch.tensor(float(expr), dtype=torch.float64)
    if isinstance(expr, str):
        if expr == "E":
            return torch.tensor(math.e, dtype=torch.float64)
        raise ValueError(f"unknown symbol {expr!r}")
    if isinstance(expr, WLExpr):
        head, args = expr.head, expr.args
        if head in ("FlexDelta", "FlexDegen"):
            return var_values[(head, *args)]
        if head == "Plus":
            result = eval_wl(args[0], var_values)
            for a in args[1:]:
                result = result + eval_wl(a, var_values)
            return result
        if head == "Times":
            result = eval_wl(args[0], var_values)
            for a in args[1:]:
                result = result * eval_wl(a, var_values)
            return result
        if head == "Power":
            base = eval_wl(args[0], var_values)
            exponent = eval_wl(args[1], var_values)
            return base**exponent
        if head == "Exp":
            return torch.exp(eval_wl(args[0], var_values))
        if head == "Log":
            return torch.log(eval_wl(args[0], var_values))
        raise ValueError(f"unknown head {head!r}")
    raise TypeError(f"unexpected expression type {type(expr)!r}")
