"""The modular-invariance loss and the batched search optimizer.

Two pieces live here:

1. **The loss** (``objective_function`` / ``loss_mean_on_taus`` /
   ``per_tau_squared_loss``): a numeric measure of how badly a candidate
   spectrum's partition function fails to be invariant under tau ->
   -1/tau, sampled on a set of torus points and averaged. Zero would
   mean perfect modular invariance; in practice we test whether a
   candidate can be tuned to keep this loss below a fixed threshold.
   The loss is optionally normalized by an error estimate
   ``sigma_trunc(tau)`` built from a reference spectrum via
   ``partition.objective_denominator`` — this converts the raw
   numerator into something like a chi-squared statistic, so "loss < 1"
   means "consistent with modular invariance to within the
   truncation error," not just "small in absolute terms."

2. **The batched optimizer** (``run_batched_survey``): given many
   (c, delta_gap) candidates built by ``spectrum.build_survey_point``,
   run all of them through the same gradient-free local search at once,
   as one vectorized computation with a shared leading "which point"
   tensor axis. Each step:

   * samples a fresh batch of random torus points per candidate,
   * evaluates the loss and its Jacobian with respect to the free
     operator-dimension parameters,
   * takes a truncated-SVD (Gauss-Newton-like) step along the best
     descent directions, accepting only prefixes of the step that
     actually reduce the training loss,
   * tracks the best point seen so far by loss on a held-out set of
     torus points.

   After the fixed number of steps, every candidate's best-found
   spectrum is re-scored on a large, fixed, deterministic grid of torus
   points (the "refined" loss) — this is the number that ultimately
   classifies the (c, delta_gap) point as a CFT candidate or an
   excluded region.

Numerical note: this local search is a chaotic dynamical system in the
free parameters — a change at the 1e-15 level in an intermediate
computation can and does flip which region of parameter space a
300-iteration trajectory ends up in. None of the arithmetic here should
be reordered, rewritten "equivalently," or have its default
hyperparameters changed; doing so silently changes which (c, delta_gap)
points get classified as consistent.
"""

from __future__ import annotations

import gc
import hashlib
import math
import multiprocessing as _mp
import os
import pickle
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ._branches import partition_primaries
from ._expr_eval import eval_wl
from .data_types import CFTData, DSD, EtaDeformedCFTData, WLExpr
from .partition import objective_denominator as _deformed_denominator
from .partition import reduced_partition_function
from .spectrum import SurveyPoint

_TWO_PI = 2.0 * math.pi


# ============================================================================
# Loss: modular-invariance residual, optionally chi-squared-normalized.
# ============================================================================

Reference = Optional[EtaDeformedCFTData]


def objective_numerator(
    model: CFTData, tau: complex | np.ndarray
) -> np.ndarray:
    """``Z_reduced(tau) - Z_reduced(-1/tau)`` — zero iff exactly modular
    invariant at this tau."""
    tau_arr = np.asarray(tau, dtype=np.complex128)
    return reduced_partition_function(model, tau_arr) - reduced_partition_function(
        model, -1.0 / tau_arr
    )


def _apply_sigma_epsilon(sigma: np.ndarray, sigma_epsilon: float) -> np.ndarray:
    """Smooth reliability-weighted denominator ``sigma_eff = sqrt(sigma^2 + eps^2)``.

    ``eps = 0`` is a no-op (returns ``sigma`` unchanged bit-identically,
    apart from the dtype cast that ``np.asarray`` performs on the raw
    ``sigma``)."""
    if sigma_epsilon == 0.0:
        return sigma
    sigma = np.asarray(sigma)
    return np.sqrt(sigma * sigma + float(sigma_epsilon) ** 2)


_FP64_EPS = float(np.finfo(np.float64).eps)


def _vacuum_reduced_abs(c: float, tau1, tau2):
    """``|Z_reduced|`` of the vacuum alone, ``sqrt(tau2) e^{2 pi tau2 (c-1)/12}
    |1-q|^2`` — a model-independent proxy (within a factor 1-3, measured on
    three exact spectra) for the size of the numbers whose difference forms
    ``R``. Works on numpy arrays and torch tensors alike."""
    e2 = _np_or_torch_exp(tau2, -2.0 * math.pi)
    one_minus_q_sq = 1.0 - 2.0 * e2 * _np_or_torch_cos(tau1) + e2 * e2
    growth = _np_or_torch_exp(tau2, 2.0 * math.pi * (c - 1.0) / 12.0)
    return tau2 ** 0.5 * growth * one_minus_q_sq


def _np_or_torch_exp(x, k):
    return torch.exp(k * x) if isinstance(x, torch.Tensor) else np.exp(k * x)


def _np_or_torch_cos(tau1):
    return (torch.cos(2.0 * math.pi * tau1) if isinstance(tau1, torch.Tensor)
            else np.cos(2.0 * math.pi * tau1))


def _sigma_round(c: float, tau1, tau2, kappa: float):
    """Per-tau fp64 rounding error of ``R``: ``kappa * eps * max(|Z(tau)|,
    |Z(-1/tau)|)`` with the vacuum proxy for ``|Z|``. Same array type in as
    out."""
    r2 = tau1 * tau1 + tau2 * tau2
    tau1_s, tau2_s = -tau1 / r2, tau2 / r2
    z = _vacuum_reduced_abs(c, tau1, tau2)
    z_s = _vacuum_reduced_abs(c, tau1_s, tau2_s)
    zmax = torch.maximum(z, z_s) if isinstance(z, torch.Tensor) else np.maximum(z, z_s)
    return float(kappa) * _FP64_EPS * zmax


def _apply_sigma_round(sigma: np.ndarray, c: float, tau: np.ndarray,
                       sigma_round_kappa: float) -> np.ndarray:
    """``sigma_eff = sqrt(sigma^2 + sigma_round^2)``: the rounding error of
    ``R`` added in quadrature as a real uncertainty. ``kappa = 0`` is a
    bit-identical no-op."""
    if sigma_round_kappa == 0.0:
        return sigma
    tau_arr = np.asarray(tau, dtype=np.complex128)
    s_round = _sigma_round(float(c), tau_arr.real, tau_arr.imag, sigma_round_kappa)
    sigma = np.asarray(sigma, dtype=np.float64)
    return np.sqrt(sigma * sigma + s_round * s_round)


def objective_denominator(
    model: CFTData,
    reference: Reference,
    tau: complex | np.ndarray,
    sigma_epsilon: float = 0.0,
    sigma_round_kappa: float = 0.0,
) -> np.ndarray:
    """Returns ``1`` for the plain (unnormalized) loss, or the reference-
    built error estimate ``sigma_trunc(tau)`` for the chi-squared-style
    loss, optionally smoothed to ``sigma_eff = sqrt(sigma^2 + eps^2)``
    when ``sigma_epsilon > 0`` and/or with the per-tau fp64 rounding error
    added in quadrature when ``sigma_round_kappa > 0``."""
    tau_arr = np.asarray(tau, dtype=np.complex128)
    if reference is None:
        sigma = np.ones_like(tau_arr, dtype=np.float64)
    else:
        sigma = _deformed_denominator(model.max_delta, reference, tau_arr)
    sigma = _apply_sigma_epsilon(sigma, sigma_epsilon)
    return _apply_sigma_round(sigma, model.central_charge, tau_arr, sigma_round_kappa)


def objective_function(
    model: CFTData,
    reference: Reference,
    tau: complex | np.ndarray,
    sigma_epsilon: float = 0.0,
    sigma_round_kappa: float = 0.0,
) -> np.ndarray:
    """``objective_numerator / objective_denominator``."""
    num = objective_numerator(model, tau)
    if reference is None and sigma_epsilon == 0.0 and sigma_round_kappa == 0.0:
        return num
    denom = objective_denominator(model, reference, tau, sigma_epsilon=sigma_epsilon,
                                  sigma_round_kappa=sigma_round_kappa)
    return num / denom


_VALID_LOSS_TRANSFORMS = ("none", "log1p")


def _apply_loss_transform(per_tau_sq: np.ndarray, loss_transform: str) -> np.ndarray:
    """Apply the optional per-tau squared-loss transform. ``'none'`` is a
    no-op; ``'log1p'`` returns ``log(1 + (R/sigma)^2)`` elementwise."""
    if loss_transform == "none":
        return per_tau_sq
    if loss_transform == "log1p":
        return np.log1p(per_tau_sq)
    raise ValueError(
        f"loss_transform must be one of {_VALID_LOSS_TRANSFORMS}, got {loss_transform!r}"
    )


def per_tau_squared_loss(
    model: CFTData,
    reference: Reference,
    taus: np.ndarray,
    sigma_epsilon: float = 0.0,
    loss_transform: str = "none",
    sigma_round_kappa: float = 0.0,
) -> np.ndarray:
    """``(Re[objective_function(tau)])^2`` per tau.

    When ``loss_transform='log1p'``, returns ``log(1 + (R/sigma)^2)`` per
    tau instead of ``(R/sigma)^2``. At ``loss_transform='none'``
    (default) this function is bit-identical to the plain squared
    residual."""
    b = np.real(objective_function(model, reference, taus, sigma_epsilon=sigma_epsilon,
                                   sigma_round_kappa=sigma_round_kappa))
    per_tau_sq = b * b
    return _apply_loss_transform(per_tau_sq, loss_transform)


def loss_mean_on_taus(
    model: CFTData,
    reference: Reference,
    taus: np.ndarray,
    sigma_epsilon: float = 0.0,
    loss_transform: str = "none",
) -> float:
    """``mean((Re[objective_function(tau)])^2)`` — scalar test-set loss.

    See the ``per_tau_squared_loss`` docstring for ``sigma_epsilon`` and
    ``loss_transform`` semantics; both default to no-op (plain mean
    squared residual)."""
    return float(per_tau_squared_loss(model, reference, taus,
                                      sigma_epsilon=sigma_epsilon,
                                      loss_transform=loss_transform).mean())


# ============================================================================
# Sigmoid pattern match — primaries whose Delta = min + scale * sigmoid(x_v)
# for some variable x_v. Survey primaries always match this; falls back
# with an informative error if a point has a non-sigmoid Delta expression.
# ============================================================================


def _match_flex_sigmoid(expr, var_keys):
    if not isinstance(expr, WLExpr) or expr.head != "Plus" or len(expr.args) != 2:
        return None
    min_val, rhs = expr.args
    if not isinstance(min_val, (int, float)):
        return None
    if not isinstance(rhs, WLExpr) or rhs.head != "Times" or len(rhs.args) != 2:
        return None
    scale, pwr = rhs.args
    if not isinstance(scale, (int, float)):
        return None
    if not isinstance(pwr, WLExpr) or pwr.head != "Power" or len(pwr.args) != 2:
        return None
    inner, exponent = pwr.args
    if not (isinstance(exponent, (int, float)) and float(exponent) == -1.0):
        return None
    if not isinstance(inner, WLExpr) or inner.head != "Plus" or len(inner.args) != 2:
        return None
    one, exp_term = inner.args
    if not (isinstance(one, (int, float)) and float(one) == 1.0):
        return None
    if not isinstance(exp_term, WLExpr) or exp_term.head != "Exp":
        return None
    neg_x = exp_term.args[0]
    if not isinstance(neg_x, WLExpr) or neg_x.head != "Times" or len(neg_x.args) != 2:
        return None
    neg_one, flex = neg_x.args
    if not (isinstance(neg_one, (int, float)) and float(neg_one) == -1.0):
        return None
    if not isinstance(flex, WLExpr) or flex.head != "FlexDelta":
        return None
    key = ("FlexDelta", *flex.args)
    if key not in var_keys:
        return None
    return float(min_val), float(scale), var_keys.index(key)


# ----------------------------------------------------------------------------
# Batched context: all the per-point stacks + the shared branch structure.
# ----------------------------------------------------------------------------


@dataclass
class _BranchStructure:
    """Primary-partition metadata shared across all points at fixed Delta_max."""

    n_prim: int
    n_var: int
    vac_degen_sum: float
    # Vacuum/conserved-current branches — tau-only (no c, no Delta).
    curr_spins: torch.Tensor
    curr_degens: torch.Tensor
    # Scalar (non-integer Delta, spin=0) branch, flex primaries only.
    svs_var_idx: torch.Tensor
    svs_sig_pos: torch.Tensor
    svs_degens: torch.Tensor
    # Tensor (non-integer Delta, spin>0) branch, flex primaries only.
    tvs_var_idx: torch.Tensor
    tvs_sig_pos: torch.Tensor
    tvs_spins: torch.Tensor
    tvs_degens: torch.Tensor
    # Per-point constant scalar primaries (e.g. pseudogap at Delta=Delta_gap).
    # Indexed via point-independent "which primaries are constant" list; the
    # numeric values are stacked per-point in BatchedSurveyContext.
    const_prim_idx: list[int]
    # Sigmoid bookkeeping.
    sig_prim_idx: list[int]
    sig_var_idx: list[int]


@dataclass
class BatchedSurveyContext:
    """Stacks N survey points into tensors with a leading ``N`` axis."""

    points: list[SurveyPoint]
    structure: _BranchStructure

    # (N,) per-point central charge
    c_batch: torch.Tensor
    # (N, P_sig) sigmoid mins / scales
    sig_mins: torch.Tensor
    sig_scales: torch.Tensor
    # (N, N_const) constant scalar Delta per point (e.g. pseudogap = Delta_gap)
    const_deltas: torch.Tensor
    # (N_const,) shared degens — always 1 in the current survey
    const_degens: torch.Tensor

    # Mutable params: (N, P_var), the FlexDelta x values per point.
    params: torch.Tensor

    @property
    def n_points(self) -> int:
        return self.c_batch.shape[0]

    @property
    def p_var(self) -> int:
        return self.params.shape[1]


def _extract_structure(point: SurveyPoint) -> _BranchStructure:
    cft = point.cft
    n_prim = len(cft.primaries)
    n_var = len(cft.variables)

    partition = partition_primaries(cft)
    integer_mask = partition.integer_mask
    vac_degen_sum = partition.vac_degen_sum
    curr_spins = partition.curr_spins
    curr_degens = partition.curr_degens
    scal_idx = partition.scal_idx
    scal_degens = partition.scal_degens
    tens_idx = partition.tens_idx
    tens_spins = partition.tens_spins
    tens_degens = partition.tens_degens

    var_keys = tuple((v.head, *v.args) for v in cft.variables)
    sig_prim_idx: list[int] = []
    sig_var_idx_list: list[int] = []
    const_prim_idx: list[int] = []
    for i, p in enumerate(cft.primaries):
        if isinstance(p.delta, (int, float)):
            if integer_mask[i]:
                # Absorbed into vacuum / current branches above.
                continue
            const_prim_idx.append(i)
            continue
        matched = _match_flex_sigmoid(p.delta, var_keys)
        if matched is None:
            raise ValueError(
                f"Primary {i} Delta expression did not match the sigmoid pattern. "
                "Batched optimizer only supports sigmoid-parameterized survey points."
            )
        _, _, vi = matched
        sig_prim_idx.append(i)
        sig_var_idx_list.append(vi)

    sig_pos_of_prim = {pi: k for k, pi in enumerate(sig_prim_idx)}
    scal_pos_of_prim = {pi: pos for pos, pi in enumerate(scal_idx)}
    tens_pos_of_prim = {pi: pos for pos, pi in enumerate(tens_idx)}

    svs_var, svs_degen, svs_sig_pos = [], [], []
    tvs_var, tvs_degen, tvs_sig_pos, tvs_spin = [], [], [], []
    for k, pi in enumerate(sig_prim_idx):
        if pi in scal_pos_of_prim:
            svs_var.append(sig_var_idx_list[k])
            svs_degen.append(scal_degens[scal_pos_of_prim[pi]])
            svs_sig_pos.append(k)
        elif pi in tens_pos_of_prim:
            tvs_var.append(sig_var_idx_list[k])
            tvs_degen.append(tens_degens[tens_pos_of_prim[pi]])
            tvs_sig_pos.append(k)
            tvs_spin.append(tens_spins[tens_pos_of_prim[pi]])
        else:
            raise ValueError(f"Sigmoid primary {pi} not in scalar/tensor branch.")

    return _BranchStructure(
        n_prim=n_prim,
        n_var=n_var,
        vac_degen_sum=vac_degen_sum,
        curr_spins=torch.tensor(curr_spins, dtype=torch.float64),
        curr_degens=torch.tensor(curr_degens, dtype=torch.float64),
        svs_var_idx=torch.tensor(svs_var, dtype=torch.long),
        svs_sig_pos=torch.tensor(svs_sig_pos, dtype=torch.long),
        svs_degens=torch.tensor(svs_degen, dtype=torch.float64),
        tvs_var_idx=torch.tensor(tvs_var, dtype=torch.long),
        tvs_sig_pos=torch.tensor(tvs_sig_pos, dtype=torch.long),
        tvs_spins=torch.tensor(tvs_spin, dtype=torch.float64),
        tvs_degens=torch.tensor(tvs_degen, dtype=torch.float64),
        const_prim_idx=const_prim_idx,
        sig_prim_idx=sig_prim_idx,
        sig_var_idx=sig_var_idx_list,
    )


def _per_point_sigmoid(
    point: SurveyPoint, structure: _BranchStructure
) -> tuple[torch.Tensor, torch.Tensor]:
    mins, scales = [], []
    var_keys = tuple((v.head, *v.args) for v in point.cft.variables)
    for pi in structure.sig_prim_idx:
        matched = _match_flex_sigmoid(point.cft.primaries[pi].delta, var_keys)
        assert matched is not None
        mn, sc, _ = matched
        mins.append(mn)
        scales.append(sc)
    return (
        torch.tensor(mins, dtype=torch.float64),
        torch.tensor(scales, dtype=torch.float64),
    )


def build_context(points: list[SurveyPoint]) -> BatchedSurveyContext:
    """Build a ``BatchedSurveyContext`` from ``N`` survey points.

    All points must share branch structure (enforced by ``Delta_max`` +
    ``num_spin`` being identical across them). This is automatic via
    ``spectrum.build_survey_point``.
    """
    assert len(points) > 0
    structure = _extract_structure(points[0])
    # Sanity check: branch sizes are identical across points.
    for p in points[1:]:
        st = _extract_structure(p)
        assert st.n_prim == structure.n_prim and st.n_var == structure.n_var, (
            "All points must share n_prim and n_var (same Delta_max + num_spin)."
        )

    c_batch = torch.tensor([p.c for p in points], dtype=torch.float64)
    sig_mins_list, sig_scales_list = [], []
    for p in points:
        mn, sc = _per_point_sigmoid(p, structure)
        sig_mins_list.append(mn)
        sig_scales_list.append(sc)
    sig_mins = torch.stack(sig_mins_list)
    sig_scales = torch.stack(sig_scales_list)
    params = torch.stack([
        torch.as_tensor(p.init_values, dtype=torch.float64) for p in points
    ])

    const_vals_list = []
    for p in points:
        vals = [float(p.cft.primaries[i].delta) for i in structure.const_prim_idx]
        const_vals_list.append(torch.tensor(vals, dtype=torch.float64))
    if const_vals_list and structure.const_prim_idx:
        const_deltas = torch.stack(const_vals_list)
    else:
        const_deltas = torch.zeros(len(points), 0, dtype=torch.float64)
    # Constant degens: assume 1.0 per const primary (always true in the survey:
    # the pseudogap primary has degeneracy 1).
    const_degens = torch.ones(const_deltas.shape[1], dtype=torch.float64)

    return BatchedSurveyContext(
        points=list(points),
        structure=structure,
        c_batch=c_batch,
        sig_mins=sig_mins,
        sig_scales=sig_scales,
        const_deltas=const_deltas,
        const_degens=const_degens,
        params=params,
    )


# ----------------------------------------------------------------------------
# Batched tau cache + forward + Jacobian.
# ----------------------------------------------------------------------------


@torch.no_grad()
def _prepare_tau_cache(
    ctx: BatchedSurveyContext,
    taus: torch.Tensor,          # (N, B, 2) — per-point tau
    sigma: torch.Tensor,         # (N, B) — per-point deformed denominator
) -> dict:
    """Build tau-only cache shared by forward_and_jac and batched_losses.

    The cache folds ``1/sigma`` into the prefactors so downstream
    products are already sigma-normalised. All output tensors have a
    leading ``N`` axis.
    """
    N = ctx.n_points
    assert taus.shape == (N, taus.shape[1], 2), f"taus shape {taus.shape}"
    B = taus.shape[1]
    tau1 = taus[:, :, 0]                             # (N, B)
    tau2 = taus[:, :, 1]
    abs_sq = tau1 * tau1 + tau2 * tau2
    tau1_inv = -tau1 / abs_sq
    tau2_inv = tau2 / abs_sq

    c_col = ctx.c_batch[:, None]                     # (N, 1)
    pref_c_arg = -_TWO_PI * 2.0 * (1.0 - c_col) / 24.0
    pref_tau = torch.sqrt(tau2) * torch.exp(pref_c_arg * tau2)
    pref_inv = torch.sqrt(tau2_inv) * torch.exp(pref_c_arg * tau2_inv)
    inv_sigma = 1.0 / sigma
    pref_tau_n = pref_tau * inv_sigma                # (N, B)
    pref_inv_n = pref_inv * inv_sigma
    neg_2pi_tau2 = -_TWO_PI * tau2                   # (N, B)
    neg_2pi_tau2_inv = -_TWO_PI * tau2_inv

    st = ctx.structure

    # Vacuum + conserved-current branches — no c, no Delta dependence, but
    # tau is per-point so these are still (N, B) in batched form.
    base_tau = torch.zeros_like(tau2)
    base_inv = torch.zeros_like(tau2_inv)
    if st.vac_degen_sum != 0.0:
        e_tau = torch.exp(-_TWO_PI * tau2)
        base_tau = base_tau + st.vac_degen_sum * (
            1.0 - 2.0 * e_tau * torch.cos(_TWO_PI * tau1) + e_tau * e_tau
        )
        e_inv = torch.exp(-_TWO_PI * tau2_inv)
        base_inv = base_inv + st.vac_degen_sum * (
            1.0 - 2.0 * e_inv * torch.cos(_TWO_PI * tau1_inv) + e_inv * e_inv
        )
    if st.curr_spins.numel() > 0:
        j = st.curr_spins                             # (P_c,)
        term1 = 2.0 * torch.exp(-_TWO_PI * j[None, :, None] * tau2[:, None, :]) * torch.cos(
            _TWO_PI * j[None, :, None] * tau1[:, None, :]
        )
        term2 = 2.0 * torch.exp(
            -_TWO_PI * (j[None, :, None] + 1) * tau2[:, None, :]
        ) * torch.cos(_TWO_PI * (j[None, :, None] - 1) * tau1[:, None, :])
        base_tau = base_tau + (st.curr_degens[None, :, None] * (term1 - term2)).sum(dim=1)
        term1i = 2.0 * torch.exp(-_TWO_PI * j[None, :, None] * tau2_inv[:, None, :]) * torch.cos(
            _TWO_PI * j[None, :, None] * tau1_inv[:, None, :]
        )
        term2i = 2.0 * torch.exp(
            -_TWO_PI * (j[None, :, None] + 1) * tau2_inv[:, None, :]
        ) * torch.cos(_TWO_PI * (j[None, :, None] - 1) * tau1_inv[:, None, :])
        base_inv = base_inv + (st.curr_degens[None, :, None] * (term1i - term2i)).sum(dim=1)

    # Constant scalar primaries (pseudogap) — per-point Delta, tau-varying
    # contribution.
    if ctx.const_deltas.numel() > 0:
        const_tau = ctx.const_degens[None, :, None] * torch.exp(
            ctx.const_deltas[:, :, None] * neg_2pi_tau2[:, None, :]
        )
        const_inv = ctx.const_degens[None, :, None] * torch.exp(
            ctx.const_deltas[:, :, None] * neg_2pi_tau2_inv[:, None, :]
        )
        base_tau = base_tau + const_tau.sum(dim=1)
        base_inv = base_inv + const_inv.sum(dim=1)

    # Tensor-branch tau-only prefactors: 2*degen*cos(2*pi*j*tau1), per spin.
    if st.tvs_sig_pos.numel() > 0:
        j = st.tvs_spins                               # (N_tvs,)
        tens_prefix_tau = (
            2.0 * st.tvs_degens[None, :, None]
            * torch.cos(_TWO_PI * j[None, :, None] * tau1[:, None, :])
        )                                              # (N, N_tvs, B)
        tens_prefix_inv = (
            2.0 * st.tvs_degens[None, :, None]
            * torch.cos(_TWO_PI * j[None, :, None] * tau1_inv[:, None, :])
        )
    else:
        tens_prefix_tau = torch.empty((N, 0, B), dtype=torch.float64)
        tens_prefix_inv = torch.empty_like(tens_prefix_tau)

    return {
        "pref_tau_n": pref_tau_n,
        "pref_inv_n": pref_inv_n,
        "base_tau": base_tau,
        "base_inv": base_inv,
        "neg_2pi_tau2": neg_2pi_tau2,
        "neg_2pi_tau2_inv": neg_2pi_tau2_inv,
        "tens_prefix_tau": tens_prefix_tau,
        "tens_prefix_inv": tens_prefix_inv,
        "B": B,
    }


@torch.no_grad()
def _forward_R(
    ctx: BatchedSurveyContext, cache: dict
) -> torch.Tensor:
    """Return sigma-normalised ``R: (N, B)`` without building the
    Jacobian ``M``.

    Same arithmetic as ``_forward_and_jac`` up to the final ``R`` line,
    but skips the ``(N, B, P_var)`` zero-allocation and the two
    ``index_copy_`` fills that construct ``M``. Used on the test-loss
    path inside the main Sven loop, where the Jacobian is discarded.
    """
    st = ctx.structure
    N = ctx.n_points
    B = cache["B"]

    pref_tau_n = cache["pref_tau_n"]
    pref_inv_n = cache["pref_inv_n"]
    base_tau = cache["base_tau"]
    base_inv = cache["base_inv"]
    neg_2pi_tau2 = cache["neg_2pi_tau2"]
    neg_2pi_tau2_inv = cache["neg_2pi_tau2_inv"]

    sig_var_idx_all = torch.tensor(st.sig_var_idx, dtype=torch.long)
    x_all = ctx.params.index_select(1, sig_var_idx_all)
    sig_all = torch.sigmoid(x_all)
    sig_deltas = ctx.sig_mins + ctx.sig_scales * sig_all

    if st.svs_sig_pos.numel() > 0:
        d_svs = sig_deltas.index_select(1, st.svs_sig_pos)
        svs_sum_tau = (st.svs_degens[None, :, None] * torch.exp(
            d_svs[:, :, None] * neg_2pi_tau2[:, None, :]
        )).sum(dim=1)
        svs_sum_inv = (st.svs_degens[None, :, None] * torch.exp(
            d_svs[:, :, None] * neg_2pi_tau2_inv[:, None, :]
        )).sum(dim=1)
    else:
        svs_sum_tau = torch.zeros(N, B, dtype=torch.float64)
        svs_sum_inv = torch.zeros(N, B, dtype=torch.float64)

    if st.tvs_sig_pos.numel() > 0:
        d_tvs = sig_deltas.index_select(1, st.tvs_sig_pos)
        tvs_sum_tau = (cache["tens_prefix_tau"] * torch.exp(
            d_tvs[:, :, None] * neg_2pi_tau2[:, None, :]
        )).sum(dim=1)
        tvs_sum_inv = (cache["tens_prefix_inv"] * torch.exp(
            d_tvs[:, :, None] * neg_2pi_tau2_inv[:, None, :]
        )).sum(dim=1)
    else:
        tvs_sum_tau = torch.zeros(N, B, dtype=torch.float64)
        tvs_sum_inv = torch.zeros(N, B, dtype=torch.float64)

    return pref_tau_n * (base_tau + svs_sum_tau + tvs_sum_tau) - pref_inv_n * (
        base_inv + svs_sum_inv + tvs_sum_inv
    )


@torch.no_grad()
def _forward_and_jac(
    ctx: BatchedSurveyContext, cache: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sigma-normalised ``R: (N, B)`` and ``M: (N, B, P_var)``."""
    st = ctx.structure
    N = ctx.n_points
    P = ctx.p_var
    B = cache["B"]

    pref_tau_n = cache["pref_tau_n"]
    pref_inv_n = cache["pref_inv_n"]
    base_tau = cache["base_tau"]
    base_inv = cache["base_inv"]
    neg_2pi_tau2 = cache["neg_2pi_tau2"]
    neg_2pi_tau2_inv = cache["neg_2pi_tau2_inv"]

    sig_var_idx_all = torch.tensor(st.sig_var_idx, dtype=torch.long)
    x_all = ctx.params.index_select(1, sig_var_idx_all)      # (N, P_sig)
    sig_all = torch.sigmoid(x_all)
    sig_deltas = ctx.sig_mins + ctx.sig_scales * sig_all      # (N, P_sig)
    dsig_all = sig_all * (1.0 - sig_all)
    dDelta_dx_all = ctx.sig_scales * dsig_all                  # (N, P_sig)

    if st.svs_sig_pos.numel() > 0:
        d_svs = sig_deltas.index_select(1, st.svs_sig_pos)    # (N, N_svs)
        raw_svs_tau = st.svs_degens[None, :, None] * torch.exp(
            d_svs[:, :, None] * neg_2pi_tau2[:, None, :]
        )                                                      # (N, N_svs, B)
        raw_svs_inv = st.svs_degens[None, :, None] * torch.exp(
            d_svs[:, :, None] * neg_2pi_tau2_inv[:, None, :]
        )
        svs_sum_tau = raw_svs_tau.sum(dim=1)                   # (N, B)
        svs_sum_inv = raw_svs_inv.sum(dim=1)
    else:
        raw_svs_tau = torch.zeros(N, 0, B, dtype=torch.float64)
        raw_svs_inv = torch.zeros(N, 0, B, dtype=torch.float64)
        svs_sum_tau = torch.zeros(N, B, dtype=torch.float64)
        svs_sum_inv = torch.zeros(N, B, dtype=torch.float64)

    if st.tvs_sig_pos.numel() > 0:
        d_tvs = sig_deltas.index_select(1, st.tvs_sig_pos)    # (N, N_tvs)
        raw_tvs_tau = cache["tens_prefix_tau"] * torch.exp(
            d_tvs[:, :, None] * neg_2pi_tau2[:, None, :]
        )
        raw_tvs_inv = cache["tens_prefix_inv"] * torch.exp(
            d_tvs[:, :, None] * neg_2pi_tau2_inv[:, None, :]
        )
        tvs_sum_tau = raw_tvs_tau.sum(dim=1)
        tvs_sum_inv = raw_tvs_inv.sum(dim=1)
    else:
        raw_tvs_tau = torch.zeros(N, 0, B, dtype=torch.float64)
        raw_tvs_inv = torch.zeros(N, 0, B, dtype=torch.float64)
        tvs_sum_tau = torch.zeros(N, B, dtype=torch.float64)
        tvs_sum_inv = torch.zeros(N, B, dtype=torch.float64)

    R = pref_tau_n * (base_tau + svs_sum_tau + tvs_sum_tau) - pref_inv_n * (
        base_inv + svs_sum_inv + tvs_sum_inv
    )                                                          # (N, B)

    M = torch.zeros(N, B, P, dtype=torch.float64)
    fac_tau = pref_tau_n * neg_2pi_tau2                        # (N, B)
    fac_inv = pref_inv_n * neg_2pi_tau2_inv

    if st.svs_sig_pos.numel() > 0:
        dsig_svs = dDelta_dx_all.index_select(1, st.svs_sig_pos)   # (N, N_svs)
        dR_dDelta = (
            fac_tau[:, None, :] * raw_svs_tau - fac_inv[:, None, :] * raw_svs_inv
        )                                                           # (N, N_svs, B)
        cols = dR_dDelta * dsig_svs[:, :, None]
        M.index_copy_(2, st.svs_var_idx, cols.permute(0, 2, 1))
    if st.tvs_sig_pos.numel() > 0:
        dsig_tvs = dDelta_dx_all.index_select(1, st.tvs_sig_pos)
        dR_dDelta = (
            fac_tau[:, None, :] * raw_tvs_tau - fac_inv[:, None, :] * raw_tvs_inv
        )
        cols = dR_dDelta * dsig_tvs[:, :, None]
        M.index_copy_(2, st.tvs_var_idx, cols.permute(0, 2, 1))
    return R, M


@torch.no_grad()
def _batched_losses(
    ctx: BatchedSurveyContext,
    params_batch: torch.Tensor,     # (N, k_cand, P_var)
    cache: dict,
) -> torch.Tensor:
    """sigma-normalised ``mean((R/sigma)^2, dim=-1)`` for ``(N, k_cand)``
    candidates."""
    st = ctx.structure
    N = ctx.n_points
    k = params_batch.shape[1]
    B = cache["B"]

    pref_tau_n = cache["pref_tau_n"]
    pref_inv_n = cache["pref_inv_n"]
    base_tau = cache["base_tau"]
    base_inv = cache["base_inv"]
    neg_2pi_tau2 = cache["neg_2pi_tau2"]
    neg_2pi_tau2_inv = cache["neg_2pi_tau2_inv"]

    sig_var_idx_all = torch.tensor(st.sig_var_idx, dtype=torch.long)
    # params_batch[:, :, sig_var_idx_all] -> (N, k, P_sig).
    x = params_batch.index_select(2, sig_var_idx_all)
    sig = torch.sigmoid(x)
    sig_deltas = (
        ctx.sig_mins[:, None, :] + ctx.sig_scales[:, None, :] * sig
    )                                                     # (N, k, P_sig)

    if st.svs_sig_pos.numel() > 0:
        d_svs = sig_deltas.index_select(2, st.svs_sig_pos)         # (N, k, N_svs)
        svs_contrib_tau = (
            st.svs_degens[None, None, :, None]
            * torch.exp(d_svs[:, :, :, None] * neg_2pi_tau2[:, None, None, :])
        ).sum(dim=2)                                               # (N, k, B)
        svs_contrib_inv = (
            st.svs_degens[None, None, :, None]
            * torch.exp(d_svs[:, :, :, None] * neg_2pi_tau2_inv[:, None, None, :])
        ).sum(dim=2)
    else:
        svs_contrib_tau = torch.zeros(N, k, B, dtype=torch.float64)
        svs_contrib_inv = torch.zeros(N, k, B, dtype=torch.float64)

    if st.tvs_sig_pos.numel() > 0:
        d_tvs = sig_deltas.index_select(2, st.tvs_sig_pos)        # (N, k, N_tvs)
        tvs_contrib_tau = (
            cache["tens_prefix_tau"][:, None, :, :]
            * torch.exp(d_tvs[:, :, :, None] * neg_2pi_tau2[:, None, None, :])
        ).sum(dim=2)                                               # (N, k, B)
        tvs_contrib_inv = (
            cache["tens_prefix_inv"][:, None, :, :]
            * torch.exp(d_tvs[:, :, :, None] * neg_2pi_tau2_inv[:, None, None, :])
        ).sum(dim=2)
    else:
        tvs_contrib_tau = torch.zeros(N, k, B, dtype=torch.float64)
        tvs_contrib_inv = torch.zeros(N, k, B, dtype=torch.float64)

    contrib_tau = base_tau[:, None, :] + svs_contrib_tau + tvs_contrib_tau
    contrib_inv = base_inv[:, None, :] + svs_contrib_inv + tvs_contrib_inv

    R_norm = (
        pref_tau_n[:, None, :] * contrib_tau
        - pref_inv_n[:, None, :] * contrib_inv
    )                                                              # (N, k, B)
    return (R_norm * R_norm).mean(dim=2)                           # (N, k)


# ----------------------------------------------------------------------------
# SVD step with per-point fallback for ill-conditioned Jacobians.
# ----------------------------------------------------------------------------


def _safe_batched_svd(M: torch.Tensor, k_sv: int, rtol: float, active: torch.Tensor):
    """Batched truncated SVD with per-point fallback.

    Tries a single batched ``torch.linalg.svd(M, full_matrices=False)``. If
    that fails (``_LinAlgError`` on some GPU/CPU backends for singular
    inputs), falls back to per-point SVD and marks the failing points as
    inactive for future iterations.

    Returns (U, S, Vh, kept: (N,)) with zeroed rows for failed points.
    """
    N, B, P = M.shape
    try:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        # Shapes: U (N,B,min(B,P)), S (N,min(B,P)), Vh (N,min(B,P),P)
        U = U[:, :, :k_sv]
        S = S[:, :k_sv]
        Vh = Vh[:, :k_sv, :]
        return U, S, Vh, active.clone()
    except torch._C._LinAlgError:
        pass
    # Fallback: per-point. Slow path only hit when some (c, Delta_gap) grid
    # points have pathological Jacobians.
    U_list, S_list, Vh_list = [], [], []
    new_active = active.clone()
    for i in range(N):
        if not bool(active[i]):
            U_list.append(torch.zeros(B, k_sv, dtype=torch.float64))
            S_list.append(torch.zeros(k_sv, dtype=torch.float64))
            Vh_list.append(torch.zeros(k_sv, P, dtype=torch.float64))
            continue
        try:
            Ui, Si, Vhi = torch.linalg.svd(M[i], full_matrices=False)
            Ui = Ui[:, :k_sv]
            Si = Si[:k_sv]
            Vhi = Vhi[:k_sv, :]
            U_list.append(Ui)
            S_list.append(Si)
            Vh_list.append(Vhi)
        except torch._C._LinAlgError:
            new_active[i] = False
            U_list.append(torch.zeros(B, k_sv, dtype=torch.float64))
            S_list.append(torch.zeros(k_sv, dtype=torch.float64))
            Vh_list.append(torch.zeros(k_sv, P, dtype=torch.float64))
    return (
        torch.stack(U_list),
        torch.stack(S_list),
        torch.stack(Vh_list),
        new_active,
    )


# ----------------------------------------------------------------------------
# One Sven iteration: forward + Jacobian + SVD + per-point accept/reject.
# ----------------------------------------------------------------------------


@torch.no_grad()
def _sven_step(
    ctx: BatchedSurveyContext,
    train_taus: torch.Tensor,        # (N, B, 2)
    train_sigma: torch.Tensor,       # (N, B)
    lr: float,
    k_sv: int,
    rtol: float,
    active: torch.Tensor,            # (N,) bool
    damping_lambda: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One batched Sven step.

    Returns (new_active, train_loss_per_point, num_sv_used_per_point).
    Mutates ``ctx.params`` in place.

    ``damping_lambda`` (default 0.0) optionally applies a
    Levenberg-Marquardt / Tikhonov filter ``S/(S^2+lambda)`` to the SVD
    pseudoinverse step instead of the hard ``1/sigma`` cut. lambda=0 is
    bit-identical to the original hard-cut step.
    """
    cache = _prepare_tau_cache(ctx, train_taus, train_sigma)
    R, M = _forward_and_jac(ctx, cache)                         # (N, B), (N, B, P)

    starting_train_loss = (R * R).mean(dim=1)                   # (N,)

    U, S, Vh, active_post = _safe_batched_svd(M, k_sv, rtol, active)
    # Rank-1 directions: dir_i = V_i * (1/sigma_i) * U_i^T * (-R)
    # Per-point SV threshold: keep sigma_i if > sigma_max * rtol (and > 0).
    sigma_max = S[:, :1]
    # To avoid div-by-zero on inactive / zero-SV points, clamp. Loss will be
    # NaN-guarded below.
    sigma_floor = torch.clamp(sigma_max * rtol, min=1e-300)
    keep_mask = S > sigma_floor                                 # (N, k_sv)
    if damping_lambda > 0.0:
        # Levenberg-Marquardt / Tikhonov filter factor S/(S^2+lambda): smoothly
        # down-weights small singular values rather than the hard 1/sigma used
        # below. lambda has units of sigma^2. keep_mask still drops
        # numerically-zero directions; as lambda->0+ this branch approaches
        # the hard-cut branch on kept directions, so the lambda=0 (else)
        # path stays bit-identical.
        S_inv = torch.where(
            keep_mask, S / (S * S + damping_lambda), torch.zeros_like(S)
        )
    else:
        S_inv = torch.where(
            keep_mask, 1.0 / torch.clamp(S, min=1e-300), torch.zeros_like(S)
        )
    u_t_neg_r = -torch.einsum("nbk,nb->nk", U, R)               # (N, k_sv)
    # rank1 = Vh^T . diag(S_inv . u_t_neg_r) viewed column-wise.
    # Shape (N, P, k_sv).
    rank1_stack = Vh.transpose(1, 2) * (S_inv * u_t_neg_r).unsqueeze(1)

    # Cumulative prefixes: (N, P, k_sv) via cumsum along k_sv axis.
    cum_grads = lr * torch.cumsum(rank1_stack, dim=2)
    # Candidate parameter vectors: (N, k_sv, P_var).
    original_params = ctx.params.clone()
    candidates = (original_params[:, :, None] + cum_grads).transpose(1, 2)

    losses = _batched_losses(ctx, candidates, cache)             # (N, k_sv)

    # Longest-true-prefix per row: ok = losses < starting_train_loss.
    ok = losses < starting_train_loss[:, None]
    # Treat NaN / Inf as "not ok" (protects against failed points).
    ok = ok & torch.isfinite(losses)
    # Inactive points never accept (keep their params frozen).
    ok = ok & active_post[:, None]

    # Number of accepted prefix steps per point: sum of leading-True prefix.
    accepted = torch.cumprod(ok.to(torch.int8), dim=1).sum(dim=1)   # (N,)

    # Apply update: for each point, if accepted > 0, set params to the
    # accepted-th candidate; else keep original.
    any_accept = accepted > 0
    new_params = original_params.clone()
    if any_accept.any():
        idx_pts = torch.nonzero(any_accept, as_tuple=True)[0]
        idx_k = accepted[idx_pts] - 1
        new_params[idx_pts] = candidates[idx_pts, idx_k]
    ctx.params.copy_(new_params)

    # Per-point realized train loss after update.
    realized_loss = torch.where(
        any_accept,
        losses.gather(1, torch.clamp(accepted - 1, min=0)[:, None]).squeeze(1),
        starting_train_loss,
    )
    return active_post, realized_loss, accepted


# ----------------------------------------------------------------------------
# One PLAIN gradient-descent iteration: forward + Jacobian + backtrack.
# The pedagogical foil to _sven_step; never used by the scan (step="sven").
# ----------------------------------------------------------------------------


@torch.no_grad()
def _gd_step(
    ctx: BatchedSurveyContext,
    train_taus: torch.Tensor,        # (N, B, 2)
    train_sigma: torch.Tensor,       # (N, B)
    lr: float,
    active: torch.Tensor,            # (N,) bool
    n_backtrack: int = 80,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One batched plain gradient-descent step.

    Steps along ``-M^T R`` -- the true gradient of the sigma-normalised
    ``mean((R/sigma)^2)``, since ``_forward_and_jac`` returns both already
    normalised -- with NO pseudoinverse and NO singular-value truncation.
    That omission IS the point: this exists only to show, in the movie,
    what the SVD step in ``_sven_step`` buys on a problem whose Jacobian
    spans ~12 orders of magnitude.

    The accept contract deliberately matches ``_sven_step``: a
    backtracking ladder ``lr, lr/2, ... lr/2^(n_backtrack-1)`` takes the
    LARGEST step that decreases the train loss, else stands still. So
    neither optimizer can diverge and neither gets a step-control
    advantage -- what differs on screen is conditioning alone, which is
    the honest comparison. Returns ``(new_active, train_loss_per_point,
    accepted)`` and mutates ``ctx.params`` in place, exactly as
    ``_sven_step`` does, so the caller needs no special-casing.

    ``n_backtrack=80`` is MEASURED, not padding, and must not be trimmed
    to a "reasonable" 10-20. At the allowed anchor (c=81/70, gap=3/35)
    iteration 0 has ``|grad| ~ 2.3e15`` and ``cond(M) ~ 8.5e13``, so the
    largest step that decreases the loss is ~5.4e-17 -- while a 12-rung
    ladder from lr=0.1 bottoms out at 4.9e-5, twelve orders of magnitude
    short. A short ladder therefore accepts NOTHING and draws a
    perfectly flat curve, which reads as "GD fails" but is really "the
    line search never found the scale". 80 rungs reach 2^-80 ~ 8e-25.
    """
    cache = _prepare_tau_cache(ctx, train_taus, train_sigma)
    R, M = _forward_and_jac(ctx, cache)                         # (N, B), (N, B, P)

    starting_train_loss = (R * R).mean(dim=1)                   # (N,)

    grad = torch.einsum("nbp,nb->np", M, R)                     # (N, P)
    # Freeze points with a non-finite gradient. _sven_step gets this for
    # free from _safe_batched_svd's per-point fallback; without an SVD
    # there is no analogous guard, so it is done explicitly here.
    active_post = active & torch.isfinite(grad).all(dim=1)

    # Ladder of step LENGTHS along one direction, largest first: (n_back,).
    scales = lr * torch.pow(
        torch.tensor(0.5, dtype=torch.float64),
        torch.arange(n_backtrack, dtype=torch.float64),
    )
    candidates = (ctx.params[:, None, :]
                  - scales[None, :, None] * grad[:, None, :])   # (N, n_back, P)

    losses = _batched_losses(ctx, candidates, cache)            # (N, n_back)

    ok = (losses < starting_train_loss[:, None]) & torch.isfinite(losses)
    ok = ok & active_post[:, None]

    # FIRST true, not longest prefix (contrast _sven_step): these candidates
    # are alternative step lengths, so the first that works is the largest
    # that works. _sven_step's are cumulative prefixes of a rank-1 expansion,
    # where the longest accepted prefix is the right pick.
    any_accept = ok.any(dim=1)
    first_ok = torch.argmax(ok.to(torch.int8), dim=1)           # 0 where none

    new_params = ctx.params.clone()
    if any_accept.any():
        idx = torch.nonzero(any_accept, as_tuple=True)[0]
        new_params[idx] = candidates[idx, first_ok[idx]]
    ctx.params.copy_(new_params)

    realized_loss = torch.where(
        any_accept,
        losses.gather(1, first_ok[:, None]).squeeze(1),
        starting_train_loss,
    )
    return active_post, realized_loss, any_accept.to(torch.int64)


# ----------------------------------------------------------------------------
# Test-tau evaluation: no Jacobian, just sigma-normalised mean((R/sigma)^2).
# ----------------------------------------------------------------------------


@torch.no_grad()
def _test_loss_batched(
    ctx: BatchedSurveyContext,
    test_taus: torch.Tensor,         # (N, B_test, 2)
    test_sigma: torch.Tensor,        # (N, B_test)
    *,
    cache: dict | None = None,
) -> torch.Tensor:
    """Per-point mean((R/sigma)^2) on the held-out tau grid.

    If ``cache`` is supplied (built once via ``_prepare_tau_cache`` when
    test tau/sigma are loop-invariant), the per-call cache rebuild is
    skipped. The forward path uses ``_forward_R`` so the Jacobian ``M``
    is never materialised. The no-cache branch is kept for external
    callers (diagnostics) that invoke this function one-off.
    """
    if cache is None:
        cache = _prepare_tau_cache(ctx, test_taus, test_sigma)
    R = _forward_R(ctx, cache)
    return (R * R).mean(dim=1)


# ----------------------------------------------------------------------------
# tau sampling.
# ----------------------------------------------------------------------------


def _taus_like_wl(n: int, seed: int, tau2_max: float = 6.0) -> np.ndarray:
    # Returns a complex128 ndarray of shape (n,).
    #
    # RNG consumption pattern: two rng.uniform(size=batch) calls per
    # outer iter, same per-iter batch sizes. The acceptance-mask
    # predicate and per-batch ordering must be preserved exactly — the
    # Delta_max=4 regression gate enforces this.
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.complex128)
    filled = 0
    while filled < n:
        batch = max(4 * n, 32)
        u = rng.uniform(0.0, 1.0, size=batch)
        t1 = rng.uniform(0.0, 0.5, size=batch)
        t2 = (np.sqrt(3) / 2.0) / (1.0 - u)
        mask = (t1 * t1 + t2 * t2 >= 1.0) & (t2 <= tau2_max)
        t1_sel = t1[mask]
        t2_sel = t2[mask]
        take = min(t1_sel.size, n - filled)
        out[filled:filled + take] = t1_sel[:take] + 1j * t2_sel[:take]
        filled += take
    return out


def _point_seed(c: float, delta_gap: float) -> int:
    h = hashlib.sha256(f"{c:.6f}|{delta_gap:.6f}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def _taus_numpy_to_real(taus_complex: np.ndarray) -> torch.Tensor:
    """(B,) complex -> (B, 2) real torch."""
    return torch.tensor(
        np.stack([taus_complex.real, taus_complex.imag], axis=-1),
        dtype=torch.float64,
    )


# ----------------------------------------------------------------------------
# Public entry point.
# ----------------------------------------------------------------------------


@dataclass
class BatchedSurveyResult:
    """Per-point result.

    ``flex_delta`` and ``max_delta`` are optional serialization fields that
    let downstream read-only diagnostics reconstruct the converged
    spectrum without re-running the optimizer. Older scan JSONs written
    before these fields existed remain readable — consumers must
    tolerate their absence.
    """

    c: float
    delta_gap: float
    best_test_loss_track: float
    best_iteration: int
    best_loss_refined: float
    n_primaries_after_trunc: int
    wall_time: float
    failure: str | None = None
    best_loss_refined_log1p: float | None = None
    flex_delta: list[float] | None = None
    max_delta: float | None = None
    test_loss_history: list[float] | None = None
    flex_params_history: list[list[float]] | None = None

    def to_dict(self) -> dict:
        d = {
            "c": self.c,
            "delta_gap": self.delta_gap,
            "best_test_loss_track": self.best_test_loss_track,
            "best_iteration": self.best_iteration,
            "best_loss_refined": self.best_loss_refined,
            "n_primaries_after_trunc": self.n_primaries_after_trunc,
            "wall_time": self.wall_time,
        }
        if self.failure is not None:
            d["failure"] = self.failure
        if self.best_loss_refined_log1p is not None:
            d["best_loss_refined_log1p"] = self.best_loss_refined_log1p
        if self.flex_delta is not None:
            d["flex_delta"] = list(self.flex_delta)
        if self.max_delta is not None:
            d["max_delta"] = float(self.max_delta)
        if self.test_loss_history is not None:
            d["test_loss_history"] = list(self.test_loss_history)
        if self.flex_params_history is not None:
            d["flex_params_history"] = [list(r) for r in self.flex_params_history]
        return d


# Worker-process globals (populated by _worker_init via spawn initializer).
_WORKER_FBCB: CFTData | None = None


def _worker_init(fbcb_pickle: bytes) -> None:
    """Initializer for `spawn` worker processes.

    Unpickles the reference CFT once per worker to avoid re-serializing
    the ~15 MB object on every task. Also caps intra-worker torch
    threading so N spawn workers x default torch thread count doesn't
    over-commit the CPU.
    """
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Only settable before any parallel work; ignore if already set.
        pass
    global _WORKER_FBCB
    _WORKER_FBCB = pickle.loads(fbcb_pickle)


def _sigma_one_point_worker(
    task: tuple[float, float, int, bytes],
) -> list[bytes]:
    """Compute sigma for all iters of one (c, Delta_gap) point.

    ``task`` = (c, max_delta, num_iter, flat_tau_complex_pickle) where the
    flat tau array is the concatenation of all num_iter x B per-iter tau
    samples. The caller pre-splits per iter from the flat result; we
    return a list of pickled (B,) float64 numpy arrays (one per iter)
    because torch tensors across process boundaries occasionally trip
    over GPU-availability checks in spawned subprocesses — numpy is
    safer and just as fast.
    """
    c, max_delta, num_iter, flat_tau_pickle = task
    assert _WORKER_FBCB is not None, "worker not initialized"
    flat_tau: np.ndarray = pickle.loads(flat_tau_pickle)
    ref = EtaDeformedCFTData(
        base=_WORKER_FBCB,
        new_central_charge=float(c),
        cut=float(max_delta) + 2.0,
    )
    flat_sigma = np.asarray(
        _deformed_denominator(max_delta, ref, flat_tau), dtype=np.float64
    )
    B = flat_sigma.shape[0] // num_iter
    assert flat_sigma.shape[0] == B * num_iter, (
        f"unexpected sigma shape {flat_sigma.shape}, B*num_iter={B*num_iter}"
    )
    chunks = flat_sigma.reshape(num_iter, B)
    return [pickle.dumps(chunks[i]) for i in range(num_iter)]


def _precompute_sigma(
    points: list[SurveyPoint],
    reference_base: CFTData,
    tau_lists: list[list[np.ndarray]],    # tau_lists[p] is a list of (B,) complex per iter
    pool=None,
) -> list[list[torch.Tensor]]:
    """For each point, build its reference (rescaled to new_c=c,
    cut=max_delta+2) and evaluate sigma on the concatenated tau grid.
    Returns list[list[Tensor]] shaped [N_points][num_iter] with each
    inner tensor of shape (B,).

    If ``pool`` is not None, dispatches one task per point to a worker
    process. Worker pool must have been initialized with ``_worker_init``
    so that ``_WORKER_FBCB`` is populated.
    """
    if pool is None:
        # Serial fast path.
        result = []
        for p, tau_seq in zip(points, tau_lists):
            ref = EtaDeformedCFTData(
                base=reference_base,
                new_central_charge=p.c,
                cut=float(p.cft.max_delta) + 2.0,
            )
            flat = np.concatenate(tau_seq)
            flat_sigma = _deformed_denominator(p.cft.max_delta, ref, flat)
            flat_t = torch.as_tensor(np.asarray(flat_sigma, dtype=np.float64))
            chunks = torch.split(flat_t, [len(t) for t in tau_seq])
            result.append(list(chunks))
        return result

    # Parallel path.
    tasks = []
    for p, tau_seq in zip(points, tau_lists):
        flat = np.concatenate(tau_seq)
        num_iter = len(tau_seq)
        # Assume uniform per-iter B (always true for the survey pipeline).
        tasks.append((p.c, p.cft.max_delta, num_iter, pickle.dumps(flat)))

    result = []
    for chunk_pickles in pool.imap(_sigma_one_point_worker, tasks, chunksize=1):
        chunks = [
            torch.as_tensor(pickle.loads(b), dtype=torch.float64)
            for b in chunk_pickles
        ]
        result.append(chunks)
    return result


def _refined_one_point_worker(
    task: tuple[bytes, np.ndarray, bytes, float],
) -> tuple[float, int]:
    """Compute refined fine-grid loss for one point.

    ``task`` = (survey_point_pickle, best_params_np, fine_taus_pickle,
    sigma_epsilon). Returns (L_refined, L_refined_log1p,
    n_primaries_after_trunc). Mirrors the serial per-point body of
    ``_refined_loss_per_point``.
    """
    sp_pickle, best_params_np, fine_taus_pickle, sigma_epsilon, *extra = task
    sigma_round_kappa = float(extra[0]) if extra else 0.0
    assert _WORKER_FBCB is not None, "worker not initialized"
    sp: SurveyPoint = pickle.loads(sp_pickle)
    fine_taus: np.ndarray = pickle.loads(fine_taus_pickle)

    var_values = {
        (v.head, *v.args): torch.tensor(
            float(best_params_np[j]), dtype=torch.float64
        )
        for j, v in enumerate(sp.cft.variables)
    }
    truncated: list[DSD] = []
    for p in sp.cft.primaries:
        if isinstance(p.delta, (int, float)):
            dv = float(p.delta)
        else:
            dv = float(eval_wl(p.delta, var_values))
        if dv <= sp.cft.max_delta + 1e-9:
            truncated.append(
                DSD(
                    delta=dv,
                    spin=int(p.spin),
                    degen=int(p.degen),
                    character_kind=p.character_kind,
                )
            )
    cft_trunc = CFTData(
        name=sp.cft.name,
        central_charge=sp.c,
        primaries=tuple(truncated),
        max_delta=sp.cft.max_delta,
        variables=(),
    )
    ref = EtaDeformedCFTData(
        base=_WORKER_FBCB,
        new_central_charge=sp.c,
        cut=float(sp.cft.max_delta) + 2.0,
    )
    # Compute the per-tau squared loss once; raw mean and log1p-mean are
    # both cheap transforms on the same array. Avoids re-evaluating the
    # full numerator/denominator stack.
    per_tau_sq = per_tau_squared_loss(
        cft_trunc, ref, fine_taus, sigma_epsilon=sigma_epsilon,
        sigma_round_kappa=sigma_round_kappa,
    )
    L = float(per_tau_sq.mean())
    L_log1p = float(np.log1p(per_tau_sq).mean())
    return L, L_log1p, len(truncated)


def _refined_loss_per_point(
    ctx: BatchedSurveyContext,
    best_params: torch.Tensor,          # (N, P_var)
    reference_base: CFTData,
    fine_taus_complex: np.ndarray,      # (B_fine,) complex
    pool=None,
    sigma_epsilon: float = 0.0,
    sigma_round_kappa: float = 0.0,
) -> tuple[list[float], list[float], list[int]]:
    """Hard-truncate best-iter spectrum + evaluate improved loss on the
    large deterministic tau grid. Returns
    ``(loss_raw_list, loss_log1p_list, n_primaries_list)``.

    Both raw and log1p losses are always computed — the extra evaluation
    is cheap (<10 ms per point) and the log1p value is the preferred
    physics-interpretable metric at higher truncation cutoffs.

    ``sigma_epsilon`` applies the smooth reliability floor
    ``sigma_eff = sqrt(sigma^2 + eps^2)`` to the refined-eval denominator;
    default 0.0 is bit-identical to the pre-epsilon pipeline.

    If ``pool`` is not None, dispatches one task per point to a worker
    process. Worker pool must have been initialized with ``_worker_init``.
    """
    if pool is None:
        losses_refined: list[float] = []
        losses_log1p: list[float] = []
        n_prims_list: list[int] = []
        for i, sp in enumerate(ctx.points):
            var_values = {
                (v.head, *v.args): torch.tensor(
                    float(best_params[i, j]), dtype=torch.float64
                )
                for j, v in enumerate(sp.cft.variables)
            }
            truncated: list[DSD] = []
            for p in sp.cft.primaries:
                if isinstance(p.delta, (int, float)):
                    dv = float(p.delta)
                else:
                    dv = float(eval_wl(p.delta, var_values))
                if dv <= sp.cft.max_delta + 1e-9:
                    truncated.append(
                        DSD(
                            delta=dv,
                            spin=int(p.spin),
                            degen=int(p.degen),
                            character_kind=p.character_kind,
                        )
                    )
            cft_trunc = CFTData(
                name=sp.cft.name,
                central_charge=sp.c,
                primaries=tuple(truncated),
                max_delta=sp.cft.max_delta,
                variables=(),
            )
            ref = EtaDeformedCFTData(
                base=reference_base,
                new_central_charge=sp.c,
                cut=float(sp.cft.max_delta) + 2.0,
            )
            per_tau_sq = per_tau_squared_loss(
                cft_trunc, ref, fine_taus_complex,
                sigma_epsilon=sigma_epsilon,
                sigma_round_kappa=sigma_round_kappa,
            )
            L = float(per_tau_sq.mean())
            L_log1p = float(np.log1p(per_tau_sq).mean())
            losses_refined.append(L)
            losses_log1p.append(L_log1p)
            n_prims_list.append(len(truncated))
        return losses_refined, losses_log1p, n_prims_list

    # Parallel path. Best params pass as small (P_var,) numpy arrays, not
    # whole (N, P_var) tensor per task — minimizes per-task pickling.
    fine_taus_pickle = pickle.dumps(fine_taus_complex)
    best_np = best_params.detach().cpu().numpy()
    tasks = [
        (pickle.dumps(sp), best_np[i].copy(), fine_taus_pickle, float(sigma_epsilon),
         float(sigma_round_kappa))
        for i, sp in enumerate(ctx.points)
    ]
    losses_refined: list[float] = [0.0] * len(tasks)
    losses_log1p: list[float] = [0.0] * len(tasks)
    n_prims_list: list[int] = [0] * len(tasks)
    for i, (L, L_log1p, n) in enumerate(pool.imap(_refined_one_point_worker, tasks, chunksize=1)):
        losses_refined[i] = L
        losses_log1p[i] = L_log1p
        n_prims_list[i] = n
    return losses_refined, losses_log1p, n_prims_list


def _auto_n_processes() -> int:
    """Default worker count: half of logical CPUs, minimum 2, capped at 8.

    Cap rationale: sigma-series is GIL-bound so parallel speedup is
    sub-linear; past ~4-8 workers additional processes just add
    spawn/IPC overhead without measurable wall-time gain on the paper
    grid.
    """
    cpu = os.cpu_count() or 2
    return max(2, min(8, cpu // 2))


def _resolve_auto_n_chunk(num_iter: int, n_train: int, *, safety: float = 0.4) -> int:
    """Auto-size ``n_chunk`` from available RAM and per-point byte estimate.

    The two dominant memory consumers in the Sven loop are
    ``train_taus_all`` (fp64 complex stored as 2 fp64 channels) and
    ``train_sigma_all`` (fp64). Per point, both are shaped
    ``(num_iter, n_train)`` along the live axes, so the per-point live
    bytes are ``num_iter * n_train * 3 * 8`` (2 tau + 1 sigma, each fp64).

    ``safety`` = 0.4 reserves ~60% of available RAM for everything else
    (ctx params, test tensors, worker-pool workspace, BLAS scratch).

    Returns at least 1 (never zero). If psutil isn't importable or
    ``available`` can't be read, falls back to a conservative default of
    1000 — still helpful at ultra-dense grids, inert at paper grid size.
    """
    per_point_bytes = num_iter * n_train * 3 * 8
    try:
        import psutil
        avail = int(psutil.virtual_memory().available)
    except Exception:
        # Conservative fallback: chunking is the whole point, so pick
        # something that limits tau+sigma to a few GB.
        return 1000
    if per_point_bytes <= 0:
        return 1
    return max(1, int(avail * safety / per_point_bytes))


def _run_sven_chunk(
    chunk_points: list[SurveyPoint],
    reference_base: CFTData,
    *,
    num_iter: int,
    lr: float,
    k_sv: int,
    rtol: float,
    damping_lambda: float = 0.0,
    n_train: int,
    n_test_track: int,
    per_point_offsets_chunk: list[int],
    tau2_max: float | None,
    sigma_epsilon: float,
    make_pool,
    verbose: bool,
    chunk_label: str = "",
    record_history: bool = False,
    step: str = "sven",
    sigma_round_kappa: float = 0.0,
    train_taus_fixed: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str | None], float, torch.Tensor | None, torch.Tensor | None]:
    """Run tau-build + sigma-precompute + Sven loop for a single chunk of
    points.

    This is the inner body of ``run_batched_survey`` factored out so the
    sigma/Sven stage can be executed once per N-chunk at ultra-dense
    grids where holding ``train_taus_all`` + ``train_sigma_all`` for all
    N would exhaust memory. The chunk's ``train_*_all`` tensors are
    allocated inside and go out of scope on return so the next chunk
    starts from a clean memory baseline.

    Returns ``(best_params, best_test_loss, best_iteration,
    failure_messages, loop_wall_s, test_loss_history, params_history)``.
    ``best_params`` has shape ``(len(chunk_points), P_var)``. Caller is
    responsible for concatenating across chunks before the single
    refined-eval pass. ``params_history`` mirrors ``test_loss_history``:
    allocated only under ``record_history``, holding the raw (pre-sigmoid)
    parameter vector per iteration, forward-filled on non-accepted iters.

    The intended semantics are per-point independent (no explicit
    cross-point state in ``_sven_step`` or ``_forward_and_jac``); tau are
    hash-seeded by ``(c, Delta_gap)`` and independent of position-in-batch.
    In practice the 300-iter Sven trajectory is chaotic, so changing
    batch shape by N-chunking can amplify tiny floating-point differences.
    Treat chunking as a memory lever whose landscape classifications and
    minima reachability need separate verification, not as a bit-exact
    trajectory transform.
    """
    assert len(chunk_points) == len(per_point_offsets_chunk)

    ctx = build_context(chunk_points)
    N = ctx.n_points

    t_build = time.perf_counter()

    # Generate per-point tau sequences (same sampler as the reference
    # per-worker survey).
    train_tau_lists: list[list[np.ndarray]] = []
    test_tau_arrays: list[np.ndarray] = []
    # Optional deterministic training set: the SAME fixed grid every iteration
    # (full-batch Gauss-Newton on e.g. the 952-point reporting grid), instead
    # of a fresh random sample per iteration. Used for the honest loss, whose
    # stiff rows near the self-dual point are rarely drawn by random sampling.
    if train_taus_fixed is not None:
        train_taus_fixed = np.asarray(train_taus_fixed, dtype=np.complex128)
        n_train = int(train_taus_fixed.shape[0])
    for i, sp in enumerate(chunk_points):
        seed = _point_seed(sp.c, sp.delta_gap) + per_point_offsets_chunk[i]
        pt_tau2_max = tau2_max if tau2_max is not None else float(sp.cft.max_delta) + 2.0
        if train_taus_fixed is not None:
            train_tau_lists.append([train_taus_fixed for _ in range(num_iter)])
        else:
            train_tau_lists.append([
                np.asarray(
                    _taus_like_wl(n_train, seed=seed + it, tau2_max=pt_tau2_max),
                    dtype=np.complex128,
                )
                for it in range(num_iter)
            ])
        test_tau_arrays.append(
            np.asarray(
                _taus_like_wl(n_test_track, seed=seed + 999_983, tau2_max=pt_tau2_max),
                dtype=np.complex128,
            )
        )

    # Stack train tau into a single (N, num_iter, n_train, 2) real tensor.
    train_taus_all = torch.zeros(N, num_iter, n_train, 2, dtype=torch.float64)
    for i, seq in enumerate(train_tau_lists):
        for t, arr in enumerate(seq):
            train_taus_all[i, t, :, 0] = torch.as_tensor(arr.real, dtype=torch.float64)
            train_taus_all[i, t, :, 1] = torch.as_tensor(arr.imag, dtype=torch.float64)
    test_taus_all = torch.zeros(N, n_test_track, 2, dtype=torch.float64)
    for i, arr in enumerate(test_tau_arrays):
        test_taus_all[i, :, 0] = torch.as_tensor(arr.real, dtype=torch.float64)
        test_taus_all[i, :, 1] = torch.as_tensor(arr.imag, dtype=torch.float64)

    # ---- Pool lifetime 1: sigma-precompute (parallel or serial) ----
    with make_pool() as pool:
        train_sigma_per_point = _precompute_sigma(
            chunk_points, reference_base, train_tau_lists, pool=pool
        )
        train_sigma_all = torch.zeros(N, num_iter, n_train, dtype=torch.float64)
        for i, chunks in enumerate(train_sigma_per_point):
            for t, c in enumerate(chunks):
                train_sigma_all[i, t] = c
        # Drop the large intermediate Python list as soon as it's copied
        # into the final tensor — the list and the tensor together can
        # transiently double memory at the sigma-to-Sven transition.
        del train_sigma_per_point

        # Test-tau sigma: funnel through the same pool (as a synthetic
        # one-iter precompute) or compute serially.
        if pool is not None:
            test_tau_as_lists = [[arr] for arr in test_tau_arrays]
            test_sigma_per_point = _precompute_sigma(
                chunk_points, reference_base, test_tau_as_lists, pool=pool
            )
            test_sigma_list = [chunks[0] for chunks in test_sigma_per_point]
        else:
            test_sigma_list = []
            for p, arr in zip(chunk_points, test_tau_arrays):
                ref = EtaDeformedCFTData(
                    base=reference_base,
                    new_central_charge=p.c,
                    cut=float(p.cft.max_delta) + 2.0,
                )
                s = _deformed_denominator(p.cft.max_delta, ref, arr)
                test_sigma_list.append(
                    torch.as_tensor(np.asarray(s, dtype=np.float64))
                )
        test_sigma_all = torch.stack(test_sigma_list)
    # Pool workers exit here; their memory is reclaimed before Sven loop.

    # Apply sigma_epsilon floor once, post-precompute, pre-Sven: every
    # downstream sigma consumer (_prepare_tau_cache, _forward_R,
    # _forward_and_jac, _batched_losses, _test_loss_batched) reads from
    # these tensors via ``inv_sigma = 1/sigma``, so floored sigma flows
    # through training Sven loss, test-loss tracker, and candidate
    # evaluation uniformly. At sigma_epsilon == 0.0 this is a no-op
    # (branch taken below), preserving bit-identity with the pre-epsilon
    # pipeline.
    if sigma_epsilon > 0.0:
        eps_sq = float(sigma_epsilon) ** 2
        train_sigma_all = torch.sqrt(train_sigma_all * train_sigma_all + eps_sq)
        test_sigma_all = torch.sqrt(test_sigma_all * test_sigma_all + eps_sq)
    # Per-tau fp64 rounding error of R added in quadrature (the principled
    # form of the floor above; see _sigma_round). kappa == 0.0 is a no-op.
    if sigma_round_kappa > 0.0:
        c_col = torch.tensor([float(sp.c) for sp in chunk_points], dtype=torch.float64)
        sr_train = torch.stack([
            _sigma_round(float(c_col[i]), train_taus_all[i, ..., 0],
                         train_taus_all[i, ..., 1], sigma_round_kappa)
            for i in range(N)
        ])
        sr_test = torch.stack([
            _sigma_round(float(c_col[i]), test_taus_all[i, ..., 0],
                         test_taus_all[i, ..., 1], sigma_round_kappa)
            for i in range(N)
        ])
        train_sigma_all = torch.sqrt(train_sigma_all * train_sigma_all + sr_train * sr_train)
        test_sigma_all = torch.sqrt(test_sigma_all * test_sigma_all + sr_test * sr_test)

    t_build = time.perf_counter() - t_build
    if verbose:
        print(
            f"  {chunk_label}build+sigma precompute: {t_build:.2f}s "
            f"(N={N}, iters={num_iter})"
        )

    # ---- Best-iter tracking ----
    best_test_loss = torch.full((N,), float("inf"), dtype=torch.float64)
    best_iteration = torch.full((N,), -1, dtype=torch.long)
    best_params = ctx.params.clone()

    # Optional per-iter test-loss recording for time-to-target scoring.
    # Allocated only when ``record_history=True``; nan-filled for iters
    # where Sven didn't accept any step (test_loss not recomputed). The
    # recording is a pure side-effect — the loss values flow into
    # ``best_test_loss`` exactly as before, so bit-identity is preserved.
    test_loss_history: torch.Tensor | None = None
    if record_history:
        test_loss_history = torch.full(
            (N, num_iter + 1), float("nan"), dtype=torch.float64
        )
    params_history: torch.Tensor | None = None
    if record_history:
        params_history = torch.full(
            (N, num_iter + 1, ctx.params.shape[1]), float("nan"),
            dtype=torch.float64,
        )

    active = torch.ones(N, dtype=torch.bool)
    failure_messages: list[str | None] = [None] * N

    # Test tau/sigma are fixed for the whole loop (built once above). The
    # tau-only cache (prefactors, vacuum/current/constant branches,
    # tensor-branch prefixes) depends only on ``ctx.c_batch`` and
    # the static ``ctx.const_*`` tensors, so it's loop-invariant.
    # Precompute once; num_iter+1 rebuilds collapse to 1.
    test_cache = _prepare_tau_cache(ctx, test_taus_all, test_sigma_all)

    # Iteration 0: log starting test loss.
    initial_test_loss = _test_loss_batched(
        ctx, test_taus_all, test_sigma_all, cache=test_cache
    )
    if test_loss_history is not None:
        test_loss_history[:, 0] = initial_test_loss
    if params_history is not None:
        params_history[:, 0] = ctx.params
    mask = torch.isfinite(initial_test_loss) & (initial_test_loss < best_test_loss)
    if mask.any():
        best_test_loss = torch.where(mask, initial_test_loss, best_test_loss)
        best_iteration = torch.where(
            mask, torch.zeros_like(best_iteration), best_iteration
        )
        best_params[mask] = ctx.params[mask]

    # ---- Main Sven loop (single-process, no pool alive) ----
    t_loop = time.perf_counter()
    for it in range(1, num_iter + 1):
        try:
            if step == "gd":
                active, _, accepted = _gd_step(
                    ctx,
                    train_taus_all[:, it - 1],
                    train_sigma_all[:, it - 1],
                    lr=lr,
                    active=active,
                )
            else:
                active, _, accepted = _sven_step(
                    ctx,
                    train_taus_all[:, it - 1],
                    train_sigma_all[:, it - 1],
                    lr=lr,
                    k_sv=k_sv,
                    rtol=rtol,
                    active=active,
                    damping_lambda=damping_lambda,
                )
        except Exception as e:
            # Last-resort: _safe_batched_svd should catch per-point
            # failures. If something else throws, freeze all points.
            for i in range(N):
                if failure_messages[i] is None:
                    failure_messages[i] = f"{type(e).__name__}: {e}"
            break

        # When no Sven step was accepted, ctx.params is unchanged from
        # the previous iter; test_loss would be identical to the prior
        # value and cannot improve best_test_loss. Skip the test_loss
        # eval entirely. Bit-identical because per-point trajectory only
        # depends on (params, tau, sigma) and we touch none of those.
        if not bool(accepted.any()):
            # If recording history, forward-fill — the loss is by
            # construction identical to iter-1's, so this preserves the
            # invariant that history[:, it] is the test loss at iter it.
            # Critical for cross-run comparability of trajectories.
            if test_loss_history is not None and it > 0:
                test_loss_history[:, it] = test_loss_history[:, it - 1]
            if params_history is not None and it > 0:
                params_history[:, it] = params_history[:, it - 1]
            continue

        # Per-point test loss update.
        test_loss = _test_loss_batched(
            ctx, test_taus_all, test_sigma_all, cache=test_cache
        )
        if test_loss_history is not None:
            test_loss_history[:, it] = test_loss
        if params_history is not None:
            params_history[:, it] = ctx.params
        mask = torch.isfinite(test_loss) & (test_loss < best_test_loss) & active
        if mask.any():
            best_test_loss = torch.where(mask, test_loss, best_test_loss)
            best_iteration = torch.where(
                mask, torch.full_like(best_iteration, it), best_iteration
            )
            best_params[mask] = ctx.params[mask]

        if verbose and it % 50 == 0:
            active_count = int(active.sum())
            print(
                f"  {chunk_label}iter {it:3d}/{num_iter} "
                f"median L_test={float(best_test_loss.nanmedian()):.3g}  "
                f"active={active_count}/{N}"
            )

    t_loop = time.perf_counter() - t_loop
    if verbose:
        print(f"  {chunk_label}loop: {t_loop:.2f}s  ({t_loop/num_iter*1e3:.1f} ms/iter)")

    return (best_params, best_test_loss, best_iteration, failure_messages,
            t_loop, test_loss_history, params_history)


def run_batched_survey(
    points: list[SurveyPoint],
    reference_base: CFTData,
    fine_test_taus: np.ndarray,
    *,
    num_iter: int = 300,
    lr: float = 0.1,
    k_sv: int = 16,
    rtol: float = 1e-12,
    damping_lambda: float = 0.0,
    n_train: int = 32,
    n_test_track: int = 48,
    n_processes: int | None = None,
    verbose: bool = False,
    seed_offset: int | list[int] | None = None,
    tau2_max: float | None = None,
    sigma_epsilon: float = 0.0,
    n_chunk: int | None = None,
    record_history: bool = False,
    step: str = "sven",
    sigma_round_kappa: float = 0.0,
    train_taus_fixed: np.ndarray | None = None,
) -> list[BatchedSurveyResult]:
    """Run the full (c, delta_gap) search as one batched local-search
    optimization.

    Processes all ``N`` points in a single vectorized run. Per-point tau
    are generated with the same hash-seeded sampler regardless of batch
    size; sigma is computed per point against a fresh reference rescaled
    to that point's central charge. Correctness contract: the
    classification of ``best_loss_refined`` against the CFT-candidate /
    excluded thresholds should be stable up to the expected wall-clock
    variance and chaotic 300-iter drift.

    ``n_processes``: if ``None``, auto-select via ``_auto_n_processes``
    (half of logical CPUs, 2..8). Set to ``1`` to force the serial code
    paths — useful as an emergency kill-switch or for reproducibility
    debugging. Parallelism only speeds up the sigma-precompute and
    refined-eval stages; the main Sven loop runs in-process regardless.

    ``num_iter``: 300 is the reference value and the default; lower
    values are accepted for smoke tests but emit a warning — bucket
    counts and CFT-candidate classifications are not meaningful below
    ~200 iters.

    ``sigma_epsilon``: smooth reliability-weighted denominator floor.
    When ``> 0``, replaces sigma by ``sigma_eff = sqrt(sigma^2 + eps^2)``
    throughout the Sven training loss, the test-loss tracker, and the
    refined fine-grid eval. At tau where sigma >> eps behavior is
    essentially unchanged; at boundary tau where sigma << eps (fp64
    noise-floor territory at high truncation cutoffs), the ratio
    ``(R/sigma_eff)^2`` is capped at ``(R/eps)^2``. Default ``0.0`` is
    bit-identical to the pre-change pipeline (no tensor copies, no op).
    Applied on the sigma tensors themselves after precompute and before
    the Sven loop — so `_prepare_tau_cache`, `_forward_R`,
    `_forward_and_jac`, `_batched_losses`, and `_test_loss_batched` all
    inherit the floor through their ``inv_sigma = 1/sigma`` path
    automatically. The sigma-precompute stage itself still computes raw
    sigma.

    ``n_chunk``: internal N-chunking for the sigma-precompute +
    Sven-loop stage. When ``None`` (default) or ``>= len(points)``, a
    single chunk runs the original monolithic path. When set to a
    positive integer smaller than ``len(points)``, the outer c-axis is
    split into contiguous slices of at most ``n_chunk`` points; each
    chunk's ``train_taus_all`` and ``train_sigma_all`` tensors are
    allocated, consumed, and freed before the next chunk starts. Refined
    eval is NOT chunked — one pool over concatenated ``best_params`` so
    the per-chunk pool-spawn cost isn't paid twice. Per-chunk RNG is
    decoupled from chunk size: per-point tau are hash-seeded on
    ``(c, delta_gap)``, not on position-in-batch. This removes the
    obvious RNG coupling, but does not guarantee bit-exact trajectories
    across different chunkings — classifications and below-threshold
    minima are the invariant that's actually preserved, not the
    trajectory itself.
    """
    if num_iter < 300:
        import warnings
        warnings.warn(
            f"num_iter={num_iter} < 300 (reference value); trajectories will "
            "not be meaningful for landscape classification.",
            RuntimeWarning,
            stacklevel=2,
        )

    if n_processes is None:
        n_processes = _auto_n_processes()
    if n_processes < 1:
        raise ValueError(f"n_processes must be >= 1, got {n_processes}")

    if n_chunk is not None and n_chunk <= 0:
        raise ValueError(f"n_chunk must be None or >= 1, got {n_chunk}")

    t_total = time.perf_counter()

    N = len(points)

    if seed_offset is None:
        per_point_offsets = [0] * N
    elif isinstance(seed_offset, int):
        per_point_offsets = [int(seed_offset)] * N
    else:
        assert len(seed_offset) == N, (
            f"seed_offset list length {len(seed_offset)} != N={N}"
        )
        per_point_offsets = [int(x) for x in seed_offset]

    # Two separate ProcessPool lifetimes: one for sigma-precompute, one for
    # refined eval. The Sven loop runs single-process between them; closing
    # the pool before the loop reclaims `n_processes x ~300 MB` of
    # reference-CFT+torch workspace that was otherwise held idle through
    # num_iter iterations of Sven. At cluster scale (many workers + large N)
    # this was the difference between fitting in the memory budget and
    # an out-of-memory kill right at the sigma-to-Sven transition. Paper-grid
    # cost of respawn is ~1 s; amortizes fully at any ultra-dense sweep.
    fbcb_pickle = pickle.dumps(reference_base) if n_processes > 1 else None

    def _make_pool():
        if n_processes > 1:
            return _mp.get_context("spawn").Pool(
                processes=n_processes,
                initializer=_worker_init,
                initargs=(fbcb_pickle,),
            )
        return nullcontext(None)

    # ---- Chunk planning ----
    # ``n_chunk is None`` OR ``n_chunk >= N`` goes through a single-chunk
    # code path that calls ``_run_sven_chunk`` once with all N points.
    # Multi-chunk runs deliberately change batch shape to save memory; exact
    # trajectory equality is not a correctness contract for chaotic Sven runs.
    if n_chunk is None or n_chunk >= N:
        chunk_slices = [slice(0, N)]
    else:
        chunk_slices = [
            slice(i, min(i + n_chunk, N))
            for i in range(0, N, n_chunk)
        ]

    if verbose and len(chunk_slices) > 1:
        proc_note = f"{n_processes} procs" if n_processes > 1 else "serial"
        print(
            f"chunked sigma+Sven: N={N}, n_chunk={n_chunk}, "
            f"{len(chunk_slices)} chunks, {proc_note}"
        )
    elif verbose:
        proc_note = f"{n_processes} procs" if n_processes > 1 else "serial"
        print(f"monolithic sigma+Sven: N={N}, {proc_note}")

    # Per-chunk outputs (concatenated at end before refined eval).
    chunk_best_params: list[torch.Tensor] = []
    chunk_best_test_loss: list[torch.Tensor] = []
    chunk_best_iteration: list[torch.Tensor] = []
    chunk_history: list[torch.Tensor] = []
    chunk_params_history: list[torch.Tensor] = []
    failure_messages: list[str | None] = []
    t_loop_total = 0.0

    for ci, sl in enumerate(chunk_slices):
        chunk_points = points[sl]
        chunk_offsets = per_point_offsets[sl]
        if len(chunk_slices) > 1 and verbose:
            print(
                f"chunk {ci+1}/{len(chunk_slices)}: "
                f"points[{sl.start}:{sl.stop}] "
                f"(c {chunk_points[0].c:.4f}..{chunk_points[-1].c:.4f})"
            )
        chunk_label = (
            f"[chunk {ci+1}/{len(chunk_slices)}] "
            if len(chunk_slices) > 1 else ""
        )
        bp, btl, bit, fm, t_loop_chunk, hist, phist = _run_sven_chunk(
            chunk_points,
            reference_base,
            num_iter=num_iter,
            lr=lr,
            k_sv=k_sv,
            rtol=rtol,
            damping_lambda=damping_lambda,
            n_train=n_train,
            n_test_track=n_test_track,
            per_point_offsets_chunk=chunk_offsets,
            tau2_max=tau2_max,
            sigma_epsilon=sigma_epsilon,
            make_pool=_make_pool,
            verbose=verbose,
            chunk_label=chunk_label,
            record_history=record_history,
            step=step,
            sigma_round_kappa=sigma_round_kappa,
            train_taus_fixed=train_taus_fixed,
        )
        chunk_best_params.append(bp)
        chunk_best_test_loss.append(btl)
        chunk_best_iteration.append(bit)
        if hist is not None:
            chunk_history.append(hist)
        if phist is not None:
            chunk_params_history.append(phist)
        failure_messages.extend(fm)
        t_loop_total += t_loop_chunk
        # Force GC between chunks to drop the chunk's train_taus_all /
        # train_sigma_all before the next chunk allocates its own.
        # `bp/btl/bit` are kept (small: (n_chunk, P_var) + 2x(n_chunk,));
        # the large tensors were only held in the helper's local scope.
        if len(chunk_slices) > 1:
            gc.collect()

    # Concatenate chunk outputs into the full-N tensors the refined-eval
    # stage expects.
    best_params = torch.cat(chunk_best_params, dim=0)
    best_test_loss = torch.cat(chunk_best_test_loss, dim=0)
    best_iteration = torch.cat(chunk_best_iteration, dim=0)
    test_loss_history: torch.Tensor | None = None
    if record_history and chunk_history:
        test_loss_history = torch.cat(chunk_history, dim=0)
    params_history: torch.Tensor | None = None
    if record_history and chunk_params_history:
        params_history = torch.cat(chunk_params_history, dim=0)
    assert best_params.shape[0] == N

    if verbose and len(chunk_slices) > 1:
        print(f"total Sven loop wall across chunks: {t_loop_total:.2f}s")

    # ---- Pool lifetime 2: refined fine-grid evaluation (ALL points) ----
    # Single pool over concatenated best_params — never per-chunk, else
    # the pool-spawn cost would be paid once per chunk and swamp the
    # chunking benefit. This is the invariant that keeps chunking
    # ~free in wall-time at paper-grid N.
    t_refined = time.perf_counter()
    for i in range(N):
        if not torch.isfinite(best_test_loss[i]):
            if failure_messages[i] is None:
                failure_messages[i] = "non-finite training loss"

    # Build a full-N context only for its ``.points`` field — refined
    # eval reads ``ctx.points`` and ``best_params`` and nothing else
    # from ``ctx``. Cheap: build_context is ms per point.
    ctx_all = build_context(points)

    with _make_pool() as pool:
        losses_refined, losses_log1p, n_prims_list = _refined_loss_per_point(
            ctx_all, best_params, reference_base, fine_test_taus, pool=pool,
            sigma_epsilon=sigma_epsilon, sigma_round_kappa=sigma_round_kappa,
        )
    t_refined = time.perf_counter() - t_refined
    if verbose:
        print(f"refined eval: {t_refined:.2f}s")

    best_params_np = best_params.detach().cpu().numpy()

    results: list[BatchedSurveyResult] = []
    for i, sp in enumerate(points):
        if failure_messages[i] is not None:
            results.append(BatchedSurveyResult(
                c=sp.c,
                delta_gap=sp.delta_gap,
                best_test_loss_track=1e30,
                best_iteration=-1,
                best_loss_refined=1e30,
                n_primaries_after_trunc=0,
                wall_time=time.perf_counter() - t_total,
                failure=failure_messages[i],
                max_delta=float(sp.cft.max_delta),
            ))
            continue
        L_ref = losses_refined[i]
        L_ref_log1p = losses_log1p[i]
        if not math.isfinite(L_ref):
            L_ref = 1e30
        if not math.isfinite(L_ref_log1p):
            L_ref_log1p = 1e30
        hist = None
        if test_loss_history is not None:
            hist = [float(x) for x in test_loss_history[i].tolist()]
        phist_i = None
        if params_history is not None:
            phist_i = params_history[i].tolist()
        results.append(BatchedSurveyResult(
            c=sp.c,
            delta_gap=sp.delta_gap,
            best_test_loss_track=float(best_test_loss[i]),
            best_iteration=int(best_iteration[i]),
            best_loss_refined=L_ref,
            best_loss_refined_log1p=L_ref_log1p,
            n_primaries_after_trunc=n_prims_list[i],
            wall_time=time.perf_counter() - t_total,
            flex_delta=[float(x) for x in best_params_np[i]],
            max_delta=float(sp.cft.max_delta),
            test_loss_history=hist,
            flex_params_history=phist_i,
        ))
    return results
