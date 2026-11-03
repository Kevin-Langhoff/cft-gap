"""Evaluate a CFT's (reduced) partition function on the torus.

Modular invariance says the partition function of a 2D CFT, viewed as a
function of the torus modular parameter tau, must be invariant under the
modular group — in particular under tau -> -1/tau (swapping the two
cycles of the torus). That single constraint, imposed at enough sample
points of tau, is what the search in ``optimize.py`` tries to satisfy.

This module computes the "reduced" partition function: the overall
Dedekind-eta prefactor common to every consistent CFT is peeled off, so
what remains is a sum over primary operators of the state-counting
function of the true dimension. Two variants are supported:

* ``reduced_partition_function`` — the exact reduced partition function
  of a CFT model whose primaries all have concrete numeric dimensions.
* ``deformed_reduced_partition`` / ``objective_denominator`` — the same
  quantity for a "reference" model that has been analytically continued
  to a nearby central charge and re-expanded as a finite series (an
  eta-function deformation). This reference furnishes the built-in
  error estimate used by the loss in ``optimize.py``: it measures how
  much a candidate's still-missing high-dimension operators could
  plausibly shift the numerator, based on what a nearby known spectrum
  looks like.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.special import gammainc, gammaincc

from .data_types import CFTData, DSD, EtaDeformedCFTData, resolved_character_kind


# --------------------------------------------------------------------- #
# Exact reduced partition function (numeric-dimension primaries only).  #
# --------------------------------------------------------------------- #


def q_qbar(tau: complex | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``q = exp(2*pi*i*tau)``, ``qbar = exp(-2*pi*i*conj(tau))``.

    For real-valued tau1, tau2 both are finite and ``qbar = conj(q)``.
    """
    tau = np.asarray(tau, dtype=np.complex128)
    q = np.exp(2j * np.pi * tau)
    qbar = np.exp(-2j * np.pi * np.conj(tau))
    return q, qbar


def reduced_symmetrized_term(
    primary: DSD, q: np.ndarray, qbar: np.ndarray
) -> np.ndarray:
    """Per-primary contribution to the reduced partition function.

    ``primary.delta`` must already be a concrete number.
    """
    delta = float(primary.delta)
    j = primary.spin
    degen = primary.degen
    kind = resolved_character_kind(primary)
    if kind == "vacuum":
        term = (1 - q) * (1 - qbar)
    elif kind == "conserved":
        term = q**j * (1 - qbar) + qbar**j * (1 - q)
    else:
        if j == 0:
            term = q ** (delta / 2) * qbar ** (delta / 2)
        else:
            term = (
                q ** ((delta + j) / 2) * qbar ** ((delta - j) / 2)
                + q ** ((delta - j) / 2) * qbar ** ((delta + j) / 2)
            )
    return degen * term


def reduced_partition_function(
    model: CFTData, tau: complex | np.ndarray
) -> np.ndarray:
    """``Z_reduced(tau) = sqrt(Im tau) * (q*qbar)^((1-c)/24) * sum_p term_p``.

    Accepts a scalar ``tau`` or an array; returns the same shape.

    Vectorized over scalar primaries: the per-primary loop with a Python
    ``sum()`` is replaced by grouped numpy ops. Vacuum and conserved
    primaries (usually a handful) still go through the per-primary path
    since their formulas differ; scalar primaries (the bulk) are batched.
    Output is bit-identical to the per-primary form modulo summation
    order — the partial sums are still combined left-to-right.
    """
    tau_arr = np.asarray(tau, dtype=np.complex128)
    q, qbar = q_qbar(tau_arr)
    c = model.central_charge
    prefactor = np.sqrt(tau_arr.imag) * (q * qbar) ** ((1 - c) / 24)

    # Partition primaries by character kind.
    vac_degens: list[float] = []
    cons_spins: list[int] = []
    cons_degens: list[float] = []
    scalar_deltas: list[float] = []
    scalar_spins: list[int] = []
    scalar_degens: list[float] = []
    for p in model.primaries:
        kind = resolved_character_kind(p)
        if kind == "vacuum":
            vac_degens.append(float(p.degen))
        elif kind == "conserved":
            cons_spins.append(int(p.spin))
            cons_degens.append(float(p.degen))
        else:
            # Generic scalar/tensor primary with numeric delta.
            scalar_deltas.append(float(p.delta))
            scalar_spins.append(int(p.spin))
            scalar_degens.append(float(p.degen))

    # Accumulate term sum.
    terms = np.zeros_like(q)
    # Vacuum: (1 - q)(1 - qbar). Same for every vacuum primary; just scale
    # by the sum of degeneracies.
    if vac_degens:
        vac_sum_deg = float(sum(vac_degens))
        terms = terms + vac_sum_deg * ((1 - q) * (1 - qbar))
    # Conserved: q^j (1-qbar) + qbar^j (1-q), per primary, weighted by degen.
    # Vectorize across the (usually 3-5) conserved primaries.
    if cons_spins:
        jc = np.asarray(cons_spins, dtype=np.int64)            # (C,)
        dc = np.asarray(cons_degens, dtype=np.float64)         # (C,)
        # q[..., None]^jc[None,:] for j in cons_spins. Most spins here
        # are small positive integers; np.power on complex is fine.
        q_j = q[..., None] ** jc[None, :]                       # (B, C)
        qb_j = qbar[..., None] ** jc[None, :]
        cons_term = (
            q_j * (1 - qbar)[..., None] + qb_j * (1 - q)[..., None]
        ) * dc[None, :]                                         # (B, C)
        terms = terms + cons_term.sum(axis=-1)

    if scalar_deltas:
        # For a scalar/tensor primary at (delta, j, degen) with numeric
        # delta, the symmetrized term simplifies (j=0 gives q^(d/2) qbar^(d/2),
        # j>0 gives the sum of two cross-terms). Vectorized form:
        #   degen * (1 if j==0 else 2) * (q*qbar)^(d/2) * cos_term
        # where cos_term = 1 for j==0 and cos_term=cos(2*pi*j*tau1) otherwise.
        # We compute the (q*qbar)^(d/2) factor for all primaries in one
        # broadcast op, then build the spin-dependent multiplier.
        d = np.asarray(scalar_deltas, dtype=np.float64)        # (S,)
        j = np.asarray(scalar_spins, dtype=np.int64)           # (S,)
        degen = np.asarray(scalar_degens, dtype=np.float64)    # (S,)
        # q*qbar = exp(-4*pi*tau2) for tau = tau1 + i*tau2; we use it
        # directly so the (q*qbar)^(d/2) factor is real-valued.
        qq = (q * qbar)                                         # (B,)
        # Broadcast: (B, 1) * (1, S) -> (B, S)
        factor = qq[..., None] ** (d[None, :] / 2.0)            # complex but real-valued
        # Spin contribution: 2*cos(2*pi*j*tau1) for j>0; 1 for j=0.
        # Multiply factor by spin contribution column-wise.
        tau1 = tau_arr.real
        is_j0 = (j == 0)
        mult = np.where(is_j0, degen, 2.0 * degen)              # (S,)
        if np.any(~is_j0):
            cos_mat = np.ones((tau1.shape[0] if tau1.ndim else 1, d.size), dtype=np.float64)
            jpos = j[~is_j0]
            cols = np.where(~is_j0)[0]
            # cos(2*pi*j*tau1) for the spin>0 columns
            t1 = tau1 if tau1.ndim else np.array([tau1])
            cos_mat[:, cols] = np.cos(2 * np.pi * jpos[None, :] * t1[:, None])
            if tau1.ndim == 0:
                cos_mat = cos_mat[0]                            # back to (S,) for scalar tau
            scalar_terms = (factor * (mult[None, :] * cos_mat)).sum(axis=-1)
        else:
            scalar_terms = (factor * mult[None, :]).sum(axis=-1)
        terms = terms + scalar_terms

    return prefactor * terms


def evaluate_on_grid(
    model: CFTData, taus: Iterable[complex]
) -> np.ndarray:
    """Convenience: ``Z_reduced`` on an iterable of taus, returned as a
    1-D complex array."""
    return reduced_partition_function(model, np.fromiter(taus, dtype=np.complex128))


# --------------------------------------------------------------------- #
# Deformed reduced partition function for the eta-deformed reference.   #
# --------------------------------------------------------------------- #
#
# For a base CFT with central charge old_c and a target new_c > old_c,
# the deformed reduced partition function is
#
#     Z_def(tau) = sqrt(tau2) * (q*qbar)^((1-new_c)/24) * tau2^((old_c-new_c)/2)
#                   * sum_Delta  P_Delta(q, qbar) * F(Delta, cut, tau2)
#
# where:
#
# * P_Delta(q, qbar) is the sum of monomials at total dimension
#   Delta = h + hbar in the expansion of
#   (sum_primaries term_p(q, qbar)) * (prod_{n>=1}(1-q^n)(1-qbar^n))^(old_c - new_c);
# * F(Delta, cut, tau2) is the regularized lower incomplete gamma with
#   a = (new_c - old_c)/2, vanishing for Delta >= cut.
#
# The series is computed symbolically using polynomial arithmetic over
# Fraction-indexed monomials so fractional dimensions (e.g. h = k^2/4 for
# a free-boson reference) stay exact. Evaluation at concrete tau is a
# finite sum of complex exponentials.

# --- polynomial arithmetic ------------------------------------------------

Key = Tuple[Fraction, Fraction]
Poly = Dict[Key, float]


_COEF_FLOOR = 1e-14


def _prune(poly: Poly) -> Poly:
    return {k: v for k, v in poly.items() if abs(v) > _COEF_FLOOR}


def _poly_mul(a: Poly, b: Poly, cut: float) -> Poly:
    out: Poly = {}
    for (h1, hb1), c1 in a.items():
        for (h2, hb2), c2 in b.items():
            h = h1 + h2
            hb = hb1 + hb2
            if float(h + hb) > cut + 1e-9:
                continue
            key = (h, hb)
            out[key] = out.get(key, 0.0) + c1 * c2
    return _prune(out)


def _one() -> Poly:
    return {(Fraction(0), Fraction(0)): 1.0}


def _binomial_series(g: Poly, alpha: float, cut: float) -> Poly:
    """``(1 + g)^alpha`` as a formal power series, truncated at total degree
    ``cut``. ``g`` must have no constant term so the series terminates
    naturally (``g^k`` has min degree ``k``)."""
    assert (Fraction(0), Fraction(0)) not in g, "g must lack a constant term"
    result = _one()
    g_k = _one()
    coef = 1.0
    for k in range(1, int(cut) + 2):
        coef *= (alpha - (k - 1)) / k
        g_k = _poly_mul(g_k, g, cut)
        if not g_k:
            break
        for key, v in g_k.items():
            result[key] = result.get(key, 0.0) + coef * v
    return _prune(result)


# --- base building blocks -------------------------------------------------


def _base_partition_terms(cft: CFTData, cut: float) -> Poly:
    """Reduced partition-function terms of ``cft`` as a polynomial in
    ``(q, qbar)`` truncated to total degree ``cut``."""
    out: Poly = {}
    for p in cft.primaries:
        delta_f = float(p.delta)
        j = int(p.spin)
        degen = float(p.degen)
        kind = resolved_character_kind(p)
        if kind == "vacuum":
            monomials = (
                ((Fraction(0), Fraction(0)),  1.0),
                ((Fraction(1), Fraction(0)), -1.0),
                ((Fraction(0), Fraction(1)), -1.0),
                ((Fraction(1), Fraction(1)),  1.0),
            )
        elif kind == "conserved":
            monomials = (
                ((Fraction(j), Fraction(0)),  1.0),
                ((Fraction(j), Fraction(1)), -1.0),
                ((Fraction(0), Fraction(j)),  1.0),
                ((Fraction(1), Fraction(j)), -1.0),
            )
        else:
            delta_q = Fraction(delta_f).limit_denominator(1024)
            half = Fraction(1, 2)
            if j == 0:
                hh = delta_q * half
                monomials = (((hh, hh), 1.0),)
            else:
                j_q = Fraction(j)
                h_plus = (delta_q + j_q) * half
                h_minus = (delta_q - j_q) * half
                monomials = (
                    ((h_plus, h_minus), 1.0),
                    ((h_minus, h_plus), 1.0),
                )
        for key, coeff in monomials:
            if float(key[0] + key[1]) > cut + 1e-9:
                continue
            out[key] = out.get(key, 0.0) + degen * coeff
    return _prune(out)


def _dedekind_power(alpha: float, cut: float) -> Poly:
    """``(prod_{n>=1} (1-q^n)(1-qbar^n))^alpha`` as a polynomial truncated
    to total degree ``cut``. Comes from

        eta(tau)*eta(-taubar)*(q*qbar)^(-1/24) = prod_{n>=1} (1-q^n)(1-qbar^n)

    once the ``q^(1/24) qbar^(1/24)`` prefactor of the Dedekind etas is
    cancelled by the matching ``(q qbar)^(-1/24)``."""
    eta_q: Poly = _one()
    for n in range(1, int(cut) + 1):
        factor: Poly = {
            (Fraction(0), Fraction(0)): 1.0,
            (Fraction(n), Fraction(0)): -1.0,
        }
        eta_q = _poly_mul(eta_q, factor, cut)
    eta_qbar: Poly = {(hb, h): v for (h, hb), v in eta_q.items()}
    base = _poly_mul(eta_q, eta_qbar, cut)
    g = {k: v for k, v in base.items() if not (k[0] == 0 and k[1] == 0)}
    return _binomial_series(g, alpha, cut)


# --- precomputed deformed partition ---------------------------------------


@dataclass(frozen=True)
class _DeformedSeries:
    """Flat numpy representation of the deformed-partition series.

    Each monomial ``coeffs[i] * q^h[i] * qbar^hbar[i]`` belongs to exactly
    one Delta-bucket. ``bucket_starts`` gives the index into ``coeffs`` where
    each bucket begins; ``bucket_deltas[b]`` is the total dimension of
    bucket ``b`` (``h + hbar``). Monomials within a bucket are stored
    contiguously so per-bucket sums are a single ``np.add.reduceat``.
    """

    new_c: float
    old_c: float
    cut: float
    # Flat per-monomial arrays (contiguous by bucket):
    coeffs: np.ndarray        # (N_mon,) float64
    h: np.ndarray             # (N_mon,) float64
    hbar: np.ndarray          # (N_mon,) float64
    # Per-bucket metadata:
    bucket_deltas: np.ndarray  # (N_buckets,) float64, sorted ascending
    bucket_starts: np.ndarray  # (N_buckets,) int64, offsets into coeffs
    # Deduplicated total-exponent table (h+hbar). Many monomials share
    # exponent values, so q^exp can be computed once per unique value
    # and gathered into the per-monomial array.
    exps_unique: np.ndarray    # (N_unique,) float64, sorted ascending
    exps_inverse: np.ndarray   # (N_mon,) int64, indices into exps_unique


def _compute_series(
    base_cft: CFTData, new_c: float, cut: float
) -> _DeformedSeries:
    old_c = float(base_cft.central_charge)
    base_terms = _base_partition_terms(base_cft, cut)
    if new_c > old_c:
        alpha = old_c - new_c
        dedekind = _dedekind_power(alpha, cut)
        product = _poly_mul(base_terms, dedekind, cut)
    else:
        product = base_terms

    # Group monomials by Delta = h + hbar, sort buckets by Delta, sort
    # monomials within each bucket for determinism.
    buckets: Dict[Fraction, Dict[Key, float]] = {}
    for key, coeff in product.items():
        delta = key[0] + key[1]
        buckets.setdefault(delta, {})[key] = coeff

    bucket_deltas: list[float] = []
    bucket_starts: list[int] = []
    coeffs: list[float] = []
    h_list: list[float] = []
    hbar_list: list[float] = []
    for delta in sorted(buckets.keys()):
        bucket = buckets[delta]
        bucket_deltas.append(float(delta))
        bucket_starts.append(len(coeffs))
        for key, coeff in sorted(bucket.items()):
            h_list.append(float(key[0]))
            hbar_list.append(float(key[1]))
            coeffs.append(float(coeff))

    h_arr = np.asarray(h_list, dtype=np.float64)
    hbar_arr = np.asarray(hbar_list, dtype=np.float64)
    exps = h_arr + hbar_arr
    exps_unique, exps_inverse = np.unique(exps, return_inverse=True)

    return _DeformedSeries(
        new_c=new_c,
        old_c=old_c,
        cut=float(cut),
        coeffs=np.asarray(coeffs, dtype=np.float64),
        h=h_arr,
        hbar=hbar_arr,
        bucket_deltas=np.asarray(bucket_deltas, dtype=np.float64),
        bucket_starts=np.asarray(bucket_starts, dtype=np.int64),
        exps_unique=np.asarray(exps_unique, dtype=np.float64),
        exps_inverse=np.asarray(exps_inverse, dtype=np.int64),
    )


# Small cache: the same base CFT / (new_c, cut) combination shows up
# repeatedly in training (all tau values hit the same reference model).
_SERIES_CACHE: Dict[tuple, _DeformedSeries] = {}


def _cache_key(base: CFTData, new_c: float, cut: float) -> tuple:
    # The base CFT's primaries + central charge fully determine the series.
    prim_key = tuple(
        (float(p.delta), int(p.spin), int(p.degen), str(p.character_kind))
        for p in base.primaries
    )
    return (float(base.central_charge), prim_key, float(new_c), float(cut))


def _get_series(deformed: EtaDeformedCFTData) -> _DeformedSeries:
    key = _cache_key(deformed.base, deformed.new_central_charge, deformed.cut)
    cached = _SERIES_CACHE.get(key)
    if cached is not None:
        return cached
    series = _compute_series(deformed.base, deformed.new_central_charge, deformed.cut)
    _SERIES_CACHE[key] = series
    return series


# --- tau-evaluation ---------------------------------------------------------


def _evaluate_series_at_tau(
    series: _DeformedSeries, tau: np.ndarray, cut: float
) -> np.ndarray:
    """Evaluate ``sum_Delta P_Delta(q, qbar) * F(Delta, cut, tau2)`` at ``tau``.

    Fully vectorized: one ``np.exp`` + one ``np.add.reduceat`` over the
    full (B, N_mon) grid, no Python per-monomial loop.
    """
    tau = np.asarray(tau, dtype=np.complex128)
    B = tau.shape
    tau2 = tau.imag.astype(np.float64)
    q = np.exp(2j * np.pi * tau)
    qbar = np.exp(-2j * np.pi * np.conj(tau))

    # Per-monomial value (B, N_mon): coeff * q^h * qbar^hbar.
    # Use log+exp to avoid pow-by-fractional-exponent precision loss:
    # q^h = exp(h * log q), qbar^hbar = exp(hbar * log qbar).
    log_q = np.log(q)                # (B,) complex
    log_qbar = np.log(qbar)          # (B,)
    mono = np.exp(
        log_q[..., None] * series.h[None, :]
        + log_qbar[..., None] * series.hbar[None, :]
    ) * series.coeffs[None, :]       # (B, N_mon)

    # Per-bucket polynomial values via ``reduceat``: sum contiguous
    # segments [starts[k], starts[k+1]) along axis=1.
    per_bucket = np.add.reduceat(mono, series.bucket_starts, axis=-1)  # (B, N_buckets)

    # Apply the form factor (or hard cutoff) per bucket.
    deltas = series.bucket_deltas                       # (N_buckets,)
    keep = deltas < cut                                 # (N_buckets,)
    if not np.any(keep):
        return np.zeros(B, dtype=np.complex128)
    if series.new_c > series.old_c:
        a = 0.5 * (series.new_c - series.old_c)
        z = 2.0 * np.pi * np.where(keep, cut - deltas, 0.0)[None, :] * tau2[..., None]
        form = np.where(keep[None, :], gammainc(a, z), 0.0)  # (B, N_buckets)
    else:
        form = keep.astype(np.float64)[None, :]              # (1, N_buckets) broadcasts

    return (per_bucket * form).sum(axis=-1)


def deformed_reduced_partition(
    model: EtaDeformedCFTData, tau: complex | np.ndarray
) -> np.ndarray:
    """``Z_reduced`` for an eta-deformed reference model.

    Returns a complex array the same shape as ``tau``."""
    series = _get_series(model)
    tau_arr = np.asarray(tau, dtype=np.complex128)
    tau2 = tau_arr.imag.astype(np.float64)
    q = np.exp(2j * np.pi * tau_arr)
    qbar = np.exp(-2j * np.pi * np.conj(tau_arr))
    prefactor = (
        np.sqrt(tau2)
        * (q * qbar) ** ((1.0 - series.new_c) / 24.0)
        * tau2 ** ((series.old_c - series.new_c) / 2.0)
    )
    return prefactor * _evaluate_series_at_tau(series, tau_arr, series.cut)


# --- TruncationDifference / ObjectiveDenominator --------------------------


def truncation_difference(
    deformed: EtaDeformedCFTData, low_cut: float, tau: complex | np.ndarray
) -> np.ndarray:
    """Reduced partition function of ``deformed`` minus that of ``deformed``
    truncated below ``low_cut``, with the base model's primaries held
    fixed."""
    truncated = EtaDeformedCFTData(
        base=deformed.base,
        new_central_charge=deformed.new_central_charge,
        cut=low_cut,
    )
    return deformed_reduced_partition(deformed, tau) - deformed_reduced_partition(
        truncated, tau
    )


def _modulus_q_truncation_difference(
    deformed: EtaDeformedCFTData, low_cut: float, tau: complex | np.ndarray
) -> np.ndarray:
    """``TruncationDifference`` with the ``q -> exp(-2*pi*sqrt(tau1^2+tau2^2))``
    modulus substitution that kills cancellations in ``tau1``.

    Vectorized: one ``np.exp`` for ``q_scalar^(h+hbar)``, one
    ``np.add.reduceat`` to sum per bucket, then one ``np.where`` to
    dispatch the form-factor difference across buckets. Mathematically
    identical to the old per-monomial Python loop.

    Cancellation avoidance (same as before): for buckets with
    ``Delta < low_cut`` we compute
    ``F(Delta, hi) - F(Delta, lo) = Q(a, z_lo) - Q(a, z_hi)`` via
    ``scipy.special.gammaincc``, never subtracting two near-unity
    values.
    """
    tau_arr = np.asarray(tau, dtype=np.complex128)
    tau2 = tau_arr.imag.astype(np.float64)
    modulus = np.sqrt(tau_arr.real ** 2 + tau_arr.imag ** 2).astype(np.float64)
    q_scalar = np.exp(-2.0 * np.pi * modulus)            # (B,) real

    series = _get_series(deformed)
    cut_hi = float(deformed.cut)
    cut_lo = float(low_cut)

    # Prefactor: ``sqrt(tau2) * (q*qbar)^((1-new_c)/24) * tau2^((old_c-new_c)/2)``.
    q_qbar_prod = q_scalar * q_scalar
    prefactor = (
        np.sqrt(tau2)
        * q_qbar_prod ** ((1.0 - series.new_c) / 24.0)
        * tau2 ** ((series.old_c - series.new_c) / 2.0)
    )

    # Per-monomial values at q = qbar = q_scalar: coeff * q^(h+hbar).
    # The series typically has many duplicate exponent values (e.g., 70
    # monomials with only 13 unique exp values), so deduplicating the
    # np.power call gives ~5x fewer power ops. Bit-identical: np.power is
    # deterministic, so np.power(q, exp_i) == np.power(q, unique[idx_i])
    # exactly when exp_i == unique[idx_i].
    exps_unique = series.exps_unique                     # (N_unique,)
    exps_inverse = series.exps_inverse                   # (N_mon,) int64
    unique_pow = np.power(
        q_scalar[..., None], exps_unique[None, :]
    )                                                    # (B, N_unique)
    mono = unique_pow[..., exps_inverse] * series.coeffs[None, :]  # (B, N_mon)
    per_bucket = np.add.reduceat(mono, series.bucket_starts, axis=-1)  # (B, N_b)

    deltas = series.bucket_deltas                        # (N_b,)
    keep = deltas < cut_hi                               # (N_b,)
    if not np.any(keep):
        return np.zeros(tau_arr.shape, dtype=np.float64)

    if series.new_c > series.old_c:
        a = 0.5 * (series.new_c - series.old_c)
        # Gate: the two saturation shortcuts below (`gammainc -> 1.0` at
        # z >= 40 and `gammaincc -> 0` at z_lo >= 50) are calibrated for
        # a <= 1.5. Empirically, scipy.special.gammainc(a, 40) returns
        # EXACTLY 1.0 in fp64 only for a <= ~1.5; at a = 2.0 the deviation
        # becomes 1 ULP. Project usage keeps new_c capped low enough that
        # a stays comfortably below the threshold; if a future caller
        # extends new_c much further, recalibrate or remove the shortcut.
        _shortcuts_ok = a <= 1.5
        # z_hi and z_lo per (tau, bucket); masked to valid buckets.
        z_hi = 2.0 * np.pi * np.where(keep, cut_hi - deltas, 0.0)[None, :] * tau2[..., None]
        high = (deltas >= cut_lo) & keep                 # only hi-cut contributes
        low = (deltas < cut_lo) & keep                   # both cuts -> cancellation-safe diff
        form_diff = np.zeros_like(z_hi)
        # Saturation threshold below which we must actually call scipy.
        # gammainc(a, z) is exactly 1.0 in fp64 for z >= 40 across the
        # supported a-range (a <= 1.5 per the assertion above). Skipping
        # scipy on saturated values is bit-identical because scipy would
        # also have returned 1.0.
        _SAT_THRESHOLD = 40.0 if _shortcuts_ok else float("inf")
        if np.any(high):
            high_mat = np.broadcast_to(high[None, :], z_hi.shape)
            sat = high_mat & (z_hi >= _SAT_THRESHOLD)
            unsat = high_mat & ~sat
            if np.any(unsat):
                form_diff[unsat] = gammainc(a, z_hi[unsat])
            if np.any(sat):
                form_diff[sat] = 1.0
        if np.any(low):
            # Compute the gammaincc difference only on (B, low-buckets);
            # the original code computed it on the full (B, N_b) then
            # masked with `low`, wasting ~50% of the special-function
            # work on positions that get discarded. Also skip positions
            # where z_lo >= 50: gammaincc(a, z) < 1e-25 there for a in
            # our scan range, so the difference is negligible (well
            # below fp64 precision when combined with O(1) per_bucket).
            low_idx = np.where(low)[0]
            cut_diff_lo = cut_lo - deltas[low_idx]            # (N_low,)
            cut_diff_hi = cut_hi - deltas[low_idx]            # (N_low,)
            z_lo_sub = 2.0 * np.pi * cut_diff_lo[None, :] * tau2[..., None]
            _GC_SAT_THRESHOLD = 50.0 if _shortcuts_ok else float("inf")
            sat = z_lo_sub >= _GC_SAT_THRESHOLD              # gammaincc < 1e-25
            unsat = ~sat
            if np.any(unsat):
                # Use np.empty and fill only the unsaturated cells; the
                # saturated cells stay 0 in form_diff (initialised to
                # zeros above), which matches the true value to 1e-25.
                z_hi_sub = 2.0 * np.pi * cut_diff_hi[None, :] * tau2[..., None]
                diff_sub = np.zeros_like(z_lo_sub)
                diff_sub[unsat] = (
                    gammaincc(a, z_lo_sub[unsat])
                    - gammaincc(a, z_hi_sub[unsat])
                )
                form_diff[..., low_idx] = diff_sub
            # else: all saturated -> form_diff stays zero on the low buckets
    else:
        # Hard cutoff only: contribution is 1 for cut_lo <= Delta < cut_hi.
        mask = (deltas >= cut_lo) & keep
        form_diff = mask.astype(np.float64)[None, :]     # broadcasts over tau

    return prefactor * (per_bucket * form_diff).sum(axis=-1)


def objective_denominator(
    model_cut: float,
    reference: EtaDeformedCFTData,
    tau: complex | np.ndarray,
) -> np.ndarray:
    """``sigma_trunc(tau) = sqrt(|diff(tau)|^2 + |diff(-1/tau)|^2)``.

    ``model_cut`` is the dimension cutoff of the candidate (primary)
    model; it becomes the ``low_cut`` passed to
    :func:`_modulus_q_truncation_difference`."""
    tau_arr = np.asarray(tau, dtype=np.complex128)
    diff_tau = _modulus_q_truncation_difference(reference, model_cut, tau_arr)
    diff_s = _modulus_q_truncation_difference(reference, model_cut, -1.0 / tau_arr)
    return np.sqrt(diff_tau ** 2 + diff_s ** 2)
