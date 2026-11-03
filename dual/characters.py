"""Foundation module -- measure algebra + Laguerre basis + index table.

Port of the SDPB/Virasoro-integrality measure machinery to ``mpmath``,
following the papers arXiv:2308.11692 ("The Geometry of the Modular
Bootstrap") and arXiv:2308.08725 ("Improving Modular Bootstrap Bounds with
Integrality"). This is the linchpin module the rest of the dual-bootstrap
port imports, so every numeric here is **arbitrary precision**: ``mpmath``
at ``mp.dps = DPS`` with ``DPS >= 200``. float64 is forbidden here -- it
destroys the ill-conditioned Hankel Cholesky below and the ~300-digit
tails involved.

What it owns:

- Global constants ``RHO_CROSSING = DAMP_BASE = exp(-2*pi)`` (not
  ``3-2*sqrt(2))``, which is the SDPB.m default before the notebook this
  was ported from overrides it) and ``SDPB_PREC_BITS = 1024``; ``nf`` for
  no-``e`` decimal formatting.
- ``DampedRational`` dataclass + ``damped_shift`` + numeric ``__call__``.
- ``bilinear_form_nopole`` (the only branch the production drivers hit) and
  ``integrate_prefactor_term`` for pole orders ``n in {1,2,3}`` (``n>=4``
  raises -- kept for general ``SDPB.m`` parity only).
- ``anti_band_matrix`` Hankel; ``orthogonal_polynomials`` with the
  Mathematica upper-Cholesky orientation;
  ``rescaled_laguerre_sample_points`` and first-max ``max_index_by``.
- ``chir`` / ``Gvac`` / ``Gchar`` characters; ``Com`` (exact polynomial in
  ``Delta``), ``Com0`` (number), ``Comwithprefactor_at`` (number at a
  point); ``tb`` derivative-index table.

The Mathematica conventions ``Gamma[a,z]`` = upper-incomplete gamma
(``mpmath.gammainc(a, z, inf)``), ``ExpIntegralE[n,z]`` = ``mpmath.expint(n,z)``,
``LaguerreL[n,a,x]`` = ``mpmath.laguerre(n,a,x)``, ``Log`` = natural log carry
through verbatim.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import mpmath
from mpmath import mp, mpf


# Global precision / constants. The reference Mathematica notebook ran at
# prec=200, $MinPrecision=200; the dual path is never float64.
DPS = 200
mp.dps = DPS

# The production notebook redefines rhoCrossing = exp(-2 pi) AFTER loading
# SDPB.m (whose default is 3 - 2*sqrt(2)). The measure base B of every
# DampedRational in the drivers and the sample-point base are the same
# constant exp(-2 pi); keep them unified.
RHO_CROSSING = mpmath.e ** (-2 * mpmath.pi)
DAMP_BASE = mpmath.e ** (-2 * mpmath.pi)  # B
SDPB_PREC_BITS = 1024
PREC_DIGITS = DPS


def nf(x):
    """Full-precision plain-decimal string with NO ``e``/``E``.

    Integers print verbatim; reals print with ``PREC_DIGITS`` significant digits
    and no scientific exponent (Mathematica ``ExponentFunction->Null``). SDPB's
    parser reads 1024-bit decimals, so an ``e`` in the string breaks it -- the
    ``min_fixed=-inf, max_fixed=+inf`` flags force fixed-point output.
    """
    if isinstance(x, int):
        return str(x)
    return mp.nstr(mpf(x), PREC_DIGITS, strip_zeros=False,
                   min_fixed=mp.ninf, max_fixed=mp.inf)


# ---------------------------------------------------------------------------
# DampedRational + closed forms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DampedRational:
    """``DampedRational[const, {poles}, base, x]`` = ``const * base**x / prod(x - p_i)``.

    Formal head carried as an immutable dataclass. ``poles`` is stored as a
    tuple so the instance stays hashable/immutable; a list argument is
    accepted and normalised.
    """

    const: object
    poles: tuple = field(default=())
    base: object = DAMP_BASE

    def __post_init__(self):
        # accept a list of poles but store immutably
        object.__setattr__(self, "poles", tuple(self.poles))

    def __call__(self, x):
        """Numeric value ``const * base**x / prod(x - p_i)``."""
        num = self.const * self.base ** x
        for p in self.poles:
            num /= (x - p)
        return num


def damped_shift(dr: DampedRational, a) -> DampedRational:
    """Shift identity (the workhorse).

    ``DampedRational[c, {p_i}, b, x + a] == DampedRational[b**a c, {p_i - a}, b, x]``:
    replacing ``x -> x + a`` multiplies the constant by ``b**a`` and shifts every
    pole ``p_i -> p_i - a``. Every ``pols2`` prefactor is built this way from
    ``DampedRational[2, {}, B, x + g]`` (empty poles -> const ``2 B**g``).
    """
    return DampedRational(dr.base ** a * dr.const,
                          tuple(p - a for p in dr.poles), dr.base)


def bilinear_form_nopole(const, base, m):
    """No-pole moment ``integral_0^inf x^m * const * base**x dx``.

    ``= const * m! * (-ln base)**(-(m+1))``. The standard ``int_0^inf x^m b^x dx``
    for ``b < 1`` (so ``-ln b > 0``). This is the ONLY ``bilinearForm`` branch the
    production drivers ever hit, because every prefactor has an empty pole list.
    """
    return const * mp.factorial(m) * (-mp.log(base)) ** (-(m + 1))


def integrate_prefactor_term(c, p, n, b, m):
    """``integral_0^inf x^m * c * b**x / (x - p)**n dx``, closed form.

    Pole orders ``n in {1, 2, 3}`` only (``n >= 4`` raises ``NotImplementedError``,
    matching ``SDPB.m``). For ``n in {2, 3}`` the low-``m`` cases ``m in {0,1,2}``
    have dedicated non-singular forms because ``Gamma[-2+m]`` is singular there.
    ``Gamma[a,z]`` is the upper-incomplete gamma ``mpmath.gammainc(a, z, inf)`` and
    ``ExpIntegralE[n,z]`` is ``mpmath.expint(n, z)``.

    These pole branches exist for general ``SDPB.m`` parity only; the production
    drivers use the empty-pole branch (``bilinear_form_nopole``).
    """
    L = mp.log(b)
    if n == 1:
        if m == 0:
            return b ** p * c * mp.gammainc(0, p * L, mp.inf)
        return b ** p * c * (-p) ** m * mp.factorial(m) * mp.gammainc(-m, p * L, mp.inf)
    if n == 2:
        if m == 0:
            return -(c / p) + b ** p * c * mp.gammainc(0, p * L, mp.inf) * L
        return (b ** p * c * mp.gamma(m) * (-L) ** (-m) * L
                * (b ** (-p) + mp.expint(m, p * L) * (-m - p * L)))
    if n == 3:
        if m == 0:
            return c * (1 - p * L + b ** p * p ** 2 * mp.gammainc(0, p * L, mp.inf) * L ** 2) / (2 * p ** 2)
        if m == 1:
            return c * (-1 - p * L + b ** p * p * mp.expint(1, p * L) * L * (2 + p * L)) / (2 * p)
        if m == 2:
            return (mpf(1) / 2 * c
                    * (-3 - p * L + b ** p * mp.gammainc(0, p * L, mp.inf) * (2 + p * L * (4 + p * L))))
        return (mpf(1) / 2 * c * (-2 + m) * mp.gamma(-2 + m) * (-L) ** (2 - m)
                * (-1 - m - p * L
                   + b ** p * mp.expint(-1 + m, p * L) * ((-1 + m) * m - p * L * (-2 * m - p * L))))
    raise NotImplementedError("pole order >= 4 unsupported (matches SDPB.m)")


# ---------------------------------------------------------------------------
# antiBandMatrix (Hankel)
# ---------------------------------------------------------------------------

def anti_band_matrix(bs):
    """Hankel matrix from anti-diagonal list ``bs``.

    ``A[i][j] = bs[i + j]`` (0-indexed), size ``n = ceil(len(bs)/2)``. E.g.
    ``bs = [b0, b1, b2]`` -> ``[[b0, b1], [b1, b2]]``.
    """
    n = (len(bs) + 1) // 2  # ceil(len/2)
    return mp.matrix([[bs[i + j] for j in range(n)] for i in range(n)])


# ---------------------------------------------------------------------------
# orthogonalPolynomials (Cholesky upper-vs-lower convention)
# ---------------------------------------------------------------------------

def orthogonal_polynomials(damped_rational: DampedRational, degree):
    """Degree-``0..degree`` polynomials orthonormal w.r.t. ``damped_rational dx``.

    Returns a list of ``degree+1`` coefficient lists, ascending powers
    ``x^0..x^degree``; entry ``k`` is the degree-``k`` orthonormal polynomial.

    Constant measure (``poles == ()`` is allowed; "constant" here means the
    DampedRational reduces to a number, used by SDPB only at ``degree == 0`` with
    prefactor ``1``): ``degree == 0`` returns ``[[1/sqrt(const)]]``; ``degree > 0``
    on a constant measure is illegal.

    DampedRational measure: moments ``mu[m] = bilinearForm`` over the NEGATIVE
    poles only (``Select[poles, #<0&]``; empty for the drivers), ``m = 0..2*degree``;
    Hankel ``M = anti_band_matrix(mu)``; Cholesky.

    Load-bearing convention note: Mathematica ``CholeskyDecomposition`` returns
    the UPPER factor ``U`` with ``M = U^H U``, and the polynomials are the
    COLUMNS of ``Inverse[U]``. ``mpmath.cholesky`` returns the LOWER ``L`` with
    ``M = L L^T``, so ``U = L.T`` for real SPD ``M``. The function verifies
    ``U^T U == M`` to ULP.
    """
    poles_neg = tuple(p for p in damped_rational.poles if p < 0)

    if len(poles_neg) == 0 and not damped_rational.poles:
        # No-pole measure (the production case and the constant prefactor=1 case).
        # degree 0 flows through the general Cholesky path below: a 1x1 Hankel
        # [mu0] gives {1/sqrt(mu0)} (= {1/sqrt(const)} for prefactor=1 -> {1}).
        mu = [bilinear_form_nopole(damped_rational.const, damped_rational.base, m)
              for m in range(2 * degree + 1)]
    else:
        # General (non-driver) path: keep negative poles only.
        mu = [bilinear_form_nopole(damped_rational.const, damped_rational.base, m)
              if not poles_neg
              else _bilinear_form_with_poles(damped_rational.const, poles_neg,
                                             damped_rational.base, m)
              for m in range(2 * degree + 1)]

    M = anti_band_matrix(mu)  # (degree+1) x (degree+1) Hankel
    L = mp.cholesky(M)        # lower: M = L L^T
    U = L.T                   # upper: M = U^T U  (real SPD)

    # Verify U^T U == M to ULP (guards the Cholesky orientation convention above).
    UtU = U.T * U
    tol = mpf(10) ** (-(PREC_DIGITS - 10))
    for i in range(M.rows):
        for j in range(M.cols):
            scale = max(mpf(1), abs(M[i, j]))
            if abs(UtU[i, j] - M[i, j]) > tol * scale:
                raise ValueError("Cholesky reconstruction U^T U != M (orientation bug?)")

    Uinv = U ** -1
    # poly k (k=0..degree): coeffs ascending = column k of Uinv.
    return [[Uinv[m, k] for m in range(degree + 1)] for k in range(degree + 1)]


def _bilinear_form_with_poles(const, poles_neg, base, m):
    """General-pole moment via partial fractions (non-driver path).

    Only reached when ``orthogonal_polynomials`` is given a measure with negative
    poles, which the production drivers never do. Uses exact ``mpmath``
    partial-fraction decomposition over the simple poles ``poles_neg`` (all
    distinct, simple -- sufficient for the cross-check use; ``SDPB.m``'s ``Apart``
    handles the general case). Sum of single-pole ``integrate_prefactor_term``s.
    """
    # 1 / prod(x - p_i) = sum_i [ 1/prod_{j!=i}(p_i - p_j) ] / (x - p_i).
    total = mpf(0)
    for i, p in enumerate(poles_neg):
        coeff = mpf(1)
        for j, q in enumerate(poles_neg):
            if j != i:
                coeff /= (p - q)
        # term = coeff * c * b^x / (x - p)^1  -> integrate_prefactor_term(c*coeff, p, 1, b, m)
        total += integrate_prefactor_term(const * coeff, p, 1, base, m)
    return total


# ---------------------------------------------------------------------------
# sample points + first-max argmax
# ---------------------------------------------------------------------------

def rescaled_laguerre_sample_points(n, rho_crossing=RHO_CROSSING):
    """SDPB rescaled-Laguerre sample points.

    ``samplePoints[k] = pi^2 (4k-1)^2 / (-64 n ln(rho_crossing))`` for
    ``k = 0..n-1``. Since ``ln(rho_crossing) < 0`` the denominator is positive and
    points increase in ``k``. Used with ``n = degree + 1``.
    """
    den = -64 * n * mp.log(rho_crossing)
    return [mpmath.pi ** 2 * (4 * k - 1) ** 2 / den for k in range(n)]


def max_index_by(values, key):
    """First-max argmax: 0-based index of the element maximizing ``key``.

    Mathematica ``maxIndexBy`` is a stable ``SortBy``, so ties take the FIRST
    occurrence. The pivot it returns must match between
    ``reshuffleWithNormalization`` and ``restoreCoefficients``, so the tie rule
    is load-bearing.
    """
    best_i = 0
    best_k = key(values[0])
    for i in range(1, len(values)):
        k = key(values[i])
        if k > best_k:  # strict > keeps the FIRST max on ties
            best_k = k
            best_i = i
    return best_i


# ---------------------------------------------------------------------------
# Virasoro characters
# ---------------------------------------------------------------------------

def chir(h, beta, c):
    """Virasoro vacuum character ``chir``.

    ``chir(h, beta) = beta^(1/4) exp(-2 pi beta (h - (c-1)/24))``.
    """
    return beta ** (mpf(1) / 4) * mpmath.e ** (-2 * mpmath.pi * beta * (h - (c - 1) / 24))


def Gvac(c, beta, betab):
    """Vacuum (identity) contribution.

    ``(chir(0,beta) - chir(1,beta)) (chir(0,betab) - chir(1,betab))``.
    """
    return ((chir(0, beta, c) - chir(1, beta, c))
            * (chir(0, betab, c) - chir(1, betab, c)))


def Gchar(c, beta, betab, Delta, s):
    """Non-vacuum primary of dimension ``Delta``, spin ``s``.

    ``chir((Delta+s)/2,beta) chir((Delta-s)/2,betab)
      + chir((Delta-s)/2,beta) chir((Delta+s)/2,betab)``.
    ``(Delta +/- s)/2`` are the weights ``h, hbar``.
    """
    return (chir((Delta + s) / 2, beta, c) * chir((Delta - s) / 2, betab, c)
            + chir((Delta - s) / 2, beta, c) * chir((Delta + s) / 2, betab, c))


# ---------------------------------------------------------------------------
# Laguerre basis: Com (exact poly in Delta), Com0, Comwithprefactor
# ---------------------------------------------------------------------------

# L[n](z) = LaguerreL[n, -1/2, z]; the basis Laguerre order.
def L(n, z):
    """``LaguerreL[n, -1/2, z]`` (numeric)."""
    return mpmath.laguerre(n, mpf(-1) / 2, z)


@functools.lru_cache(maxsize=None)
def _laguerre_coeffs_cached(n, _dps):
    """Memoized core of :func:`_laguerre_coeffs`, keyed on ``(n, mp.dps)``.

    ``_dps`` is the live ``mp.dps`` the wrapper passes in PURELY so it enters the
    cache key -- the body reads the ambient precision (``_dps == mp.dps`` here,
    single-threaded), which is the precision the returned ``mpf`` coefficients
    carry. mpmath is global-precision-stateful: the test suite alone runs modules
    at dps 60/200/220 in one process, and several CLI drivers set
    ``mp.dps = args.dps``. A cache keyed on ``n`` only could hand a list built at
    one precision into an assembly at another and silently break bit-identity, so
    the precision is part of the key. Returns an immutable tuple; the wrapper
    hands out a private list copy per call.
    """
    a = mpf(-1) / 2
    return tuple((-1) ** i * mpmath.binomial(n + a, n - i) / mp.factorial(i)
                 for i in range(n + 1))


def _laguerre_coeffs(n):
    """Exact ascending coefficient list of ``L_n^(-1/2)(z)`` in ``z``.

    ``L_n^(a)(z) = sum_{i=0}^n (-1)^i binomial(n+a, n-i) z^i / i!`` with ``a = -1/2``.
    Returns ``[a_0, ..., a_n]`` (mpf) meaning ``a_0 + a_1 z + ... + a_n z^n``.
    Built by EXACT expansion, never float sampling.

    Memoized on ``(n, mp.dps)`` (see :func:`_laguerre_coeffs_cached`):
    ``com_poly_in_delta`` rebuilds the same ~10 orders hundreds of times per
    ``pols2`` assembly. A fresh ``list`` copy is returned each call so a caller
    can never mutate the cached tuple (it is consumed read-only by
    :func:`_poly_compose_affine`).
    """
    return list(_laguerre_coeffs_cached(n, mp.dps))


def _poly_mul(p, q):
    """Multiply two ascending-coefficient polynomials."""
    out = [mpf(0)] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        for j, qj in enumerate(q):
            out[i + j] += pi * qj
    return out


def _poly_compose_affine(coeffs, a, b):
    """Compose ascending poly ``coeffs`` (in z) with the affine map ``z = a*x + b``.

    Returns the ascending coefficient list in ``x`` of ``p(a*x + b)``.
    Horner over the powers of the binomial ``(a x + b)``.
    """
    # Build powers (a x + b)^i incrementally.
    result = [mpf(0)]
    power = [mpf(1)]  # (a x + b)^0
    binom = [b, a]    # a x + b, ascending in x
    for i, ci in enumerate(coeffs):
        if ci != 0:
            term = [ci * t for t in power]
            # add term into result
            if len(term) > len(result):
                result = result + [mpf(0)] * (len(term) - len(result))
            for k, t in enumerate(term):
                result[k] += t
        if i < len(coeffs) - 1:
            power = _poly_mul(power, binom)
    return result


@functools.lru_cache(maxsize=None)
def _com_poly_in_delta_cached(k, kb, c, s, _dps):
    """Memoized core of :func:`com_poly_in_delta`, keyed on ``(k, kb, c, s, mp.dps)``.

    ``_dps`` is the live ``mp.dps`` the wrapper passes in PURELY so it enters the
    cache key -- the body reads the ambient precision (``_dps == mp.dps`` here,
    single-threaded), which is the precision the returned ``mpf`` coefficients
    carry. mpmath is global-precision-stateful: the test suite alone runs modules
    at dps 60/200/220 in one process, and several CLI drivers set
    ``mp.dps = args.dps``. A cache keyed on ``(k, kb, c, s)`` only could hand a
    list built at one precision into an assembly at another and silently break
    bit-identity, so the precision is part of the key. ``c`` is an ``mpf``
    (hashable; two equal ``mpf``\\ s hash equal, which is the correct behavior
    here). Returns an immutable tuple; the wrapper hands out a private list copy
    per call.
    """
    two_pi = 2 * mpmath.pi
    # u = 4 pi (Delta/2 + s/2 - (c-1)/24) = (2 pi) Delta + 4 pi (s/2 - (c-1)/24)
    u_a = two_pi                                     # coeff of Delta
    u_b = 4 * mpmath.pi * (mpf(s) / 2 - (c - 1) / 24)  # constant
    v_a = two_pi
    v_b = 4 * mpmath.pi * (-mpf(s) / 2 - (c - 1) / 24)

    Lk = _laguerre_coeffs(k)
    Lkb = _laguerre_coeffs(kb)

    Lk_u = _poly_compose_affine(Lk, u_a, u_b)
    Lkb_v = _poly_compose_affine(Lkb, v_a, v_b)
    Lkb_u = _poly_compose_affine(Lkb, u_a, u_b)
    Lk_v = _poly_compose_affine(Lk, v_a, v_b)

    t1 = _poly_mul(Lk_u, Lkb_v)
    t2 = _poly_mul(Lkb_u, Lk_v)
    n = max(len(t1), len(t2))
    t1 += [mpf(0)] * (n - len(t1))
    t2 += [mpf(0)] * (n - len(t2))
    return tuple(t1[i] + t2[i] for i in range(n))


def com_poly_in_delta(k, kb, c, s):
    """``Com[k, kb, c]`` as an EXACT ascending polynomial in ``Delta``.

    ``Com[k,kb,c] = L[k](u) L[kb](v) + L[kb](u) L[k](v)`` with
    ``u = 4 pi ((Delta+s)/2 - (c-1)/24)``, ``v = 4 pi ((Delta-s)/2 - (c-1)/24)`` --
    a polynomial in ``Delta`` of degree ``k+kb`` at fixed ``s, c``. NO leading
    factor 2. Built by exact Laguerre expansion + affine composition, never float
    sampling (the exp damping must NOT leak in; it is supplied downstream by the
    DampedRational prefactor in ``pols2``).

    Returns ``[a_0, ..., a_{k+kb}]`` (mpf), ascending in ``Delta``.

    Memoized on ``(k, kb, c, s, mp.dps)`` (see :func:`_com_poly_in_delta_cached`):
    ``pols2`` assembly rebuilds the same handful of ``(k, kb)`` pairs across every
    spin/gap combination. A fresh ``list`` copy is returned each call so a caller
    can never mutate the cached tuple.
    """
    return list(_com_poly_in_delta_cached(k, kb, c, s, mp.dps))


def com_value(k, kb, c, Delta, s):
    """Numeric value of ``Com[k, kb, c]`` at a given ``Delta, s``.

    Direct numeric Laguerre product (matches ``com_poly_in_delta`` evaluated at
    ``Delta``); used by the gates and any numeric cross-check.
    """
    u = 4 * mpmath.pi * ((Delta + s) / 2 - (c - 1) / 24)
    v = 4 * mpmath.pi * ((Delta - s) / 2 - (c - 1) / 24)
    return L(k, u) * L(kb, v) + L(kb, u) * L(k, v)


def com_poly_in_x(k, kb, c, gap, s):
    """``Com[k, kb, c] /. {Delta -> x + gap, s -> s}`` as ascending coeffs in ``x``.

    This is exactly what ``pols2`` needs: the Base entry as a polynomial in the
    shifted variable ``x = Delta - gap``. Composes the ``Delta``-polynomial with
    the affine shift ``Delta = x + gap``.
    """
    delta_poly = com_poly_in_delta(k, kb, c, s)
    return _poly_compose_affine(delta_poly, mpf(1), gap)  # Delta = 1*x + gap


@functools.lru_cache(maxsize=None)
def _A_cached(n, c, _dps):
    """Memoized core of :func:`_A`, keyed on ``(n, c, mp.dps)``.

    ``_dps`` is the live ``mp.dps`` the wrapper passes in PURELY so it enters the
    cache key -- see :func:`_com_poly_in_delta_cached` for why the precision must
    be part of the key. ``c`` is an ``mpf`` (hashable; two equal ``mpf``\\ s hash
    equal, which is correct behavior here). Returns an immutable ``mpf`` value,
    so no copy is needed on the way out.
    """
    P = mpmath.pi
    return (-mpmath.e ** ((mpf(1) / 12) * (-25 + c) * P) * L(n, 4 * (1 + (1 - c) / 24) * P)
            + mpmath.e ** ((mpf(1) / 12) * (-1 + c) * P) * L(n, (mpf(1) / 6) * (1 - c) * P))


def _A(n, c):
    """Helper ``A(n)`` for ``Com0``.

    ``A(n) = -exp((1/12)(-25+c) pi) L[n](4 (1 + (1-c)/24) pi)
              + exp((1/12)(-1+c) pi) L[n]((1/6)(1-c) pi)``.

    Memoized on ``(n, c, mp.dps)`` (see :func:`_A_cached`).
    """
    return _A_cached(n, c, mp.dps)


@functools.lru_cache(maxsize=None)
def _Com0_cached(k, kb, c, _dps):
    """Memoized core of :func:`Com0`, keyed on ``(k, kb, c, mp.dps)``.

    ``_dps`` is the live ``mp.dps`` the wrapper passes in PURELY so it enters the
    cache key -- see :func:`_com_poly_in_delta_cached` for why the precision must
    be part of the key. ``c`` is an ``mpf`` (hashable; two equal ``mpf``\\ s hash
    equal, which is correct behavior here). Returns an immutable ``mpf`` value.
    """
    return 4 * _A(k, c) * _A(kb, c)


def Com0(k, kb, c):
    """Vacuum-channel value (a pure NUMBER).

    ``Com0[k,kb,c] = 4 * A(k) * A(kb)`` -- no ``Delta``, no ``s``. NO leading
    factor beyond the explicit 4.

    Memoized on ``(k, kb, c, mp.dps)`` (see :func:`_Com0_cached`); the body
    computes ``4 * _A(k, c) * _A(kb, c)``, which now hits the ``_A`` cache too.
    """
    return _Com0_cached(k, kb, c, mp.dps)


def comwithprefactor_at(k, kb, c, Delta, s):
    """``Comwithprefactor[k, kb, c]`` evaluated at a point (a NUMBER).

    ``= 2 exp(-2 pi (Delta - (c-1)/12)) * Com_value(k, kb, c, Delta, s)``. Keep the
    explicit leading 2 (``Com`` has none, ``Comwithprefactor`` does;
    ``Comwithprefactor / Com = 2 exp(-2 pi (Delta - (c-1)/12))``). Used by ``pols1``,
    which evaluates the damped basis at specific ``Delta = scalartb[k]``.
    """
    return (2 * mpmath.e ** (-2 * mpmath.pi * (Delta - (c - 1) / 12))
            * com_value(k, kb, c, Delta, s))


# ---------------------------------------------------------------------------
# tb derivative-index table
# ---------------------------------------------------------------------------

def tb(Lmax):
    """Derivative-index pairs ``(k-i, i)`` up to total order ``Lmax``.

    ``k`` runs ODD ``1, 3, ..., (Lmax if odd else Lmax-1)``; for each ``k``,
    ``i = 0..(k-1)//2`` appends ``(k-i, i)``; finally a STABLE sort by the first
    element (Python's sort is stable, matching Mathematica ``SortBy``).
    ``len(tb)`` = number of crossing-derivative constraints.
    """
    pairs = []
    kmax = Lmax if (Lmax % 2 == 1) else Lmax - 1
    for k in range(1, kmax + 1, 2):
        for i in range(0, (k - 1) // 2 + 1):
            pairs.append((k - i, i))
    pairs.sort(key=lambda pr: pr[0])  # stable -> matches SortBy[#, First]
    return pairs
