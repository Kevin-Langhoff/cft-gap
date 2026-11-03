"""Quick start: bracket an operator count with the dual method.

The dual method bounds instead of searching. Here it computes, for a window of
scaling dimensions just above a proposed gap, a bracket [A, B] on the NUMBER of
scalar primaries any consistent theory with that gap must contain there.
Operator counts are whole numbers, so a bracket that contains no integer
excludes the gap.

This demonstration imposes the window constraint at sample points (20 to start) and
positivity on spins <= 10 only. It does not run the two checks that turn such a
bracket into a certificate: the interval proof of the window constraint between
the samples, and the post-check of the spins above the cutoff. For a certified
bracket use ``reproduce/count_bracket.py`` (``dual.scan.count_bracket``).

Two sample points (about 20 s total with the arbitrary-precision solver):

  * c = 1.01, gap 301/600:  the bracket pins the count between ~4.06 and
    ~4.07; no whole number fits, so the gap is excluded at this level.
  * c = 2, window at 2/3:   the bracket contains 18, the count realized by
    the known SU(3) level-1 theory: consistent, as it must be.

The constraint strength is set by SDP_LEVEL below; the paper's settings
(``reproduce/count_bracket.py``) use a much higher level (tighter brackets,
minutes to an hour per bracket).
"""

from mpmath import mp, mpf

from dual.bounds import (
    integer_in,
    integer_lowerbound_interval,
    integer_upperbound_interval,
)

mp.dps = 200  # the character blocks are evaluated in high precision

# Modest constraint level for a fast demonstration.
N_SAMPLES, SDP_LEVEL, SPIN_CUTOFF = 20, 7, 10


def bracket(c, gap):
    """[A, B] bracket on the scalar count in [gap, gap + 1/20] (sampled window)."""
    window_lo, window_hi = gap, gap + mpf(1) / 20
    lo = integer_lowerbound_interval(
        c, gap, window_lo, window_hi, N_SAMPLES, SDP_LEVEL, SPIN_CUTOFF)
    hi = integer_upperbound_interval(
        c, gap, window_lo, window_hi, N_SAMPLES, SDP_LEVEL, SPIN_CUTOFF)
    return mpf(lo.bound), mpf(hi.bound)


def main() -> None:
    for c, gap, label in [
        (mpf(101) / 100, mpf(301) / 600, "c=1.01, gap=301/600"),
        (mpf(2), mpf(2) / 3, "c=2.00, window at 2/3 (SU(3) level 1)"),
    ]:
        a, b = bracket(c, gap)
        verdict = ("contains a whole number -> consistent"
                   if integer_in(a, b)
                   else "NO whole number fits -> gap excluded at this level")
        print(f"{label}:  count in [{mp.nstr(a, 6)}, {mp.nstr(b, 6)}]  "
              f"[{verdict}]")


if __name__ == "__main__":
    main()
