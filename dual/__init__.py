"""Dual modular bootstrap — certified upper bounds on the 2D CFT spectral gap.

Given a central charge ``c``, the modular bootstrap asks how large the gap to
the first non-vacuum primary can be before torus modular invariance becomes
impossible. This package builds and solves the *dual* problem: it searches for a
linear functional of the crossing equation that is non-negative on every allowed
operator yet negative on the assumed spectrum, which is a rigorous proof that no
consistent CFT sits at that gap. Sharpening the classic Hellerman-type bound, it
also uses the fact that operator degeneracies are non-negative INTEGERS: if the
certified two-sided bracket on a degeneracy contains no integer, the assumption
is excluded outright.

Layout:

- :mod:`dual.characters` — the arbitrary-precision measure algebra: Virasoro
  characters, the crossing-derivative "Com" composites, the Laguerre basis, and
  the sample-point / orthonormal-polynomial machinery. Everything downstream is
  built from these. Runs at ``mpmath`` precision ``mp.dps = 200``; float64 is
  forbidden on this path (it destroys the ill-conditioned Cholesky and the
  ~300-digit tails).
- :mod:`dual.assembly` — turns the Com composites into the semidefinite program
  (objective, normalization, and the per-spin positivity blocks), including the
  emitter to the solver's polynomial-vector-matrix input.
- :mod:`dual.bounds` — the bound drivers (lower / upper / interval / spin) and
  the integrality filter that turns a two-sided bracket into an exclusion.
- :mod:`dual.verify`, :mod:`dual.postcheck` — the rigor layer: an exact high-
  precision certificate that a recovered functional is non-negative, the
  whole-window interval certificate, and the all-spin post-check.
- :mod:`dual.backends` — swappable solvers: a fast float64 linear-program
  pre-screen (default), and two arbitrary-precision semidefinite solvers.
- :mod:`dual.scan` — the window count bracket (:mod:`dual.scan.count_bracket`,
  certified end to end) and the resampling window solve it is built on.

Two fidelity tiers are preserved verbatim from the reference implementation: the
pure-numeric building blocks reproduce the gold data bit-for-bit, and the SDP
bounds match to solver tolerance.
"""
