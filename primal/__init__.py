"""Primal method: a direct numerical search for consistent 2D CFT spectra.

Modular invariance is a strong constraint on the spectrum of operators
in a 2D conformal field theory: the theory's partition function, viewed
as a function on the space of tori, must be invariant under the
torus's modular group. This package implements a direct search for
spectra that satisfy this constraint at a given (central charge,
spectral gap) point: parameterize a candidate spectrum with a handful
of free operator dimensions, and use a local optimizer to try to drive
a modular-invariance residual to zero.

Scanning this search over a grid of (central charge, spectral gap)
points produces a two-region map: points where the search finds a
consistent-looking candidate, and points where every attempt fails
(excluded points). See ``scan/`` for the grid and adaptive-refinement
drivers.
"""
