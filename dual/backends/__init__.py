"""Solver backends for the dual modular bootstrap.

A backend takes an assembled SDP (objective + normalization +
``PositiveMatrixWithPrefactor`` constraints) and returns a recovered dual
functional ``alpha`` plus a status. Construction is backend-agnostic; backends
are swappable behind a common interface so the pure-Python scipy-LP FIND path
and the high-precision SDPB-XML path can be cross-checked.
"""
