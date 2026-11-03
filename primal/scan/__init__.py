"""Scan drivers: run the search over a grid of (c, delta_gap) points.

``grid`` runs the fixed reference grid (or any regular sub-grid of it).
``adaptive`` refines a grid adaptively around the boundary between the
CFT-candidate region and the excluded region, so compute is spent where
the boundary actually is instead of uniformly over the whole plane.
"""
