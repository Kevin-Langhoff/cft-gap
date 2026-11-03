# primal: search for modular-invariant spectra

At each point (c, Delta_gap) the primal method tries to build a truncated
spectrum (primaries up to Delta_max, with a few adjustable dimensions and
degeneracies) whose torus partition function is invariant under tau -> -1/tau
within its truncation error, tested on 952 fixed values of tau. If the
optimizer gets the loss below L = 25 the point is a **candidate**; if every
restart fails it is **excluded by the search**. This is a non-convex search:
a candidate is evidence that a consistent truncated spectrum exists, an
exclusion only that none was found, and points near the boundary are
sensitive to float-level details, so verdicts are statements about classes of
points. The dual method (`../dual/`) gives the certified counterpart.

```bash
python examples/compute_primal_point.py   # two points, ~15 s
python -m primal.scan.grid                # a 165-point reference grid, ~35 s
python reproduce/primal_scan.py --small   # the landscape scan of the paper, ~40 s
```

## Layout

- `data_types.py`: data structures for a spectrum (`CFTData`, `DSD`) and the
  symbolic-expression tree used for free dimensions (`WLExpr`).
- `partition.py`: the reduced partition function of a spectrum and the
  reference series used for the truncation-error estimate.
- `spectrum.py`: the starting spectrum (ansatz) at one (c, Delta_gap) point.
- `optimize.py`: the modular-invariance loss and the batched optimizer
  `run_batched_survey`.
- `warm_start.py`: warm starts for the landscape scan: carry converged
  parameters to a neighbouring point (`lift_init`), encode a known spectrum as
  a starting point, and the compactified free-boson spectrum
  (`free_boson_primaries`, `free_boson_point`).
- `scan/continuation.py`: the landscape scan of the paper (grid, scrub,
  refine); its module docstring describes the method in full.
- `scan/grid.py`: a fixed-grid scan with independent restarts (the reference
  grid above).
- `scan/adaptive.py`: quadtree refinement of the candidate/excluded boundary.
- `resources.py`, `data/`: loaders and cached reference data (reference
  spectrum, the 952 fixed tau, known-CFT anchors).
- `_expr_eval.py`, `_branches.py`: internal helpers of `optimize.py`.
