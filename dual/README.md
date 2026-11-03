# dual: certified bounds from semidefinite programming

For a proposed (c, gap), the dual method builds linear functionals on the
modular crossing equation that minimize and maximize the number of primaries of
one spin in a window of dimensions, giving a bracket [A, B] on that count for
every spectrum with that gap. Counts are whole numbers, so a certified bracket
that contains no admissible count excludes the gap. Everything runs at
200-digit precision in `mpmath`.

```bash
python examples/compute_dual_bound.py                     # two sampled-window brackets, ~20 s
python reproduce/count_bracket.py --c 1.14 --gap 0.5171052631578946 --spin 1 --w 0.05 --small
```

## Layout

- `characters.py`: Virasoro character blocks in damped-rational form
  (importing it sets the global `mpmath` precision to 200 digits).
- `assembly.py`: builds the semidefinite programs: the sampled window
  builders, the continuous-window builder
  (`build_window_bound_sdp_continuous`, the window constraint imposed on the
  whole window through a safe polynomial bound of the exponential) and a
  legacy conditional builder kept for reference.
- `bounds.py`: bound drivers and `integer_in`, the exact integer test.
- `verify.py`: exact positivity certificates for a recovered functional,
  independent of the solver.
- `postcheck.py`: the whole-window interval certificate
  (`certify_window_constraint`) and the post-check of the spins above the SDP's
  cutoff (`postcheck_spin_truncation`, a finite extension to `l_check`, not an
  asymptotic all-spin proof).
- `scan/count_bracket.py`: `certified_count_bracket`, one two-sided window
  bracket that runs every certificate gate.
- `scan/density_bound.py`, `scan/resample.py`: the resampling window solve
  `bound_sab` and its refinement loop.
- `backends/`: `sdpb` (the paper's solver, through `SDPB_SIF`), `sdpa_gmp`
  (arbitrary precision, `pip install sdpa-multiprecision`), `scipy_lp` and
  `cutting_plane` (float64: fast, diagnostic only).

## What is certified

- `scipy_lp` (float64) results are diagnostics. Only the arbitrary-precision
  backends (SDPB, SDPA-GMP) give certified numbers, and their functionals are
  re-verified independently of the solver.
- A window count bracket [A, B] is certified only if both functionals pass the
  whole-window interval certificate, the exact positivity certificate on the
  spins the SDP imposed, and the post-check of the spins it left out up to
  `l_check` (90 in the paper). The rigorous bracket is
  [A / (|a0| + delta_lo), B / (|a0| - delta_up)]. The one function that runs
  all of these gates is `dual.scan.count_bracket.certified_count_bracket`.
- Certified exclusions hold for spectra with a twist gap (no conserved
  currents) and are checked through spin `l_check`. Spin-1 windows count
  parity pairs, so admissible spin-1 counts lie on 2Z (this assumes parity);
  spin-0 counts lie on Z.
- Lambda is quoted as in the paper, odd: the code's `Lambda = 20` is the
  paper's Lambda = 19.
