# Reproducing the paper

Three command-line drivers over the library. The methods live in `primal/` and
`dual/`; each script's docstring lists every option. `--small` runs the same
pipeline at a low cutoff on a laptop; without it a script runs the paper's
setting, which belongs on a cluster (one job per sector, bracket or width).
All costs below were measured on Xeon Gold 6330 cores.

## `primal_scan.py`: the primal boundary

```bash
python reproduce/primal_scan.py --small --out small.json                       # laptop, ~40 s
python reproduce/primal_scan.py --sector lin --workers 20 --out d6_lin.json    # paper setting, one sector
python reproduce/primal_scan.py --resume --out d6_lin.json --rounds 2          # more refinement rounds
```

Grid, then scrub, then rounds of refine and scrub (`primal/scan/continuation.py`).
Each cell climbs a ladder of restarts: the default and an evenly spaced start,
the compactified free boson with a primary at the gap, the converged parameters
of each converged neighbour, then perturbed restarts. The sweep runs from large
c - 1 toward c = 1 and from low gap upward. The scrub re-tries every excluded
cell that touches a candidate from that candidate's parameters; refinement
subdivides the cells that straddle the threshold.

- **Verdict:** candidate iff L < 25 on the 952 fixed torus points
  (`sigma_round_kappa = 4`, `sigma_epsilon = 0`, `tau2_max = Delta_max + 2`).
- **Paper setting:** Delta_max = 6; three sectors, each a 20 x 20 grid over
  Delta_gap in [0.04, 0.60]: `log` (c - 1 in [1e-4, 1e-2], log-spaced), `lin`
  ([1e-2, 0.2]) and `lin2` ([0.2, 0.5]).
- **Cost:** the `lin` sector's grid and scrub took 2.8 h on 20 cores (10,329
  solves). A solve takes 2-5 s on an idle core and about 9 s per core with 20
  concurrent workers. `--optimizer long` costs about 30x per solve and does
  not do better near c = 1, where single cells are chaotic under either
  optimizer.

## `count_bracket.py`: one certified window count bracket

```bash
python reproduce/count_bracket.py --c 1.14 --gap 0.5171052631578946 --spin 1 --w 0.05 --small   # ~10 s
python reproduce/count_bracket.py --c 1.14 --gap 0.5171052631578946 --spin 1 --w 0.05           # paper setting
```

Bounds the number of primaries of one spin in a window, `[gap, gap + w]` for
spin 0 and `[1, 1 + w]` for spin 1, for every unitary modular-invariant
spectrum with a twist gap whose non-vacuum scalars all have Delta >= gap. The
bracket is certified only if both sides pass the whole-window interval
certificate, the exact positivity certificate on spins 0..lmax and the
post-check on lmax < l <= l_check; the verdict is printed with its assumptions
(lattice, parity, l_check).

| | Lambda (order) | lmax | NN | bits | l_check | backend |
|---|---|---|---|---|---|---|
| `--small` | 8 (7) | 16 | 20 | 512 | 30 | SDPA-GMP |
| default (paper) | 20 (19) | 40 | 40 | 1024 | 90 | SDPB |

- **Cost (paper setting):** spin 0 takes 6-20 min per side with 4 MPI ranks
  (`--np 4`); spin 1 takes a median of 25-70 min per bracket on one core (the
  longest 2.5 h), under 300 MB of memory.
- **Check:** at the command above, order 7 gives [3.708435, 8.177879], order
  11 gives [4.063296, 8.079957] and the paper's order 19 gives [4.127733,
  8.064999]; each is certified and contains the next.
- SDPA-GMP reads its constraint matrix in float64, and near the edge of an
  exclusion it can return functionals that fail the exact positivity gate.
  Those brackets are reported as uncertified. Use SDPB for any number you
  quote.

## `gap_bound_integrality.py`: the gap bound from window integrality

```bash
python reproduce/gap_bound_integrality.py --small --out small_gap.json                  # ~30 s
python reproduce/gap_bound_integrality.py --small --backend sdpb --out small_gap.json   # ~8 min
python reproduce/gap_bound_integrality.py --c 1.001 --spin 1 --widths 3.6 4.0 4.4 \
    --g-top 0.2 --g-step 0.005 --sample-step 0.005 --nn 1 --out s1_c1.001.json          # paper setting
```

A certified bracket at gap g with no admissible count (spin 0: no integer;
spin 1: no even integer) proves Delta_gap < g for spectra with a twist gap,
parity-invariant for the spin-1 lattice, with spins <= l_check checked. For
each window width the script searches downward in g for the smallest such
exclusion and reports Delta* = min over w of g_exc(w), with every probe.
`--search bisect` bisects instead of walking; `--resume` continues a stopped
run.

- **Paper settings:** lmax = 40, 1024 bits. Spin 0 at Lambda = 20 (order 19)
  wins at w = 1.0 for c - 1 <= 5e-3 and about 0.75-0.83 at c = 1.02-1.03.
  Spin 1 at Lambda = 20 (order 19) wins at a width that grows toward c = 1:
  0.15 at c = 1.14, 0.3 at 1.1, 1.5 at 1.04, 4.0 for c - 1 <= 1e-3. Near c = 1
  the windows are sampled with step 0.005 (`--sample-step 0.005 --nn 1`).
- **The Lambda = 59 points of Fig. 1** used code Lambda = 60 with wide windows
  (w up to 15) and hours per side. The five at c - 1 <= 1e-2 were certified
  with the window split into 32 sub-windows and an exact closure of the spin
  tail beyond spin 90, which this repository does not include; the five at
  c >= 1.02 used the continuous-window builder
  `dual.assembly.build_window_bound_sdp_continuous`. See
  [`data/paper/README.md`](../data/paper/README.md).
- **`--small`** (c = 1.14, spin 1, order 7): with SDPB it certifies
  Delta_gap < 0.5235; SDPA-GMP finds no certified bound here and says so.
