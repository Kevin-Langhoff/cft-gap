# Data behind the paper's figures

| file | figure | content |
|---|---|---|
| [`fig1_data.json`](fig1_data.json) | Fig. 1 (the (c, gap) plane) | the record the Fig. 1 script drew (the PDF in the paper is byte-identical to its output), plus the unrounded primal polyline and refinement anchors |
| [`cft_census_points.json`](cft_census_points.json) | Fig. 1, grey crosses and bar | the known CFTs the same script drew |
| [`count_containment.json`](count_containment.json) | Supplemental Material, window counts of an admitted search spectrum (spin-1 panel) | certified spin-1 count brackets against a Delta_max = 6 search spectrum |

Numbers are bit-identical to the records the figure scripts read; only file
names, run identifiers and scheduling notes were removed, and labels were
reworded. The Supplemental Material figure now also has a spin-0 panel, whose
data is not in this directory.

## Conventions

- **Lambda** is quoted as in the paper (odd). Code values appear in the key
  `s1_L20` and in `count_containment.json` (`Lambda` = 20 is the paper's 19).
- **Counts.** As in the paper, spin-1 windows [1, 1 + w] count parity pairs
  and scalar windows [gap, gap + w] count scalars; both must be integers. The
  records count primaries (two per spin-1 pair), so an admissible spin-1 count
  there is even. `count_containment.json` stores
  spin-1 counts in primaries; divide by 2 for the pairs the figure plots.
- **Certificate or diagnostic.** A bracket is a certificate only if the window
  constraint is proven on the whole window and the all-spin post-check is
  clean on both sides. Two whole-window routes occur: (i) a functional solved
  at window samples, then proven on the whole window by
  `dual.postcheck.certify_window_constraint` up to an absolute slack delta,
  with rigorous bracket [A / (1 + delta_lo), B / (1 - delta_up)]; (ii) the
  continuous-window SDP (`dual.assembly.build_window_bound_sdp_continuous`),
  whose window blocks are certified by `dual.verify.certify_functional` at
  relative tolerance 1e-13, with the raw [A, B] used. A bracket widened by a
  modelled or densely sampled window dip (eta) is not a certificate. The
  post-check runs through spin 90, a finite extension. Certified exclusions
  assume parity and no conserved currents beyond the stress tensor.
- **Search.** L = exp(S) - 1 with S = mean log(1 + (R / sigma)^2); a spectrum
  is admitted iff L < 25. Search results are never exclusions.

## `fig1_data.json`

| key | content |
|---|---|
| `search.navy_points` | the drawn primal boundary at Delta_max = 6 for 1.0001 <= c <= 1.1, [c, gap], 1759 points rounded to 5 decimals |
| `search.navy_points_full_precision` | the same polyline unrounded |
| `search.survey_contours` | the boundary for c > 1.1: (c - 1, gap) pairs of the L = 25 contour of the survey |
| `refinement.anchors` | the 163 bisected charges the line below c = 1.1 interpolates: drawn `gap`, the last admitted `lower`, the first excluded `upper`, their scores, the bisection `tolerance` |
| `s0.points` | the spin-0 dual curve at Lambda = 19: 57 points (`c`, `gap`, `tier`, `W` where stored) |
| `s1_L20.points` | the spin-1 dual curve at Lambda = 19: 44 points (`c`, `gap`, `W`) |
| `strongest_s1` | the ten Lambda = 59 spin-1 points (diamonds): `c`, `gap`, `W`, `dip` (how each was certified) |
| `fill` | the shaded region: above the pointwise minimum of the drawn dual series, interpolated linearly between points |
| `records_beating_the_drawn_Lambda19_curve` | two Lambda = 39 records below the spin-1 curve; eta-widened, not certificates, not drawn |
| other keys | `layout`, drawing rules and one-line notes |

**How each dual series is certified:**

| series | points | certificate |
|---|---|---|
| `s0` (Lambda = 19) | 57 | the drawn records are sampled-window solves; a replay of their stored functionals through `certify_window_constraint` closes on all 57, with integer-free delta-widened brackets and a clean post-check through spin 90 |
| `s1_L20` (Lambda = 19), the 15 charges named in `s1_L20.source` | 15 | sampled-window records; the replay through `certify_window_constraint` closes on all 15 (no even integer in the delta-widened bracket) |
| `s1_L20`, the other charges | 29 | continuous-window SDP (route ii), both sides pass `certify_functional` at 1e-13 and the post-check over spins 41-90, [A, B] holds no even integer; no `certify_window_constraint` replay |
| `strongest_s1`, c - 1 = 1e-4 ... 1e-2 | 5 | whole-window interval certificate (window split into 32 sub-windows) on both sides, post-check over spins 41-90, and an exact check of the spin tail from spin 91; delta-widened brackets [6.00028, 7.91492], [6.00092, 7.91586], [6.00197, 7.94492], [6.00966, 7.97079], [6.06320, 7.94282] |
| `strongest_s1`, c = 1.02 ... 1.1 | 5 | continuous-window SDP (route ii), `certify_functional` at 1e-13, post-check over spins 41-90; no `certify_window_constraint` delta and no spin-tail record. A dense check finds a window dip of at most 2e-11 on four of them, but 6.1e-4 / 4.5e-4 at c = 1.02 (widened by those values the bracket [6.1287, 7.7888] still holds no even integer) |

The replay deltas and the spin-tail records are not stored here.

## `cft_census_points.json`

The unitary, compact, bosonic 2d CFTs with 1 < c <= 3/2 and Delta_gap <= 3/5
from a census of the literature and of explicit constructions, with exact c
and Delta_gap (the lowest non-vacuum Virasoro primary of any spin). `marks`
(171 points with c != 3/2) are the grey crosses; `c_3_2_segment` (the
continuous c = 3/2 families, 0 < gap <= 3/8) is the grey bar, on which the 16
`c_3_2_points` lie. Each entry has `c`, `gap` (exact, as strings), their float
values, `families` and `labels`. Every listed theory has a conserved current
beyond the stress tensor, so none satisfies the twist-gap assumption of the
certified exclusions.

## `count_containment.json`

Spin-1 count brackets for Delta_max = 6 search spectra: spin 1, window
[1, 1 + w], code Lambda = 20 (paper 19), lmax = 40 (60 for re-solved windows,
as each record's `lmax` says), 40 window samples, SDPB at 1024 bits, then the
interval certificate (tolerance relaxed 1e-9, 1e-7, 1e-6 until it closes) and
the post-check to spin 90. The bracket is solved at the search spectrum's own
scalar gap.

- `anatomy`: the case the figure draws, c = 1.14, Delta_gap = 0.5171 (L =
  1.32), windows w = 0.05 ... 2. All 7 brackets are certified and each
  contains the search count. Example: at w = 0.05, [A_c, B_c] = [4.1276,
  8.0651] primaries = [2.064, 4.033] pairs, and the search has 3 pairs.
- `grid`: 7 charges (c = 1.04 ... 1.16) x 7 windows. 47 of 49 brackets are
  certified and all 47 contain the search count; (1.08, w = 0.65) and (1.16,
  w = 0.35) fail the post-check and are diagnostics.
- `diagnostic_c1p01`: an uncertified diagnostic at c = 1.01 (Lambda = 35, no
  post-check, brackets at an assumed gap of 0.06). Not drawn.
- `rows`, `design`, `budget`, `spin1_overcount_at_known_cft`: the search
  spectra tested, the grid design, solver cost, and the search's spin-1 count
  bias at a known CFT.

Window record fields: `W` and `x = 1 + W`; `N` (search count, primaries); `A`,
`B` (raw bracket); `delta_lo`, `delta_up` (interval-certificate slack); `A_c`,
`B_c` (rigorous bracket, valid where `certified` is true); `in_cert`
(A_c <= N <= B_c); `certified`; `allspin_ok`, `allspin`; `lmax`; `evens` (even
integers in [A, B]). Fields with `eta` are a legacy projected-slack bracket,
never the certificate.
