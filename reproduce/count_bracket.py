"""One certified window count bracket: the operator count of one spin in one window.

A thin driver over :func:`dual.scan.count_bracket.certified_count_bracket`,
which holds the method and documents it. For a central charge ``c``, an
assumed scalar gap ``gap`` and a spin (0 or 1) it bounds the number of
primaries of that spin in the window

    spin 0:  [gap, gap + w]        spin 1:  [1, 1 + w]

for every unitary modular-invariant spectrum with a twist gap (no conserved
currents) whose non-vacuum scalars all have ``Delta >= gap``, positivity checked
through spin ``l_check``, and reports whether the bracket contains an admissible count
(spin 0: an integer; spin 1: an even integer, which assumes parity invariance:
parity pairs spin +1 with spin -1). A bracket with no admissible count that
passes every certificate EXCLUDES the gap for spectra with a twist gap (no
conserved currents), parity-invariant for the spin-1 2Z lattice, with spins
<= l_check checked: no such spectrum exists. Spins above l_check are not
checked by this code. ``--lattice Z`` for spin 1 drops the parity assumption
(conservative); ``--lattice 2Z`` for spin 0 is refused.

Certified means BOTH sides pass three gates: the whole-window interval
certificate, and two zero-tolerance half-line certificates, one on the spin
continua the SDP imposed (spins 0..lmax) and one on the spins it left out
(lmax < l <= l_check). The rigorous bracket is
``[A / (|a0_lo| + delta_lo), B / (|a0_up| - delta_up)]``, rounded outward, with
``a0`` each functional's bookkeeping slot and ``delta`` the slack each interval
certificate proves. Anything else is a diagnostic.

Settings
--------
--small         Lambda = 8 (order 7), lmax = 16, NN = 20, 512 bits,
                max_outer = 2, all-spin check to spin 30, SDPA-GMP backend
                (``pip install sdpa-multiprecision``): runs on a laptop in minutes.
(default)       the paper recipe: Lambda = 20 (order 19, the
                paper's Lambda = 19), lmax = 40, NN = 40, max_outer = 4,
                SDPB at 1024 bits (``SDPB_SIF`` names the container image),
                all-spin check to spin 90.
Any recipe flag given explicitly overrides the preset.

Lambda is the code's derivative cutoff: the real order is Lambda if odd, else
Lambda - 1, so the code's 20 and the paper's 19 are the same functional space.

Cost of the paper recipe (measured at c = 1.14, windows w = 0.05 ... 2):
spin 0 takes 6-20 min per side with 4 MPI ranks (``--np 4``); spin 1 takes a
median 25-70 min per two-sided bracket on one core (the longest 2.5 h), with
peak memory under 300 MB. Wider windows cost more. ``--small`` takes seconds
to minutes.

Usage::

    python reproduce/count_bracket.py --c 1.14 --gap 0.5171052631578946 --spin 1 --w 0.05 --small
    python reproduce/count_bracket.py --c 1.14 --gap 0.5171052631578946 --spin 0 --w 0.2 --np 4 \\
        --out s0_c1.14_w0.2.json

Numbers accept ``p/q`` fractions (``--gap 301/600``) and are parsed at the
working precision (200 digits).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SMALL = dict(Lambda=8, lmax=16, NN=20, bits=512, max_outer=2, l_check=30, backend="sdpa_gmp")
PAPER = dict(Lambda=20, lmax=40, NN=40, bits=1024, max_outer=4, l_check=90, backend="sdpb")


def number(s):
    """``mpf`` from a decimal or a ``p/q`` fraction string (after the precision is set)."""
    from mpmath import mpf
    if "/" in s:
        p, q = s.split("/")
        return mpf(p) / mpf(q)
    return mpf(s)


def add_recipe_args(ap):
    """Recipe flags shared with ``gap_bound_integrality.py`` (None = take the preset)."""
    ap.add_argument("--small", action="store_true", help="laptop preset (see module docstring)")
    ap.add_argument("--Lambda", type=int, help="code derivative cutoff (paper order: odd part)")
    ap.add_argument("--lmax", type=int, help="spin blocks imposed in the SDP")
    ap.add_argument("--nn", type=int, help="window sample points (also the resampling grid)")
    ap.add_argument("--max-outer", type=int, help="resampling rounds after the first solve")
    ap.add_argument("--bits", type=int, help="solver precision in bits")
    ap.add_argument("--backend", choices=["sdpb", "sdpa_gmp"])
    ap.add_argument("--np", type=int, default=1, help="SDPB MPI ranks (ignored by sdpa_gmp)")
    ap.add_argument("--l-check", type=int, help="all-spin post-check up to this spin")
    ap.add_argument("--lattice", choices=["Z", "2Z"],
                    help="admissible counts (default: Z for spin 0, 2Z for spin 1, which "
                         "assumes parity); Z for spin 1 is conservative, 2Z for spin 0 "
                         "is refused")


def recipe_from(args):
    base = SMALL if args.small else PAPER
    pick = lambda k, v: base[k] if v is None else v
    return dict(Lambda=pick("Lambda", args.Lambda), lmax=pick("lmax", args.lmax),
                NN=pick("NN", args.nn), max_outer=pick("max_outer", args.max_outer),
                precision_bits=pick("bits", args.bits), backend=pick("backend", args.backend),
                np_ranks=args.np, l_check=pick("l_check", args.l_check),
                lattice={None: None, "Z": 1, "2Z": 2}[args.lattice])


def check_lattice(ap, lattice, spin):
    """Refuse a lattice coarser than the admissible counts of ``spin`` before any solve."""
    from dual.scan.count_bracket import default_lattice
    if lattice == "2Z" and default_lattice(spin) == 1:
        ap.error(f"--lattice 2Z is coarser than the admissible counts of spin {spin} "
                 "(scalars do not pair): use Z")


def lattice_note(lattice, l_check):
    """The assumptions an exclusion on ``lattice`` rests on, for the printed verdicts."""
    parity = "parity-invariant spectra (spin +1/-1 pairs)" if lattice == "2Z" else "no parity assumed"
    return (f"counts on {lattice}: {parity}; twist gap (no conserved currents); "
            f"spins <= {l_check} checked")


def summary(rec):
    """One line per bracket: window, recipe, raw and rigorous ends, certificate, verdict."""
    from dual.scan.count_bracket import paper_order
    f = lambda x: "-" if x is None else f"{x:.6f}"
    e = lambda x: "-" if x is None else f"{x:.1e}"
    B_r = "inf" if rec["B_r_str"] == "inf" else f(rec["B_r"])
    return (f"c={rec['c_str']} gap={rec['gap_str']} spin {rec['spin']} window "
            f"[{rec['a']:.6g}, {rec['b']:.6g}]  Lambda = {rec['Lambda']} (order "
            f"{paper_order(rec['Lambda'])}) lmax {rec['lmax']} NN {rec['NN']} "
            f"{rec['backend']} {rec['precision_bits']}-bit\n"
            f"  raw [A, B] = [{f(rec['A'])}, {f(rec['B'])}]  status {rec['lo']['status']}/"
            f"{rec['up']['status']}\n"
            f"  rigorous [A_r, B_r] = [{f(rec['A_r'])}, {B_r}]  delta "
            f"{e(rec['delta_lo'])}/{e(rec['delta_up'])}  in-SDP spins clean: "
            f"{rec['inblock_clean']}  all-spin to {rec['l_check']} clean: "
            f"{rec['allspin_clean']}  certified: {rec['certified']}\n"
            f"  admissible counts on {rec['lattice']}: {rec.get('admissible')}  -> "
            f"{rec['verdict'].upper()}  (wall {rec['wall']:.0f} s)\n"
            f"  [{lattice_note(rec['lattice'], rec['l_check'])}]")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--c", required=True)
    ap.add_argument("--gap", required=True, help="assumed scalar gap")
    ap.add_argument("--spin", type=int, choices=[0, 1], required=True)
    ap.add_argument("--w", required=True, help="window width")
    add_recipe_args(ap)
    ap.add_argument("--out", help="write the JSON record here")
    args = ap.parse_args(argv)

    from dual.scan.count_bracket import certified_count_bracket   # sets 200-digit precision
    check_lattice(ap, args.lattice, args.spin)
    rec = certified_count_bracket(number(args.c), number(args.gap), args.spin,
                                  number(args.w), **recipe_from(args))
    rec["preset"] = "small" if args.small else "paper"
    print(summary(rec), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rec, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
