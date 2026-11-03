"""Quick start: search for a consistent CFT spectrum at two sample points.

For each point (central charge c, spectral gap) the primal method builds a
candidate spectrum and runs a local optimizer that tries to make it modular
invariant. The final loss says how close it got:

    loss < 4        a good candidate spectrum exists (the point looks allowed)
    loss >= 1e7     every attempt fails badly (the point is strongly excluded)

These are the thresholds of the reference-grid scan used here (Delta_max = 4,
independent restarts). The paper's landscape uses ``reproduce/primal_scan.py``
(Delta_max = 6, continuation scan, candidate iff L < 25 with the float64
rounding term included). Runs in about 15 s on a laptop.
"""

from primal.scan.grid import run_landscape_scan


def classify(loss: float) -> str:
    if loss < 4.0:
        return "CFT candidate found (allowed)"
    if loss >= 1e7:
        return "strongly excluded"
    return "marginal (between thresholds)"


def main() -> None:
    # One point deep in the allowed region, one high-gap point near c=1
    # where the search is expected to fail.
    sample_points = [(1.30, 0.10), (1.02, 0.55)]

    for c, gap in sample_points:
        results = run_landscape_scan(
            dmax=4,
            c_range=(c, c, 1.0),      # single-value ranges -> a 1-point grid
            gap_range=(gap, gap, 1.0),
        )
        (res,) = results
        loss = res["best_loss_refined"]
        print(f"c = {c:5.2f}, gap = {gap:4.2f}  ->  loss = {loss:10.3e}   "
              f"[{classify(loss)}]")


if __name__ == "__main__":
    main()
