# examples

Two quick-start scripts, each runnable in seconds after `pip install -e .`:

- `compute_primal_point.py`: the search at two points, one where a consistent
  spectrum is found and one where every attempt fails.
- `compute_dual_bound.py`: two dual count brackets at a low cutoff with the
  window sampled at points: one excludes a gap (no whole number fits), one
  contains the count of the known SU(3) level-1 theory. These are
  demonstrations; `reproduce/count_bracket.py` computes certified brackets.
