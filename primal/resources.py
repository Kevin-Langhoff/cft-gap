"""Paths and cached-reference loaders shared by the scan driver and tests.

The scan driver and the test suite both need the same cached inputs: a
reference CFT spectrum (used to build the error-estimate denominator in
``optimize.py``) and a fixed grid of torus points for the final
"refined" evaluation. Keep the paths and loaders in one place so they
can't drift apart.

These caches are one-time precomputed data, not something regenerated
at run time — the loaders raise ``FileNotFoundError`` when a cache is
missing so the caller can decide how to surface the failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data_types import CFTData, DSD

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"

REFERENCE_DIR = DATA_DIR / "reference"    # cached reference spectrum + analytic anchors
RUNS_DIR = DATA_DIR / "runs"              # scan-driver outputs


def resolve_fbcb_cut(dmax: int | float, explicit_cut: int | None) -> int:
    """Resolve the reference-cache cut for the scan driver.

    Explicit CLI cuts are preserved exactly; omitted cuts follow the
    project rule: cut 12 at dimension cutoff >= 6, otherwise cut 6 (the
    higher cut keeps the error estimate from vanishing identically at
    c=1 once the search probes deeper truncations).
    """
    if explicit_cut is not None:
        return int(explicit_cut)
    return 12 if float(dmax) >= 6.0 else 6


def fbcb_cache_path(cut: int = 6) -> Path:
    return REFERENCE_DIR / f"fbcb_sqrt2_{cut}.json"


def fine_test_taus_path() -> Path:
    return REFERENCE_DIR / "default_fine_deterministic_taus.json"


def load_fbcb_sqrt2(cut: int = 6) -> CFTData:
    """Load the cached free-boson-at-R=sqrt(2) reference spectrum at the
    given dimension cut. Raises ``FileNotFoundError`` if the cache is
    missing."""
    path = fbcb_cache_path(cut)
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    return CFTData(
        name=f"FBCB_Rsqrt2_cut{int(data['cut'])}",
        central_charge=float(data["c"]),
        primaries=tuple(
            DSD(float(d), int(j), int(g)) for d, j, g in data["primaries"]
        ),
        max_delta=float(data["cut"]),
        variables=(),
    )


def load_fine_test_taus() -> np.ndarray:
    path = fine_test_taus_path()
    if not path.exists():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text())
    return np.array([complex(re, im) for re, im in raw], dtype=np.complex128)
