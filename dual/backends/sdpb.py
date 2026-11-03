"""Arbitrary-precision SDP backend -- SDPB v3 via an apptainer/singularity image.

The *rigorous* tier of the two-tier solve. Unlike
:mod:`dual.backends.sdpa_gmp` (GMP arithmetic but a **float64 cast of
the constraint matrix on ingest** -- the fp64-input wall that produces spurious
"feasibility-wall" cells at high ``c`` / low gap / large ``Lmax``), SDPB reads its
SDP from arbitrary-precision **text** (``pmp2sdp -p <bits>``), so coefficients that
span ~1e-55 (high-gap ``exp(-126)`` prefactors) keep full precision. SDPB is the
gold solver of record: the reference Mathematica pipeline used it, and the
PMP-JSON schema below is the one validated against this exact container on the
interval bound (matching the published value ``4.072272``).

Pipeline (on a machine with ``apptainer``/``singularity`` and the SDPB image):

    build_*_sdp(...)  ->  to_pmp_json()  ->  pmp_in.json
        apptainer exec sdpb.sif pmp2sdp -i pmp_in.json -o sdp_dir -p <bits>
        apptainer exec sdpb.sif [mpirun -n NP] sdpb -s sdp_dir -o out_dir --precision <bits> ...
    parse out_dir/out.txt (terminateReason) + out_dir/y.txt (functional y)
        ->  SDPResult(status, alpha=y, bound = -obj.y (max) / +obj.y (min), feasible)

``bound`` is recomputed in ``mpmath`` from the recovered ``y`` against the EXACT
objective -- the identical formula :mod:`sdpa_gmp` uses -- so a clean cell must
agree between the two backends. ``SDPResult`` mirrors the SDPA-GMP one so the two
backends are swappable behind :mod:`dual.bounds`.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from mpmath import mp, mpf

from ..characters import DampedRational

# PMP-JSON serialization digits. 70 s.f. of text input >> the fp64 the SDPA path
# is stuck at; SDPB re-reads at ``--precision`` bits regardless.
DIGITS = 70


@dataclass
class SDPResult:
    """Result of an SDPB solve -- field-compatible with :class:`sdpa_gmp.SDPResult`.

    - ``status``: ``"optimal"`` / ``"feasible"`` / ``"infeasible"`` / ``"error"``,
      mapped from SDPB's ``terminateReason``.
    - ``alpha``: the recovered dual functional ``y`` as an ``mpf`` list of length
      ``len(sdp.objective)`` (``None`` when SDPB found no solution).
    - ``bound``: ``-objective.alpha`` for ``sense="max"``, ``+objective.alpha`` for
      ``sense="min"`` (``mpf``); ``None`` when ``alpha`` is ``None``.
    - ``feasible``: ``True`` when SDPB reached a usable (optimal/feasible) point.
    - ``reason``: SDPB's raw ``terminateReason`` string. Some feasibility drivers
      branch on it directly ("primal feasible" vs "dual feasible"), which the
      coarse ``status`` map collapses.
    - ``primal_x``: SDPB's PRIMAL solution vector ``x`` (the bootstrap spectral
      measure side), as an ``mpf`` list -- populated ONLY when ``solve_sdp`` is
      called with ``parse_primal=True`` on a usable solve. ``None`` by default,
      so every legacy construction is byte-compatible.
    """

    status: str
    alpha: object
    bound: object
    feasible: bool
    reason: str = ""
    primal_x: object = None


# ---------------------------------------------------------------------------
# PMP-JSON serialization (validated schema)
# ---------------------------------------------------------------------------

def _s(x):
    return mp.nstr(mpf(x), DIGITS)


def _coeff_list(poly):
    # ascending mpmath coeff list (pols2) or a bare scalar mpf (pols1)
    if isinstance(poly, (list, tuple)):
        return [_s(c) for c in poly]
    return [_s(poly)]


def to_pmp_json(sdp):
    """Serialize an assembled :class:`dual.assembly.SDP` to PMP-JSON.

    SDP-agnostic: walks ``sdp.pvms`` exactly as the validated exporter, so it
    covers the spin SDP (``build_bound_sdp_spin``) used by the legacy gap
    curve, not just the interval SDP.
    """
    N = len(sdp.objective)
    blocks = []
    for pvm in sdp.pvms:
        vec = pvm.matrix[0][0]          # 1x1 matrix entry: list over N components
        assert len(vec) == N, (len(vec), N)
        if isinstance(pvm.prefactor, DampedRational):
            dr = pvm.prefactor
            damped = {"constant": _s(dr.const), "base": _s(dr.base),
                      "poles": [_s(p) for p in dr.poles]}
        else:
            damped = {"constant": "1", "base": "1", "poles": []}
        polys = [_coeff_list(vec[n]) for n in range(N)]
        blocks.append({"DampedRational": damped, "polynomials": [[polys]]})
    return {"objective": [_s(o) for o in sdp.objective],
            "normalization": [_s(v) for v in sdp.normalization],
            "PositiveMatrixWithPrefactorArray": blocks}


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_OUT_RE = re.compile(r'(\w+)\s*=\s*"?([^";]+)"?\s*;?')


def _parse_out_txt(path):
    """Parse SDPB ``out.txt`` into a dict of ``key -> value-string``."""
    info = {}
    with open(path) as f:
        for line in f:
            m = _OUT_RE.match(line.strip())
            if m:
                info[m.group(1)] = m.group(2).strip()
    return info


_HEADER_RE = re.compile(r'^\s*\d+(\s+\d+)?\s*$')


def _parse_vector(path):
    """Parse an SDPB output vector file (e.g. ``z.txt``) to a list of ``mpf``.

    SDPB prefixes the data with a ``"<rows> [cols]"`` integer header line; the
    remaining lines are the high-precision entries (one per line, optionally
    brace/comma wrapped). We drop the header, then parse every non-empty token at
    the current ``mp`` precision.
    """
    with open(path) as f:
        lines = f.read().splitlines()
    if lines and _HEADER_RE.match(lines[0]):
        lines = lines[1:]
    text = " ".join(lines).replace("{", " ").replace("}", " ").replace(",", " ")
    return [mpf(tok) for tok in text.split() if tok]


_REASON_STATUS = [
    ("primal-dual optimal", "optimal"),
    ("found dual feasible jump", "feasible"),
    ("dual feasible", "feasible"),
    ("primal feasible", "feasible"),
    # GENUINE INFEASIBILITY (a real feasibility wall), not a solver failure.
    # Above the Hellerman bound c/6+1/3 the spin SDP is primal-infeasible /
    # dual-unbounded; SDPB reports "maxComplementarity exceeded" with the dual
    # objective diverging (~1e101) -- and the termination is BIT-IDENTICAL at
    # maxIterations 500 vs 2000, so it is not iteration-limited (there is no
    # finite optimum to converge to). Mapping it to a distinct "infeasible"
    # status keeps a real feasibility wall from being conflated with a solver
    # crash ("error").
    ("maxComplementarity exceeded", "infeasible"),
    ("primal-dual infeasible", "infeasible"),
    ("primal infeasible", "infeasible"),
    ("dual infeasible", "infeasible"),
    ("maxIterations exceeded", "error"),
    ("maxRuntime exceeded", "error"),
]


def _status_from_reason(reason):
    low = reason.lower()
    for needle, status in _REASON_STATUS:
        if needle.lower() in low:   # needles carry SDPB's mixed case; match case-insensitively
            return status
    return "error"


def _maybe_parse_primal(out_dir, *, parse_primal, status):
    """Parse SDPB's primal solution ``x.txt`` (additive, OPT-IN).

    SDPB always writes ``x.txt`` under ``--writeSolution x,y,z`` (the solve
    command below), but the legacy parse reads only ``z``/``y`` -- the recovered
    DUAL functional ``alpha``. ``x`` is the bootstrap PRIMAL side (the spectral
    measure): the envelope-theorem gap-subgradient ``dV/dgap`` and the extremal
    operator densities live here, and were previously discarded with the workdir.

    Default OFF: with ``parse_primal=False`` ``x.txt`` is never opened, so the
    solve is byte-identical to the legacy path. Returns a list of ``mpf`` on a
    usable (optimal/feasible) solve when the file exists; ``None`` otherwise
    (not requested, non-usable status, or no file -- never raises).
    """
    if not parse_primal or status not in ("optimal", "feasible"):
        return None
    x_txt = os.path.join(out_dir, "x.txt")
    if not os.path.exists(x_txt):
        return None
    return _parse_vector(x_txt)


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def _sif_path():
    return os.environ.get("SDPB_SIF", os.path.expanduser("~/sdpb.sif"))


def solve_sdp(sdp, *, precision_bits=1024, sense="max",
              duality_gap_threshold=mpf(10) ** -30, max_iteration=500,
              np_ranks=None, sif=None, workdir=None, keep=False, verbose=False,
              feasibility=False, checkpoint_dir=None,
              initial_checkpoint_dir=None, parse_primal=False):
    """Solve the assembled ``sdp`` with SDPB and recover ``alpha`` (= dual ``y``).

    ``sense`` selects the driver return formula exactly as :func:`sdpa_gmp.solve_sdp`:
    ``"max"`` (upper-bound drivers, ``norm=-e0``) returns ``bound = -objective.alpha``;
    ``"min"`` (lower-bound drivers) returns ``bound = +objective.alpha``.

    ``feasibility=True`` mirrors a jump-detection feasibility solve: jump
    detection ``--detectPrimalFeasibleJump --detectDualFeasibleJump`` with
    thresholds ``primalError/dualError 1e-40`` and ``dualityGap 1e-60``
    (normal convergence unreachable -- termination IS the verdict, carried in
    ``SDPResult.reason``). ``bound`` is None (obj = 0); ``alpha`` is recovered
    from ``z.txt`` when present, else reconstructed from ``y.txt`` via
    :func:`dual.assembly.restore_coefficients` (a jump termination may not
    write the full solution set).

    ``np_ranks`` MPI ranks (``mpirun -n`` inside the container; ``None``/1 = serial).
    ``sif`` overrides the apptainer image path (default ``$SDPB_SIF`` or ``~/sdpb.sif``).
    ``workdir`` overrides the scratch dir (default a fresh ``tempfile`` dir, removed
    unless ``keep``). Returns an :class:`SDPResult`.

    ``checkpoint_dir`` / ``initial_checkpoint_dir`` (additive): when
    ``checkpoint_dir`` is set, SDPB writes a final interior-point checkpoint there
    (and ``--noFinalCheckpoint`` is dropped); when ``initial_checkpoint_dir`` is
    set, SDPB warm-starts from that checkpoint. Both ``None`` (the default) is
    BYTE-IDENTICAL to the legacy command. A persisted checkpoint dir is the
    CALLER's responsibility -- pass a caller-owned absolute path (the default
    tempdir cleanup is unchanged), since a fresh ``workdir`` is removed in the
    ``finally`` unless ``keep``.
    """
    sif = sif or _sif_path()
    apptainer = shutil.which("apptainer") or shutil.which("singularity")
    if apptainer is None:
        raise RuntimeError("no apptainer/singularity on PATH (SDPB backend needs a "
                           "machine with one of these installed)")
    if not os.path.exists(sif):
        raise FileNotFoundError(f"SDPB image not found: {sif}")

    objective = [mpf(o) for o in sdp.objective]
    bits = int(precision_bits)
    if np_ranks is None:
        # opt-in parallel ranks for callers that cannot pass the kwarg
        # (e.g. bracket_at -> integer_*_interval -> solve_bound_sdpb);
        # cost accounting must then use ranks x wall.
        np_ranks = int(os.environ.get("SDPB_NP_RANKS", "1"))

    own_workdir = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="sdpb_")
    # Isolate OpenMPI's session/tmp dir to this unique workdir. Concurrent
    # `mpirun` instances under one uid otherwise race on the shared $TMPDIR
    # (/tmp) session directory and abort within seconds. Per-call TMPDIR removes
    # the collision; it does not affect the numerics (byte-identical results),
    # and the workdir is rmtree'd in `finally`, so the session files are cleaned
    # up. TMPDIR is honoured by every OpenMPI version; the MCA vars cover the
    # ORTE/PRRTE launchers across the SDPB image's possible OMPI builds. Also
    # applies to pmp2sdp's scratch.
    _env = dict(os.environ)
    _env["TMPDIR"] = workdir
    _env["OMPI_MCA_orte_tmpdir_base"] = workdir
    _env["PRTE_MCA_prte_tmpdir_base"] = workdir
    try:
        pmp_in = os.path.join(workdir, "pmp_in.json")
        sdp_dir = os.path.join(workdir, "sdp")
        out_dir = os.path.join(workdir, "out")

        import json
        with open(pmp_in, "w") as f:
            json.dump(to_pmp_json(sdp), f)

        # 1) preprocess PMP-JSON -> SDPB block dir (text in -> no fp64 cast)
        conv = [apptainer, "exec", sif, "pmp2sdp",
                "-i", pmp_in, "-o", sdp_dir, "-p", str(bits),
                "-v", "1" if verbose else "0"]
        r1 = subprocess.run(conv, capture_output=True, text=True, env=_env)
        if r1.returncode != 0:
            if verbose:
                print("pmp2sdp failed:\n", r1.stdout, r1.stderr)
            return SDPResult(status="error", alpha=None, bound=None, feasible=False)

        # 2) solve. Write the 'z' vector too: SDPB eliminates one dual variable
        # via the normalization, so y.txt has only N-1 entries; z is the FULL
        # N-vector functional restored in the original coordinates (via
        # sdp/normalization.json) and is what aligns with sdp.objective.
        sdpb = ["sdpb", "-s", sdp_dir, "-o", out_dir,
                "--precision", str(bits),
                "--maxIterations", str(int(max_iteration)),
                "--writeSolution", "x,y,z"]
        # Hot-start checkpoint wiring (additive). The None/None default is
        # BYTE-IDENTICAL to the legacy command (--noFinalCheckpoint, no
        # checkpoint flags) so existing runs are unaffected; only a
        # non-None kwarg changes the argv.
        if checkpoint_dir is not None:
            sdpb += ["--checkpointDir", checkpoint_dir]   # write a final checkpoint
        else:
            sdpb += ["--noFinalCheckpoint"]
        if initial_checkpoint_dir is not None:
            sdpb += ["--initialCheckpointDir", initial_checkpoint_dir]  # warm start
        if feasibility:
            # jump-detection feasibility flags: the thresholds make normal
            # convergence unreachable; the jump detectors terminate with
            # "primal/dual feasible jump detected".
            sdpb += ["--primalErrorThreshold", "1e-40",
                     "--dualErrorThreshold", "1e-40",
                     "--dualityGapThreshold", "1e-60",
                     "--detectPrimalFeasibleJump", "--detectDualFeasibleJump"]
        else:
            sdpb += ["--dualityGapThreshold",
                     mp.nstr(mpf(duality_gap_threshold), 6)]
        if np_ranks and int(np_ranks) > 1:
            # --oversubscribe: OpenMPI's slot count under the cgroup is too
            # conservative and otherwise aborts before launching.
            cmd = [apptainer, "exec", sif, "mpirun", "--oversubscribe",
                   "-n", str(int(np_ranks))] + sdpb
        else:
            cmd = [apptainer, "exec", sif] + sdpb
        r2 = subprocess.run(cmd, capture_output=True, text=True, env=_env)
        if verbose:
            print(r2.stdout[-2000:])
            if r2.stderr:
                print("STDERR:", r2.stderr[-2000:])

        out_txt = os.path.join(out_dir, "out.txt")
        z_txt = os.path.join(out_dir, "z.txt")
        y_txt = os.path.join(out_dir, "y.txt")
        if not os.path.exists(out_txt):
            return SDPResult(status="error", alpha=None, bound=None,
                             feasible=False, reason="no out.txt")

        info = _parse_out_txt(out_txt)
        reason = info.get("terminateReason", "")
        status = _status_from_reason(reason)
        # SDPB's primal vector x.txt, default OFF (additive, opt-in).
        primal_x = _maybe_parse_primal(out_dir, parse_primal=parse_primal,
                                       status=status)

        # recover the functional: z.txt (full N-vector) preferred; on a
        # jump-terminated feasibility solve fall back to y.txt (N-1 entries)
        # + restore_coefficients (the original restoreCoefficients pivot).
        alpha = None
        if status in ("optimal", "feasible"):
            if os.path.exists(z_txt):
                alpha = _parse_vector(z_txt)
            elif feasibility and os.path.exists(y_txt):
                from ..assembly import restore_coefficients
                y = _parse_vector(y_txt)
                if len(y) == len(objective) - 1:
                    alpha = restore_coefficients(sdp.normalization, y)
            if alpha is not None and len(alpha) != len(objective):
                return SDPResult(status="error", alpha=None, bound=None,
                                 feasible=False, reason=reason)

        if feasibility:
            # termination IS the verdict; bound is meaningless (obj = 0)
            return SDPResult(status=status, alpha=alpha, bound=None,
                             feasible=alpha is not None, reason=reason,
                             primal_x=primal_x)

        if status not in ("optimal", "feasible") or alpha is None:
            return SDPResult(status=status, alpha=None, bound=None,
                             feasible=False, reason=reason)

        obj_dot = mp.fsum(objective[k] * alpha[k] for k in range(len(objective)))
        bound = -obj_dot if sense == "max" else obj_dot
        return SDPResult(status=status, alpha=alpha, bound=bound, feasible=True,
                         reason=reason, primal_x=primal_x)
    finally:
        if own_workdir and not keep:
            shutil.rmtree(workdir, ignore_errors=True)
