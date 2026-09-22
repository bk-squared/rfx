"""The openEMS MSL notch filter tutorial as a reproduce gate, shared by cases.

WHAT THIS MODULE IS
-------------------
Two cross-validation cases need the same first step: run openEMS's own
``python/Tutorials/MSL_NotchFilter.py``, verbatim, and refuse to record anything
about the case's own structure unless that tutorial reproduces its published
notch. That step, its gate, the sanity helpers every real pass carries, and the
record/evidence writers live here once. Each case keeps its own Stage B -- its
structure, its deltas from the tutorial, its dry-run and self-check text -- in
its own maker.

NEVER RUN BY CI. The external solver runs by hand, on the cluster.

WHAT THE TUTORIAL IS
--------------------
A 600 um wide microstrip line on 254 um of lossless RO4350B (eps_r = 3.66) over
a PEC ground, with a 12 mm open-circuit stub of the same width branching off the
middle of the line. The stub is a quarter wavelength near 3.7 GHz, where its
open end transforms into a short across the line and |S21| collapses into a deep
transmission notch.

THE TUTORIAL, ITS SOURCE AND ITS ATTRIBUTION
--------------------------------------------
Copied from the docstring of ``validation/crossval/20_msl_phase_referee.py``
(that script's STAGE A section), which is where this repository's faithful port
of the tutorial already lives:

    ``python/Tutorials/MSL_NotchFilter.py`` (fetched verbatim 2026-08-04 via
    ``gh api repos/thliebig/openEMS/contents/python/Tutorials/MSL_NotchFilter.py``;
    "(c) 2016-2023 Thorsten Liebig", "Tested with python 3.10, openEMS v0.0.35+")
    builds an open-circuit-stub microstrip notch filter on RO4350B (eps_r=3.66,
    h_sub=254um), MSL_width=600um, MSL_length=50mm (each side), stub_length=12mm,
    f_max=7GHz, boundary ``['PML_8','PML_8','MUR','MUR','PEC','MUR']``,
    ``SetGaussExcite(f_max/2, f_max/2)`` (both center AND corner set to the SAME
    3.5GHz -- an unusual-looking but literal reading, reproduced verbatim rather
    than "corrected"). ``AddMSLPort`` (thirds-rule mesh already built into the
    tutorial's own ``third_mesh`` refinement at the trace's Y-edges) with port[0]
    ``excite=-1`` (verbatim, not "corrected" to +1) and ``FeedShift=10*resolution``,
    ``MeasPlaneShift=MSL_length/3`` on BOTH ports. openEMS(NrTS, EndCriteria) are
    left at the library's own defaults (NrTS~=1e9, EndCriteria=1e-5 --
    ``python/openEMS/openEMS.pyx`` docstring, fetched 2026-08-04) because the
    tutorial itself never overrides them.

THE RECORDED REPRODUCTION (task recipe ``external_solver_comparator.md``, step 2)
---------------------------------------------------------------------------------
The port above has already reproduced the tutorial's own notch, and that is the
number the Stage A gate below is built on:

    measured f_notch  3.6711 GHz
    analytic f_notch  3.6872 GHz  (quarter-wave open stub, Hammerstad-Jensen
                                   eps_eff; recomputed below as F_NOTCH_AN_HZ)
    deviation         0.44 %
    VESSL run         369367251705 (2026-08-04)
    log               validation/crossval/_20_msl_phase_referee_logs/
                      20260804T070702Z_run.log  (its lines 10-17)
    wall time         41.6 s for the real pass, 8 threads

THAT RECORD IS AN AUDIT TRAIL, NEVER A CASE'S GATE
--------------------------------------------------
``validation/crossval/07_sheen_lpf.py``'s own reproduce-gate record names the
failure this distinction exists to stop, and it is quoted here rather than
summarised because a summary is how the decisive sentence gets dropped:

    "07_sheen_lpf.py's own run_openems_tutorial(), pre-#971: the docstring and
    its print both claimed 'known-good ... S21 notch ~3.43 GHz (repo fixture:
    openEMS 3.4286 GHz)' -- that number was never this function's own
    measurement, it was cv06b's realized-board reading, later shown to be an
    outlier by an independent Palace-FEM referee. Do not cite a sibling case's
    number as this function's own known-good without running THIS function and
    checking the result against F_NOTCH_TUTORIAL_DECLARED_HZ."

So: ``REPRODUCE_GATE_RECORD`` below travels in every record this module writes,
as provenance for the tutorial. It is NOT what any run is judged against. Stage
A runs the tutorial in the same invocation as the case's own stages and is
gated, by ``stage_a_gate_verdict``, against ``F_NOTCH_AN_HZ`` -- recomputed here
from the tutorial's own declared geometry, not copied from any case.

THE SANITY GATES, AND FROM WHERE
--------------------------------
All from ``validation/crossval/20_msl_phase_referee.py`` (byte-identical copies,
pinned by ``tests/crossval/msl_notch_filter/test_reference_maker.py``), applied
unchanged to EVERY real pass of EVERY case:

  * ``_scan_stdout_for_bad_patterns`` -- stdout AND stderr scan for "Unused
    primitive" / "not on the mesh" / "unused excitation" before the run is
    trusted, plus the excitation-clipping patterns on real passes only;
  * ``_run_openems_capturing_stdout`` -- captures fd 1 AND fd 2 (CSXCAD writes
    some port/mesh warnings to stderr; an fd-1-only capture saw none of them);
  * ``_check_excitation_and_trace`` -- nonzero excitation energy and non-empty,
    non-zero port voltage traces, with no absolute floor;
  * ``_log_indicates_truncation`` -- openEMS's own "reached before the
    end-criteria of" text. A real pass that trips it FAILS;
  * ``_non_physical_guard`` -- |S| finite and <= 2;
  * ``_passivity_witness`` -- max(|S11|^2+|S21|^2) <= 1 + tol over the calling
    case's own witness band, supplied through the function's own ``idx``
    argument. The function itself is untouched.

WHAT A CASE SUPPLIES, AND WHAT IT MUST NOT CHANGE
-------------------------------------------------
``run_stage`` takes the case's builder, its frequency grid, its witness band,
its real-pass NrTS/EndCriteria, its mesh-summary and meta callbacks, and its
spectral-feature callback. It does not take a way to skip a gate. A case that
needs a different band gives a band; a case that needs a different verdict does
not get one here.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants -- the tutorial's own, copied from validation/crossval/
# 20_msl_phase_referee.py's Stage A block.
# ---------------------------------------------------------------------------
_C0 = 2.998e8  # m/s -- matches the coax/notch precedent's own constant

A_UNIT = 1.0e-6  # CSXCAD length unit: um (tutorial's own choice)
A_MSL_LENGTH_UM = 50000.0
A_MSL_WIDTH_UM = 600.0
A_SUBSTRATE_THICKNESS_UM = 254.0
A_SUBSTRATE_EPR = 3.66
A_STUB_LENGTH_UM = 12.0e3
A_F_MAX_HZ = 7.0e9
A_N_FREQS = 1601  # tutorial's own np.linspace(1e6, f_max, 1601)

A_BOUNDARY = ["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"]
A_PML_CELLS = 8  # "PML_8" -- the x faces only

# Quarter-wave open-stub notch, Hammerstad-Jensen eps_eff. Recomputed here, not
# copy-pasted, exactly as the precedent does (it cross-checks to 5 significant
# figures against this repo's own recorded "fringing-free analytic 3.69 GHz").
_A_STUB_LEN_M = 12.0e-3
_A_W_TRACE_M = 600e-6
_A_H_SUB_M = 254e-6
_A_EPS_R = 3.66
_A_U = _A_W_TRACE_M / _A_H_SUB_M
_A_EPS_EFF = (_A_EPS_R + 1.0) / 2.0 + (_A_EPS_R - 1.0) / 2.0 * (1.0 + 12.0 / _A_U) ** -0.5
F_NOTCH_AN_HZ = _C0 / (4.0 * _A_STUB_LEN_M * np.sqrt(_A_EPS_EFF))

# The recorded reproduction, as an audit artifact carried into every record this
# script writes (the precedent's REPRODUCE_GATE_RECORD, narrowed to Stage A).
REPRODUCE_GATE_RECORD = {
    "tutorial": {
        "repo": "thliebig/openEMS",
        "path": "python/Tutorials/MSL_NotchFilter.py",
        "attribution": "(c) 2016-2023 Thorsten Liebig",
        "fetched_verbatim_on": "2026-08-04",
        "fetched_via": "gh api repos/thliebig/openEMS/contents/python/Tutorials/MSL_NotchFilter.py",
    },
    "port_in_this_repository": "validation/crossval/20_msl_phase_referee.py",
    "reproduced_f_notch_hz": 3671100625.0,
    "analytic_f_notch_hz": 3687193135.4851503,
    "reproduced_f_notch_dev_pct": 0.4364433837294213,
    "vessl_run_id": "369367251705",
    "log_path": "validation/crossval/_20_msl_phase_referee_logs/20260804T070702Z_run.log",
    "log_lines": "10-17",
    "real_pass_wall_time_s": 41.6,
    "real_pass_wall_time_source": (
        "validation/crossval/_20_msl_phase_referee_logs/20260804T055009Z_result.json, "
        "stage_a.elapsed_s -- the wall time is not in the run log's text"
    ),
    "verified_on": "2026-08-04",
}

# STAGE_A_GATE -- copied, with its justification, from the port's own
# REPRODUCE_GATE_RECORD["gate"] (validation/crossval/20_msl_phase_referee.py):
#
#   One-sided-low-biased band: docs/agent-memory/rfx-known-issues.md records
#   openEMS reading ~7% LOW vs the fringing-free analytic on a DIFFERENT (not
#   identical) line-length/domain combination of this SAME substrate/trace/stub
#   -- PREDICTED, not yet measured for THIS exact tutorial geometry, hence the
#   generosity.
#
# The band is unchanged here. What the band is applied TO changed: the shared
# estimator's refined frequency over 2-7 GHz, not a bare argmin over
# 0.5-1.5 x F_NOTCH_AN_HZ.
STAGE_A_GATE = {
    "f_notch_lo_hz": 0.80 * float(F_NOTCH_AN_HZ),
    "f_notch_hi_hz": 1.05 * float(F_NOTCH_AN_HZ),
}

# A frequency band alone does not say a notch is there. A through line with no
# stub has a |S21| minimum somewhere in 2-7 GHz too, and it would sit inside the
# band above. The gate therefore also requires the minimum to be a NOTCH: at
# least 20 dB below unity -- the same deep-null level the case itself uses
# (PI, 2026-09-22). A thru line fails it, which is what the self-check plants.
STAGE_A_MIN_DEPTH_DB = -20.0


# The passivity tolerance, unchanged: max(|S11|^2+|S21|^2) <= 1.05.
PASSIVITY_TOL = 0.05
# The tutorial's own substrate z recipe is linspace(0, h_sub, 5) -- 4 cells.
# The rung factor scales it too (see DELTA 2).
A_SUBSTRATE_Z_CELLS = 4


def substrate_z_cells(resolution_factor: float) -> int:
    """Substrate cells at a rung: round(4 / factor). 4, 6, 8 at 1.0, 1/sqrt2, 0.5."""
    return max(1, int(round(A_SUBSTRATE_Z_CELLS / resolution_factor)))

# The band the TUTORIAL's own Stage A is read over: 2-7 GHz. The tutorial's
# CalcPort grid starts near 1 MHz -- three and a half decades below the
# Gaussian excitation's 3.5 GHz centre -- where both port voltages are at the
# numerical floor and ``uf_ref/uf_inc`` is a ratio of two such numbers rather
# than an S-parameter the structure produced. A case's OWN Stage B band is the
# case's to declare; this one is not.
STAGE_A_WITNESS_BAND_HZ = (2.0e9, 7.0e9)
STAGE_A_NOTCH_BAND_GHZ = (STAGE_A_WITNESS_BAND_HZ[0] / 1e9,
                          STAGE_A_WITNESS_BAND_HZ[1] / 1e9)

# ---------------------------------------------------------------------------
# openEMS import plumbing (deferred, matches the coax/thru precedent so
# this module stays importable -- and testable -- without openEMS).
# ---------------------------------------------------------------------------
def _ensure_openems_numpy_compat() -> None:
    for name, value in {"float": float, "int": int, "complex": complex, "mat": np.matrix}.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _import_openems():
    _ensure_openems_numpy_compat()
    from CSXCAD.CSXCAD import ContinuousStructure
    from openEMS.openEMS import openEMS
    from openEMS.ports import MSLPort
    return ContinuousStructure, openEMS, MSLPort


# ---------------------------------------------------------------------------
# Shared sanity-check helpers (ticked against the coax/thru precedent above)
# ---------------------------------------------------------------------------
_BAD_STDOUT_PATTERNS = ("Unused primitive", "not on the mesh", "unused excitation")

# Truncation patterns (D3/TRUNCATION PATTERNS fix, run-1 forensics, review
# 2026-08-04): openEMS's own excitation-pulse-clipping warnings. Checked ONLY
# via ``check_truncation=True`` (real-run scans), NEVER for smoke scans --
# the smoke runs in THIS script are deliberately tiny (NrTS=200,
# EndCriteria=0.0/"-infdB", see ``_build_stage_a_notch_tutorial``/
# ``_build_stage_b``'s own smoke calls), so their excitation pulse (tens of
# thousands of timesteps at this script's own f0/fc) is ALWAYS clipped by
# construction -- run-1's own committed log (``validation/crossval/
# _20_msl_phase_referee_logs/20260804T070702Z_run.log``) shows both strings
# firing for the smoke portion of BOTH stages while the REAL portion (which
# reached its own EndCriteria, per REPRODUCE_GATE_RECORD["settling_evidence"])
# shows neither -- the exact positive control this scoping relies on: same
# binary, same stream, smoke trips it and real does not.
_TRUNCATION_STDOUT_PATTERNS = (
    "Cutting to max number of timesteps",
    "max. number of timesteps is smaller than three times the excitation",
)

# Benign, narrowly-scoped exception (GUARD CHANNEL-GAP FIX, run-1 forensics,
# 2026-08-04): CSXCAD's "Unused primitive (type: Box) detected in property:
# ..." warning fires for port0_metal/port1_metal on EVERY real Stage B run,
# for a structural reason, not a defect -- Stage B's M3 topology fix (module
# docstring "Port topology") gives the thru line's own trace box (property
# "copper") a span EXACTLY feed_x0..feed_x1, which fully covers each port's
# own SHORTER metal box ([start, start+port_w]); CSXCAD's own priority-based
# rasterizer resolves the overlap to the higher-priority primitive (both are
# priority=10 in this script, and the trace is added to the CSX tree first),
# so the port's own box never contributes a rasterized cell of its own and
# CSXCAD reports it "unused". This is a REPORTING artifact of two PEC
# primitives sharing the same physical footprint, not a missing-conductor
# defect -- the conduction proof is |S21| ~= 1.0 band-wide (run-1: 0.9985 to
# 1.0087, self_consistency_openems/passivity BOTH passed), which would be
# impossible if the line were actually open-circuited at either port.
# Scoped to the EXACT two property names below -- any OTHER "Unused
# primitive" (e.g. property: substrate!, a genuinely dropped conductor
# elsewhere) still trips the gate.
_ALLOWLISTED_UNUSED_PRIMITIVE_PROPERTIES = ("port0_metal!", "port1_metal!")


def _scan_stdout_for_bad_patterns(log_text: str, label: str, *, check_truncation: bool = False) -> None:
    """Pre-solve fail-fast gate. ``check_truncation=True`` ADDS
    ``_TRUNCATION_STDOUT_PATTERNS`` to the scan -- callers must pass this
    ONLY for a REAL run's own captured log, never a smoke run's (see
    ``_TRUNCATION_STDOUT_PATTERNS``'s own docstring for why: smoke's tiny
    NrTS/EndCriteria budget makes those patterns fire by design there).
    """
    patterns = _BAD_STDOUT_PATTERNS + (_TRUNCATION_STDOUT_PATTERNS if check_truncation else ())
    hits = []
    for line in log_text.splitlines():
        line_lower = line.lower()
        if not any(p.lower() in line_lower for p in patterns):
            continue
        if "unused primitive" in line_lower and any(
            f"property: {prop}".lower() in line_lower
            for prop in _ALLOWLISTED_UNUSED_PRIMITIVE_PROPERTIES
        ):
            continue  # allowlisted -- see _ALLOWLISTED_UNUSED_PRIMITIVE_PROPERTIES docstring
        hits.append(line.strip())
    if hits:
        raise RuntimeError(
            f"[{label}] pre-solve mesh/port fail-fast gate tripped: openEMS "
            f"stdout/stderr contains {hits!r}. Aborting BEFORE the full NrTS "
            f"budget is spent."
        )


def _run_openems_capturing_stdout(fdtd, sim_path: str, *, threads: int) -> str:
    """Run openEMS while capturing its OS-level stdout AND stderr to a file
    we can grep.

    ``fdtd.Run()`` invokes the openEMS C++ binary; its stdout/stderr go to
    the process's OS-level fd 1 / fd 2, not Python's ``sys.stdout``/
    ``sys.stderr`` -- redirect BOTH fds, restoring both afterward even on
    error.

    GUARD CHANNEL-GAP FIX (run-1 forensics, 2026-08-04): the pre-fix version
    of this function redirected fd 1 ONLY. CSXCAD emits some port/mesh
    warnings on STDERR, not stdout -- e.g. "Unused primitive (type: Box)
    detected in property: port0_metal!/port1_metal!", which appears 4x in
    run-1's own VESSL-captured ``run.log`` (a terminal capture that sees
    BOTH fds; see ``validation/crossval/_20_msl_phase_referee_logs/
    20260804T070702Z_run.log`` lines 24-25/29-30) but was ABSENT from this
    function's own fd-1-only capture file -- ``_scan_stdout_for_bad_
    patterns`` never saw it and the pre-solve fail-fast gate below silently
    passed (rc=0) despite the warning having actually occurred. Both fds are
    now dup2'd into the SAME capture file so the scan below sees everything
    the terminal would have.
    """
    os.makedirs(sim_path, exist_ok=True)
    log_path = os.path.join(sim_path, "_openems_stdout.log")
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    with open(log_path, "w") as logf:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(logf.fileno(), stdout_fd)
        os.dup2(logf.fileno(), stderr_fd)
        try:
            fdtd.Run(sim_path, cleanup=True, verbose=1, numThreads=threads)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
    with open(log_path) as logf:
        return logf.read()


_END_CRITERIA_NOT_REACHED_TEXT = "reached before the end-criteria of"


def _log_indicates_truncation(real_log: str) -> bool:
    """D3 fix (run-1 forensics, review 2026-08-04): the single, pure,
    directly-testable source of truth for "did this REAL run hit its NrTS
    cap without reaching its own EndCriteria decay target" -- openEMS's own
    ``RunFDTD: Warning: Max. number of timesteps was reached before the
    end-criteria of ... was reached`` text (visible only after the fd-2
    capture fix; this exact phrase appears verbatim in run-1's committed
    ``run.log``, in the SMOKE portion of both stages, never the real
    portion). Both ``_run_stage_a_reproduce_gate``'s ``truncated_suspected``
    and ``_run_stage_b``'s ``end_criteria_not_reached``/``truncated`` call
    this SAME function on their own real (never smoke) captured log, rather
    than each maintaining its own copy of the substring or (the pre-fix
    Stage B behavior) an unrelated, structurally-unreachable probe-row-count
    comparison.
    """
    return _END_CRITERIA_NOT_REACHED_TEXT in real_log


def _check_excitation_and_trace(port, sim_path: str, label: str, *,
                                channel: str = "uf_inc") -> tuple[float, int]:
    """Scale-free excitation/trace guard (issue #465/PR #473, reaffirmed
    PR #547 -- see the module docstring PRECEDENT TICK-LIST item 8). No
    absolute floor on the launched-channel peak: only exact-zero/non-finite
    is a defect signature, because the healthy and broken populations
    overlap in openEMS's raw units (verified-good runs as small as
    ~3e-14 are recorded in ``rfx/interop/emitters/openems.py``'s own [D5]
    guard).
    """
    launched = np.asarray(getattr(port, channel), dtype=np.complex128)
    peak = float(np.max(np.abs(launched))) if launched.size else 0.0
    if not np.isfinite(peak) or peak == 0.0:
        raise RuntimeError(
            f"[{label}] openEMS port injected/received NO wave energy on "
            f"its own launched channel ({channel}={peak!r}): excitation "
            f"did not couple or the port never saw the wave."
        )
    n_samples = 0
    for name in list(getattr(port, "U_filenames", []) or []):
        trace_path = os.path.join(sim_path, name)
        if not os.path.exists(trace_path):
            continue
        raw = np.loadtxt(trace_path, comments="%")
        if not raw.size:
            continue
        raw2 = np.atleast_2d(raw)
        n_samples = max(n_samples, raw2.shape[0])
        peak_here = float(np.max(np.abs(raw2[:, 1])))
        if peak_here == 0.0:
            raise RuntimeError(
                f"[{label}] openEMS port voltage time trace is identically "
                f"zero: the excitation never entered the grid."
            )
    return peak, n_samples


def _non_physical_guard(s_mag: np.ndarray, label: str) -> None:
    peak = float(np.max(s_mag)) if s_mag.size else float("nan")
    if not np.all(np.isfinite(s_mag)) or peak > 2.0:
        raise RuntimeError(
            f"[{label}] non-physical/unstable |S| max={peak!r}: field blew "
            f"up or diverged."
        )


def _passivity_witness(s11: np.ndarray, s21: np.ndarray, label: str, *,
                       tol: float = 0.05, idx=None) -> dict:
    balance_full = np.abs(s11) ** 2 + np.abs(s21) ** 2
    balance = balance_full[idx] if idx is not None else balance_full
    max_balance = float(np.max(balance))
    passed = bool(max_balance <= 1.0 + tol)
    if not passed:
        raise RuntimeError(
            f"[{label}] passivity witness failed: max(|S11|^2+|S21|^2)="
            f"{max_balance:.4f} > {1.0 + tol} -- non-physical energy gain."
        )
    return {"balance": balance.tolist(), "max_balance": max_balance,
            "tol": tol, "passed": passed}

# ---------------------------------------------------------------------------
# STAGE A: faithful port of MSL_NotchFilter.py.
# ---------------------------------------------------------------------------
def _build_stage_a_notch_tutorial(ContinuousStructure, openEMS, MSLPort, *,
                                  nrts: int | None, end_criteria: float | None):
    """Build MSL_NotchFilter.py's geometry fresh (smoke + real run, matching
    the coax/thru precedent's "separate openEMS instances" pattern)."""
    kw = {}
    if nrts is not None:
        kw["NrTS"] = nrts
    if end_criteria is not None:
        kw["EndCriteria"] = end_criteria
    fdtd = openEMS(**kw)
    fdtd.SetGaussExcite(A_F_MAX_HZ / 2.0, A_F_MAX_HZ / 2.0)
    fdtd.SetBoundaryCond(["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"])

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(A_UNIT)

    resolution = _C0 / (A_F_MAX_HZ * np.sqrt(A_SUBSTRATE_EPR)) / A_UNIT / 50.0
    third_mesh = np.array([2.0 * resolution / 3.0, -resolution / 3.0]) / 4.0

    mesh.AddLine("x", [0.0])
    mesh.AddLine("x", A_MSL_WIDTH_UM / 2.0 + third_mesh)
    mesh.AddLine("x", -A_MSL_WIDTH_UM / 2.0 - third_mesh)
    mesh.SmoothMeshLines("x", resolution / 4.0)
    mesh.AddLine("x", [-A_MSL_LENGTH_UM, A_MSL_LENGTH_UM])
    mesh.SmoothMeshLines("x", resolution)

    mesh.AddLine("y", [0.0])
    mesh.AddLine("y", A_MSL_WIDTH_UM / 2.0 + third_mesh)
    mesh.AddLine("y", -A_MSL_WIDTH_UM / 2.0 - third_mesh)
    mesh.SmoothMeshLines("y", resolution / 4.0)
    mesh.AddLine("y", [-15.0 * A_MSL_WIDTH_UM, 15.0 * A_MSL_WIDTH_UM + A_STUB_LENGTH_UM])
    mesh.AddLine("y", (A_MSL_WIDTH_UM / 2.0 + A_STUB_LENGTH_UM) + third_mesh)
    mesh.SmoothMeshLines("y", resolution)

    mesh.AddLine("z", np.linspace(0.0, A_SUBSTRATE_THICKNESS_UM, 5))
    mesh.AddLine("z", [3000.0])
    mesh.SmoothMeshLines("z", resolution)

    substrate = csx.AddMaterial("RO4350B", epsilon=A_SUBSTRATE_EPR)
    substrate.AddBox(
        [-A_MSL_LENGTH_UM, -15.0 * A_MSL_WIDTH_UM, 0.0],
        [A_MSL_LENGTH_UM, 15.0 * A_MSL_WIDTH_UM + A_STUB_LENGTH_UM, A_SUBSTRATE_THICKNESS_UM],
    )

    pec = csx.AddMetal("PEC")
    port0 = MSLPort(
        csx, port_nr=1, metal_prop=pec,
        start=[-A_MSL_LENGTH_UM, -A_MSL_WIDTH_UM / 2.0, A_SUBSTRATE_THICKNESS_UM],
        stop=[0.0, A_MSL_WIDTH_UM / 2.0, 0.0],
        prop_dir="x", exc_dir="z", excite=-1.0,
        FeedShift=10.0 * resolution, MeasPlaneShift=A_MSL_LENGTH_UM / 3.0,
        priority=10,
    )
    port1 = MSLPort(
        csx, port_nr=2, metal_prop=pec,
        start=[A_MSL_LENGTH_UM, -A_MSL_WIDTH_UM / 2.0, A_SUBSTRATE_THICKNESS_UM],
        stop=[0.0, A_MSL_WIDTH_UM / 2.0, 0.0],
        prop_dir="x", exc_dir="z",
        MeasPlaneShift=A_MSL_LENGTH_UM / 3.0,
        priority=10,
    )

    pec.AddBox(
        [-A_MSL_WIDTH_UM / 2.0, A_MSL_WIDTH_UM / 2.0, A_SUBSTRATE_THICKNESS_UM],
        [A_MSL_WIDTH_UM / 2.0, A_MSL_WIDTH_UM / 2.0 + A_STUB_LENGTH_UM, A_SUBSTRATE_THICKNESS_UM],
        priority=10,
    )
    return fdtd, port0, port1

def _builder_body_after_kw(func) -> str:
    """The builder's source from its first statement (``kw = {}``) onward.

    Signature and docstring are excluded on purpose: they are where the
    parameters are named, and naming them is not a change to what is built.
    """
    import inspect

    src = inspect.getsource(func)
    marker = "\n    kw = {}\n"
    if marker not in src:
        raise RuntimeError(f"{func.__name__}: no '    kw = {{}}' first statement to anchor on")
    return src[src.index(marker) + 1:]


# ---------------------------------------------------------------------------
# The repository's one spectral-feature estimator module, loaded by path
# (validation/ is not a package; it imports numpy only). Same loader the cases'
# own tests use.
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    env = os.environ.get("RFX_REPO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def load_spectral_features():
    """``validation/crossval/comparators/spectral_features.py`` as a module.

    ``refined_extremum`` (log-parabolic sub-bin extremum) and
    ``level_crossing`` (linearly interpolated level crossing) both live there.
    A case must not roll its own: the estimator the record is written with has
    to be the estimator the case's test reads it back with, or the record and
    the case drift apart.
    """
    path = _repo_root() / "validation" / "crossval" / "comparators" / "spectral_features.py"
    if not path.is_file():
        raise RuntimeError(
            f"shared spectral-feature estimators not found at {path} -- this "
            f"script must run from inside the repository (or set RFX_REPO_ROOT)"
        )
    spec = importlib.util.spec_from_file_location("_rfx_crossval_spectral_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_refined_extremum():
    return load_spectral_features().refined_extremum


# ---------------------------------------------------------------------------
# Pure-numpy geometry / mesh plan. No openEMS, no CSXCAD.
#
# WHAT IT CANNOT DO: CSXCAD's own SmoothMeshLines grades the transition between
# a fine and a coarse region, and it is a CSXCAD call. The estimate below only
# subdivides each gap into equal parts no wider than the target, so every line
# count it reports is a LOWER BOUND and every cell width it reports near a
# fine/coarse transition is an UPPER bound. Away from such a transition -- which
# is where the x boundaries and therefore the PML sit -- the two agree.
# ---------------------------------------------------------------------------
def _smooth_estimate(lines, max_res: float) -> np.ndarray:
    lines = np.unique(np.asarray(lines, dtype=float))
    out = [lines[0]]
    for a, b in zip(lines[:-1], lines[1:]):
        n = max(1, int(np.ceil((b - a) / max_res - 1e-9)))
        out.extend(np.linspace(a, b, n + 1)[1:])
    return np.unique(np.asarray(out, dtype=float))

# ---------------------------------------------------------------------------
# What the solver actually built, read back
# ---------------------------------------------------------------------------
_VERSION_RE = re.compile(r"openEMS[^\n]*?(v\d[\w.\-+]*)", re.IGNORECASE)
_TIMESTEP_RE = re.compile(r"Timestep:\s*(\d+)")


def _openems_version(log_text: str) -> dict:
    """The solver's version, or a loud failure.

    Four sources, in this order, and the record names which one answered. A
    record that cannot say which openEMS produced it is not a reference, so
    "none of the four" raises instead of writing ``null``.
    """
    tried = []
    try:
        import openEMS as _pkg
        v = getattr(_pkg, "__version__", None)
        if v:
            return {"version": str(v), "source": "openEMS.__version__"}
        tried.append("openEMS.__version__ (attribute absent)")
    except Exception as exc:
        tried.append(f"openEMS.__version__ ({exc!r})")
    try:
        from openEMS.openEMS import openEMS as _cls
        v = getattr(_cls, "__version__", None)
        if v:
            return {"version": str(v), "source": "openEMS.openEMS.openEMS.__version__"}
        tried.append("openEMS.openEMS.openEMS.__version__ (attribute absent)")
    except Exception as exc:
        tried.append(f"openEMS.openEMS.openEMS.__version__ ({exc!r})")
    try:
        import importlib.metadata as md
        return {"version": md.version("openEMS"), "source": "importlib.metadata"}
    except Exception as exc:
        tried.append(f"importlib.metadata.version('openEMS') ({exc!r})")
    m = _VERSION_RE.search(log_text)
    if m:
        return {"version": m.group(1),
                "source": "openEMS's own startup banner in the captured log"}
    tried.append("the solver's startup banner in the captured log (no match)")
    raise RuntimeError(
        "the container reports NO openEMS version. Tried, in order: "
        + "; ".join(tried)
        + ". A reference record that cannot name the solver that produced it is "
          "not a reference -- refusing to write one."
    )


def _timesteps_executed(log_text: str):
    hits = [int(m.group(1)) for m in _TIMESTEP_RE.finditer(log_text)]
    return max(hits) if hits else None


def _mesh_lines(fdtd):
    """The x/y/z lines CSXCAD actually built, in the CSX unit."""
    try:
        grid = fdtd.GetCSX().GetGrid()
        return {ax: np.asarray(grid.GetLines(ax), dtype=float) for ax in ("x", "y", "z")}
    except Exception:
        return None


def lines_in_um(lines, csx_unit_m: float):
    """The realized lines converted to um, whatever unit the case builds in.

    Every reporting helper below reads um. A case that sets ``SetDeltaUnit(1.0)``
    and works in metres converts here, once, instead of every field carrying a
    unit that depends on which case wrote it.
    """
    if lines is None:
        return None
    scale = csx_unit_m / 1.0e-6
    return {ax: np.asarray(lines[ax], dtype=float) * scale for ax in ("x", "y", "z")}


def _mesh_realized(lines_um, *, substrate_thickness_um: float) -> dict:
    """Summary of the realized mesh. Every number here was read back, not declared."""
    if lines_um is None:
        return {"error": "CSXCAD did not return its grid lines"}
    out: dict = {}
    cells = 1
    for ax in ("x", "y", "z"):
        ln = lines_um[ax]
        steps = np.diff(ln)
        out[ax] = {
            "n_lines": int(ln.size),
            "min_um": float(ln.min()), "max_um": float(ln.max()),
            "step_min_um": float(steps.min()) if steps.size else None,
            "step_max_um": float(steps.max()) if steps.size else None,
        }
        cells *= max(int(ln.size) - 1, 0)
    out["n_cells"] = int(cells)
    z = lines_um["z"]
    in_sub = z[(z >= -1e-9) & (z <= substrate_thickness_um + 1e-9)]
    out["substrate_z_cells_realized"] = int(in_sub.size - 1) if in_sub.size else 0
    out["substrate_top_on_a_line"] = bool(
        in_sub.size and abs(in_sub.max() - substrate_thickness_um) < 1e-6)
    return out


def _nearest_line(lines_um, target_um: float) -> tuple:
    """(nearest realized line, |target - line|) in um."""
    i = int(np.argmin(np.abs(lines_um - target_um)))
    return float(lines_um[i]), float(abs(lines_um[i] - target_um))


def nearest_line(lines_um, target_um: float) -> tuple:
    """Public name for ``_nearest_line``; a case reports edges with it."""
    return _nearest_line(lines_um, target_um)


def _port_declared_and_snap(lines_x_um, *, start_x_um: float, direction: float,
                            feed_shift_um: float, measplane_shift_um: float,
                            port_obj=None, csx_unit_m: float = 1.0e-6) -> dict:
    """What the script declared for this port, and where the grid put it.

    The ``*_declared_mm`` fields are the declared numbers. The ``*_nearest_line_mm``
    / ``*_snap_um`` fields are measured against the x lines CSXCAD built: the
    distance is how far the plane had to move to reach a line.
    """
    feed_x = start_x_um + direction * feed_shift_um
    meas_x = start_x_um + direction * measplane_shift_um
    out = {
        "start_x_declared_mm": start_x_um / 1e3,
        "prop_direction": "+x" if direction > 0 else "-x",
        "feed_shift_declared_mm": feed_shift_um / 1e3,
        "meas_plane_shift_declared_mm": measplane_shift_um / 1e3,
        "feed_plane_x_declared_mm": feed_x / 1e3,
        "meas_plane_x_declared_mm": meas_x / 1e3,
    }
    if lines_x_um is not None and np.size(lines_x_um):
        f_line, f_snap = _nearest_line(lines_x_um, feed_x)
        m_line, m_snap = _nearest_line(lines_x_um, meas_x)
        out.update({
            "feed_plane_nearest_line_mm": f_line / 1e3,
            "feed_plane_snap_um": f_snap,
            "meas_plane_nearest_line_mm": m_line / 1e3,
            "meas_plane_snap_um": m_snap,
        })
    else:
        out.update({"feed_plane_nearest_line_mm": None, "feed_plane_snap_um": None,
                    "meas_plane_nearest_line_mm": None, "meas_plane_snap_um": None})
    # What MSLPort itself reports after its own snap, kept separate from both the
    # declared numbers and the line measurement above.
    for attr, key in (("feed_shift", "port_object_feed_shift_mm"),
                      ("measplane_shift", "port_object_meas_plane_shift_mm")):
        try:
            out[key] = float(getattr(port_obj, attr)) * csx_unit_m / 1e-3
        except Exception:
            out[key] = None
    return out


class StageFailure(RuntimeError):
    """A sanity gate fired. Carries what had been measured when it did.

    The precedent's own convention (its PRECEDENT TICK-LIST item 11: "partial
    per-bin data attached to a raised RuntimeError so a failing run still leaves
    numbers to inspect"). ``main`` writes it to ``<output stem>_FAILED.json``.
    """

    def __init__(self, message: str, *, stage: str, partial: dict, meta: dict):
        super().__init__(message)
        self.stage = stage
        self.partial = partial
        self.meta = meta


def energy_summary(freqs_hz: np.ndarray, s11: np.ndarray, s21: np.ndarray, *,
                   witness_band_hz, passivity_tol: float) -> dict:
    """|S11|^2+|S21|^2 per bin, and where its maxima sit.

    Reported for the whole grid AND split at the band edge, so "the witness read
    X" is never a single number a reader cannot place. The witness bounds the
    EXCESS above unity only, so the band minimum and the whole-grid minimum are
    reported beside the maxima: on a structure that radiates, a sum below unity
    is a deficit the record states rather than a gate reading.
    """
    energy = np.abs(s11) ** 2 + np.abs(s21) ** 2
    band = (freqs_hz >= witness_band_hz[0]) & (freqs_hz <= witness_band_hz[1])
    below_1 = freqs_hz < 1.0e9
    one_to_two = (freqs_hz >= 1.0e9) & (freqs_hz < witness_band_hz[0])
    above = freqs_hz > witness_band_hz[1]
    band_ghz = [witness_band_hz[0] / 1e9, witness_band_hz[1] / 1e9]

    def _max(mask) -> float:
        return float(np.max(energy[mask])) if np.any(mask) else float("nan")

    def _min(mask) -> float:
        return float(np.min(energy[mask])) if np.any(mask) else float("nan")

    return {
        "energy_sum": energy,
        "band_mask": band,
        "max_energy_sum_band": _max(band),
        "max_energy_sum_full": float(np.max(energy)) if energy.size else float("nan"),
        "max_energy_sum_below_band": {
            "0_1_ghz": _max(below_1),
            "1_2_ghz": _max(one_to_two),
        },
        "min_energy_sum_band": _min(band),
        "min_energy_sum_full": float(np.min(energy)) if energy.size else float("nan"),
        "max_energy_sum_above_band": _max(above),
        "min_energy_sum_above_band": _min(above),
        "witness_band_ghz": band_ghz,
        "passivity_tol": 1.0 + passivity_tol,
    }


def stage_a_gate_verdict(f_notch_hz: float, depth_db: float) -> dict:
    """The reproduce gate: the right frequency AND an actual notch.

    Pure arithmetic so a case's self-check can plant curves at it.
    """
    f_ok = bool(STAGE_A_GATE["f_notch_lo_hz"] <= f_notch_hz <= STAGE_A_GATE["f_notch_hi_hz"])
    depth_ok = bool(depth_db <= STAGE_A_MIN_DEPTH_DB)
    return {
        "measured_f_notch_hz": float(f_notch_hz),
        "measured_depth_db": float(depth_db),
        "analytic_f_notch_hz": float(F_NOTCH_AN_HZ),
        "deviation_pct": abs(f_notch_hz - F_NOTCH_AN_HZ) / F_NOTCH_AN_HZ * 100.0,
        "gate_band_hz": [STAGE_A_GATE["f_notch_lo_hz"], STAGE_A_GATE["f_notch_hi_hz"]],
        "gate_min_depth_db": STAGE_A_MIN_DEPTH_DB,
        "f_notch_ok": f_ok,
        "depth_ok": depth_ok,
        "passed": bool(f_ok and depth_ok),
    }


def stage_a_notch_features(refined_extremum):
    """The Stage A feature callback: the tutorial's notch, over 2-7 GHz.

    Returned as a callable so ``run_stage`` stays ignorant of which estimator a
    case loaded, and so the failure mode (an estimator that raises) lands in the
    record instead of losing the arrays that were already measured.
    """
    def features(freqs_ghz, s11, s21) -> dict:
        try:
            notch = refined_extremum(freqs_ghz, np.abs(s21),
                                     STAGE_A_NOTCH_BAND_GHZ[0], STAGE_A_NOTCH_BAND_GHZ[1],
                                     transform="log")
            return {"notch": {
                "bin_f_ghz": float(notch["bin_f"]),
                "refined_f_ghz": float(notch["refined_f"]),
                "depth_db": float(notch["depth_db"]),
                "sub_bin_shift_bins": float(notch["sub_bin_shift"]),
                "bin_width_ghz": float(notch["bin_width"]),
                "band_ghz": list(STAGE_A_NOTCH_BAND_GHZ),
                "estimator": "validation/crossval/comparators/spectral_features.py::"
                             "refined_extremum, transform='log'",
            }}
        except Exception as exc:
            return {"notch": {"error": repr(exc)}}
    return features


def build_stage_a(ContinuousStructure, openEMS, MSLPort, *, nrts, end_criteria):
    """The Stage A builder, as a case calls it: the tutorial, verbatim."""
    return _build_stage_a_notch_tutorial(
        ContinuousStructure, openEMS, MSLPort, nrts=nrts, end_criteria=end_criteria)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def run_stage(*, label: str, sim_root: str, threads: int, build,
              freqs_hz, witness_band_hz, passivity_tol: float,
              real_nrts, real_end_criteria,
              mesh_realized_fn, meta_extra_fn, features_fn,
              calcport_ref_impedance=None, record_deficit: bool = False,
              smoke_nrts: int = 200, smoke_end_criteria: float = 0.0):
    """Smoke pass, real pass, CalcPort, the ported sanity gates, the features.

    The sequence and every gate call is the precedent's
    ``_run_stage_a_reproduce_gate`` (validation/crossval/20_msl_phase_referee.py).
    Six things differ, all declared:

      1. the builder, the frequency grid, the witness band and the real pass's
         NrTS/EndCriteria are the CASE's, passed in;
      2. a truncated real pass RAISES here instead of being reported as a flag;
      3. the spectral features come from the repository's shared estimators over
         the case's own band instead of a bare argmin over 0.5-1.5 x the
         analytic frequency;
      4. ``_passivity_witness`` runs on every stage (the precedent's Stage A
         runner does not call it at all) and is given the witness band's ``idx``
         mask through the function's own existing argument -- the function is
         untouched;
      5. the arrays, the features and the energy-sum numbers are computed and
         stashed BEFORE ``_non_physical_guard`` and ``_passivity_witness`` run,
         so a gate that fires still leaves them behind. Only the order of pure
         post-processing moved; both guards still block the record;
      6. ``calcport_ref_impedance``, when a case passes one, adds the second
         ``CalcPort`` pass that references S to a system impedance, and the
         FIRST pass's ``Z_ref`` real part is recorded as ``re_z0``. A case that
         passes nothing keeps the precedent's single unreferenced pass.

    ``record_deficit`` adds the band and whole-grid MINIMA of the energy sum, and
    its maximum and minimum above the band, to the record. A case whose structure
    radiates wants them; a case whose grid ends at the band edge would only get
    NaNs, so it is off unless asked for.
    """
    ContinuousStructure, openEMS, MSLPort = _import_openems()

    sim_dir = os.path.join(sim_root, label)
    smoke_dir = os.path.join(sim_root, label + "_smoke")

    record: dict = {}
    meta: dict = {"stage": label}
    band_ghz = (witness_band_hz[0] / 1e9, witness_band_hz[1] / 1e9)

    def fail(exc: Exception):
        return StageFailure(str(exc), stage=label, partial=record, meta=meta)

    try:
        smoke_fdtd, _p0, _p1 = build(ContinuousStructure, openEMS, MSLPort,
                                     nrts=smoke_nrts, end_criteria=smoke_end_criteria)
        smoke_log = _run_openems_capturing_stdout(smoke_fdtd, smoke_dir, threads=threads)
        _scan_stdout_for_bad_patterns(smoke_log, label + "_smoke")

        fdtd, port0, port1 = build(ContinuousStructure, openEMS, MSLPort,
                                   nrts=real_nrts, end_criteria=real_end_criteria)
        lines = _mesh_lines(fdtd)
        mesh = mesh_realized_fn(lines)
        meta["mesh_realized"] = mesh

        t0 = time.time()
        real_log = _run_openems_capturing_stdout(fdtd, sim_dir, threads=threads)
        _scan_stdout_for_bad_patterns(real_log, label, check_truncation=True)
        elapsed = time.time() - t0

        freqs = np.asarray(freqs_hz, dtype=float)
        # Pass 1 carries no ref_impedance, so port.Z_ref is the measured line
        # impedance rather than the value it was told to assume.
        port0.CalcPort(sim_dir, freqs)
        port1.CalcPort(sim_dir, freqs)
        re_z0 = None
        if calcport_ref_impedance is not None:
            re_z0 = np.real(np.asarray(port0.Z_ref, dtype=np.complex128))
            port0.CalcPort(sim_dir, freqs, ref_impedance=calcport_ref_impedance)
            port1.CalcPort(sim_dir, freqs, ref_impedance=calcport_ref_impedance)

        inc_peak, n_samples = _check_excitation_and_trace(port0, sim_dir, label)

        if _log_indicates_truncation(real_log):
            raise RuntimeError(
                f"[{label}] SANITY GATE 'end criteria reached' FAILED: openEMS's own "
                f"'reached before the end-criteria of' warning is in this real pass's "
                f"captured log -- the run hit its NrTS cap before the field decayed, so "
                f"the spectrum is truncated and no record is written."
            )

        s11 = np.asarray(port0.uf_ref, dtype=np.complex128) / np.asarray(port0.uf_inc, dtype=np.complex128)
        s21 = np.asarray(port1.uf_ref, dtype=np.complex128) / np.asarray(port0.uf_inc, dtype=np.complex128)

        freqs_ghz = freqs / 1e9
        summary = energy_summary(freqs, s11, s21, witness_band_hz=witness_band_hz,
                                 passivity_tol=passivity_tol)

        record.update({
            "freqs_ghz": freqs_ghz.tolist(),
            "s11_mag": np.abs(s11).tolist(),
            "s11_deg": np.degrees(np.angle(s11)).tolist(),
            "s21_mag": np.abs(s21).tolist(),
            "s21_deg": np.degrees(np.angle(s21)).tolist(),
            "energy_sum": summary["energy_sum"].tolist(),
            "max_energy_sum_band": summary["max_energy_sum_band"],
            "max_energy_sum_full": summary["max_energy_sum_full"],
            "max_energy_sum_below_band": summary["max_energy_sum_below_band"],
            "witness_band_ghz": summary["witness_band_ghz"],
            "passivity_tol": summary["passivity_tol"],
        })
        if record_deficit:
            for key in ("min_energy_sum_band", "min_energy_sum_full",
                        "max_energy_sum_above_band", "min_energy_sum_above_band"):
                if np.isfinite(summary[key]):
                    record[key] = summary[key]
        if re_z0 is not None:
            record["re_z0"] = re_z0.tolist()
            record["re_z0_median_ohm"] = float(np.median(re_z0))

        # Printed before the guards run, so the numbers reach the log even when
        # the next line raises.
        print(f"  energy sum max(|S11|^2+|S21|^2): "
              f"band {band_ghz[0]:.1f}-{band_ghz[1]:.1f} GHz "
              f"{summary['max_energy_sum_band']:.4f} | "
              f"full grid {summary['max_energy_sum_full']:.4f} | "
              f"0-1 GHz {summary['max_energy_sum_below_band']['0_1_ghz']:.4f} | "
              f"1-2 GHz {summary['max_energy_sum_below_band']['1_2_ghz']:.4f} "
              f"(tol {summary['passivity_tol']:.2f}, judged on the band)", flush=True)

        # Inside the guarded block on purpose: a container that reports no
        # openEMS version refuses the record, but the arrays measured above
        # still reach the evidence file instead of a bare traceback (review,
        # 2026-09-22).
        openems_info = _openems_version(real_log)

        record.update(features_fn(freqs_ghz, s11, s21))

        _non_physical_guard(np.abs(s11), label + "_s11")
        _non_physical_guard(np.abs(s21), label + "_s21")
        # The witness function is byte-identical to the precedent's; the band is
        # supplied through its own idx argument.
        _passivity_witness(s11, s21, label,
                           tol=passivity_tol, idx=summary["band_mask"])
    except RuntimeError as exc:
        raise fail(exc) from exc

    meta.update(meta_extra_fn(lines=lines, port0=port0, port1=port1))
    meta.update({
        "mesh_realized": mesh,
        "timesteps_executed": _timesteps_executed(real_log),
        "end_criteria_reached": True,
        "excitation_energy_peak": inc_peak,
        "port_trace_samples": n_samples,
        "wall_time_s": round(elapsed, 1),
        "stdout_log_path": os.path.join(sim_dir, "_openems_stdout.log"),
        "smoke_stdout_log_path": os.path.join(smoke_dir, "_openems_stdout.log"),
        "openems": openems_info,
    })
    return record, meta


def build_artifact(records: dict, stage_meta: dict, stage_a_gate: dict, stages: list,
                   *, stage_names, meta_common: dict, produced_by: str,
                   failed_gate: str | None = None) -> dict:
    """The record, whole or partial. ``failed_gate`` marks the partial one."""
    first_meta = next((m for m in stage_meta.values() if m.get("openems")), None)
    meta = {
        "tool": "openEMS",
        "openems": (first_meta["openems"] if first_meta
                    else {"version": None, "source": "no stage reached the solver"}),
        "rfx_openems_commit": os.environ.get("RFX_OPENEMS_COMMIT"),
        "rfx_openems_image": os.environ.get("RFX_OPENEMS_IMAGE"),
    }
    meta.update(meta_common)
    meta.update({
        "reproduce_gate_record": REPRODUCE_GATE_RECORD,
        "stage_a_min_depth_db": STAGE_A_MIN_DEPTH_DB,
        "stage_a_gate": stage_a_gate,
        "reproduce_gate_ran": "stage_a" in stages,
        "stages_requested": stages,
        "stages": stage_meta,
        "produced_by": produced_by,
        "ci_runs_this": False,
    })
    artifact: dict = {"meta": meta}
    for name in stage_names:
        artifact[name] = records.get(name)
    artifact["run_id"] = None
    artifact["run_id_note"] = (
        "null by design. VESSL does not export VESSL_RUN_ID into the pod, so the "
        "job cannot write its own id; the submitter fills this field "
        "(scripts/vessl_submit.sh drops run_id.txt beside the record)."
    )
    if failed_gate is not None:
        artifact["failed_gate"] = failed_gate
        artifact["meta"]["record_is_partial"] = True
    return artifact


def failed_output_path(out: Path) -> Path:
    return out.with_name(out.stem + "_FAILED" + out.suffix)


def write_record(artifact: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(artifact, fh, indent=1)


def stages_for(stage: str) -> list:
    if stage == "A":
        return ["stage_a"]
    if stage == "B":
        return ["stage_b_coarse", "stage_b_mid", "stage_b_fine"]
    return ["stage_a", "stage_b_coarse", "stage_b_mid", "stage_b_fine"]


# ---------------------------------------------------------------------------
# The TUTORIAL's own geometry and mesh, as numbers. One copy: both the dry-run
# table and the Stage A record's plan_estimate read it, and the MSL notch
# filter's Stage B is the tutorial at a mesh rung, so its rungs read it too.
# ---------------------------------------------------------------------------
def tutorial_plan(label: str, msl_length_um: float, resolution_factor: float) -> dict:
    """The geometry and mesh the named stage builds, as numbers."""
    resolution = _C0 / (A_F_MAX_HZ * np.sqrt(A_SUBSTRATE_EPR)) / A_UNIT / 50.0 * resolution_factor
    third_mesh = np.array([2.0 * resolution / 3.0, -resolution / 3.0]) / 4.0

    x = [0.0]
    x += list(A_MSL_WIDTH_UM / 2.0 + third_mesh)
    x += list(-A_MSL_WIDTH_UM / 2.0 - third_mesh)
    x = _smooth_estimate(x, resolution / 4.0)
    x = _smooth_estimate(np.concatenate([x, [-msl_length_um, msl_length_um]]), resolution)

    y = [0.0]
    y += list(A_MSL_WIDTH_UM / 2.0 + third_mesh)
    y += list(-A_MSL_WIDTH_UM / 2.0 - third_mesh)
    y = _smooth_estimate(y, resolution / 4.0)
    y = _smooth_estimate(
        np.concatenate([
            y,
            [-15.0 * A_MSL_WIDTH_UM, 15.0 * A_MSL_WIDTH_UM + A_STUB_LENGTH_UM],
            (A_MSL_WIDTH_UM / 2.0 + A_STUB_LENGTH_UM) + third_mesh,
        ]),
        resolution,
    )

    n_sub = substrate_z_cells(resolution_factor)
    z = _smooth_estimate(
        np.concatenate([np.linspace(0.0, A_SUBSTRATE_THICKNESS_UM, n_sub + 1), [3000.0]]),
        resolution,
    )

    feed_shift = 10.0 * resolution
    measplane_shift = msl_length_um / 3.0
    # PML_8 on the x faces only: it eats the outermost 8 cells at each end.
    pml_lo = float(x[A_PML_CELLS]) if x.size > A_PML_CELLS else float(x[-1])
    pml_hi = float(x[-1 - A_PML_CELLS]) if x.size > A_PML_CELLS else float(x[0])
    feed_x = -msl_length_um + feed_shift
    measplane_x = -msl_length_um + measplane_shift
    stub_tip_y = A_MSL_WIDTH_UM / 2.0 + A_STUB_LENGTH_UM
    substrate_y_hi = 15.0 * A_MSL_WIDTH_UM + A_STUB_LENGTH_UM

    return {
        "label": label,
        "msl_length_um": float(msl_length_um),
        "resolution_factor": float(resolution_factor),
        "resolution_um": float(resolution),
        "third_mesh_um": [float(v) for v in third_mesh],
        "box_mm": {
            "x": [-msl_length_um / 1e3, msl_length_um / 1e3],
            "y": [-15.0 * A_MSL_WIDTH_UM / 1e3, substrate_y_hi / 1e3],
            "z": [0.0, 3000.0 / 1e3],
        },
        "substrate_box_mm": {
            "x": [-msl_length_um / 1e3, msl_length_um / 1e3],
            "y": [-15.0 * A_MSL_WIDTH_UM / 1e3, substrate_y_hi / 1e3],
            "z": [0.0, A_SUBSTRATE_THICKNESS_UM / 1e3],
        },
        "stub_tip_y_mm": stub_tip_y / 1e3,
        "stub_tip_to_substrate_edge_mm": (substrate_y_hi - stub_tip_y) / 1e3,
        "mesh_lines_estimate": {"x": int(x.size), "y": int(y.size), "z": int(z.size)},
        "cells_estimate": int((x.size - 1) * (y.size - 1) * (z.size - 1)),
        "mesh_step_estimate_um": {
            "x_min": float(np.min(np.diff(x))), "x_max": float(np.max(np.diff(x))),
            "y_min": float(np.min(np.diff(y))), "y_max": float(np.max(np.diff(y))),
            "z_min": float(np.min(np.diff(z))), "z_max": float(np.max(np.diff(z))),
        },
        "substrate_z_cells_estimate": int(np.sum(
            (z >= -1e-9) & (z <= A_SUBSTRATE_THICKNESS_UM + 1e-9)) - 1),
        "substrate_z_step_estimate_um": float(A_SUBSTRATE_THICKNESS_UM / n_sub),
        "port0_start_x_mm": -msl_length_um / 1e3,
        "port1_start_x_mm": msl_length_um / 1e3,
        "feed_shift_mm": feed_shift / 1e3,
        "measplane_shift_mm": measplane_shift / 1e3,
        "feed_x_mm": feed_x / 1e3,
        "measplane_x_mm": measplane_x / 1e3,
        "pml_inner_face_x_mm_estimate": [pml_lo / 1e3, pml_hi / 1e3],
        "pml_depth_mm_estimate": (pml_lo + msl_length_um) / 1e3,
        "feed_inside_pml_estimate": bool(feed_x < pml_lo),
        "measplane_inside_pml_estimate": bool(measplane_x < pml_lo),
        "measplane_downstream_of_feed": bool(measplane_shift > feed_shift),
    }

def print_tutorial_plan(plan: dict) -> None:
    b, s = plan["box_mm"], plan["substrate_box_mm"]
    m, st = plan["mesh_lines_estimate"], plan["mesh_step_estimate_um"]
    print(f"  {plan['label']}  (MSL_length = {plan['msl_length_um']:.0f} um, "
          f"resolution factor {plan['resolution_factor']:g})")
    print(f"    resolution              {plan['resolution_um']:.3f} um   "
          f"third_mesh {plan['third_mesh_um'][0]:+.3f} / {plan['third_mesh_um'][1]:+.3f} um")
    print(f"    box (mm)                x [{b['x'][0]:+.3f}, {b['x'][1]:+.3f}]  "
          f"y [{b['y'][0]:+.3f}, {b['y'][1]:+.3f}]  z [{b['z'][0]:.3f}, {b['z'][1]:.3f}]")
    print(f"    substrate box (mm)      x [{s['x'][0]:+.3f}, {s['x'][1]:+.3f}]  "
          f"y [{s['y'][0]:+.3f}, {s['y'][1]:+.3f}]  z [{s['z'][0]:.3f}, {s['z'][1]:.3f}]")
    print(f"    stub tip y              {plan['stub_tip_y_mm']:.3f} mm  "
          f"({plan['stub_tip_to_substrate_edge_mm']:.3f} mm short of the substrate's y edge)")
    print(f"    mesh lines (estimate)   x {m['x']}  y {m['y']}  z {m['z']}   "
          f"-> cells >= {plan['cells_estimate']:,}")
    print(f"    cell size (estimate)    x {st['x_min']:.2f}-{st['x_max']:.2f}  "
          f"y {st['y_min']:.2f}-{st['y_max']:.2f}  z {st['z_min']:.2f}-{st['z_max']:.2f} um")
    print(f"    substrate z cells       {plan['substrate_z_cells_estimate']} of "
          f"{plan['substrate_z_step_estimate_um']:.2f} um across the 254 um board")
    print(f"    ports                   port0 start x {plan['port0_start_x_mm']:+.3f} mm, "
          f"port1 start x {plan['port1_start_x_mm']:+.3f} mm")
    print(f"    FeedShift               {plan['feed_shift_mm']:.3f} mm "
          f"-> feed at x {plan['feed_x_mm']:+.3f} mm")
    print(f"    MeasPlaneShift          {plan['measplane_shift_mm']:.3f} mm "
          f"-> measurement plane at x {plan['measplane_x_mm']:+.3f} mm")
    print(f"    x PML_8 inner face      {plan['pml_inner_face_x_mm_estimate'][0]:+.3f} mm "
          f"(depth {plan['pml_depth_mm_estimate']:.3f} mm, estimate)")
    print(f"    feed inside the PML     {plan['feed_inside_pml_estimate']}")
    print(f"    meas plane inside PML   {plan['measplane_inside_pml_estimate']}")
    print(f"    meas plane downstream of the feed  {plan['measplane_downstream_of_feed']}")


# ---------------------------------------------------------------------------
# Stage A, end to end. A case calls this and gets the tutorial's own record.
# ---------------------------------------------------------------------------
def stage_a_freqs_hz() -> np.ndarray:
    """The tutorial's own CalcPort grid: linspace(1e6, 7e9, 1601)."""
    return np.linspace(1.0e6, A_F_MAX_HZ, A_N_FREQS)


def stage_a_resolution_um() -> float:
    """The tutorial's own resolution, ~lambda/50 in the substrate."""
    return _C0 / (A_F_MAX_HZ * np.sqrt(A_SUBSTRATE_EPR)) / A_UNIT / 50.0


def run_stage_a(*, sim_root: str, threads: int, refined_extremum,
                label: str = "stage_a") -> tuple[dict, dict]:
    """The reproduce gate's own stage: the tutorial verbatim, on its own mesh.

    A case that wants the gate calls this and then reads
    ``stage_a_gate_verdict`` on the ``notch`` the record carries. Nothing here
    is a case parameter: the geometry, the mesh, the 1601-point grid, the
    2-7 GHz witness band, the single unreferenced ``CalcPort`` pass and
    openEMS's own NrTS/EndCriteria defaults are the tutorial's.
    """
    resolution_um = stage_a_resolution_um()

    def build(ContinuousStructure, openEMS, MSLPort, *, nrts, end_criteria):
        return _build_stage_a_notch_tutorial(
            ContinuousStructure, openEMS, MSLPort, nrts=nrts, end_criteria=end_criteria)

    def mesh_realized_fn(lines):
        return _mesh_realized(lines_in_um(lines, A_UNIT),
                              substrate_thickness_um=A_SUBSTRATE_THICKNESS_UM)

    def meta_extra_fn(*, lines, port0, port1) -> dict:
        lines_um = lines_in_um(lines, A_UNIT)
        x = None if lines_um is None else lines_um["x"]
        return {
            "model": ("openEMS python/Tutorials/MSL_NotchFilter.py, verbatim -- "
                      "the reproduce gate, NOT the case's own structure"),
            "resolution_um": resolution_um,
            "resolution_factor": 1.0,
            "msl_length_um": float(A_MSL_LENGTH_UM),
            "csx_unit_m": A_UNIT,
            "box_mm": {
                "x": [-A_MSL_LENGTH_UM / 1e3, A_MSL_LENGTH_UM / 1e3],
                "y": [-15.0 * A_MSL_WIDTH_UM / 1e3,
                      (15.0 * A_MSL_WIDTH_UM + A_STUB_LENGTH_UM) / 1e3],
                "z": [0.0, 3.0],
            },
            "port0": _port_declared_and_snap(
                x, start_x_um=-A_MSL_LENGTH_UM, direction=+1.0,
                feed_shift_um=10.0 * resolution_um,
                measplane_shift_um=A_MSL_LENGTH_UM / 3.0, port_obj=port0,
                csx_unit_m=A_UNIT),
            "port1": _port_declared_and_snap(
                x, start_x_um=+A_MSL_LENGTH_UM, direction=-1.0,
                feed_shift_um=10.0 * resolution_um,
                measplane_shift_um=A_MSL_LENGTH_UM / 3.0, port_obj=port1,
                csx_unit_m=A_UNIT),
            "substrate_z_cells_declared": A_SUBSTRATE_Z_CELLS,
            "nrts_declared": "openEMS library default (~1e9)",
            "end_criteria_declared": "openEMS library default (1e-5)",
            "calcport_grid": f"linspace(1e6, {A_F_MAX_HZ}, {A_N_FREQS})",
            "calcport_passes": "one, no ref_impedance (the precedent's own tick)",
            "plan_estimate": tutorial_plan(label, A_MSL_LENGTH_UM, 1.0),
        }

    return run_stage(
        label=label, sim_root=sim_root, threads=threads, build=build,
        freqs_hz=stage_a_freqs_hz(),
        witness_band_hz=STAGE_A_WITNESS_BAND_HZ, passivity_tol=PASSIVITY_TOL,
        real_nrts=None, real_end_criteria=None,
        mesh_realized_fn=mesh_realized_fn, meta_extra_fn=meta_extra_fn,
        features_fn=stage_a_notch_features(refined_extremum))
